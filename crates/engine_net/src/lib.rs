//! Сетевой загрузчик данных (решение D5): модели, текстуры, JSON, бинарные файлы.
//!
//! - **Таймауты** соединения и всего запроса.
//! - **Повторы** с экспоненциальной задержкой при сетевых ошибках и ответах 5xx/429.
//! - **Дедупликация**: одновременные запросы одного URL превращаются в один.
//! - **Дисковый кеш** (native): лимит размера, вытеснение давно не использованного,
//!   проверка актуальности через `ETag`/`Last-Modified`, работа офлайн по кешу.
//!   Оборванная загрузка докачивается (`Range`). На Web кешем управляет браузер.
//! - **Целостность**: SHA-256, если его назвали при запросе или в манифесте.
//! - **Предзагрузка** списка адресов ([`Manifest`]).
//!
//! ```ignore
//! let net = world.resource::<Net>().clone();
//! let task = tasks::spawn_io(async move { net.get("https://example.com/a.glb").await });
//! ```

#[cfg(not(target_arch = "wasm32"))]
mod cache;
mod error;
mod hash;
mod manifest;
#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
mod resume;
#[cfg(target_arch = "wasm32")]
mod web;

pub use hash::{parse_sha256, sha256};
pub use manifest::{Manifest, ManifestEntry};

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use engine_core::tasks::{self, BoxedFuture};
use engine_core::{App, CVar, Plugin, Stage};
use engine_ecs::prelude::*;
use engine_platform::Duration;
use futures_util::FutureExt;
use futures_util::future::Shared;

pub use error::NetError;

/// Тело ответа. Разделяемое: при дедупликации один буфер получают все ожидающие.
pub type Bytes = Arc<[u8]>;

pub static NET_CONNECT_TIMEOUT_MS: CVar<u64> = CVar::new(
    "net.connect_timeout_ms",
    5_000,
    "Таймаут установки соединения, мс",
);
pub static NET_TIMEOUT_MS: CVar<u64> = CVar::new(
    "net.timeout_ms",
    60_000,
    "Таймаут всего запроса (с загрузкой тела), мс",
);
pub static NET_RETRIES: CVar<u32> = CVar::new(
    "net.retries",
    3,
    "Число повторов при сетевых ошибках и ответах 5xx",
);
pub static NET_CACHE_MAX_MB: CVar<u64> = CVar::new(
    "net.cache_max_mb",
    512,
    "Размер дискового кеша, МБ (0 — без кеша)",
);
pub static NET_MAX_RESPONSE_MB: CVar<u64> =
    CVar::new("net.max_response_mb", 256, "Максимальный размер ответа, МБ");

/// Подключает сеть: ресурс [`Net`] и статистику [`NetStats`].
#[derive(Debug, Default)]
pub struct NetPlugin;

impl Plugin for NetPlugin {
    fn build(&self, app: &mut App) {
        app.register_cvar(&NET_CONNECT_TIMEOUT_MS)
            .register_cvar(&NET_TIMEOUT_MS)
            .register_cvar(&NET_RETRIES)
            .register_cvar(&NET_CACHE_MAX_MB)
            .register_cvar(&NET_MAX_RESPONSE_MB)
            .init_resource::<NetStats>()
            .add_systems(Stage::First, update_stats);
    }

    fn finish(&self, app: &mut App) {
        // В finish: к этому моменту применены значения CVar из файла и аргументов
        app.insert_resource(Net::new(Settings::from_cvars()));
    }
}

/// Настройки, снятые с CVar при создании клиента.
// На Web соединением и кешем управляет браузер — часть полей там не нужна
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
#[derive(Debug, Clone)]
pub(crate) struct Settings {
    pub connect_timeout: Duration,
    pub timeout: Duration,
    pub retries: u32,
    pub cache_max_bytes: u64,
    pub max_response_bytes: u64,
}

impl Settings {
    fn from_cvars() -> Self {
        Self {
            connect_timeout: Duration::from_millis(NET_CONNECT_TIMEOUT_MS.get()),
            timeout: Duration::from_millis(NET_TIMEOUT_MS.get()),
            retries: NET_RETRIES.get(),
            cache_max_bytes: NET_CACHE_MAX_MB.get() * 1024 * 1024,
            max_response_bytes: NET_MAX_RESPONSE_MB.get() * 1024 * 1024,
        }
    }

    /// Задержка перед повтором номер `attempt` (с 1): 0.5 с, 1 с, 2 с, ... не больше 8 с.
    pub fn backoff(attempt: u32) -> Duration {
        Duration::from_millis(500 * 2u64.pow(attempt.saturating_sub(1).min(4)))
    }
}

/// Счётчики, общие для всех копий клиента.
#[derive(Debug, Default)]
pub(crate) struct Counters {
    pub requests: AtomicU64,
    pub deduplicated: AtomicU64,
    pub cache_hits: AtomicU64,
    pub revalidated: AtomicU64,
    pub bytes_downloaded: AtomicU64,
    pub retries: AtomicU64,
    pub errors: AtomicU64,
}

/// Статистика сети, обновляется в начале кадра.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct NetStats {
    /// Запросов к сети (без учёта дедупликации).
    pub requests: u64,
    /// Запросов, присоединившихся к уже идущей загрузке того же URL.
    pub deduplicated: u64,
    /// Отдано из кеша без обращения к серверу или после ответа 304.
    pub cache_hits: u64,
    /// Из них подтверждено сервером (304 Not Modified).
    pub revalidated: u64,
    pub bytes_downloaded: u64,
    pub retries: u64,
    pub errors: u64,
    /// Загрузок в процессе.
    pub in_flight: usize,
}

type SharedFetch = Shared<SendFuture>;

/// Future загрузки, который можно хранить в ресурсе ECS (тот обязан быть `Send`).
#[cfg(not(target_arch = "wasm32"))]
type SendFuture = BoxedFuture<'static, Result<Bytes, NetError>>;
#[cfg(target_arch = "wasm32")]
type SendFuture = wasm_send::WasmSend<BoxedFuture<'static, Result<Bytes, NetError>>>;

#[cfg(target_arch = "wasm32")]
mod wasm_send {
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll};

    /// Обёртка, объявляющая значение `Send` на Web.
    ///
    /// Future браузерного `fetch` держит JS-объекты и не может быть `Send`. Но в
    /// `wasm32-unknown-unknown` без атомиков второго потока не существует, поэтому
    /// передать значение в другой поток физически невозможно.
    pub struct WasmSend<T>(pub T);

    // [Rust] `unsafe impl` — мы обещаем компилятору то, что он сам проверить не может.
    // Обещание верно только без потоков, поэтому проверяем это на этапе компиляции.
    #[cfg(not(target_feature = "atomics"))]
    #[allow(unsafe_code)]
    unsafe impl<T> Send for WasmSend<T> {}
    #[cfg(not(target_feature = "atomics"))]
    #[allow(unsafe_code)]
    unsafe impl<T> Sync for WasmSend<T> {}

    impl<T: Future + Unpin> Future for WasmSend<T> {
        type Output = T::Output;
        fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            Pin::new(&mut self.0).poll(cx)
        }
    }
}

struct Inner {
    settings: Settings,
    counters: Counters,
    /// Идущие загрузки: URL → общий результат для всех ожидающих.
    in_flight: Mutex<HashMap<String, SharedFetch>>,
    #[cfg(not(target_arch = "wasm32"))]
    backend: native::Backend,
}

/// HTTP-клиент. Дёшево клонируется (внутри `Arc`), можно передавать в задачи.
#[derive(Clone, Resource)]
pub struct Net {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Net {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Net")
            .field("settings", &self.inner.settings)
            .finish_non_exhaustive()
    }
}

impl Net {
    pub(crate) fn new(settings: Settings) -> Self {
        Self {
            inner: Arc::new(Inner {
                #[cfg(not(target_arch = "wasm32"))]
                backend: native::Backend::new(&settings),
                settings,
                counters: Counters::default(),
                in_flight: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Клиент с кешем в заданной папке — для тестов, без каталога ОС.
    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn test_with_cache(dir: std::path::PathBuf) -> Self {
        let settings = Settings {
            connect_timeout: std::time::Duration::from_secs(2),
            timeout: std::time::Duration::from_secs(2),
            retries: 1,
            cache_max_bytes: 1024 * 1024,
            max_response_bytes: 1024 * 1024,
        };
        Self {
            inner: Arc::new(Inner {
                backend: native::Backend::with_cache(
                    &settings,
                    cache::DiskCache::open(dir, settings.cache_max_bytes),
                ),
                settings,
                counters: Counters::default(),
                in_flight: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Загружает URL. Одновременные вызовы с одним URL выполняют один запрос.
    ///
    /// [Rust] Возвращается future: работа начнётся, когда его начнут ждать (`.await`)
    /// внутри задачи. Сам запрос выполняется в пуле IO, поэтому ждать можно откуда угодно.
    pub fn get(&self, url: &str) -> impl std::future::Future<Output = Result<Bytes, NetError>> {
        self.begin(url, None)
    }

    /// То же, что [`get`](Self::get), но байты обязаны совпасть с SHA-256.
    /// Несовпадение не сохраняется в кеш и повторяется, как сетевая ошибка.
    pub fn get_verified(
        &self,
        url: &str,
        sha256: [u8; 32],
    ) -> impl std::future::Future<Output = Result<Bytes, NetError>> {
        self.begin(url, Some(sha256))
    }

    /// Скачивает все строки манифеста. Ошибка одной не отменяет остальные:
    /// наружу вернётся первая, когда закончатся все.
    pub fn preload(
        &self,
        manifest: &Manifest,
    ) -> impl std::future::Future<Output = Result<(), NetError>> {
        let jobs: Vec<_> = manifest
            .entries
            .iter()
            .map(|entry| {
                let net = self.clone();
                let url = entry.url.clone();
                let hash = entry.sha256;
                async move {
                    match hash {
                        Some(hash) => net.get_verified(&url, hash).await.map(|_| ()),
                        None => net.get(&url).await.map(|_| ()),
                    }
                }
            })
            .collect();
        async move {
            futures_util::future::join_all(jobs)
                .await
                .into_iter()
                .collect::<Result<Vec<()>, _>>()
                .map(|_| ())
        }
    }

    fn begin(
        &self,
        url: &str,
        sha256: Option<[u8; 32]>,
    ) -> impl std::future::Future<Output = Result<Bytes, NetError>> {
        let key = flight_key(url, sha256);
        {
            let mut in_flight = self
                .inner
                .in_flight
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if let Some(existing) = in_flight.get(&key) {
                self.inner
                    .counters
                    .deduplicated
                    .fetch_add(1, Ordering::Relaxed);
                existing.clone()
            } else {
                let fetch = self.start(url.to_owned(), sha256, key.clone());
                in_flight.insert(key, fetch.clone());
                fetch
            }
        }
    }

    /// Запускает загрузку в пуле IO и оборачивает в разделяемый future.
    fn start(&self, url: String, sha256: Option<[u8; 32]>, key: String) -> SharedFetch {
        let net = self.clone();
        let task = tasks::spawn_io(async move {
            net.inner.counters.requests.fetch_add(1, Ordering::Relaxed);
            // На native — блокирующий запрос (мы уже в потоке пула IO), на Web — fetch браузера
            #[cfg(not(target_arch = "wasm32"))]
            let result =
                net.inner
                    .backend
                    .fetch(&url, &net.inner.settings, &net.inner.counters, sha256);
            #[cfg(target_arch = "wasm32")]
            let result = web::fetch(&url, &net.inner.settings, &net.inner.counters, sha256).await;
            if let Err(error) = &result {
                net.inner.counters.errors.fetch_add(1, Ordering::Relaxed);
                tracing::warn!("Сеть: {url}: {error}");
            }
            // Загрузка завершена — следующий запрос этого URL пойдёт заново (или из кеша)
            net.inner
                .in_flight
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .remove(&key);
            result
        });
        // [Rust] `Shared` позволяет нескольким ожидающим получить один результат:
        // каждый получает копию (Clone). Bytes — это Arc, копируется только счётчик.
        let boxed: BoxedFuture<'static, Result<Bytes, NetError>> = Box::pin(task);
        #[cfg(target_arch = "wasm32")]
        let boxed = wasm_send::WasmSend(boxed);
        boxed.shared()
    }

    fn snapshot(&self) -> NetStats {
        let c = &self.inner.counters;
        NetStats {
            requests: c.requests.load(Ordering::Relaxed),
            deduplicated: c.deduplicated.load(Ordering::Relaxed),
            cache_hits: c.cache_hits.load(Ordering::Relaxed),
            revalidated: c.revalidated.load(Ordering::Relaxed),
            bytes_downloaded: c.bytes_downloaded.load(Ordering::Relaxed),
            retries: c.retries.load(Ordering::Relaxed),
            errors: c.errors.load(Ordering::Relaxed),
            in_flight: self
                .inner
                .in_flight
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .len(),
        }
    }
}

fn update_stats(net: Option<Res<Net>>, mut stats: ResMut<NetStats>) {
    if let Some(net) = net {
        *stats = net.snapshot();
    }
}

/// Признак, что повтор имеет смысл: ошибка сети, обрыв или чужой хэш.
pub(crate) fn is_retryable(error: &NetError) -> bool {
    match error {
        NetError::Status(code) => is_retryable_status(*code),
        NetError::Timeout | NetError::Transport(_) | NetError::HashMismatch => true,
        NetError::TooLarge(_) | NetError::Manifest(_) => false,
    }
}

fn is_retryable_status(status: u16) -> bool {
    status == 429 || status >= 500
}

/// Разные ожидаемые хэши одного адреса не делят одну загрузку.
fn flight_key(url: &str, sha256: Option<[u8; 32]>) -> String {
    match sha256 {
        None => url.to_owned(),
        Some(hash) => format!("{url} {}", hash::to_hex(&hash)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_manifest_preloads_nothing() {
        let net = Net::new(Settings {
            connect_timeout: Duration::from_secs(1),
            timeout: Duration::from_secs(1),
            retries: 0,
            cache_max_bytes: 0,
            max_response_bytes: 1024,
        });
        engine_core::tasks::block_on(net.preload(&Manifest::default())).unwrap();
    }

    /// Реальная сеть: `cargo test -p engine_net -- --ignored`
    #[test]
    #[ignore = "нужен доступ в интернет"]
    fn download_dedup_and_revalidate() {
        let _app = App::new(); // создаёт пулы задач
        let net = Net::new(Settings::from_cvars());
        let url = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/Box/glTF-Binary/Box.glb";

        // Два одновременных запроса — одна загрузка
        let (a, b) = engine_core::tasks::block_on(async {
            futures_util::future::join(net.get(url), net.get(url)).await
        });
        let (a, b) = (a.unwrap(), b.unwrap());
        assert_eq!(&a[..4], b"glTF");
        assert!(
            Arc::ptr_eq(&a, &b),
            "оба ожидающих получили один и тот же буфер"
        );
        let stats = net.snapshot();
        assert_eq!(stats.deduplicated, 1);

        // Повтор: условный запрос, сервер отвечает 304, данные из кеша
        let c = engine_core::tasks::block_on(net.get(url)).unwrap();
        assert_eq!(a[..], c[..]);
        let stats = net.snapshot();
        println!("{stats:?}");
        assert!(stats.cache_hits >= 1);
    }
}
