use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context, Poll, Waker};

/// Устройство GPU: всё, что нужно для работы с видеокартой.
#[derive(Debug)]
pub(crate) struct Gpu {
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// Причина потери устройства. Пишет колбэк wgpu (из любого потока), читает рендер.
    pub lost: LostFlag,
}

/// Устройство потеряно: `Some(причина)`. У каждого устройства свой флаг, поэтому колбэк
/// старого устройства (он срабатывает и при его уничтожении) новое не трогает.
#[derive(Debug, Clone, Default)]
pub(crate) struct LostFlag(Arc<Mutex<Option<String>>>);

impl LostFlag {
    pub fn set(&self, reason: String) {
        let mut slot = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        slot.get_or_insert(reason);
    }

    pub fn is_set(&self) -> bool {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some()
    }

    pub fn reason(&self) -> Option<String> {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

/// Колбэк потери и обработчик ошибок. После потери любой вызов wgpu даёт ошибку
/// «устройство недействительно»; обработчик по умолчанию на ней паникует, а нам нужно
/// дожить до начала следующего кадра и пересоздать устройство.
fn watch_device(device: &wgpu::Device) -> LostFlag {
    let lost = LostFlag::default();
    let flag = lost.clone();
    device.set_device_lost_callback(move |reason, message| {
        flag.set(format!("{reason:?}: {message}"));
    });
    let flag = lost.clone();
    device.on_uncaptured_error(Arc::new(move |error| {
        if flag.is_set() {
            tracing::debug!("wgpu после потери устройства: {error}");
            return;
        }
        // Как обработчик по умолчанию: ошибка валидации — баг в коде рендера
        panic!("wgpu: {error}");
    }));
    lost
}

/// Асинхронная инициализация GPU.
///
/// [Rust] Запрос адаптера и устройства в `wgpu` — асинхронный (`Future`). На native он
/// завершается сразу, а в браузере ждёт промис JavaScript. Блокирующе ждать в браузере
/// нельзя, поэтому future хранится здесь и опрашивается (`poll`) каждый кадр, пока не
/// будет готов. Исполнитель (executor) для этого не нужен.
pub(crate) struct GpuInit {
    future: InitFuture,
}

// [Rust] `Pin<Box<dyn Future>>` — future в куче с закреплённым адресом. Async-код может
// хранить ссылки на собственные поля, поэтому перемещать его в памяти после первого
// poll нельзя — Pin это гарантирует.
//
// На native future должен быть `Send`: рендер вместе с ним переезжает в свой поток.
// На Web объекты wgpu — это JS-объекты, они не `Send`, но и потоков там нет.
#[cfg(not(target_arch = "wasm32"))]
type InitFuture = Pin<Box<dyn Future<Output = Result<Gpu, String>> + Send>>;
#[cfg(target_arch = "wasm32")]
type InitFuture = Pin<Box<dyn Future<Output = Result<Gpu, String>>>>;

impl GpuInit {
    /// `surface` — `None` для рендера без окна.
    pub fn start(instance: &wgpu::Instance, surface: Option<&wgpu::Surface<'static>>) -> Self {
        // Future адаптера создаётся до async-блока: он не держит ссылку на surface,
        // поэтому весь блок может жить сколько угодно ('static).
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface,
            // Остальное — по умолчанию (программный адаптер не нужен и т.п.)
            ..Default::default()
        });
        let future = async move {
            let adapter = adapter.await.map_err(|e| format!("адаптер: {e}"))?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("engine"),
                    // Лимиты, которые поддерживает этот адаптер (GLES3 и мобильные GPU слабее desktop)
                    required_limits: adapter.limits(),
                    // Метки времени и сжатые текстуры — только то, что умеет этот адаптер.
                    // Без явного запроса BC/ASTC/ETC2 устройство их не включит, и KTX2
                    // всегда разжимался бы в RGBA.
                    required_features: adapter.features()
                        & (wgpu::Features::TIMESTAMP_QUERY
                            | wgpu::Features::TEXTURE_COMPRESSION_BC
                            | wgpu::Features::TEXTURE_COMPRESSION_ASTC
                            | wgpu::Features::TEXTURE_COMPRESSION_ETC2),
                    ..Default::default()
                })
                .await
                .map_err(|e| format!("устройство: {e}"))?;
            let lost = watch_device(&device);
            Ok(Gpu {
                adapter,
                device,
                queue,
                lost,
            })
        };
        Self {
            future: Box::pin(future),
        }
    }

    /// Проверяет готовность. `None` — ещё ждём.
    pub fn poll(&mut self) -> Option<Result<Gpu, String>> {
        // Waker::noop: нас не нужно будить — опросим сами в следующем кадре
        let mut context = Context::from_waker(Waker::noop());
        match self.future.as_mut().poll(&mut context) {
            Poll::Ready(result) => Some(result),
            Poll::Pending => None,
        }
    }
}

impl std::fmt::Debug for GpuInit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("GpuInit")
    }
}
