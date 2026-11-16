//! Дисковый кеш HTTP-ответов.
//!
//! Для каждого URL два файла: `<хэш>.bin` (тело) и `<хэш>.meta` (`ETag`, `Last-Modified`).
//! Недокачанный файл лежит отдельно (`<хэш>.part`): его нельзя отдать как готовый ответ,
//! следующий запрос продолжает его через `Range`.
//! Время изменения `.bin` обновляется при каждом чтении — по нему при переполнении удаляются
//! давно не использованные записи (LRU). Запись атомарная: во временный файл, затем rename —
//! при падении посреди записи в кеше не окажется обрезанного файла.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Debug, Clone, Default)]
pub(crate) struct CacheEntry {
    pub etag: Option<String>,
    pub last_modified: Option<String>,
}

/// Недокачанное тело. `total` — размер файла, если сервер его уже назвал.
#[derive(Debug, Clone)]
pub(crate) struct Partial {
    pub bytes: Vec<u8>,
    pub total: Option<u64>,
    pub etag: Option<String>,
}

pub(crate) struct DiskCache {
    dir: PathBuf,
    max_bytes: u64,
}

impl DiskCache {
    pub fn open(dir: PathBuf, max_bytes: u64) -> Option<Self> {
        if let Err(error) = fs::create_dir_all(&dir) {
            tracing::warn!("Кеш: не удалось создать {}: {error}", dir.display());
            return None;
        }
        tracing::info!("Кеш сети: {}", dir.display());
        Some(Self { dir, max_bytes })
    }

    pub fn lookup(&self, url: &str) -> Option<CacheEntry> {
        let (body, meta) = self.paths(url);
        if !body.exists() {
            return None;
        }
        let text = fs::read_to_string(meta).unwrap_or_default();
        let mut lines = text.lines();
        let read = |line: Option<&str>| line.filter(|l| !l.is_empty()).map(str::to_owned);
        Some(CacheEntry {
            etag: read(lines.next()),
            last_modified: read(lines.next()),
        })
    }

    pub fn read(&self, url: &str) -> io::Result<Vec<u8>> {
        let (body, _) = self.paths(url);
        let bytes = fs::read(&body)?;
        // Отмечаем использование для LRU; ошибка не критична
        let _ = fs::File::options()
            .write(true)
            .open(&body)
            .and_then(|file| file.set_modified(SystemTime::now()));
        Ok(bytes)
    }

    pub fn store(&self, url: &str, bytes: &[u8], entry: &CacheEntry) -> io::Result<()> {
        if bytes.len() as u64 > self.max_bytes {
            return Ok(()); // не поместится целиком — не кешируем
        }
        let (body, meta) = self.paths(url);
        let meta_text = format!(
            "{}\n{}\n{url}\n",
            entry.etag.as_deref().unwrap_or(""),
            entry.last_modified.as_deref().unwrap_or("")
        );
        write_atomic(&body, bytes)?;
        write_atomic(&meta, meta_text.as_bytes())?;
        // Готовый файл заменяет хвост докачки
        self.discard_partial(url);
        self.evict()
    }

    pub fn read_partial(&self, url: &str) -> Option<Partial> {
        let (body, meta) = self.partial_paths(url);
        let bytes = fs::read(&body).ok()?;
        if bytes.is_empty() {
            return None;
        }
        let text = fs::read_to_string(meta).unwrap_or_default();
        let mut lines = text.lines();
        let total = lines
            .next()
            .filter(|line| !line.is_empty())
            .and_then(|line| line.parse().ok());
        let etag = lines
            .next()
            .filter(|line| !line.is_empty())
            .map(str::to_owned);
        Some(Partial { bytes, total, etag })
    }

    pub fn store_partial(
        &self,
        url: &str,
        bytes: &[u8],
        total: Option<u64>,
        etag: Option<&str>,
    ) -> io::Result<()> {
        if bytes.is_empty() || bytes.len() as u64 > self.max_bytes {
            self.discard_partial(url);
            return Ok(());
        }
        let (body, meta) = self.partial_paths(url);
        let total = total.map(|value| value.to_string()).unwrap_or_default();
        let meta_text = format!("{total}\n{}\n{url}\n", etag.unwrap_or(""));
        write_atomic(&body, bytes)?;
        write_atomic(&meta, meta_text.as_bytes())?;
        self.evict()
    }

    pub fn discard_partial(&self, url: &str) {
        let (body, meta) = self.partial_paths(url);
        let _ = fs::remove_file(body);
        let _ = fs::remove_file(meta);
    }

    /// Удаляет и готовую копию, и хвост. Нужно, когда хэш не совпал.
    pub fn remove(&self, url: &str) {
        let (body, meta) = self.paths(url);
        let _ = fs::remove_file(body);
        let _ = fs::remove_file(meta);
        self.discard_partial(url);
    }

    /// Удаляет давно не использованные записи, пока кеш больше лимита.
    fn evict(&self) -> io::Result<()> {
        let mut entries = Vec::new();
        let mut total = 0u64;
        for item in fs::read_dir(&self.dir)? {
            let path = item?.path();
            let kind = path.extension().and_then(|ext| ext.to_str());
            if !matches!(kind, Some("bin" | "part")) {
                continue;
            }
            let meta = fs::metadata(&path)?;
            total += meta.len();
            entries.push((
                meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                meta.len(),
                path,
            ));
        }
        if total <= self.max_bytes {
            return Ok(());
        }
        // [Rust] Кортежи сравниваются по полям слева направо — сортировка по времени
        entries.sort();
        for (_, len, path) in entries {
            if total <= self.max_bytes {
                break;
            }
            let sidecar = if path.extension().is_some_and(|ext| ext == "part") {
                path.with_extension("partmeta")
            } else {
                path.with_extension("meta")
            };
            let _ = fs::remove_file(sidecar);
            fs::remove_file(&path)?;
            total = total.saturating_sub(len);
        }
        Ok(())
    }

    fn paths(&self, url: &str) -> (PathBuf, PathBuf) {
        let name = Self::name(url);
        (
            self.dir.join(format!("{name}.bin")),
            self.dir.join(format!("{name}.meta")),
        )
    }

    fn partial_paths(&self, url: &str) -> (PathBuf, PathBuf) {
        let name = Self::name(url);
        (
            self.dir.join(format!("{name}.part")),
            self.dir.join(format!("{name}.partmeta")),
        )
    }

    fn name(url: &str) -> String {
        format!("{:016x}", fnv1a(url.as_bytes()))
    }
}

fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)
}

/// FNV-1a: простой стабильный хэш для имён файлов.
/// [Rust] `std::hash::DefaultHasher` не подходит: его алгоритм может поменяться между
/// версиями Rust, и имена файлов кеша перестанут совпадать.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_cache(max_bytes: u64) -> (DiskCache, PathBuf) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "engine-net-test-{}-{max_bytes}-{n}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        (DiskCache::open(dir.clone(), max_bytes).unwrap(), dir)
    }

    #[test]
    fn store_lookup_read() {
        let (cache, dir) = temp_cache(1024);
        let entry = CacheEntry {
            etag: Some("\"abc\"".to_owned()),
            last_modified: None,
        };
        cache.store("https://x/a", b"hello", &entry).unwrap();
        let found = cache.lookup("https://x/a").unwrap();
        assert_eq!(found.etag.as_deref(), Some("\"abc\""));
        assert_eq!(cache.read("https://x/a").unwrap(), b"hello");
        assert!(cache.lookup("https://x/b").is_none());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn evicts_least_recently_used() {
        let (cache, dir) = temp_cache(25);
        let entry = CacheEntry::default();
        cache.store("u1", &[1; 10], &entry).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        cache.store("u2", &[2; 10], &entry).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        cache.read("u1").unwrap(); // u1 теперь свежее u2
        std::thread::sleep(std::time::Duration::from_millis(20));
        cache.store("u3", &[3; 10], &entry).unwrap(); // 30 байт > 25 — вытесняется u2
        assert!(cache.lookup("u1").is_some());
        assert!(cache.lookup("u2").is_none());
        assert!(cache.lookup("u3").is_some());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn partial_is_not_a_finished_entry() {
        let (cache, dir) = temp_cache(1024);
        cache
            .store_partial("https://x/a", b"0123", Some(8), Some("\"v\""))
            .unwrap();
        assert!(cache.lookup("https://x/a").is_none());
        let partial = cache.read_partial("https://x/a").unwrap();
        assert_eq!(partial.bytes, b"0123");
        assert_eq!(partial.total, Some(8));
        assert_eq!(partial.etag.as_deref(), Some("\"v\""));
        cache
            .store(
                "https://x/a",
                b"01234567",
                &CacheEntry {
                    etag: Some("\"v\"".to_owned()),
                    last_modified: None,
                },
            )
            .unwrap();
        assert!(cache.read_partial("https://x/a").is_none());
        assert_eq!(cache.read("https://x/a").unwrap(), b"01234567");
        fs::remove_dir_all(dir).unwrap();
    }
}
