//! Загрузка на native: блокирующий `ureq` в потоке пула IO + дисковый кеш.
//!
//! Оборванное тело сохраняется и следующий запрос просит остаток (`Range`).
//! Если сервер не умеет докачку и отвечает 200, накопленное заменяется целиком.

use std::io::Read;
use std::sync::atomic::Ordering;

use crate::cache::{CacheEntry, DiskCache};
use crate::hash;
use crate::resume::{Assembly, Piece, Step, parse_content_range};
use crate::{Bytes, Counters, NetError, Settings, is_retryable};

pub(crate) struct Backend {
    agent: ureq::Agent,
    cache: Option<DiskCache>,
}

impl Backend {
    pub fn new(settings: &Settings) -> Self {
        let cache = if settings.cache_max_bytes == 0 {
            None
        } else {
            engine_platform::paths::cache_dir()
                .and_then(|dir| DiskCache::open(dir.join("net"), settings.cache_max_bytes))
        };
        if cache.is_none() && settings.cache_max_bytes > 0 {
            tracing::warn!("Сеть: дисковый кеш недоступен");
        }
        Self::with_cache(settings, cache)
    }

    pub(crate) fn with_cache(settings: &Settings, cache: Option<DiskCache>) -> Self {
        // [Rust] ureq — синхронный HTTP-клиент на чистом Rust, TLS через rustls (без OpenSSL).
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_connect(Some(settings.connect_timeout))
            .timeout_global(Some(settings.timeout))
            // Коды 4xx/5xx — не ошибка транспорта: разбираем статус сами (304, повторы)
            .http_status_as_error(false)
            .user_agent("istok-engine")
            .build()
            .into();
        Self { agent, cache }
    }

    pub fn fetch(
        &self,
        url: &str,
        settings: &Settings,
        counters: &Counters,
        expected: Option<[u8; 32]>,
    ) -> Result<Bytes, NetError> {
        let _zone = tracing::trace_span!("net_fetch", url).entered();
        // Готовая копия важнее хвоста: его проверит условный запрос
        let (mut cached, mut assembly, mut partial_etag) = self.start_state(url);
        let mut failures = 0u32;
        let mut short_ranges = 0u32;
        let mut last_error = NetError::Transport("нет попыток".to_owned());
        while failures <= settings.retries {
            if failures > 0 {
                counters.retries.fetch_add(1, Ordering::Relaxed);
                std::thread::sleep(Settings::backoff(failures));
            }
            if !assembly.bytes.is_empty() {
                tracing::debug!("Сеть: докачка {url}, уже {} байт", assembly.bytes.len());
            }
            let range = assembly.range_header();
            match self.request(
                url,
                range.as_deref(),
                cached.as_ref(),
                partial_etag.as_deref(),
                settings,
            ) {
                Ok(Outcome::NotModified) => {
                    match self.cached_bytes(url, cached.as_ref(), expected) {
                        Ok(Some(bytes)) => {
                            counters.cache_hits.fetch_add(1, Ordering::Relaxed);
                            counters.revalidated.fetch_add(1, Ordering::Relaxed);
                            return Ok(bytes);
                        }
                        Ok(None) => {
                            last_error = NetError::Transport("304 без данных в кеше".to_owned());
                        }
                        Err(error) => {
                            self.forget(url);
                            cached = None;
                            last_error = error;
                        }
                    }
                    failures += 1;
                }
                Ok(Outcome::Reset) => {
                    // Сервер не принял хвост — качаем заново
                    assembly.clear();
                    partial_etag = None;
                    self.discard_partial(url);
                    failures += 1;
                }
                Ok(Outcome::Piece { piece, entry }) => {
                    counters
                        .bytes_downloaded
                        .fetch_add(piece.data.len() as u64, Ordering::Relaxed);
                    match self.accept_piece(
                        url,
                        &mut ResumeState {
                            assembly: &mut assembly,
                            partial_etag: &mut partial_etag,
                            cached: &mut cached,
                            short_ranges: &mut short_ranges,
                        },
                        expected,
                        piece,
                        entry,
                    ) {
                        PieceRound::Done(bytes) => return Ok(bytes),
                        PieceRound::Again {
                            error,
                            count_failure,
                            give_up,
                        } => {
                            if let Some(error) = error {
                                last_error = error;
                            }
                            if give_up {
                                failures = settings.retries.saturating_add(1);
                            } else if count_failure {
                                failures += 1;
                            }
                        }
                    }
                }
                Err(error) if !is_retryable(&error) => return Err(error),
                Err(error) => {
                    last_error = error;
                    failures += 1;
                }
            }
        }

        // Хвост докачки не отдаём: это не целый файл.
        self.offline(url, cached.as_ref(), expected, last_error, counters)
    }

    fn start_state(&self, url: &str) -> (Option<CacheEntry>, Assembly, Option<String>) {
        let cached = self.cache.as_ref().and_then(|cache| cache.lookup(url));
        let mut assembly = Assembly::default();
        let mut partial_etag = None;
        if cached.is_some() {
            self.discard_partial(url);
        } else if let Some(partial) = self
            .cache
            .as_ref()
            .and_then(|cache| cache.read_partial(url))
        {
            partial_etag = partial.etag;
            assembly.total = partial.total;
            assembly.bytes = partial.bytes;
        }
        (cached, assembly, partial_etag)
    }

    /// Сеть недоступна, но есть старая копия — лучше она, чем ничего.
    fn offline(
        &self,
        url: &str,
        cached: Option<&CacheEntry>,
        expected: Option<[u8; 32]>,
        last_error: NetError,
        counters: &Counters,
    ) -> Result<Bytes, NetError> {
        if !matches!(last_error, NetError::Status(code) if code < 500)
            && let Ok(Some(bytes)) = self.cached_bytes(url, cached, expected)
        {
            tracing::warn!("Сеть: {url}: {last_error}, используется кеш");
            counters.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(bytes);
        }
        Err(last_error)
    }

    fn accept_piece(
        &self,
        url: &str,
        state: &mut ResumeState<'_>,
        expected: Option<[u8; 32]>,
        piece: Piece,
        entry: CacheEntry,
    ) -> PieceRound {
        if let Some(etag) = &entry.etag {
            *state.partial_etag = Some(etag.clone());
        }
        let finished = piece.finished;
        let added = piece.data.len();
        match state.assembly.absorb(piece) {
            Ok(Step::Done) => {
                if let Err(error) = hash::check(&state.assembly.bytes, expected) {
                    state.assembly.clear();
                    *state.partial_etag = None;
                    *state.cached = None;
                    self.forget(url);
                    return PieceRound::Again {
                        error: Some(error),
                        count_failure: true,
                        give_up: false,
                    };
                }
                self.store(url, &state.assembly.bytes, &entry);
                PieceRound::Done(std::mem::take(&mut state.assembly.bytes).into())
            }
            Ok(Step::More) if finished && added > 0 => {
                // Сервер прислал не весь файл, но соединение цело
                self.store_partial(url, state.assembly, state.partial_etag.as_deref());
                *state.short_ranges += 1;
                PieceRound::Again {
                    error: (*state.short_ranges > 32)
                        .then(|| NetError::Transport("слишком много кусков докачки".to_owned())),
                    count_failure: false,
                    give_up: *state.short_ranges > 32,
                }
            }
            Ok(Step::More) => {
                self.store_partial(url, state.assembly, state.partial_etag.as_deref());
                PieceRound::Again {
                    error: Some(NetError::Transport("соединение оборвалось".to_owned())),
                    count_failure: true,
                    give_up: false,
                }
            }
            Err(error) => {
                if state.assembly.bytes.is_empty() {
                    self.discard_partial(url);
                } else {
                    self.store_partial(url, state.assembly, state.partial_etag.as_deref());
                }
                PieceRound::Again {
                    error: Some(error),
                    count_failure: true,
                    give_up: false,
                }
            }
        }
    }

    fn request(
        &self,
        url: &str,
        range: Option<&str>,
        cached: Option<&CacheEntry>,
        partial_etag: Option<&str>,
        settings: &Settings,
    ) -> Result<Outcome, NetError> {
        let mut request = self.agent.get(url);
        if let Some(range) = range {
            request = request.header("Range", range);
            // Если файл на сервере сменился, придёт 200 целиком, а не хвост к старой копии
            if let Some(etag) = partial_etag {
                request = request.header("If-Range", etag);
            }
        } else if let Some(entry) = cached {
            if let Some(etag) = &entry.etag {
                request = request.header("If-None-Match", etag);
            }
            if let Some(modified) = &entry.last_modified {
                request = request.header("If-Modified-Since", modified);
            }
        }
        let mut response = request.call().map_err(map_error)?;
        let status = response.status().as_u16();
        if status == 304 {
            return Ok(Outcome::NotModified);
        }
        if status == 416 {
            return Ok(Outcome::Reset);
        }
        if !(200..300).contains(&status) {
            return Err(NetError::Status(status));
        }
        let header = |name: &str| {
            response
                .headers()
                .get(name)
                .and_then(|value| value.to_str().ok())
                .map(str::to_owned)
        };
        let entry = CacheEntry {
            etag: header("etag"),
            last_modified: header("last-modified"),
        };
        let content_length = header("content-length").and_then(|text| text.parse().ok());
        let content_range = header("content-range");
        let (start, total) = if status == 206 {
            match content_range.as_deref().and_then(parse_content_range) {
                Some((start, _end, total)) => (start, total),
                None => (range_start(range), None),
            }
        } else {
            (0, content_length)
        };
        if total.is_some_and(|total| total > settings.max_response_bytes)
            || content_length.is_some_and(|len| len > settings.max_response_bytes)
        {
            return Err(NetError::TooLarge(settings.max_response_bytes));
        }
        let mut reader = response
            .body_mut()
            .with_config()
            .limit(settings.max_response_bytes)
            .reader();
        let (data, read) = read_body(&mut reader, settings.max_response_bytes);
        let finished = match read {
            Err(NetError::TooLarge(limit)) => return Err(NetError::TooLarge(limit)),
            Err(_) => false,
            Ok(()) => content_length.is_none_or(|len| data.len() as u64 >= len),
        };
        Ok(Outcome::Piece {
            piece: Piece {
                start,
                data,
                total,
                finished,
            },
            entry,
        })
    }

    fn cached_bytes(
        &self,
        url: &str,
        cached: Option<&CacheEntry>,
        expected: Option<[u8; 32]>,
    ) -> Result<Option<Bytes>, NetError> {
        let Some(bytes) = self.read_cached(url, cached) else {
            return Ok(None);
        };
        hash::check(&bytes, expected)?;
        Ok(Some(bytes))
    }

    fn read_cached(&self, url: &str, cached: Option<&CacheEntry>) -> Option<Bytes> {
        cached?;
        self.cache.as_ref()?.read(url).ok().map(Bytes::from)
    }

    fn store(&self, url: &str, bytes: &[u8], entry: &CacheEntry) {
        if let Some(cache) = &self.cache
            && let Err(error) = cache.store(url, bytes, entry)
        {
            tracing::warn!("Кеш: не удалось сохранить {url}: {error}");
        }
    }

    fn store_partial(&self, url: &str, assembly: &Assembly, etag: Option<&str>) {
        let Some(cache) = &self.cache else {
            return;
        };
        if let Err(error) = cache.store_partial(url, &assembly.bytes, assembly.total, etag) {
            tracing::warn!("Кеш: не удалось сохранить хвост {url}: {error}");
        }
    }

    fn discard_partial(&self, url: &str) {
        if let Some(cache) = &self.cache {
            cache.discard_partial(url);
        }
    }

    fn forget(&self, url: &str) {
        if let Some(cache) = &self.cache {
            cache.remove(url);
        }
    }
}

struct ResumeState<'a> {
    assembly: &'a mut Assembly,
    partial_etag: &'a mut Option<String>,
    cached: &'a mut Option<CacheEntry>,
    short_ranges: &'a mut u32,
}

enum PieceRound {
    Done(Bytes),
    Again {
        error: Option<NetError>,
        count_failure: bool,
        give_up: bool,
    },
}

enum Outcome {
    NotModified,
    /// `416`: хвост серверу не подошёл.
    Reset,
    Piece {
        piece: Piece,
        entry: CacheEntry,
    },
}

fn range_start(header: Option<&str>) -> u64 {
    header
        .and_then(|header| header.strip_prefix("bytes="))
        .and_then(|rest| rest.split_once('-'))
        .and_then(|(start, _)| start.parse().ok())
        .unwrap_or(0)
}

fn read_body(reader: &mut impl Read, limit: u64) -> (Vec<u8>, Result<(), NetError>) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 16 * 1024];
    loop {
        match reader.read(&mut chunk) {
            Ok(0) => return (buf, Ok(())),
            Ok(read) => {
                let len = u64::try_from(buf.len()).unwrap_or(u64::MAX);
                let add = u64::try_from(read).unwrap_or(u64::MAX);
                if len.saturating_add(add) > limit {
                    return (buf, Err(NetError::TooLarge(limit)));
                }
                buf.extend_from_slice(&chunk[..read]);
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => {
                let mapped = if error.kind() == std::io::ErrorKind::TimedOut {
                    NetError::Timeout
                } else {
                    NetError::Transport(error.to_string())
                };
                return (buf, Err(mapped));
            }
        }
    }
}

fn map_error(error: ureq::Error) -> NetError {
    match error {
        ureq::Error::Timeout(_) => NetError::Timeout,
        ureq::Error::StatusCode(code) => NetError::Status(code),
        // У ошибки ввода-вывода Display короткий («io») — берём текст исходной ошибки ОС
        ureq::Error::Io(error) => NetError::Transport(format!("ввод-вывод: {error}")),
        other => NetError::Transport(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    use super::*;
    use crate::hash::sha256;

    struct Server {
        port: u16,
        stop: Arc<AtomicBool>,
        ranges: Arc<Mutex<Vec<Option<u64>>>>,
        thread: Option<thread::JoinHandle<()>>,
    }

    impl Server {
        /// `drop_first` — сколько байт отдать в первом ответе без `Range`, затем закрыть соединение.
        fn spawn(body: Vec<u8>, drop_first: Option<usize>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            let stop = Arc::new(AtomicBool::new(false));
            let ranges = Arc::new(Mutex::new(Vec::new()));
            let stop_thread = Arc::clone(&stop);
            let ranges_thread = Arc::clone(&ranges);
            let thread = thread::spawn(move || {
                listener.set_nonblocking(true).unwrap();
                while !stop_thread.load(Ordering::Relaxed) {
                    match listener.accept() {
                        Ok((stream, _)) => {
                            let _ = handle(stream, &body, drop_first, &ranges_thread);
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            thread::sleep(Duration::from_millis(5));
                        }
                        Err(error) => panic!("{error}"),
                    }
                }
            });
            Self {
                port,
                stop,
                ranges,
                thread: Some(thread),
            }
        }

        fn url(&self) -> String {
            format!("http://127.0.0.1:{}/file", self.port)
        }

        fn ranges(&self) -> Vec<Option<u64>> {
            self.ranges.lock().unwrap().clone()
        }
    }

    impl Drop for Server {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(thread) = self.thread.take() {
                let _ = thread.join();
            }
        }
    }

    fn handle(
        mut stream: TcpStream,
        body: &[u8],
        drop_first: Option<usize>,
        ranges: &Mutex<Vec<Option<u64>>>,
    ) -> std::io::Result<()> {
        stream.set_read_timeout(Some(Duration::from_secs(2)))?;
        let mut buf = Vec::new();
        let mut tmp = [0u8; 1024];
        while !buf.windows(4).any(|window| window == b"\r\n\r\n") {
            let read = stream.read(&mut tmp)?;
            if read == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..read]);
            if buf.len() > 8192 {
                break;
            }
        }
        let head = String::from_utf8_lossy(&buf);
        let range = head.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.trim()
                .eq_ignore_ascii_case("range")
                .then(|| value.trim().to_owned())
        });
        let start = range.as_deref().and_then(parse_range_start);
        ranges.lock().unwrap().push(start);
        let _ = stream.set_write_timeout(Some(Duration::from_secs(2)));
        if let Some(start) = start {
            let start = usize::try_from(start).unwrap_or(body.len()).min(body.len());
            let rest = &body[start..];
            let end = body.len().saturating_sub(1);
            write!(
                stream,
                "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {start}-{end}/{}\r\nETag: \"v1\"\r\nConnection: close\r\n\r\n",
                rest.len(),
                body.len()
            )?;
            stream.write_all(rest)?;
        } else if let Some(drop_after) = drop_first {
            let sent = drop_after.min(body.len());
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nETag: \"v1\"\r\nAccept-Ranges: bytes\r\nConnection: close\r\n\r\n",
                body.len()
            )?;
            stream.write_all(&body[..sent])?;
        } else {
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nETag: \"v1\"\r\nConnection: close\r\n\r\n",
                body.len()
            )?;
            stream.write_all(body)?;
        }
        stream.flush()
    }

    fn parse_range_start(value: &str) -> Option<u64> {
        let rest = value.strip_prefix("bytes=")?;
        let (start, _) = rest.split_once('-')?;
        if start.is_empty() {
            return None;
        }
        start.parse().ok()
    }

    fn test_settings() -> Settings {
        Settings {
            connect_timeout: Duration::from_secs(2),
            timeout: Duration::from_secs(2),
            retries: 3,
            cache_max_bytes: 1024 * 1024,
            max_response_bytes: 1024 * 1024,
        }
    }

    fn backend(dir: &std::path::Path) -> Backend {
        let settings = test_settings();
        Backend::with_cache(
            &settings,
            DiskCache::open(dir.to_path_buf(), settings.cache_max_bytes),
        )
    }

    #[test]
    fn resumes_after_a_dropped_connection() {
        let body = b"0123456789abcdef".to_vec();
        let server = Server::spawn(body.clone(), Some(4));
        let dir = std::env::temp_dir().join(format!("engine-net-resume-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let settings = test_settings();
        let fetched = backend(&dir)
            .fetch(
                &server.url(),
                &settings,
                &Counters::default(),
                Some(sha256(&body)),
            )
            .unwrap();
        assert_eq!(fetched.as_ref(), body.as_slice());
        assert_eq!(server.ranges(), vec![None, Some(4)]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn hash_mismatch_is_not_cached() {
        let body = b"not-the-expected-bytes".to_vec();
        let server = Server::spawn(body, None);
        let dir = std::env::temp_dir().join(format!("engine-net-hash-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut settings = test_settings();
        settings.retries = 1;
        let error = Backend::with_cache(
            &settings,
            DiskCache::open(dir.clone(), settings.cache_max_bytes),
        )
        .fetch(
            &server.url(),
            &settings,
            &Counters::default(),
            Some(sha256(b"other")),
        )
        .unwrap_err();
        assert_eq!(error, NetError::HashMismatch);
        assert!(
            DiskCache::open(dir.clone(), settings.cache_max_bytes)
                .unwrap()
                .lookup(&server.url())
                .is_none()
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn preload_checks_the_manifest_hash() {
        let _app = engine_core::App::new();
        let body = b"preload-body".to_vec();
        let server = Server::spawn(body.clone(), None);
        let dir = std::env::temp_dir().join(format!("engine-net-preload-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let net = crate::Net::test_with_cache(dir.clone());
        let manifest = crate::Manifest::parse(&format!(
            "{} {}\n",
            server.url(),
            crate::hash::to_hex(&sha256(&body))
        ))
        .unwrap();
        engine_core::tasks::block_on(net.preload(&manifest)).unwrap();
        let bad = crate::Manifest::parse(&format!(
            "{} {}\n",
            server.url(),
            crate::hash::to_hex(&sha256(b"other"))
        ))
        .unwrap();
        // Кеш уже есть, но хэш другой — отдавать его нельзя
        let error = engine_core::tasks::block_on(net.preload(&bad)).unwrap_err();
        assert_eq!(error, NetError::HashMismatch);
        let _ = std::fs::remove_dir_all(dir);
    }
}
