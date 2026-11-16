//! Сборка файла из кусков HTTP Range.
//!
//! Ответ 200 заменяет накопленное (сервер прислал файл целиком). Ответ 206
//! дописывается, только если начинается ровно там, где мы остановились.
//! Обрыв соединения кусок не добивает: следующее чтение попросит хвост.

use crate::NetError;

/// Один ответ сервера: кусок тела и то, дочитали ли его.
pub(crate) struct Piece {
    /// Смещение первого байта куска в файле. 0 — ответ без Range (обычно 200).
    pub start: u64,
    pub data: Vec<u8>,
    /// Размер всего файла, если сервер его назвал (`Content-Range` или `Content-Length` у 200).
    pub total: Option<u64>,
    /// Тело этого ответа дочитано, а не оборвалось.
    pub finished: bool,
}

#[derive(Debug, Default)]
pub(crate) struct Assembly {
    pub bytes: Vec<u8>,
    pub total: Option<u64>,
}

pub(crate) enum Step {
    Done,
    /// Кусок принят, файла целиком ещё нет.
    More,
}

impl Assembly {
    /// Заголовок `Range` для хвоста. `None` — качать с начала.
    pub fn range_header(&self) -> Option<String> {
        if self.bytes.is_empty() {
            None
        } else {
            Some(format!("bytes={}-", self.bytes.len()))
        }
    }

    pub fn clear(&mut self) {
        self.bytes.clear();
        self.total = None;
    }

    pub fn absorb(&mut self, piece: Piece) -> Result<Step, NetError> {
        let have = u64::try_from(self.bytes.len()).unwrap_or(u64::MAX);
        if piece.start == 0 {
            self.bytes = piece.data;
        } else if piece.start == have {
            if piece.data.is_empty() && !piece.finished {
                return Err(NetError::Transport("обрыв без новых байт".to_owned()));
            }
            self.bytes.extend(piece.data);
        } else {
            // Дырка или перекрытие: накопленное больше не стыкуется, начнём заново
            self.clear();
            return Err(NetError::Transport(format!(
                "докачка с байта {}, а есть {have}",
                piece.start
            )));
        }
        if let Some(total) = piece.total {
            self.total = Some(total);
        }
        let len = u64::try_from(self.bytes.len()).unwrap_or(u64::MAX);
        if self.total.is_some_and(|total| len > total) {
            self.clear();
            return Err(NetError::Transport(
                "ответ длиннее объявленного размера".to_owned(),
            ));
        }
        // Дочитанный ответ без известной длины — это и есть конец файла
        // (200 без Content-Length или 206 `bytes start-end/*`).
        if piece.finished && self.total.is_none() {
            self.total = Some(len);
        }
        if piece.finished && self.total == Some(len) {
            Ok(Step::Done)
        } else {
            Ok(Step::More)
        }
    }
}

/// `Content-Range: bytes START-END/TOTAL` или `bytes START-END/*`.
pub(crate) fn parse_content_range(header: &str) -> Option<(u64, u64, Option<u64>)> {
    let rest = header.trim();
    let rest = rest
        .strip_prefix("bytes ")
        .or_else(|| rest.strip_prefix("bytes="))?;
    let (range, total) = rest.split_once('/')?;
    let (start, end) = range.split_once('-')?;
    let start = start.parse().ok()?;
    let end = end.parse().ok()?;
    if end < start {
        return None;
    }
    let total = if total == "*" {
        None
    } else {
        Some(total.parse().ok()?)
    };
    Some((start, end, total))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn piece(start: u64, data: &str, total: Option<u64>, finished: bool) -> Piece {
        Piece {
            start,
            data: data.as_bytes().to_vec(),
            total,
            finished,
        }
    }

    #[test]
    fn resume_appends_and_full_response_replaces() {
        let mut file = Assembly::default();
        assert!(file.range_header().is_none());
        file.absorb(piece(0, "0123", Some(8), false)).unwrap();
        assert_eq!(file.range_header().as_deref(), Some("bytes=4-"));
        assert!(matches!(
            file.absorb(piece(4, "4567", Some(8), true)).unwrap(),
            Step::Done
        ));
        assert_eq!(file.bytes, b"01234567");

        // Сервер проигнорировал Range и прислал файл заново
        assert!(matches!(
            file.absorb(piece(0, "abcdefgh", Some(8), true)).unwrap(),
            Step::Done
        ));
        assert_eq!(file.bytes, b"abcdefgh");
    }

    #[test]
    fn gap_drops_partial() {
        let mut file = Assembly {
            bytes: b"0123".to_vec(),
            total: Some(8),
        };
        assert!(file.absorb(piece(2, "xx", Some(8), true)).is_err());
        assert_eq!(file.bytes.as_slice(), b"");
    }

    #[test]
    fn content_range_header() {
        assert_eq!(
            parse_content_range("bytes 8-15/16"),
            Some((8, 15, Some(16)))
        );
        assert_eq!(parse_content_range("bytes 0-3/*"), Some((0, 3, None)));
        assert!(parse_content_range("bytes 5-1/10").is_none());
    }
}
