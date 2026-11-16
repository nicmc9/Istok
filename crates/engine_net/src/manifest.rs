//! Список файлов, которые стоит скачать заранее.
//!
//! Текст, по строке на файл: адрес и необязательный SHA-256 (64 шестнадцатеричных
//! символа). Пустые строки и строки с `#` пропускаются.
//!
//! ```text
//! # уровень
//! https://cdn.example.com/hero.glb 2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae
//! https://cdn.example.com/music.ogg
//! ```

use crate::NetError;
use crate::hash::parse_sha256;

/// Одна строка манифеста.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestEntry {
    pub url: String,
    pub sha256: Option<[u8; 32]>,
}

/// Набор адресов для [`crate::Net::preload`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Manifest {
    pub entries: Vec<ManifestEntry>,
}

impl Manifest {
    pub fn parse(text: &str) -> Result<Self, NetError> {
        let mut entries = Vec::new();
        for (index, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut parts = line.split_whitespace();
            let url = parts.next().unwrap_or(line);
            if !(url.starts_with("http://") || url.starts_with("https://")) {
                return Err(NetError::Manifest(format!(
                    "строка {}: нужен адрес http(s)",
                    index + 1
                )));
            }
            let sha256 = match parts.next() {
                None => None,
                Some(hex) => Some(parse_sha256(hex).ok_or_else(|| {
                    NetError::Manifest(format!(
                        "строка {}: хэш — 64 шестнадцатеричных символа",
                        index + 1
                    ))
                })?),
            };
            if parts.next().is_some() {
                return Err(NetError::Manifest(format!(
                    "строка {}: лишние поля",
                    index + 1
                )));
            }
            if entries.iter().any(|entry: &ManifestEntry| entry.url == url) {
                return Err(NetError::Manifest(format!(
                    "строка {}: адрес повторяется",
                    index + 1
                )));
            }
            entries.push(ManifestEntry {
                url: url.to_owned(),
                sha256,
            });
        }
        Ok(Self { entries })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_comments_and_hashes() {
        let text = "\
# вступление
https://cdn.example.com/a.bin e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855

https://cdn.example.com/b.bin
";
        let manifest = Manifest::parse(text).unwrap();
        assert_eq!(manifest.entries.len(), 2);
        assert!(manifest.entries[0].sha256.is_some());
        assert!(manifest.entries[1].sha256.is_none());
        assert!(Manifest::parse("assets/local.bin").is_err());
        assert!(Manifest::parse("https://cdn.example.com/a.bin abc").is_err());
    }
}
