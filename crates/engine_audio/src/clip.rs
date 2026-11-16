//! Звуковой клип как ресурс: `server.load::<AudioClip>("audio/bounce.wav")`.

use engine_assets::{Asset, AssetError, AssetLoader, Bytes, LoadContext};
use engine_core::tasks::BoxedFuture;
use engine_ecs::prelude::{TypePath, bevy_reflect};

/// Звуковой файл в исходном (сжатом) виде. Декодирует бэкенд: короткие звуки — целиком
/// при первом проигрывании, музыку — потоком по ходу воспроизведения.
#[derive(Debug, Clone, TypePath)]
pub struct AudioClip {
    pub data: Bytes,
}

impl Asset for AudioClip {}

#[derive(Debug)]
pub struct AudioClipLoader;

/// Формат по сигнатуре в начале файла.
fn known_format(bytes: &[u8]) -> bool {
    let wav = bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WAVE".as_slice());
    let ogg = bytes.starts_with(b"OggS");
    // MP3: тег ID3 или заголовок кадра (11 единичных бит синхронизации)
    let mp3 =
        bytes.starts_with(b"ID3") || matches!(bytes, [0xFF, second, ..] if second & 0xE0 == 0xE0);
    wav || ogg || mp3
}

impl AssetLoader for AudioClipLoader {
    type Asset = AudioClip;

    fn extensions(&self) -> &[&'static str] {
        &["wav", "ogg", "mp3"]
    }

    fn load<'a>(
        &'a self,
        bytes: Bytes,
        ctx: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<AudioClip, AssetError>> {
        Box::pin(async move {
            if !known_format(&bytes) {
                return Err(AssetError::format(ctx.path(), "не WAV, OGG или MP3"));
            }
            Ok(AudioClip { data: bytes })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_formats_by_signature() {
        assert!(known_format(b"RIFF\0\0\0\0WAVEfmt "));
        assert!(known_format(b"OggS\0\x02"));
        assert!(known_format(b"ID3\x04"));
        assert!(known_format(&[0xFF, 0xFB, 0x90]));
        assert!(!known_format(b"RIFF\0\0\0\0AVI "));
        assert!(!known_format(b"PK\x03\x04"));
    }
}
