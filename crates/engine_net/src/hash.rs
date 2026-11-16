//! SHA-256 для проверки, что скачанное совпало с ожидаемым.
//!
//! Алгоритм фиксированный: манифест пишет 64 шестнадцатеричных символа, и тот же
//! хэш считается на всех платформах.

use sha2::{Digest, Sha256};

use crate::NetError;

pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

pub(crate) fn check(bytes: &[u8], expected: Option<[u8; 32]>) -> Result<(), NetError> {
    if let Some(expected) = expected
        && sha256(bytes) != expected
    {
        return Err(NetError::HashMismatch);
    }
    Ok(())
}

/// 64 шестнадцатеричных символа, регистр не важен.
pub fn parse_sha256(text: &str) -> Option<[u8; 32]> {
    if text.len() != 64 || !text.is_ascii() {
        return None;
    }
    let mut out = [0u8; 32];
    for (index, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&text[index * 2..index * 2 + 2], 16).ok()?;
    }
    Some(out)
}

pub(crate) fn to_hex(hash: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for byte in hash {
        out.push(char::from(HEX[usize::from(byte >> 4)]));
        out.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_sha256_and_hex_roundtrip() {
        // Известное значение SHA-256 от пустой строки
        let hash = sha256(b"");
        let hex = to_hex(&hash);
        assert_eq!(
            hex,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(parse_sha256(&hex.to_ascii_uppercase()), Some(hash));
        assert!(parse_sha256("abc").is_none());
        assert!(check(b"", Some(hash)).is_ok());
        assert_eq!(check(b"x", Some(hash)), Err(NetError::HashMismatch));
    }
}
