//! KTX2: несжатый RGBA8 или Basis Universal.
//!
//! Basis не разжимается здесь. Байты файла остаются в [`Image::basis`], а формат
//! блока (BC7, ASTC, ETC2 или RGBA) выбирает рендер по возможностям GPU.
//! Так один файл годится и для Vulkan на Linux, и для Metal, и для GLES3.

use super::image::{ColorSpace, Image};

/// Сигнатура KTX 2.0.
pub(crate) const MAGIC: &[u8] = b"\xABKTX 20\xBB\r\n\x1A\n";

const HEADER: usize = 80;
const LEVEL_INDEX: usize = 24;
const RGBA8_UNORM: u32 = 37;
const RGBA8_SRGB: u32 = 43;

pub(crate) fn is_ktx2(bytes: &[u8]) -> bool {
    bytes.starts_with(MAGIC)
}

impl Image {
    /// `color_space` — как текстура используется (glTF). `None` берёт признак из файла.
    pub fn decode_ktx2(bytes: &[u8], color_space: Option<ColorSpace>) -> Result<Self, String> {
        if !is_ktx2(bytes) {
            return Err("это не KTX2".to_owned());
        }
        let format = read_u32(bytes, 12).unwrap_or(0);
        if format == RGBA8_UNORM || format == RGBA8_SRGB {
            return uncompressed_rgba(bytes, color_space);
        }
        if let Ok(texture) = basisu::Transcoder::new(bytes) {
            return basis_image(bytes, &texture, color_space);
        }
        Err("KTX2 без Basis: нужен R8G8B8A8. Сжатый файл кодируйте Basis Universal".to_owned())
    }
}

fn basis_image(
    bytes: &[u8],
    texture: &basisu::Transcoder<'_>,
    color_space: Option<ColorSpace>,
) -> Result<Image, String> {
    if texture.face_count() != 1 || texture.layer_count() > 1 || texture.is_video() {
        return Err(
            "KTX2: нужен обычный 2D-файл, без кубической карты, массива и видео".to_owned(),
        );
    }
    let (width, height) = texture.base_dimensions();
    if width == 0 || height == 0 {
        return Err("KTX2: нулевой размер".to_owned());
    }
    Ok(Image {
        width,
        height,
        color_space: color_space.unwrap_or_else(|| transfer_space(bytes)),
        data: Vec::new(),
        mips: Vec::new(),
        basis: bytes.to_vec(),
    })
}

fn uncompressed_rgba(bytes: &[u8], color_space: Option<ColorSpace>) -> Result<Image, String> {
    let format = read_u32(bytes, 12)?;
    let width = read_u32(bytes, 20)?;
    let height = read_u32(bytes, 24)?;
    let depth = read_u32(bytes, 28)?;
    let layers = read_u32(bytes, 32)?;
    let faces = read_u32(bytes, 36)?;
    let levels = read_u32(bytes, 40)?;
    let scheme = read_u32(bytes, 44)?;
    if depth > 1 || layers > 1 || faces > 1 {
        return Err("KTX2: нужен обычный 2D-файл".to_owned());
    }
    if width == 0 || height == 0 || levels == 0 {
        return Err("KTX2: пустой файл".to_owned());
    }
    let from_format = if format == RGBA8_SRGB {
        ColorSpace::Srgb
    } else {
        ColorSpace::Linear
    };
    if scheme != 0 {
        return Err("KTX2: это сжатие не Basis Universal и не сырой RGBA8".to_owned());
    }
    let mut image = Image {
        width,
        height,
        color_space: color_space.unwrap_or(from_format),
        data: Vec::new(),
        mips: Vec::new(),
        basis: Vec::new(),
    };
    for level in 0..levels {
        let entry = HEADER + usize::try_from(level).unwrap_or(0) * LEVEL_INDEX;
        let offset = usize::try_from(read_u64(bytes, entry)?).unwrap_or(usize::MAX);
        let length = usize::try_from(read_u64(bytes, entry + 8)?).unwrap_or(0);
        let mip_w = (width >> level).max(1);
        let mip_h = (height >> level).max(1);
        let expect = mip_w
            .checked_mul(mip_h)
            .and_then(|n| n.checked_mul(4))
            .ok_or("KTX2: слишком большой уровень")?;
        let expect = usize::try_from(expect).unwrap_or(usize::MAX);
        if length != expect {
            return Err(format!(
                "KTX2: уровень {level} — {length} байт, ожидалось {expect}"
            ));
        }
        let end = offset
            .checked_add(length)
            .ok_or("KTX2: уровень за концом файла")?;
        let pixels = bytes
            .get(offset..end)
            .ok_or("KTX2: уровень за концом файла")?
            .to_vec();
        if level == 0 {
            image.data = pixels;
        } else {
            image.mips.push(pixels);
        }
    }
    if levels == 1 {
        image.generate_mips();
    }
    Ok(image)
}

/// Байт transfer function в DFD: 1 — линейный, 2 — sRGB. Нет блока — считаем цветом.
fn transfer_space(bytes: &[u8]) -> ColorSpace {
    let Ok(offset) = read_u32(bytes, 48) else {
        return ColorSpace::Srgb;
    };
    let Ok(length) = read_u32(bytes, 52) else {
        return ColorSpace::Srgb;
    };
    let Ok(offset) = usize::try_from(offset) else {
        return ColorSpace::Srgb;
    };
    if length < 17 || bytes.len() < offset.saturating_add(17) {
        return ColorSpace::Srgb;
    }
    if bytes[offset + 16] == 1 {
        ColorSpace::Linear
    } else {
        ColorSpace::Srgb
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let chunk = bytes
        .get(offset..offset + 4)
        .ok_or("короткий заголовок KTX2")?;
    let mut raw = [0u8; 4];
    raw.copy_from_slice(chunk);
    Ok(u32::from_le_bytes(raw))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, String> {
    let chunk = bytes
        .get(offset..offset + 8)
        .ok_or("короткий заголовок KTX2")?;
    let mut raw = [0u8; 8];
    raw.copy_from_slice(chunk);
    Ok(u64::from_le_bytes(raw))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba_file(format: u32, width: u32, height: u32, pixels: &[u8]) -> Vec<u8> {
        let mut file = Vec::new();
        file.extend_from_slice(MAGIC);
        for value in [format, 1, width, height, 0, 0, 1, 1, 0, 0, 0, 0, 0] {
            file.extend_from_slice(&value.to_le_bytes());
        }
        // sgdByteOffset и sgdByteLength — u64, до этого в списке выше не вошли
        file.extend_from_slice(&0u64.to_le_bytes());
        file.extend_from_slice(&0u64.to_le_bytes());
        let offset = (HEADER + LEVEL_INDEX) as u64;
        file.extend_from_slice(&offset.to_le_bytes());
        let length = pixels.len() as u64;
        file.extend_from_slice(&length.to_le_bytes());
        file.extend_from_slice(&length.to_le_bytes());
        file.extend_from_slice(pixels);
        file
    }

    #[test]
    fn uncompressed_srgb_keeps_pixels_and_builds_a_mip() {
        let pixels = [
            10, 20, 30, 255, 40, 50, 60, 255, 70, 80, 90, 255, 100, 110, 120, 255,
        ];
        let image = Image::decode_ktx2(&rgba_file(RGBA8_SRGB, 2, 2, &pixels), None).unwrap();
        assert_eq!(image.basis.as_slice(), b"");
        assert_eq!(image.color_space, ColorSpace::Srgb);
        assert_eq!(image.data, pixels);
        assert_eq!(image.mip_level_count(), 2);
    }

    #[test]
    fn caller_color_space_overrides_the_file() {
        let pixels = [0, 0, 0, 255];
        let image = Image::decode_ktx2(
            &rgba_file(RGBA8_SRGB, 1, 1, &pixels),
            Some(ColorSpace::Linear),
        )
        .unwrap();
        assert_eq!(image.color_space, ColorSpace::Linear);
    }

    #[test]
    fn raw_bc7_is_rejected() {
        // 0 — vkFormat «формат только в DFD», без Basis это не наш файл
        let error = Image::decode_ktx2(&rgba_file(0, 4, 4, &[0; 16]), None).unwrap_err();
        assert!(error.contains("Basis"));
    }
}
