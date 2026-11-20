//! Заливка KTX2 с Basis на GPU.
//!
//! Формат выбирается по устройству: BC7 (Linux, Vulkan, Metal), иначе ASTC 4×4,
//! иначе ETC2 (GLES3), иначе сырой RGBA. sRGB — это вид формата, не другие байты.

use basisu::{DecodeFlags, TargetFormat, Transcoder};
use engine_assets::Asset;
use engine_assets::formats::{ColorSpace, Image};

/// Куда перекодировать Basis. Порядок — от десктопа к GLES3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuBlock {
    Bc7,
    Astc,
    Etc2,
    Rgba,
}

pub(crate) fn gpu_block(features: wgpu::Features) -> GpuBlock {
    if features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
        GpuBlock::Bc7
    } else if features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC) {
        GpuBlock::Astc
    } else if features.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2) {
        GpuBlock::Etc2
    } else {
        GpuBlock::Rgba
    }
}

pub(crate) fn upload_basis(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &Image,
) -> crate::scene::GpuTexture {
    match prepare(device.features(), image) {
        Ok(ready) => upload_levels(device, image, &ready, |texture, mip, level| {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: mip,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &level.bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(level.row_bytes),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: level.width,
                    height: level.height,
                    depth_or_array_layers: 1,
                },
            );
        }),
        Err(error) => {
            tracing::error!("KTX2 не залился, вместо него шахматка: {error}");
            crate::scene::upload_rgba(device, queue, &error_image())
        }
    }
}

pub(crate) fn upload_basis_staged(
    device: &wgpu::Device,
    encoder: &mut wgpu::CommandEncoder,
    staging: &mut crate::upload::Staging,
    image: &Image,
) -> crate::scene::GpuTexture {
    match prepare(device.features(), image) {
        Ok(ready) => upload_levels(device, image, &ready, |texture, mip, level| {
            staging.copy_texture(
                encoder,
                texture,
                mip,
                &level.bytes,
                level.row_bytes,
                wgpu::Extent3d {
                    width: level.width,
                    height: level.height,
                    depth_or_array_layers: 1,
                },
            );
        }),
        Err(error) => {
            tracing::error!("KTX2 не залился, вместо него шахматка: {error}");
            crate::scene::upload_rgba_staged(device, encoder, staging, &error_image())
        }
    }
}

struct Level {
    width: u32,
    height: u32,
    bytes: Vec<u8>,
    row_bytes: u32,
}

struct Ready {
    format: wgpu::TextureFormat,
    levels: Vec<Level>,
}

fn prepare(features: wgpu::Features, image: &Image) -> Result<Ready, String> {
    let texture = Transcoder::new(&image.basis).map_err(|error| format!("{error:?}"))?;
    let mut block = gpu_block(features);
    let mut target = target_format(block);
    if !texture.supports(target) {
        block = GpuBlock::Rgba;
        target = TargetFormat::Rgba32;
    }
    let count = texture.level_count().max(1);
    let mut levels = Vec::with_capacity(usize::try_from(count).unwrap_or(1));
    for level in 0..count {
        let info = texture
            .image_level_info(level)
            .map_err(|error| format!("уровень {level}: {error:?}"))?;
        let bytes = texture
            .transcode(level, target, DecodeFlags::NONE)
            .map_err(|error| format!("уровень {level}: {error:?}"))?;
        let row_bytes = row_bytes(target, info.width);
        levels.push(Level {
            width: info.width,
            height: info.height,
            bytes,
            row_bytes,
        });
    }
    tracing::debug!(
        "KTX2: {block:?}, {}×{}, уровней {}",
        image.width,
        image.height,
        levels.len()
    );
    Ok(Ready {
        format: wgpu_format(block, image.color_space),
        levels,
    })
}

fn target_format(block: GpuBlock) -> TargetFormat {
    match block {
        GpuBlock::Bc7 => TargetFormat::Bc7Rgba,
        GpuBlock::Astc => TargetFormat::Astc4x4Rgba,
        GpuBlock::Etc2 => TargetFormat::Etc2Rgba,
        GpuBlock::Rgba => TargetFormat::Rgba32,
    }
}

fn wgpu_format(block: GpuBlock, space: ColorSpace) -> wgpu::TextureFormat {
    let srgb = space == ColorSpace::Srgb;
    match block {
        GpuBlock::Bc7 => {
            if srgb {
                wgpu::TextureFormat::Bc7RgbaUnormSrgb
            } else {
                wgpu::TextureFormat::Bc7RgbaUnorm
            }
        }
        GpuBlock::Astc => wgpu::TextureFormat::Astc {
            block: wgpu::AstcBlock::B4x4,
            channel: if srgb {
                wgpu::AstcChannel::UnormSrgb
            } else {
                wgpu::AstcChannel::Unorm
            },
        },
        GpuBlock::Etc2 => {
            if srgb {
                wgpu::TextureFormat::Etc2Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Etc2Rgba8Unorm
            }
        }
        GpuBlock::Rgba => {
            if srgb {
                wgpu::TextureFormat::Rgba8UnormSrgb
            } else {
                wgpu::TextureFormat::Rgba8Unorm
            }
        }
    }
}

fn row_bytes(target: TargetFormat, width: u32) -> u32 {
    let (block_w, _) = if target.is_block_based() {
        target.block_dims()
    } else {
        (1, 1)
    };
    let bytes = u32::try_from(target.bytes_per_block_or_pixel()).unwrap_or(1);
    width.div_ceil(block_w.max(1)) * bytes
}

fn upload_levels(
    device: &wgpu::Device,
    image: &Image,
    ready: &Ready,
    mut write: impl FnMut(&wgpu::Texture, u32, &Level),
) -> crate::scene::GpuTexture {
    let base = ready.levels.first();
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ktx2"),
        size: wgpu::Extent3d {
            width: base.map_or(image.width, |level| level.width),
            height: base.map_or(image.height, |level| level.height),
            depth_or_array_layers: 1,
        },
        mip_level_count: u32::try_from(ready.levels.len()).unwrap_or(1),
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: ready.format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (index, level) in ready.levels.iter().enumerate() {
        write(&texture, u32::try_from(index).unwrap_or(0), level);
    }
    crate::scene::GpuTexture {
        view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        bytes: image.memory_size(),
    }
}

fn error_image() -> Image {
    Image {
        width: 8,
        height: 8,
        color_space: ColorSpace::Srgb,
        data: image_checker(),
        mips: Vec::new(),
        basis: Vec::new(),
    }
}

fn image_checker() -> Vec<u8> {
    let mut data = Vec::with_capacity(8 * 8 * 4);
    for y in 0..8 {
        for x in 0..8 {
            let pink = (x + y) % 2 == 0;
            data.extend_from_slice(if pink {
                &[255, 0, 255, 255]
            } else {
                &[0, 0, 0, 255]
            });
        }
    }
    data
}

#[cfg(test)]
mod tests {
    use super::{GpuBlock, gpu_block};

    #[test]
    fn desktop_block_wins_over_mobile() {
        let both =
            wgpu::Features::TEXTURE_COMPRESSION_BC | wgpu::Features::TEXTURE_COMPRESSION_ASTC;
        assert_eq!(gpu_block(both), GpuBlock::Bc7);
        assert_eq!(
            gpu_block(wgpu::Features::TEXTURE_COMPRESSION_ASTC),
            GpuBlock::Astc
        );
        assert_eq!(
            gpu_block(wgpu::Features::TEXTURE_COMPRESSION_ETC2),
            GpuBlock::Etc2
        );
        assert_eq!(gpu_block(wgpu::Features::empty()), GpuBlock::Rgba);
    }
}
