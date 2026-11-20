//! Общий staging-буфер для мешей и текстур и бюджет заливки на кадр.
//!
//! Постоянные буферы создаются сразу, байты пишутся в переиспользуемые куски
//! и копируются командой GPU. За кадр копируется не больше [`RENDER_UPLOAD_BUDGET_MB`].
//! Один кусок больше бюджета всё равно уходит в том кадре, где до него дошла очередь:
//! иначе он стоял бы вечно.

use std::num::NonZeroU64;

use engine_core::CVar;

/// Сколько мешей и текстур копировать на GPU за один кадр.
pub static RENDER_UPLOAD_BUDGET_MB: CVar<u32> = CVar::new(
    "render.upload_budget_mb",
    8,
    "Мегабайт мешей и текстур на GPU за кадр. 0 — без ограничения",
);

/// Кусок пояса. Бюджет по умолчанию — два таких.
const CHUNK: u64 = 4 * 1024 * 1024;

pub(crate) fn budget_bytes() -> u64 {
    match RENDER_UPLOAD_BUDGET_MB.get() {
        0 => u64::MAX,
        megabytes => u64::from(megabytes).saturating_mul(1024 * 1024),
    }
}

/// Можно ли взять ещё `cost` байт в этом кадре.
/// Первый кусок берётся даже если он больше бюджета.
pub(crate) fn fits(spent: u64, cost: u64, budget: u64) -> bool {
    spent == 0 || spent.saturating_add(cost) <= budget
}

/// Переиспользуемые временные буферы `MAP_WRITE | COPY_SRC`.
pub(crate) struct Staging {
    belt: wgpu::util::StagingBelt,
}

impl Staging {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        Self {
            belt: wgpu::util::StagingBelt::new(device.clone(), CHUNK),
        }
    }

    /// Разметить пояс и вернуть куски после отправки этого энкодера.
    pub(crate) fn finish_on_submit(&mut self, encoder: &wgpu::CommandEncoder) {
        self.belt.finish_and_recall_on_submit(encoder);
    }

    /// Скопировать байты в постоянный буфер. Размер буфера — [`aligned_size`] от `bytes`.
    pub(crate) fn copy_buffer(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Buffer,
        bytes: &[u8],
    ) {
        let size = aligned_size(bytes.len());
        let mut view = self.belt.write_buffer(encoder, target, 0, size);
        if bytes.len() == view.len() {
            view.copy_from_slice(bytes);
        } else {
            let mut padded = vec![0u8; view.len()];
            let count = bytes.len().min(padded.len());
            padded[..count].copy_from_slice(&bytes[..count]);
            view.copy_from_slice(&padded);
        }
    }

    /// Скопировать уровень текстуры. Строка в поясе выровнена на 256 байт.
    pub(crate) fn copy_texture(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        mip: u32,
        data: &[u8],
        row_bytes: u32,
        size: wgpu::Extent3d,
    ) {
        if data.is_empty() || row_bytes == 0 || size.width == 0 || size.height == 0 {
            return;
        }
        let row = usize::try_from(row_bytes).unwrap_or(usize::MAX);
        let rows = data.len() / row;
        if rows == 0 {
            return;
        }
        let padded = align_u32(row_bytes, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let padded_row = usize::try_from(padded).unwrap_or(0);
        let bytes = u64::from(padded).saturating_mul(u64::try_from(rows).unwrap_or(0));
        let Some(alloc) = NonZeroU64::new(align_u64(bytes, wgpu::COPY_BUFFER_ALIGNMENT)) else {
            return;
        };
        let align = NonZeroU64::new(u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT))
            .unwrap_or(NonZeroU64::MIN);
        let slice = self.belt.allocate(alloc, align);
        {
            let mut view = slice
                .get_mapped_range_mut()
                .expect("staging-буфер не отображён");
            let mut row_buf = vec![0u8; padded_row];
            for row_index in 0..rows {
                let src = row_index * row;
                let dst = row_index * padded_row;
                let width = row.min(padded_row);
                row_buf[..width].copy_from_slice(&data[src..src + width]);
                row_buf[width..].fill(0);
                view.slice(dst..dst + padded_row).copy_from_slice(&row_buf);
            }
        }
        encoder.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: slice.buffer(),
                layout: wgpu::TexelCopyBufferLayout {
                    offset: slice.offset(),
                    bytes_per_row: Some(padded),
                    rows_per_image: None,
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: mip,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            size,
        );
    }
}

/// Размер буфера и копии: кратно 4 и не ноль.
pub(crate) fn aligned_size(len: usize) -> NonZeroU64 {
    let len = u64::try_from(len).unwrap_or(u64::MAX);
    let aligned = align_u64(len, wgpu::COPY_BUFFER_ALIGNMENT).max(wgpu::COPY_BUFFER_ALIGNMENT);
    NonZeroU64::new(aligned).unwrap_or(NonZeroU64::MIN)
}

fn align_u64(value: u64, align: u64) -> u64 {
    value.div_ceil(align.max(1)) * align.max(1)
}

fn align_u32(value: u32, align: u32) -> u32 {
    value.div_ceil(align.max(1)) * align.max(1)
}

#[cfg(test)]
mod tests {
    use super::fits;

    #[test]
    fn first_chunk_is_taken_even_over_budget() {
        assert!(fits(0, 20, 8));
    }

    #[test]
    fn later_chunk_waits_when_it_does_not_fit() {
        assert!(!fits(8, 4, 10));
        assert!(fits(8, 2, 10));
    }

    #[test]
    fn unlimited_budget_takes_everything() {
        assert!(fits(u64::MAX - 1, 5, u64::MAX));
    }
}
