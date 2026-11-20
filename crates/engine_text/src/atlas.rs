//! Атлас глифов: одна текстура R8, в которую по мере надобности добавляются SDF глифов.
//!
//! Атлас живёт в главном потоке (там же вёрстка); рендеру уходят только изменения —
//! прямоугольники новых глифов ([`AtlasUpdate`]). Когда место кончается, атлас удваивается
//! (до [`MAX_SIZE`]) и заполняется заново: глифы, которые всё ещё нужны, растеризуются
//! снова при следующей вёрстке.

use std::collections::HashMap;

use cosmic_text::{CacheKey, FontSystem, SwashCache, SwashContent};
use engine_ecs::prelude::*;

/// Размер, в котором растеризуется глиф для SDF, пикселей на em. Текст любого размера
/// рисуется из этой растеризации.
pub const SDF_PX: f32 = 48.0;
/// Поле SDF вокруг глифа, пикселей. Определяет максимальную толщину обводки.
pub const SPREAD: usize = 6;
const START_SIZE: u32 = 1024;
const MAX_SIZE: u32 = 4096;

/// Место глифа в атласе.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphSlot {
    /// Текстурные координаты (0..1): левый верхний и правый нижний углы.
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    /// Прямоугольник глифа относительно точки на базовой линии, в пикселях растеризации
    /// (`SDF_PX`), Y вниз: `left`, `top` (отрицательный — выше базовой линии), ширина, высота.
    /// Уже с полями SDF.
    pub left: f32,
    pub top: f32,
    pub width: f32,
    pub height: f32,
}

/// Изменение текстуры атласа для рендера.
#[derive(Debug, Clone)]
pub struct AtlasUpdate {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    /// По байту на пиксель, строки подряд.
    pub data: Vec<u8>,
}

/// Атлас глифов (ресурс ECS). Рендер забирает изменения через [`GlyphAtlas::take_changes`].
#[derive(Resource, Debug)]
pub struct GlyphAtlas {
    size: u32,
    /// Растёт при каждой пересборке атласа: вёрстка, сделанная раньше, устарела.
    generation: u64,
    packer: ShelfPacker,
    /// `None` — у глифа нет изображения (пробел) или он цветной (эмодзи — пока не рисуем).
    slots: HashMap<CacheKey, Option<GlyphSlot>>,
    /// Атлас пересоздан — рендеру надо создать текстуру заново (нового размера).
    reset: bool,
    updates: Vec<AtlasUpdate>,
}

impl Default for GlyphAtlas {
    fn default() -> Self {
        Self {
            size: START_SIZE,
            generation: 0,
            packer: ShelfPacker::new(START_SIZE),
            slots: HashMap::new(),
            reset: true,
            updates: Vec::new(),
        }
    }
}

/// Изменения атласа за кадр.
#[derive(Debug, Default)]
pub struct AtlasChanges {
    /// `Some(размер)` — создать текстуру заново (содержимое пустое).
    pub reset: Option<u32>,
    pub updates: Vec<AtlasUpdate>,
}

impl GlyphAtlas {
    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Изменения с прошлого вызова — для отправки рендеру.
    pub fn take_changes(&mut self) -> AtlasChanges {
        AtlasChanges {
            reset: std::mem::take(&mut self.reset).then_some(self.size),
            updates: std::mem::take(&mut self.updates),
        }
    }

    /// Место глифа; растеризует и добавляет, если его ещё нет.
    /// `Err(())` — атлас переполнен и пересоздан: вёрстку нужно повторить.
    pub(crate) fn glyph(
        &mut self,
        key: CacheKey,
        fonts: &mut FontSystem,
        swash: &mut SwashCache,
    ) -> Result<Option<GlyphSlot>, ()> {
        if let Some(slot) = self.slots.get(&key) {
            return Ok(*slot);
        }
        let slot = self.rasterize(key, fonts, swash)?;
        self.slots.insert(key, slot);
        Ok(slot)
    }

    fn rasterize(
        &mut self,
        key: CacheKey,
        fonts: &mut FontSystem,
        swash: &mut SwashCache,
    ) -> Result<Option<GlyphSlot>, ()> {
        let Some(image) = swash.get_image_uncached(fonts, key) else {
            return Ok(None);
        };
        if image.content != SwashContent::Mask || image.placement.width == 0 {
            return Ok(None);
        }
        let (w, h) = (
            image.placement.width as usize,
            image.placement.height as usize,
        );
        let sdf = crate::sdf::build(&image.data, w, h, SPREAD);
        // Размер с полями SDF
        let (padded_w, padded_h) = (w + 2 * SPREAD, h + 2 * SPREAD);
        let (slot_w, slot_h) = (
            u32::try_from(padded_w).unwrap_or(0),
            u32::try_from(padded_h).unwrap_or(0),
        );
        let Some((x, y)) = self.packer.allocate(slot_w, slot_h) else {
            self.grow();
            return Err(());
        };
        self.updates.push(AtlasUpdate {
            x,
            y,
            width: slot_w,
            height: slot_h,
            data: sdf,
        });
        #[allow(clippy::cast_precision_loss)]
        let (size, spread) = (self.size as f32, SPREAD as f32);
        #[allow(clippy::cast_precision_loss)]
        Ok(Some(GlyphSlot {
            uv_min: [x as f32 / size, y as f32 / size],
            uv_max: [(x + slot_w) as f32 / size, (y + slot_h) as f32 / size],
            left: image.placement.left as f32 - spread,
            top: -(image.placement.top as f32) - spread,
            width: padded_w as f32,
            height: padded_h as f32,
        }))
    }

    /// Места нет: удваиваем атлас (или очищаем на максимальном размере).
    fn grow(&mut self) {
        let size = (self.size * 2).min(MAX_SIZE);
        if size == self.size {
            tracing::warn!("Атлас глифов {size}² заполнен — очищаем");
        } else {
            tracing::info!("Атлас глифов: {} → {size}²", self.size);
        }
        self.size = size;
        self.invalidate();
    }

    /// Забыть все глифы и пересоздать текстуру того же размера. Вёрстка устаревает и
    /// повторяется. Нужен, когда текстура на GPU пропала (потеря устройства).
    pub fn invalidate(&mut self) {
        self.packer = ShelfPacker::new(self.size);
        self.slots.clear();
        self.updates.clear();
        self.reset = true;
        self.generation += 1;
    }
}

/// Упаковка прямоугольников «полками»: строки фиксированной высоты, заполняемые слева
/// направо. Глифы одного размера шрифта близки по высоте — потери небольшие.
#[derive(Debug)]
struct ShelfPacker {
    size: u32,
    /// (y, высота, занято по x)
    shelves: Vec<(u32, u32, u32)>,
    next_y: u32,
}

impl ShelfPacker {
    fn new(size: u32) -> Self {
        Self {
            size,
            shelves: Vec::new(),
            next_y: 0,
        }
    }

    fn allocate(&mut self, w: u32, h: u32) -> Option<(u32, u32)> {
        const GAP: u32 = 1; // зазор: билинейная выборка не захватит соседа
        let (w, h) = (w + GAP, h + GAP);
        if w > self.size {
            return None;
        }
        // Подходящая полка: не ниже глифа и не слишком высокая (иначе пустое место)
        for shelf in &mut self.shelves {
            if h <= shelf.1 && h * 3 >= shelf.1 * 2 && shelf.2 + w <= self.size {
                let x = shelf.2;
                shelf.2 += w;
                return Some((x, shelf.0));
            }
        }
        if self.next_y + h > self.size {
            return None;
        }
        let y = self.next_y;
        self.next_y += h;
        self.shelves.push((y, h, w));
        Some((0, y))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packer_fills_shelves_and_reports_full() {
        let mut packer = ShelfPacker::new(64);
        let a = packer.allocate(20, 10).unwrap();
        let b = packer.allocate(20, 10).unwrap();
        assert_eq!(a, (0, 0));
        assert_eq!(b, (21, 0), "та же полка, правее");
        let c = packer.allocate(20, 30).unwrap();
        assert_eq!(c.1, 11, "высокий — на новую полку");
        // Заполняем до отказа
        let mut count = 0;
        while packer.allocate(20, 10).is_some() {
            count += 1;
            assert!(count < 100);
        }
        assert!(packer.allocate(70, 5).is_none(), "шире атласа");
    }
}
