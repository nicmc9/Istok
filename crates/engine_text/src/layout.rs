//! Вёрстка: строка → глифы (шейпинг, переносы, выравнивание) → прямоугольники и места в атласе.
//!
//! Шейпинг делает `cosmic-text`: подбирает глифы с учётом лигатур и кернинга, письма справа
//! налево, подменяет шрифт, если в основном нет символа. Вёрстка идёт в «опорном» размере
//! [`REF_PX`] и масштабируется до `Text::size`: результат не зависит от размера (без хинтинга),
//! а `cosmic-text` не работает с дробными размерами вроде 0.3 м.

use std::collections::HashMap;

use cosmic_text::{
    Align, Attrs, Buffer, CacheKey, CacheKeyFlags, Family, FontSystem, Metrics, Shaping,
    SwashCache, fontdb,
};
use engine_assets::{AssetEvent, AssetId, Assets};
use engine_ecs::prelude::*;
use engine_math::Vec2;

use crate::atlas::{GlyphAtlas, SDF_PX};
use crate::{Font, PlacedGlyph, Text, TextAlign, TextLayout};

/// Опорный размер вёрстки, пикселей на em.
const REF_PX: f32 = 64.0;

/// Состояние вёрстки: база шрифтов, кеш растеризатора, зарегистрированные шрифты.
#[derive(Resource)]
pub struct TextSystem {
    fonts: FontSystem,
    swash: SwashCache,
    default_family: String,
    /// Шрифт-ресурс → имя семейства в базе шрифтов.
    families: HashMap<AssetId<Font>, String>,
    /// Добавились шрифты — пересверстать всё (могли появиться недостающие символы).
    fonts_changed: bool,
}

impl std::fmt::Debug for TextSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextSystem")
            .field("default_family", &self.default_family)
            .field("families", &self.families.len())
            .finish_non_exhaustive()
    }
}

impl Default for TextSystem {
    fn default() -> Self {
        // Пустая база: системные шрифты не ищем — на Web и мобильных их не прочитать,
        // а результат должен быть одинаковым на всех платформах
        let mut db = fontdb::Database::new();
        let default_family = register(&mut db, epaint_default_fonts::UBUNTU_LIGHT.to_vec())
            .unwrap_or_else(|| "Ubuntu".to_owned());
        Self {
            fonts: FontSystem::new_with_locale_and_db("en-US".to_owned(), db),
            swash: SwashCache::new(),
            default_family,
            families: HashMap::new(),
            fonts_changed: false,
        }
    }
}

/// Добавляет шрифт в базу, возвращает имя семейства.
fn register(db: &mut fontdb::Database, data: Vec<u8>) -> Option<String> {
    // [Rust] `Arc<dyn Trait>` — общий указатель на объект «какого-то типа с этим интерфейсом»
    let ids = db.load_font_source(fontdb::Source::Binary(std::sync::Arc::new(data)));
    let face = db.face(*ids.first()?)?;
    face.families.first().map(|(name, _)| name.clone())
}

/// Регистрирует загруженные шрифты-ресурсы. Стадия `PostUpdate`, до вёрстки.
pub(crate) fn register_fonts(
    mut events: MessageReader<AssetEvent<Font>>,
    fonts: Res<Assets<Font>>,
    mut system: ResMut<TextSystem>,
) {
    for event in events.read() {
        let AssetEvent::Loaded(id) = event else {
            continue;
        };
        let Some(font) = fonts
            .get_by_id(*id)
            .filter(engine_assets::Resolved::is_ready)
        else {
            continue;
        };
        let font = font.asset;
        let system = &mut *system;
        if let Some(family) = register(system.fonts.db_mut(), font.data.to_vec()) {
            tracing::info!("Шрифт зарегистрирован: {family}");
            system.families.insert(*id, family);
            system.fonts_changed = true;
        } else {
            tracing::error!("Шрифт не разобран");
        }
    }
}

/// Пересчитывает вёрстку изменившихся текстов. Стадия `PostUpdate`.
pub(crate) fn layout_texts(
    mut commands: Commands,
    mut texts: Query<(Entity, Ref<Text>, Option<&mut TextLayout>)>,
    mut system: ResMut<TextSystem>,
    mut atlas: ResMut<GlyphAtlas>,
) {
    let fonts_changed = std::mem::take(&mut system.fonts_changed);
    for (entity, text, layout) in &mut texts {
        let stale = layout
            .as_ref()
            .is_none_or(|l| l.atlas_generation != atlas.generation());
        if !(stale || fonts_changed || text.is_changed()) {
            continue;
        }
        let _zone = tracing::trace_span!("layout_text").entered();
        // Переполнение атласа → он пересоздан → повторяем (второй раз места хватит)
        let mut result = layout_one(&text, &mut system, &mut atlas);
        if result.is_err() {
            result = layout_one(&text, &mut system, &mut atlas);
        }
        let Ok(new_layout) = result else {
            tracing::error!("Текст не поместился в атлас глифов: {:?}", text.content);
            continue;
        };
        match layout {
            Some(mut layout) => *layout = new_layout,
            None => {
                commands.entity(entity).insert(new_layout);
            }
        }
    }
}

fn layout_one(
    text: &Text,
    system: &mut TextSystem,
    atlas: &mut GlyphAtlas,
) -> Result<TextLayout, ()> {
    let family = text
        .font
        .as_ref()
        .and_then(|font| system.families.get(&font.id()))
        .unwrap_or(&system.default_family)
        .clone();
    let size = text.size.max(1e-4);
    let scale = size / REF_PX;
    let mut buffer = Buffer::new_empty(Metrics::new(REF_PX, REF_PX * text.line_height.max(0.1)));
    let attrs = Attrs::new().family(Family::Name(&family));
    let align = match text.align {
        TextAlign::Left => Align::Left,
        TextAlign::Center => Align::Center,
        TextAlign::Right => Align::Right,
    };
    buffer.set_size(text.max_width.map(|w| w / scale), None);
    buffer.set_text(&text.content, &attrs, Shaping::Advanced, Some(align));
    buffer.shape_until_scroll(&mut system.fonts, false);
    // Без ширины блока выравнивать не по чему: берём ширину самой длинной строки и
    // верстаем ещё раз
    if text.max_width.is_none() && text.align != TextAlign::Left {
        let widest = buffer
            .layout_runs()
            .map(|run| run.line_w)
            .fold(0.0, f32::max);
        buffer.set_size(Some(widest.ceil()), None);
        buffer.shape_until_scroll(&mut system.fonts, false);
    }

    let mut width: f32 = 0.0;
    let mut height: f32 = 0.0;
    // (левый верхний, правый нижний, uv) в опорных пикселях, Y вниз
    let mut raw = Vec::new();
    let k = REF_PX / SDF_PX; // опорных пикселей в пикселе растеризации
    for run in buffer.layout_runs() {
        width = width.max(run.line_w);
        height = height.max(run.line_top + run.line_height);
        for glyph in run.glyphs {
            let (key, _, _) = CacheKey::new(
                glyph.font_id,
                glyph.glyph_id,
                SDF_PX,
                (0.0, 0.0),
                glyph.font_weight,
                // Хинтинг подгоняет контур под пиксели конкретного размера — для SDF вреден
                CacheKeyFlags::DISABLE_HINTING,
            );
            let Some(slot) = atlas.glyph(key, &mut system.fonts, &mut system.swash)? else {
                continue; // пробел или цветной глиф
            };
            let origin_x = glyph.x + glyph.x_offset * glyph.font_size;
            let baseline = run.line_y - glyph.y_offset * glyph.font_size;
            let min = Vec2::new(origin_x + slot.left * k, baseline + slot.top * k);
            let max = min + Vec2::new(slot.width, slot.height) * k;
            raw.push((min, max, slot.uv_min, slot.uv_max));
        }
    }
    if text.align != TextAlign::Left
        && let Some(w) = text.max_width
    {
        width = w / scale;
    }

    // В единицы `size`, Y вверх, относительно точки привязки
    let block = Vec2::new(width, height);
    let anchor = text.anchor * block;
    let convert = |p: Vec2| Vec2::new(p.x - anchor.x, anchor.y - p.y) * scale;
    let glyphs = raw
        .into_iter()
        .map(|(min, max, uv_min, uv_max)| {
            // Y переворачивается: верх глифа (min.y) становится max.y
            let (a, b) = (convert(min), convert(max));
            PlacedGlyph {
                min: Vec2::new(a.x, b.y),
                max: Vec2::new(b.x, a.y),
                uv_min,
                uv_max,
            }
        })
        .collect();
    Ok(TextLayout {
        glyphs,
        size: block * scale,
        atlas_generation: atlas.generation(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(text: &Text) -> TextLayout {
        let mut system = TextSystem::default();
        let mut atlas = GlyphAtlas::default();
        layout_one(text, &mut system, &mut atlas).expect("атлас не переполнен")
    }

    #[test]
    fn cyrillic_from_default_font() {
        let result = layout(&Text::new("Привет", 1.0));
        assert_eq!(
            result.glyphs.len(),
            6,
            "все буквы нашлись во встроенном шрифте"
        );
        // Высота блока — около межстрочного интервала (1.2 em), ширина — несколько em
        assert!((result.size.y - 1.2).abs() < 0.05, "{:?}", result.size);
        assert!(
            result.size.x > 2.0 && result.size.x < 5.0,
            "{:?}",
            result.size
        );
    }

    #[test]
    fn anchor_centers_block_and_size_scales() {
        let small = layout(&Text::new("Карта", 1.0));
        let big = layout(&Text::new("Карта", 10.0));
        assert!(
            (big.size.x / small.size.x - 10.0).abs() < 1e-3,
            "масштаб линейный"
        );
        // Центр блока — в начале координат: глифы по обе стороны
        let min_x = small
            .glyphs
            .iter()
            .map(|g| g.min.x)
            .fold(f32::MAX, f32::min);
        let max_x = small
            .glyphs
            .iter()
            .map(|g| g.max.x)
            .fold(f32::MIN, f32::max);
        assert!(
            min_x < 0.0 && max_x > 0.0 && (min_x + max_x).abs() < 0.3,
            "{min_x} {max_x}"
        );
    }

    #[test]
    fn wraps_to_max_width() {
        let mut text = Text::new("один два три четыре пять", 1.0);
        text.max_width = Some(4.0);
        let result = layout(&text);
        assert!(result.size.y > 2.0, "несколько строк: {:?}", result.size);
    }
}
