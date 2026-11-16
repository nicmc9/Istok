//! Текст (R2-03): шрифты, вёрстка, SDF-атлас глифов.
//!
//! - [`Text`] — компонент: строка, шрифт, размер, цвет, обводка, режим ([`TextMode`]):
//!   в мире, подпись над точкой или на экране.
//! - [`Font`] — шрифт как ресурс (`.ttf`, `.otf`). Без шрифта — встроенный Ubuntu Light
//!   (латиница и кириллица); недостающие символы ищутся в остальных загруженных шрифтах.
//! - Вёрстка — `cosmic-text` (шейпинг HarfBuzz-совместимым `harfrust`, переносы, двунаправленный
//!   текст), растеризация — `swash`. Результат — компонент [`TextLayout`].
//! - Глифы хранятся как поля расстояний (SDF, см. `sdf.rs`) в общем атласе [`GlyphAtlas`]:
//!   одна растеризация на глиф для любых размеров, чёткие края, обводка без затрат.
//!
//! Рисует текст `engine_render`: читает `TextLayout` и изменения атласа.
//!
//! Ограничения: MSDF (острые углы у крупного текста) и цветные глифы (эмодзи) — позже.

mod atlas;
mod font;
mod layout;
mod sdf;
mod text;

use engine_assets::AssetAppExt;
use engine_core::{App, Plugin, Stage};
use engine_ecs::prelude::*;

pub use atlas::{AtlasChanges, AtlasUpdate, GlyphAtlas, SDF_PX, SPREAD};
pub use font::{Font, FontLoader};
pub use layout::TextSystem;
pub use text::{Outline, PlacedGlyph, Text, TextAlign, TextLayout, TextMode};

/// Подключает текст. Требует `AssetPlugin`.
#[derive(Debug, Default)]
pub struct TextPlugin;

impl Plugin for TextPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Text>()
            .register_loader(FontLoader)
            .init_resource::<TextSystem>()
            .init_resource::<GlyphAtlas>()
            .add_systems(
                Stage::PostUpdate,
                // [Rust] `.chain()` — системы кортежа выполняются строго по порядку
                (layout::register_fonts, layout::layout_texts).chain(),
            );
    }
}
