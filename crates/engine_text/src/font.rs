//! Шрифт как ресурс: `server.load::<Font>("fonts/NotoSans-Regular.ttf")`.

use engine_assets::{Asset, AssetError, AssetLoader, Bytes, LoadContext};
use engine_core::tasks::BoxedFuture;
use engine_ecs::prelude::{TypePath, bevy_reflect};

/// Файл шрифта TrueType / OpenType. Разбирает его библиотека вёрстки при регистрации.
#[derive(Debug, Clone, TypePath)]
pub struct Font {
    pub data: Bytes,
}

impl Asset for Font {}

#[derive(Debug)]
pub struct FontLoader;

impl AssetLoader for FontLoader {
    type Asset = Font;

    fn extensions(&self) -> &[&'static str] {
        &["ttf", "otf"]
    }

    fn load<'a>(
        &'a self,
        bytes: Bytes,
        ctx: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<Font, AssetError>> {
        Box::pin(async move {
            // Быстрая проверка сигнатуры: полный разбор — при регистрации в FontSystem
            let tag = bytes.get(..4).unwrap_or_default();
            let known: [&[u8]; 3] = [&[0, 1, 0, 0], b"OTTO", b"true"];
            if !known.contains(&tag) {
                return Err(AssetError::format(ctx.path(), "не TrueType/OpenType"));
            }
            Ok(Font { data: bytes })
        })
    }
}
