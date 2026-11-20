//! Компоненты текста.

use engine_assets::Handle;
use engine_ecs::prelude::*;
use engine_math::Vec2;
use engine_scene::Transform;

use crate::Font;

/// Где и как рисуется текст.
#[derive(Debug, Clone, Copy, PartialEq, Reflect)]
pub enum TextMode {
    /// В мире, в плоскости XY своего `Transform` (смотрит вдоль +Z). Размер — в метрах,
    /// закрывается объектами сцены. Вывески, надписи на стенах.
    World,
    /// Подпись над точкой мира: позиция — из `Transform`, всегда лицом к экрану, размер —
    /// в точках экрана (не меняется с расстоянием), поверх сцены. Метки на карте, имена.
    /// `offset` — сдвиг на экране в точках (Y вниз).
    Label { offset: Vec2 },
    /// На экране: `position` — точки от левого верхнего угла безопасной зоны (без выреза,
    /// часов и системных панелей; на desktop это угол окна). `Transform` не нужен.
    /// Счётчики, подсказки, HUD.
    Screen { position: Vec2 },
}

/// Обводка вокруг букв (читаемость на пёстром фоне).
#[derive(Debug, Clone, Copy, PartialEq, Reflect)]
pub struct Outline {
    /// Толщина в долях высоты шрифта (em), до 0.1.
    pub width: f32,
    /// sRGB, как `Text::color`.
    pub color: [f32; 4],
}

/// Выравнивание строк внутри блока.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Reflect)]
#[reflect(Default)]
pub enum TextAlign {
    #[default]
    Left,
    Center,
    Right,
}

/// Текст. Вёрстка (шейпинг, переносы, глифы в атласе) пересчитывается при изменении.
///
/// ```ignore
/// commands.spawn((Text::new("Привет, мир!", 0.5), Transform::from_xyz(0.0, 2.0, 0.0)));
/// commands.spawn(Text::screen("Очки: 0", 24.0, Vec2::new(16.0, 16.0)));
/// ```
#[derive(Component, Debug, Clone, PartialEq, Reflect)]
#[reflect(Component)]
#[require(Transform)]
pub struct Text {
    pub content: String,
    /// `None` — шрифт по умолчанию (Ubuntu Light, с кириллицей).
    pub font: Option<Handle<Font>>,
    /// Высота шрифта (em): метры для `World`, точки экрана для `Label` и `Screen`.
    pub size: f32,
    /// Цвет в sRGB (как в CSS и редакторах), с альфой.
    pub color: [f32; 4],
    pub outline: Option<Outline>,
    pub align: TextAlign,
    /// Какая точка блока текста стоит в позиции: (0, 0) — левый верхний угол,
    /// (0.5, 0.5) — центр, (0.5, 1) — середина низа.
    pub anchor: Vec2,
    /// Ширина, после которой строка переносится (в единицах `size`); `None` — без переносов.
    pub max_width: Option<f32>,
    /// Межстрочный интервал, множитель высоты шрифта.
    pub line_height: f32,
    pub mode: TextMode,
}

impl Text {
    /// Текст в мире высотой `size` метров, центр блока — в позиции `Transform`.
    pub fn new(content: impl Into<String>, size: f32) -> Self {
        Self {
            content: content.into(),
            font: None,
            size,
            color: [1.0; 4],
            outline: None,
            align: TextAlign::Center,
            anchor: Vec2::splat(0.5),
            max_width: None,
            line_height: 1.2,
            mode: TextMode::World,
        }
    }

    /// Подпись над точкой мира, размер в точках экрана; низ подписи — над точкой.
    pub fn label(content: impl Into<String>, size: f32) -> Self {
        Self {
            anchor: Vec2::new(0.5, 1.0),
            mode: TextMode::Label { offset: Vec2::ZERO },
            ..Self::new(content, size)
        }
    }

    /// Текст на экране: левый верхний угол — в `position` (точки от угла окна).
    pub fn screen(content: impl Into<String>, size: f32, position: Vec2) -> Self {
        Self {
            align: TextAlign::Left,
            anchor: Vec2::ZERO,
            mode: TextMode::Screen { position },
            ..Self::new(content, size)
        }
    }

    #[must_use]
    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }

    #[must_use]
    pub fn with_outline(mut self, width: f32, color: [f32; 4]) -> Self {
        self.outline = Some(Outline { width, color });
        self
    }

    #[must_use]
    pub fn with_font(mut self, font: Handle<Font>) -> Self {
        self.font = Some(font);
        self
    }
}

/// Глиф в свёрстанном тексте: прямоугольник относительно точки привязки (Y вверх) — в метрах
/// для `World`, в точках экрана для `Label`/`Screen` — и место в атласе.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlacedGlyph {
    /// Левый нижний угол.
    pub min: Vec2,
    /// Правый верхний.
    pub max: Vec2,
    /// Текстурные координаты левого **верхнего** угла `(min.x, max.y)` — в атласе Y вниз.
    pub uv_min: [f32; 2],
    /// …и правого нижнего `(max.x, min.y)`.
    pub uv_max: [f32; 2],
}

/// Результат вёрстки — добавляется и обновляется системой текста, рендер его только читает.
#[derive(Component, Debug, Clone, Default)]
pub struct TextLayout {
    pub glyphs: Vec<PlacedGlyph>,
    /// Размер блока (метры или точки, как у глифов).
    pub size: Vec2,
    /// Поколение атласа, для которого посчитаны текстурные координаты.
    pub(crate) atlas_generation: u64,
}

impl TextLayout {
    /// Текстурные координаты актуальны: атлас не пересоздавался после вёрстки.
    /// Иначе текст нужно пропустить в этом кадре — следующая вёрстка его исправит.
    pub fn matches(&self, atlas: &crate::GlyphAtlas) -> bool {
        self.atlas_generation == atlas.generation()
    }
}
