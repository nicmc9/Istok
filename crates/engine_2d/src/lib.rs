//! Плагин 2D (R2-13): спрайты, атласы и ортографическая камера.
//!
//! Рисование — в `engine_render`: после 3D-сцены, без освещения, порядок — поле `layer`
//! (больший слой ближе к камере). Камера смотрит вдоль −Z; поставьте её впереди спрайтов,
//! например `Transform::from_xyz(0.0, 0.0, 10.0)`.
//!
//! Покадровый UV берётся у [`engine_scene::SpritePlayer`], если он есть на той же сущности.

use engine_assets::formats::Image;
use engine_assets::{Asset, AssetAppExt, Handle};
use engine_core::{App, Plugin, Stage};
use engine_ecs::prelude::*;
use engine_math::Vec2;
use engine_scene::{FitAxis, SpriteClip, SpritePlayer, Transform};

/// Прямоугольный спрайт в плоскости XY. `uv` — лево, верх, право, низ текстуры
/// (`(0, 0)` — левый верхний угол, как у [`SpriteClip`]).
#[derive(Component, Debug, Clone, Reflect)]
#[reflect(Component)]
#[require(Transform)]
pub struct Sprite {
    pub image: Handle<Image>,
    pub uv: [f32; 4],
    /// Линейный цвет-множитель.
    pub color: [f32; 4],
    /// Кто выше — рисуется позже, поверх остальных спрайтов.
    pub layer: i32,
    /// Размер в мировых единицах, до масштаба `Transform`.
    pub size: Vec2,
}

impl Sprite {
    pub fn new(image: Handle<Image>, size: Vec2) -> Self {
        Self {
            image,
            uv: [0.0, 0.0, 1.0, 1.0],
            color: [1.0; 4],
            layer: 0,
            size,
        }
    }

    #[must_use]
    pub fn with_uv(mut self, uv: [f32; 4]) -> Self {
        self.uv = uv;
        self
    }

    #[must_use]
    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }

    #[must_use]
    pub fn with_layer(mut self, layer: i32) -> Self {
        self.layer = layer;
        self
    }
}

/// Кадры одной текстуры. Сам по себе не рисуется: [`Atlas::sprite`] собирает [`Sprite`].
#[derive(Debug, Clone, TypePath)]
pub struct Atlas {
    pub image: Handle<Image>,
    /// Прямоугольники UV, как у [`SpriteClip::grid`].
    pub frames: Vec<[f32; 4]>,
    names: Vec<(String, usize)>,
}

impl Atlas {
    /// Сетка: слева направо, сверху вниз.
    pub fn grid(image: Handle<Image>, columns: u32, rows: u32) -> Self {
        let clip = SpriteClip::grid(columns, rows, 1.0);
        Self {
            image,
            frames: clip.frames.iter().map(|frame| frame.uv).collect(),
            names: Vec::new(),
        }
    }

    /// Имя кадра. Индекс должен попадать в сетку.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>, frame: usize) -> Self {
        debug_assert!(
            frame < self.frames.len(),
            "кадра {frame} нет в атласе из {}",
            self.frames.len()
        );
        self.names.push((name.into(), frame));
        self
    }

    pub fn index(&self, name: &str) -> Option<usize> {
        self.names
            .iter()
            .find(|(named, _)| named == name)
            .map(|(_, index)| *index)
    }

    pub fn sprite(&self, frame: usize, size: Vec2) -> Option<Sprite> {
        let uv = *self.frames.get(frame)?;
        Some(Sprite::new(self.image.clone(), size).with_uv(uv))
    }

    pub fn sprite_named(&self, name: &str, size: Vec2) -> Option<Sprite> {
        self.sprite(self.index(name)?, size)
    }
}

impl Asset for Atlas {
    fn memory_size(&self) -> u64 {
        let names: usize = self.names.iter().map(|(name, _)| name.len()).sum();
        u64::try_from(self.frames.len() * 16 + names).unwrap_or(u64::MAX)
    }
}

/// Ортографическая камера. `size` — видимая область в единицах мира по стороне `fit`,
/// другая сторона следует за окном. Смотрит вдоль −Z своего [`Transform`].
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(Component, Default)]
#[require(Transform)]
pub struct Camera2d {
    pub size: f32,
    /// По умолчанию по меньшей стороне: в портрете на телефоне края не обрезаются.
    #[reflect(default)]
    pub fit: FitAxis,
    /// Если камер несколько, берётся меньший номер. Разные вьюпорты — задача Б-09.
    pub order: i32,
}

impl Default for Camera2d {
    fn default() -> Self {
        Self {
            size: 10.0,
            fit: FitAxis::default(),
            order: 0,
        }
    }
}

impl Camera2d {
    /// Ширина и высота видимой области при соотношении сторон `aspect` (ширина / высота).
    pub fn visible_size(&self, aspect: f32) -> (f32, f32) {
        let aspect = aspect.max(1e-4);
        let height = self.fit.height(self.size.max(1e-4), aspect);
        (height * aspect, height)
    }
}

/// Подключает 2D. Требует `AssetPlugin` и `ScenePlugin` (трансформы).
/// Рисовать начинает рендер, плагин только регистрирует типы и переносит кадр анимации на спрайт.
#[derive(Debug, Default)]
pub struct Plugin2d;

impl Plugin for Plugin2d {
    fn build(&self, app: &mut App) {
        app.register_type::<Sprite>()
            .register_type::<Camera2d>()
            .init_asset::<Atlas>()
            .add_systems(Stage::PostUpdate, apply_sprite_frames);
    }
}

/// Проигрыватель кадров (R2-06) пишет UV в спрайт, когда кадр уже выбран.
fn apply_sprite_frames(mut sprites: Query<(&SpritePlayer, &mut Sprite)>) {
    for (player, mut sprite) in &mut sprites {
        if let Some(uv) = player.sampled_uv() {
            sprite.uv = uv;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_assets::formats::{ColorSpace, Image};
    use engine_assets::{AssetPlugin, Assets, Handle};
    use engine_core::{App, Duration, FixedTime, Instant};
    use engine_math::Vec2;
    use engine_scene::{ScenePlugin, SpriteClip, SpritePlayer, Transform};

    #[test]
    fn grid_matches_sprite_clip_and_names() {
        let mut app = App::new();
        app.add_plugin(engine_net::NetPlugin)
            .add_plugin(AssetPlugin)
            .add_plugin(Plugin2d);
        let image = white_pixel(&mut app);
        let handle = app
            .world_mut()
            .resource_mut::<Assets<Atlas>>()
            .add(Atlas::grid(image, 2, 2).with_name("нижний-левый", 2));
        let atlas = app
            .world()
            .resource::<Assets<Atlas>>()
            .get_loaded(&handle)
            .unwrap();
        let clip = SpriteClip::grid(2, 2, 1.0);
        assert_eq!(atlas.frames[0], clip.frames[0].uv);
        assert_eq!(atlas.frames[2], [0.0, 0.5, 0.5, 1.0]);
        assert_eq!(atlas.index("нижний-левый"), Some(2));
        assert!(atlas.index("нет").is_none());
        let sprite = atlas.sprite_named("нижний-левый", Vec2::ONE).unwrap();
        assert_eq!(sprite.uv, [0.0, 0.5, 0.5, 1.0]);
    }

    fn white_pixel(app: &mut App) -> Handle<Image> {
        let mut image = Image {
            width: 1,
            height: 1,
            color_space: ColorSpace::Srgb,
            data: vec![255, 255, 255, 255],
            mips: Vec::new(),
            basis: Vec::new(),
        };
        image.generate_mips();
        app.world_mut().resource_mut::<Assets<Image>>().add(image)
    }

    #[test]
    fn visible_height_sets_width_from_aspect() {
        let camera = Camera2d {
            size: 10.0,
            ..Camera2d::default()
        };
        let (width, height) = camera.visible_size(2.0);
        assert!((width - 20.0).abs() < 1e-4);
        assert!((height - 10.0).abs() < 1e-4);
        // Портрет: те же 10 по ширине, высота растёт
        let (width, height) = camera.visible_size(0.5);
        assert!((width - 10.0).abs() < 1e-4);
        assert!((height - 20.0).abs() < 1e-4);
    }

    #[test]
    fn player_frame_is_copied_onto_sprite() {
        let mut app = App::new();
        app.add_plugin(engine_net::NetPlugin)
            .add_plugin(AssetPlugin)
            .add_plugin(ScenePlugin)
            .add_plugin(Plugin2d);
        let image = white_pixel(&mut app);
        let clip = app
            .world_mut()
            .resource_mut::<Assets<SpriteClip>>()
            .add(SpriteClip::grid(2, 2, 0.25));
        let entity = app
            .world_mut()
            .spawn((
                Sprite::new(image, Vec2::ONE),
                SpritePlayer::new(clip),
                Transform::default(),
            ))
            .id();
        app.world_mut()
            .insert_resource(FixedTime::new(Duration::from_secs_f64(0.5), 4));
        let start = Instant::now();
        app.update_at(start);
        app.update_at(start + Duration::from_secs_f64(0.5));
        let uv = app.world().get::<Sprite>(entity).unwrap().uv;
        assert_eq!(uv, [0.0, 0.5, 0.5, 1.0]);
    }
}
