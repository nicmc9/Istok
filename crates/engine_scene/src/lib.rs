//! Сцена в ECS (`docs/acrhitecture.md`, раздел 6).
//!
//! - [`Transform`] — положение относительно родителя, [`GlobalTransform`] — в мире
//!   (считается в `PostUpdate`).
//! - Иерархия — компоненты `ChildOf`/`Children` из ECS. Отдельного графа сцены нет.
//! - [`Camera`], [`DirectionalLight`], [`PointLight`], [`SpotLight`], ресурс [`AmbientLight`].
//! - [`SceneRoot`] — разворачивает glTF в дерево сущностей, когда файл загрузится.
//! - [`SpritePlayer`] — кадры атласа, [`PropertyPlayer`] — кривые в поля компонентов.
//! - [`ParticleEmitter`] и [`Decal`] — частицы на CPU и проекция текстуры на геометрию.

mod animation;
mod components;
mod fx;
mod gltf_scene;
mod propagate;
mod property;
mod sprite;

use engine_assets::AssetAppExt;
use engine_core::{App, Plugin, Stage};
use engine_ecs::prelude::{ChildOf, IntoScheduleConfigs, Name};

pub use animation::{AnimationPlayer, SceneNodes};
pub use components::{
    AmbientLight, Camera, DirectionalLight, FitAxis, GlobalTransform, Mesh3d, PointLight,
    SpotLight, Transform, Visibility,
};
pub use fx::{Decal, Particle, ParticleCloud, ParticleEmitter};
pub use gltf_scene::{SceneRoot, SceneSpawned, SkinnedMesh};
pub use propagate::{is_visible, propagate_transforms};
pub use property::{Curve, Keys, PropertyClip, PropertyPlayer, PropertyTrack, Tween};
pub use sprite::{SpriteClip, SpriteFrame, SpritePlayer};

/// Подключает сцену. Требует `AssetPlugin` (для glTF).
#[derive(Debug, Default)]
pub struct ScenePlugin;

impl Plugin for ScenePlugin {
    fn build(&self, app: &mut App) {
        // Типы для сцен (сохранение, загрузка) — решение D7. Служебные (GlobalTransform,
        // кости, состояние анимации) не регистрируются: их восстанавливают системы
        app.register_type::<Transform>()
            .register_type::<Visibility>()
            .register_type::<Mesh3d>()
            .register_type::<Camera>()
            .register_type::<FitAxis>()
            .register_type::<DirectionalLight>()
            .register_type::<PointLight>()
            .register_type::<SpotLight>()
            .register_type::<SceneRoot>()
            .register_type::<ChildOf>()
            .register_type::<Name>()
            .register_type::<fx::ParticleEmitter>()
            .register_type::<fx::Decal>();
        app.init_resource::<AmbientLight>()
            .add_systems(Stage::PreUpdate, gltf_scene::spawn_gltf_scenes)
            .init_asset::<SpriteClip>()
            .init_asset::<PropertyClip>()
            .add_systems(
                Stage::FixedUpdate,
                (
                    animation::animate,
                    sprite::animate_sprites,
                    property::animate_properties,
                    fx::simulate,
                )
                    .chain(),
            )
            .add_systems(
                Stage::PostUpdate,
                (
                    propagate::propagate_transforms,
                    gltf_scene::bind_pose_transforms,
                )
                    .chain(),
            );
    }
}
