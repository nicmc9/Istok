//! Фасад движка. Проект зависит только от этого крейта.
//!
//! Обычно достаточно `use engine::prelude::*;`. Подсистемы доступны и по отдельности:
//! `engine::ecs`, `engine::math`, `engine::platform`, `engine::render`.

pub use engine_assets as assets;
pub use engine_console as console;
pub use engine_core::{
    AnyCVar, App, AppControl, AppLifecycle, CHEATS, CVar, CVarError, CVarFlags, CVarSource,
    CVarString, CVarValue, CVars, FixedTime, LOG_FILTER, MainWindow, Plugin, SIM_HZ, SIM_MAX_STEPS,
    Stage, Time, Window, WindowResized,
};
pub use engine_debug as debug;
pub use engine_ecs as ecs;
pub use engine_input as input;
pub use engine_math as math;
pub use engine_net as net;
pub use engine_platform as platform;
/// Звук: фасад и бэкенд kira (`KiraAudioPlugin`). Feature `audio`.
#[cfg(feature = "audio")]
pub mod audio {
    pub use engine_audio::*;
    pub use engine_audio_kira::{KiraAudioPlugin, KiraBackend};
}
/// Физика: фасад и бэкенд rapier (`RapierPhysicsPlugin`). Feature `physics`.
#[cfg(feature = "physics")]
pub mod physics {
    pub use engine_physics::*;
    pub use engine_physics_rapier::{RapierBackend, RapierPhysicsPlugin};
}
pub use engine_2d as twod;
pub use engine_render as render;
pub use engine_scene as scene;
pub use engine_serialize as serialize;
pub use engine_text as text;

/// Логи и зоны профайлера: `engine::log::info!("...")`, `engine::log::trace_span!("zone")`.
///
/// [Rust] Макросы реэкспортируются как обычные элементы. Внутри они ссылаются на свой крейт
/// через `$crate`, поэтому работают, даже если проект не зависит от `tracing` напрямую.
pub mod log {
    pub use tracing::{debug, debug_span, error, info, info_span, trace, trace_span, warn};
}

pub mod prelude {
    //! Всё, что нужно в большинстве файлов проекта.

    #[cfg(feature = "audio")]
    pub use crate::audio::{AudioClip, AudioListener, AudioPlayer, KiraAudioPlugin};
    #[cfg(feature = "physics")]
    pub use crate::physics::{
        Collider, CollisionEvent, Impulse, Physics, RapierPhysicsPlugin, RigidBody, Velocity,
    };
    pub use crate::{
        App, AppControl, AppLifecycle, CVar, CVarFlags, CVars, FixedTime, Plugin, Stage, Time,
        Window, WindowResized,
    };
    pub use engine_2d::{Atlas, Camera2d, Plugin2d, Sprite};
    pub use engine_assets::{
        AssetAppExt, AssetEvent, AssetPlugin, AssetServer, AssetStatus, Assets, Handle,
    };
    pub use engine_console::{
        CommandResult, Console, ConsoleAppExt, ConsoleCommand, ConsolePlugin,
    };
    pub use engine_debug::DebugPlugin;
    pub use engine_ecs::prelude::*;
    pub use engine_input::{Input, InputPlugin, Key, MouseButton};
    pub use engine_math::*;
    pub use engine_net::{NetPlugin, NetStats};
    pub use engine_render::{ClearColor, Color, DebugDraw, RenderPlugin, RenderStats, Ui};
    pub use engine_scene::{
        AmbientLight, Camera, Decal, DirectionalLight, GlobalTransform, Mesh3d, ParticleEmitter,
        PointLight, ScenePlugin, SceneRoot, SceneSpawned, SpotLight, Transform, Visibility,
    };
    pub use engine_serialize::{LoadScene, SerializePlugin};
    pub use engine_text::{Font, Text, TextAlign, TextMode, TextPlugin};
}
