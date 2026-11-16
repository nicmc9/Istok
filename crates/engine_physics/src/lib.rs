//! Физика (R2-09): фасад. Бэкенд подключается отдельным крейтом (`engine_physics_rapier`).
//!
//! - Тело — сущность с [`RigidBody`], [`Collider`] и `Transform`. Дополнительно: [`Velocity`]
//!   (читать и задавать скорость), [`Impulse`] (толчок), [`Damping`], [`Ccd`].
//! - Шаг — в `FixedUpdate`, с частотой логики (`sim.hz`); рендер интерполирует между шагами.
//! - Код меняет тело через компоненты: новый `Transform` — телепорт (у кинематических —
//!   движение к цели), новая `Velocity` — скорость, новый масштаб — пересоздание формы.
//! - Столкновения — сообщения [`CollisionEvent`], лучи — [`Physics::raycast`].
//! - CVar: `physics.enabled` — создать мир (по умолчанию нет), `physics.debug` — коллайдеры
//!   линиями, `physics.paused` — остановить симуляцию.

mod backend;
mod components;
mod debug;
mod sync;

use engine_core::{App, CVar, CVarFlags, Plugin, Stage};
use engine_ecs::prelude::*;

pub use backend::{BodyDesc, Physics, PhysicsBackend, RayHit};
pub use components::{
    Ccd, Collider, ColliderShape, CollisionEvent, Damping, Gravity, Impulse, Pose, RigidBody,
    Velocity,
};

pub static PHYSICS_ENABLED: CVar<bool> = CVar::new(
    "physics.enabled",
    false,
    "Создать мир физики. Выкл — бэкенд не запускается",
)
.with_flags(CVarFlags::STARTUP);
pub static PHYSICS_DEBUG: CVar<bool> =
    CVar::new("physics.debug", false, "Рисовать коллайдеры линиями");
pub static PHYSICS_PAUSED: CVar<bool> =
    CVar::new("physics.paused", false, "Остановить симуляцию физики");

/// Системы физики. Не подключается сам по себе: его добавляет плагин бэкенда, положив
/// ресурс [`Physics`] со своей реализацией.
#[derive(Debug, Default)]
pub struct PhysicsPlugin;

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        assert!(
            app.world().contains_resource::<Physics>(),
            "PhysicsPlugin требует бэкенд: добавьте плагин бэкенда (например, RapierPhysicsPlugin)"
        );
        app.register_type::<RigidBody>()
            .register_type::<Collider>()
            .register_type::<Velocity>()
            .register_type::<Damping>()
            .register_type::<Ccd>()
            .register_cvar(&PHYSICS_DEBUG)
            .register_cvar(&PHYSICS_PAUSED)
            .init_resource::<Gravity>()
            .add_message::<CollisionEvent>()
            .add_systems(
                Stage::FixedUpdate,
                (
                    sync::sync_bodies,
                    sync::push_changes,
                    sync::step,
                    sync::pull_results,
                )
                    .chain(),
            )
            .add_systems(Stage::PostUpdate, debug::draw_colliders);
    }
}
