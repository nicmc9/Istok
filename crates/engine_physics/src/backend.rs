//! Трейт бэкенда физики и ресурс [`Physics`] — точка доступа к нему из систем.
//!
//! Бэкенд хранит своё представление мира и знает тела по `Entity`. Фасад сам решает, когда
//! добавить, удалить или передвинуть тело (системы в `sync.rs`), бэкенд только исполняет.
//! Новый бэкенд — отдельный крейт с реализацией трейта и плагином, который кладёт
//! `Physics::new(бэкенд)` и подключает `PhysicsPlugin` (см. `engine_physics_rapier`).

use engine_ecs::prelude::*;
use engine_math::Vec3;

use crate::components::{Collider, CollisionEvent, Damping, Pose, RigidBody, Velocity};

/// Всё, что нужно бэкенду для создания тела.
#[derive(Debug, Clone)]
pub struct BodyDesc {
    pub kind: RigidBody,
    pub pose: Pose,
    /// Коллайдер с уже применённым масштабом `Transform`.
    pub collider: Collider,
    pub velocity: Velocity,
    pub damping: Damping,
    pub ccd: bool,
}

/// Попадание луча.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    pub entity: Entity,
    /// Расстояние от начала луча (направление нормализуется).
    pub distance: f32,
    pub point: Vec3,
    /// Нормаль поверхности в точке попадания.
    pub normal: Vec3,
}

/// Бэкенд физики.
// [Rust] `Send + Sync + 'static` — можно хранить в ресурсе ECS и обращаться из любого потока.
pub trait PhysicsBackend: Send + Sync + 'static {
    /// Создаёт тело. Тело с таким `entity` уже удалено вызывающим.
    fn add_body(&mut self, entity: Entity, desc: &BodyDesc);
    /// Удаляет тело, если оно есть.
    fn remove_body(&mut self, entity: Entity);
    /// Переносит тело. Для кинематического — цель на следующий шаг (тело «доедет»
    /// и растолкает соседей), для остальных — мгновенно.
    fn set_pose(&mut self, entity: Entity, pose: Pose);
    fn set_velocity(&mut self, entity: Entity, velocity: Velocity);
    fn apply_impulse(&mut self, entity: Entity, linear: Vec3, angular: Vec3);
    /// Шаг симуляции; события столкновений — в `events`.
    fn step(&mut self, dt: f32, gravity: Vec3, events: &mut Vec<CollisionEvent>);
    /// Динамические тела, которые двигались на последнем шаге (спящие пропускаются).
    // [Rust] `&mut dyn FnMut` вместо обобщённого параметра: трейт остаётся «объектным» —
    // его можно хранить как `Box<dyn PhysicsBackend>` (аналог виртуального метода).
    fn for_each_moved(&self, f: &mut dyn FnMut(Entity, Pose, Velocity));
    /// Первое попадание луча. `direction` не обязан быть единичным.
    fn raycast(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
        exclude: Option<Entity>,
    ) -> Option<RayHit>;
    fn body_count(&self) -> usize;
}

/// Физический мир (ресурс ECS): запросы к физике из систем.
///
/// ```ignore
/// fn aim(physics: Res<Physics>, camera: Query<&Transform, With<Camera>>) {
///     if let Some(hit) = physics.raycast(origin, direction, 100.0) { /* hit.entity */ }
/// }
/// ```
#[derive(Resource)]
pub struct Physics {
    backend: Box<dyn PhysicsBackend>,
}

impl Physics {
    pub fn new(backend: impl PhysicsBackend) -> Self {
        Self {
            backend: Box::new(backend),
        }
    }

    pub fn raycast(&self, origin: Vec3, direction: Vec3, max_distance: f32) -> Option<RayHit> {
        self.backend.raycast(origin, direction, max_distance, None)
    }

    /// Луч, не видящий тело `exclude` (например, самого стреляющего).
    pub fn raycast_excluding(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
        exclude: Entity,
    ) -> Option<RayHit> {
        self.backend
            .raycast(origin, direction, max_distance, Some(exclude))
    }

    pub fn body_count(&self) -> usize {
        self.backend.body_count()
    }

    pub(crate) fn backend(&self) -> &dyn PhysicsBackend {
        &*self.backend
    }

    pub(crate) fn backend_mut(&mut self) -> &mut dyn PhysicsBackend {
        &mut *self.backend
    }
}

impl std::fmt::Debug for Physics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Physics")
            .field("bodies", &self.backend.body_count())
            .finish_non_exhaustive()
    }
}
