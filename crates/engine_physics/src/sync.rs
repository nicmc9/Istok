//! Системы шага физики (стадия `FixedUpdate`, строго по порядку):
//! 1. `sync_bodies` — новые, изменённые и удалённые тела → бэкенд;
//! 2. `push_changes` — изменения из кода (телепорт, скорость, масштаб, толчки) → бэкенд;
//! 3. `step` — шаг симуляции, события столкновений;
//! 4. `pull_results` — положения и скорости динамических тел → `Transform`, `Velocity`.
//!
//! Рендер интерполирует `Transform` между шагами — физика выглядит плавно на любой частоте экрана.

use engine_core::FixedTime;
use engine_ecs::prelude::*;
use engine_scene::Transform;

use crate::PHYSICS_PAUSED;
use crate::backend::{BodyDesc, Physics};
use crate::components::{
    BodyState, Ccd, Collider, CollisionEvent, Damping, Gravity, Impulse, Pose, RigidBody, Velocity,
};

/// Описание тела для бэкенда из компонентов сущности.
fn describe(
    kind: RigidBody,
    collider: &Collider,
    transform: &Transform,
    velocity: Velocity,
    damping: Option<&Damping>,
    ccd: bool,
) -> BodyDesc {
    BodyDesc {
        kind,
        pose: Pose::from(transform),
        collider: Collider {
            shape: collider.shape.scaled(transform.scale),
            ..collider.clone()
        },
        velocity,
        damping: damping.copied().unwrap_or_default(),
        ccd,
    }
}

fn state(transform: &Transform, velocity: Velocity) -> BodyState {
    BodyState {
        pose: Pose::from(transform),
        scale: transform.scale,
        velocity,
    }
}

type ChangedBody<'a> = (
    Entity,
    &'a RigidBody,
    &'a Collider,
    &'a Transform,
    &'a Velocity,
    Option<&'a Damping>,
    Has<Ccd>,
    Has<ChildOf>,
);

/// Тело нужно (пере)создать: появились или изменились компоненты, из которых оно строится.
type BodyChanged = Or<(Changed<RigidBody>, Changed<Collider>, Changed<Damping>)>;

pub(crate) fn sync_bodies(
    mut commands: Commands,
    mut physics: ResMut<Physics>,
    mut removed_bodies: RemovedComponents<RigidBody>,
    mut removed_colliders: RemovedComponents<Collider>,
    // [Rust] `Or<(Changed<A>, Changed<B>)>` — фильтр запроса: изменилось хотя бы одно
    // (только что добавленный компонент тоже считается изменённым)
    changed: Query<ChangedBody, BodyChanged>,
) {
    let backend = physics.backend_mut();
    for entity in removed_bodies.read().chain(removed_colliders.read()) {
        backend.remove_body(entity);
        // Сущность могла быть удалена целиком — тогда снимать нечего
        if let Ok(mut entity) = commands.get_entity(entity) {
            entity.try_remove::<BodyState>();
        }
    }
    for (entity, kind, collider, transform, velocity, damping, ccd, child) in &changed {
        if child {
            tracing::warn!("Тело {entity} — дочерняя сущность: физика работает только с корневыми");
            continue;
        }
        backend.remove_body(entity);
        backend.add_body(
            entity,
            &describe(*kind, collider, transform, *velocity, damping, ccd),
        );
        commands.entity(entity).insert(state(transform, *velocity));
    }
}

type ExistingBody<'a> = (
    Entity,
    &'a RigidBody,
    &'a Collider,
    &'a Transform,
    &'a Velocity,
    Option<&'a Damping>,
    Has<Ccd>,
    &'a mut BodyState,
    Option<&'a mut Impulse>,
);

pub(crate) fn push_changes(mut physics: ResMut<Physics>, mut bodies: Query<ExistingBody>) {
    let backend = physics.backend_mut();
    for (entity, kind, collider, transform, velocity, damping, ccd, mut state, impulse) in
        &mut bodies
    {
        if transform.scale != state.scale {
            // Масштаб меняет форму — тело пересоздаётся (скорость сохраняется)
            backend.remove_body(entity);
            backend.add_body(
                entity,
                &describe(*kind, collider, transform, *velocity, damping, ccd),
            );
            *state = self::state(transform, *velocity);
            continue;
        }
        let pose = Pose::from(transform);
        if pose != state.pose {
            backend.set_pose(entity, pose);
            state.pose = pose;
        }
        if *velocity != state.velocity && *kind == RigidBody::Dynamic {
            backend.set_velocity(entity, *velocity);
            state.velocity = *velocity;
        }
        if let Some(mut impulse) = impulse
            && *impulse != Impulse::default()
        {
            backend.apply_impulse(entity, impulse.linear, impulse.angular);
            *impulse = Impulse::default();
        }
    }
}

pub(crate) fn step(
    mut physics: ResMut<Physics>,
    gravity: Res<Gravity>,
    fixed: Res<FixedTime>,
    mut writer: MessageWriter<CollisionEvent>,
    // Буфер событий переиспользуется между шагами — без аллокаций
    mut events: Local<Vec<CollisionEvent>>,
) {
    if PHYSICS_PAUSED.get() {
        return;
    }
    let _zone = tracing::trace_span!("physics_step").entered();
    physics
        .backend_mut()
        .step(fixed.step_secs(), gravity.0, &mut events);
    writer.write_batch(events.drain(..));
}

pub(crate) fn pull_results(
    physics: Res<Physics>,
    mut bodies: Query<(&mut Transform, &mut Velocity, &mut BodyState)>,
) {
    if PHYSICS_PAUSED.get() {
        return;
    }
    physics
        .backend()
        .for_each_moved(&mut |entity, pose, velocity| {
            let Ok((mut transform, mut body_velocity, mut state)) = bodies.get_mut(entity) else {
                return;
            };
            transform.translation = pose.translation;
            transform.rotation = pose.rotation;
            *body_velocity = velocity;
            state.pose = pose;
            state.velocity = velocity;
        });
}
