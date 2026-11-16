//! Бэкенд физики на `rapier3d` (R2-09).
//!
//! ```ignore
//! App::new().add_plugin(RapierPhysicsPlugin) // + компоненты из engine_physics
//! ```
//!
//! Типы rapier наружу не выходят: всё общение — через трейт `PhysicsBackend`. Математика
//! rapier — glam другой версии, поэтому векторы на границе переводятся через массивы
//! (это копирование 12–16 байт, не дороже обычного присваивания).

use std::collections::HashMap;
use std::sync::Mutex;

use engine_core::{App, Plugin};
use engine_ecs::prelude::*;
use engine_math::{Quat, Vec3};
use engine_physics::{
    BodyDesc, ColliderShape, CollisionEvent, PHYSICS_ENABLED, Physics, PhysicsBackend,
    PhysicsPlugin, Pose, RayHit, RigidBody, Velocity,
};
use rapier3d::prelude as rp;

/// Подключает физику с бэкендом rapier.
#[derive(Debug, Default)]
pub struct RapierPhysicsPlugin;

impl Plugin for RapierPhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.register_cvar(&PHYSICS_ENABLED);
        if !PHYSICS_ENABLED.get() {
            tracing::info!("Физика выключена (physics.enabled)");
            return;
        }
        app.insert_resource(Physics::new(RapierBackend::default()))
            .add_plugin(PhysicsPlugin);
    }
}

/// Мир rapier и соответствие сущностей его телам.
#[derive(Default)]
pub struct RapierBackend {
    world: rp::PhysicsWorld,
    bodies: HashMap<Entity, rp::RigidBodyHandle>,
}

// ---------- перевод типов на границе ----------

fn to_rp(v: Vec3) -> rp::Vector {
    rp::Vector::from_array(v.to_array())
}

fn from_rp(v: rp::Vector) -> Vec3 {
    Vec3::from_array(v.to_array())
}

fn pose_to_rp(pose: Pose) -> rp::Pose {
    rp::Pose::from_parts(
        to_rp(pose.translation),
        rp::Rotation::from_array(pose.rotation.to_array()),
    )
}

fn pose_from_rp(pose: &rp::Pose) -> Pose {
    Pose {
        translation: from_rp(pose.translation),
        rotation: Quat::from_array(pose.rotation.to_array()),
    }
}

/// Сущность хранится в `user_data` тела и коллайдера — так события и лучи rapier
/// переводятся обратно в сущности без отдельной таблицы.
fn entity_bits(entity: Entity) -> u128 {
    u128::from(entity.to_bits())
}

fn entity_from(user_data: u128) -> Option<Entity> {
    Entity::try_from_bits(u64::try_from(user_data).ok()?)
}

fn collider_builder(desc: &BodyDesc) -> Option<rp::ColliderBuilder> {
    let builder = match &desc.collider.shape {
        ColliderShape::Cuboid { half_extents: h } => rp::ColliderBuilder::cuboid(h.x, h.y, h.z),
        ColliderShape::Ball { radius } => rp::ColliderBuilder::ball(*radius),
        ColliderShape::Capsule {
            half_height,
            radius,
        } => rp::ColliderBuilder::capsule_y(*half_height, *radius),
        ColliderShape::TriMesh { vertices, indices } => {
            let vertices = vertices.iter().map(|v| to_rp(*v)).collect();
            match rp::ColliderBuilder::trimesh(vertices, indices.clone()) {
                Ok(builder) => builder,
                Err(error) => {
                    tracing::error!("Коллайдер из меша не создан: {error}");
                    return None;
                }
            }
        }
    };
    Some(
        builder
            .friction(desc.collider.friction)
            .restitution(desc.collider.restitution)
            .density(desc.collider.density)
            .sensor(desc.collider.sensor)
            .active_events(rp::ActiveEvents::COLLISION_EVENTS),
    )
}

impl std::fmt::Debug for RapierBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RapierBackend")
            .field("bodies", &self.bodies.len())
            .finish_non_exhaustive()
    }
}

impl PhysicsBackend for RapierBackend {
    fn add_body(&mut self, entity: Entity, desc: &BodyDesc) {
        let builder = match desc.kind {
            RigidBody::Dynamic => rp::RigidBodyBuilder::dynamic(),
            RigidBody::Fixed => rp::RigidBodyBuilder::fixed(),
            RigidBody::Kinematic => rp::RigidBodyBuilder::kinematic_position_based(),
        };
        let body = builder
            .pose(pose_to_rp(desc.pose))
            .linvel(to_rp(desc.velocity.linear))
            .angvel(to_rp(desc.velocity.angular))
            .linear_damping(desc.damping.linear)
            .angular_damping(desc.damping.angular)
            .ccd_enabled(desc.ccd)
            .user_data(entity_bits(entity));
        let handle = match collider_builder(desc) {
            Some(collider) => {
                self.world
                    .insert(body, collider.user_data(entity_bits(entity)))
                    .0
            }
            // Без формы тело всё равно создаётся: компоненты сущности остаются рабочими
            None => self.world.insert_body(body),
        };
        self.bodies.insert(entity, handle);
    }

    fn remove_body(&mut self, entity: Entity) {
        if let Some(handle) = self.bodies.remove(&entity) {
            self.world.remove_body(handle);
        }
    }

    fn set_pose(&mut self, entity: Entity, pose: Pose) {
        let Some(body) = self
            .bodies
            .get(&entity)
            .and_then(|h| self.world.bodies.get_mut(*h))
        else {
            return;
        };
        if body.is_kinematic() {
            body.set_next_kinematic_position(pose_to_rp(pose));
        } else {
            body.set_position(pose_to_rp(pose), true);
        }
    }

    fn set_velocity(&mut self, entity: Entity, velocity: Velocity) {
        if let Some(body) = self
            .bodies
            .get(&entity)
            .and_then(|h| self.world.bodies.get_mut(*h))
        {
            body.set_linvel(to_rp(velocity.linear), true);
            body.set_angvel(to_rp(velocity.angular), true);
        }
    }

    fn apply_impulse(&mut self, entity: Entity, linear: Vec3, angular: Vec3) {
        if let Some(body) = self
            .bodies
            .get(&entity)
            .and_then(|h| self.world.bodies.get_mut(*h))
        {
            body.apply_impulse(to_rp(linear), true);
            body.apply_torque_impulse(to_rp(angular), true);
        }
    }

    fn step(&mut self, dt: f32, gravity: Vec3, events: &mut Vec<CollisionEvent>) {
        self.world.integration_parameters.dt = dt;
        self.world.gravity = to_rp(gravity);
        let collector = Collector::default();
        self.world.step_with_events(&(), &collector);
        let raw = collector
            .events
            .into_inner()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Пары коллайдеров → пары сущностей
        let colliders = &self.world.colliders;
        let entity = |handle: rp::ColliderHandle| {
            colliders.get(handle).and_then(|c| entity_from(c.user_data))
        };
        events.extend(raw.into_iter().filter_map(|event| {
            let (a, b) = (entity(event.collider1())?, entity(event.collider2())?);
            let sensor = event.sensor();
            Some(if event.started() {
                CollisionEvent::Started { a, b, sensor }
            } else {
                CollisionEvent::Stopped { a, b, sensor }
            })
        }));
    }

    fn for_each_moved(&self, f: &mut dyn FnMut(Entity, Pose, Velocity)) {
        for (_, body) in self.world.active_bodies() {
            if !body.is_dynamic() {
                continue;
            }
            let Some(entity) = entity_from(body.user_data) else {
                continue;
            };
            let velocity = Velocity {
                linear: from_rp(body.linvel()),
                angular: from_rp(body.angvel()),
            };
            f(entity, pose_from_rp(body.position()), velocity);
        }
    }

    fn raycast(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
        exclude: Option<Entity>,
    ) -> Option<RayHit> {
        let direction = direction.try_normalize()?;
        let ray = rp::Ray::new(to_rp(origin), to_rp(direction));
        let mut filter = rp::QueryFilter::default().exclude_sensors();
        if let Some(handle) = exclude.and_then(|e| self.bodies.get(&e)) {
            filter = filter.exclude_rigid_body(*handle);
        }
        let (handle, hit) = self
            .world
            .cast_ray_and_get_normal(&ray, max_distance, true, filter)?;
        let entity = entity_from(self.world.colliders.get(handle)?.user_data)?;
        Some(RayHit {
            entity,
            distance: hit.time_of_impact,
            point: origin + direction * hit.time_of_impact,
            normal: from_rp(hit.normal),
        })
    }

    fn body_count(&self) -> usize {
        self.bodies.len()
    }
}

/// Сборщик событий столкновений за шаг. rapier вызывает обработчик через `&self` (возможно,
/// из нескольких потоков), поэтому накопление — под мьютексом; за шаг он берётся единицы раз.
#[derive(Default)]
struct Collector {
    events: Mutex<Vec<rp::CollisionEvent>>,
}

impl rp::EventHandler for Collector {
    fn handle_collision_event(
        &self,
        _bodies: &rp::RigidBodySet,
        _colliders: &rp::ColliderSet,
        event: rp::CollisionEvent,
        _contact_pair: Option<&rp::ContactPair>,
    ) {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(event);
    }

    fn handle_contact_force_event(
        &self,
        _dt: f32,
        _bodies: &rp::RigidBodySet,
        _colliders: &rp::ColliderSet,
        _contact_pair: &rp::ContactPair,
        _total_force_magnitude: f32,
    ) {
    }

    fn handle_soft_body_tear_event(
        &self,
        _soft_bodies: &rp::SoftBodySet,
        _event: &rp::SoftBodyTearEvent,
    ) {
    }
}
