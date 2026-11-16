//! Физика через App: настоящие системы фасада, бэкенд rapier, фиксированный шаг.

use engine_core::{App, CVarSource, CVars, FixedTime};
use engine_ecs::prelude::*;
use engine_math::Vec3;
use engine_physics::{Collider, CollisionEvent, Impulse, Physics, RigidBody, Velocity};
use engine_physics_rapier::RapierPhysicsPlugin;
use engine_platform::Instant;
use engine_scene::Transform;

/// События столкновений, накопленные за все кадры теста.
#[derive(Resource, Default)]
struct Collisions(Vec<CollisionEvent>);

fn collect(mut reader: MessageReader<CollisionEvent>, mut all: ResMut<Collisions>) {
    all.0.extend(reader.read().copied());
}

struct Sim {
    app: App,
    now: Instant,
}

impl Sim {
    fn new() -> Self {
        let mut app = App::new();
        // Плагин читает CVar при добавлении: без этого мир не создаётся
        app.world_mut()
            .resource_mut::<CVars>()
            .set("physics.enabled", "true", CVarSource::LaunchArguments)
            .unwrap();
        app.add_plugin(RapierPhysicsPlugin)
            .init_resource::<Collisions>()
            .add_systems(engine_core::Stage::Update, collect);
        let now = Instant::now();
        app.update_at(now); // первый кадр: запуск, шагов нет
        Self { app, now }
    }

    /// Продвинуть время на `seconds` кадрами по одному шагу.
    fn run(&mut self, seconds: f32) {
        let step = self.app.world().resource::<FixedTime>().step();
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // секунды теста
        let steps = (seconds / step.as_secs_f32()).ceil() as u32;
        for _ in 0..steps {
            self.now += step;
            self.app.update_at(self.now);
        }
    }

    fn world(&mut self) -> &mut World {
        self.app.world_mut()
    }

    fn ground(&mut self) -> Entity {
        self.world()
            .spawn((
                RigidBody::Fixed,
                Collider::cuboid(Vec3::new(10.0, 0.5, 10.0)),
                Transform::from_xyz(0.0, -0.5, 0.0),
            ))
            .id()
    }

    fn ball(&mut self, height: f32) -> Entity {
        self.world()
            .spawn((
                RigidBody::Dynamic,
                Collider::ball(0.5),
                Transform::from_xyz(0.0, height, 0.0),
            ))
            .id()
    }

    fn position(&mut self, entity: Entity) -> Vec3 {
        self.world().get::<Transform>(entity).unwrap().translation
    }
}

#[test]
fn ball_falls_and_rests_on_ground() {
    let mut sim = Sim::new();
    let ground = sim.ground();
    let ball = sim.ball(3.0);
    sim.run(0.3);
    let falling = sim.position(ball).y;
    assert!(falling < 3.0 && falling > 0.5, "падает: {falling}");
    sim.run(3.0);
    let rest = sim.position(ball).y;
    assert!(
        (rest - 0.5).abs() < 0.02,
        "лежит на земле (центр на радиусе): {rest}"
    );

    let collisions = &sim.world().resource::<Collisions>().0;
    assert!(
        collisions
            .iter()
            .any(|e| matches!(e, CollisionEvent::Started { .. })
                && [e.entities().0, e.entities().1].contains(&ball)
                && [e.entities().0, e.entities().1].contains(&ground)),
        "событие касания шара и земли: {collisions:?}"
    );
}

#[test]
fn raycast_hits_ball_and_reports_normal() {
    let mut sim = Sim::new();
    sim.ground();
    let ball = sim.ball(2.0);
    sim.world().resource_mut::<engine_physics::Gravity>().0 = Vec3::ZERO;
    sim.run(0.05);
    let physics = sim.world().resource::<Physics>();
    let hit = physics
        .raycast(Vec3::new(-5.0, 2.0, 0.0), Vec3::X * 3.0, 100.0)
        .expect("луч попадает в шар");
    assert_eq!(hit.entity, ball);
    assert!((hit.distance - 4.5).abs() < 1e-3, "{}", hit.distance);
    assert!((hit.normal - Vec3::NEG_X).length() < 1e-3);
    // Тот же луч, исключая шар, — мимо (земля ниже)
    assert!(
        physics
            .raycast_excluding(Vec3::new(-5.0, 2.0, 0.0), Vec3::X, 100.0, ball)
            .is_none()
    );
}

#[test]
fn code_changes_reach_physics() {
    let mut sim = Sim::new();
    sim.ground();
    let ball = sim.ball(0.5);
    sim.run(0.1);

    // Телепорт: новый Transform из кода
    sim.world().get_mut::<Transform>(ball).unwrap().translation = Vec3::new(4.0, 5.0, 0.0);
    sim.run(0.05);
    let p = sim.position(ball);
    assert!((p.x - 4.0).abs() < 1e-3 && p.y > 4.5, "телепорт: {p}");

    // Скорость из кода
    sim.world().get_mut::<Velocity>(ball).unwrap().linear = Vec3::new(0.0, 0.0, 3.0);
    sim.run(0.1);
    assert!(
        sim.position(ball).z > 0.2,
        "скорость: {}",
        sim.position(ball)
    );

    // Толчок: применяется один раз и обнуляется
    sim.world().entity_mut(ball).insert(Impulse {
        linear: Vec3::new(-2.0, 0.0, 0.0),
        angular: Vec3::ZERO,
    });
    sim.run(0.05);
    assert_eq!(
        *sim.world().get::<Impulse>(ball).unwrap(),
        Impulse::default()
    );
    assert!(sim.world().get::<Velocity>(ball).unwrap().linear.x < 0.0);

    // Удаление сущности убирает тело
    sim.world().despawn(ball);
    sim.run(0.05);
    assert_eq!(sim.world().resource::<Physics>().body_count(), 1);
}

#[test]
fn kinematic_platform_pushes_ball() {
    let mut sim = Sim::new();
    sim.ground();
    let platform = sim
        .world()
        .spawn((
            RigidBody::Kinematic,
            Collider::cuboid(Vec3::new(0.2, 1.0, 1.0)),
            Transform::from_xyz(-2.0, 1.0, 0.0),
        ))
        .id();
    let ball = sim.ball(0.5);
    // Платформа едет вправо, шар стоит на пути
    for _ in 0..90 {
        sim.world()
            .get_mut::<Transform>(platform)
            .unwrap()
            .translation
            .x += 0.05;
        sim.run(1.0 / 60.0);
    }
    assert!(
        sim.position(ball).x > 1.5,
        "шар вытолкнут: {}",
        sim.position(ball)
    );
}
