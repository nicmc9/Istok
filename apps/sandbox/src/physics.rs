//! Физика в витрине: пирамида из кубов, мячи, команда консоли `drop`.

use engine::assets::formats::{Material, Mesh};
use engine::math::Vec3;
use engine::physics::Damping;
use engine::prelude::*;

/// Мяч — по этой метке столкновения превращаются в звук удара.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Default)]
pub struct Ball;

/// Звук удара мяча.
#[derive(Resource)]
pub struct BounceSound(Handle<AudioClip>);

/// Меши и материалы демо. Созданы кодом, но с именами (`sandbox://...`): так они попадают
/// в сохранённые сцены и находятся при загрузке.
#[derive(Resource)]
pub struct PhysicsDemo {
    pub floor_mesh: Handle<Mesh>,
    pub floor_material: Handle<Material>,
    cube_mesh: Handle<Mesh>,
    cube_material: Handle<Material>,
    ball_mesh: Handle<Mesh>,
    ball_materials: Vec<Handle<Material>>,
    /// Счётчик брошенных мячей — для разброса цвета и места.
    dropped: usize,
}

/// Размер пола, м.
pub const FLOOR_SIZE: f32 = 30.0;

const BALL_RADIUS: f32 = 0.18;
const CUBE_HALF: f32 = 0.2;
/// Где стоит пирамида: за моделями витрины.
const PYRAMID: Vec3 = Vec3::new(0.0, 0.0, -2.6);

fn material(
    materials: &mut Assets<Material>,
    name: &str,
    color: [f32; 3],
    roughness: f32,
) -> Handle<Material> {
    materials.add_named(
        &format!("sandbox://materials/{name}"),
        Material {
            name: name.to_owned(),
            base_color: [color[0], color[1], color[2], 1.0],
            metallic: 0.0,
            roughness,
            ..Material::default()
        },
    )
}

/// Ресурсы демо (меши, материалы, звук удара) — при любой сцене, в том числе загруженной
/// из файла: сохранённая сцена ссылается на них по именам.
pub fn create_assets(
    meshes: &mut Assets<Mesh>,
    materials: &mut Assets<Material>,
    server: &AssetServer,
) -> (PhysicsDemo, Option<BounceSound>) {
    let demo = PhysicsDemo {
        floor_mesh: meshes.add_named("sandbox://meshes/floor", Mesh::plane(FLOOR_SIZE)),
        floor_material: material(materials, "floor", [0.25, 0.25, 0.27], 0.85),
        cube_mesh: meshes.add_named(
            "sandbox://meshes/cube",
            Mesh::cuboid(Vec3::splat(CUBE_HALF)),
        ),
        cube_material: material(materials, "crate", [0.75, 0.55, 0.3], 0.5),
        ball_mesh: meshes.add_named("sandbox://meshes/ball", Mesh::sphere(BALL_RADIUS, 24, 12)),
        ball_materials: vec![
            material(materials, "ball_red", [0.85, 0.2, 0.2], 0.5),
            material(materials, "ball_blue", [0.2, 0.4, 0.9], 0.5),
            material(materials, "ball_green", [0.25, 0.75, 0.3], 0.5),
        ],
        dropped: 0,
    };
    // Без audio.enabled тип AudioClip не зарегистрирован: загрузка дала бы ошибку в логе.
    let bounce = engine::audio::AUDIO_ENABLED
        .get()
        .then(|| BounceSound(server.load("audio/bounce.wav")));
    (demo, bounce)
}

/// Пирамида 4 × 4 → 1 из кубов и три мяча над ней.
pub fn spawn_pyramid(commands: &mut Commands, demo: &mut PhysicsDemo) {
    let size = CUBE_HALF * 2.0;
    for level in 0..4u8 {
        let count = 4 - level;
        let offset = (f32::from(count) - 1.0) * size * 0.5;
        for x in 0..count {
            for z in 0..count {
                let position = PYRAMID
                    + Vec3::new(
                        f32::from(x) * size - offset,
                        CUBE_HALF + f32::from(level) * size,
                        f32::from(z) * size - offset,
                    );
                commands.spawn((
                    Mesh3d {
                        mesh: demo.cube_mesh.clone(),
                        materials: vec![Some(demo.cube_material.clone())],
                    },
                    RigidBody::Dynamic,
                    Collider::cuboid(Vec3::splat(CUBE_HALF)).with_friction(0.7),
                    Transform::from_translation(position),
                ));
            }
        }
    }
    for layer in 0..3 {
        let bundle = demo.next_ball(layer);
        commands.spawn(bundle);
    }
}

impl PhysicsDemo {
    /// Мяч над пирамидой; место и цвет немного меняются от мяча к мячу. `layer` — номер
    /// в пачке: мячи одной пачки встают столбиком. Перекрывающиеся тела решатель расталкивает
    /// с огромной скоростью — мячи разлетелись бы по всей сцене.
    fn next_ball(&mut self, layer: usize) -> impl Bundle + use<> {
        let i = self.dropped;
        self.dropped += 1;
        #[allow(clippy::cast_precision_loss)]
        let angle = i as f32 * 2.4; // «золотой угол» — мячи не падают в одну точку
        #[allow(clippy::cast_precision_loss)] // не больше 200 мячей в пачке
        let height = 3.0 + layer as f32 * (BALL_RADIUS * 2.0 + 0.05);
        let position = PYRAMID + Vec3::new(angle.cos() * 0.3, height, angle.sin() * 0.3);
        (
            Mesh3d {
                mesh: self.ball_mesh.clone(),
                materials: vec![Some(
                    self.ball_materials[i % self.ball_materials.len()].clone(),
                )],
            },
            Ball,
            RigidBody::Dynamic,
            Collider::ball(BALL_RADIUS)
                .with_restitution(0.6)
                .with_density(4.0),
            // Сопротивление качению в rapier не моделируется — без затухания мяч катится вечно
            Damping {
                linear: 0.3,
                angular: 1.0,
            },
            Transform::from_translation(position),
        )
    }
}

/// Удар мяча — разовый 3D-звук в точке удара. Громкость — от скорости, высота тона чуть
/// меняется от удара к удару, чтобы одинаковые удары не звучали «как пулемёт».
pub fn bounce_sounds(
    mut commands: Commands,
    mut events: MessageReader<CollisionEvent>,
    balls: Query<(&Transform, &Velocity), With<Ball>>,
    sound: Option<Res<BounceSound>>,
    mut counter: Local<u32>,
) {
    let Some(sound) = sound else { return };
    // Не больше нескольких звуков за кадр: при обвале пирамиды ударов десятки
    let mut budget = 6;
    for event in events.read() {
        let CollisionEvent::Started {
            a,
            b,
            sensor: false,
        } = *event
        else {
            continue;
        };
        for entity in [a, b] {
            let Ok((transform, velocity)) = balls.get(entity) else {
                continue;
            };
            let speed = velocity.linear.length();
            if speed < 0.8 || budget == 0 {
                continue;
            }
            budget -= 1;
            *counter = counter.wrapping_add(1);
            #[allow(clippy::cast_precision_loss)] // остаток от деления на 7
            let pitch = 0.9 + (*counter % 7) as f32 * 0.035;
            commands.spawn((
                AudioPlayer::once(sound.0.clone())
                    .spatial()
                    .with_volume((speed / 6.0).min(1.0))
                    .with_speed(pitch),
                Transform::from_translation(transform.translation),
            ));
        }
    }
}

/// Тела, упавшие с края пола, удаляются: иначе они падают вечно и занимают физику.
pub fn despawn_fallen(
    mut commands: Commands,
    bodies: Query<(Entity, &Transform), With<RigidBody>>,
) {
    for (entity, transform) in &bodies {
        if transform.translation.y < -30.0 {
            commands.entity(entity).despawn();
        }
    }
}

/// Команда консоли: `drop [n]` — бросить мячи на пирамиду.
pub fn drop_command() -> ConsoleCommand {
    ConsoleCommand::new(
        "drop",
        "Бросить мячи на пирамиду (физика)",
        |world, args| {
            let count: usize = match args {
                [] => 1,
                [n] => n.parse().map_err(|_| format!("«{n}» — не число"))?,
                _ => return Err("использование: drop [количество]".to_owned()),
            };
            if !world.contains_resource::<Physics>() {
                return Err("физика выключена (physics.enabled)".to_owned());
            }
            if !world.contains_resource::<PhysicsDemo>() {
                return Err("физика есть только в сцене showcase".to_owned());
            }
            let count = count.min(200);
            for layer in 0..count {
                // [Rust] Ресурс и `spawn` оба заимствуют мир — по очереди, не одновременно
                let ball = world.resource_mut::<PhysicsDemo>().next_ball(layer);
                world.spawn(ball);
            }
            Ok(format!("брошено мячей: {count}"))
        },
    )
    .with_usage("drop [n]")
}
