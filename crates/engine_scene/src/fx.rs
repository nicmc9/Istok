//! Частицы и декали (L-02).
//!
//! Частицы считает CPU в `FixedUpdate`: на GLES3 compute нет, а отдельная симуляция
//! на GPU имеет смысл только на тысячах частиц. Уже выпущенные живут в мировых
//! координатах — сдвиг эмиттера их не тащит за собой.
//!
//! [`ParticleCloud`] — состояние проигрывания, в сцену не пишется (как `AnimationPlayer`).
//! Декаль не симулируется: её `Transform` и есть коробка (масштаб — полный размер).
//! Локальные XY — плоскость текстуры, Z — толщина проекции.

use engine_assets::Handle;
use engine_assets::formats::Image;
use engine_core::FixedTime;
use engine_ecs::prelude::*;
use engine_math::Vec3;

use crate::{GlobalTransform, Transform};

/// Рождает частицы. Направление — локальная ось +Y, тяжесть — вниз по миру (−Y).
#[derive(Component, Debug, Clone, Reflect)]
#[reflect(Component)]
#[require(Transform)]
pub struct ParticleEmitter {
    pub image: Handle<Image>,
    /// Штук в секунду.
    pub rate: f32,
    /// Сколько секунд живёт частица.
    pub lifetime: f32,
    /// Начальная скорость вдоль направления вылета.
    pub speed: f32,
    /// Полная ширина билборда в мировых единицах.
    pub size: f32,
    /// Линейный цвет, альфа — непрозрачность.
    pub color: [f32; 4],
    /// Ускорение вниз, мировые единицы в секунду². Ноль — частицы летят прямо.
    pub gravity: f32,
    /// Потолок живых частиц. Дальше новые не рождаются, пока старые не умрут.
    pub max: u32,
}

impl ParticleEmitter {
    pub fn new(image: Handle<Image>) -> Self {
        Self {
            image,
            rate: 30.0,
            lifetime: 1.0,
            speed: 2.0,
            size: 0.15,
            color: [1.0, 1.0, 1.0, 1.0],
            gravity: 4.0,
            max: 256,
        }
    }
}

/// Проекция текстуры на то, что уже нарисовано. Коробка — `Transform`:
/// масштаб по X и Y — размер пятна, по Z — насколько глубоко оно садится в поверхность.
#[derive(Component, Debug, Clone, Reflect)]
#[reflect(Component)]
#[require(Transform)]
pub struct Decal {
    pub image: Handle<Image>,
    /// Линейный цвет, альфа — непрозрачность.
    pub color: [f32; 4],
}

impl Decal {
    pub fn new(image: Handle<Image>) -> Self {
        Self {
            image,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

/// Одна живая частица. `previous` — позиция на прошлом шаге логики, для кадра между шагами.
#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub position: Vec3,
    pub previous: Vec3,
    pub velocity: Vec3,
    pub age: f32,
    pub lifetime: f32,
    pub size: f32,
    pub color: [f32; 4],
}

/// Живые частицы эмиттера. Появляется само, в сохранённую сцену не входит.
#[derive(Component, Debug, Clone)]
pub struct ParticleCloud {
    acc: f32,
    rng: u32,
    particles: Vec<Particle>,
}

/// Числа эмиттера без хэндла текстуры: шаг симуляции текстуру не читает.
#[derive(Clone, Copy)]
struct Emit {
    rate: f32,
    lifetime: f32,
    speed: f32,
    size: f32,
    color: [f32; 4],
    gravity: f32,
    max: u32,
}

impl Emit {
    fn from(emitter: &ParticleEmitter) -> Self {
        Self {
            rate: emitter.rate,
            lifetime: emitter.lifetime,
            speed: emitter.speed,
            size: emitter.size,
            color: emitter.color,
            gravity: emitter.gravity,
            max: emitter.max,
        }
    }
}

impl ParticleCloud {
    pub(crate) fn new() -> Self {
        Self {
            acc: 0.0,
            // Ноль заморозил бы xorshift: он так и остался бы нулём.
            rng: 0xA341_316C,
            particles: Vec::new(),
        }
    }

    pub fn live(&self) -> &[Particle] {
        &self.particles
    }

    pub(crate) fn step(&mut self, emitter: &ParticleEmitter, origin: Vec3, up: Vec3, dt: f32) {
        self.integrate(Emit::from(emitter), origin, up, dt);
    }

    fn integrate(&mut self, emit: Emit, origin: Vec3, up: Vec3, dt: f32) {
        if dt <= 0.0 {
            return;
        }
        let up = unit_axis(up);
        for particle in &mut self.particles {
            // Прошлая точка — до интегрирования, чтобы кадр между шагами не прыгал.
            particle.previous = particle.position;
            particle.velocity.y -= emit.gravity * dt;
            particle.position += particle.velocity * dt;
            particle.age += dt;
        }
        let lifetime = emit.lifetime;
        self.particles.retain(|particle| particle.age < lifetime);
        self.spawn(emit, origin, up, dt);
    }

    fn spawn(&mut self, emit: Emit, origin: Vec3, up: Vec3, dt: f32) {
        if emit.lifetime <= 0.0 || emit.rate <= 0.0 {
            return;
        }
        let max = usize::try_from(emit.max).unwrap_or(usize::MAX);
        self.acc += emit.rate * dt;
        while self.acc >= 1.0 && self.particles.len() < max {
            self.acc -= 1.0;
            let dir = self.cone(up);
            let velocity = dir * emit.speed;
            self.particles.push(Particle {
                position: origin,
                previous: origin,
                velocity,
                age: 0.0,
                lifetime: emit.lifetime,
                size: emit.size,
                color: emit.color,
            });
        }
        // Иначе при упоре в потолок дробная часть копится и выплёскивается пачкой.
        if self.particles.len() >= max {
            self.acc = self.acc.min(1.0);
        }
    }

    /// Направление в конусе вокруг `up`. Разброс небольшой, чтобы фонтан читался.
    fn cone(&mut self, up: Vec3) -> Vec3 {
        const SPREAD: f32 = 0.35;
        let mut side = up.cross(Vec3::X);
        if side.length_squared() < 1e-6 {
            side = up.cross(Vec3::Z);
        }
        let side = side.normalize_or_zero();
        let forward = up.cross(side);
        let dir = up + side * (self.signed() * SPREAD) + forward * (self.signed() * SPREAD);
        let dir = dir.normalize_or_zero();
        if dir.length_squared() < 1e-6 { up } else { dir }
    }

    fn signed(&mut self) -> f32 {
        self.unit() * 2.0 - 1.0
    }

    /// 16 старших бит xorshift → 0..1. Младшие отбрасываем: в f32 их всё равно нет.
    fn unit(&mut self) -> f32 {
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.rng = x;
        let bits = u16::try_from(x >> 16).unwrap_or(u16::MAX);
        f32::from(bits) / 65535.0
    }
}

fn unit_axis(up: Vec3) -> Vec3 {
    if up.length_squared() < 1e-8 {
        Vec3::Y
    } else {
        up.normalize()
    }
}

pub(crate) fn simulate(
    time: Res<FixedTime>,
    mut commands: Commands,
    missing: Query<Entity, (With<ParticleEmitter>, Without<ParticleCloud>)>,
    mut clouds: Query<(&ParticleEmitter, &GlobalTransform, &mut ParticleCloud)>,
) {
    let _zone = tracing::trace_span!("particles").entered();
    // Облако появляется в конце стадии, поэтому первый шаг эмиттера ещё ничего не считает.
    for entity in &missing {
        commands.entity(entity).insert(ParticleCloud::new());
    }
    let dt = time.step_secs();
    for (emitter, global, mut cloud) in &mut clouds {
        let up = unit_axis(global.0.y_axis.truncate());
        cloud.step(emitter, global.translation(), up, dt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn emit(rate: f32, lifetime: f32, speed: f32, gravity: f32, max: u32) -> Emit {
        Emit {
            rate,
            lifetime,
            speed,
            size: 0.1,
            color: [1.0, 1.0, 1.0, 1.0],
            gravity,
            max,
        }
    }

    #[test]
    fn falls_on_the_next_step_and_stays_in_world() {
        let mut cloud = ParticleCloud::new();
        let emit = emit(1.0, 10.0, 0.0, 4.0, 8);
        cloud.integrate(emit, Vec3::new(0.0, 2.0, 0.0), Vec3::Y, 1.0);
        assert_eq!(cloud.live().len(), 1);
        assert!((cloud.live()[0].position.y - 2.0).abs() < 1e-4);

        cloud.integrate(emit, Vec3::new(40.0, 2.0, 0.0), Vec3::Y, 1.0);
        let old: Vec<_> = cloud.live().iter().filter(|p| p.age > 0.5).collect();
        assert_eq!(old.len(), 1);
        // Скорость была 0, за шаг тяжесть дала −4 и сдвинула на −4. К новому месту эмиттера не прыгнула.
        assert!((old[0].position.y - (2.0 - 4.0)).abs() < 1e-3);
        assert!(old[0].position.x.abs() < 1e-3);
        assert!(
            cloud
                .live()
                .iter()
                .any(|p| (p.position.x - 40.0).abs() < 1e-3)
        );
    }

    #[test]
    fn dies_after_lifetime_and_respects_the_cap() {
        let mut cloud = ParticleCloud::new();
        let burst = emit(1.0, 1.2, 0.0, 0.0, 8);
        cloud.integrate(burst, Vec3::ZERO, Vec3::Y, 1.0);
        cloud.integrate(burst, Vec3::ZERO, Vec3::Y, 1.0);
        cloud.integrate(burst, Vec3::ZERO, Vec3::Y, 1.0);
        assert_eq!(cloud.live().len(), 2, "первый шаг уже старше жизни");
        assert!(cloud.live().iter().all(|p| p.age < 1.2));

        let mut capped = ParticleCloud::new();
        capped.integrate(emit(10.0, 10.0, 1.0, 0.0, 2), Vec3::ZERO, Vec3::Y, 1.0);
        assert_eq!(capped.live().len(), 2);
        capped.integrate(emit(10.0, 10.0, 1.0, 0.0, 2), Vec3::ZERO, Vec3::Y, 1.0);
        assert_eq!(capped.live().len(), 2);
    }

    #[test]
    fn floor_decal_is_flat() {
        use crate::Transform;

        let place = Vec3::new(0.55, 0.0, 0.15);
        let matrix = Transform::from_translation(place)
            .looking_at(place + Vec3::NEG_Y, Vec3::Z)
            .with_scale(Vec3::new(1.1, 1.1, 0.3))
            .to_matrix();
        let up = matrix.z_axis.truncate();
        assert!(
            up.x.abs() < 0.05 && (up.y - 0.3).abs() < 0.05 && up.z.abs() < 0.05,
            "локальный +Z должен смотреть вверх, получилось {up:?}"
        );
    }
    #[test]
    fn system_steps_on_the_fixed_tick() {
        use crate::ScenePlugin;
        use engine_assets::formats::{ColorSpace, Image};
        use engine_assets::{AssetPlugin, Assets};
        use engine_core::{App, Duration, FixedTime, Instant};

        let mut app = App::new();
        app.add_plugin(engine_net::NetPlugin)
            .add_plugin(AssetPlugin)
            .add_plugin(ScenePlugin);
        let image = app.world_mut().resource_mut::<Assets<Image>>().add(Image {
            width: 1,
            height: 1,
            color_space: ColorSpace::Srgb,
            data: vec![255, 255, 255, 255],
            mips: Vec::new(),
            basis: Vec::new(),
        });
        app.world_mut().spawn((
            ParticleEmitter::new(image),
            Transform::from_xyz(0.0, 2.0, 0.0),
        ));
        app.world_mut()
            .insert_resource(FixedTime::new(Duration::from_secs_f64(0.5), 4));
        let start = Instant::now();
        // Первый кадр без шага (delta = 0), второй только заводит облако, третий его считает.
        app.update_at(start);
        app.update_at(start + Duration::from_secs_f64(0.5));
        app.update_at(start + Duration::from_secs(1));
        let mut query = app.world_mut().query::<&ParticleCloud>();
        let count = query
            .iter(app.world())
            .map(|cloud| cloud.live().len())
            .sum::<usize>();
        assert!(count > 0);
        let near_emitter = query.iter(app.world()).all(|cloud| {
            cloud
                .live()
                .iter()
                .all(|p| (p.position.y - 2.0).abs() < 0.2)
        });
        assert!(near_emitter);
    }
}
