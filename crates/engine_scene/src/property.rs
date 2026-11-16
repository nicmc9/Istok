//! Анимация свойств (R2-06): кривые пишут в поля отражённых компонентов.
//!
//! Имя компонента — короткое (`PointLight`) или полный путь типа. Поле — путь рефлексии:
//! `intensity`, `translation`, `translation.x`. Тип кривой должен совпадать с типом поля
//! (`Vec3` и `[f32; 3]` — разные кривые).

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock, PoisonError};

use engine_assets::{Asset, Assets, Handle};
use engine_core::FixedTime;
use engine_ecs::prelude::bevy_ecs::prelude::Mut;
use engine_ecs::prelude::bevy_reflect::GetPath;
use engine_ecs::prelude::*;
use engine_ecs::reflect::{AppTypeRegistry, PartialReflect, ReflectComponent};
use engine_math::{Quat, Vec2, Vec3, Vec4};

use crate::animation::advance;

/// Как переходить между ключами. Кубический сплайн здесь не нужен: он есть у клипов glTF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tween {
    /// Держать значение до следующего ключа.
    Step,
    /// Линейно. Повороты — по дуге (slerp).
    Linear,
}

/// Ключи одного свойства. Время — секунды, по возрастанию.
#[derive(Debug, Clone)]
pub struct Keys<T> {
    pub times: Vec<f32>,
    pub values: Vec<T>,
    pub tween: Tween,
}

impl<T> Keys<T> {
    pub fn linear(times: Vec<f32>, values: Vec<T>) -> Self {
        Self {
            times,
            values,
            tween: Tween::Linear,
        }
    }

    pub fn step(times: Vec<f32>, values: Vec<T>) -> Self {
        Self {
            times,
            values,
            tween: Tween::Step,
        }
    }

    fn duration(&self) -> f32 {
        let count = self.times.len().min(self.values.len());
        if count == 0 {
            0.0
        } else {
            self.times[count - 1]
        }
    }
}

fn sample_keys<T: Blend>(keys: &Keys<T>, time: f32) -> Option<T> {
    let count = keys.times.len().min(keys.values.len());
    if count == 0 {
        return None;
    }
    if count == 1 || time <= keys.times[0] {
        return Some(keys.values[0]);
    }
    let last = count - 1;
    if time >= keys.times[last] {
        return Some(keys.values[last]);
    }
    let next = keys.times[..count].partition_point(|&stamp| stamp <= time);
    let index = next - 1;
    if keys.tween == Tween::Step {
        return Some(keys.values[index]);
    }
    let span = keys.times[index + 1] - keys.times[index];
    let fraction = if span > 0.0 {
        (time - keys.times[index]) / span
    } else {
        0.0
    };
    Some(keys.values[index].blend(keys.values[index + 1], fraction))
}

trait Blend: Copy {
    fn blend(self, other: Self, fraction: f32) -> Self;
}

impl Blend for f32 {
    fn blend(self, other: Self, fraction: f32) -> Self {
        self + (other - self) * fraction
    }
}

impl Blend for Vec2 {
    fn blend(self, other: Self, fraction: f32) -> Self {
        self.lerp(other, fraction)
    }
}

impl Blend for Vec3 {
    fn blend(self, other: Self, fraction: f32) -> Self {
        self.lerp(other, fraction)
    }
}

impl Blend for Vec4 {
    fn blend(self, other: Self, fraction: f32) -> Self {
        self.lerp(other, fraction)
    }
}

impl Blend for Quat {
    fn blend(self, other: Self, fraction: f32) -> Self {
        self.slerp(other, fraction)
    }
}

impl Blend for [f32; 3] {
    fn blend(self, other: Self, fraction: f32) -> Self {
        [
            self[0].blend(other[0], fraction),
            self[1].blend(other[1], fraction),
            self[2].blend(other[2], fraction),
        ]
    }
}

impl Blend for [f32; 4] {
    fn blend(self, other: Self, fraction: f32) -> Self {
        [
            self[0].blend(other[0], fraction),
            self[1].blend(other[1], fraction),
            self[2].blend(other[2], fraction),
            self[3].blend(other[3], fraction),
        ]
    }
}

/// Кривая под тип поля. `Float3` — это `[f32; 3]` (цвет света), не [`Vec3`].
#[derive(Debug, Clone)]
pub enum Curve {
    Float(Keys<f32>),
    Vec2(Keys<Vec2>),
    Vec3(Keys<Vec3>),
    Vec4(Keys<Vec4>),
    Quat(Keys<Quat>),
    Float3(Keys<[f32; 3]>),
    Float4(Keys<[f32; 4]>),
}

impl Curve {
    fn sample(&self, time: f32) -> Option<Sampled> {
        Some(match self {
            Self::Float(keys) => Sampled::Float(sample_keys(keys, time)?),
            Self::Vec2(keys) => Sampled::Vec2(sample_keys(keys, time)?),
            Self::Vec3(keys) => Sampled::Vec3(sample_keys(keys, time)?),
            Self::Vec4(keys) => Sampled::Vec4(sample_keys(keys, time)?),
            Self::Quat(keys) => Sampled::Quat(sample_keys(keys, time)?),
            Self::Float3(keys) => Sampled::Float3(sample_keys(keys, time)?),
            Self::Float4(keys) => Sampled::Float4(sample_keys(keys, time)?),
        })
    }

    fn duration(&self) -> f32 {
        match self {
            Self::Float(keys) => keys.duration(),
            Self::Vec2(keys) => keys.duration(),
            Self::Vec3(keys) => keys.duration(),
            Self::Vec4(keys) => keys.duration(),
            Self::Quat(keys) => keys.duration(),
            Self::Float3(keys) => keys.duration(),
            Self::Float4(keys) => keys.duration(),
        }
    }

    fn bytes(&self) -> u64 {
        match self {
            Self::Float(keys) => bytes_of(&keys.times) + bytes_of(&keys.values),
            Self::Vec2(keys) => bytes_of(&keys.times) + bytes_of(&keys.values),
            Self::Vec3(keys) => bytes_of(&keys.times) + bytes_of(&keys.values),
            Self::Vec4(keys) => bytes_of(&keys.times) + bytes_of(&keys.values),
            Self::Quat(keys) => bytes_of(&keys.times) + bytes_of(&keys.values),
            Self::Float3(keys) => bytes_of(&keys.times) + bytes_of(&keys.values),
            Self::Float4(keys) => bytes_of(&keys.times) + bytes_of(&keys.values),
        }
    }
}

enum Sampled {
    Float(f32),
    Vec2(Vec2),
    Vec3(Vec3),
    Vec4(Vec4),
    Quat(Quat),
    Float3([f32; 3]),
    Float4([f32; 4]),
}

impl Sampled {
    fn apply_to(&self, slot: &mut dyn PartialReflect) -> bool {
        let result = match self {
            Self::Float(value) => slot.try_apply(value),
            Self::Vec2(value) => slot.try_apply(value),
            Self::Vec3(value) => slot.try_apply(value),
            Self::Vec4(value) => slot.try_apply(value),
            Self::Quat(value) => slot.try_apply(value),
            Self::Float3(value) => slot.try_apply(value),
            Self::Float4(value) => slot.try_apply(value),
        };
        result.is_ok()
    }
}

/// Одно свойство одного компонента.
#[derive(Debug, Clone)]
pub struct PropertyTrack {
    pub component: String,
    pub field: String,
    pub curve: Curve,
}

impl PropertyTrack {
    pub fn new(component: impl Into<String>, field: impl Into<String>, curve: Curve) -> Self {
        Self {
            component: component.into(),
            field: field.into(),
            curve,
        }
    }
}

/// Набор кривых. Длительность — время последнего ключа.
#[derive(Debug, Clone, TypePath)]
pub struct PropertyClip {
    pub name: String,
    pub tracks: Vec<PropertyTrack>,
}

impl PropertyClip {
    pub fn new(name: impl Into<String>, tracks: Vec<PropertyTrack>) -> Self {
        Self {
            name: name.into(),
            tracks,
        }
    }

    pub fn duration(&self) -> f32 {
        self.tracks
            .iter()
            .map(|track| track.curve.duration())
            .fold(0.0, f32::max)
    }
}

impl Asset for PropertyClip {
    fn memory_size(&self) -> u64 {
        let tracks = self.tracks.iter().fold(0, |sum, track| {
            sum + track.component.len() + track.field.len()
        });
        u64::try_from(self.name.len() + tracks).unwrap_or(u64::MAX)
            + self
                .tracks
                .iter()
                .map(|track| track.curve.bytes())
                .sum::<u64>()
    }
}

/// Пишет кривые клипа в компоненты этой сущности. В сцену не пишется — состояние проигрывания.
#[derive(Component, Debug, Clone)]
pub struct PropertyPlayer {
    clip: Handle<PropertyClip>,
    time: f32,
    pub speed: f32,
    pub paused: bool,
    pub looping: bool,
}

impl PropertyPlayer {
    pub fn new(clip: Handle<PropertyClip>) -> Self {
        Self {
            clip,
            time: 0.0,
            speed: 1.0,
            paused: false,
            looping: true,
        }
    }

    pub fn play(&mut self, clip: Handle<PropertyClip>) -> &mut Self {
        self.clip = clip;
        self.time = 0.0;
        self
    }

    pub fn time(&self) -> f32 {
        self.time
    }
}

struct Write {
    entity: Entity,
    component: String,
    field: String,
    value: Sampled,
}

/// Кривые пишут в любые компоненты, поэтому системе нужен весь мир, а не один запрос.
pub(crate) fn animate_properties(world: &mut World) {
    let _zone = tracing::trace_span!("animate_properties").entered();
    let step = world.resource::<FixedTime>().step_secs();
    let mut writes = Vec::new();
    // Клип забираем из мира на время прохода: иначе нельзя одновременно менять проигрыватель
    world.resource_scope(|world, clips: Mut<Assets<PropertyClip>>| {
        let mut players = world.query::<(Entity, &mut PropertyPlayer)>();
        for (entity, mut player) in players.iter_mut(world) {
            let Some(clip) = clips.get_loaded(&player.clip) else {
                continue;
            };
            if !player.paused {
                let dt = step * player.speed;
                player.time = advance(player.time, dt, clip.duration(), player.looping);
            }
            let time = player.time;
            for track in &clip.tracks {
                let Some(value) = track.curve.sample(time) else {
                    continue;
                };
                writes.push(Write {
                    entity,
                    component: track.component.clone(),
                    field: track.field.clone(),
                    value,
                });
            }
        }
    });
    apply_writes(world, writes);
}

fn apply_writes(world: &mut World, writes: Vec<Write>) {
    let registry = world.resource::<AppTypeRegistry>().0.clone();
    let registry = registry.read();
    for write in writes {
        let Some(registration) = registry
            .get_with_type_path(&write.component)
            .or_else(|| registry.get_with_short_type_path(&write.component))
        else {
            warn_once(
                &write.component,
                &format!("Анимация: нет типа «{}»", write.component),
            );
            continue;
        };
        let Some(component) = registration.data::<ReflectComponent>() else {
            warn_once(
                &write.component,
                &format!("«{}» не компонент", write.component),
            );
            continue;
        };
        let Ok(entity) = world.get_entity_mut(write.entity) else {
            continue;
        };
        let Some(mut reflected) = component.reflect_mut(entity) else {
            warn_once(
                &format!("missing:{}", write.component),
                &format!("Анимация: у сущности нет компонента «{}»", write.component),
            );
            continue;
        };
        let path = write.field.as_str();
        match reflected.reflect_path_mut(path) {
            Ok(slot) => {
                if !write.value.apply_to(slot) {
                    warn_once(
                        &format!("{}.{}:type", write.component, write.field),
                        &format!(
                            "Анимация: тип кривой не совпал с {}.{}",
                            write.component, write.field
                        ),
                    );
                }
            }
            Err(_) => warn_once(
                &format!("{}.{}", write.component, write.field),
                &format!("Анимация: нет поля {}.{}", write.component, write.field),
            ),
        }
    }
}

fn warn_once(key: &str, message: &str) {
    static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    let mut seen = SEEN
        .get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .unwrap_or_else(PoisonError::into_inner);
    if seen.insert(key.to_owned()) {
        tracing::warn!("{message}");
    }
}

fn bytes_of<T>(items: &[T]) -> u64 {
    u64::try_from(std::mem::size_of_val(items)).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_step_and_ends() {
        let linear = Curve::Float(Keys::linear(vec![0.0, 1.0], vec![0.0, 10.0]));
        let at = |time| match linear.sample(time) {
            Some(Sampled::Float(value)) => value,
            _ => panic!("нет значения"),
        };
        assert!((at(0.5) - 5.0).abs() < 1e-4);
        assert!((at(-1.0) - 0.0).abs() < 1e-4);
        assert!((at(3.0) - 10.0).abs() < 1e-4);
        let step = Curve::Float(Keys::step(vec![0.0, 1.0], vec![1.0, 9.0]));
        let at = |time| match step.sample(time) {
            Some(Sampled::Float(value)) => value,
            _ => panic!("нет значения"),
        };
        assert!((at(0.9) - 1.0).abs() < 1e-4);
        assert!((at(1.0) - 9.0).abs() < 1e-4);
    }

    #[test]
    fn quaternion_uses_slerp() {
        let end = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let keys = Curve::Quat(Keys::linear(vec![0.0, 1.0], vec![Quat::IDENTITY, end]));
        let Some(Sampled::Quat(mid)) = keys.sample(0.5) else {
            panic!("нет значения");
        };
        let expect = Quat::IDENTITY.slerp(end, 0.5);
        assert!(mid.dot(expect).abs() > 0.999);
    }

    #[test]
    fn writes_reflected_fields_and_skips_unknown() {
        use crate::{PointLight, ScenePlugin, Transform};
        use engine_assets::AssetPlugin;
        use engine_core::{App, Duration, FixedTime, Instant};

        let mut app = App::new();
        app.add_plugin(engine_net::NetPlugin)
            .add_plugin(AssetPlugin)
            .add_plugin(ScenePlugin);
        let clip = app
            .world_mut()
            .resource_mut::<Assets<PropertyClip>>()
            .add(PropertyClip::new(
                "move",
                vec![
                    PropertyTrack::new(
                        "Transform",
                        "no_such_field",
                        Curve::Float(Keys::linear(vec![0.0, 1.0], vec![0.0, 1.0])),
                    ),
                    PropertyTrack::new(
                        "Transform",
                        "translation",
                        Curve::Vec3(Keys::linear(
                            vec![0.0, 1.0],
                            vec![Vec3::ZERO, Vec3::X * 10.0],
                        )),
                    ),
                    PropertyTrack::new(
                        "PointLight",
                        "intensity",
                        Curve::Float(Keys::step(vec![0.0, 0.8], vec![1.0, 9.0])),
                    ),
                    PropertyTrack::new(
                        "PointLight",
                        "color",
                        Curve::Float3(Keys::linear(
                            vec![0.0, 1.0],
                            vec![[0.0; 3], [0.0, 0.0, 1.0]],
                        )),
                    ),
                ],
            ));
        let entity = app
            .world_mut()
            .spawn((
                Transform::default(),
                PointLight::default(),
                PropertyPlayer::new(clip),
            ))
            .id();
        app.world_mut()
            .insert_resource(FixedTime::new(Duration::from_secs_f64(0.5), 4));
        let start = Instant::now();
        app.update_at(start);
        app.update_at(start + Duration::from_secs_f64(0.5));

        let world = app.world();
        let translation = world.get::<Transform>(entity).unwrap().translation;
        assert!((translation.x - 5.0).abs() < 1e-3, "{}", translation.x);
        let light = world.get::<PointLight>(entity).unwrap();
        assert!(
            (light.intensity - 1.0).abs() < 1e-3,
            "шаг ещё держит первый ключ"
        );
        assert!((light.color[2] - 0.5).abs() < 1e-3);
    }
}
