//! Компоненты физики. Тело = `RigidBody` + `Collider` на сущности с `Transform`.

use engine_assets::formats::Mesh;
use engine_ecs::prelude::*;
use engine_math::{Quat, Vec3};
use engine_scene::Transform;

/// Вид тела.
///
/// ```ignore
/// commands.spawn((
///     RigidBody::Dynamic,
///     Collider::ball(0.5).with_restitution(0.6),
///     Transform::from_xyz(0.0, 5.0, 0.0),
/// ));
/// ```
///
/// Тело должно быть корневой сущностью (без `ChildOf`): физика пишет `Transform`, а у дочерних
/// он локальный.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Default, Reflect)]
#[reflect(Component, Default)]
#[require(Transform, Velocity)]
pub enum RigidBody {
    /// Движется под действием сил и столкновений; `Transform` пишет физика.
    #[default]
    Dynamic,
    /// Неподвижно: пол, стены. Передвинуть можно — изменением `Transform` (телепорт).
    Fixed,
    /// Движется только кодом (изменением `Transform`), расталкивает динамические тела, само
    /// от них не отлетает: платформы, двери.
    Kinematic,
}

/// Форма коллайдера в локальных координатах тела.
#[derive(Debug, Clone, PartialEq, Reflect)]
pub enum ColliderShape {
    Cuboid {
        half_extents: Vec3,
    },
    Ball {
        radius: f32,
    },
    /// Вдоль оси Y: цилиндр высотой `2 × half_height` и полусферы радиуса `radius` на концах.
    Capsule {
        half_height: f32,
        radius: f32,
    },
    /// Треугольники меша. Только для неподвижных и кинематических тел: у динамических
    /// у сетки нет объёма — масса и столкновения считаются плохо.
    /// Вершины и индексы сохраняются в сцене вместе с коллайдером.
    TriMesh {
        vertices: Vec<Vec3>,
        indices: Vec<[u32; 3]>,
    },
}

impl ColliderShape {
    /// Форма с учётом масштаба `Transform`. Шар и капсула остаются круглыми: берётся
    /// наибольший масштаб по осям сечения.
    #[must_use]
    pub fn scaled(&self, scale: Vec3) -> Self {
        let scale = scale.abs();
        match self {
            Self::Cuboid { half_extents } => Self::Cuboid {
                half_extents: *half_extents * scale,
            },
            Self::Ball { radius } => Self::Ball {
                radius: radius * scale.max_element(),
            },
            Self::Capsule {
                half_height,
                radius,
            } => Self::Capsule {
                half_height: half_height * scale.y,
                radius: radius * scale.x.max(scale.z),
            },
            Self::TriMesh { vertices, indices } => Self::TriMesh {
                vertices: vertices.iter().map(|v| *v * scale).collect(),
                indices: indices.clone(),
            },
        }
    }
}

/// Коллайдер: форма и свойства поверхности.
#[derive(Component, Debug, Clone, PartialEq, Reflect)]
#[reflect(Component)]
pub struct Collider {
    pub shape: ColliderShape,
    /// Трение: 0 — лёд, 1 — резина.
    pub friction: f32,
    /// Упругость: 0 — не отскакивает, 1 — отскакивает на ту же высоту.
    pub restitution: f32,
    /// Плотность, кг/м³ условных единиц: масса = объём × плотность.
    pub density: f32,
    /// Датчик: не сталкивается, только сообщает о пересечениях (`CollisionEvent`).
    pub sensor: bool,
}

impl Collider {
    fn new(shape: ColliderShape) -> Self {
        Self {
            shape,
            friction: 0.5,
            restitution: 0.0,
            density: 1.0,
            sensor: false,
        }
    }

    /// Параллелепипед с половинами размеров `half_extents`.
    pub fn cuboid(half_extents: Vec3) -> Self {
        Self::new(ColliderShape::Cuboid { half_extents })
    }

    pub fn ball(radius: f32) -> Self {
        Self::new(ColliderShape::Ball { radius })
    }

    /// Капсула вдоль Y — обычная форма персонажа.
    pub fn capsule(half_height: f32, radius: f32) -> Self {
        Self::new(ColliderShape::Capsule {
            half_height,
            radius,
        })
    }

    /// Треугольники всех примитивов меша (уровень, рельеф). Только для `Fixed`/`Kinematic`.
    pub fn trimesh(mesh: &Mesh) -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for primitive in &mesh.primitives {
            let base = u32::try_from(vertices.len()).unwrap_or(u32::MAX);
            vertices.extend(primitive.positions.iter().map(|p| Vec3::from_array(*p)));
            indices.extend(
                primitive
                    .indices
                    // [Rust] `as_chunks::<3>` — срез как массив троек (остаток отдельно)
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|[a, b, c]| [base + a, base + b, base + c]),
            );
        }
        Self::new(ColliderShape::TriMesh { vertices, indices })
    }

    #[must_use]
    pub fn with_friction(mut self, friction: f32) -> Self {
        self.friction = friction;
        self
    }

    #[must_use]
    pub fn with_restitution(mut self, restitution: f32) -> Self {
        self.restitution = restitution;
        self
    }

    #[must_use]
    pub fn with_density(mut self, density: f32) -> Self {
        self.density = density;
        self
    }

    #[must_use]
    pub fn sensor(mut self) -> Self {
        self.sensor = true;
        self
    }
}

/// Скорость тела: линейная (м/с) и угловая (рад/с, вектор оси × скорость). Физика
/// обновляет её каждый шаг; запись из кода задаёт скорость телу.
#[derive(Component, Debug, Clone, Copy, PartialEq, Default, Reflect)]
#[reflect(Component, Default)]
pub struct Velocity {
    pub linear: Vec3,
    pub angular: Vec3,
}

/// Мгновенный толчок (импульс, Н·с): применяется на ближайшем шаге и обнуляется.
/// Несколько толчков за кадр складываются.
#[derive(Component, Debug, Clone, Copy, PartialEq, Default)]
pub struct Impulse {
    pub linear: Vec3,
    pub angular: Vec3,
}

/// Затухание скорости — «сопротивление воздуха». По умолчанию нет.
#[derive(Component, Debug, Clone, Copy, PartialEq, Default, Reflect)]
#[reflect(Component, Default)]
pub struct Damping {
    pub linear: f32,
    pub angular: f32,
}

/// Непрерывное определение столкновений (CCD) для быстрых тел: снаряд не пролетит сквозь
/// тонкую стену между шагами. Дороже обычного — только где нужно.
#[derive(Component, Debug, Clone, Copy, Default, Reflect)]
#[reflect(Component, Default)]
pub struct Ccd;

/// Положение и поворот тела (без масштаба).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Pose {
    pub translation: Vec3,
    pub rotation: Quat,
}

impl From<&Transform> for Pose {
    fn from(transform: &Transform) -> Self {
        Self {
            translation: transform.translation,
            rotation: transform.rotation,
        }
    }
}

/// Начало или конец касания двух тел. Для датчиков — вход и выход.
#[derive(Message, Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionEvent {
    Started { a: Entity, b: Entity, sensor: bool },
    Stopped { a: Entity, b: Entity, sensor: bool },
}

impl CollisionEvent {
    /// Участники события.
    pub fn entities(&self) -> (Entity, Entity) {
        match *self {
            Self::Started { a, b, .. } | Self::Stopped { a, b, .. } => (a, b),
        }
    }
}

/// Ускорение свободного падения, м/с².
#[derive(Resource, Debug, Clone, Copy, PartialEq)]
pub struct Gravity(pub Vec3);

impl Default for Gravity {
    fn default() -> Self {
        Self(Vec3::new(0.0, -9.81, 0.0))
    }
}

/// Что запомнила физика при последней синхронизации: по разнице с текущими компонентами
/// видно, что их поменял код (телепорт, новая скорость, другой масштаб).
#[derive(Component, Debug, Clone, Copy)]
pub(crate) struct BodyState {
    pub pose: Pose,
    pub scale: Vec3,
    pub velocity: Velocity,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_shapes_stay_round() {
        let ball = ColliderShape::Ball { radius: 1.0 }.scaled(Vec3::new(1.0, 3.0, 2.0));
        assert_eq!(ball, ColliderShape::Ball { radius: 3.0 });
        let cube = ColliderShape::Cuboid {
            half_extents: Vec3::ONE,
        }
        .scaled(Vec3::new(2.0, -1.0, 0.5));
        assert_eq!(
            cube,
            ColliderShape::Cuboid {
                half_extents: Vec3::new(2.0, 1.0, 0.5)
            },
            "отражение (отрицательный масштаб) размер не меняет"
        );
    }

    #[test]
    fn trimesh_merges_primitives() {
        let mut mesh = Mesh::plane(2.0);
        mesh.primitives.push(mesh.primitives[0].clone());
        let ColliderShape::TriMesh { vertices, indices } = Collider::trimesh(&mesh).shape else {
            panic!("ожидался TriMesh");
        };
        assert_eq!(vertices.len(), 8);
        assert_eq!(indices.len(), 4);
        assert!(
            indices[2].iter().all(|&i| i >= 4),
            "индексы второго — со сдвигом"
        );
    }
}
