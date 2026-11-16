//! Отладочная отрисовка коллайдеров линиями (CVar `physics.debug`). Цвет — по виду тела:
//! динамические — зелёные, неподвижные — серые, кинематические — синие, датчики — жёлтые.
//! Форма берётся из компонентов с тем же масштабом, что ушёл в бэкенд.

use std::f32::consts::{PI, TAU};

use engine_ecs::prelude::*;
use engine_math::{Quat, Vec3};
use engine_render::{Color, DebugDraw};
use engine_scene::Transform;

use crate::PHYSICS_DEBUG;
use crate::components::{Collider, ColliderShape, RigidBody};

/// Отрезков на окружность.
const SEGMENTS: usize = 24;
/// Больше треугольников сетки не рисуем: линий стало бы больше, чем полезно.
const MAX_TRIANGLES: usize = 20_000;

pub(crate) fn draw_colliders(
    debug: Option<ResMut<DebugDraw>>,
    bodies: Query<(&RigidBody, &Collider, &Transform)>,
) {
    let Some(mut debug) = debug.filter(|_| PHYSICS_DEBUG.get()) else {
        return;
    };
    for (kind, collider, transform) in &bodies {
        let color = if collider.sensor {
            Color::rgb(1.0, 0.85, 0.2)
        } else {
            match kind {
                RigidBody::Dynamic => Color::rgb(0.3, 1.0, 0.4),
                RigidBody::Fixed => Color::rgb(0.6, 0.6, 0.6),
                RigidBody::Kinematic => Color::rgb(0.3, 0.6, 1.0),
            }
        };
        let shape = collider.shape.scaled(transform.scale);
        let mut lines = Lines {
            debug: &mut debug,
            origin: transform.translation,
            rotation: transform.rotation,
            color,
        };
        lines.shape(&shape);
    }
}

/// Линии в локальных координатах тела.
struct Lines<'a> {
    debug: &'a mut DebugDraw,
    origin: Vec3,
    rotation: Quat,
    color: Color,
}

impl Lines<'_> {
    fn line(&mut self, a: Vec3, b: Vec3) {
        let to_world = |p: Vec3| self.origin + self.rotation * p;
        let (a, b) = (to_world(a), to_world(b));
        self.debug.line(a, b, self.color);
    }

    /// Дуга окружности радиуса `r` с центром `center` в плоскости осей `u`, `v`.
    fn arc(&mut self, center: Vec3, u: Vec3, v: Vec3, r: f32, from: f32, to: f32) {
        #[allow(clippy::cast_precision_loss)]
        let point = |i: usize| {
            let angle = from + (to - from) * i as f32 / SEGMENTS as f32;
            center + (u * angle.cos() + v * angle.sin()) * r
        };
        for i in 0..SEGMENTS {
            self.line(point(i), point(i + 1));
        }
    }

    fn shape(&mut self, shape: &ColliderShape) {
        match shape {
            ColliderShape::Cuboid { half_extents: h } => {
                let corner = |i: u8| {
                    Vec3::new(
                        if i & 1 == 0 { -h.x } else { h.x },
                        if i & 2 == 0 { -h.y } else { h.y },
                        if i & 4 == 0 { -h.z } else { h.z },
                    )
                };
                // 12 рёбер: пары углов, отличающихся одним битом (одной координатой)
                for i in 0..8u8 {
                    for bit in [1, 2, 4] {
                        if i & bit == 0 {
                            self.line(corner(i), corner(i | bit));
                        }
                    }
                }
            }
            ColliderShape::Ball { radius } => {
                for (u, v) in [(Vec3::X, Vec3::Y), (Vec3::Y, Vec3::Z), (Vec3::Z, Vec3::X)] {
                    self.arc(Vec3::ZERO, u, v, *radius, 0.0, TAU);
                }
            }
            ColliderShape::Capsule {
                half_height,
                radius,
            } => {
                let (top, bottom) = (Vec3::Y * *half_height, -Vec3::Y * *half_height);
                self.arc(top, Vec3::X, Vec3::Z, *radius, 0.0, TAU);
                self.arc(bottom, Vec3::X, Vec3::Z, *radius, 0.0, TAU);
                for side in [Vec3::X, Vec3::NEG_X, Vec3::Z, Vec3::NEG_Z] {
                    self.line(top + side * *radius, bottom + side * *radius);
                }
                // Полусферы: дуги в двух вертикальных плоскостях
                for side in [Vec3::X, Vec3::Z] {
                    self.arc(top, side, Vec3::Y, *radius, 0.0, PI);
                    self.arc(bottom, side, Vec3::Y, *radius, PI, TAU);
                }
            }
            ColliderShape::TriMesh { vertices, indices } => {
                for [a, b, c] in indices.iter().take(MAX_TRIANGLES) {
                    let point = |i: u32| vertices.get(i as usize).copied().unwrap_or_default();
                    let (a, b, c) = (point(*a), point(*b), point(*c));
                    self.line(a, b);
                    self.line(b, c);
                    self.line(c, a);
                }
            }
        }
    }
}
