//! Локальные источники света (R2-02): точечные и прожекторы.
//!
//! Каждый кадр источники отбираются на CPU:
//! 1. отбрасываются те, чья сфера действия не пересекает фрустум камеры;
//! 2. оставшиеся сортируются по удалённости от камеры (ближние важнее);
//! 3. в кадр попадают первые [`MAX_LIGHTS`]; слоёв теней ([`LOCAL_SHADOW_LAYERS`]) хватает
//!    ближним источникам с тенями: прожектору нужен 1 слой, точечному — 6 (грани куба).
//!
//! Шейдер перебирает все источники кадра для каждого пикселя — цена растёт линейно.
//! Для сотен источников нужен кластерный forward (Б-11): источники раскладываются по ячейкам
//! экрана × глубины, пиксель смотрит только свои.

use engine_core::CVar;
use engine_math::{Aabb, Frustum, Mat4, Vec3, camera};

/// Источников в кадре. Совпадает с размером массива в `common.wgsl`.
pub(crate) const MAX_LIGHTS: usize = 32;
/// Слоёв в карте теней локальных источников. Совпадает с `common.wgsl`.
pub(crate) const LOCAL_SHADOW_LAYERS: usize = 16;

pub static RENDER_LOCAL_SHADOW_SIZE: CVar<u32> = CVar::new(
    "render.local_shadow_size",
    1024,
    "Размер карты теней прожектора / грани точечного источника, пикселей (128–4096)",
);

/// Прожектор: конус с мягким краем между внутренним и внешним углом.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Spot {
    /// Куда светит (единичный).
    pub direction: Vec3,
    /// Половина угла раствора, радианы: полная яркость внутри `inner`, ноль за `outer`.
    pub inner: f32,
    pub outer: f32,
}

/// Точечный источник или прожектор — в снимке кадра.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LocalLight {
    pub position: Vec3,
    pub range: f32,
    /// Цвет × интенсивность.
    pub color: [f32; 3],
    /// `None` — точечный.
    pub spot: Option<Spot>,
    pub shadows: bool,
}

/// Источник, попавший в кадр.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SelectedLight {
    pub light: LocalLight,
    /// Первый слой карты теней; `None` — без тени (выключена или не хватило слоёв).
    pub shadow_layer: Option<u32>,
    /// Размер текселя карты на расстоянии 1 м — для смещения по нормали.
    pub texel_per_meter: f32,
}

/// Результат отбора: источники кадра и матрицы слоёв теней (по порядку слоёв).
#[derive(Debug, Default)]
pub(crate) struct LightSelection {
    pub lights: Vec<SelectedLight>,
    pub shadow_views: Vec<Mat4>,
}

pub(crate) fn select_lights(
    lights: &[LocalLight],
    frustum: &Frustum,
    camera_position: Vec3,
    shadows_enabled: bool,
    shadow_size: u32,
) -> LightSelection {
    let mut candidates: Vec<(f32, &LocalLight)> = lights
        .iter()
        .filter(|light| light.range > 0.0)
        .filter(|light| {
            let r = Vec3::splat(light.range);
            frustum.intersects_aabb(&Aabb {
                min: light.position - r,
                max: light.position + r,
            })
        })
        // Удалённость края сферы: камера внутри сферы — 0, самый важный
        .map(|light| {
            (
                (light.position.distance(camera_position) - light.range).max(0.0),
                light,
            )
        })
        .collect();
    // [Rust] `total_cmp` — полный порядок для f32 (NaN не ломает сортировку, в отличие
    // от `partial_cmp().unwrap()`)
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    if candidates.len() > MAX_LIGHTS {
        tracing::debug!(
            "Источников в кадре {}, рисуем ближние {MAX_LIGHTS}",
            candidates.len()
        );
    }

    let mut selection = LightSelection::default();
    for (_, light) in candidates.into_iter().take(MAX_LIGHTS) {
        let views = if light.shadows && shadows_enabled {
            shadow_views(light, shadow_size)
        } else {
            Vec::new()
        };
        let free = LOCAL_SHADOW_LAYERS - selection.shadow_views.len();
        let shadow_layer = (!views.is_empty() && views.len() <= free)
            .then(|| u32::try_from(selection.shadow_views.len()).unwrap_or(0));
        if shadow_layer.is_some() {
            selection.shadow_views.extend(views);
        }
        // Ширина видимой области на расстоянии 1 м = 2·tg(половины угла)
        let half_fov = light
            .spot
            .map_or_else(|| cube_half_fov(shadow_size), |spot| spot_half_fov(&spot));
        #[allow(clippy::cast_precision_loss)]
        let texel_per_meter = 2.0 * half_fov.tan() / shadow_size.max(1) as f32;
        selection.lights.push(SelectedLight {
            light: *light,
            shadow_layer,
            texel_per_meter,
        });
    }
    selection
}

/// Половина угла обзора карты теней прожектора: внешний конус с небольшим запасом,
/// чтобы PCF у края не выходил за карту.
fn spot_half_fov(spot: &Spot) -> f32 {
    (spot.outer * 1.05).clamp(0.01, 1.5)
}

/// Половина угла грани куба: чуть больше 45°, чтобы у края грани выборки PCF (±1 тексель)
/// и смещение по нормали оставались внутри карты — иначе на стыке граней светлый шов.
fn cube_half_fov(shadow_size: u32) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let margin = 3.0 / shadow_size.max(1) as f32;
    (1.0 + margin).atan()
}

/// Грани куба точечного источника: +X, −X, +Y, −Y, +Z, −Z — тот же порядок в `pbr.wgsl`.
const CUBE_FACES: [(Vec3, Vec3); 6] = [
    (Vec3::X, Vec3::Y),
    (Vec3::NEG_X, Vec3::Y),
    (Vec3::Y, Vec3::Z),
    (Vec3::NEG_Y, Vec3::Z),
    (Vec3::Z, Vec3::Y),
    (Vec3::NEG_Z, Vec3::Y),
];

/// Матрицы «проекция × вид» слоёв теней источника.
fn shadow_views(light: &LocalLight, shadow_size: u32) -> Vec<Mat4> {
    // Ближняя плоскость: слишком маленькая съедает точность глубины
    let near = (light.range * 0.005).max(0.02);
    let far = light.range.max(near * 2.0);
    let eye = light.position;
    // [Rust] `if let ... else` — разбор одного варианта Option, остальное — в else
    if let Some(spot) = light.spot {
        let up = if spot.direction.y.abs() > 0.99 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        let projection = camera::perspective(2.0 * spot_half_fov(&spot), 1.0, near, far);
        vec![projection * camera::look_at(eye, eye + spot.direction, up)]
    } else {
        // Чуть шире 90°: соседние грани немного перекрываются (см. `cube_half_fov`)
        let fov = 2.0 * cube_half_fov(shadow_size);
        let projection = camera::perspective(fov, 1.0, near, far);
        CUBE_FACES
            .iter()
            .map(|(axis, up)| projection * camera::look_at(eye, eye + *axis, *up))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(x: f32, shadows: bool) -> LocalLight {
        LocalLight {
            position: Vec3::new(x, 0.0, -10.0),
            range: 3.0,
            color: [1.0; 3],
            spot: None,
            shadows,
        }
    }

    fn frustum() -> Frustum {
        // Камера в начале координат смотрит вдоль −Z
        Frustum::from_view_projection(&camera::perspective(1.2, 1.0, 0.1, 100.0))
    }

    #[test]
    fn culls_outside_and_sorts_by_distance() {
        let lights = [point(1.0, false), point(0.0, false), point(500.0, false)];
        let selection = select_lights(&lights, &frustum(), Vec3::ZERO, true, 1024);
        assert_eq!(selection.lights.len(), 2, "дальний сбоку отсечён");
        assert_eq!(selection.lights[0].light.position.x, 0.0, "ближний первым");
    }

    #[test]
    fn limits_count() {
        let lights: Vec<_> = (0..100u16)
            .map(|i| point(0.0, false).with_z(-5.0 - f32::from(i) * 0.5))
            .collect();
        let selection = select_lights(&lights, &frustum(), Vec3::ZERO, true, 1024);
        assert_eq!(selection.lights.len(), MAX_LIGHTS);
    }

    #[test]
    fn shadow_layers_run_out() {
        // 3 точечных с тенями: 6 + 6 слоёв, третьему (ещё 6) места нет
        let lights = [point(0.0, true), point(0.1, true), point(0.2, true)];
        let selection = select_lights(&lights, &frustum(), Vec3::ZERO, true, 1024);
        let layers: Vec<_> = selection.lights.iter().map(|l| l.shadow_layer).collect();
        assert_eq!(layers, [Some(0), Some(6), None]);
        assert_eq!(selection.shadow_views.len(), 12);
        // Прожектору хватает одного слоя
        let spot = LocalLight {
            spot: Some(Spot {
                direction: Vec3::NEG_Y,
                inner: 0.3,
                outer: 0.5,
            }),
            ..point(0.3, true)
        };
        let lights = [point(0.0, true), point(0.1, true), spot];
        let selection = select_lights(&lights, &frustum(), Vec3::ZERO, true, 1024);
        assert_eq!(selection.lights[2].shadow_layer, Some(12));
    }

    /// Точка в направлении грани попадает в фрустум этой грани.
    #[test]
    fn cube_faces_cover_their_axes() {
        let light = point(0.0, true);
        let views = shadow_views(&light, 1024);
        for (face, (axis, _)) in CUBE_FACES.iter().enumerate() {
            let target = light.position + *axis * 1.5 + Vec3::splat(0.2);
            let ndc = views[face].project_point3(target);
            assert!(
                ndc.x.abs() < 1.0 && ndc.y.abs() < 1.0 && (0.0..1.0).contains(&ndc.z),
                "грань {face}: {ndc}"
            );
        }
    }

    impl LocalLight {
        fn with_z(mut self, z: f32) -> Self {
            self.position.z = z;
            self
        }
    }
}
