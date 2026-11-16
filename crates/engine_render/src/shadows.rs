//! Тени от солнца: каскадные карты теней (R2-01, cascaded shadow maps).
//!
//! Одна карта на всю видимую область дала бы либо мыльные тени вблизи, либо короткую
//! дальность. Поэтому фрустум камеры режется по глубине на [`CASCADES`] участков: ближний
//! участок маленький и получает всю карту — тени чёткие; дальние — крупнее и грубее.
//!
//! - Каждый каскад — слой текстуры-массива глубины (`texture_depth_2d_array` в шейдере).
//! - Область каскада — сфера вокруг участка фрустума: её размер не зависит от поворота
//!   камеры, а центр сдвигается шагами в один тексель — тени не «дрожат» при движении.
//! - В карту рисуются все объекты, а не только видимые камерой: тень может отбрасывать то,
//!   что за кадром. Отсечение — по фрустуму каждого каскада.

use engine_core::CVar;
use engine_math::{Mat4, Vec3, Vec4Swizzles, camera};

use crate::scene::DEPTH_FORMAT;

/// Число каскадов. Совпадает с размерами массивов в `common.wgsl`.
pub(crate) const CASCADES: usize = 4;

pub static RENDER_SHADOWS: CVar<bool> = CVar::new("render.shadows", true, "Тени от солнца");
pub static RENDER_SHADOW_SIZE: CVar<u32> = CVar::new(
    "render.shadow_size",
    2048,
    "Размер карты теней одного каскада, пикселей (256–8192)",
);
pub static RENDER_SHADOW_DISTANCE: CVar<f32> = CVar::new(
    "render.shadow_distance",
    40.0,
    "Дальность теней от камеры, м",
);
pub static RENDER_SHADOW_DEBUG: CVar<bool> = CVar::new(
    "render.shadow_debug",
    false,
    "Подсветить каскады цветом (красный — ближний)",
);

/// Настройки теней кадра (из CVar), едут в снимке.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ShadowSettings {
    pub enabled: bool,
    pub size: u32,
    pub distance: f32,
    pub debug: bool,
    /// Размер карты прожектора / грани точечного источника.
    pub local_size: u32,
}

impl ShadowSettings {
    pub fn from_cvars() -> Self {
        Self {
            enabled: RENDER_SHADOWS.get(),
            size: RENDER_SHADOW_SIZE.get().clamp(256, 8192),
            distance: RENDER_SHADOW_DISTANCE.get().max(1.0),
            debug: RENDER_SHADOW_DEBUG.get(),
            local_size: crate::lights::RENDER_LOCAL_SHADOW_SIZE
                .get()
                .clamp(128, 4096),
        }
    }
}

impl Default for ShadowSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            size: 2048,
            distance: 40.0,
            debug: false,
            local_size: 1024,
        }
    }
}

/// Камера для расчёта каскадов.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CameraView {
    /// Мировая матрица камеры (камера смотрит вдоль −Z).
    pub world: Mat4,
    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

/// Рассчитанные каскады.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Cascades {
    /// «Проекция × вид» солнца для каждого каскада.
    pub view_proj: [Mat4; CASCADES],
    /// Дальняя граница каскада — глубина вдоль взгляда камеры.
    pub splits: [f32; CASCADES],
    /// Размер текселя каскада в метрах (для смещения по нормали).
    pub texel: [f32; CASCADES],
}

/// Границы участков: смесь логарифмического (чёткость вблизи) и равномерного (не слишком
/// мелкий первый каскад) разбиения — «practical split scheme» из GPU Gems 3, гл. 10.
pub(crate) fn split_distances(near: f32, far: f32) -> [f32; CASCADES] {
    const LAMBDA: f32 = 0.75;
    // [Rust] `std::array::from_fn` строит массив, вызывая замыкание для каждого индекса
    std::array::from_fn(|i| {
        #[allow(clippy::cast_precision_loss)] // i < 4
        let t = (i + 1) as f32 / CASCADES as f32;
        let log = near * (far / near).powf(t);
        let uniform = near + (far - near) * t;
        LAMBDA * log + (1.0 - LAMBDA) * uniform
    })
}

/// Каскады для камеры и направления солнца (`sun_direction` — куда светит).
pub(crate) fn compute_cascades(
    camera: &CameraView,
    sun_direction: Vec3,
    distance: f32,
    map_size: u32,
) -> Cascades {
    let near = camera.near.max(0.01);
    let far = distance.min(camera.far).max(near * 2.0);
    let splits = split_distances(near, far);
    let direction = sun_direction.try_normalize().unwrap_or(Vec3::NEG_Y);
    // Поворот «в систему солнца» без сдвига: вид из начала координат вдоль луча.
    // Сдвиг добавим позже, привязав к текселям.
    let up = if direction.y.abs() > 0.99 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let light_rotation = camera::look_at(Vec3::ZERO, direction, up);

    let mut view_proj = [Mat4::IDENTITY; CASCADES];
    let mut texel = [0.0; CASCADES];
    let mut slice_near = near;
    for cascade in 0..CASCADES {
        let slice_far = splits[cascade];
        let corners = slice_corners(camera, slice_near, slice_far);
        // [Rust] `iter().sum()` работает для Vec3: glam реализует трейт `Sum`
        #[allow(clippy::cast_precision_loss)]
        let center = corners.iter().copied().sum::<Vec3>() / corners.len() as f32;
        let radius = corners
            .iter()
            .map(|c| c.distance(center))
            .fold(0.0_f32, f32::max);
        // Округление радиуса вверх: мелкие колебания не меняют масштаб карты
        let radius = (radius * 16.0).ceil() / 16.0;
        #[allow(clippy::cast_precision_loss)]
        let texel_size = 2.0 * radius / map_size as f32;
        // Центр в системе солнца, сдвинутый на целое число текселей
        let mut center_ls = light_rotation.transform_point3(center);
        center_ls.x = (center_ls.x / texel_size).floor() * texel_size;
        center_ls.y = (center_ls.y / texel_size).floor() * texel_size;
        // Глубина: вперёд по лучу до дальнего края сферы, назад — с запасом на тех, кто
        // отбрасывает тень из-за пределов области (горы, крыши)
        let behind = distance.max(radius * 2.0);
        let projection = camera::orthographic(
            center_ls.x - radius,
            center_ls.x + radius,
            center_ls.y - radius,
            center_ls.y + radius,
            -center_ls.z - radius - behind,
            -center_ls.z + radius,
        );
        view_proj[cascade] = projection * light_rotation;
        texel[cascade] = texel_size;
        slice_near = slice_far;
    }
    Cascades {
        view_proj,
        splits,
        texel,
    }
}

/// Восемь углов участка фрустума камеры между глубинами `near` и `far`, в мире.
fn slice_corners(camera: &CameraView, near: f32, far: f32) -> [Vec3; 8] {
    let projection = camera::perspective(camera.fov_y, camera.aspect.max(1e-4), near, far);
    let to_world = camera.world * projection.inverse();
    let mut corners = [Vec3::ZERO; 8];
    for (i, corner) in corners.iter_mut().enumerate() {
        // Углы куба NDC: x, y ∈ {−1, 1}, z ∈ {0, 1} (глубина wgpu)
        let ndc = engine_math::Vec4::new(
            if i & 1 == 0 { -1.0 } else { 1.0 },
            if i & 2 == 0 { -1.0 } else { 1.0 },
            if i & 4 == 0 { 0.0 } else { 1.0 },
            1.0,
        );
        let world = to_world * ndc;
        *corner = world.xyz() / world.w;
    }
    corners
}

/// Текстура-массив карт теней.
pub(crate) struct ShadowMap {
    pub size: u32,
    /// Вид на весь массив — для выборки в PBR.
    pub array_view: wgpu::TextureView,
    /// Вид на каждый слой — цель прохода каскада.
    pub layer_views: Vec<wgpu::TextureView>,
    /// Оценка в байтах: слой — `Depth32Float`, 4 байта на пиксель.
    pub bytes: u64,
}

impl ShadowMap {
    /// `layers` слоёв `size × size`: каскады солнца или слои локальных источников.
    pub fn new(device: &wgpu::Device, size: u32, layers: usize, label: &'static str) -> Self {
        let size = size.min(device.limits().max_texture_dimension_2d);
        let bytes = u64::from(size) * u64::from(size) * u64::try_from(layers).unwrap_or(1) * 4;
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: u32::try_from(layers).unwrap_or(1),
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow_map_array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let layer_views = (0..layers)
            .map(|layer| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("shadow_cascade"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    #[allow(clippy::cast_possible_truncation)]
                    base_array_layer: layer as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        tracing::info!("Карта теней «{label}»: {layers} × {size}²");
        Self {
            size,
            array_view,
            layer_views,
            bytes,
        }
    }
}

/// Сэмплер сравнения для PCF: билинейная фильтрация результатов сравнения 2×2.
pub(crate) fn comparison_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("shadow"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_camera() -> CameraView {
        CameraView {
            world: camera::look_at(Vec3::new(3.0, 2.0, 8.0), Vec3::ZERO, Vec3::Y).inverse(),
            fov_y: 0.9,
            aspect: 16.0 / 9.0,
            near: 0.1,
            far: 500.0,
        }
    }

    #[test]
    fn splits_grow_and_end_at_distance() {
        let splits = split_distances(0.1, 40.0);
        assert!(splits.windows(2).all(|w| w[0] < w[1]), "{splits:?}");
        assert!((splits[CASCADES - 1] - 40.0).abs() < 1e-3);
        assert!(splits[0] < 10.0, "ближний каскад маленький: {splits:?}");
    }

    /// Каждый участок фрустума целиком попадает в карту своего каскада.
    #[test]
    fn slices_fit_their_cascade() {
        let camera = test_camera();
        let cascades = compute_cascades(&camera, Vec3::new(-0.4, -1.0, -0.3), 40.0, 2048);
        let mut near = camera.near;
        for i in 0..CASCADES {
            for corner in slice_corners(&camera, near, cascades.splits[i]) {
                let clip = cascades.view_proj[i] * corner.extend(1.0);
                let ndc = clip.xyz() / clip.w;
                assert!(
                    ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0 && (0.0..=1.0).contains(&ndc.z),
                    "каскад {i}: угол {corner} → {ndc}"
                );
            }
            near = cascades.splits[i];
        }
    }

    /// Небольшой сдвиг камеры меняет матрицу только на целое число текселей.
    #[test]
    fn cascades_move_in_whole_texels() {
        let mut camera = test_camera();
        let sun = Vec3::new(-0.4, -1.0, -0.3);
        let a = compute_cascades(&camera, sun, 40.0, 2048);
        camera.world = Mat4::from_translation(Vec3::new(0.013, 0.0, 0.007)) * camera.world;
        let b = compute_cascades(&camera, sun, 40.0, 2048);
        for i in 0..CASCADES {
            // Начало координат в пикселях карты: разница должна быть целой
            let pa = a.view_proj[i].project_point3(Vec3::ZERO);
            let pb = b.view_proj[i].project_point3(Vec3::ZERO);
            let shift = (pb.truncate() - pa.truncate()) * 1024.0; // NDC 2 → 2048 пикселей
            let fraction = (shift - shift.round()).abs();
            assert!(fraction.max_element() < 1e-2, "каскад {i}: сдвиг {shift}");
        }
    }
}
