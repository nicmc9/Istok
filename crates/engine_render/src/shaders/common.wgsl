// Общее для всех шейдеров движка: данные кадра (камера, свет) и кодирование sRGB.
// Подключается через `#include "common.wgsl"`.

#include "color.wgsl"

// Точечный источник или прожектор
struct Light {
    position_range: vec4<f32>,   // xyz — позиция, w — радиус действия
    color_kind: vec4<f32>,       // rgb — цвет × интенсивность, w: 0 — точечный, 1 — прожектор
    direction_outer: vec4<f32>,  // xyz — куда светит прожектор, w — cos внешнего угла
    params: vec4<f32>,           // x — cos внутреннего угла, y — слой тени (< 0 — нет),
                                 // z — размер текселя тени на 1 м
};

struct Frame {
    view_proj: mat4x4<f32>,
    camera_position: vec4<f32>,
    ambient: vec4<f32>,           // rgb — цвет × интенсивность
    sun_direction: vec4<f32>,     // xyz — куда светит солнце, w = 1 — солнце есть
    sun_color: vec4<f32>,
    // x — число источников, y = 1 — кодировать sRGB вручную (Web, GLES)
    counts: vec4<u32>,
    lights: array<Light, 32>,
    camera_forward: vec4<f32>,
    // Тени от солнца: матрица «вид × проекция» каждого каскада
    cascades: array<mat4x4<f32>, 4>,
    cascade_splits: vec4<f32>,    // дальняя граница каскада (глубина вдоль взгляда камеры)
    cascade_texel: vec4<f32>,     // размер текселя каскада в метрах
    shadow_params: vec4<f32>,     // x = 1 — тени есть, y = 1 — подсветить каскады
    // Слои теней прожекторов (1 на источник) и точечных (6 граней: +X −X +Y −Y +Z −Z)
    local_shadows: array<mat4x4<f32>, 16>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var shadow_map: texture_depth_2d_array;
// Сэмплер сравнения: возвращает не глубину, а долю выборок, прошедших `ref <= глубина`
@group(0) @binding(2) var shadow_sampler: sampler_comparison;
@group(0) @binding(3) var local_shadow_map: texture_depth_2d_array;
