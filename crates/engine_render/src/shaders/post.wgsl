// Постэффекты: bloom, тональная компрессия, FXAA. Рисуются полноэкранным треугольником.
// Группа 0: исходная текстура, текстура bloom, сэмплер, параметры.

#include "color.wgsl"

struct PostParams {
    exposure: f32,
    bloom_intensity: f32,
    bloom_threshold: f32,
    // 1 — выход в sRGB-текстуру (кодирует оборудование): пишем линейные значения
    tonemap_output_linear: u32,
    fxaa_output_linear: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var bloom: texture_2d<f32>;
@group(0) @binding(2) var linear_sampler: sampler;
@group(0) @binding(3) var<uniform> params: PostParams;

struct FullscreenOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Один треугольник на весь экран (вершины 0, 1, 2 без вершинного буфера):
// покрывает экран целиком, лишнее отсекается — дешевле двух треугольников квада.
@vertex
fn vs_fullscreen(@builtin(vertex_index) index: u32) -> FullscreenOutput {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: FullscreenOutput;
    out.position = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    out.uv = uv;
    return out;
}

fn sample_source(uv: vec2<f32>) -> vec3<f32> {
    return textureSample(source, linear_sampler, uv).rgb;
}

// Уменьшение вдвое по 13 выборкам (Jimenez, «Next Generation Post Processing in Call of Duty:
// Advanced Warfare»): меньше мерцания, чем у простого усреднения 2×2.
fn downsample13(uv: vec2<f32>) -> vec3<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(source));
    let a = sample_source(uv + t * vec2<f32>(-2.0, -2.0));
    let b = sample_source(uv + t * vec2<f32>(0.0, -2.0));
    let c = sample_source(uv + t * vec2<f32>(2.0, -2.0));
    let d = sample_source(uv + t * vec2<f32>(-2.0, 0.0));
    let e = sample_source(uv);
    let f = sample_source(uv + t * vec2<f32>(2.0, 0.0));
    let g = sample_source(uv + t * vec2<f32>(-2.0, 2.0));
    let h = sample_source(uv + t * vec2<f32>(0.0, 2.0));
    let i = sample_source(uv + t * vec2<f32>(2.0, 2.0));
    let j = sample_source(uv + t * vec2<f32>(-1.0, -1.0));
    let k = sample_source(uv + t * vec2<f32>(1.0, -1.0));
    let l = sample_source(uv + t * vec2<f32>(-1.0, 1.0));
    let m = sample_source(uv + t * vec2<f32>(1.0, 1.0));
    return e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
}

// Первый уровень bloom: только то, что ярче порога (мягкое колено вместо резкой границы)
@fragment
fn fs_bloom_prefilter(in: FullscreenOutput) -> @location(0) vec4<f32> {
    // Ограничение сверху гасит «светлячков» — одиночные очень яркие пиксели
    let color = min(downsample13(in.uv), vec3<f32>(64.0));
    let brightness = max(color.r, max(color.g, color.b));
    let knee = params.bloom_threshold * 0.5;
    let soft = clamp(brightness - params.bloom_threshold + knee, 0.0, 2.0 * knee);
    let contribution = max(soft * soft / (4.0 * knee + 1e-4), brightness - params.bloom_threshold)
        / max(brightness, 1e-4);
    return vec4<f32>(color * contribution, 1.0);
}

@fragment
fn fs_bloom_down(in: FullscreenOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(downsample13(in.uv), 1.0);
}

// Увеличение меньшего уровня (source) фильтром-«палаткой» 3×3 плюс уровень того же размера
// из цепочки уменьшения (bloom). Отдельная текстура на выходе, а не смешивание в уровень
// уменьшения: так граф видит простую цепочку без циклов чтения/записи.
@fragment
fn fs_bloom_up(in: FullscreenOutput) -> @location(0) vec4<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(source));
    var sum = sample_source(in.uv) * 4.0;
    sum += (sample_source(in.uv + vec2<f32>(-t.x, 0.0)) + sample_source(in.uv + vec2<f32>(t.x, 0.0))
        + sample_source(in.uv + vec2<f32>(0.0, -t.y)) + sample_source(in.uv + vec2<f32>(0.0, t.y))) * 2.0;
    sum += sample_source(in.uv + vec2<f32>(-t.x, -t.y)) + sample_source(in.uv + vec2<f32>(t.x, -t.y))
        + sample_source(in.uv + vec2<f32>(-t.x, t.y)) + sample_source(in.uv + vec2<f32>(t.x, t.y));
    let same_level = textureSample(bloom, linear_sampler, in.uv).rgb;
    return vec4<f32>(sum / 16.0 + same_level, 1.0);
}

@fragment
fn fs_tonemap(in: FullscreenOutput) -> @location(0) vec4<f32> {
    let hdr = sample_source(in.uv);
    let glow = textureSample(bloom, linear_sampler, in.uv).rgb;
    let mapped = tonemap_aces((hdr + glow * params.bloom_intensity) * params.exposure);
    if (params.tonemap_output_linear == 1u) {
        return vec4<f32>(mapped, 1.0);
    }
    return vec4<f32>(linear_to_srgb(mapped), 1.0);
}

// FXAA (упрощённый вариант Тимоти Лоттеса): ищет перепады яркости и сглаживает вдоль края.
// Работает с цветами в sRGB — так перепады ближе к тому, что видит глаз.
@fragment
fn fs_fxaa(in: FullscreenOutput) -> @location(0) vec4<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(source));
    let rgb_nw = sample_source(in.uv + vec2<f32>(-t.x, -t.y));
    let rgb_ne = sample_source(in.uv + vec2<f32>(t.x, -t.y));
    let rgb_sw = sample_source(in.uv + vec2<f32>(-t.x, t.y));
    let rgb_se = sample_source(in.uv + vec2<f32>(t.x, t.y));
    let rgb_m = sample_source(in.uv);
    let luma_nw = luminance(rgb_nw);
    let luma_ne = luminance(rgb_ne);
    let luma_sw = luminance(rgb_sw);
    let luma_se = luminance(rgb_se);
    let luma_m = luminance(rgb_m);
    let luma_min = min(luma_m, min(min(luma_nw, luma_ne), min(luma_sw, luma_se)));
    let luma_max = max(luma_m, max(max(luma_nw, luma_ne), max(luma_sw, luma_se)));

    var dir = vec2<f32>(-((luma_nw + luma_ne) - (luma_sw + luma_se)), (luma_nw + luma_sw) - (luma_ne + luma_se));
    let reduce = max((luma_nw + luma_ne + luma_sw + luma_se) * 0.25 * (1.0 / 8.0), 1.0 / 128.0);
    let rcp_min = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * rcp_min, vec2<f32>(-8.0), vec2<f32>(8.0)) * t;

    let rgb_a = 0.5 * (sample_source(in.uv + dir * (1.0 / 3.0 - 0.5)) + sample_source(in.uv + dir * (2.0 / 3.0 - 0.5)));
    let rgb_b = rgb_a * 0.5 + 0.25 * (sample_source(in.uv - dir * 0.5) + sample_source(in.uv + dir * 0.5));
    let luma_b = luminance(rgb_b);
    let result = select(rgb_b, rgb_a, luma_b < luma_min || luma_b > luma_max);
    if (params.fxaa_output_linear == 1u) {
        return vec4<f32>(srgb_to_linear(result), 1.0);
    }
    return vec4<f32>(result, 1.0);
}
