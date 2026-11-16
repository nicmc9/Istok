// Текст из SDF-атласа (R2-03). Один шейдер для трёх режимов, режим — в position.w:
// 0 — мир, 1 — подпись над точкой мира, 2 — экран (пиксели от левого верхнего угла).

#include "color.wgsl"

struct TextUniform {
    view_proj: mat4x4<f32>,
    // xy — размер цели в пикселях, z = 1 — кодировать sRGB в шейдере
    screen: vec4<f32>,
};

@group(0) @binding(0) var<uniform> text: TextUniform;
@group(0) @binding(1) var atlas: texture_2d<f32>;
@group(0) @binding(2) var atlas_sampler: sampler;

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) offset_uv: vec4<f32>,
    @location(2) color: vec4<f32>,
    @location(3) outline: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) outline: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let mode = u32(in.position.w + 0.5);
    let to_ndc = 2.0 / text.screen.xy;
    if (mode == 0u) {
        out.clip_position = text.view_proj * vec4<f32>(in.position.xyz, 1.0);
    } else if (mode == 1u) {
        // Сдвиг в пикселях переводим в clip-пространство (умножение на w — до деления
        // перспективы), на экране Y вниз, в NDC — вверх
        var clip = text.view_proj * vec4<f32>(in.position.xyz, 1.0);
        clip = vec4<f32>(clip.xy + vec2<f32>(in.offset_uv.x, -in.offset_uv.y) * to_ndc * clip.w, clip.zw);
        // Точка позади камеры — подпись не рисуем (вершина за пределами отсечения)
        if (clip.w <= 0.0) {
            clip = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        }
        out.clip_position = clip;
    } else {
        let ndc = vec2<f32>(in.position.x * to_ndc.x - 1.0, 1.0 - in.position.y * to_ndc.y);
        out.clip_position = vec4<f32>(ndc, 0.0, 1.0);
    }
    out.uv = in.offset_uv.zw;
    out.color = in.color;
    out.outline = in.outline;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let distance = textureSample(atlas, atlas_sampler, in.uv).r;
    // Ширина сглаживания — изменение значения SDF на один пиксель экрана: край чёткий
    // и у мелкого, и у крупного текста
    let aa = max(fwidth(distance) * 0.5, 1e-4);
    let fill = smoothstep(0.5 - aa, 0.5 + aa, distance);
    var color = in.color.rgb;
    var alpha = fill;
    if (in.outline.a > 0.0) {
        // Обводка — тот же контур, но на уровне ниже 0.5 (дальше от буквы)
        let edge = 0.5 - in.outline.a;
        alpha = smoothstep(edge - aa, edge + aa, distance);
        color = mix(in.outline.rgb, color, fill);
    }
    alpha *= in.color.a;
    if (alpha <= 0.0) {
        discard;
    }
    if (text.screen.z > 0.5) {
        color = linear_to_srgb(color);
    }
    return vec4<f32>(color, alpha);
}
