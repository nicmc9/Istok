// Спрайт без освещения (R2-13). Цвет текстуры уже линейный (sRGB-формат на GPU).

#include "color.wgsl"

struct SpriteUniform {
    view_proj: mat4x4<f32>,
    // x = 1 — кодировать sRGB в шейдере (цель без sRGB-вида и без HDR)
    extra: vec4<f32>,
};

@group(0) @binding(0) var<uniform> sprite: SpriteUniform;
@group(0) @binding(1) var sprite_texture: texture_2d<f32>;
@group(0) @binding(2) var sprite_sampler: sampler;

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = sprite.view_proj * vec4<f32>(in.position.xyz, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sample = textureSample(sprite_texture, sprite_sampler, in.uv);
    var color = sample.rgb * in.color.rgb;
    let alpha = sample.a * in.color.a;
    if (alpha <= 0.0) {
        discard;
    }
    if (sprite.extra.x > 0.5) {
        color = linear_to_srgb(color);
    }
    return vec4<f32>(color, alpha);
}
