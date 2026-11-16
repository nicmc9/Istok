// Билборд частицы (L-02). Освещение не считается: цвет задаёт эмиттер.
// Симуляция на GPU сюда не входит — позиции уже мировые, их прислал CPU.

#include "color.wgsl"

struct ParticleUniform {
    view_proj: mat4x4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    // x = 1 — кодировать sRGB в шейдере (цель без sRGB-вида и без HDR)
    extra: vec4<f32>,
}

struct Instance {
    @location(0) center_size: vec4<f32>,
    @location(1) color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> particle: ParticleUniform;
@group(0) @binding(1) var particle_texture: texture_2d<f32>;
@group(0) @binding(2) var particle_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

// Один квад на частицу, без буфера вершин: угол берётся из номера вершины.
const CORNER = array<vec2<f32>, 4>(
    vec2<f32>(-0.5, 0.5),
    vec2<f32>(-0.5, -0.5),
    vec2<f32>(0.5, -0.5),
    vec2<f32>(0.5, 0.5),
);
const TRI = array<u32, 6>(0u, 1u, 2u, 0u, 2u, 3u);
const UV = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, 0.0),
);

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, inst: Instance) -> VertexOutput {
    let id = TRI[vi];
    let corner = CORNER[id];
    let center = inst.center_size.xyz;
    let size = inst.center_size.w;
    let world = center + particle.right.xyz * corner.x * size + particle.up.xyz * corner.y * size;
    var out: VertexOutput;
    out.clip_position = particle.view_proj * vec4<f32>(world, 1.0);
    out.uv = UV[id];
    out.color = inst.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sample = textureSample(particle_texture, particle_sampler, in.uv);
    var color = sample.rgb * in.color.rgb;
    let alpha = sample.a * in.color.a;
    if (alpha <= 0.0) {
        discard;
    }
    if (particle.extra.x > 0.5) {
        color = linear_to_srgb(color);
    }
    return vec4<f32>(color, alpha);
}
