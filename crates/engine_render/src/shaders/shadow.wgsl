// Карта теней: глубина сцены с точки зрения солнца, по каскаду за проход.
// Группа 0 — матрица каскада (смещение в буфере), 1 — материал, 2 — кости скелетного меша.
// Матрица модели — атрибут экземпляра. Нормали тени не нужны, в раскладке их нет.

#include "mesh.wgsl"

struct ShadowInstanceInput {
    @location(5) model0: vec4<f32>,
    @location(6) model1: vec4<f32>,
    @location(7) model2: vec4<f32>,
    @location(8) model3: vec4<f32>,
};

fn shadow_model(inst: ShadowInstanceInput) -> mat4x4<f32> {
    return mat4x4<f32>(inst.model0, inst.model1, inst.model2, inst.model3);
}

struct ShadowView {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> shadow_view: ShadowView;

struct ShadowOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput, inst: ShadowInstanceInput) -> ShadowOutput {
    var out: ShadowOutput;
    out.clip_position = shadow_view.view_proj * shadow_model(inst) * vec4<f32>(in.position, 1.0);
    out.uv = in.uv;
    return out;
}

@vertex
fn vs_skinned(
    in: VertexInput,
    @location(3) joint_indices: vec4<u32>,
    @location(4) joint_weights: vec4<f32>,
    inst: ShadowInstanceInput,
    @location(13) joint_offset: vec4<u32>,
) -> ShadowOutput {
    var out: ShadowOutput;
    let world = shadow_model(inst)
        * skin_matrix(joint_offset.x, joint_indices, joint_weights)
        * vec4<f32>(in.position, 1.0);
    out.clip_position = shadow_view.view_proj * world;
    out.uv = in.uv;
    return out;
}

// Альфа-маска (листья, решётки): дырки в материале — дырки и в тени.
// Непрозрачным объектам фрагментный шейдер не нужен — пишется только глубина.
@fragment
fn fs_masked(in: ShadowOutput) {
    let alpha = material.base_color.a * textureSample(base_color_texture, material_sampler, in.uv).a;
    if (alpha < material.emissive.w) {
        discard;
    }
}
