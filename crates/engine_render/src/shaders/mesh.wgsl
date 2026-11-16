// Общее для проходов, рисующих меши (PBR, тени): материал, кости, вершины.
// Группа 0 у каждого прохода своя (кадр камеры или вид из источника).
// Группа 1 — материал. Группа 2 — кости, только у скелетного меша.
// Матрица модели — атрибут экземпляра, не uniform: одинаковые меши рисуются одним вызовом.

struct Material {
    base_color: vec4<f32>,
    emissive: vec4<f32>,          // rgb — свечение, w — порог альфы (маска), < 0 — без маски
    // x — металличность, y — шероховатость, z — масштаб нормалей, w = 1 — есть карта нормалей
    params: vec4<f32>,
};

@group(1) @binding(0) var<uniform> material: Material;
@group(1) @binding(1) var base_color_texture: texture_2d<f32>;
@group(1) @binding(2) var metallic_roughness_texture: texture_2d<f32>;
@group(1) @binding(3) var normal_texture: texture_2d<f32>;
@group(1) @binding(4) var occlusion_texture: texture_2d<f32>;
@group(1) @binding(5) var emissive_texture: texture_2d<f32>;
@group(1) @binding(6) var material_sampler: sampler;

// Экземпляр: столбцы матрицы модели и нормалей. Кости — отдельно, location 13.
struct InstanceInput {
    @location(5) model0: vec4<f32>,
    @location(6) model1: vec4<f32>,
    @location(7) model2: vec4<f32>,
    @location(8) model3: vec4<f32>,
    @location(9) normal0: vec4<f32>,
    @location(10) normal1: vec4<f32>,
    @location(11) normal2: vec4<f32>,
    @location(12) normal3: vec4<f32>,
};

fn instance_model(inst: InstanceInput) -> mat4x4<f32> {
    return mat4x4<f32>(inst.model0, inst.model1, inst.model2, inst.model3);
}

fn instance_normal(inst: InstanceInput) -> mat4x4<f32> {
    return mat4x4<f32>(inst.normal0, inst.normal1, inst.normal2, inst.normal3);
}

// Группа 2. PRESKINNED — уже посчитанные вершины (compute). Иначе кости:
// с storage все скелеты кадра лежат подряд, экземпляр хранит начало своих;
// без storage привязан один скелет на вызов (смещение uniform).
#ifdef PRESKINNED
struct SkinnedVertex {
    position: vec4<f32>,
    normal: vec4<f32>,
};
@group(2) @binding(0) var<storage, read> skinned_vertices: array<SkinnedVertex>;
#else
#ifdef JOINT_STORAGE
@group(2) @binding(0) var<storage, read> joint_matrices: array<mat4x4<f32>>;
#else
struct Joints {
    matrices: array<mat4x4<f32>, 256>,
};
@group(2) @binding(0) var<uniform> joints: Joints;
#endif
#endif

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

// Матрица скиннинга: взвешенная сумма матриц четырёх костей.
#ifndef PRESKINNED
fn skin_matrix(offset: u32, joint_indices: vec4<u32>, joint_weights: vec4<f32>) -> mat4x4<f32> {
#ifdef JOINT_STORAGE
    let base = offset;
#else
    // Один скелет уже выбран смещением привязки. offset читаем, чтобы атрибут не выкинули.
    let base = offset * 0u;
#endif
    return joint_at(base + joint_indices.x) * joint_weights.x
        + joint_at(base + joint_indices.y) * joint_weights.y
        + joint_at(base + joint_indices.z) * joint_weights.z
        + joint_at(base + joint_indices.w) * joint_weights.w;
}

fn joint_at(index: u32) -> mat4x4<f32> {
#ifdef JOINT_STORAGE
    return joint_matrices[index];
#else
    return joints.matrices[index];
#endif
}
#endif
