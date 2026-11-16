// Общее для проходов, рисующих меши (PBR, тени): материал, объект, кости, вершины.
// Группы привязок 1–3; группа 0 у каждого прохода своя (кадр камеры или вид из источника).

struct Material {
    base_color: vec4<f32>,
    emissive: vec4<f32>,          // rgb — свечение, w — порог альфы (маска), < 0 — без маски
    // x — металличность, y — шероховатость, z — масштаб нормалей, w = 1 — есть карта нормалей
    params: vec4<f32>,
};

struct Object {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,   // обратная транспонированная: верна и при неравномерном масштабе
};


@group(1) @binding(0) var<uniform> material: Material;
@group(1) @binding(1) var base_color_texture: texture_2d<f32>;
@group(1) @binding(2) var metallic_roughness_texture: texture_2d<f32>;
@group(1) @binding(3) var normal_texture: texture_2d<f32>;
@group(1) @binding(4) var occlusion_texture: texture_2d<f32>;
@group(1) @binding(5) var emissive_texture: texture_2d<f32>;
@group(1) @binding(6) var material_sampler: sampler;

@group(2) @binding(0) var<uniform> object: Object;

// Матрицы костей скелетного меша: мировая матрица кости × обратная матрица привязки.
// Uniform на 256 костей. Вершины скелетного меша уже в мире.
struct Joints {
    matrices: array<mat4x4<f32>, 256>,
};
@group(3) @binding(0) var<uniform> joints: Joints;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

// Матрица скиннинга: взвешенная сумма матриц четырёх костей.
fn skin_matrix(joint_indices: vec4<u32>, joint_weights: vec4<f32>) -> mat4x4<f32> {
    return joints.matrices[joint_indices.x] * joint_weights.x
        + joints.matrices[joint_indices.y] * joint_weights.y
        + joints.matrices[joint_indices.z] * joint_weights.z
        + joints.matrices[joint_indices.w] * joint_weights.w;
}
