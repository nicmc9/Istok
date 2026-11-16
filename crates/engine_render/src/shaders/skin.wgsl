// Скиннинг на compute: один поток на вершину одного экземпляра.
// Формула та же, что у vs_skinned: четыре кости, нормаль той же матрицей.
// Поза привязки читается как сырые f32 (vec3 в WGSL выровнен на 16 и не ложится на буфер).
// Веса — 24 байта: четыре u16 и четыре f32. Результат — позиция и нормаль по 16 байт.

struct SkinnedVertex {
    position: vec4<f32>,
    normal: vec4<f32>,
};

struct SkinInstance {
    joint_offset: u32,
    output_base: u32,
};

struct SkinParams {
    vertex_count: u32,
    instance_count: u32,
    instance_base: u32,
    // Шаг цикла: сколько потоков запущено вдоль X. Больше лимита — один поток считает несколько вершин.
    x_threads: u32,
};

@group(0) @binding(0) var<storage, read> bind_vertices: array<f32>;
@group(0) @binding(1) var<storage, read> skin_words: array<u32>;
@group(0) @binding(2) var<storage, read> joint_matrices: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage, read_write> skinned: array<SkinnedVertex>;
@group(0) @binding(4) var<storage, read> skin_instances: array<SkinInstance>;
@group(0) @binding(5) var<uniform> params: SkinParams;

fn joint_at(index: u32) -> mat4x4<f32> {
    return joint_matrices[index];
}

fn skin_matrix(offset: u32, joints: vec4<u32>, weights: vec4<f32>) -> mat4x4<f32> {
    return joint_at(offset + joints.x) * weights.x
        + joint_at(offset + joints.y) * weights.y
        + joint_at(offset + joints.z) * weights.z
        + joint_at(offset + joints.w) * weights.w;
}

fn skin_one(index: u32) {
    let vertex = index % params.vertex_count;
    let instance = index / params.vertex_count;
    let info = skin_instances[params.instance_base + instance];

    let base = vertex * 8u;
    let position = vec3<f32>(bind_vertices[base], bind_vertices[base + 1u], bind_vertices[base + 2u]);
    let normal = vec3<f32>(
        bind_vertices[base + 3u],
        bind_vertices[base + 4u],
        bind_vertices[base + 5u],
    );

    // Little-endian: младшие 16 бит — первый индекс кости.
    let word = vertex * 6u;
    let packed0 = skin_words[word];
    let packed1 = skin_words[word + 1u];
    let joints = vec4<u32>(
        packed0 & 0xFFFFu,
        packed0 >> 16u,
        packed1 & 0xFFFFu,
        packed1 >> 16u,
    );
    let weights = vec4<f32>(
        bitcast<f32>(skin_words[word + 2u]),
        bitcast<f32>(skin_words[word + 3u]),
        bitcast<f32>(skin_words[word + 4u]),
        bitcast<f32>(skin_words[word + 5u]),
    );

    let posed = skin_matrix(info.joint_offset, joints, weights);
    skinned[info.output_base + vertex] = SkinnedVertex(
        vec4<f32>((posed * vec4<f32>(position, 1.0)).xyz, 0.0),
        vec4<f32>((posed * vec4<f32>(normal, 0.0)).xyz, 0.0),
    );
}

// 64 — то же число, что WORKGROUP в skin.rs.
@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (params.vertex_count == 0u || params.x_threads == 0u) {
        return;
    }
    let total = params.vertex_count * params.instance_count;
    for (var index = id.x; index < total; index += params.x_threads) {
        skin_one(index);
    }
}
