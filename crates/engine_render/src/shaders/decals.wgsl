// Проецируемая декаль (L-02). Куб декали только накрывает пиксели;
// точка на поверхности берётся из глубины кадра. Обратная матрица посчитана на CPU:
// у WGSL нет переносимого inverse.
// Грани замотаны наружу (векторное произведение рёбер смотрит из куба), рисуем только лицевые.

#include "color.wgsl"

struct DecalUniform {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    // x = 1 — кодировать sRGB в шейдере (цель без sRGB-вида и без HDR)
    extra: vec4<f32>,
}

struct Instance {
    @location(0) world0: vec4<f32>,
    @location(1) world1: vec4<f32>,
    @location(2) world2: vec4<f32>,
    @location(3) world3: vec4<f32>,
    @location(4) local0: vec4<f32>,
    @location(5) local1: vec4<f32>,
    @location(6) local2: vec4<f32>,
    @location(7) local3: vec4<f32>,
    @location(8) color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: DecalUniform;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(1) @binding(0) var decal_texture: texture_2d<f32>;
@group(1) @binding(1) var decal_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) local0: vec4<f32>,
    @location(2) local1: vec4<f32>,
    @location(3) local2: vec4<f32>,
    @location(4) local3: vec4<f32>,
}

fn columns(a: vec4<f32>, b: vec4<f32>, c: vec4<f32>, d: vec4<f32>) -> mat4x4<f32> {
    return mat4x4<f32>(a, b, c, d);
}

// 12 треугольников. Обход такой, что векторное произведение рёбер смотрит из куба:
// у грани +Z это (1, 0) × (0, 1) по XY. Обратный порядок GPU срезает как тыльную сторону.
const CORNER = array<vec3<f32>, 36>(
    vec3<f32>(-0.5, -0.5, 0.5), vec3<f32>(0.5, 0.5, 0.5), vec3<f32>(-0.5, 0.5, 0.5),
    vec3<f32>(-0.5, -0.5, 0.5), vec3<f32>(0.5, -0.5, 0.5), vec3<f32>(0.5, 0.5, 0.5),
    vec3<f32>(0.5, -0.5, -0.5), vec3<f32>(-0.5, 0.5, -0.5), vec3<f32>(-0.5, -0.5, -0.5),
    vec3<f32>(0.5, -0.5, -0.5), vec3<f32>(0.5, 0.5, -0.5), vec3<f32>(-0.5, 0.5, -0.5),
    vec3<f32>(-0.5, 0.5, 0.5), vec3<f32>(0.5, 0.5, -0.5), vec3<f32>(0.5, 0.5, 0.5),
    vec3<f32>(-0.5, 0.5, 0.5), vec3<f32>(-0.5, 0.5, -0.5), vec3<f32>(0.5, 0.5, -0.5),
    vec3<f32>(-0.5, -0.5, 0.5), vec3<f32>(0.5, -0.5, 0.5), vec3<f32>(0.5, -0.5, -0.5),
    vec3<f32>(-0.5, -0.5, 0.5), vec3<f32>(0.5, -0.5, -0.5), vec3<f32>(-0.5, -0.5, -0.5),
    vec3<f32>(0.5, -0.5, -0.5), vec3<f32>(0.5, 0.5, 0.5), vec3<f32>(0.5, 0.5, -0.5),
    vec3<f32>(0.5, -0.5, -0.5), vec3<f32>(0.5, -0.5, 0.5), vec3<f32>(0.5, 0.5, 0.5),
    vec3<f32>(-0.5, -0.5, 0.5), vec3<f32>(-0.5, 0.5, -0.5), vec3<f32>(-0.5, 0.5, 0.5),
    vec3<f32>(-0.5, -0.5, 0.5), vec3<f32>(-0.5, -0.5, -0.5), vec3<f32>(-0.5, 0.5, -0.5),
);

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, inst: Instance) -> VertexOutput {
    let world = columns(inst.world0, inst.world1, inst.world2, inst.world3) * vec4<f32>(CORNER[vi], 1.0);
    var out: VertexOutput;
    out.clip_position = camera.view_proj * world;
    out.color = inst.color;
    out.local0 = inst.local0;
    out.local1 = inst.local1;
    out.local2 = inst.local2;
    out.local3 = inst.local3;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(depth_tex));
    // position — пиксель, Y вниз. В NDC Y вверх, глубина уже 0..1 (DirectX).
    let uv = in.clip_position.xy / size;
    let depth = textureLoad(depth_tex, vec2<i32>(in.clip_position.xy), 0);
    if (depth >= 1.0) {
        discard;
    }
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world_h = camera.inv_view_proj * ndc;
    if (abs(world_h.w) < 1e-5) {
        discard;
    }
    let world = world_h.xyz / world_h.w;
    let local_h = columns(in.local0, in.local1, in.local2, in.local3) * vec4<f32>(world, 1.0);
    if (abs(local_h.w) < 1e-5) {
        discard;
    }
    let local = local_h.xyz / local_h.w;
    if (abs(local.x) > 0.5 || abs(local.y) > 0.5 || abs(local.z) > 0.5) {
        discard;
    }
    // v = 0 — верх картинки, он же локальный +Y.
    let stamp_uv = vec2<f32>(local.x + 0.5, 0.5 - local.y);
    let sample = textureSample(decal_texture, decal_sampler, stamp_uv);
    var color = sample.rgb * in.color.rgb;
    let alpha = sample.a * in.color.a;
    if (alpha <= 0.0) {
        discard;
    }
    if (camera.extra.x > 0.5) {
        color = linear_to_srgb(color);
    }
    return vec4<f32>(color, alpha);
}
