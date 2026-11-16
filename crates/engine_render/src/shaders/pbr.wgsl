// PBR (metallic-roughness, как в glTF): BRDF Кука–Торранса с GGX.
// Группы привязок: 0 — кадр (камера, свет, карта теней), 1 — материал, 2 — кости
// скелетного меша. Матрица модели — атрибут экземпляра.

#include "common.wgsl"

#include "mesh.wgsl"

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput, inst: InstanceInput) -> VertexOutput {
    let world = instance_model(inst) * vec4<f32>(in.position, 1.0);
    var out: VertexOutput;
    out.clip_position = frame.view_proj * world;
    out.world_position = world.xyz;
    out.world_normal = (instance_normal(inst) * vec4<f32>(in.normal, 0.0)).xyz;
    out.uv = in.uv;
    return out;
}

// Скиннинг: вершина — взвешенная сумма положений по четырём костям.
// Матрицы костей уже мировые, у экземпляра матрица модели единичная.
#ifndef PRESKINNED
@vertex
fn vs_skinned(
    in: VertexInput,
    @location(3) joint_indices: vec4<u32>,
    @location(4) joint_weights: vec4<f32>,
    inst: InstanceInput,
    @location(13) joint_offset: vec4<u32>,
) -> VertexOutput {
    let posed = skin_matrix(joint_offset.x, joint_indices, joint_weights);
    let world = instance_model(inst) * posed * vec4<f32>(in.position, 1.0);
    var out: VertexOutput;
    out.clip_position = frame.view_proj * world;
    out.world_position = world.xyz;
    // Нормаль — той же матрицей (верно при равномерном масштабе костей)
    out.world_normal = (instance_normal(inst) * posed * vec4<f32>(in.normal, 0.0)).xyz;
    out.uv = in.uv;
    return out;
}
#endif

// Вершина уже посчитана compute. y смещения — индекс её позиции в буфере.
// UV берём из меша: в буфер результата они не пишутся.
#ifdef PRESKINNED
struct UvInput {
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_preskinned(
    uv_in: UvInput,
    inst: InstanceInput,
    @builtin(vertex_index) vertex_index: u32,
    @location(13) joint_offset: vec4<u32>,
) -> VertexOutput {
    let posed = skinned_vertices[joint_offset.y + vertex_index];
    let world = instance_model(inst) * vec4<f32>(posed.position.xyz, 1.0);
    var out: VertexOutput;
    out.clip_position = frame.view_proj * world;
    out.world_position = world.xyz;
    out.world_normal = (instance_normal(inst) * vec4<f32>(posed.normal.xyz, 0.0)).xyz;
    out.uv = uv_in.uv;
    return out;
}
#endif

const PI: f32 = 3.14159265;

// Распределение микрограней GGX
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Затенение микрограней (Смит, Шлик-GGX)
fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let gv = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let gl = n_dot_l / (n_dot_l * (1.0 - k) + k);
    return gv * gl;
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Вклад одного источника света
fn shade(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, radiance: vec3<f32>,
         albedo: vec3<f32>, metallic: f32, roughness: f32) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 1e-4);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let f = fresnel_schlick(max(dot(h, v), 0.0), f0);
    let specular = distribution_ggx(max(dot(n, h), 0.0), roughness)
        * geometry_smith(n_dot_v, n_dot_l, roughness) * f
        / max(4.0 * n_dot_v * n_dot_l, 1e-4);
    let diffuse = (vec3<f32>(1.0) - f) * (1.0 - metallic) * albedo / PI;
    return (diffuse + specular) * radiance * n_dot_l;
}

// Нормаль из карты нормалей без касательных в вершинах: базис строится по производным
// позиции и UV в экранном пространстве (метод Christian Schüler, «Followup: Normal Mapping
// Without Precomputed Tangents»).
fn perturb_normal(n: vec3<f32>, position: vec3<f32>, uv: vec2<f32>, mapped: vec3<f32>) -> vec3<f32> {
    let dp1 = dpdx(position);
    let dp2 = dpdy(position);
    let duv1 = dpdx(uv);
    let duv2 = dpdy(uv);
    let dp2perp = cross(dp2, n);
    let dp1perp = cross(n, dp1);
    let t = dp2perp * duv1.x + dp1perp * duv2.x;
    let b = dp2perp * duv1.y + dp1perp * duv2.y;
    let inv_max = inverseSqrt(max(dot(t, t), dot(b, b)) + 1e-12);
    return normalize(mat3x3<f32>(t * inv_max, b * inv_max, n) * mapped);
}


// Тень от солнца (каскадные карты): 1 — освещено, 0 — в тени.
// `geometric_n` — нормаль поверхности без карты нормалей: по ней сдвигаем точку выборки.
fn sun_shadow(world_position: vec3<f32>, geometric_n: vec3<f32>) -> f32 {
    if (frame.shadow_params.x < 0.5) {
        return 1.0;
    }
    // Глубина вдоль взгляда камеры — по ней выбирается каскад
    let depth = dot(world_position - frame.camera_position.xyz, frame.camera_forward.xyz);
    var cascade = 4u;
    for (var i = 0u; i < 4u; i++) {
        if (depth < frame.cascade_splits[i]) {
            cascade = i;
            break;
        }
    }
    if (cascade == 4u) {
        return 1.0; // дальше дистанции теней
    }
    // Смещение по нормали на пару текселей каскада убирает «теневые прыщи» (shadow acne)
    let offset = geometric_n * frame.cascade_texel[cascade] * 2.0;
    let clip = frame.cascades[cascade] * vec4<f32>(world_position + offset, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || ndc.z > 1.0) {
        return 1.0;
    }
    // PCF 3×3: каждая выборка сравнения ещё и билинейно смешивает 2×2 соседа
    let texel = 1.0 / vec2<f32>(textureDimensions(shadow_map));
    var lit = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            lit += textureSampleCompareLevel(shadow_map, shadow_sampler,
                uv + vec2<f32>(f32(x), f32(y)) * texel, cascade, ndc.z);
        }
    }
    lit /= 9.0;
    // Плавно гасим тень у дальней границы последнего каскада
    let far = frame.cascade_splits[3];
    return mix(lit, 1.0, smoothstep(far * 0.85, far, depth));
}

// Тень локального источника: 1 — освещено. `from_light` — вектор от источника к точке.
fn local_shadow(light: Light, world_position: vec3<f32>, geometric_n: vec3<f32>,
                from_light: vec3<f32>) -> f32 {
    if (light.params.y < 0.0) {
        return 1.0;
    }
    // Тексель перспективной карты растёт с расстоянием — и смещение тоже
    let offset = geometric_n * light.params.z * length(from_light) * 2.0;
    let sample_position = world_position + offset;
    var layer = u32(light.params.y);
    if (light.color_kind.w < 0.5) {
        // Точечный: грань куба — по наибольшей по модулю координате направления.
        // Берём уже смещённую точку: ту же, что спроецируем, иначе у стыка граней
        // точка может оказаться за краем выбранной грани
        let d = sample_position - light.position_range.xyz;
        let a = abs(d);
        if (a.x >= a.y && a.x >= a.z) {
            layer += select(1u, 0u, d.x > 0.0);
        } else if (a.y >= a.z) {
            layer += select(3u, 2u, d.y > 0.0);
        } else {
            layer += select(5u, 4u, d.z > 0.0);
        }
    }
    let clip = frame.local_shadows[layer] * vec4<f32>(sample_position, 1.0);
    if (clip.w <= 0.0) {
        return 1.0;
    }
    let ndc = clip.xyz / clip.w;
    let uv = ndc.xy * vec2<f32>(0.5, -0.5) + 0.5;
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || ndc.z > 1.0) {
        return 1.0;
    }
    let texel = 1.0 / vec2<f32>(textureDimensions(local_shadow_map));
    var lit = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            lit += textureSampleCompareLevel(local_shadow_map, shadow_sampler,
                uv + vec2<f32>(f32(x), f32(y)) * texel, layer, ndc.z);
        }
    }
    return lit / 9.0;
}

// Номер каскада для отладочной подсветки (`render.shadow_debug`)
fn cascade_tint(world_position: vec3<f32>) -> vec3<f32> {
    let depth = dot(world_position - frame.camera_position.xyz, frame.camera_forward.xyz);
    // `var`, а не `let`: индекс не константа, а значение-массив так индексировать нельзя
    var colors = array<vec3<f32>, 5>(vec3<f32>(1.0, 0.5, 0.5), vec3<f32>(0.5, 1.0, 0.5),
        vec3<f32>(0.5, 0.5, 1.0), vec3<f32>(1.0, 1.0, 0.5), vec3<f32>(1.0));
    var cascade = 4u;
    for (var i = 0u; i < 4u; i++) {
        if (depth < frame.cascade_splits[i]) {
            cascade = i;
            break;
        }
    }
    return colors[cascade];
}

@fragment
fn fs_main(in: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    // Все выборки и производные — в начале, до ветвлений: WGSL требует для них
    // «однородного» потока управления
    let base_sample = textureSample(base_color_texture, material_sampler, in.uv);
    let mr_sample = textureSample(metallic_roughness_texture, material_sampler, in.uv);
    let normal_sample = textureSample(normal_texture, material_sampler, in.uv).xyz;
    let occlusion = textureSample(occlusion_texture, material_sampler, in.uv).r;
    let emissive_sample = textureSample(emissive_texture, material_sampler, in.uv).rgb;

    var n = normalize(in.world_normal);
    if (!front) {
        n = -n; // двусторонний материал: обратная сторона
    }
    let geometric_n = n;
    let mapped = normalize(vec3<f32>((normal_sample.xy * 2.0 - 1.0) * material.params.z,
                                     normal_sample.z * 2.0 - 1.0));
    let perturbed = perturb_normal(n, in.world_position, in.uv, mapped);
    n = select(n, perturbed, material.params.w > 0.5);

    let base = material.base_color * base_sample;
    if (material.emissive.w >= 0.0 && base.a < material.emissive.w) {
        discard; // альфа-маска (листья, решётки)
    }
    let albedo = base.rgb;
    // В glTF: G — шероховатость, B — металличность
    let roughness = clamp(material.params.y * mr_sample.g, 0.04, 1.0);
    let metallic = clamp(material.params.x * mr_sample.b, 0.0, 1.0);
    let v = normalize(frame.camera_position.xyz - in.world_position);

    var color = vec3<f32>(0.0);
    if (frame.sun_direction.w > 0.5) {
        let shadow = sun_shadow(in.world_position, geometric_n);
        color += shade(n, v, -normalize(frame.sun_direction.xyz), frame.sun_color.rgb * shadow,
                       albedo, metallic, roughness);
    }
    for (var i = 0u; i < min(frame.counts.x, 32u); i++) {
        let light = frame.lights[i];
        let to_light = light.position_range.xyz - in.world_position;
        let distance = length(to_light);
        if (distance >= light.position_range.w) {
            continue; // вне радиуса — дешёвый выход до расчёта BRDF и тени
        }
        let l = to_light / max(distance, 1e-4);
        // Плавное затухание до нуля на границе радиуса
        let ratio = distance / max(light.position_range.w, 1e-4);
        let window = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
        var attenuation = window * window / (distance * distance + 1.0);
        if (light.color_kind.w > 0.5) {
            // Прожектор: мягкий край конуса между внешним и внутренним углом
            let cos_angle = dot(-l, light.direction_outer.xyz);
            attenuation *= smoothstep(light.direction_outer.w, light.params.x, cos_angle);
        }
        if (attenuation <= 0.0) {
            continue;
        }
        let shadow = local_shadow(light, in.world_position, geometric_n, -to_light);
        color += shade(n, v, l, light.color_kind.rgb * attenuation * shadow,
                       albedo, metallic, roughness);
    }
    color += frame.ambient.rgb * albedo * occlusion;
    color += material.emissive.rgb * emissive_sample;
    if (frame.shadow_params.y > 0.5) {
        color *= cascade_tint(in.world_position);
    }

#ifdef HDR_OUTPUT
    // HDR: яркость без ограничения сверху, тональная компрессия — отдельным проходом
    return vec4<f32>(color, base.a);
#else
    // Без HDR (старые GPU): компрессия и sRGB прямо здесь
    var out_color = tonemap_aces(color);
    if (frame.counts.y == 1u) {
        out_color = linear_to_srgb(out_color);
    }
    return vec4<f32>(out_color, base.a);
#endif
}
