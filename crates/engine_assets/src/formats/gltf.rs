//! glTF 2.0 (`.gltf` с внешними файлами и `.glb`). Разбор — крейт `gltf`, но буферы и
//! внешние текстуры читаются через нашу VFS: тот же кеш, сеть и пути, что у всех ресурсов.
//!
//! Результат — данные на CPU. В GPU их переведёт рендер (этап 5).

use engine_ecs::prelude::{TypePath, bevy_reflect};
use std::collections::HashSet;

use engine_core::tasks::BoxedFuture;
use engine_math::{Aabb, Quat, Vec3};

use super::animation::{AnimationClip, Channel, Interpolation, Keyframes};
use super::image::{ColorSpace, Image};
use crate::{Asset, AssetError, AssetLoader, Bytes, Handle, LoadContext};

/// Часть меша с одним материалом.
#[derive(Debug, Clone)]
pub struct Primitive {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tangents: Vec<[f32; 4]>,
    pub uvs: Vec<[f32; 2]>,
    /// Индексы костей и веса для скиннинга (4 кости на вершину).
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    /// Индекс в `Gltf::materials`.
    pub material: Option<usize>,
    /// Границы в локальных координатах меша — для отсечения.
    pub bounds: Aabb,
}

/// Меш. Вложенный ресурс: путь `файл.glb#Mesh0`.
#[derive(Debug, Clone, Default, TypePath)]
pub struct Mesh {
    pub name: String,
    pub primitives: Vec<Primitive>,
}

impl Asset for Mesh {
    fn memory_category(&self) -> crate::MemoryCategory {
        crate::MemoryCategory::Mesh
    }
    fn memory_size(&self) -> u64 {
        self.primitives
            .iter()
            .map(|p| {
                size_of_slice(&p.positions)
                    + size_of_slice(&p.normals)
                    + size_of_slice(&p.tangents)
                    + size_of_slice(&p.uvs)
                    + size_of_slice(&p.joints)
                    + size_of_slice(&p.weights)
                    + size_of_slice(&p.indices)
            })
            .sum()
    }
}

fn size_of_slice<T>(items: &[T]) -> u64 {
    u64::try_from(std::mem::size_of_val(items)).unwrap_or(u64::MAX)
}

impl Mesh {
    pub fn vertex_count(&self) -> usize {
        self.primitives.iter().map(|p| p.positions.len()).sum()
    }
    pub fn triangle_count(&self) -> usize {
        self.primitives.iter().map(|p| p.indices.len() / 3).sum()
    }

    /// Квадратная плоскость `size × size` в XZ, нормаль вверх (+Y), центр в начале координат.
    /// UV повторяется раз в метр — текстура не растягивается на большой плоскости.
    pub fn plane(size: f32) -> Self {
        let h = size * 0.5;
        let positions = vec![[-h, 0.0, -h], [h, 0.0, -h], [h, 0.0, h], [-h, 0.0, h]];
        let uvs = vec![[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]];
        let bounds = Aabb::from_points(positions.iter().map(|p| Vec3::from_array(*p)));
        Self {
            name: "plane".to_owned(),
            primitives: vec![Primitive {
                positions,
                normals: vec![[0.0, 1.0, 0.0]; 4],
                tangents: Vec::new(),
                uvs,
                joints: Vec::new(),
                weights: Vec::new(),
                // Против часовой стрелки, если смотреть сверху — лицевая сторона вверх
                indices: vec![0, 2, 1, 0, 3, 2],
                material: None,
                bounds,
            }],
        }
    }
}

impl Mesh {
    /// Параллелепипед с половинами размеров `half`, центр в начале координат. У каждой грани
    /// свои вершины — нормали резкие, UV на грани 0..1.
    pub fn cuboid(half: Vec3) -> Self {
        // (нормаль, ось U, ось V) грани; U × V = нормаль — обход против часовой стрелки
        let faces = [
            (Vec3::X, Vec3::NEG_Z, Vec3::Y),
            (Vec3::NEG_X, Vec3::Z, Vec3::Y),
            (Vec3::Y, Vec3::X, Vec3::NEG_Z),
            (Vec3::NEG_Y, Vec3::X, Vec3::Z),
            (Vec3::Z, Vec3::X, Vec3::Y),
            (Vec3::NEG_Z, Vec3::NEG_X, Vec3::Y),
        ];
        let mut positions = Vec::with_capacity(24);
        let mut normals = Vec::with_capacity(24);
        let mut uvs = Vec::with_capacity(24);
        let mut indices = Vec::with_capacity(36);
        for (normal, u, v) in faces {
            let base = u32::try_from(positions.len()).unwrap_or(0);
            for (su, sv) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                positions.push(((normal + u * su + v * sv) * half).to_array());
                normals.push(normal.to_array());
                uvs.push([f32::midpoint(su, 1.0), f32::midpoint(1.0, -sv)]);
            }
            indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        }
        Self::from_parts("cuboid", positions, normals, uvs, indices)
    }

    /// Сфера из `rings` колец по `segments` вершин (UV-сфера), центр в начале координат.
    pub fn sphere(radius: f32, segments: u32, rings: u32) -> Self {
        let (segments, rings) = (segments.max(3), rings.max(2));
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        #[allow(clippy::cast_precision_loss)] // сотни вершин
        for ring in 0..=rings {
            let v = ring as f32 / rings as f32;
            let theta = v * std::f32::consts::PI; // от северного полюса
            for segment in 0..=segments {
                let u = segment as f32 / segments as f32;
                let phi = u * std::f32::consts::TAU;
                let normal = Vec3::new(
                    theta.sin() * phi.cos(),
                    theta.cos(),
                    -theta.sin() * phi.sin(),
                );
                positions.push((normal * radius).to_array());
                normals.push(normal.to_array());
                uvs.push([u, v]);
            }
        }
        let row = segments + 1;
        let mut indices = Vec::new();
        for ring in 0..rings {
            for segment in 0..segments {
                let a = ring * row + segment;
                let b = a + row;
                indices.extend([a, b, a + 1, a + 1, b, b + 1]);
            }
        }
        Self::from_parts("sphere", positions, normals, uvs, indices)
    }

    fn from_parts(
        name: &str,
        positions: Vec<[f32; 3]>,
        normals: Vec<[f32; 3]>,
        uvs: Vec<[f32; 2]>,
        indices: Vec<u32>,
    ) -> Self {
        let bounds = Aabb::from_points(positions.iter().map(|p| Vec3::from_array(*p)));
        Self {
            name: name.to_owned(),
            primitives: vec![Primitive {
                positions,
                normals,
                tangents: Vec::new(),
                uvs,
                joints: Vec::new(),
                weights: Vec::new(),
                indices,
                material: None,
                bounds,
            }],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlphaMode {
    Opaque,
    Mask(u8),
    Blend,
}

/// PBR-материал (metallic-roughness, как в glTF). Вложенный ресурс: `файл.glb#Material0`.
#[derive(Debug, Clone, TypePath)]
pub struct Material {
    pub name: String,
    pub base_color: [f32; 4],
    pub base_color_texture: Option<Handle<Image>>,
    pub metallic: f32,
    pub roughness: f32,
    pub metallic_roughness_texture: Option<Handle<Image>>,
    pub normal_texture: Option<Handle<Image>>,
    pub occlusion_texture: Option<Handle<Image>>,
    pub emissive: [f32; 3],
    pub emissive_texture: Option<Handle<Image>>,
    pub alpha_mode: AlphaMode,
    pub double_sided: bool,
}

impl Asset for Material {}

impl Default for Material {
    /// Белый матовый неметалл — как материал по умолчанию в glTF.
    fn default() -> Self {
        Self {
            name: String::new(),
            base_color: [1.0; 4],
            base_color_texture: None,
            metallic: 1.0,
            roughness: 1.0,
            metallic_roughness_texture: None,
            normal_texture: None,
            occlusion_texture: None,
            emissive: [0.0; 3],
            emissive_texture: None,
            alpha_mode: AlphaMode::Opaque,
            double_sided: false,
        }
    }
}

/// Узел иерархии сцены.
#[derive(Debug, Clone)]
pub struct Node {
    pub name: String,
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    pub mesh: Option<usize>,
    pub skin: Option<usize>,
    pub children: Vec<usize>,
}

/// Скелет: узлы-кости.
#[derive(Debug, Clone)]
pub struct Skin {
    pub name: String,
    pub joints: Vec<usize>,
    /// Обратные матрицы привязки: переводят вершину из пространства меша в пространство кости.
    pub inverse_bind_matrices: Vec<engine_math::Mat4>,
}

/// Загруженный файл glTF.
#[derive(Debug, Clone, TypePath)]
pub struct Gltf {
    pub meshes: Vec<Handle<Mesh>>,
    pub materials: Vec<Handle<Material>>,
    pub images: Vec<Handle<Image>>,
    pub nodes: Vec<Node>,
    pub skins: Vec<Skin>,
    /// Корневые узлы каждой сцены.
    pub scenes: Vec<Vec<usize>>,
    pub default_scene: Option<usize>,
    /// Анимационные клипы (вложенные ресурсы `файл.glb#Animation0`).
    pub animations: Vec<Handle<AnimationClip>>,
}

impl Asset for Gltf {}

#[derive(Debug)]
pub struct GltfLoader;

impl AssetLoader for GltfLoader {
    type Asset = Gltf;

    fn extensions(&self) -> &[&'static str] {
        &["gltf", "glb"]
    }

    fn load<'a>(
        &'a self,
        bytes: Bytes,
        ctx: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<Gltf, AssetError>> {
        Box::pin(async move { load_gltf(&bytes, ctx).await })
    }
}

async fn load_gltf(bytes: &[u8], ctx: &mut LoadContext) -> Result<Gltf, AssetError> {
    let path = ctx.path().clone();
    let gltf::Gltf { document, blob } =
        gltf::Gltf::from_slice(bytes).map_err(|e| AssetError::format(&path, e))?;

    let buffers = read_buffers(&document, blob, ctx).await?;
    let images = load_images(&document, &buffers, ctx)?;
    let materials = read_materials(&document, &images)
        .into_iter()
        .enumerate()
        .map(|(index, material)| ctx.add_labeled(&format!("Material{index}"), material))
        .collect();
    let meshes = read_meshes(&document, &buffers, ctx)?;
    let nodes = read_nodes(&document);

    Ok(Gltf {
        meshes,
        materials,
        images,
        nodes,
        skins: document
            .skins()
            .map(|skin| {
                let reader = skin.reader(|buffer| buffers.get(buffer.index()).map(|b| &b[..]));
                Skin {
                    name: skin.name().unwrap_or_default().to_owned(),
                    joints: skin.joints().map(|j| j.index()).collect(),
                    inverse_bind_matrices: reader
                        .read_inverse_bind_matrices()
                        .map(|m| {
                            m.map(|c| engine_math::Mat4::from_cols_array_2d(&c))
                                .collect()
                        })
                        .unwrap_or_default(),
                }
            })
            .collect(),
        scenes: document
            .scenes()
            .map(|scene| scene.nodes().map(|n| n.index()).collect())
            .collect(),
        default_scene: document.default_scene().map(|s| s.index()),
        animations: read_animations(&document, &buffers)
            .into_iter()
            .enumerate()
            .map(|(index, clip)| ctx.add_labeled(&format!("Animation{index}"), clip))
            .collect(),
    })
}

/// Буферы: встроенный блок `.glb` или внешние `.bin`.
async fn read_buffers(
    document: &gltf::Document,
    blob: Option<Vec<u8>>,
    ctx: &mut LoadContext,
) -> Result<Vec<Bytes>, AssetError> {
    let mut buffers = Vec::with_capacity(document.buffers().len());
    for buffer in document.buffers() {
        let data = match buffer.source() {
            gltf::buffer::Source::Bin => blob
                .as_deref()
                .map(Bytes::from)
                .ok_or_else(|| AssetError::format(ctx.path(), "нет бинарного блока glb"))?,
            gltf::buffer::Source::Uri(uri) if uri.starts_with("data:") => {
                return Err(AssetError::format(
                    ctx.path(),
                    "встроенные data: URI не поддерживаются — используйте .glb",
                ));
            }
            gltf::buffer::Source::Uri(uri) => ctx.read(&decode_uri(uri)).await?,
        };
        buffers.push(data);
    }
    Ok(buffers)
}

/// Изображения: встроенные декодируются сразу, внешние грузятся отдельными ресурсами.
/// Цветовое пространство — по тому, как текстура используется в материалах.
fn load_images(
    document: &gltf::Document,
    buffers: &[Bytes],
    ctx: &mut LoadContext,
) -> Result<Vec<Handle<Image>>, AssetError> {
    let linear = linear_images(document);
    let mut images = Vec::new();
    for image in document.images() {
        let color_space = if linear.contains(&image.index()) {
            ColorSpace::Linear
        } else {
            ColorSpace::Srgb
        };
        let handle = match image.source() {
            gltf::image::Source::View { view, .. } => {
                let buffer = &buffers[view.buffer().index()];
                let data = &buffer[view.offset()..view.offset() + view.length()];
                let decoded = if super::ktx2::is_ktx2(data) {
                    Image::decode_ktx2(data, Some(color_space))
                } else {
                    Image::decode(data, color_space)
                }
                .map_err(|e| AssetError::format(ctx.path(), e))?;
                ctx.add_labeled(&format!("Image{}", image.index()), decoded)
            }
            // TODO(R-08): цветовое пространство для внешних текстур (сейчас всегда sRGB)
            gltf::image::Source::Uri { uri, .. } => ctx.load::<Image>(&decode_uri(uri)),
        };
        images.push(handle);
    }
    Ok(images)
}

fn read_materials(document: &gltf::Document, images: &[Handle<Image>]) -> Vec<Material> {
    let image = |texture: gltf::Texture<'_>| images[texture.source().index()].clone();
    document
        .materials()
        .map(|material| {
            let pbr = material.pbr_metallic_roughness();
            Material {
                name: material.name().unwrap_or_default().to_owned(),
                base_color: pbr.base_color_factor(),
                base_color_texture: pbr.base_color_texture().map(|t| image(t.texture())),
                metallic: pbr.metallic_factor(),
                roughness: pbr.roughness_factor(),
                metallic_roughness_texture: pbr
                    .metallic_roughness_texture()
                    .map(|t| image(t.texture())),
                normal_texture: material.normal_texture().map(|t| image(t.texture())),
                occlusion_texture: material.occlusion_texture().map(|t| image(t.texture())),
                emissive: material.emissive_factor(),
                emissive_texture: material.emissive_texture().map(|t| image(t.texture())),
                alpha_mode: match material.alpha_mode() {
                    gltf::material::AlphaMode::Opaque => AlphaMode::Opaque,
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    gltf::material::AlphaMode::Mask => AlphaMode::Mask(
                        (material.alpha_cutoff().unwrap_or(0.5).clamp(0.0, 1.0) * 255.0) as u8,
                    ),
                    gltf::material::AlphaMode::Blend => AlphaMode::Blend,
                },
                double_sided: material.double_sided(),
            }
        })
        .collect()
}

fn read_meshes(
    document: &gltf::Document,
    buffers: &[Bytes],
    ctx: &mut LoadContext,
) -> Result<Vec<Handle<Mesh>>, AssetError> {
    let mut meshes = Vec::new();
    for mesh in document.meshes() {
        let mut primitives = Vec::new();
        for primitive in mesh.primitives() {
            // [Rust] Замыкание даёт ридеру доступ к буферам по индексу
            let reader = primitive.reader(|buffer| buffers.get(buffer.index()).map(|b| &b[..]));
            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .ok_or_else(|| AssetError::format(ctx.path(), "у примитива нет позиций"))?
                .collect();
            let indices = reader.read_indices().map_or_else(
                || (0..u32::try_from(positions.len()).unwrap_or(u32::MAX)).collect(),
                |indices| indices.into_u32().collect(),
            );
            let bounds = Aabb::from_points(positions.iter().map(|p| Vec3::from_array(*p)));
            primitives.push(Primitive {
                bounds,
                normals: reader
                    .read_normals()
                    .map(Iterator::collect)
                    .unwrap_or_default(),
                tangents: reader
                    .read_tangents()
                    .map(Iterator::collect)
                    .unwrap_or_default(),
                uvs: reader
                    .read_tex_coords(0)
                    .map(|uv| uv.into_f32().collect())
                    .unwrap_or_default(),
                joints: reader
                    .read_joints(0)
                    .map(|j| j.into_u16().collect())
                    .unwrap_or_default(),
                weights: reader
                    .read_weights(0)
                    .map(|w| w.into_f32().collect())
                    .unwrap_or_default(),
                positions,
                indices,
                material: primitive.material().index(),
            });
        }
        let name = mesh.name().unwrap_or_default().to_owned();
        meshes.push(ctx.add_labeled(&format!("Mesh{}", mesh.index()), Mesh { name, primitives }));
    }
    Ok(meshes)
}

fn read_animations(document: &gltf::Document, buffers: &[Bytes]) -> Vec<AnimationClip> {
    use gltf::animation::util::ReadOutputs;
    document
        .animations()
        .map(|animation| {
            let mut duration = 0.0f32;
            let channels = animation
                .channels()
                .filter_map(|channel| {
                    let reader =
                        channel.reader(|buffer| buffers.get(buffer.index()).map(|b| &b[..]));
                    let times: Vec<f32> = reader.read_inputs()?.collect();
                    let keyframes = match reader.read_outputs()? {
                        ReadOutputs::Translations(v) => {
                            Keyframes::Translation(v.map(Vec3::from_array).collect())
                        }
                        ReadOutputs::Scales(v) => {
                            Keyframes::Scale(v.map(Vec3::from_array).collect())
                        }
                        ReadOutputs::Rotations(v) => {
                            Keyframes::Rotation(v.into_f32().map(Quat::from_array).collect())
                        }
                        // Морфинг (веса морф-таргетов) — позже
                        ReadOutputs::MorphTargetWeights(_) => return None,
                    };
                    duration = duration.max(times.last().copied().unwrap_or(0.0));
                    let interpolation = match channel.sampler().interpolation() {
                        gltf::animation::Interpolation::Step => Interpolation::Step,
                        gltf::animation::Interpolation::Linear => Interpolation::Linear,
                        gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
                    };
                    Some(Channel {
                        node: channel.target().node().index(),
                        times,
                        keyframes,
                        interpolation,
                    })
                })
                .collect();
            AnimationClip {
                name: animation.name().unwrap_or_default().to_owned(),
                duration,
                channels,
            }
        })
        .collect()
}

fn read_nodes(document: &gltf::Document) -> Vec<Node> {
    document
        .nodes()
        .map(|node| {
            let (translation, rotation, scale) = node.transform().decomposed();
            Node {
                name: node.name().unwrap_or_default().to_owned(),
                translation: Vec3::from_array(translation),
                rotation: Quat::from_array(rotation),
                scale: Vec3::from_array(scale),
                mesh: node.mesh().map(|m| m.index()),
                skin: node.skin().map(|s| s.index()),
                children: node.children().map(|c| c.index()).collect(),
            }
        })
        .collect()
}

/// Изображения, которые хранят данные, а не цвет: нормали, металличность, затенение.
fn linear_images(document: &gltf::Document) -> HashSet<usize> {
    let mut linear = HashSet::new();
    for material in document.materials() {
        let mut add = |texture: gltf::Texture<'_>| {
            linear.insert(texture.source().index());
        };
        if let Some(t) = material.normal_texture() {
            add(t.texture());
        }
        if let Some(t) = material.occlusion_texture() {
            add(t.texture());
        }
        if let Some(t) = material
            .pbr_metallic_roughness()
            .metallic_roughness_texture()
        {
            add(t.texture());
        }
    }
    linear
}

/// URI в glTF кодируются как URL: пробел — `%20`.
fn decode_uri(uri: &str) -> String {
    let mut out = Vec::with_capacity(uri.len());
    let bytes = uri.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%'
            && i + 2 < bytes.len()
            && let Ok(byte) = u8::from_str_radix(&uri[i + 1..i + 3], 16)
        {
            out.push(byte);
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Каждый треугольник обходится против часовой стрелки, если смотреть снаружи:
    /// векторное произведение рёбер смотрит туда же, куда нормаль.
    fn assert_outward(mesh: &Mesh) {
        let p = &mesh.primitives[0];
        for t in p.indices.chunks(3) {
            let [a, b, c] = [t[0], t[1], t[2]].map(|i| Vec3::from_array(p.positions[i as usize]));
            let face = (b - a).cross(c - a);
            if face.length() < 1e-6 {
                continue; // вырожденный треугольник у полюса сферы
            }
            let normal = Vec3::from_array(p.normals[t[0] as usize]);
            assert!(
                face.dot(normal) > 0.0,
                "{}: треугольник {t:?} вывернут",
                mesh.name
            );
        }
    }

    #[test]
    fn generated_meshes_face_outward() {
        let cube = Mesh::cuboid(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(cube.triangle_count(), 12);
        assert_eq!(cube.primitives[0].bounds.max, Vec3::new(1.0, 2.0, 3.0));
        assert_outward(&cube);
        let sphere = Mesh::sphere(0.5, 16, 8);
        assert_outward(&sphere);
        assert!((sphere.primitives[0].bounds.max.y - 0.5).abs() < 1e-5);
        assert_outward(&Mesh::plane(2.0));
    }
}
