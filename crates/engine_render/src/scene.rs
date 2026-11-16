//! GPU-ресурсы сцены и отрисовка PBR: меши, текстуры, материалы, пайплайны.

use std::collections::HashMap;
use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use engine_assets::formats::{AlphaMode, ColorSpace, Image, Material, Mesh};
use engine_assets::{Asset, AssetId};
use engine_math::{Aabb, Frustum, Mat4, Vec3, camera};
use wgpu::util::DeviceExt;

use crate::command::MaterialData;
use crate::lights::{LOCAL_SHADOW_LAYERS, LightSelection, MAX_LIGHTS};
use crate::shadows::{CASCADES, CameraView, Cascades, ShadowMap, ShadowSettings};
use crate::snapshot::{FrameSnapshot, interpolate};

pub(crate) const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// 256 матриц 4×4 — массив `Joints` в шейдере без storage.
const JOINTS_SIZE: u64 = (crate::extract::MAX_JOINTS * 64) as u64;
const MATRIX_SIZE: u64 = 64;

// ---------- данные для шейдера ----------
// [Rust] `#[repr(C)]` — раскладка полей как в C (без перестановок компилятором).
// `Pod` (plain old data) из bytemuck разрешает смотреть на структуру как на байты.
// Размеры и выравнивание совпадают с WGSL (std140): только vec4 и mat4.

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LightUniform {
    /// xyz — позиция, w — радиус действия.
    position_range: [f32; 4],
    /// rgb — цвет × интенсивность, w: 0 — точечный, 1 — прожектор.
    color_kind: [f32; 4],
    /// xyz — куда светит прожектор, w — косинус внешнего угла.
    direction_outer: [f32; 4],
    /// x — косинус внутреннего угла, y — первый слой тени (< 0 — без тени),
    /// z — размер текселя тени на 1 м расстояния.
    params: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    view_proj: [[f32; 4]; 4],
    camera_position: [f32; 4],
    ambient: [f32; 4],
    sun_direction: [f32; 4],
    sun_color: [f32; 4],
    counts: [u32; 4],
    lights: [LightUniform; MAX_LIGHTS],
    camera_forward: [f32; 4],
    cascades: [[[f32; 4]; 4]; CASCADES],
    cascade_splits: [f32; 4],
    cascade_texel: [f32; 4],
    /// x = 1 — тени есть, y = 1 — подсветить каскады.
    shadow_params: [f32; 4],
    /// «Проекция × вид» слоёв теней прожекторов и граней точечных источников.
    local_shadows: [[[f32; 4]; 4]; LOCAL_SHADOW_LAYERS],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MaterialUniform {
    base_color: [f32; 4],
    emissive: [f32; 4],
    params: [f32; 4],
}

/// Один экземпляр меша. Раскладка совпадает с атрибутами шейдера:
/// модель (0), нормали (64), начало костей (128, `x` — индекс матрицы).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct InstanceData {
    model: [[f32; 4]; 4],
    normal_matrix: [[f32; 4]; 4],
    joint_offset: [u32; 4],
}

impl InstanceData {
    fn from_model(model: Mat4, joint_offset: u32) -> Self {
        Self {
            model: model.to_cols_array_2d(),
            normal_matrix: model.inverse().transpose().to_cols_array_2d(),
            joint_offset: [joint_offset, 0, 0, 0],
        }
    }
}

/// Модель и нормали. Смещение костей не входит: обычный меш его не читает.
const PBR_STATIC_ATTRS: &[wgpu::VertexAttribute] = &wgpu::vertex_attr_array![
    5 => Float32x4,
    6 => Float32x4,
    7 => Float32x4,
    8 => Float32x4,
    9 => Float32x4,
    10 => Float32x4,
    11 => Float32x4,
    12 => Float32x4
];

/// То же плюс `joint_offset` на байте 128.
const PBR_SKINNED_ATTRS: &[wgpu::VertexAttribute] = &wgpu::vertex_attr_array![
    5 => Float32x4,
    6 => Float32x4,
    7 => Float32x4,
    8 => Float32x4,
    9 => Float32x4,
    10 => Float32x4,
    11 => Float32x4,
    12 => Float32x4,
    13 => Uint32x4
];

/// Тени: только модель. Нормали в шейдере теней нет.
const SHADOW_STATIC_ATTRS: &[wgpu::VertexAttribute] = &wgpu::vertex_attr_array![
    5 => Float32x4,
    6 => Float32x4,
    7 => Float32x4,
    8 => Float32x4
];

/// Тени скелета: модель и начало костей. Нормали между ними пропускаем — смещение вручную.
const SHADOW_SKINNED_ATTRS: &[wgpu::VertexAttribute] = &[
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 0,
        shader_location: 5,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 16,
        shader_location: 6,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 32,
        shader_location: 7,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Float32x4,
        offset: 48,
        shader_location: 8,
    },
    wgpu::VertexAttribute {
        format: wgpu::VertexFormat::Uint32x4,
        offset: 128,
        shader_location: 13,
    },
];

/// Вершина: позиция, нормаль, UV — 32 байта.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

// ---------- GPU-ресурсы ----------

struct GpuPrimitive {
    vertices: wgpu::Buffer,
    /// Индексы костей и веса — только у скелетных мешей.
    skin: Option<wgpu::Buffer>,
    indices: wgpu::Buffer,
    index_count: u32,
    bounds: Aabb,
}

impl GpuPrimitive {
    fn gpu_bytes(&self) -> u64 {
        self.vertices.size()
            + self.indices.size()
            + self.skin.as_ref().map_or(0, wgpu::Buffer::size)
    }
}

pub(crate) struct GpuTexture {
    pub(crate) view: wgpu::TextureView,
    pub(crate) bytes: u64,
}

struct GpuMaterial {
    data: MaterialData,
    uniform: wgpu::Buffer,
    /// Группа привязок и какие текстуры в ней были настоящими (остальные — заглушки).
    bind_group: Option<(wgpu::BindGroup, [bool; 5])>,
}

/// Как лежат кости: один storage на кадр или uniform на скелет.
#[derive(Clone, Copy, PartialEq, Eq)]
enum JointMode {
    Storage,
    Uniform,
}

impl JointMode {
    fn storage(self) -> bool {
        matches!(self, Self::Storage)
    }
}

/// Вариант PBR-пайплайна: 2 × 2 × 2 = 8 сочетаний.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct PipelineKind {
    blend: bool,
    double_sided: bool,
    skinned: bool,
}

/// Объект кадра. В список попадают и невидимые камере: тень может отбрасывать и то, что за кадром.
#[derive(Clone, Copy)]
struct DrawEntry {
    index: usize,
    kind: PipelineKind,
    /// Материал с альфа-маской — в тени нужны «дырки».
    masked: bool,
    distance: f32,
    /// Начало матриц костей: индекс в storage или номер слота uniform.
    joint_slot: Option<u32>,
    model: Mat4,
    bounds: Aabb,
    /// Прошёл отсечение по фрустуму камеры.
    in_view: bool,
}

/// Ключ склейки: один вызов на одинаковые меш, примитив, материал и пайплайн.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct BatchKey {
    kind: PipelineKind,
    masked: bool,
    mesh: AssetId<Mesh>,
    primitive: u32,
    material: Option<AssetId<Material>>,
}

/// Кусок вершинного буфера экземпляров.
struct DrawBatch {
    /// Индекс в `entries` — по нему берутся меш и материал.
    entry: u32,
    first: u32,
    count: u32,
}

/// Индексы четырёх костей и их веса — 24 байта на вершину.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SkinVertex {
    joints: [u16; 4],
    weights: [f32; 4],
}

/// Статистика кадра.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct DrawStats {
    pub drawn: u32,
    pub culled: u32,
    pub missing: u32,
    /// Вызовы `draw` прохода сцены. Тени сюда не входят.
    pub draw_calls: u32,
}

pub(crate) struct SceneRenderer {
    frame_layout: wgpu::BindGroupLayout,
    material_layout: wgpu::BindGroupLayout,
    joints_layout: wgpu::BindGroupLayout,
    /// Кости всех скелетов — один storage-буфер, экземпляр хранит индекс первой матрицы.
    /// Без vertex storage остаётся uniform на 256 костей и отдельный вызов на скелет.
    joints: JointMode,
    joint_buffer: wgpu::Buffer,
    joint_bind_group: wgpu::BindGroup,
    joint_capacity: u64,
    joint_stride: u64,
    joint_data: Vec<u8>,
    /// Начало скелета в снимке → индекс матрицы (storage) или номер слота (uniform).
    joint_slots: HashMap<u32, u32>,
    pipelines: HashMap<PipelineKind, wgpu::RenderPipeline>,
    frame_buffer: wgpu::Buffer,
    frame_bind_group: wgpu::BindGroup,
    instance_data: Vec<InstanceData>,
    instance_buffer: Option<(wgpu::Buffer, u64)>,
    camera_batches: Vec<DrawBatch>,
    shadow_batches: Vec<Vec<DrawBatch>>,
    sampler: wgpu::Sampler,
    /// Заглушки: белая (sRGB и линейная), «плоская» нормаль.
    white_srgb: wgpu::TextureView,
    white_linear: wgpu::TextureView,
    flat_normal: wgpu::TextureView,
    default_material: GpuMaterial,
    meshes: HashMap<AssetId<Mesh>, Vec<GpuPrimitive>>,
    textures: HashMap<AssetId<Image>, GpuTexture>,
    materials: HashMap<AssetId<Material>, GpuMaterial>,
    manual_srgb: bool,
    /// Объекты кадра (см. [`DrawEntry`]).
    entries: Vec<DrawEntry>,
    shadows: ShadowResources,
    /// Каскады этого кадра; `None` — теней нет (выключены или нет солнца).
    cascades: Option<Cascades>,
    shadow_debug: bool,
    /// Локальные источники кадра и их слои теней.
    lights: LightSelection,
    line_pipeline: wgpu::RenderPipeline,
    color_format: wgpu::TextureFormat,
    /// Рисуем в HDR-текстуру (тональная компрессия — отдельным проходом).
    hdr: bool,
    line_buffer: Option<(wgpu::Buffer, u64)>,
    line_count: u32,
}

/// Всё для прохода теней: карта, матрицы каскадов (по смещению в буфере), пайплайны.
struct ShadowResources {
    /// Каскады солнца.
    map: ShadowMap,
    /// Прожекторы и грани точечных источников.
    local: ShadowMap,
    sampler: wgpu::Sampler,
    view_layout: wgpu::BindGroupLayout,
    view_buffer: wgpu::Buffer,
    view_bind_group: wgpu::BindGroup,
    view_stride: u64,
    /// Ключ: (скелетный, альфа-маска).
    pipelines: HashMap<(bool, bool), wgpu::RenderPipeline>,
}

/// Куда рисовать кадр.
pub(crate) struct FrameTarget<'a> {
    pub color: &'a wgpu::TextureView,
    pub depth: &'a wgpu::TextureView,
    /// Метки времени GPU для прохода (если поддерживаются).
    pub timestamps: Option<wgpu::RenderPassTimestampWrites<'a>>,
    /// Сохранить глубину после прохода: её читают текст в мире, частицы и декали.
    /// Иначе она выбрасывается — на мобильных GPU запись из тайловой памяти стоит заметно.
    pub keep_depth: bool,
}

/// Матрица вид-проекция камеры кадра — для проходов вне сцены (текст).
pub(crate) fn camera_view_proj(snapshot: &FrameSnapshot, size: (u32, u32), alpha: f32) -> Mat4 {
    #[allow(clippy::cast_precision_loss)]
    let aspect = size.0 as f32 / size.1.max(1) as f32;
    frame_camera(snapshot, aspect, alpha).view_proj()
}

/// Мировая матрица камеры кадра. Без камеры — вид издалека на начало координат.
/// Нужна проходам, которым мало одной матрицы вид-проекция (билборд смотрит осями камеры).
pub(crate) fn camera_world(snapshot: &FrameSnapshot, alpha: f32) -> Mat4 {
    snapshot.camera.map_or_else(
        || camera::look_at(Vec3::new(0.0, 1.0, 5.0), Vec3::ZERO, Vec3::Y).inverse(),
        |camera| interpolate(&camera.prev, &camera.curr, alpha),
    )
}

/// Камера кадра (интерполированная). Без камеры — вид издалека на начало координат.
fn frame_camera(snapshot: &FrameSnapshot, aspect: f32, alpha: f32) -> CameraView {
    let camera = snapshot.camera.as_ref();
    CameraView {
        world: camera_world(snapshot, alpha),
        fov_y: camera.map_or(0.9, |c| c.fov_y),
        aspect,
        near: camera.map_or(0.05, |c| c.near),
        far: camera.map_or(1000.0, |c| c.far),
    }
}

impl CameraView {
    fn view_proj(&self) -> Mat4 {
        camera::perspective(self.fov_y, self.aspect, self.near, self.far) * self.world.inverse()
    }

    fn position(&self) -> Vec3 {
        self.world.w_axis.truncate()
    }

    /// Направление взгляда (−Z камеры).
    fn forward(&self) -> Vec3 {
        -self.world.z_axis.truncate().normalize_or_zero()
    }
}

impl SceneRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        hdr: bool,
        joint_storage: bool,
        sources: &HashMap<String, String>,
    ) -> Self {
        let _zone = tracing::trace_span!("scene_renderer_new").entered();
        let joints = if joint_storage {
            JointMode::Storage
        } else {
            JointMode::Uniform
        };
        let (frame_layout, material_layout, joints_layout) = create_layouts(device, joint_storage);

        // Смещение uniform обязано быть кратно min_uniform_buffer_offset_alignment (обычно 256).
        let alignment = u64::from(device.limits().min_uniform_buffer_offset_alignment);

        let frame_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame"),
            size: std::mem::size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let defaults = ShadowSettings::default();
        let shadows = ShadowResources::new(device, alignment, defaults.size, defaults.local_size);
        let frame_bind_group =
            create_frame_bind_group(device, &frame_layout, &frame_buffer, &shadows);
        // Скелет без storage — 256 матриц (16 КБ). Со storage ёмкость — число матриц.
        let joint_stride = JOINTS_SIZE.div_ceil(alignment) * alignment;
        let joint_capacity = if joint_storage { 256 } else { 4 };
        let (joint_buffer, joint_bind_group) = if joint_storage {
            create_joint_storage(device, &joints_layout, joint_capacity * MATRIX_SIZE)
        } else {
            create_object_buffer(
                device,
                &joints_layout,
                joint_stride,
                joint_capacity,
                JOINTS_SIZE,
            )
        };

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("material"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let [white_srgb, white_linear, flat_normal] = fallback_textures(device, queue);

        let layouts = [
            &frame_layout,
            &material_layout,
            &joints_layout,
            &shadows.view_layout,
        ];
        // Встроенные шейдеры обязаны собираться — иначе это ошибка в самом движке
        let (pipelines, line_pipeline, shadow_pipelines) =
            build_pipelines(device, layouts, color_format, hdr, joint_storage, sources)
                .unwrap_or_else(|error| panic!("шейдеры движка не собрались: {error}"));
        let mut shadows = shadows;
        shadows.pipelines = shadow_pipelines;
        let default_material = GpuMaterial::new(device, MaterialData::from(&Material::default()));
        Self {
            frame_layout,
            material_layout,
            joints_layout,
            joints,
            joint_buffer,
            joint_bind_group,
            joint_capacity,
            joint_stride,
            joint_data: Vec::new(),
            joint_slots: HashMap::new(),
            pipelines,
            frame_buffer,
            frame_bind_group,
            instance_data: Vec::new(),
            instance_buffer: None,
            camera_batches: Vec::new(),
            shadow_batches: Vec::new(),
            sampler,
            white_srgb,
            white_linear,
            flat_normal,
            default_material,
            meshes: HashMap::new(),
            textures: HashMap::new(),
            materials: HashMap::new(),
            // В HDR кодирование делает проход тональной компрессии
            manual_srgb: !hdr && !color_format.is_srgb(),
            entries: Vec::new(),
            shadows,
            cascades: None,
            shadow_debug: false,
            lights: LightSelection::default(),
            line_pipeline,
            color_format,
            hdr,
            line_buffer: None,
            line_count: 0,
        }
    }

    /// Пересобирает пайплайны из новых исходников шейдеров. При ошибке старые остаются.
    pub fn rebuild_shaders(
        &mut self,
        device: &wgpu::Device,
        sources: &HashMap<String, String>,
    ) -> Result<(), String> {
        let layouts = [
            &self.frame_layout,
            &self.material_layout,
            &self.joints_layout,
            &self.shadows.view_layout,
        ];
        let (pipelines, line_pipeline, shadow_pipelines) = build_pipelines(
            device,
            layouts,
            self.color_format,
            self.hdr,
            self.joints.storage(),
            sources,
        )?;
        self.pipelines = pipelines;
        self.line_pipeline = line_pipeline;
        self.shadows.pipelines = shadow_pipelines;
        Ok(())
    }

    pub fn upload_mesh(&mut self, device: &wgpu::Device, id: AssetId<Mesh>, mesh: &Mesh) {
        let _zone = tracing::trace_span!("upload_mesh").entered();
        let primitives = mesh
            .primitives
            .iter()
            .map(|p| {
                let vertices: Vec<Vertex> = (0..p.positions.len())
                    .map(|i| Vertex {
                        position: p.positions[i],
                        normal: p.normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
                        uv: p.uvs.get(i).copied().unwrap_or_default(),
                    })
                    .collect();
                GpuPrimitive {
                    vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("vertices"),
                        contents: bytemuck::cast_slice(&vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
                    indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("indices"),
                        contents: bytemuck::cast_slice(&p.indices),
                        usage: wgpu::BufferUsages::INDEX,
                    }),
                    skin: (!p.joints.is_empty() && p.joints.len() == p.positions.len()).then(
                        || {
                            let skin: Vec<SkinVertex> = p
                                .joints
                                .iter()
                                .zip(
                                    p.weights
                                        .iter()
                                        .chain(std::iter::repeat(&[1.0, 0.0, 0.0, 0.0])),
                                )
                                .map(|(joints, weights)| SkinVertex {
                                    joints: *joints,
                                    weights: *weights,
                                })
                                .collect();
                            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("skin"),
                                contents: bytemuck::cast_slice(&skin),
                                usage: wgpu::BufferUsages::VERTEX,
                            })
                        },
                    ),
                    index_count: u32::try_from(p.indices.len()).unwrap_or(u32::MAX),
                    bounds: p.bounds,
                }
            })
            .collect();
        self.meshes.insert(id, primitives);
    }

    pub fn upload_image(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: AssetId<Image>,
        image: &Image,
    ) {
        let _zone = tracing::trace_span!("upload_image").entered();
        self.textures
            .insert(id, create_texture(device, queue, image));
    }

    pub fn upload_material(
        &mut self,
        device: &wgpu::Device,
        id: AssetId<Material>,
        data: MaterialData,
    ) {
        self.materials.insert(id, GpuMaterial::new(device, data));
    }

    pub fn remove_mesh(&mut self, id: AssetId<Mesh>) {
        self.meshes.remove(&id);
    }
    pub fn remove_image(&mut self, id: AssetId<Image>) {
        self.textures.remove(&id);
    }

    pub(crate) fn image_view(&self, id: AssetId<Image>) -> Option<&wgpu::TextureView> {
        self.textures.get(&id).map(|texture| &texture.view)
    }
    pub fn remove_material(&mut self, id: AssetId<Material>) {
        self.materials.remove(&id);
    }

    pub fn counts(&self) -> (usize, usize, usize) {
        (self.meshes.len(), self.textures.len(), self.materials.len())
    }

    /// Видеопамять мешей, текстур и карт теней. Цели кадра и постэффекты сюда не входят:
    /// они живут один кадр в пуле графа.
    pub fn gpu_bytes(&self) -> u64 {
        let meshes: u64 = self
            .meshes
            .values()
            .flat_map(|primitives| primitives.iter().map(GpuPrimitive::gpu_bytes))
            .sum();
        let textures: u64 = self.textures.values().map(|texture| texture.bytes).sum();
        meshes + textures + self.shadows.map.bytes + self.shadows.local.bytes
    }

    /// Рисует сцену. `alpha` — коэффициент интерполяции между шагами логики.
    /// Подготовка кадра до проходов: отсечение, матрицы объектов и костей, каскады теней,
    /// данные кадра, материалы. Потом — [`Self::render_shadows`] и [`Self::render`].
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        size: (u32, u32),
        snapshot: &FrameSnapshot,
        alpha: f32,
    ) -> DrawStats {
        let _zone = tracing::trace_span!("prepare_scene").entered();
        let mut stats = DrawStats::default();
        #[allow(clippy::cast_precision_loss)]
        let aspect = size.0 as f32 / size.1.max(1) as f32;
        let camera = frame_camera(snapshot, aspect, alpha);
        let view_proj = camera.view_proj();
        let camera_position = camera.position();

        self.prepare_shadows(device, queue, snapshot, &camera, &view_proj);
        self.cull_and_collect(snapshot, alpha, &view_proj, camera_position, &mut stats);
        self.build_batches(snapshot);
        self.upload_instances(device, queue);
        let joint_count = if self.joints.storage() {
            u64::try_from(self.joint_data.len() / usize::try_from(MATRIX_SIZE).unwrap_or(64))
                .unwrap_or(0)
        } else {
            u64::try_from(self.joint_slots.len()).unwrap_or(0)
        };
        self.ensure_joint_capacity(device, joint_count);
        if !self.joint_data.is_empty() {
            queue.write_buffer(&self.joint_buffer, 0, &self.joint_data);
        }
        let frame = self.frame_uniform(snapshot, &view_proj, &camera);
        queue.write_buffer(&self.frame_buffer, 0, bytemuck::bytes_of(&frame));
        self.prepare_materials(device, snapshot);
        self.upload_lines(device, queue, snapshot);
        stats
    }

    /// Каскады теней кадра; пересоздаёт карту, если сменился размер.
    /// Каскады солнца и отбор локальных источников; пересоздаёт карты при смене размера.
    /// Матрицы всех слоёв — в буфер видов: 0..4 — каскады, 4.. — локальные слои.
    fn prepare_shadows(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        snapshot: &FrameSnapshot,
        camera: &CameraView,
        view_proj: &Mat4,
    ) {
        let settings = snapshot.shadows;
        self.shadow_debug = settings.debug;
        let max_size = device.limits().max_texture_dimension_2d;
        let (sun_size, local_size) = (
            settings.size.min(max_size),
            settings.local_size.min(max_size),
        );
        if self.shadows.map.size != sun_size || self.shadows.local.size != local_size {
            if self.shadows.map.size != sun_size {
                self.shadows.map = ShadowMap::new(device, sun_size, CASCADES, "shadow_sun");
            }
            if self.shadows.local.size != local_size {
                self.shadows.local =
                    ShadowMap::new(device, local_size, LOCAL_SHADOW_LAYERS, "shadow_local");
            }
            // Группа кадра ссылается на старые текстуры — пересоздаём
            self.frame_bind_group = create_frame_bind_group(
                device,
                &self.frame_layout,
                &self.frame_buffer,
                &self.shadows,
            );
        }

        self.cascades = snapshot
            .lights
            .sun
            .filter(|sun| sun.shadows && settings.enabled)
            .map(|sun| {
                crate::shadows::compute_cascades(camera, sun.direction, settings.distance, sun_size)
            });
        self.lights = crate::lights::select_lights(
            &snapshot.lights.local,
            &Frustum::from_view_projection(view_proj),
            camera.position(),
            settings.enabled,
            local_size,
        );

        let stride = usize::try_from(self.shadows.view_stride).unwrap_or(256);
        let mut data = vec![0u8; stride * (CASCADES + LOCAL_SHADOW_LAYERS)];
        let cascades = self.cascades.map(|c| c.view_proj).unwrap_or_default();
        for (slot, matrix) in cascades.iter().chain(&self.lights.shadow_views).enumerate() {
            let offset = slot * stride;
            data[offset..offset + 64]
                .copy_from_slice(bytemuck::bytes_of(&matrix.to_cols_array_2d()));
        }
        queue.write_buffer(&self.shadows.view_buffer, 0, &data);
    }

    /// Проход теней: по проходу на каждый слой (каскады солнца, прожекторы, грани точечных),
    /// в каждый — объекты внутри его фрустума. Прозрачные (blend) тень не отбрасывают.
    pub fn render_shadows(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        snapshot: &FrameSnapshot,
        timestamps: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        // (матрица, слой-цель, слот в буфере видов)
        let mut layers: Vec<(&Mat4, &wgpu::TextureView, usize)> = Vec::new();
        if let Some(cascades) = &self.cascades {
            for (i, view) in self.shadows.map.layer_views.iter().enumerate() {
                layers.push((&cascades.view_proj[i], view, i));
            }
        }
        for (i, matrix) in self.lights.shadow_views.iter().enumerate() {
            layers.push((matrix, &self.shadows.local.layer_views[i], CASCADES + i));
        }
        if layers.is_empty() {
            return;
        }
        let _zone = tracing::trace_span!("render_shadows").entered();
        let count = layers.len();
        for (index, (_matrix, view, slot)) in layers.into_iter().enumerate() {
            // Метка начала — в первом проходе, конца — в последнем: время всех слоёв.
            // Средним проходам метки не передаём вовсе (пустые wgpu не принимает).
            let (first, last) = (index == 0, index == count - 1);
            let timestamps = timestamps.as_ref().filter(|_| first || last).map(|t| {
                wgpu::RenderPassTimestampWrites {
                    query_set: t.query_set,
                    beginning_of_pass_write_index: t
                        .beginning_of_pass_write_index
                        .filter(|_| first),
                    end_of_pass_write_index: t.end_of_pass_write_index.filter(|_| last),
                }
            });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow_layer"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store, // карта нужна проходу сцены
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: timestamps,
                ..Default::default()
            });
            let view_offset = u32::try_from(slot as u64 * self.shadows.view_stride).unwrap_or(0);
            pass.set_bind_group(0, &self.shadows.view_bind_group, &[view_offset]);
            if let Some(batches) = self.shadow_batches.get(index) {
                self.draw_batches(&mut pass, snapshot, batches, true, None);
            }
        }
    }

    /// Проход сцены (после [`Self::prepare`] и теней).
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &FrameTarget<'_>,
        snapshot: &FrameSnapshot,
        alpha: f32,
        stats: &mut DrawStats,
    ) {
        let _zone = tracing::trace_span!("render_scene").entered();
        self.draw(encoder, target, snapshot, alpha, stats);
    }

    /// Отсечение по фрустуму. Матрицы экземпляров собирает [`Self::build_batches`].
    fn cull_and_collect(
        &mut self,
        snapshot: &FrameSnapshot,
        alpha: f32,
        view_proj: &Mat4,
        camera_position: Vec3,
        stats: &mut DrawStats,
    ) {
        let _zone = tracing::trace_span!("cull").entered();
        let frustum = Frustum::from_view_projection(view_proj);
        self.entries.clear();
        self.joint_data.clear();
        self.joint_slots.clear();
        for (index, draw) in snapshot.draws.iter().enumerate() {
            let Some(primitive) = self
                .meshes
                .get(&draw.mesh)
                .and_then(|p| p.get(draw.primitive as usize))
            else {
                stats.missing += 1;
                continue;
            };
            let model = interpolate(&draw.prev, &draw.curr, alpha);
            let skin = draw.skin.filter(|_| primitive.skin.is_some());
            let mut world_bounds = primitive.bounds.transformed(&model);
            if skin.is_some() {
                // Анимация выводит конечности за AABB позы привязки — берём с запасом
                let (center, half) = (world_bounds.center(), world_bounds.half_extents() * 1.5);
                world_bounds = engine_math::Aabb {
                    min: center - half,
                    max: center + half,
                };
            }
            let in_view = frustum.intersects_aabb(&world_bounds);
            if !in_view {
                stats.culled += 1;
                // Без теней невидимый объект не нужен вовсе
                if self.cascades.is_none() {
                    continue;
                }
            }
            let material = draw
                .material
                .and_then(|id| self.materials.get(&id))
                .map_or(&self.default_material.data, |m| &m.data);
            let kind = PipelineKind {
                blend: material.alpha_mode == AlphaMode::Blend,
                double_sided: material.double_sided,
                skinned: skin.is_some(),
            };
            // [Rust] Считаем сразу: `material` — ссылка внутрь `self.materials`, и пока она
            // жива, вызвать `self.joint_slot(..)` (нужен `&mut self`) borrow checker не даст
            let masked = matches!(material.alpha_mode, AlphaMode::Mask(_));
            // Скелетный меш: вершины уже в мире после скиннинга, матрица модели — единичная
            let model = if skin.is_some() {
                Mat4::IDENTITY
            } else {
                model
            };
            let joint_slot = skin.map(|(start, count)| self.joint_slot(snapshot, start, count));
            let distance = (world_bounds.center() - camera_position).length_squared();
            self.entries.push(DrawEntry {
                index,
                kind,
                masked,
                distance,
                joint_slot,
                model,
                bounds: world_bounds,
                in_view,
            });
        }
    }

    /// Слот матриц скелета: один на скелет, даже если у меша несколько примитивов.
    /// Со storage матрицы лежат подряд. Без него слот дополняется до размера uniform.
    fn joint_slot(&mut self, snapshot: &FrameSnapshot, start: u32, count: u32) -> u32 {
        if let Some(&slot) = self.joint_slots.get(&start) {
            return slot;
        }
        let matrices = &snapshot.joints[start as usize..(start + count) as usize];
        if self.joints.storage() {
            let matrices_before =
                self.joint_data.len() / usize::try_from(MATRIX_SIZE).unwrap_or(64);
            let offset = u32::try_from(matrices_before).unwrap_or(0);
            self.joint_slots.insert(start, offset);
            for matrix in matrices {
                self.joint_data
                    .extend_from_slice(bytemuck::bytes_of(&matrix.to_cols_array_2d()));
            }
            return offset;
        }
        let slot = u32::try_from(self.joint_slots.len()).unwrap_or(0);
        self.joint_slots.insert(start, slot);
        let begin = self.joint_data.len();
        for matrix in matrices {
            self.joint_data
                .extend_from_slice(bytemuck::bytes_of(&matrix.to_cols_array_2d()));
        }
        self.joint_data.resize(
            begin + usize::try_from(self.joint_stride).unwrap_or(16384),
            0,
        );
        slot
    }

    /// Непрозрачные с одним мешем и материалом — один вызов. Прозрачные по одному:
    /// порядок «дальние раньше» нельзя отдать одному вызову. Скелеты без storage тоже
    /// по одному: у каждого свой uniform костей.
    fn build_batches(&mut self, snapshot: &FrameSnapshot) {
        let mut opaque = Vec::new();
        let mut transparent = Vec::new();
        for (index, entry) in self.entries.iter().enumerate() {
            if !entry.in_view {
                continue;
            }
            if entry.kind.blend {
                transparent.push(index);
            } else {
                opaque.push(index);
            }
        }
        opaque.sort_by_key(|&index| self.batch_key(snapshot, index));
        transparent.sort_by(|&a, &b| {
            self.entries[b]
                .distance
                .total_cmp(&self.entries[a].distance)
        });

        self.instance_data.clear();
        let mut camera = self.push_batches(&opaque, snapshot);
        camera.append(&mut self.push_batches(&transparent, snapshot));
        self.camera_batches = camera;

        let layers = self.shadow_layer_matrices();
        self.shadow_batches.clear();
        for matrix in &layers {
            let frustum = Frustum::from_view_projection(matrix);
            let mut indices = Vec::new();
            for (index, entry) in self.entries.iter().enumerate() {
                if entry.kind.blend || !frustum.intersects_aabb(&entry.bounds) {
                    continue;
                }
                indices.push(index);
            }
            indices.sort_by_key(|&index| self.batch_key(snapshot, index));
            let batches = self.push_batches(&indices, snapshot);
            self.shadow_batches.push(batches);
        }
    }

    fn shadow_layer_matrices(&self) -> Vec<Mat4> {
        let mut layers = Vec::new();
        if let Some(cascades) = &self.cascades {
            layers.extend(cascades.view_proj);
        }
        layers.extend(self.lights.shadow_views.iter().copied());
        layers
    }

    fn batch_key(&self, snapshot: &FrameSnapshot, index: usize) -> BatchKey {
        let entry = &self.entries[index];
        let draw = &snapshot.draws[entry.index];
        BatchKey {
            kind: entry.kind,
            masked: entry.masked,
            mesh: draw.mesh,
            primitive: draw.primitive,
            material: draw.material,
        }
    }

    /// Скелет без общего буфера костей не склеивается: смещение uniform одно на вызов.
    fn can_merge(&self, index: usize) -> bool {
        let entry = &self.entries[index];
        !entry.kind.blend && (!entry.kind.skinned || self.joints.storage())
    }

    fn push_batches(&mut self, indices: &[usize], snapshot: &FrameSnapshot) -> Vec<DrawBatch> {
        let ranges = group_runs(indices.len(), |i| {
            let prev = indices[i - 1];
            let curr = indices[i];
            self.can_merge(prev) && self.batch_key(snapshot, prev) == self.batch_key(snapshot, curr)
        });
        ranges
            .into_iter()
            .map(|range| {
                let first = u32::try_from(self.instance_data.len()).unwrap_or(u32::MAX);
                let entry = u32::try_from(indices[range.start]).unwrap_or(0);
                let count = u32::try_from(range.len()).unwrap_or(0);
                for &index in &indices[range] {
                    let item = &self.entries[index];
                    self.instance_data.push(InstanceData::from_model(
                        item.model,
                        item.joint_slot.unwrap_or(0),
                    ));
                }
                DrawBatch {
                    entry,
                    first,
                    count,
                }
            })
            .collect()
    }

    fn upload_instances(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.instance_data.is_empty() {
            return;
        }
        let bytes = bytemuck::cast_slice(&self.instance_data);
        let needed = bytes.len() as u64;
        if self
            .instance_buffer
            .as_ref()
            .is_none_or(|(_, capacity)| *capacity < needed)
        {
            let capacity = needed.next_power_of_two();
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instances"),
                size: capacity,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instance_buffer = Some((buffer, capacity));
        }
        if let Some((buffer, _)) = &self.instance_buffer {
            queue.write_buffer(buffer, 0, bytes);
        }
    }

    /// Группы привязок материалов: создаются при первой нужде и пересоздаются,
    /// когда догрузились текстуры.
    fn prepare_materials(&mut self, device: &wgpu::Device, snapshot: &FrameSnapshot) {
        // [Rust] Обход по индексу, а не `for x in &self.entries`: ссылка на `self.entries`
        // жила бы весь цикл, а `prepare_material` требует `&mut self` — borrow checker
        // запрещает такое пересечение (в C++ это была бы потенциальная инвалидация итератора).
        for slot in 0..self.entries.len() {
            let index = self.entries[slot].index;
            if let Some(id) = snapshot.draws[index].material {
                self.prepare_material(device, id);
            }
        }
        if self.default_material.bind_group.is_none() {
            let bind_group = self.material_bind_group(
                device,
                &self.default_material.data,
                &self.default_material.uniform,
            );
            self.default_material.bind_group = Some((bind_group, [false; 5]));
        }
    }

    fn draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &FrameTarget<'_>,
        snapshot: &FrameSnapshot,
        alpha: f32,
        stats: &mut DrawStats,
    ) {
        let mut clear = snapshot.interpolated_clear_color(alpha).to_wgpu();
        if self.manual_srgb {
            // Цвет очистки не проходит через шейдер: на поверхности без sRGB (Web, GLES)
            // кодируем его сами, как шейдер кодирует пиксели
            let encode = |c: f64| {
                if c <= 0.003_130_8 {
                    c * 12.92
                } else {
                    1.055 * c.powf(1.0 / 2.4) - 0.055
                }
            };
            clear = wgpu::Color {
                r: encode(clear.r),
                g: encode(clear.g),
                b: encode(clear.b),
                a: clear.a,
            };
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target.color,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: target.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: if target.keep_depth {
                        wgpu::StoreOp::Store
                    } else {
                        wgpu::StoreOp::Discard
                    },
                }),
                stencil_ops: None,
            }),
            timestamp_writes: target.timestamps.clone(),
            ..Default::default()
        });
        pass.set_bind_group(0, &self.frame_bind_group, &[]);
        self.draw_batches(
            &mut pass,
            snapshot,
            &self.camera_batches,
            false,
            Some(stats),
        );
        // Отладочные линии — поверх сцены, с проверкой глубины
        if let Some((buffer, _)) = &self.line_buffer
            && self.line_count > 0
        {
            pass.set_pipeline(&self.line_pipeline);
            pass.set_vertex_buffer(0, buffer.slice(..));
            pass.draw(0..self.line_count, 0..1);
        }
    }

    /// Копирует отладочные линии в GPU. Буфер растёт по мере надобности.
    fn upload_lines(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        snapshot: &FrameSnapshot,
    ) {
        self.line_count = u32::try_from(snapshot.lines.len()).unwrap_or(0);
        if snapshot.lines.is_empty() {
            return;
        }
        let bytes: &[u8] = bytemuck::cast_slice(&snapshot.lines);
        let needed = bytes.len() as u64;
        if self
            .line_buffer
            .as_ref()
            .is_none_or(|(_, capacity)| *capacity < needed)
        {
            let capacity = needed.next_power_of_two();
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("debug_lines"),
                size: capacity,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.line_buffer = Some((buffer, capacity));
        }
        if let Some((buffer, _)) = &self.line_buffer {
            queue.write_buffer(buffer, 0, bytes);
        }
    }

    fn frame_uniform(
        &self,
        snapshot: &FrameSnapshot,
        view_proj: &Mat4,
        camera: &CameraView,
    ) -> FrameUniform {
        let camera_position = camera.position();
        let lights = &snapshot.lights;
        let mut light_data = [LightUniform::zeroed(); MAX_LIGHTS];
        for (slot, selected) in light_data.iter_mut().zip(&self.lights.lights) {
            let light = &selected.light;
            let (kind, direction, cos_inner, cos_outer) = light.spot.map_or(
                (0.0, Vec3::ZERO, 1.0, 1.0),
                // smoothstep в шейдере не определён при равных границах — разводим их
                |spot| {
                    let cos_outer = spot.outer.cos();
                    (
                        1.0,
                        spot.direction,
                        spot.inner.cos().max(cos_outer + 1e-4),
                        cos_outer,
                    )
                },
            );
            // [Rust] `as f32` из u32 точен до 2^24 — слоёв единицы
            #[allow(clippy::cast_precision_loss)]
            let shadow_layer = selected.shadow_layer.map_or(-1.0, |layer| layer as f32);
            let p = light.position;
            *slot = LightUniform {
                position_range: [p.x, p.y, p.z, light.range],
                color_kind: [light.color[0], light.color[1], light.color[2], kind],
                direction_outer: [direction.x, direction.y, direction.z, cos_outer],
                params: [cos_inner, shadow_layer, selected.texel_per_meter, 0.0],
            };
        }
        let mut local_shadows = [Mat4::IDENTITY.to_cols_array_2d(); LOCAL_SHADOW_LAYERS];
        for (slot, matrix) in local_shadows.iter_mut().zip(&self.lights.shadow_views) {
            *slot = matrix.to_cols_array_2d();
        }
        let (sun_direction, sun_color) = lights.sun.map_or(([0.0; 4], [0.0; 4]), |sun| {
            let (dir, color) = (sun.direction, sun.color);
            (
                [dir.x, dir.y, dir.z, 1.0],
                [color[0], color[1], color[2], 0.0],
            )
        });
        let cascades = self.cascades.unwrap_or(Cascades {
            view_proj: [Mat4::IDENTITY; CASCADES],
            splits: [0.0; CASCADES],
            texel: [0.0; CASCADES],
        });
        FrameUniform {
            view_proj: view_proj.to_cols_array_2d(),
            camera_position: camera_position.extend(1.0).to_array(),
            ambient: [lights.ambient[0], lights.ambient[1], lights.ambient[2], 0.0],
            sun_direction,
            sun_color,
            counts: [
                u32::try_from(self.lights.lights.len()).unwrap_or(0),
                u32::from(self.manual_srgb),
                0,
                0,
            ],
            lights: light_data,
            camera_forward: camera.forward().extend(0.0).to_array(),
            cascades: cascades.view_proj.map(|m| m.to_cols_array_2d()),
            cascade_splits: cascades.splits,
            cascade_texel: cascades.texel,
            shadow_params: [
                if self.cascades.is_some() { 1.0 } else { 0.0 },
                if self.shadow_debug { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
            local_shadows,
        }
    }

    fn ensure_joint_capacity(&mut self, device: &wgpu::Device, count: u64) {
        if count <= self.joint_capacity {
            return;
        }
        self.joint_capacity = count.next_power_of_two();
        let (buffer, bind_group) = if self.joints.storage() {
            create_joint_storage(
                device,
                &self.joints_layout,
                self.joint_capacity * MATRIX_SIZE,
            )
        } else {
            create_object_buffer(
                device,
                &self.joints_layout,
                self.joint_stride,
                self.joint_capacity,
                JOINTS_SIZE,
            )
        };
        self.joint_buffer = buffer;
        self.joint_bind_group = bind_group;
    }

    /// `shadow` — пайплайны карты теней. Статистика считается только для прохода сцены.
    fn draw_batches<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        snapshot: &FrameSnapshot,
        batches: &[DrawBatch],
        shadow: bool,
        mut stats: Option<&mut DrawStats>,
    ) {
        // [Rust] `Buffer` — хэндл, клон дешёвый. Держать ссылку на поле `self` нельзя:
        // ниже тот же `self` отдаёт пайплайны и меши.
        let Some(instances) = self
            .instance_buffer
            .as_ref()
            .map(|(buffer, _)| buffer.clone())
        else {
            return;
        };
        let mut scene_pipeline = None;
        let mut shadow_pipeline = None;
        for batch in batches {
            let entry = &self.entries[batch.entry as usize];
            let draw = &snapshot.draws[entry.index];
            let primitive = &self.meshes[&draw.mesh][draw.primitive as usize];
            let material = draw
                .material
                .and_then(|id| self.materials.get(&id))
                .unwrap_or(&self.default_material);
            let Some((bind_group, _)) = material.bind_group.as_ref() else {
                continue;
            };
            if shadow {
                let key = (entry.kind.skinned, entry.masked);
                if shadow_pipeline != Some(key) {
                    pass.set_pipeline(&self.shadows.pipelines[&key]);
                    shadow_pipeline = Some(key);
                }
            } else if scene_pipeline != Some(entry.kind) {
                pass.set_pipeline(&self.pipelines[&entry.kind]);
                scene_pipeline = Some(entry.kind);
            }
            pass.set_bind_group(1, bind_group, &[]);
            pass.set_vertex_buffer(0, primitive.vertices.slice(..));
            let instance_slot = if let (Some(joint_slot), Some(skin)) =
                (entry.joint_slot, primitive.skin.as_ref())
            {
                if self.joints.storage() {
                    pass.set_bind_group(2, &self.joint_bind_group, &[]);
                } else {
                    let offset =
                        u32::try_from(u64::from(joint_slot) * self.joint_stride).unwrap_or(0);
                    pass.set_bind_group(2, &self.joint_bind_group, &[offset]);
                }
                pass.set_vertex_buffer(1, skin.slice(..));
                2
            } else {
                1
            };
            pass.set_vertex_buffer(instance_slot, instances.slice(..));
            pass.set_index_buffer(primitive.indices.slice(..), wgpu::IndexFormat::Uint32);
            let end = batch.first.saturating_add(batch.count);
            pass.draw_indexed(0..primitive.index_count, 0, batch.first..end);
            if let Some(stats) = stats.as_mut() {
                stats.drawn += batch.count;
                stats.draw_calls += 1;
            }
        }
    }

    fn prepare_material(&mut self, device: &wgpu::Device, id: AssetId<Material>) {
        let Some(material) = self.materials.get(&id) else {
            return;
        };
        let available = texture_slots(&material.data)
            .map(|t| t.is_some_and(|id| self.textures.contains_key(&id)));
        if material
            .bind_group
            .as_ref()
            .is_some_and(|(_, ready)| *ready == available)
        {
            return;
        }
        let bind_group = self.material_bind_group(device, &material.data, &material.uniform);
        if let Some(material) = self.materials.get_mut(&id) {
            material.bind_group = Some((bind_group, available));
        }
    }

    fn material_bind_group(
        &self,
        device: &wgpu::Device,
        data: &MaterialData,
        uniform: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let [base, mr, normal, occlusion, emissive] = texture_slots(data);
        let view =
            |id: Option<AssetId<Image>>, fallback: &'_ wgpu::TextureView| -> wgpu::TextureView {
                id.and_then(|id| self.textures.get(&id))
                    .map_or_else(|| fallback.clone(), |t| t.view.clone())
            };
        let views = [
            view(base, &self.white_srgb),
            view(mr, &self.white_linear),
            view(normal, &self.flat_normal),
            view(occlusion, &self.white_linear),
            view(emissive, &self.white_srgb),
        ];
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }];
        for (i, view) in views.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: u32::try_from(i + 1).unwrap_or(1),
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: 6,
            resource: wgpu::BindingResource::Sampler(&self.sampler),
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material"),
            layout: &self.material_layout,
            entries: &entries,
        })
    }

    #[allow(dead_code)]
    fn layouts(&self) -> (&wgpu::BindGroupLayout, &wgpu::BindGroupLayout) {
        (&self.frame_layout, &self.material_layout)
    }
}

/// Раскладки групп привязок: 0 — кадр, 1 — материал, 2 — кости.
fn create_layouts(
    device: &wgpu::Device,
    joint_storage: bool,
) -> (
    wgpu::BindGroupLayout,
    wgpu::BindGroupLayout,
    wgpu::BindGroupLayout,
) {
    let uniform = |binding, visibility, dynamic: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: None,
        },
        count: None,
    };
    let texture = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let frame = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("frame"),
        entries: &[
            uniform(0, wgpu::ShaderStages::VERTEX_FRAGMENT, false),
            // Карта теней: массив глубины + сэмплер сравнения
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                count: None,
            },
            // Тени прожекторов и точечных источников
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
        ],
    });
    let material = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("material"),
        entries: &[
            uniform(0, wgpu::ShaderStages::FRAGMENT, false),
            texture(1),
            texture(2),
            texture(3),
            texture(4),
            texture(5),
            wgpu::BindGroupLayoutEntry {
                binding: 6,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    // Со storage все скелеты в одном буфере. Без него — один скелет на вызов, по смещению.
    let joints = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("joints"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: if joint_storage {
                    wgpu::BufferBindingType::Storage { read_only: true }
                } else {
                    wgpu::BufferBindingType::Uniform
                },
                has_dynamic_offset: !joint_storage,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    (frame, material, joints)
}

fn create_joint_storage(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    size: u64,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("joints"),
        size: size.max(MATRIX_SIZE),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("joints"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    (buffer, bind_group)
}

fn create_frame_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
    shadows: &ShadowResources,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("frame"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&shadows.map.array_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&shadows.sampler),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&shadows.local.array_view),
            },
        ],
    })
}

impl ShadowResources {
    /// Пайплайны заполняются позже — вместе с остальными, из исходников шейдеров.
    fn new(device: &wgpu::Device, alignment: u64, size: u32, local_size: u32) -> Self {
        let view_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_view"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true, // матрица каскада — по смещению
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let view_stride = 64u64.div_ceil(alignment) * alignment;
        let (view_buffer, view_bind_group) = create_object_buffer(
            device,
            &view_layout,
            view_stride,
            (CASCADES + LOCAL_SHADOW_LAYERS) as u64,
            64,
        );
        Self {
            map: ShadowMap::new(device, size, CASCADES, "shadow_sun"),
            local: ShadowMap::new(device, local_size, LOCAL_SHADOW_LAYERS, "shadow_local"),
            sampler: crate::shadows::comparison_sampler(device),
            view_layout,
            view_buffer,
            view_bind_group,
            view_stride,
            pipelines: HashMap::new(),
        }
    }
}

/// Текстуры-заглушки 1×1 для пустых слотов материала: белая sRGB, белая линейная,
/// «плоская» нормаль (0, 0, 1).
fn fallback_textures(device: &wgpu::Device, queue: &wgpu::Queue) -> [wgpu::TextureView; 3] {
    let solid = |rgba: [u8; 4], color_space| {
        let image = Image {
            width: 1,
            height: 1,
            color_space,
            data: rgba.to_vec(),
            mips: Vec::new(),
            basis: Vec::new(),
        };
        create_texture(device, queue, &image).view
    };
    [
        solid([255; 4], ColorSpace::Srgb),
        solid([255; 4], ColorSpace::Linear),
        solid([128, 128, 255, 255], ColorSpace::Linear),
    ]
}

fn texture_slots(data: &MaterialData) -> [Option<AssetId<Image>>; 5] {
    [
        data.base_color_texture,
        data.metallic_roughness_texture,
        data.normal_texture,
        data.occlusion_texture,
        data.emissive_texture,
    ]
}

impl GpuMaterial {
    fn new(device: &wgpu::Device, data: MaterialData) -> Self {
        let alpha_cutoff = match data.alpha_mode {
            AlphaMode::Mask(cutoff) => f32::from(cutoff) / 255.0,
            _ => -1.0,
        };
        let uniform = MaterialUniform {
            base_color: data.base_color,
            emissive: [
                data.emissive[0],
                data.emissive[1],
                data.emissive[2],
                alpha_cutoff,
            ],
            params: [
                data.metallic,
                data.roughness,
                1.0,
                if data.normal_texture.is_some() {
                    1.0
                } else {
                    0.0
                },
            ],
        };
        Self {
            uniform: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("material"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            }),
            data,
            bind_group: None,
        }
    }
}

/// Буфер «много элементов по смещению»: объекты (матрица модели) или скелеты (матрицы костей).
/// Шейдер видит один элемент размером `item_size`; какой — задаёт смещение при отрисовке.
fn create_object_buffer(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    stride: u64,
    capacity: u64,
    item_size: u64,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("objects"),
        size: stride * capacity,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("objects"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: NonZeroU64::new(item_size),
            }),
        }],
    });
    (buffer, bind_group)
}

fn create_texture(device: &wgpu::Device, queue: &wgpu::Queue, image: &Image) -> GpuTexture {
    if image.basis.is_empty() {
        upload_rgba(device, queue, image)
    } else {
        crate::ktx::upload_basis(device, queue, image)
    }
}

pub(crate) fn upload_rgba(device: &wgpu::Device, queue: &wgpu::Queue, image: &Image) -> GpuTexture {
    let format = match image.color_space {
        ColorSpace::Srgb => wgpu::TextureFormat::Rgba8UnormSrgb,
        ColorSpace::Linear => wgpu::TextureFormat::Rgba8Unorm,
    };
    let size = wgpu::Extent3d {
        width: image.width,
        height: image.height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("image"),
        size,
        mip_level_count: image.mip_level_count(),
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let levels = std::iter::once(&image.data).chain(image.mips.iter());
    for (level, data) in levels.enumerate() {
        let level = u32::try_from(level).unwrap_or(0);
        let width = (image.width >> level).max(1);
        let height = (image.height >> level).max(1);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }
    GpuTexture {
        view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        bytes: image.memory_size(),
    }
}

type Pipelines = (
    HashMap<PipelineKind, wgpu::RenderPipeline>,
    wgpu::RenderPipeline,
    HashMap<(bool, bool), wgpu::RenderPipeline>,
);

/// Все пайплайны сцены из исходников шейдеров. Ошибка — текстом, без паники.
fn build_pipelines(
    device: &wgpu::Device,
    layouts: [&wgpu::BindGroupLayout; 4],
    color_format: wgpu::TextureFormat,
    hdr: bool,
    joint_storage: bool,
    sources: &HashMap<String, String>,
) -> Result<Pipelines, String> {
    let mut pbr_defines = Vec::new();
    if hdr {
        pbr_defines.push("HDR_OUTPUT");
    }
    if joint_storage {
        pbr_defines.push("JOINT_STORAGE");
    }
    let shadow_defines: &[&str] = if joint_storage {
        &["JOINT_STORAGE"]
    } else {
        &[]
    };
    let pbr = crate::shaders::compile(device, sources, "shaders/pbr.wgsl", &pbr_defines)?;
    let debug = crate::shaders::compile(device, sources, "shaders/debug.wgsl", &pbr_defines)?;
    let shadow = crate::shaders::compile(device, sources, "shaders/shadow.wgsl", shadow_defines)?;
    let [frame, material, joints, shadow_view] = layouts;
    crate::shaders::checked(device, || {
        (
            create_pipelines(device, [frame, material, joints], color_format, &pbr),
            create_line_pipeline(device, frame, color_format, &debug),
            create_shadow_pipelines(device, [shadow_view, material, joints], &shadow),
        )
    })
}

fn instance_layout(
    attributes: &'static [wgpu::VertexAttribute],
) -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<InstanceData>() as u64,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes,
    }
}

/// Пайплайны теней: скелетный × альфа-маска. Только глубина, без цвета.
fn create_shadow_pipelines(
    device: &wgpu::Device,
    layouts: [&wgpu::BindGroupLayout; 3],
    shader: &wgpu::ShaderModule,
) -> HashMap<(bool, bool), wgpu::RenderPipeline> {
    let static_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("shadow"),
        bind_group_layouts: &[Some(layouts[0]), Some(layouts[1])],
        ..Default::default()
    });
    let skinned_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("shadow_skinned"),
        bind_group_layouts: &layouts.map(Some),
        ..Default::default()
    });
    let vertex_layout = Some(wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2],
    });
    let skin_layout = Some(wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<SkinVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![3 => Uint16x4, 4 => Float32x4],
    });
    let static_instance = Some(instance_layout(SHADOW_STATIC_ATTRS));
    let skinned_instance = Some(instance_layout(SHADOW_SKINNED_ATTRS));
    let skinned_buffers = [vertex_layout.clone(), skin_layout, skinned_instance];
    let static_buffers = [vertex_layout, static_instance];
    let mut pipelines = HashMap::new();
    for skinned in [false, true] {
        for masked in [false, true] {
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("shadow"),
                layout: Some(if skinned {
                    &skinned_layout
                } else {
                    &static_layout
                }),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some(if skinned { "vs_skinned" } else { "vs_main" }),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: if skinned {
                        &skinned_buffers
                    } else {
                        &static_buffers
                    },
                },
                // Без отсечения граней: тонкие плоскости (листья, ткань) тоже отбрасывают тень
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    // Сдвиг глубины по наклону поверхности — против «теневых прыщей»
                    // на поверхностях, почти параллельных лучам солнца
                    bias: wgpu::DepthBiasState {
                        constant: 2,
                        slope_scale: 2.0,
                        clamp: 0.0,
                    },
                }),
                multisample: wgpu::MultisampleState::default(),
                // Непрозрачным фрагментный шейдер не нужен — пишется только глубина
                fragment: masked.then_some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_masked"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[],
                }),
                multiview_mask: None,
                cache: None,
            });
            pipelines.insert((skinned, masked), pipeline);
        }
    }
    pipelines
}

/// PBR-пайплайны для всех сочетаний: прозрачность × двусторонность × скиннинг.
fn create_pipelines(
    device: &wgpu::Device,
    layouts: [&wgpu::BindGroupLayout; 3],
    color_format: wgpu::TextureFormat,
    shader: &wgpu::ShaderModule,
) -> HashMap<PipelineKind, wgpu::RenderPipeline> {
    // Обычным мешам группа костей не нужна.
    let static_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pbr"),
        bind_group_layouts: &[Some(layouts[0]), Some(layouts[1])],
        ..Default::default()
    });
    let skinned_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pbr_skinned"),
        bind_group_layouts: &layouts.map(Some),
        ..Default::default()
    });
    let mut pipelines = HashMap::new();
    for skinned in [false, true] {
        for double_sided in [false, true] {
            for blend in [false, true] {
                let kind = PipelineKind {
                    blend,
                    double_sided,
                    skinned,
                };
                let layout = if skinned {
                    &skinned_layout
                } else {
                    &static_layout
                };
                pipelines.insert(
                    kind,
                    create_pipeline(device, layout, shader, color_format, kind),
                );
            }
        }
    }
    pipelines
}

fn create_line_pipeline(
    device: &wgpu::Device,
    frame_layout: &wgpu::BindGroupLayout,
    color_format: wgpu::TextureFormat,
    shader: &wgpu::ShaderModule,
) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("debug"),
        bind_group_layouts: &[Some(frame_layout)],
        ..Default::default()
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("debug_lines"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<crate::debug_draw::LineVertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4],
            })],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            ..Default::default()
        },
        // Глубина проверяется (линии прячутся за объектами), но не записывается
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn create_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    color_format: wgpu::TextureFormat,
    kind: PipelineKind,
) -> wgpu::RenderPipeline {
    let blend = kind.blend.then_some(wgpu::BlendState::ALPHA_BLENDING);
    let (double_sided, depth_write) = (kind.double_sided, !kind.blend);
    let vertex_layout = Some(wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2],
    });
    let skin_layout = Some(wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<SkinVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &wgpu::vertex_attr_array![3 => Uint16x4, 4 => Float32x4],
    });
    let instance = Some(instance_layout(if kind.skinned {
        PBR_SKINNED_ATTRS
    } else {
        PBR_STATIC_ATTRS
    }));
    let skinned_buffers = [vertex_layout.clone(), skin_layout, instance.clone()];
    let static_buffers = [vertex_layout, instance];
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("pbr"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some(if kind.skinned {
                "vs_skinned"
            } else {
                "vs_main"
            }),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: if kind.skinned {
                &skinned_buffers
            } else {
                &static_buffers
            },
        },
        primitive: wgpu::PrimitiveState {
            cull_mode: if double_sided {
                None
            } else {
                Some(wgpu::Face::Back)
            },
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(depth_write),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// Склеивает подряд идущие элементы. `continues(i)` — элемент `i` в том же вызове, что и `i - 1`.
fn group_runs(len: usize, continues: impl Fn(usize) -> bool) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0;
    for i in 1..=len {
        if i == len || !continues(i) {
            ranges.push(start..i);
            start = i;
        }
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::group_runs;

    #[test]
    fn identical_instances_become_one_draw() {
        let ranges = group_runs(5, |i| matches!(i, 1 | 2 | 4));
        assert_eq!(ranges, vec![0..3, 3..5]);
    }

    #[test]
    fn transparent_and_mixed_meshes_stay_split() {
        let ranges = group_runs(3, |_| false);
        assert_eq!(ranges, vec![0..1, 1..2, 2..3]);
    }
}
