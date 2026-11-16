//! Спрайты (R2-13): квады в снимке кадра и один проход без освещения.
//!
//! Слои рисуются от меньшего к большему. Внутри слоя одинаковые текстуры идут одним вызовом.
//! Глубина 3D-сцены не участвует: спрайты всегда поверх неё, порядок только между собой.

use std::collections::{HashMap, HashSet};

use bytemuck::{Pod, Zeroable};
use engine_2d::{Camera2d, Sprite};
use engine_assets::AssetId;
use engine_assets::formats::Image;
use engine_core::FixedTime;
use engine_ecs::prelude::*;
use engine_math::{Mat4, Vec4, camera};
use engine_scene::{GlobalTransform, Visibility, is_visible};

use crate::frontend::RenderFrontend;
use crate::scene::SceneRenderer;
use crate::snapshot::interpolate;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Camera2dSnapshot {
    pub prev: Mat4,
    pub curr: Mat4,
    pub height: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct SpriteQuad {
    prev: Mat4,
    curr: Mat4,
    size: [f32; 2],
    uv: [f32; 4],
    color: [f32; 4],
    image: AssetId<Image>,
    layer: i32,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct SpriteList {
    pub camera: Option<Camera2dSnapshot>,
    pub quads: Vec<SpriteQuad>,
}

#[derive(Default)]
pub(crate) struct Motion {
    steps: u64,
    transforms: HashMap<Entity, (Mat4, Mat4)>,
    seen: HashSet<Entity>,
}

impl Motion {
    fn pair(&mut self, entity: Entity, current: Mat4, stepped: bool) -> (Mat4, Mat4) {
        self.seen.insert(entity);
        let entry = self.transforms.entry(entity).or_insert((current, current));
        if stepped {
            entry.0 = entry.1;
        }
        entry.1 = current;
        *entry
    }

    fn finish(&mut self) {
        self.transforms
            .retain(|entity, _| self.seen.contains(entity));
        self.seen.clear();
    }
}

pub(crate) fn extract_sprites(
    fixed: Res<FixedTime>,
    sprites: Query<(Entity, &Sprite, &GlobalTransform)>,
    cameras: Query<(Entity, &Camera2d, &GlobalTransform)>,
    visibility: Query<(&Visibility, Option<&ChildOf>)>,
    mut frontend: NonSendMut<RenderFrontend>,
    mut motion: Local<Motion>,
) {
    let _zone = tracing::trace_span!("extract_sprites").entered();
    let list = &mut frontend.snapshots.write().sprites;
    list.quads.clear();
    list.camera = None;
    let stepped = fixed.steps_total() != motion.steps;
    motion.steps = fixed.steps_total();

    let Some((entity, camera, global)) = cameras.iter().min_by_key(|(_, camera, _)| camera.order)
    else {
        motion.finish();
        return;
    };
    let (prev, curr) = motion.pair(entity, global.0, stepped);
    list.camera = Some(Camera2dSnapshot {
        prev,
        curr,
        height: camera.height,
    });

    for (entity, sprite, global) in &sprites {
        if !is_visible(entity, &visibility) {
            continue;
        }
        if sprite.size.x.abs() < 1e-8 && sprite.size.y.abs() < 1e-8 {
            continue;
        }
        let (prev, curr) = motion.pair(entity, global.0, stepped);
        list.quads.push(SpriteQuad {
            prev,
            curr,
            size: sprite.size.to_array(),
            uv: sprite.uv,
            color: sprite.color,
            image: sprite.image.id(),
            layer: sprite.layer,
        });
    }
    cluster_by(&mut list.quads, |quad| quad.layer, |quad| quad.image);
    motion.finish();
}

/// Сначала по слою, затем одинаковые текстуры слоя — подряд, чтобы рисовать их одним вызовом.
pub(crate) fn cluster_by<T, I: PartialEq>(
    items: &mut [T],
    layer: impl Fn(&T) -> i32,
    image: impl Fn(&T) -> I,
) {
    items.sort_by_key(&layer);
    let mut index = 0;
    while index < items.len() {
        let mut next = index + 1;
        let mut cursor = next;
        while cursor < items.len() && layer(&items[cursor]) == layer(&items[index]) {
            if image(&items[cursor]) == image(&items[index]) {
                items.swap(next, cursor);
                next += 1;
            }
            cursor += 1;
        }
        index = next;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SpriteVertex {
    position: [f32; 4],
    uv: [f32; 2],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SpriteUniform {
    view_proj: [[f32; 4]; 4],
    extra: [f32; 4],
}

struct Batch {
    image: AssetId<Image>,
    range: std::ops::Range<u32>,
}

pub(crate) struct SpriteRenderer {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    vertices: Option<(wgpu::Buffer, u64)>,
    batches: Vec<Batch>,
    groups: Vec<(wgpu::BindGroup, std::ops::Range<u32>)>,
    /// Цель без sRGB-вида и без HDR: шейдер сам кодирует цвет.
    encode: bool,
}

impl SpriteRenderer {
    pub fn new(
        device: &wgpu::Device,
        sources: &HashMap<String, String>,
        format: wgpu::TextureFormat,
        hdr: bool,
    ) -> Result<Self, String> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sprites"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sprites"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let module = crate::shaders::compile(device, sources, "shaders/sprites.wgsl", &[])?;
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sprites"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<SpriteVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x2, 2 => Float32x4],
                })],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sprites"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sprites_uniform"),
            size: std::mem::size_of::<SpriteUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            layout,
            pipeline,
            sampler,
            uniform,
            vertices: None,
            batches: Vec::new(),
            groups: Vec::new(),
            encode: !hdr && !format.is_srgb(),
        })
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        list: &SpriteList,
        aspect: f32,
        alpha: f32,
    ) {
        // Группы прошлого кадра держали виды текстур до submit — к этому моменту он уже прошёл
        self.groups.clear();
        self.batches.clear();
        let Some(camera) = list.camera else {
            return;
        };
        if list.quads.is_empty() {
            return;
        }
        let mut vertices = Vec::with_capacity(list.quads.len() * 6);
        let mut start = 0u32;
        let mut current = list.quads[0].image;
        for (index, quad) in list.quads.iter().enumerate() {
            if quad.image != current {
                let end = u32::try_from(vertices.len()).unwrap_or(u32::MAX);
                self.batches.push(Batch {
                    image: current,
                    range: start..end,
                });
                current = quad.image;
                start = end;
            }
            push_quad(&mut vertices, quad, alpha);
            if index + 1 == list.quads.len() {
                let end = u32::try_from(vertices.len()).unwrap_or(u32::MAX);
                self.batches.push(Batch {
                    image: current,
                    range: start..end,
                });
            }
        }
        let bytes = (vertices.len() * std::mem::size_of::<SpriteVertex>()) as u64;
        if self
            .vertices
            .as_ref()
            .is_none_or(|(_, capacity)| *capacity < bytes)
        {
            let capacity = bytes.next_power_of_two().max(256);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sprite_vertices"),
                size: capacity,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.vertices = Some((buffer, capacity));
        }
        if let Some((buffer, _)) = &self.vertices {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&vertices));
        }
        let view_proj = view_proj(&camera, aspect, alpha);
        let uniform = SpriteUniform {
            view_proj: view_proj.to_cols_array_2d(),
            extra: [if self.encode { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0],
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&uniform));
    }

    pub fn has(&self) -> bool {
        !self.batches.is_empty()
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::TextureView,
        scene: &SceneRenderer,
    ) {
        let Some(buffer) = self.vertices.as_ref().map(|(buffer, _)| buffer.clone()) else {
            return;
        };
        for batch in &self.batches {
            let Some(view) = scene.image_view(batch.image) else {
                continue;
            };
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("sprites"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            self.groups.push((group, batch.range.clone()));
        }
        if self.groups.is_empty() {
            return;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sprites"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, buffer.slice(..));
        for (group, range) in &self.groups {
            pass.set_bind_group(0, group, &[]);
            pass.draw(range.clone(), 0..1);
        }
    }
}

fn push_quad(out: &mut Vec<SpriteVertex>, quad: &SpriteQuad, alpha: f32) {
    let model = interpolate(&quad.prev, &quad.curr, alpha);
    let (hx, hy) = (quad.size[0] * 0.5, quad.size[1] * 0.5);
    let [u0, v0, u1, v1] = quad.uv;
    // Верх квада (больше Y) — верх кадра в текстуре (меньший v)
    let corners = [
        (-hx, hy, u0, v0),
        (-hx, -hy, u0, v1),
        (hx, -hy, u1, v1),
        (-hx, hy, u0, v0),
        (hx, -hy, u1, v1),
        (hx, hy, u1, v0),
    ];
    for (x, y, u, v) in corners {
        let point = model * Vec4::new(x, y, 0.0, 1.0);
        out.push(SpriteVertex {
            position: [point.x, point.y, point.z, 1.0],
            uv: [u, v],
            color: quad.color,
        });
    }
}

fn view_proj(camera: &Camera2dSnapshot, aspect: f32, alpha: f32) -> Mat4 {
    let height = camera.height.max(1e-4);
    let half_h = height * 0.5;
    let half_w = half_h * aspect.max(1e-4);
    let world = interpolate(&camera.prev, &camera.curr, alpha);
    // Спрайт на той же глубине, что камера, не отсекается: слой решает порядок, не z
    camera::orthographic(-half_w, half_w, -half_h, half_h, -100.0, 100.0) * world.inverse()
}

#[cfg(test)]
mod tests {
    use super::cluster_by;

    #[test]
    fn clusters_images_inside_a_layer() {
        let mut items = vec![(1, 'b'), (0, 'a'), (1, 'a'), (0, 'a'), (1, 'b')];
        cluster_by(&mut items, |item| item.0, |item| item.1);
        assert_eq!(
            items,
            vec![(0, 'a'), (0, 'a'), (1, 'b'), (1, 'b'), (1, 'a')]
        );
    }
}
