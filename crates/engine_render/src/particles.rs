//! Частицы (L-02): билборды в снимке кадра. Позиции уже посчитал CPU (`engine_scene`).
//!
//! Проход после декалей и до спрайтов: глубина сцены проверяется, но не пишется,
//! поэтому частицы прячутся за геометрией и не заслоняют друг друга глубиной.
//! Одинаковые текстуры — одним вызовом, внутри текстуры дальние рисуются раньше.

use std::collections::HashMap;
use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use engine_assets::AssetId;
use engine_assets::formats::Image;
use engine_ecs::prelude::*;
use engine_math::Vec3;
use engine_scene::{GlobalTransform, ParticleCloud, ParticleEmitter, Visibility, is_visible};

use crate::frontend::RenderFrontend;
use crate::scene::{DEPTH_FORMAT, SceneRenderer, camera_view_proj, camera_world};
use crate::snapshot::FrameSnapshot;

#[derive(Clone, Copy, Debug)]
pub(crate) struct ParticleItem {
    previous: Vec3,
    position: Vec3,
    age: f32,
    lifetime: f32,
    size: f32,
    color: [f32; 4],
    image: AssetId<Image>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ParticleList {
    pub items: Vec<ParticleItem>,
}

pub(crate) fn extract_particles(
    clouds: Query<(Entity, &ParticleEmitter, &ParticleCloud)>,
    cameras: Query<(&engine_scene::Camera, &GlobalTransform)>,
    visibility: Query<(&Visibility, Option<&ChildOf>)>,
    mut frontend: NonSendMut<RenderFrontend>,
) {
    let _zone = tracing::trace_span!("extract_particles").entered();
    let list = &mut frontend.snapshots.write().particles;
    list.items.clear();
    let eye = cameras
        .iter()
        .min_by_key(|(camera, _)| camera.order)
        .map(|(_, global)| global.translation());
    for (entity, emitter, cloud) in &clouds {
        if !is_visible(entity, &visibility) {
            continue;
        }
        let image = emitter.image.id();
        for particle in cloud.live() {
            list.items.push(ParticleItem {
                previous: particle.previous,
                position: particle.position,
                age: particle.age,
                lifetime: particle.lifetime,
                size: particle.size,
                color: particle.color,
                image,
            });
        }
    }
    crate::sprites::cluster_by(&mut list.items, |_| 0, |item| item.image);
    if let Some(eye) = eye {
        sort_far_first(&mut list.items, eye);
    }
}

fn sort_far_first(items: &mut [ParticleItem], eye: Vec3) {
    let mut start = 0;
    while start < items.len() {
        let image = items[start].image;
        let mut end = start + 1;
        while end < items.len() && items[end].image == image {
            end += 1;
        }
        items[start..end].sort_by(|a, b| {
            let behind = (b.position - eye).length_squared();
            let ahead = (a.position - eye).length_squared();
            behind.total_cmp(&ahead)
        });
        start = end;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParticleInstance {
    center_size: [f32; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParticleUniform {
    view_proj: [[f32; 4]; 4],
    right: [f32; 4],
    up: [f32; 4],
    extra: [f32; 4],
}

struct Batch {
    image: AssetId<Image>,
    range: Range<u32>,
}

pub(crate) struct ParticleRenderer {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    instances: Option<(wgpu::Buffer, u64)>,
    batches: Vec<Batch>,
    groups: Vec<(wgpu::BindGroup, Range<u32>)>,
    encode: bool,
}

impl ParticleRenderer {
    pub fn new(
        device: &wgpu::Device,
        sources: &HashMap<String, String>,
        format: wgpu::TextureFormat,
        hdr: bool,
    ) -> Result<Self, String> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particles"),
            entries: &[
                buffer_entry(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
                texture_entry(1),
                sampler_entry(2),
            ],
        });
        let pipeline = pipeline(device, sources, &layout, format)?;
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("particles"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles_uniform"),
            size: std::mem::size_of::<ParticleUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            layout,
            pipeline,
            sampler,
            uniform,
            instances: None,
            batches: Vec::new(),
            groups: Vec::new(),
            encode: !hdr && !format.is_srgb(),
        })
    }

    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        snapshot: &FrameSnapshot,
        size: (u32, u32),
        alpha: f32,
    ) {
        self.groups.clear();
        self.batches.clear();
        let list = &snapshot.particles;
        if list.items.is_empty() {
            return;
        }
        let instances = instances(list, snapshot.step.as_secs_f32(), alpha);
        self.batches = batches(list, instances.len());
        let bytes = (instances.len() * std::mem::size_of::<ParticleInstance>()) as u64;
        ensure_buffer(device, &mut self.instances, bytes, "particle_instances");
        if let Some((buffer, _)) = &self.instances {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&instances));
        }
        let world = camera_world(snapshot, alpha);
        let uniform = ParticleUniform {
            view_proj: camera_view_proj(snapshot, size, alpha).to_cols_array_2d(),
            right: axis(world.x_axis.truncate(), Vec3::X),
            up: axis(world.y_axis.truncate(), Vec3::Y),
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
        depth: &wgpu::TextureView,
        scene: &SceneRenderer,
        store_depth: bool,
    ) {
        let Some(buffer) = self.instances.as_ref().map(|(buffer, _)| buffer.clone()) else {
            return;
        };
        for batch in &self.batches {
            let Some(view) = scene.image_view(batch.image) else {
                continue;
            };
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("particles"),
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
            label: Some("particles"),
            color_attachments: &[Some(color_attachment(color))],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    // Текст в мире читает глубину после нас. Иначе сцена уже сохранила её,
                    // и повторная запись из тайловой памяти не нужна.
                    store: if store_depth {
                        wgpu::StoreOp::Store
                    } else {
                        wgpu::StoreOp::Discard
                    },
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, buffer.slice(..));
        for (group, range) in &self.groups {
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..6, range.clone());
        }
    }
}

fn instances(list: &ParticleList, step: f32, alpha: f32) -> Vec<ParticleInstance> {
    list.items
        .iter()
        .map(|item| {
            let center = item.previous + (item.position - item.previous) * alpha;
            // alpha = 0 — возраст на прошлом шаге, 1 — на текущем. Новорождённая не уходит в минус.
            let age = (item.age - step * (1.0 - alpha)).max(0.0);
            let fade = 1.0 - (age / item.lifetime.max(1e-4)).clamp(0.0, 1.0);
            let mut color = item.color;
            color[3] *= fade;
            ParticleInstance {
                center_size: [center.x, center.y, center.z, item.size],
                color,
            }
        })
        .collect()
}

fn batches(list: &ParticleList, count: usize) -> Vec<Batch> {
    let mut batches = Vec::new();
    let mut start = 0u32;
    let mut current = list.items[0].image;
    for (index, item) in list.items.iter().enumerate() {
        if item.image != current {
            let end = u32::try_from(index).unwrap_or(u32::MAX);
            batches.push(Batch {
                image: current,
                range: start..end,
            });
            current = item.image;
            start = end;
        }
        if index + 1 == count {
            let end = u32::try_from(count).unwrap_or(u32::MAX);
            batches.push(Batch {
                image: current,
                range: start..end,
            });
        }
    }
    batches
}

fn axis(value: Vec3, fallback: Vec3) -> [f32; 4] {
    let value = value.normalize_or_zero();
    let value = if value.length_squared() < 1e-8 {
        fallback
    } else {
        value
    };
    [value.x, value.y, value.z, 0.0]
}

fn buffer_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn pipeline(
    device: &wgpu::Device,
    sources: &HashMap<String, String>,
    layout: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
) -> Result<wgpu::RenderPipeline, String> {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("particles"),
        bind_group_layouts: &[Some(layout)],
        ..Default::default()
    });
    let module = crate::shaders::compile(device, sources, "shaders/particles.wgsl", &[])?;
    Ok(
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particles"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<ParticleInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x4, 1 => Float32x4],
                })],
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
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
        }),
    )
}

fn color_attachment(view: &wgpu::TextureView) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        },
    }
}

fn ensure_buffer(
    device: &wgpu::Device,
    slot: &mut Option<(wgpu::Buffer, u64)>,
    bytes: u64,
    label: &'static str,
) {
    if slot.as_ref().is_none_or(|(_, capacity)| *capacity < bytes) {
        let capacity = bytes.next_power_of_two().max(256);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: capacity,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        *slot = Some((buffer, capacity));
    }
}
