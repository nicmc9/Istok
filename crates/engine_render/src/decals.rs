//! Декали (L-02): коробка проецирует текстуру на уже нарисованную геометрию.
//!
//! Глубина кадра читается как текстура, мировая точка восстанавливается обратной
//! матрицей вида-проекции (её считает CPU). Проход не пишет глубину и не крепит её
//! как цель: в одном проходе нельзя и писать вложение, и читать ту же текстуру.
//! Лицевые грани куба рисуются один раз — иначе тонкая коробка закрасилась бы дважды.

use std::collections::{HashMap, HashSet};
use std::ops::Range;

use bytemuck::{Pod, Zeroable};
use engine_assets::AssetId;
use engine_assets::formats::Image;
use engine_core::FixedTime;
use engine_ecs::prelude::*;
use engine_math::{Mat4, Vec3};
use engine_scene::{Decal, GlobalTransform, Visibility, is_visible};

use crate::frontend::RenderFrontend;
use crate::scene::{SceneRenderer, camera_view_proj};
use crate::snapshot::{FrameSnapshot, interpolate};

#[derive(Clone, Copy, Debug)]
pub(crate) struct DecalItem {
    prev: Mat4,
    curr: Mat4,
    image: AssetId<Image>,
    color: [f32; 4],
}

#[derive(Clone, Debug, Default)]
pub(crate) struct DecalList {
    pub items: Vec<DecalItem>,
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

pub(crate) fn extract_decals(
    fixed: Res<FixedTime>,
    decals: Query<(Entity, &Decal, &GlobalTransform)>,
    cameras: Query<(&engine_scene::Camera, &GlobalTransform)>,
    visibility: Query<(&Visibility, Option<&ChildOf>)>,
    mut frontend: NonSendMut<RenderFrontend>,
    mut motion: Local<Motion>,
) {
    let _zone = tracing::trace_span!("extract_decals").entered();
    let list = &mut frontend.snapshots.write().decals;
    list.items.clear();
    let stepped = fixed.steps_total() != motion.steps;
    motion.steps = fixed.steps_total();
    let eye = cameras
        .iter()
        .min_by_key(|(camera, _)| camera.order)
        .map(|(_, global)| global.translation());
    for (entity, decal, global) in &decals {
        if !is_visible(entity, &visibility) {
            continue;
        }
        let (prev, curr) = motion.pair(entity, global.0, stepped);
        list.items.push(DecalItem {
            prev,
            curr,
            image: decal.image.id(),
            color: decal.color,
        });
    }
    crate::sprites::cluster_by(&mut list.items, |_| 0, |item| item.image);
    if let Some(eye) = eye {
        sort_far_first(&mut list.items, eye);
    }
    motion.finish();
}

fn sort_far_first(items: &mut [DecalItem], eye: Vec3) {
    let mut start = 0;
    while start < items.len() {
        let image = items[start].image;
        let mut end = start + 1;
        while end < items.len() && items[end].image == image {
            end += 1;
        }
        items[start..end].sort_by(|a, b| {
            let behind = (b.curr.w_axis.truncate() - eye).length_squared();
            let ahead = (a.curr.w_axis.truncate() - eye).length_squared();
            behind.total_cmp(&ahead)
        });
        start = end;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DecalInstance {
    world: [[f32; 4]; 4],
    to_local: [[f32; 4]; 4],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DecalUniform {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    extra: [f32; 4],
}

struct Batch {
    image: AssetId<Image>,
    range: Range<u32>,
}

pub(crate) struct DecalRenderer {
    camera_layout: wgpu::BindGroupLayout,
    texture_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    instances: Option<(wgpu::Buffer, u64)>,
    batches: Vec<Batch>,
    /// Живут до следующего `prepare`: submit кадра идёт после `render`.
    camera_group: Option<wgpu::BindGroup>,
    groups: Vec<(wgpu::BindGroup, Range<u32>)>,
    encode: bool,
}

impl DecalRenderer {
    pub fn new(
        device: &wgpu::Device,
        sources: &HashMap<String, String>,
        format: wgpu::TextureFormat,
        hdr: bool,
    ) -> Result<Self, String> {
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("decals_camera"),
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
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("decals_texture"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline = pipeline(device, sources, &camera_layout, &texture_layout, format)?;
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("decals"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("decals_uniform"),
            size: std::mem::size_of::<DecalUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            camera_layout,
            texture_layout,
            pipeline,
            sampler,
            uniform,
            instances: None,
            batches: Vec::new(),
            camera_group: None,
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
        self.camera_group = None;
        self.groups.clear();
        self.batches.clear();
        let list = &snapshot.decals;
        if list.items.is_empty() {
            return;
        }
        let instances: Vec<_> = list
            .items
            .iter()
            .map(|item| {
                let world = interpolate(&item.prev, &item.curr, alpha);
                DecalInstance {
                    world: world.to_cols_array_2d(),
                    to_local: world.inverse().to_cols_array_2d(),
                    color: item.color,
                }
            })
            .collect();
        self.batches = batches(list);
        let bytes = (instances.len() * std::mem::size_of::<DecalInstance>()) as u64;
        if self
            .instances
            .as_ref()
            .is_none_or(|(_, capacity)| *capacity < bytes)
        {
            let capacity = bytes.next_power_of_two().max(256);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("decal_instances"),
                size: capacity,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.instances = Some((buffer, capacity));
        }
        if let Some((buffer, _)) = &self.instances {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&instances));
        }
        let view_proj = camera_view_proj(snapshot, size, alpha);
        let uniform = DecalUniform {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
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
    ) {
        let Some(buffer) = self.instances.as_ref().map(|(buffer, _)| buffer.clone()) else {
            return;
        };
        let camera = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("decals_camera"),
            layout: &self.camera_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(depth),
                },
            ],
        });
        for batch in &self.batches {
            let Some(view) = scene.image_view(batch.image) else {
                continue;
            };
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("decals"),
                layout: &self.texture_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            self.groups.push((group, batch.range.clone()));
        }
        if self.groups.is_empty() {
            return;
        }
        self.camera_group = Some(camera);
        let Some(camera) = &self.camera_group else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("decals"),
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
        pass.set_bind_group(0, camera, &[]);
        for (group, range) in &self.groups {
            pass.set_bind_group(1, group, &[]);
            pass.draw(0..36, range.clone());
        }
    }
}

fn batches(list: &DecalList) -> Vec<Batch> {
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
        if index + 1 == list.items.len() {
            let end = u32::try_from(list.items.len()).unwrap_or(u32::MAX);
            batches.push(Batch {
                image: current,
                range: start..end,
            });
        }
    }
    batches
}

fn pipeline(
    device: &wgpu::Device,
    sources: &HashMap<String, String>,
    camera: &wgpu::BindGroupLayout,
    texture: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
) -> Result<wgpu::RenderPipeline, String> {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("decals"),
        bind_group_layouts: &[Some(camera), Some(texture)],
        ..Default::default()
    });
    let module = crate::shaders::compile(device, sources, "shaders/decals.wgsl", &[])?;
    Ok(
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("decals"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<DecalInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &wgpu::vertex_attr_array![
                        0 => Float32x4,
                        1 => Float32x4,
                        2 => Float32x4,
                        3 => Float32x4,
                        4 => Float32x4,
                        5 => Float32x4,
                        6 => Float32x4,
                        7 => Float32x4,
                        8 => Float32x4,
                    ],
                })],
            },
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
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
        }),
    )
}

#[cfg(test)]
mod tests {
    use engine_math::{Vec3, Vec4, camera};

    /// Глубина 0..1 и Y текстуры вниз должны собираться обратно в ту же точку пола.
    #[test]
    fn depth_roundtrip_matches_directx_ndc() {
        let eye = Vec3::new(0.0, 2.0, 4.0);
        let world = camera::look_at(eye, Vec3::ZERO, Vec3::Y).inverse();
        let view_proj = camera::perspective(0.9, 16.0 / 9.0, 0.05, 100.0) * world.inverse();
        let point = Vec3::new(0.5, 0.0, 0.2);
        let clip = view_proj * point.extend(1.0);
        let ndc = clip / clip.w;
        let uv_x = ndc.x * 0.5 + 0.5;
        let uv_y = 0.5 - ndc.y * 0.5;
        let rebuilt = Vec4::new(uv_x * 2.0 - 1.0, 1.0 - uv_y * 2.0, ndc.z, 1.0);
        let back = view_proj.inverse() * rebuilt;
        let got = back.truncate() / back.w;
        assert!(
            (got - point).length() < 1e-3,
            "обратная проекция уехала: {got:?}, ждали {point:?}"
        );
    }
}
