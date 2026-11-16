//! Рисование текста (R2-03): глифы из SDF-атласа `engine_text`.
//!
//! - Главный поток (extract): свёрстанные тексты → вершины глифов в снимке кадра;
//!   изменения атласа → команда рендеру (терять нельзя, как текстуры интерфейса).
//! - Рендер: два прохода.
//!   - `TextWorld` — текст в мире, после сцены, с проверкой глубины (закрывается объектами),
//!     в HDR-кадр: проходит тональную компрессию вместе со сценой.
//!   - `TextOverlay` — подписи и экранный текст, после постэффектов, поверх всего (кроме
//!     отладочного интерфейса): постэффекты не размывают мелкий текст.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use engine_core::Window;
use engine_ecs::prelude::*;
use engine_math::{Mat4, Vec2, Vec3};
use engine_scene::{GlobalTransform, Visibility, is_visible};
use engine_text::{AtlasChanges, GlyphAtlas, SDF_PX, SPREAD, Text, TextLayout, TextMode};

use crate::command::RenderCommand;
use crate::frontend::RenderFrontend;

// ---------- главный поток ----------

/// Вершина глифа (64 байта). Режим — в `position.w`: 0 — мир, 1 — подпись, 2 — экран.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct TextVertex {
    /// Мир: точка в мире; подпись: точка привязки в мире; экран: пиксели от левого верхнего угла.
    position: [f32; 4],
    /// xy — сдвиг в пикселях (подпись), zw — текстурные координаты.
    offset_uv: [f32; 4],
    /// Линейный цвет.
    color: [f32; 4],
    /// rgb — линейный цвет обводки, a — толщина обводки в единицах SDF (0..0.5).
    outline: [f32; 4],
}

const MODE_WORLD: f32 = 0.0;
const MODE_LABEL: f32 = 1.0;
const MODE_SCREEN: f32 = 2.0;

/// Вершины текста кадра.
#[derive(Debug, Clone, Default)]
pub(crate) struct TextFrame {
    pub world: Vec<TextVertex>,
    pub overlay: Vec<TextVertex>,
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_rgba(srgb: [f32; 4]) -> [f32; 4] {
    [
        srgb_to_linear(srgb[0]),
        srgb_to_linear(srgb[1]),
        srgb_to_linear(srgb[2]),
        srgb[3],
    ]
}

/// Изменения атласа → рендеру; тексты → вершины в снимок. Стадия `Extract`, до публикации.
#[allow(clippy::type_complexity)]
pub(crate) fn extract_text(
    atlas: Option<ResMut<GlyphAtlas>>,
    texts: Query<(Entity, &Text, &TextLayout, &GlobalTransform)>,
    visibility: Query<(&Visibility, Option<&ChildOf>)>,
    window: Res<Window>,
    mut frontend: NonSendMut<RenderFrontend>,
) {
    // Без TextPlugin атласа нет — и текста тоже
    let Some(mut atlas) = atlas else {
        return;
    };
    let changes = atlas.take_changes();
    if changes.reset.is_some() || !changes.updates.is_empty() {
        frontend.send(RenderCommand::TextAtlas(Box::new(changes)));
    }
    let _zone = tracing::trace_span!("extract_text").entered();
    #[allow(clippy::cast_possible_truncation)]
    let pixels_per_point = window.scale_factor as f32;
    let frame = &mut frontend.snapshots.write().text;
    frame.world.clear();
    frame.overlay.clear();
    for (entity, text, layout, global) in &texts {
        if !layout.matches(&atlas) || !is_visible(entity, &visibility) {
            continue;
        }
        let color = linear_rgba(text.color);
        // Обводка: доля em → единицы SDF (0.5 значения = SPREAD пикселей растеризации)
        #[allow(clippy::cast_precision_loss)]
        let outline = text.outline.map_or([0.0; 4], |o| {
            let width = (o.width * SDF_PX / (2.0 * SPREAD as f32)).clamp(0.0, 0.45);
            let c = linear_rgba(o.color);
            [c[0], c[1], c[2], width]
        });
        let push = |out: &mut Vec<TextVertex>, corner: &dyn Fn(Vec2) -> ([f32; 4], [f32; 2])| {
            for glyph in &layout.glyphs {
                let (a, b) = (glyph.min, glyph.max);
                let (u0, v0, u1, v1) = (
                    glyph.uv_min[0],
                    glyph.uv_min[1],
                    glyph.uv_max[0],
                    glyph.uv_max[1],
                );
                // Два треугольника; верх глифа (max.y) — верх в атласе (uv_min.y)
                let quad = [
                    (Vec2::new(a.x, b.y), [u0, v0]),
                    (Vec2::new(a.x, a.y), [u0, v1]),
                    (Vec2::new(b.x, a.y), [u1, v1]),
                    (Vec2::new(a.x, b.y), [u0, v0]),
                    (Vec2::new(b.x, a.y), [u1, v1]),
                    (Vec2::new(b.x, b.y), [u1, v0]),
                ];
                for (point, uv) in quad {
                    let (position, offset) = corner(point);
                    out.push(TextVertex {
                        position,
                        offset_uv: [offset[0], offset[1], uv[0], uv[1]],
                        color,
                        outline,
                    });
                }
            }
        };
        match text.mode {
            TextMode::World => {
                let model = global.0;
                push(&mut frame.world, &|p| {
                    let world = model.transform_point3(Vec3::new(p.x, p.y, 0.0));
                    ([world.x, world.y, world.z, MODE_WORLD], [0.0, 0.0])
                });
            }
            TextMode::Label { offset } => {
                let anchor = global.translation();
                push(&mut frame.overlay, &|p| {
                    // Точки → пиксели; на экране Y вниз
                    let px = (offset + Vec2::new(p.x, -p.y)) * pixels_per_point;
                    ([anchor.x, anchor.y, anchor.z, MODE_LABEL], [px.x, px.y])
                });
            }
            TextMode::Screen { position } => {
                push(&mut frame.overlay, &|p| {
                    let px = (position + Vec2::new(p.x, -p.y)) * pixels_per_point;
                    ([px.x, px.y, 0.0, MODE_SCREEN], [0.0, 0.0])
                });
            }
        }
    }
}

// ---------- рендер ----------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TextUniform {
    view_proj: [[f32; 4]; 4],
    /// xy — размер цели в пикселях, z = 1 — кодировать sRGB в шейдере.
    screen: [f32; 4],
}

/// Цель прохода текста.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum TextLayer {
    World,
    Overlay,
}

pub(crate) struct TextRenderer {
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    module: wgpu::ShaderModule,
    sampler: wgpu::Sampler,
    atlas: wgpu::Texture,
    uniforms: HashMap<TextLayer, wgpu::Buffer>,
    bind_groups: HashMap<TextLayer, wgpu::BindGroup>,
    pipelines: HashMap<TextLayer, wgpu::RenderPipeline>,
    /// (формат цели, нужна ли глубина) для каждого слоя.
    targets: HashMap<TextLayer, (wgpu::TextureFormat, bool)>,
    /// Кодировать sRGB в шейдере (цель без sRGB-формата, не HDR).
    encode: HashMap<TextLayer, bool>,
    vertices: Option<(wgpu::Buffer, u64)>,
    /// Где в буфере вершины каждого слоя.
    ranges: HashMap<TextLayer, std::ops::Range<u32>>,
}

impl TextRenderer {
    /// `world_format` — формат кадра сцены (HDR или цель), `overlay_format` — формат цели.
    pub fn new(
        device: &wgpu::Device,
        sources: &HashMap<String, String>,
        world_format: wgpu::TextureFormat,
        hdr: bool,
        overlay_format: wgpu::TextureFormat,
    ) -> Result<Self, String> {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("text"),
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
            label: Some("text"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let module = crate::shaders::compile(device, sources, "shaders/text.wgsl", &[])?;
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("text"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniforms = [TextLayer::World, TextLayer::Overlay]
            .into_iter()
            .map(|layer| {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("text_uniform"),
                    size: std::mem::size_of::<TextUniform>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                (layer, buffer)
            })
            .collect();
        let targets = HashMap::from([
            (TextLayer::World, (world_format, true)),
            (TextLayer::Overlay, (overlay_format, false)),
        ]);
        let encode = HashMap::from([
            (TextLayer::World, !hdr && !world_format.is_srgb()),
            (TextLayer::Overlay, !overlay_format.is_srgb()),
        ]);
        let mut renderer = Self {
            layout,
            pipeline_layout,
            module,
            sampler,
            atlas: create_atlas(device, 1),
            uniforms,
            bind_groups: HashMap::new(),
            pipelines: HashMap::new(),
            targets,
            encode,
            vertices: None,
            ranges: HashMap::new(),
        };
        renderer.create_bind_groups(device);
        renderer.create_pipelines(device);
        Ok(renderer)
    }

    /// Горячая перезагрузка шейдера. При ошибке остаётся прежний.
    pub fn rebuild(
        &mut self,
        device: &wgpu::Device,
        sources: &HashMap<String, String>,
    ) -> Result<(), String> {
        self.module = crate::shaders::compile(device, sources, "shaders/text.wgsl", &[])?;
        self.create_pipelines(device);
        Ok(())
    }

    /// Изменения атласа из главного потока.
    pub fn apply_atlas(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        changes: &AtlasChanges,
    ) {
        if let Some(size) = changes.reset {
            self.atlas = create_atlas(device, size);
            self.create_bind_groups(device);
        }
        for update in &changes.updates {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.atlas,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: update.x,
                        y: update.y,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &update.data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(update.width),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: update.width,
                    height: update.height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    /// Вершины и параметры кадра. Возвращает, есть ли текст в мире.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frame: &TextFrame,
        view_proj: &Mat4,
        size: (u32, u32),
    ) {
        self.ranges.clear();
        let total = frame.world.len() + frame.overlay.len();
        if total == 0 {
            return;
        }
        let bytes = (total * std::mem::size_of::<TextVertex>()) as u64;
        if self
            .vertices
            .as_ref()
            .is_none_or(|(_, capacity)| *capacity < bytes)
        {
            let capacity = bytes.next_power_of_two();
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("text_vertices"),
                size: capacity,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.vertices = Some((buffer, capacity));
        }
        if let Some((buffer, _)) = &self.vertices {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&frame.world));
            let offset = (frame.world.len() * std::mem::size_of::<TextVertex>()) as u64;
            queue.write_buffer(buffer, offset, bytemuck::cast_slice(&frame.overlay));
        }
        let world = u32::try_from(frame.world.len()).unwrap_or(0);
        let overlay = u32::try_from(frame.overlay.len()).unwrap_or(0);
        self.ranges.insert(TextLayer::World, 0..world);
        self.ranges
            .insert(TextLayer::Overlay, world..world + overlay);
        #[allow(clippy::cast_precision_loss)]
        for (layer, buffer) in &self.uniforms {
            let uniform = TextUniform {
                view_proj: view_proj.to_cols_array_2d(),
                screen: [
                    size.0 as f32,
                    size.1 as f32,
                    if self.encode[layer] { 1.0 } else { 0.0 },
                    0.0,
                ],
            };
            queue.write_buffer(buffer, 0, bytemuck::bytes_of(&uniform));
        }
    }

    /// Есть ли что рисовать в этом слое (после `prepare`).
    pub fn has(&self, layer: TextLayer) -> bool {
        self.ranges.get(&layer).is_some_and(|r| !r.is_empty())
    }

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layer: TextLayer,
        color: &wgpu::TextureView,
        depth: Option<&wgpu::TextureView>,
    ) {
        let (Some(range), Some((buffer, _))) = (self.ranges.get(&layer), &self.vertices) else {
            return;
        };
        if range.is_empty() {
            return;
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("text"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // поверх кадра
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: depth.map(|view| wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load, // глубина сцены
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(&self.pipelines[&layer]);
        pass.set_bind_group(0, &self.bind_groups[&layer], &[]);
        pass.set_vertex_buffer(0, buffer.slice(..));
        pass.draw(range.clone(), 0..1);
    }

    fn create_bind_groups(&mut self, device: &wgpu::Device) {
        let view = self
            .atlas
            .create_view(&wgpu::TextureViewDescriptor::default());
        for (layer, buffer) in &self.uniforms {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("text"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });
            self.bind_groups.insert(*layer, group);
        }
    }

    fn create_pipelines(&mut self, device: &wgpu::Device) {
        for (layer, (format, depth)) in &self.targets {
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("text"),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &self.module,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<TextVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![
                            0 => Float32x4, 1 => Float32x4, 2 => Float32x4, 3 => Float32x4
                        ],
                    })],
                },
                // Без отсечения граней: текст в мире виден и с обратной стороны (зеркально)
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: depth.then_some(wgpu::DepthStencilState {
                    format: crate::scene::DEPTH_FORMAT,
                    // Прозрачные края букв не должны закрывать то, что нарисуют позже
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &self.module,
                    entry_point: Some("fs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: *format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
            self.pipelines.insert(*layer, pipeline);
        }
    }
}

fn create_atlas(device: &wgpu::Device, size: u32) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("glyph_atlas"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}
