//! Постэффекты (R2-04): bloom, тональная компрессия, FXAA.
//!
//! Каждый проход — полноэкранный треугольник: читает текстуру (и, для тональной компрессии,
//! bloom) и пишет в цель. Пайплайны создаются по требованию для пары «вход шейдера + формат
//! цели» — форматы зависят от платформы (sRGB или нет у окна).

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use engine_core::{CVar, CVarFlags};

pub static RENDER_HDR: CVar<bool> = CVar::new(
    "render.hdr",
    true,
    "HDR-кадр и постэффекты (если GPU умеет); выкл — старый путь без постэффектов",
)
// Решается при создании рендера: пайплайны сцены собираются под формат кадра
.with_flags(CVarFlags::STARTUP);
pub static RENDER_BLOOM: CVar<bool> = CVar::new("render.bloom", true, "Свечение ярких мест");
pub static RENDER_BLOOM_INTENSITY: CVar<f32> =
    CVar::new("render.bloom_intensity", 0.15, "Сила свечения");
pub static RENDER_BLOOM_THRESHOLD: CVar<f32> = CVar::new(
    "render.bloom_threshold",
    1.0,
    "Яркость (линейная), с которой начинается свечение",
);
pub static RENDER_EXPOSURE: CVar<f32> = CVar::new(
    "render.exposure",
    1.0,
    "Экспозиция перед тональной компрессией",
);
pub static RENDER_FXAA: CVar<bool> = CVar::new("render.fxaa", true, "Сглаживание FXAA");

/// Формат HDR-кадра и уровней bloom.
pub(crate) const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Промежуточный кадр после тональной компрессии (для FXAA), значения в sRGB.
pub(crate) const LDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Уровней bloom: 1/2 … 1/32 размера кадра.
pub(crate) const BLOOM_LEVELS: usize = 5;
/// Цепочка уменьшения: `bloom0` (1/2) … `bloom4` (1/32).
pub(crate) const BLOOM_NAMES: [&str; BLOOM_LEVELS] =
    ["bloom0", "bloom1", "bloom2", "bloom3", "bloom4"];
/// Цепочка увеличения: `bloom_up3` = увеличенный `bloom4` + `bloom3`, … `bloom_up0` — итог.
pub(crate) const BLOOM_UP_NAMES: [&str; BLOOM_LEVELS - 1] =
    ["bloom_up0", "bloom_up1", "bloom_up2", "bloom_up3"];

/// Настройки постэффектов кадра (из CVar `render.*`).
#[derive(Debug, Clone, Copy)]
pub(crate) struct PostSettings {
    pub bloom: bool,
    pub bloom_intensity: f32,
    pub bloom_threshold: f32,
    pub exposure: f32,
    pub fxaa: bool,
}

impl PostSettings {
    /// Текущие значения CVar. Читаются в главном потоке и едут в снимке кадра.
    pub fn from_cvars() -> Self {
        Self {
            bloom: RENDER_BLOOM.get(),
            bloom_intensity: RENDER_BLOOM_INTENSITY.get(),
            bloom_threshold: RENDER_BLOOM_THRESHOLD.get(),
            exposure: RENDER_EXPOSURE.get(),
            fxaa: RENDER_FXAA.get(),
        }
    }
}

impl Default for PostSettings {
    fn default() -> Self {
        Self {
            bloom: true,
            bloom_intensity: 0.15,
            bloom_threshold: 1.0,
            exposure: 1.0,
            fxaa: true,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PostParams {
    exposure: f32,
    bloom_intensity: f32,
    bloom_threshold: f32,
    tonemap_output_linear: u32,
    fxaa_output_linear: u32,
    _pad: [u32; 3],
}

/// Можно ли на этом GPU рисовать в HDR-текстуру и фильтровать её.
pub(crate) fn hdr_supported(adapter: &wgpu::Adapter) -> bool {
    let features = adapter.get_texture_format_features(HDR_FORMAT);
    features
        .allowed_usages
        .contains(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
        && features
            .flags
            .contains(wgpu::TextureFormatFeatureFlags::FILTERABLE)
}

pub(crate) struct PostRenderer {
    layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    module: wgpu::ShaderModule,
    sampler: wgpu::Sampler,
    params: wgpu::Buffer,
    /// Чёрная текстура 1×1 — на место bloom, когда он выключен.
    black: wgpu::TextureView,
    pipelines: HashMap<(&'static str, wgpu::TextureFormat), wgpu::RenderPipeline>,
}

/// Один полноэкранный проход.
pub(crate) struct PostPass<'a> {
    pub entry: &'static str,
    pub source: &'a wgpu::TextureView,
    pub bloom: Option<&'a wgpu::TextureView>,
    pub target: &'a wgpu::TextureView,
    pub target_format: wgpu::TextureFormat,
    pub timestamps: Option<wgpu::RenderPassTimestampWrites<'a>>,
}

impl PostRenderer {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sources: &HashMap<String, String>,
    ) -> Result<Self, String> {
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
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post"),
            entries: &[
                texture(0),
                texture(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("post"),
            bind_group_layouts: &[Some(&layout)],
            ..Default::default()
        });
        let module = crate::shaders::compile(device, sources, "shaders/post.wgsl", &[])?;
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("post"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default() // ClampToEdge: выборки за краем берут крайний пиксель
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post_params"),
            size: std::mem::size_of::<PostParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let black = {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("post_black"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HDR_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                texture.as_image_copy(),
                &[0u8; 8],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(8),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            texture.create_view(&wgpu::TextureViewDescriptor::default())
        };
        Ok(Self {
            layout,
            pipeline_layout,
            module,
            sampler,
            params,
            black,
            pipelines: HashMap::new(),
        })
    }

    /// Новые исходники шейдера (горячая перезагрузка). При ошибке остаётся прежний.
    pub fn rebuild(
        &mut self,
        device: &wgpu::Device,
        sources: &HashMap<String, String>,
    ) -> Result<(), String> {
        self.module = crate::shaders::compile(device, sources, "shaders/post.wgsl", &[])?;
        self.pipelines.clear();
        Ok(())
    }

    /// Параметры кадра. `*_linear` — пишем ли в sRGB-текстуру (тогда кодирует оборудование).
    pub fn write_params(
        &self,
        queue: &wgpu::Queue,
        settings: &PostSettings,
        tonemap_linear: bool,
        fxaa_linear: bool,
    ) {
        let params = PostParams {
            exposure: settings.exposure,
            bloom_intensity: if settings.bloom {
                settings.bloom_intensity
            } else {
                0.0
            },
            bloom_threshold: settings.bloom_threshold.max(0.0),
            tonemap_output_linear: u32::from(tonemap_linear),
            fxaa_output_linear: u32::from(fxaa_linear),
            _pad: [0; 3],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
    }

    pub fn run(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pass: &PostPass<'_>,
    ) {
        let pipeline = self
            .pipeline(device, pass.entry, pass.target_format)
            .clone();
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(pass.entry),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(pass.source),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(pass.bloom.unwrap_or(&self.black)),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.params.as_entire_binding(),
                },
            ],
        });
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(pass.entry),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: pass.target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Треугольник закрашивает каждый пиксель — старое содержимое не нужно,
                    // но Clear на тайловых мобильных GPU дешевле, чем Load
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            timestamp_writes: pass.timestamps.clone(),
            ..Default::default()
        });
        render_pass.set_pipeline(&pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        render_pass.draw(0..3, 0..1); // полноэкранный треугольник
    }

    fn pipeline(
        &mut self,
        device: &wgpu::Device,
        entry: &'static str,
        format: wgpu::TextureFormat,
    ) -> &wgpu::RenderPipeline {
        // [Rust] entry API: создать значение, только если по ключу ничего нет
        // (как `try_emplace` в C++, но с ленивым созданием через замыкание)
        self.pipelines.entry((entry, format)).or_insert_with(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(entry),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &self.module,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &self.module,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        })
    }
}
