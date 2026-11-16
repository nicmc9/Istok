//! Compute-скиннинг: поза считается один раз, сцена и тени её читают.
//!
//! Группа (один примитив и все его экземпляры кадра) целиком на compute или целиком
//! в вершинном шейдере. Сначала берутся маленькие группы, пока есть `render.skin_budget_mb`:
//! толпа одинаковых орков в бюджет не входит и остаётся на вершинном шейдере.

use std::collections::HashMap;
use std::num::NonZeroU64;

use bytemuck::{Pod, Zeroable};
use engine_core::CVar;

/// Байт на уже посчитанную вершину: позиция и нормаль, по `vec4`.
pub(crate) const OUTPUT_STRIDE: u64 = 32;
/// Потоков в группе. То же число, что `@workgroup_size` в `skin.wgsl`.
const WORKGROUP: u32 = 64;
const PARAMS_SIZE: u64 = 16;

pub static RENDER_SKIN_COMPUTE: CVar<bool> = CVar::new(
    "render.skin_compute",
    true,
    "Скиннинг скелета на compute, если он помещается в бюджет",
);
pub static RENDER_SKIN_BUDGET_MB: CVar<u32> = CVar::new(
    "render.skin_budget_mb",
    16,
    "Память уже посчитанных вершин, МБ. Больше — больше скелетов на compute",
);

pub(crate) fn skin_budget_bytes() -> u64 {
    u64::from(RENDER_SKIN_BUDGET_MB.get()).saturating_mul(1024 * 1024)
}

/// Какие группы считать на compute. `sizes` — байты результата группы.
/// Сначала маленькие: толпа не забирает бюджет у одного персонажа.
pub(crate) fn select_skin_groups(sizes: &[u64], budget: u64) -> Vec<bool> {
    let mut order: Vec<usize> = (0..sizes.len()).collect();
    order.sort_by_key(|&index| (sizes[index], index));
    let mut left = budget;
    let mut chosen = vec![false; sizes.len()];
    for index in order {
        let size = sizes[index];
        if size > 0 && size <= left {
            chosen[index] = true;
            left -= size;
        }
    }
    chosen
}

/// Один экземпляр в таблице compute: где его кости и куда писать вершины.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SkinInstance {
    pub joint_offset: u32,
    pub output_base: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SkinParams {
    vertex_count: u32,
    instance_count: u32,
    instance_base: u32,
    x_threads: u32,
}

/// Один dispatch: буферы позы привязки этого примитива и диапазон экземпляров.
pub(crate) struct SkinPass<'a> {
    pub vertices: &'a wgpu::Buffer,
    pub weights: &'a wgpu::Buffer,
    pub vertex_count: u32,
    pub instance_base: u32,
    pub instance_count: u32,
}

struct Dispatch {
    bind_group: wgpu::BindGroup,
    offset: u32,
    groups: u32,
}

pub(crate) struct SkinGpu {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    read_layout: wgpu::BindGroupLayout,
    max_groups: u32,
    uniform_stride: u64,
    out_buffer: Option<(wgpu::Buffer, u64)>,
    read_group: Option<wgpu::BindGroup>,
    params_buffer: Option<(wgpu::Buffer, u64)>,
    instance_buffer: Option<(wgpu::Buffer, u64)>,
    dispatches: Vec<Dispatch>,
}

impl SkinGpu {
    pub(crate) fn new(
        device: &wgpu::Device,
        sources: &HashMap<String, String>,
    ) -> Result<Self, String> {
        let layout = compute_layout(device);
        let read_layout = read_layout(device);
        let pipeline = compile(device, sources, &layout)?;
        let alignment = u64::from(device.limits().min_uniform_buffer_offset_alignment);
        Ok(Self {
            pipeline,
            layout,
            read_layout,
            max_groups: device.limits().max_compute_workgroups_per_dimension.max(1),
            uniform_stride: PARAMS_SIZE.div_ceil(alignment) * alignment,
            out_buffer: None,
            read_group: None,
            params_buffer: None,
            instance_buffer: None,
            dispatches: Vec::new(),
        })
    }

    pub(crate) fn read_layout(&self) -> &wgpu::BindGroupLayout {
        &self.read_layout
    }

    /// Новый compute-пайплайн. Старый остаётся, пока вызывающий не подменит его.
    pub(crate) fn recompile(
        &self,
        device: &wgpu::Device,
        sources: &HashMap<String, String>,
    ) -> Result<wgpu::ComputePipeline, String> {
        compile(device, sources, &self.layout)
    }

    pub(crate) fn install(&mut self, pipeline: wgpu::ComputePipeline) {
        self.pipeline = pipeline;
    }

    pub(crate) fn read_group(&self) -> Option<&wgpu::BindGroup> {
        self.read_group.as_ref()
    }

    pub(crate) fn reset(&mut self) {
        self.dispatches.clear();
    }

    /// Пишет таблицу экземпляров и собирает dispatch на каждую группу.
    pub(crate) fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        joints: &wgpu::Buffer,
        instances: &[SkinInstance],
        passes: &[SkinPass<'_>],
    ) {
        self.dispatches.clear();
        if passes.is_empty() {
            return;
        }
        let out = self.output_buffer(device, passes);
        let params = self.params_buffer(device, queue, passes);
        let table = self.instance_buffer(device, queue, instances);
        self.read_group = Some(read_group(device, &self.read_layout, &out));
        let stride = u32::try_from(self.uniform_stride).unwrap_or(256);
        for (index, pass) in passes.iter().enumerate() {
            let threads = u64::from(pass.vertex_count) * u64::from(pass.instance_count);
            let Ok(threads) = u32::try_from(threads) else {
                continue;
            };
            if threads == 0 {
                continue;
            }
            let (groups, _) = dispatch_width(threads, self.max_groups);
            let offset = u32::try_from(index).unwrap_or(0).saturating_mul(stride);
            let bind_group =
                dispatch_group(device, &self.layout, pass, joints, &out, &table, &params);
            self.dispatches.push(Dispatch {
                bind_group,
                offset,
                groups,
            });
        }
    }

    pub(crate) fn dispatch(&self, encoder: &mut wgpu::CommandEncoder) {
        if self.dispatches.is_empty() {
            return;
        }
        let _zone = tracing::trace_span!("skin_compute").entered();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("skin"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        for dispatch in &self.dispatches {
            pass.set_bind_group(0, &dispatch.bind_group, &[dispatch.offset]);
            pass.dispatch_workgroups(dispatch.groups, 1, 1);
        }
    }

    fn output_buffer(&mut self, device: &wgpu::Device, passes: &[SkinPass<'_>]) -> wgpu::Buffer {
        let vertices: u64 = passes
            .iter()
            .map(|pass| u64::from(pass.vertex_count) * u64::from(pass.instance_count))
            .sum();
        ensure(
            device,
            &mut self.out_buffer,
            "skin_out",
            vertices.saturating_mul(OUTPUT_STRIDE).max(OUTPUT_STRIDE),
            wgpu::BufferUsages::STORAGE,
        )
    }

    fn instance_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances: &[SkinInstance],
    ) -> wgpu::Buffer {
        let bytes = bytemuck::cast_slice(instances);
        let buffer = ensure(
            device,
            &mut self.instance_buffer,
            "skin_instances",
            u64::try_from(bytes.len().max(8)).unwrap_or(u64::MAX),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        if !bytes.is_empty() {
            queue.write_buffer(&buffer, 0, bytes);
        }
        buffer
    }

    fn params_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        passes: &[SkinPass<'_>],
    ) -> wgpu::Buffer {
        let stride = usize::try_from(self.uniform_stride).unwrap_or(256);
        let mut data = vec![0u8; stride * passes.len().max(1)];
        for (index, pass) in passes.iter().enumerate() {
            let threads = u64::from(pass.vertex_count) * u64::from(pass.instance_count);
            let (_, span) =
                dispatch_width(u32::try_from(threads).unwrap_or(u32::MAX), self.max_groups);
            let params = SkinParams {
                vertex_count: pass.vertex_count,
                instance_count: pass.instance_count,
                instance_base: pass.instance_base,
                x_threads: span,
            };
            let offset = index * stride;
            data[offset..offset + usize::try_from(PARAMS_SIZE).unwrap_or(16)]
                .copy_from_slice(bytemuck::bytes_of(&params));
        }
        let buffer = ensure(
            device,
            &mut self.params_buffer,
            "skin_params",
            u64::try_from(data.len()).unwrap_or(u64::MAX),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(&buffer, 0, &data);
        buffer
    }
}

/// `(группы по X, шаг потока)`. Шаг — сколько потоков реально запущено.
fn dispatch_width(threads: u32, max_groups: u32) -> (u32, u32) {
    let needed = threads.div_ceil(WORKGROUP).max(1);
    let groups = needed.min(max_groups.max(1));
    let span = groups.saturating_mul(WORKGROUP).max(1);
    (groups, span)
}

fn ensure(
    device: &wgpu::Device,
    slot: &mut Option<(wgpu::Buffer, u64)>,
    label: &'static str,
    needed: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let needed = needed.max(4);
    if slot.as_ref().is_none_or(|(_, capacity)| *capacity < needed) {
        let capacity = needed.next_power_of_two();
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: capacity,
            usage,
            mapped_at_creation: false,
        });
        *slot = Some((buffer, capacity));
    }
    slot.as_ref()
        .map_or_else(|| panic!("буфер {label}"), |(buffer, _)| buffer.clone())
}

fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn compute_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("skin"),
        entries: &[
            storage(0, true),
            storage(1, true),
            storage(2, true),
            storage(3, false),
            storage(4, true),
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: NonZeroU64::new(PARAMS_SIZE),
                },
                count: None,
            },
        ],
    })
}

fn read_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("skin_read"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

fn compile(
    device: &wgpu::Device,
    sources: &HashMap<String, String>,
    layout: &wgpu::BindGroupLayout,
) -> Result<wgpu::ComputePipeline, String> {
    let module = crate::shaders::compile(device, sources, "shaders/skin.wgsl", &[])?;
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("skin"),
        bind_group_layouts: &[Some(layout)],
        ..Default::default()
    });
    crate::shaders::checked(device, || {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("skin"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    })
}

fn read_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("skin_read"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    })
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn dispatch_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    pass: &SkinPass<'_>,
    joints: &wgpu::Buffer,
    out: &wgpu::Buffer,
    instances: &wgpu::Buffer,
    params: &wgpu::Buffer,
) -> wgpu::BindGroup {
    let entries = [
        buffer_entry(0, pass.vertices),
        buffer_entry(1, pass.weights),
        buffer_entry(2, joints),
        buffer_entry(3, out),
        buffer_entry(4, instances),
        wgpu::BindGroupEntry {
            binding: 5,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: params,
                offset: 0,
                size: NonZeroU64::new(PARAMS_SIZE),
            }),
        },
    ];
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("skin_dispatch"),
        layout,
        entries: &entries,
    })
}

#[cfg(test)]
mod tests {
    use super::select_skin_groups;

    const MB: u64 = 1024 * 1024;
    /// Больший примитив орка: 6484 вершины × 32 байта.
    const ORC_BODY: u64 = 6484 * 32;
    /// Меньший примитив орка: 5437 вершин × 32 байта.
    const ORC_EXTRA: u64 = 5437 * 32;

    #[test]
    fn crowd_stays_on_vertex_shader() {
        let chosen = select_skin_groups(&[ORC_BODY * 1000, ORC_EXTRA * 1000], 16 * MB);
        assert_eq!(chosen, vec![false, false]);
    }

    #[test]
    fn one_character_fits_before_the_crowd() {
        let chosen = select_skin_groups(&[ORC_BODY * 1000, ORC_BODY], 16 * MB);
        assert_eq!(chosen, vec![false, true]);
    }

    #[test]
    fn smaller_group_takes_the_budget_first() {
        let copies = 44u64;
        let chosen = select_skin_groups(&[ORC_BODY * copies, ORC_EXTRA * copies], 16 * MB);
        assert_eq!(chosen, vec![false, true]);
    }
}
