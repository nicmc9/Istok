use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use engine_core::sync::mailbox::Mailbox;
use engine_core::sync::reply::ReplySender;
use engine_core::sync::triple_buffer::Consumer;
use engine_platform::{Instant, WindowSize};

use crate::caps::GpuCaps;
use crate::command::{RenderCommand, Screenshot};
use crate::gpu::{Gpu, GpuInit};
use crate::graph::{PassKind, RenderGraph, TextureDesc};
use crate::post::{
    BLOOM_LEVELS, BLOOM_NAMES, BLOOM_UP_NAMES, HDR_FORMAT, LDR_FORMAT, PostPass, PostRenderer,
};
use crate::scene::{DEPTH_FORMAT, SceneRenderer};
use crate::snapshot::FrameSnapshot;
use crate::text::{TextLayer, TextRenderer};

/// Счётчики, которые рендер пишет, а главный поток читает для статистики.
#[derive(Debug, Default)]
pub(crate) struct SharedStats {
    pub frames_rendered: AtomicU64,
    pub snapshots_skipped: AtomicU64,
    pub frame_cpu_nanos: AtomicU64,
    pub drawn: AtomicU64,
    /// Время GPU проходов, мкс; 0 или `u64::MAX` — нет данных.
    pub gpu_shadow_us: AtomicU64,
    pub gpu_scene_us: AtomicU64,
    pub gpu_post_us: AtomicU64,
    pub gpu_ui_us: AtomicU64,
    pub culled: AtomicU64,
    pub gpu_meshes: AtomicU64,
    pub gpu_textures: AtomicU64,
    pub gpu_bytes: AtomicU64,
    /// 0 — GPU ещё нет, 1 — compute есть, 2 — compute нет.
    pub gpu_tier: AtomicU8,
}

/// Продолжать работу или завершиться.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Flow {
    Continue,
    Exit,
}

/// Куда рисуем: окно или текстура в памяти (offscreen).
enum TargetKind {
    Surface {
        surface: wgpu::Surface<'static>,
        /// `None` — ещё не настроен (нет GPU или окно нулевого размера).
        config: Option<wgpu::SurfaceConfiguration>,
    },
    Offscreen {
        texture: Option<wgpu::Texture>,
    },
}

struct Target {
    kind: TargetKind,
    size: WindowSize,
    /// Формат, в котором рисуют пайплайны (для окна — sRGB-вариант, если есть).
    view_format: Option<wgpu::TextureFormat>,
}

/// Рендер: владеет GPU и целью отрисовки. Работает в своём потоке или в главном —
/// код одинаковый, отличается только то, кто вызывает методы.
pub(crate) struct Renderer {
    commands: Mailbox<RenderCommand>,
    snapshots: Consumer<FrameSnapshot>,
    stats: Arc<SharedStats>,
    gpu: Option<Gpu>,
    /// Возможности GPU. До готовности compute выключен.
    caps: GpuCaps,
    init: Option<GpuInit>,
    target: Option<Target>,
    scene: Option<SceneRenderer>,
    /// Постэффекты; `None` — HDR недоступен или выключен (`render.hdr`), сцена рисуется
    /// прямо в цель с тональной компрессией в шейдере.
    post: Option<PostRenderer>,
    /// Ресурсы, пришедшие до готовности GPU, — применятся после инициализации.
    pending: Vec<RenderCommand>,
    screenshots: Vec<ReplySender<Result<Screenshot, String>>>,
    ui: Option<egui_wgpu::Renderer>,
    /// Текст; `None` — шейдер текста не собрался (ошибка в логе).
    text: Option<TextRenderer>,
    /// Спрайты; `None` — шейдер не собрался (ошибка в логе).
    sprites: Option<crate::sprites::SpriteRenderer>,
    /// Частицы; `None` — шейдер не собрался.
    particles: Option<crate::particles::ParticleRenderer>,
    /// Декали; `None` — шейдер не собрался.
    decals: Option<crate::decals::DecalRenderer>,
    /// Временные текстуры render graph, переиспользуемые между кадрами.
    pool: crate::graph::TexturePool,
    timer: Option<crate::gpu_timer::GpuTimer>,
    /// Исходники шейдеров по имени (`shaders/pbr.wgsl`): встроенные, затем обновления.
    shader_sources: std::collections::HashMap<String, String>,
    /// Текстуры интерфейса, которые нужно освободить после отрисовки кадра.
    ui_free: Vec<egui::TextureId>,
}

impl Renderer {
    pub fn new(
        commands: Mailbox<RenderCommand>,
        snapshots: Consumer<FrameSnapshot>,
        stats: Arc<SharedStats>,
    ) -> Self {
        Self {
            commands,
            snapshots,
            stats,
            gpu: None,
            caps: GpuCaps::default(),
            init: None,
            target: None,
            scene: None,
            post: None,
            pending: Vec::new(),
            screenshots: Vec::new(),
            ui: None,
            text: None,
            sprites: None,
            particles: None,
            decals: None,
            ui_free: Vec::new(),
            pool: crate::graph::TexturePool::default(),
            timer: None,
            shader_sources: crate::shaders::embedded_sources(),
        }
    }

    /// Можно ли рисовать прямо сейчас.
    pub fn can_render(&self) -> bool {
        self.gpu.is_some()
            && self.scene.is_some()
            && self
                .target
                .as_ref()
                .is_some_and(|t| t.view_format.is_some())
    }

    /// Нечего делать, пока не придёт команда: нет окна или GPU и инициализация не идёт.
    #[cfg(not(target_arch = "wasm32"))] // нужен только потоку рендера
    pub fn is_idle(&self) -> bool {
        !self.can_render() && self.init.is_none()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait_command(&mut self) -> Flow {
        match self.commands.recv() {
            Some(command) => self.handle(command),
            // Все отправители уничтожены — главный поток завершился
            None => Flow::Exit,
        }
    }

    /// Выполняет все накопившиеся команды без ожидания.
    pub fn process_commands(&mut self) -> Flow {
        while let Some(command) = self.commands.try_recv() {
            if self.handle(command) == Flow::Exit {
                return Flow::Exit;
            }
        }
        Flow::Continue
    }

    fn handle(&mut self, command: RenderCommand) -> Flow {
        tracing::trace!("Рендер: команда {command:?}");
        match command {
            RenderCommand::SurfaceCreated {
                instance,
                surface,
                size,
            } => {
                if self.gpu.is_none() && self.init.is_none() {
                    self.init = Some(GpuInit::start(&instance, Some(&surface)));
                }
                self.target = Some(Target {
                    kind: TargetKind::Surface {
                        surface,
                        config: None,
                    },
                    size,
                    view_format: None,
                });
                self.configure();
            }
            RenderCommand::InitHeadless { instance, size } => {
                self.init = Some(GpuInit::start(&instance, None));
                self.target = Some(Target {
                    kind: TargetKind::Offscreen { texture: None },
                    size,
                    view_format: None,
                });
            }
            RenderCommand::SurfaceLost { reply } => {
                // Surface уничтожается здесь — до ответа главному потоку
                self.target = None;
                reply.send(());
                tracing::info!("Рендер: surface отпущен");
            }
            RenderCommand::Resize(size) => {
                if let Some(target) = &mut self.target {
                    // При переносе окна между экранами winit присылает и смену размера,
                    // и смену масштаба — второй раз перенастраивать незачем
                    if target.size == size && target.view_format.is_some() {
                        return Flow::Continue;
                    }
                    target.size = size;
                }
                self.configure();
            }
            RenderCommand::Screenshot { reply } => self.screenshots.push(reply),
            RenderCommand::UpdateShader { name, source } => self.update_shader(name, source),
            RenderCommand::Shutdown => return Flow::Exit,
            resource => self.apply_resource(resource),
        }
        Flow::Continue
    }

    fn apply_resource(&mut self, command: RenderCommand) {
        let (Some(gpu), Some(scene)) = (&self.gpu, &mut self.scene) else {
            self.pending.push(command);
            return;
        };
        match command {
            RenderCommand::UploadMesh(id, mesh) => scene.upload_mesh(&gpu.device, id, &mesh),
            RenderCommand::UploadImage(id, image) => {
                scene.upload_image(&gpu.device, &gpu.queue, id, &image);
            }
            RenderCommand::UploadMaterial(id, data) => scene.upload_material(&gpu.device, id, data),
            RenderCommand::RemoveMesh(id) => scene.remove_mesh(id),
            RenderCommand::RemoveImage(id) => scene.remove_image(id),
            RenderCommand::RemoveMaterial(id) => scene.remove_material(id),
            RenderCommand::UiTextures(delta) => {
                let delta = &delta.0;
                if let Some(ui) = &mut self.ui {
                    for (id, images) in &delta.set {
                        for image in images {
                            ui.update_texture(&gpu.device, &gpu.queue, *id, image);
                        }
                    }
                }
                // Обёртка UiTextures пометит изменения обработанными при уничтожении
                self.ui_free.extend(delta.free.iter().copied());
            }
            RenderCommand::TextAtlas(changes) => {
                if let Some(text) = &mut self.text {
                    text.apply_atlas(&gpu.device, &gpu.queue, &changes);
                }
            }
            _ => {}
        }
        let (meshes, textures, _) = scene.counts();
        self.stats
            .gpu_meshes
            .store(meshes as u64, Ordering::Relaxed);
        self.stats
            .gpu_textures
            .store(textures as u64, Ordering::Relaxed);
        self.stats
            .gpu_bytes
            .store(scene.gpu_bytes(), Ordering::Relaxed);
    }

    /// Новый исходник шейдера (горячая перезагрузка). Пайплайны пересобираются; при ошибке
    /// в шейдере рендер продолжает работать со старыми — приложение не падает.
    fn update_shader(&mut self, name: String, source: String) {
        if self.shader_sources.get(&name) == Some(&source) {
            return; // первая загрузка совпадает со встроенным — пересобирать нечего
        }
        let previous = self.shader_sources.insert(name.clone(), source);
        let (Some(gpu), Some(scene)) = (&self.gpu, &mut self.scene) else {
            return;
        };
        // [Rust] `and_then` на Option<&mut _>: пересобрать постэффекты, если они есть
        let result = scene
            .rebuild_shaders(&gpu.device, &self.shader_sources)
            .and_then(|()| {
                self.post.as_mut().map_or(Ok(()), |post| {
                    post.rebuild(&gpu.device, &self.shader_sources)
                })
            })
            .and_then(|()| {
                self.text.as_mut().map_or(Ok(()), |text| {
                    text.rebuild(&gpu.device, &self.shader_sources)
                })
            });
        match result {
            Ok(()) => tracing::info!("Шейдер обновлён: {name}"),
            Err(error) => {
                tracing::error!("Ошибка в шейдере, остаются прежние пайплайны:\n{error}");
                // Откатываем исходник: следующая правка другого файла не должна
                // подхватить сломанный
                if let Some(previous) = previous {
                    self.shader_sources.insert(name, previous);
                }
            }
        }
    }

    /// Проверяет асинхронную инициализацию GPU.
    pub fn poll_init(&mut self) {
        let Some(init) = &mut self.init else { return };
        match init.poll() {
            None => {}
            Some(Ok(gpu)) => {
                let info = gpu.adapter.get_info();
                tracing::info!(
                    "GPU: {} ({:?}, драйвер {})",
                    info.name,
                    info.backend,
                    info.driver_info
                );
                let caps = GpuCaps::from_probe(crate::caps::Probe::from_adapter(&gpu.adapter));
                tracing::info!("{}", caps.report());
                self.stats.gpu_tier.store(caps.tier_code(), Ordering::Relaxed);
                self.caps = caps;
                self.gpu = Some(gpu);
                self.init = None;
                self.configure();
            }
            Some(Err(error)) => {
                tracing::error!("Не удалось инициализировать GPU: {error}");
                self.init = None;
            }
        }
    }

    /// (Пере)настраивает цель отрисовки под текущий размер.
    fn configure(&mut self) {
        // Копируем до заимствования `gpu`: иначе `self.caps` и `self.gpu` спорят.
        let surface_view_formats = self.caps.surface_view_formats();
        let (Some(gpu), Some(target)) = (&self.gpu, &mut self.target) else {
            return;
        };
        let WindowSize { width, height } = target.size;
        if width == 0 || height == 0 {
            // Свёрнутое окно: настраивать цель нулевого размера нельзя
            target.view_format = None;
            return;
        }
        let view_format = match &mut target.kind {
            TargetKind::Surface { surface, config } => {
                let Some(mut new_config) = surface.get_default_config(&gpu.adapter, width, height)
                else {
                    tracing::error!("Surface несовместим с адаптером");
                    return;
                };
                // Fifo = вертикальная синхронизация: get_current_texture ждёт свободный кадр
                // экрана, и это задаёт темп потока рендера. Поддерживается везде.
                new_config.present_mode = wgpu::PresentMode::Fifo;
                new_config.desired_maximum_frame_latency = 2;
                let capabilities = surface.get_capabilities(&gpu.adapter);
                if capabilities.usages.contains(wgpu::TextureUsages::COPY_SRC) {
                    new_config.usage |= wgpu::TextureUsages::COPY_SRC; // для скриншотов
                }
                // На Web формат окна не sRGB. Рисуем через sRGB-«вид» той же текстуры, если
                // GPU умеет; иначе шейдер кодирует sRGB сам (флаг в uniform)
                let srgb = new_config.format.add_srgb_suffix();
                let format = if srgb != new_config.format && surface_view_formats {
                    new_config.view_formats.push(srgb);
                    srgb
                } else {
                    new_config.format
                };
                surface.configure(&gpu.device, &new_config);
                tracing::info!(
                    "Surface: {width}x{height}, {:?} (рисуем в {format:?})",
                    new_config.format
                );
                *config = Some(new_config);
                format
            }
            TargetKind::Offscreen { texture } => {
                let format = wgpu::TextureFormat::Rgba8UnormSrgb;
                *texture = Some(gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("offscreen"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                }));
                tracing::info!("Offscreen: {width}x{height}, {format:?}");
                format
            }
        };
        target.view_format = Some(view_format);
        // Временные текстуры графа (глубина и т.п.) старого размера больше не нужны
        self.pool.retain_size((width, height));

        if self.scene.is_none() {
            self.create_renderers(view_format);
        }
        // Рендер (снова) готов рисовать: пропуски за время простоя не считаем
        self.snapshots.resync();
    }

    /// Первая настройка цели: рендеры сцены, постэффектов, текста и интерфейса под её формат.
    fn create_renderers(&mut self, view_format: wgpu::TextureFormat) {
        // HDR уже решён в `GpuCaps`: формат Rgba16Float, не тир compute.
        let hdr_ok = self.caps.hdr();
        let timestamps = self.caps.timestamps();
        let Some(gpu) = &self.gpu else { return };
        self.timer = crate::gpu_timer::GpuTimer::new(&gpu.device, &gpu.queue, timestamps);
        // HDR: сцена пишет линейный цвет в Rgba16Float, постэффекты сводят его в цель.
        // Без поддержки (часть GLES) — прежний путь: тональная компрессия в PBR-шейдере.
        self.post = if crate::post::RENDER_HDR.get() && hdr_ok {
            PostRenderer::new(&gpu.device, &gpu.queue, &self.shader_sources)
                .inspect_err(|error| tracing::error!("Постэффекты недоступны: {error}"))
                .ok()
        } else {
            None
        };
        let hdr = self.post.is_some();
        tracing::info!(
            "Рендер: HDR и постэффекты {}",
            if hdr {
                "включены"
            } else {
                "выключены"
            }
        );
        self.scene = Some(SceneRenderer::new(
            &gpu.device,
            &gpu.queue,
            if hdr { HDR_FORMAT } else { view_format },
            hdr,
            &self.shader_sources,
        ));
        self.text = TextRenderer::new(
            &gpu.device,
            &self.shader_sources,
            if hdr { HDR_FORMAT } else { view_format },
            hdr,
            view_format,
        )
        .inspect_err(|error| tracing::error!("Текст недоступен: {error}"))
        .ok();
        self.sprites = crate::sprites::SpriteRenderer::new(
            &gpu.device,
            &self.shader_sources,
            if hdr { HDR_FORMAT } else { view_format },
            hdr,
        )
        .inspect_err(|error| tracing::error!("Спрайты недоступны: {error}"))
        .ok();
        self.particles = crate::particles::ParticleRenderer::new(
            &gpu.device,
            &self.shader_sources,
            if hdr { HDR_FORMAT } else { view_format },
            hdr,
        )
        .inspect_err(|error| tracing::error!("Частицы недоступны: {error}"))
        .ok();
        self.decals = crate::decals::DecalRenderer::new(
            &gpu.device,
            &self.shader_sources,
            if hdr { HDR_FORMAT } else { view_format },
            hdr,
        )
        .inspect_err(|error| tracing::error!("Декали недоступны: {error}"))
        .ok();
        self.ui = Some(egui_wgpu::Renderer::new(
            &gpu.device,
            view_format,
            egui_wgpu::RendererOptions::default(),
        ));
        // Ресурсы, пришедшие до готовности GPU
        for command in std::mem::take(&mut self.pending) {
            self.apply_resource(command);
        }
    }

    /// После отправки кадра: освобождение текстур интерфейса и статистика.
    fn finish_frame(&mut self, work_started: Instant, draw: crate::scene::DrawStats) {
        if let Some(ui) = &mut self.ui {
            for id in self.ui_free.drain(..) {
                ui.free_texture(&id);
            }
        }
        let work = u64::try_from(work_started.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.stats.frame_cpu_nanos.store(work, Ordering::Relaxed);
        self.stats
            .drawn
            .store(u64::from(draw.drawn), Ordering::Relaxed);
        self.stats
            .culled
            .store(u64::from(draw.culled), Ordering::Relaxed);
        self.stats.frames_rendered.fetch_add(1, Ordering::Relaxed);
        if let Some(timer) = &self.timer {
            let [shadow, scene, post, ui] = timer.results;
            self.stats
                .gpu_shadow_us
                .store(shadow.unwrap_or(u64::MAX), Ordering::Relaxed);
            self.stats
                .gpu_scene_us
                .store(scene.unwrap_or(u64::MAX), Ordering::Relaxed);
            self.stats
                .gpu_post_us
                .store(post.unwrap_or(u64::MAX), Ordering::Relaxed);
            self.stats
                .gpu_ui_us
                .store(ui.unwrap_or(u64::MAX), Ordering::Relaxed);
        }
    }

    /// Рисует кадр. `false` — кадр не нарисован (не готово, окно скрыто и т.п.).
    pub fn render_frame(&mut self) -> bool {
        if !self.can_render() {
            return false;
        }
        let _zone = tracing::trace_span!("render_frame").entered();
        self.snapshots.take();
        self.stats
            .snapshots_skipped
            .store(self.snapshots.skipped(), Ordering::Relaxed);

        // [Rust] Выше проверено, что всё есть, но компилятор об этом не знает:
        // `let ... else` достаёт значения или выходит из функции.
        let (Some(gpu), Some(target), Some(scene)) = (&self.gpu, &self.target, &mut self.scene)
        else {
            return false;
        };
        let Some(view_format) = target.view_format else {
            return false;
        };

        // Цель: кадр окна или offscreen-текстура
        let acquire_started = Instant::now();
        let surface_frame = match &target.kind {
            TargetKind::Surface { surface, .. } => match acquire(surface) {
                Acquired::Frame(frame) => Some(frame),
                Acquired::Skip => return false,
                Acquired::Reconfigure => {
                    self.configure();
                    return false;
                }
            },
            TargetKind::Offscreen { .. } => None,
        };
        let waited = acquire_started.elapsed();
        if waited.as_millis() > 100 {
            tracing::debug!("Рендер: ожидание кадра экрана {waited:?}");
        }
        let work_started = Instant::now();
        let texture = match (&surface_frame, &target.kind) {
            (Some(frame), _) => &frame.texture,
            (
                None,
                TargetKind::Offscreen {
                    texture: Some(texture),
                },
            ) => texture,
            _ => return false,
        };
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(view_format),
            ..Default::default()
        });

        let snapshot = self.snapshots.current();
        let alpha = snapshot.alpha(Instant::now());
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        if let Some(timer) = &mut self.timer {
            timer.collect(&gpu.device);
        }
        let inputs = FrameInputs {
            gpu,
            view: &view,
            view_format,
            size: target.size,
            snapshot,
            alpha,
            timer: self.timer.as_ref(),
        };
        let (draw, ui_commands) = execute_graph(
            &inputs,
            &mut encoder,
            scene,
            self.post.as_mut(),
            self.text.as_mut(),
            self.sprites.as_mut(),
            self.particles.as_mut(),
            self.decals.as_mut(),
            self.ui.as_mut(),
            &mut self.pool,
        );
        let timer_slot = self.timer.as_ref().and_then(|t| t.resolve(&mut encoder));

        // Скриншот: копия кадра в буфер, читаемый процессором
        let readback = (!self.screenshots.is_empty())
            .then(|| crate::screenshot::copy_to_buffer(&gpu.device, &mut encoder, texture));
        {
            let _zone = tracing::trace_span!("submit_present").entered();
            gpu.queue
                .submit(ui_commands.into_iter().chain([encoder.finish()]));
            if let Some(readback) = readback {
                let result = readback.read(&gpu.device);
                for reply in self.screenshots.drain(..) {
                    reply.send(result.clone());
                }
            }
            if let Some(frame) = surface_frame {
                gpu.queue.present(frame);
            }
            if let (Some(timer), Some(slot)) = (&self.timer, timer_slot) {
                timer.after_submit(slot);
            }
        }
        self.finish_frame(work_started, draw);
        engine_platform::profiling::render_frame_mark();
        true
    }
}

enum Acquired {
    Frame(wgpu::SurfaceTexture),
    Skip,
    Reconfigure,
}

/// Следующий кадр окна. Здесь поток ждёт свободный кадр экрана (vsync).
fn acquire(surface: &wgpu::Surface<'static>) -> Acquired {
    let acquired = {
        let _zone = tracing::trace_span!("acquire_frame").entered();
        surface.get_current_texture()
    };
    match acquired {
        wgpu::CurrentSurfaceTexture::Success(frame)
        | wgpu::CurrentSurfaceTexture::Suboptimal(frame) => Acquired::Frame(frame),
        wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
            tracing::debug!("Рендер: кадр экрана недоступен (окно скрыто или таймаут)");
            Acquired::Skip
        }
        wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
            tracing::warn!("Surface устарел, перенастраиваем");
            Acquired::Reconfigure
        }
        wgpu::CurrentSurfaceTexture::Validation => {
            tracing::error!("Ошибка валидации при получении кадра");
            Acquired::Skip
        }
    }
}

/// Интерфейс поверх сцены. Возвращает командные буферы egui (загрузка вершин),
/// которые нужно отправить раньше основного.
fn render_ui(
    ui: &mut egui_wgpu::Renderer,
    gpu: &Gpu,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    size: WindowSize,
    frame: &crate::ui::UiFrame,
    timestamps: Option<wgpu::RenderPassTimestampWrites<'_>>,
) -> Vec<wgpu::CommandBuffer> {
    if frame.primitives.is_empty() {
        return Vec::new();
    }
    let _zone = tracing::trace_span!("render_ui").entered();
    let screen = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [size.width, size.height],
        pixels_per_point: frame.pixels_per_point,
    };
    let commands = ui.update_buffers(&gpu.device, &gpu.queue, encoder, &frame.primitives, &screen);
    let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("ui"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load, // поверх сцены
                store: wgpu::StoreOp::Store,
            },
        })],
        timestamp_writes: timestamps,
        ..Default::default()
    });
    // [Rust] egui-wgpu требует проход с временем жизни 'static. `forget_lifetime` снимает
    // привязку прохода к encoder — wgpu сам проверит порядок во время выполнения.
    let mut pass = pass.forget_lifetime();
    ui.render(&mut pass, &frame.primitives, &screen);
    commands
}

/// Всё, что нужно проходам кадра, кроме изменяемых частей рендера.
struct FrameInputs<'a> {
    gpu: &'a Gpu,
    view: &'a wgpu::TextureView,
    view_format: wgpu::TextureFormat,
    size: WindowSize,
    snapshot: &'a FrameSnapshot,
    alpha: f32,
    timer: Option<&'a crate::gpu_timer::GpuTimer>,
}

/// Строит граф кадра и выполняет проходы по порядку. Сюда добавляются новые проходы
/// (тени и т.п.): ресурс + проход в графе + ветка в `match`.
///
/// С HDR: сцена → декали → частицы → спрайты → текст в мире → `hdr` → bloom →
/// тональная компрессия → (`ldr` → FXAA) → подписи → интерфейс.
/// Без HDR: сцена → декали → частицы → спрайты → текст в мире → подписи → интерфейс.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)] // один `match` по проходам — читается целиком
fn execute_graph(
    frame: &FrameInputs<'_>,
    encoder: &mut wgpu::CommandEncoder,
    scene: &mut SceneRenderer,
    mut post: Option<&mut PostRenderer>,
    mut text: Option<&mut TextRenderer>,
    mut sprites: Option<&mut crate::sprites::SpriteRenderer>,
    mut particles: Option<&mut crate::particles::ParticleRenderer>,
    mut decals: Option<&mut crate::decals::DecalRenderer>,
    mut ui: Option<&mut egui_wgpu::Renderer>,
    pool: &mut crate::graph::TexturePool,
) -> (crate::scene::DrawStats, Vec<wgpu::CommandBuffer>) {
    let gpu = frame.gpu;
    let settings = frame.snapshot.post;
    let sampled = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    let desc = |format, scale| TextureDesc {
        format,
        usage: sampled,
        scale,
    };

    let mut graph = RenderGraph::default();
    let target = graph.import("target");
    // Декаль читает глубину как текстуру. Без декалей вложение только для теста глубины:
    // лишняя возможность чтения заставляет мобильный GPU сбросить тайловую память.
    let depth_usage = if frame.snapshot.decals.items.is_empty() {
        wgpu::TextureUsages::RENDER_ATTACHMENT
    } else {
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
    };
    let depth = graph.transient(
        "depth",
        TextureDesc {
            format: DEPTH_FORMAT,
            usage: depth_usage,
            scale: 0,
        },
    );
    // [Rust] Ресурсы постэффектов есть только с HDR — `Option` вместо «пустых» id
    let hdr = post
        .is_some()
        .then(|| graph.transient("hdr", desc(HDR_FORMAT, 0)));
    let mut bloom = [None; BLOOM_LEVELS];
    let mut bloom_up = [None; BLOOM_LEVELS - 1];
    let mut ldr = None;
    // Карта теней живёт в SceneRenderer (массив слоёв, свой размер) — в графе она
    // «внешний» ресурс: граф только упорядочивает запись (тени) и чтение (сцена)
    let shadow_map = graph.import("shadow_map");
    graph.add_pass(PassKind::Shadow, &[], &[shadow_map]);
    match hdr {
        None => graph.add_pass(PassKind::Scene, &[shadow_map], &[target, depth]),
        Some(hdr) => {
            graph.add_pass(PassKind::Scene, &[shadow_map], &[hdr, depth]);
            let mut glow = None;
            if settings.bloom {
                // Уровень i — 1/2^(i+1) кадра
                for level in 0..BLOOM_LEVELS {
                    #[allow(clippy::cast_possible_truncation)] // уровней единицы
                    let id =
                        graph.transient(BLOOM_NAMES[level], desc(HDR_FORMAT, level as u32 + 1));
                    let source = if level == 0 {
                        hdr
                    } else {
                        bloom[level - 1].unwrap_or(hdr)
                    };
                    #[allow(clippy::cast_possible_truncation)]
                    graph.add_pass(PassKind::BloomDown(level as u8), &[source], &[id]);
                    bloom[level] = Some(id);
                }
                // Сверху вниз: up[i] = увеличенный (up[i+1] или bloom[последний]) + bloom[i]
                let mut smaller = bloom[BLOOM_LEVELS - 1];
                for level in (0..BLOOM_LEVELS - 1).rev() {
                    #[allow(clippy::cast_possible_truncation)]
                    let id =
                        graph.transient(BLOOM_UP_NAMES[level], desc(HDR_FORMAT, level as u32 + 1));
                    let reads: Vec<_> = [smaller, bloom[level]].into_iter().flatten().collect();
                    #[allow(clippy::cast_possible_truncation)]
                    graph.add_pass(PassKind::BloomUp(level as u8), &reads, &[id]);
                    bloom_up[level] = Some(id);
                    smaller = Some(id);
                }
                glow = bloom_up[0];
            }
            let tonemap_out = if settings.fxaa {
                let id = graph.transient("ldr", desc(LDR_FORMAT, 0));
                ldr = Some(id);
                id
            } else {
                target
            };
            let reads: Vec<_> = [Some(hdr), glow].into_iter().flatten().collect();
            graph.add_pass(PassKind::Tonemap, &reads, &[tonemap_out]);
            if let Some(ldr) = ldr {
                graph.add_pass(PassKind::Fxaa, &[ldr], &[target]);
            }
        }
    }
    // Декали и частицы пишут тот же цвет, что сцена, и читают её глубину. Добавляем до
    // спрайтов: при двух записях в одну цель порядок добавления решает, кто сверху.
    // Постэффекты эту цель читают, поэтому всё равно идут после.
    let size = (frame.size.width, frame.size.height);
    if let Some(decals) = decals.as_deref_mut() {
        decals.prepare(&gpu.device, &gpu.queue, frame.snapshot, size, frame.alpha);
        if decals.has() {
            graph.add_pass(PassKind::Decals, &[depth], &[hdr.unwrap_or(target)]);
        }
    }
    if let Some(particles) = particles.as_deref_mut() {
        particles.prepare(&gpu.device, &gpu.queue, frame.snapshot, size, frame.alpha);
        if particles.has() {
            graph.add_pass(PassKind::Particles, &[depth], &[hdr.unwrap_or(target)]);
        }
    }
    let sprite_list = &frame.snapshot.sprites;
    if let Some(sprites) = sprites.as_deref_mut() {
        #[allow(clippy::cast_precision_loss)]
        let aspect = size.0 as f32 / size.1.max(1) as f32;
        sprites.prepare(&gpu.device, &gpu.queue, sprite_list, aspect, frame.alpha);
        if sprites.has() {
            graph.add_pass(PassKind::Sprites, &[], &[hdr.unwrap_or(target)]);
        }
    }
    // Текст. В мире — в кадр сцены (HDR или цель) поверх неё, читая глубину; проходы
    // постэффектов читают этот кадр, поэтому идут после. Подписи и экранный текст пишут цель
    // без чтения: среди пишущих в цель решает порядок добавления — после постэффектов,
    // до интерфейса.
    let text_frame = &frame.snapshot.text;
    if let Some(text) = text.as_deref_mut() {
        let view_proj = crate::scene::camera_view_proj(frame.snapshot, size, frame.alpha);
        text.prepare(&gpu.device, &gpu.queue, text_frame, &view_proj, size);
    }
    let world_text = text.as_ref().is_some_and(|t| t.has(TextLayer::World));
    if world_text {
        graph.add_pass(PassKind::TextWorld, &[depth], &[hdr.unwrap_or(target)]);
    }
    if text.as_ref().is_some_and(|t| t.has(TextLayer::Overlay)) {
        graph.add_pass(PassKind::TextOverlay, &[], &[target]);
    }
    // Интерфейс читает цель (рисует поверх), поэтому идёт после всех, кто её пишет
    graph.add_pass(PassKind::Ui, &[target], &[target]);
    let order = graph.compile(target).unwrap_or_else(|error| {
        tracing::error!("Render graph: {error}");
        Vec::new()
    });
    let transients: std::collections::HashMap<_, _> = graph
        .transients()
        .map(|(id, name, desc)| (id, pool.get(&gpu.device, name, desc, size)))
        .collect();
    let view = |id: Option<crate::graph::ResourceId>| id.map(|id| &transients[&id]);

    if let Some(post) = &post {
        // Выход в sRGB-текстуру — оборудование само кодирует, шейдер пишет линейные значения
        let tonemap_linear = !settings.fxaa && frame.view_format.is_srgb();
        post.write_params(
            &gpu.queue,
            &settings,
            tonemap_linear,
            frame.view_format.is_srgb(),
        );
    }
    // Первый и последний проход цепочки постэффектов — для замера её времени целиком
    let post_passes: Vec<PassKind> = order
        .iter()
        .copied()
        .filter(|p| {
            matches!(
                p,
                PassKind::BloomDown(_) | PassKind::BloomUp(_) | PassKind::Tonemap | PassKind::Fxaa
            )
        })
        .collect();
    let post_timestamps = |pass: PassKind| {
        frame.timer.and_then(|t| {
            t.post_writes(
                post_passes.first() == Some(&pass),
                post_passes.last() == Some(&pass),
            )
        })
    };

    let mut draw = scene.prepare(&gpu.device, &gpu.queue, size, frame.snapshot, frame.alpha);
    let mut ui_commands = Vec::new();
    for pass in order {
        let _zone = tracing::trace_span!("pass", ?pass).entered();
        // [Rust] Не замыкание, а функция `post_pass`: у замыкания нельзя записать, что
        // результат живёт столько же, сколько ссылки-аргументы (`'a` в сигнатуре функции).
        let stamps = post_timestamps(pass);
        // [Rust] `post.as_deref_mut()` даёт `Option<&mut PostRenderer>` на время прохода,
        // не забирая значение из `post` — следующий проход тоже его получит.
        match pass {
            PassKind::Shadow => {
                scene.render_shadows(
                    encoder,
                    frame.snapshot,
                    frame.timer.and_then(|t| t.writes(PassKind::Shadow)),
                );
            }
            PassKind::Scene => {
                let keep_depth = world_text
                    || !frame.snapshot.particles.items.is_empty()
                    || !frame.snapshot.decals.items.is_empty();
                let target = crate::scene::FrameTarget {
                    color: view(hdr).unwrap_or(frame.view),
                    depth: &transients[&depth],
                    timestamps: frame.timer.and_then(|t| t.writes(PassKind::Scene)),
                    keep_depth,
                };
                scene.render(encoder, &target, frame.snapshot, frame.alpha, &mut draw);
            }
            PassKind::BloomDown(level) => {
                let level = usize::from(level);
                let (Some(post), Some(source), Some(out)) = (
                    post.as_deref_mut(),
                    if level == 0 {
                        view(hdr)
                    } else {
                        view(bloom[level - 1])
                    },
                    view(bloom[level]),
                ) else {
                    continue;
                };
                let entry = if level == 0 {
                    "fs_bloom_prefilter"
                } else {
                    "fs_bloom_down"
                };
                post.run(
                    &gpu.device,
                    encoder,
                    &post_pass(entry, source, None, (out, HDR_FORMAT), stamps),
                );
            }
            PassKind::BloomUp(level) => {
                let level = usize::from(level);
                let smaller = if level + 1 == BLOOM_LEVELS - 1 {
                    bloom[level + 1]
                } else {
                    bloom_up[level + 1]
                };
                let (Some(post), Some(source), Some(same), Some(out)) = (
                    post.as_deref_mut(),
                    view(smaller),
                    view(bloom[level]),
                    view(bloom_up[level]),
                ) else {
                    continue;
                };
                post.run(
                    &gpu.device,
                    encoder,
                    &post_pass("fs_bloom_up", source, Some(same), (out, HDR_FORMAT), stamps),
                );
            }
            PassKind::Tonemap => {
                let (Some(post), Some(source)) = (post.as_deref_mut(), view(hdr)) else {
                    continue;
                };
                let (out, format) = match view(ldr) {
                    Some(ldr) => (ldr, LDR_FORMAT),
                    None => (frame.view, frame.view_format),
                };
                let glow = view(bloom_up[0]);
                post.run(
                    &gpu.device,
                    encoder,
                    &post_pass("fs_tonemap", source, glow, (out, format), stamps),
                );
            }
            PassKind::Fxaa => {
                let (Some(post), Some(source)) = (post.as_deref_mut(), view(ldr)) else {
                    continue;
                };
                post.run(
                    &gpu.device,
                    encoder,
                    &post_pass(
                        "fs_fxaa",
                        source,
                        None,
                        (frame.view, frame.view_format),
                        stamps,
                    ),
                );
            }
            PassKind::Decals => {
                if let Some(decals) = decals.as_deref_mut() {
                    let color = view(hdr).unwrap_or(frame.view);
                    decals.render(&gpu.device, encoder, color, &transients[&depth], scene);
                }
            }
            PassKind::Particles => {
                if let Some(particles) = particles.as_deref_mut() {
                    let color = view(hdr).unwrap_or(frame.view);
                    particles.render(
                        &gpu.device,
                        encoder,
                        color,
                        &transients[&depth],
                        scene,
                        world_text,
                    );
                }
            }
            PassKind::Sprites => {
                if let Some(sprites) = sprites.as_deref_mut() {
                    let color = view(hdr).unwrap_or(frame.view);
                    sprites.render(&gpu.device, encoder, color, scene);
                }
            }
            PassKind::TextWorld => {
                if let Some(text) = text.as_deref() {
                    let color = view(hdr).unwrap_or(frame.view);
                    text.render(encoder, TextLayer::World, color, Some(&transients[&depth]));
                }
            }
            PassKind::TextOverlay => {
                if let Some(text) = text.as_deref() {
                    text.render(encoder, TextLayer::Overlay, frame.view, None);
                }
            }
            PassKind::Ui => {
                if let Some(ui) = ui.as_deref_mut() {
                    ui_commands = render_ui(
                        ui,
                        gpu,
                        encoder,
                        frame.view,
                        frame.size,
                        &frame.snapshot.ui,
                        frame.timer.and_then(|t| t.writes(PassKind::Ui)),
                    );
                }
            }
        }
    }
    (draw, ui_commands)
}

fn post_pass<'a>(
    entry: &'static str,
    source: &'a wgpu::TextureView,
    bloom: Option<&'a wgpu::TextureView>,
    (target, target_format): (&'a wgpu::TextureView, wgpu::TextureFormat),
    timestamps: Option<wgpu::RenderPassTimestampWrites<'a>>,
) -> PostPass<'a> {
    PostPass {
        entry,
        source,
        bloom,
        target,
        target_format,
        timestamps,
    }
}
