//! Рендер движка на `wgpu`.
//!
//! Схема (подробно — `docs/acrhitecture.md`, разделы 3–5, 11):
//! - главный поток в стадии `Extract` записывает снимок кадра в тройной буфер, а загруженные
//!   ресурсы (меши, текстуры, материалы) отправляет командами;
//! - рендер (в своём потоке или в главном — [`RenderMode`]) забирает свежий снимок,
//!   интерполирует трансформы, отсекает невидимое и рисует PBR;
//! - цель — окно или текстура в памяти (`render.headless`), кадр можно сохранить в PNG.
//!
//! Типы `wgpu` наружу не выходят.

mod caps;
mod color;
mod command;
mod debug_draw;
mod decals;
mod extract;
mod frontend;
mod gpu;
mod gpu_timer;
mod graph;
mod ktx;
mod lights;
mod particles;
mod post;
mod preprocess;
mod renderer;
mod scene;
mod screenshot;
mod shaders;
mod shadows;
mod snapshot;
mod sprites;
mod text;
mod ui;

pub use caps::{GpuTier, RENDER_TIER};
pub use color::Color;
pub use command::Screenshot;
pub use debug_draw::{DEBUG_AXES, DEBUG_BOUNDS, DEBUG_GRID, DebugDraw};
/// egui для своих окон: `engine::render::egui::Window::new(..)`.
pub use egui;
pub use frontend::{
    ClearColor, RENDER_BACKEND, RENDER_HEADLESS, RENDER_HEADLESS_HEIGHT, RENDER_HEADLESS_WIDTH,
    RENDER_MODE, RenderMode, RenderPlugin, RenderStats, SCREENSHOT_EXIT, SCREENSHOT_FRAME,
    SCREENSHOT_PATH, ScreenshotReceiver, request_screenshot,
};
pub use lights::RENDER_LOCAL_SHADOW_SIZE;
pub use post::{
    RENDER_BLOOM, RENDER_BLOOM_INTENSITY, RENDER_BLOOM_THRESHOLD, RENDER_EXPOSURE, RENDER_FXAA,
    RENDER_HDR,
};
pub use shadows::{
    RENDER_SHADOW_DEBUG, RENDER_SHADOW_DISTANCE, RENDER_SHADOW_SIZE, RENDER_SHADOWS,
};
pub use ui::{DEBUG_UI, Ui};

/// Сохраняет снимок в PNG (только native — на Web файловой системы нет).
pub fn save_png(shot: &Screenshot, path: &str) -> Result<(), String> {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (shot, path);
        Err("на Web сохранение файлов не поддерживается".to_owned())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        image::save_buffer(
            path,
            &shot.data,
            shot.width,
            shot.height,
            image::ExtendedColorType::Rgba8,
        )
        .map_err(|e| e.to_string())
    }
}
