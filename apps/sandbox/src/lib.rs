//! Тестовое приложение: 3D-сцена из моделей glTF, орбитальная камера, статистика.
//!
//! Сцена выбирается CVar `sandbox.scene`: `showcase` (по умолчанию), `orcs`, `sponza`, `helmet`, `sprites`.
//! Скриншот: `--cvar render.screenshot=shot.png --cvar render.screenshot_exit=true`.

mod camera;
mod physics;
mod save;
mod scenes;

use engine::platform::{LaunchContext, PlatformConfig, PlatformError};
use engine::prelude::*;

/// Общий запуск для всех платформ.
pub fn run(launch: LaunchContext) -> Result<(), PlatformError> {
    let mut app = App::new();
    // Файл встроен в бинарник: так он доступен на всех платформах
    app.load_config_toml(include_str!("../sandbox.toml"))
        .add_plugin(NetPlugin)
        .add_plugin(AssetPlugin)
        .add_plugin(ScenePlugin)
        .add_plugin(Plugin2d)
        .add_plugin(InputPlugin)
        .add_plugin(KiraAudioPlugin)
        .add_plugin(RapierPhysicsPlugin)
        .add_plugin(RenderPlugin)
        .add_plugin(TextPlugin)
        .add_plugin(SerializePlugin)
        .add_plugin(DebugPlugin)
        .add_plugin(ConsolePlugin)
        .add_plugin(SandboxPlugin);
    // CVar из аргументов применяется при регистрации в RenderPlugin, раньше его не прочитать.
    let headless = engine::render::RENDER_HEADLESS.get();
    let mut config = PlatformConfig {
        title: "Sandbox".to_owned(),
        headless,
        ..PlatformConfig::default() // [Rust] остальные поля — из значения по умолчанию
    };
    if headless {
        config.width = engine::render::RENDER_HEADLESS_WIDTH.get();
        config.height = engine::render::RENDER_HEADLESS_HEIGHT.get();
    }
    app.run(config, launch)
}

struct SandboxPlugin;

impl Plugin for SandboxPlugin {
    fn build(&self, app: &mut App) {
        for command in scenes::console_commands() {
            app.add_console_command(command);
        }
        app.add_console_command(physics::drop_command());
        for command in save::console_commands() {
            app.add_console_command(command);
        }
        // Компоненты sandbox — в сохранённые сцены
        app.register_type::<camera::OrbitCamera>()
            .register_type::<physics::Ball>()
            .register_type::<scenes::Sign>()
            .register_type::<scenes::FitTo>()
            .register_type::<Hud>();
        app.register_cvar(&scenes::SCENE)
            .insert_resource(ClearColor(Color::rgb(0.05, 0.06, 0.08)))
            .init_resource::<FrameStats>()
            .add_systems(Stage::Startup, scenes::setup)
            .add_systems(
                Stage::Update,
                (
                    scenes::fit_models,
                    scenes::autoplay,
                    scenes::next_clip,
                    camera::orbit_camera,
                    report_stats,
                    update_hud,
                    physics::despawn_fallen,
                ),
            );
        // События столкновений регистрирует плагин физики — без него система не соберётся в мир
        if engine::physics::PHYSICS_ENABLED.get() {
            app.add_systems(Stage::Update, physics::bounce_sounds);
        }
    }
}

/// Экранный счётчик кадров (пример текста `TextMode::Screen`).
#[derive(Component, Reflect, Default)]
#[reflect(Component, Default)]
struct Hud;

/// Раз в полсекунды — кадры рендера в секунду и число звуков. Текст меняется редко: вёрстка
/// пересчитывается только при изменении `Text`.
fn update_hud(
    mut commands: Commands,
    time: Res<Time>,
    render: Res<RenderStats>,
    audio: Option<Res<engine::audio::AudioStats>>,
    mut hud: Query<&mut Text, With<Hud>>,
    // (время следующего обновления, кадров рендера на тот момент)
    mut last: Local<(f32, u64)>,
) {
    let now = time.elapsed().as_secs_f32();
    if now < last.0 {
        return;
    }
    let fps = (render.frames_rendered - last.1) * 2;
    *last = (now + 0.5, render.frames_rendered);
    let sound = match audio.as_deref() {
        Some(stats) if stats.started => format!("звуков: {}", stats.playing),
        Some(_) => "звук выключен".to_owned(),
        None => "звук не подключён".to_owned(),
    };
    let content = format!("Кадров в секунду: {fps} · {sound}");
    match hud.single_mut() {
        Ok(mut text) => text.content = content,
        Err(_) => {
            commands.spawn((
                Text::screen(content, 18.0, Vec2::new(12.0, 12.0))
                    .with_outline(0.08, [0.0, 0.0, 0.0, 0.8]),
                Hud,
            ));
        }
    }
}

#[derive(Resource, Default)]
struct FrameStats {
    frames: u32,
    last_report_rendered: u64,
    next_report_secs: f32,
}

fn report_stats(
    time: Res<Time>,
    server: Res<AssetServer>,
    render: Res<RenderStats>,
    net: Res<NetStats>,
    mut stats: ResMut<FrameStats>,
) {
    stats.frames += 1;
    let now = time.elapsed().as_secs_f32();
    if now < stats.next_report_secs {
        return;
    }
    let (loading, ready, failed) = server.counts();
    engine::log::info!(
        "кадры: логика {} ({:.2} мс), рендер {} ({:.2} мс) | примитивов: нарисовано {}, отсечено {} | GPU: мешей {}, текстур {} | ресурсы: {ready} готово, {loading} грузится, {failed} ошибок | сеть: {} КБ",
        stats.frames,
        time.cpu_time().as_secs_f64() * 1000.0,
        render.frames_rendered - stats.last_report_rendered,
        render.frame_cpu_time.as_secs_f64() * 1000.0,
        render.drawn,
        render.culled,
        render.gpu_meshes,
        render.gpu_textures,
        net.bytes_downloaded / 1024,
    );
    stats.frames = 0;
    stats.last_report_rendered = render.frames_rendered;
    stats.next_report_secs = now + 1.0;
}

/// Точка входа на Android. Её по имени ищет `android-activity` после загрузки .so,
/// функции `main` на Android нет.
#[cfg(target_os = "android")]
// no_mangle небезопасен: два символа с одним именем — UB при линковке.
// ABI именно Rust, а не C: android-activity объявляет функцию как `extern "Rust"`.
#[allow(unsafe_code, clippy::no_mangle_with_rust_abi)]
#[unsafe(no_mangle)]
fn android_main(app: engine::platform::AndroidApp) {
    if let Err(error) = run(LaunchContext::android(app)) {
        engine::log::error!("Приложение завершилось с ошибкой: {error}");
    }
}

/// Точка входа на Web. `wasm-bindgen` вызывает её сразу после загрузки модуля.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn web_start() {
    if let Err(error) = run(LaunchContext::default()) {
        engine::log::error!("Приложение завершилось с ошибкой: {error}");
    }
}
