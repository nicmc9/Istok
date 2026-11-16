//! Тест сравнения картинок (golden image, задача T-01): сцена рисуется без окна в текстуру
//! и сравнивается с эталоном `tests/golden/*.png` с допуском — у разных GPU и драйверов
//! пиксели немного отличаются.
//!
//! Нет эталона — тест создаёт его и падает: посмотрите картинку и закоммитьте.
//! Обновить эталон: удалить файл и запустить тест. Нет GPU — тест пропускается.

use engine_assets::formats::Gltf;
use engine_assets::{AssetPlugin, AssetServer};
use engine_core::{App, CVarSource, CVars};
use engine_math::Vec3;
use engine_render::{ClearColor, Color, RenderPlugin, RenderStats, request_screenshot};
use engine_scene::{Camera, DirectionalLight, ScenePlugin, SceneRoot, Transform};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

fn render_scene(model: &str) -> Option<engine_render::Screenshot> {
    let mut app = App::new();
    {
        let mut cvars = app.world_mut().resource_mut::<CVars>();
        for (name, value) in [
            ("render.headless", "true"),
            ("render.headless_width", &WIDTH.to_string()),
            ("render.headless_height", &HEIGHT.to_string()),
            // Тест запускается из папки крейта
            ("assets.root", "../../assets"),
            ("assets.strict", "true"),
            // Для проверки поведения без GPU: GOLDEN_BACKEND=gl на macOS (там GL нет)
            (
                "render.backend",
                &std::env::var("GOLDEN_BACKEND").unwrap_or_else(|_| "auto".to_owned()),
            ),
        ] {
            cvars.set(name, value, CVarSource::LaunchArguments).unwrap();
        }
    }
    app.add_plugin(engine_net::NetPlugin)
        .add_plugin(AssetPlugin)
        .add_plugin(ScenePlugin)
        .add_plugin(RenderPlugin)
        .insert_resource(ClearColor(Color::rgb(0.1, 0.1, 0.12)));
    app.update();

    let handle = app.world().resource::<AssetServer>().load::<Gltf>(model);
    let world = app.world_mut();
    world.spawn((SceneRoot(handle.clone()), Transform::default()));
    world.spawn((
        Camera::default(),
        Transform::from_xyz(1.6, 1.3, 2.2).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    world.spawn((
        DirectionalLight::default(),
        Transform::from_xyz(2.0, 4.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Кадры, пока не загрузится модель и рендер не нарисует её
    for _ in 0..2000 {
        app.update();
        let stats = *app.world().resource::<RenderStats>();
        if stats.drawn > 0 && stats.gpu_textures > 0 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    if app.world().resource::<RenderStats>().frames_rendered == 0 {
        eprintln!("Пропуск: нет GPU (кадр не нарисован)");
        return None;
    }
    let receiver = request_screenshot(app.world_mut());
    for _ in 0..100 {
        app.update();
        if let Some(result) = receiver.try_get() {
            return Some(result.expect("скриншот"));
        }
    }
    panic!("скриншот не получен");
}

fn compare(name: &str, shot: &engine_render::Screenshot) {
    let path = format!("{}/tests/golden/{name}.png", env!("CARGO_MANIFEST_DIR"));
    let Ok(golden) = image::open(&path) else {
        engine_render::save_png(shot, &path).unwrap();
        panic!("Эталона не было — создан {path}. Проверьте картинку и закоммитьте.");
    };
    let golden = golden.to_rgba8();
    assert_eq!(
        (golden.width(), golden.height()),
        (shot.width, shot.height),
        "размер кадра"
    );

    // Допуск: отличие пикселя больше 8/255 по любому каналу — «другой» пиксель.
    // Таких допускается не больше 0.5% (сглаживание краёв, точность драйвера).
    let different = golden
        .as_raw()
        .as_chunks::<4>()
        .0
        .iter()
        .zip(shot.data.as_chunks::<4>().0)
        .filter(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| x.abs_diff(*y) > 8))
        .count();
    let total = (shot.width * shot.height) as usize;
    #[allow(clippy::cast_precision_loss)]
    let share = different as f64 / total as f64;
    if share > 0.005 {
        // CARGO_TARGET_TMPDIR — временная папка интеграционных тестов внутри общего target/
        let actual = format!("{}/golden-{name}-actual.png", env!("CARGO_TARGET_TMPDIR"));
        let _ = engine_render::save_png(shot, &actual);
        panic!(
            "{name}: отличается {:.2}% пикселей (допуск 0.5%). Кадр: {actual}",
            share * 100.0
        );
    }
}

#[test]
fn golden_box_textured() {
    if let Some(shot) = render_scene("models/khronos/BoxTextured/BoxTextured.glb") {
        compare("box_textured", &shot);
    }
}

/// Две копии одного glTF — один вызов на примитив, не по вызову на копию.
#[test]
fn duplicates_share_one_draw() {
    assert_batched("models/khronos/BoxTextured/BoxTextured.glb");
    assert_batched("models/khronos/Fox/Fox.glb");
}

fn assert_batched(model: &str) {
    let Some(stats) = render_copies(model) else {
        return;
    };
    assert!(
        stats.drawn >= 2,
        "{model}: обе копии должны быть нарисованы, нарисовано {}",
        stats.drawn
    );
    assert!(
        stats.draw_calls < stats.drawn,
        "{model}: вызовов {} при {} примитивах — копии не склеились",
        stats.draw_calls,
        stats.drawn
    );
}

/// Две лисы: compute и вершинный шейдер дают одну картинку, оба пути склеивают вызов.
#[test]
fn fox_compute_matches_vertex_skinning() {
    let _reset = ResetSkinCompute;
    let Some((on, on_shot)) = render_fox(true) else {
        return;
    };
    let Some((off, off_shot)) = render_fox(false) else {
        return;
    };
    assert!(
        on.draw_calls < on.drawn,
        "compute: вызовов {} при {} примитивах",
        on.draw_calls,
        on.drawn
    );
    assert!(
        off.draw_calls < off.drawn,
        "вершинный: вызовов {} при {} примитивах",
        off.draw_calls,
        off.drawn
    );
    assert_eq!(
        off.skin_compute, 0,
        "render.skin_compute false всё ещё считает на compute"
    );
    if on.gpu_tier == engine_render::GpuTier::Compute {
        assert!(
            on.skin_compute >= 2,
            "две лисы должны попасть в бюджет compute, сейчас {}",
            on.skin_compute
        );
    }
    let share = difference_share(&on_shot, &off_shot);
    if share > 0.005 {
        let dir = env!("CARGO_TARGET_TMPDIR");
        let _ = engine_render::save_png(&on_shot, &format!("{dir}/fox-compute.png"));
        let _ = engine_render::save_png(&off_shot, &format!("{dir}/fox-vertex.png"));
        panic!(
            "лиса: compute и вершинный шейдер отличаются на {:.2}% пикселей. Кадры: {dir}/fox-*.png",
            share * 100.0
        );
    }
}

struct ResetSkinCompute;

impl Drop for ResetSkinCompute {
    fn drop(&mut self) {
        engine_render::RENDER_SKIN_COMPUTE.set(true);
    }
}

fn difference_share(left: &engine_render::Screenshot, right: &engine_render::Screenshot) -> f64 {
    assert_eq!(
        (left.width, left.height),
        (right.width, right.height),
        "размер кадра"
    );
    let different = left
        .data
        .as_chunks::<4>()
        .0
        .iter()
        .zip(right.data.as_chunks::<4>().0)
        .filter(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| x.abs_diff(*y) > 8))
        .count();
    let total = (left.width * left.height) as usize;
    #[allow(clippy::cast_precision_loss)]
    {
        different as f64 / total as f64
    }
}

fn render_fox(compute: bool) -> Option<(engine_render::RenderStats, engine_render::Screenshot)> {
    let mut app = App::new();
    {
        let mut cvars = app.world_mut().resource_mut::<CVars>();
        let flag = if compute { "true" } else { "false" };
        for (name, value) in [
            ("render.headless", "true"),
            ("render.headless_width", &WIDTH.to_string()),
            ("render.headless_height", &HEIGHT.to_string()),
            ("assets.root", "../../assets"),
            ("assets.strict", "true"),
            ("render.skin_compute", flag),
        ] {
            cvars.set(name, value, CVarSource::LaunchArguments).unwrap();
        }
    }
    app.add_plugin(engine_net::NetPlugin)
        .add_plugin(AssetPlugin)
        .add_plugin(ScenePlugin)
        .add_plugin(RenderPlugin)
        .insert_resource(ClearColor(Color::rgb(0.1, 0.1, 0.12)));
    app.update();

    let handle = app
        .world()
        .resource::<AssetServer>()
        .load::<Gltf>("models/khronos/Fox/Fox.glb");
    let world = app.world_mut();
    world.spawn((
        SceneRoot(handle.clone()),
        Transform::from_xyz(-40.0, 0.0, 0.0),
    ));
    world.spawn((SceneRoot(handle), Transform::from_xyz(40.0, 0.0, 0.0)));
    world.spawn((
        Camera::default(),
        Transform::from_xyz(0.0, 60.0, 220.0).looking_at(Vec3::new(0.0, 40.0, 0.0), Vec3::Y),
    ));
    world.spawn((
        DirectionalLight::default(),
        Transform::from_xyz(2.0, 4.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    for _ in 0..2000 {
        app.update();
        let stats = *app.world().resource::<engine_render::RenderStats>();
        if stats.drawn >= 2 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    if app
        .world()
        .resource::<engine_render::RenderStats>()
        .frames_rendered
        == 0
    {
        eprintln!("Пропуск: нет GPU (кадр не нарисован)");
        return None;
    }
    let stats = *app.world().resource::<engine_render::RenderStats>();
    let receiver = request_screenshot(app.world_mut());
    for _ in 0..100 {
        app.update();
        if let Some(result) = receiver.try_get() {
            return Some((stats, result.expect("скриншот")));
        }
    }
    panic!("скриншот не получен");
}

fn render_copies(model: &str) -> Option<engine_render::RenderStats> {
    let mut app = App::new();
    {
        let mut cvars = app.world_mut().resource_mut::<CVars>();
        for (name, value) in [
            ("render.headless", "true"),
            ("render.headless_width", &WIDTH.to_string()),
            ("render.headless_height", &HEIGHT.to_string()),
            ("assets.root", "../../assets"),
            ("assets.strict", "true"),
        ] {
            cvars.set(name, value, CVarSource::LaunchArguments).unwrap();
        }
    }
    app.add_plugin(engine_net::NetPlugin)
        .add_plugin(AssetPlugin)
        .add_plugin(ScenePlugin)
        .add_plugin(RenderPlugin)
        .insert_resource(ClearColor(Color::rgb(0.1, 0.1, 0.12)));
    app.update();

    let handle = app.world().resource::<AssetServer>().load::<Gltf>(model);
    let world = app.world_mut();
    world.spawn((
        SceneRoot(handle.clone()),
        Transform::from_xyz(-1.5, 0.0, 0.0),
    ));
    world.spawn((SceneRoot(handle), Transform::from_xyz(1.5, 0.0, 0.0)));
    world.spawn((
        Camera::default(),
        Transform::from_xyz(4.0, 2.5, 6.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    world.spawn((
        DirectionalLight::default(),
        Transform::from_xyz(2.0, 4.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    for _ in 0..2000 {
        app.update();
        let stats = *app.world().resource::<engine_render::RenderStats>();
        if stats.drawn >= 2 {
            return Some(stats);
        }
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    if app
        .world()
        .resource::<engine_render::RenderStats>()
        .frames_rendered
        == 0
    {
        eprintln!("Пропуск: нет GPU (кадр не нарисован)");
        return None;
    }
    Some(*app.world().resource::<engine_render::RenderStats>())
}
