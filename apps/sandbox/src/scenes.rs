//! Сцены sandbox.

use std::borrow::Cow;

use engine::CVarString;
use engine::assets::formats::{AnimationClip, ColorSpace, Gltf, Image, Material, Mesh};
use engine::math::{Aabb, Vec2, Vec3};
use engine::prelude::*;
use engine::scene::{
    AnimationPlayer, Curve, Keys, PropertyClip, PropertyPlayer, PropertyTrack, SceneNodes,
    SpriteClip, SpritePlayer,
};

use crate::camera::OrbitCamera;

pub static SCENE: CVar<CVarString> = CVar::new(
    "sandbox.scene",
    Cow::Borrowed("showcase"),
    "Сцена: showcase, orcs, sponza, helmet, sprites, fx или файл scenes/<имя>.scene",
);

/// Доля клипа `0..1`, с которой стартует анимация. Без неё все копии начинают с нуля.
#[derive(Component)]
pub(crate) struct AnimationPhase(f32);

/// Вписать модель: высота и точка, куда поставить основание. Сохраняется в сцене: модель
/// из файла подгонится так же, даже если сохранили до подгонки.
#[derive(Component, Reflect)]
#[reflect(Component)]
pub struct FitTo {
    pub height: f32,
    pub position: Vec3,
}

#[allow(clippy::too_many_arguments)] // система ECS: параметры — запрашиваемые данные
pub fn setup(
    mut commands: Commands,
    server: Res<AssetServer>,
    mut ambient: ResMut<AmbientLight>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<Material>>,
    mut clips: ResMut<Assets<PropertyClip>>,
    mut images: ResMut<Assets<Image>>,
    mut atlases: ResMut<Assets<Atlas>>,
    mut sprite_clips: ResMut<Assets<SpriteClip>>,
) {
    let scene = SCENE.get();
    engine::log::info!("Сцена: {scene}");
    // Ресурсы, созданные кодом, — с именами: их находят и сохранённые сцены
    let (mut demo, bounce) = crate::physics::create_assets(&mut meshes, &mut materials, &server);
    commands.insert_resource(bounce);
    if scene.as_ref() == "sprites" {
        spawn_sprites(&mut commands, &mut images, &mut atlases, &mut sprite_clips);
        commands.insert_resource(demo);
        return;
    }
    if scene.ends_with(".scene") {
        // Сцена из файла (`sandbox.scene=scenes/имя.scene`) — через сервер ресурсов,
        // поэтому работает на всех платформах
        commands.spawn(LoadScene(server.load(scene.as_ref())));
        commands.insert_resource(demo);
        return;
    }
    let mut model = |path: &str, height: f32, position: Vec3| {
        commands.spawn((
            SceneRoot(server.load(path)),
            FitTo { height, position },
            Transform::default(),
        ));
    };
    let (target, distance) = match scene.as_ref() {
        "sponza" => {
            model("models/khronos/Sponza/Sponza.gltf", 12.0, Vec3::ZERO);
            ambient.intensity = 0.3;
            (Vec3::new(0.0, 2.0, 0.0), 6.0)
        }
        "helmet" => {
            model(
                "models/khronos/DamagedHelmet/DamagedHelmet.glb",
                2.0,
                Vec3::ZERO,
            );
            (Vec3::new(0.0, 1.0, 0.0), 4.0)
        }
        "fx" => (Vec3::new(0.0, 0.4, 0.0), 3.5),
        "orcs" => {
            spawn_orc_crowd(&mut commands, &server);
            // Один направленный источник: рассеянный свет гасим, иначе это второй свет.
            ambient.intensity = 0.0;
            // Поле шире дальности теней по умолчанию (40 м) — иначе дальние ряды без тени.
            engine::render::RENDER_SHADOW_DISTANCE.set(250.0);
            (Vec3::new(0.0, 1.0, 0.0), 120.0)
        }
        _ => {
            model(
                "models/khronos/DamagedHelmet/DamagedHelmet.glb",
                1.6,
                Vec3::new(0.0, 0.0, 0.0),
            );
            model("models/orc/orc.glb", 2.0, Vec3::new(-2.2, 0.0, 0.0));
            model("models/khronos/Fox/Fox.glb", 1.0, Vec3::new(2.2, 0.0, 0.0));
            model(
                "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/8f492390ddc7a2ae221a1369ef560589cf8c70fb/Models/Duck/glTF-Binary/Duck.glb",
                0.8,
                Vec3::new(1.2, 0.0, 1.6),
            );
            model(
                "models/khronos/BoxTextured/BoxTextured.glb",
                0.6,
                Vec3::new(-1.2, 0.0, 1.6),
            );
            spawn_texts(&mut commands, &server);
            (Vec3::new(0.0, 0.9, 0.0), 6.5)
        }
    };

    // Пол — чтобы было видно тени (в Sponza свой)
    if scene.as_ref() == "orcs" {
        // Прямоугольник 40×25 с шагом 3 м: около 117×72 м.
        let floor = Mesh::plane(160.0);
        commands.spawn((
            RigidBody::Fixed,
            Collider::trimesh(&floor),
            Mesh3d {
                mesh: meshes.add(floor),
                materials: vec![Some(demo.floor_material.clone())],
            },
            Transform::default(),
        ));
    } else if scene.as_ref() != "sponza" {
        commands.spawn((
            // Неподвижное тело из треугольников того же меша, что рисуется
            RigidBody::Fixed,
            Collider::trimesh(&Mesh::plane(crate::physics::FLOOR_SIZE)),
            Mesh3d {
                mesh: demo.floor_mesh.clone(),
                materials: vec![Some(demo.floor_material.clone())],
            },
            Transform::default(),
        ));
    }

    if !matches!(scene.as_ref(), "sponza" | "helmet" | "orcs") {
        let at = if scene.as_ref() == "fx" {
            Vec3::ZERO
        } else {
            Vec3::new(0.0, 0.0, 2.3)
        };
        spawn_effects(&mut commands, &mut images, at);
    }
    if !matches!(scene.as_ref(), "sponza" | "helmet" | "fx" | "orcs") {
        crate::physics::spawn_pyramid(&mut commands, &mut demo);
    }
    commands.insert_resource(demo);
    // Фоновая музыка: по кругу, шина Music (громкость — `audio.music_volume`). На Web
    // заиграет после первого клика или касания. Без audio.enabled плеер не запускается.
    if engine::audio::AUDIO_ENABLED.get() && scene.as_ref() != "orcs" {
        commands.spawn(AudioPlayer::music(server.load("audio/ambient_loop.wav")).with_volume(0.5));
    }

    commands.spawn((
        Camera::default(),
        {
            let mut orbit = OrbitCamera::new(target, distance);
            // В `fx` камера смотрит круче вниз, иначе пятно на полу схлопывается в полоску.
            if scene.as_ref() == "fx" {
                orbit.pitch = -0.55;
            } else if scene.as_ref() == "orcs" {
                // Сверху и вдоль рядов, чтобы прямоугольник сетки был в кадре целиком.
                orbit.yaw = 0.0;
                orbit.pitch = -1.0;
            }
            orbit
        },
        Transform::default(),
    ));
    if scene.as_ref() == "orcs" {
        commands.spawn((
            DirectionalLight::default(),
            Transform::from_xyz(4.0, 8.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
        ));
    } else {
        spawn_lights(&mut commands, &mut clips);
    }
}

/// Тысяча копий одного орка. Файл грузится один раз, каждый экземпляр — своё дерево узлов.
fn spawn_orc_crowd(commands: &mut Commands, server: &AssetServer) {
    const COUNT: usize = 1000;
    const COLUMNS: usize = 40;
    // Шире силуэта орка (~0.8 м при росте 2 м), иначе ряды сливаются.
    const SPACING: f32 = 3.0;
    const HEIGHT: f32 = 2.0;

    let rows = COUNT.div_ceil(COLUMNS);
    let width = f32::from(u16::try_from(COLUMNS - 1).unwrap_or(0)) * SPACING;
    let depth = f32::from(u16::try_from(rows - 1).unwrap_or(0)) * SPACING;
    let origin = Vec3::new(-width * 0.5, 0.0, -depth * 0.5);
    let orc = server.load("models/orc/orc.glb");
    for index in 0..COUNT {
        let column = index % COLUMNS;
        let row = index / COLUMNS;
        let position = origin
            + Vec3::new(
                f32::from(u16::try_from(column).unwrap_or(0)) * SPACING,
                0.0,
                f32::from(u16::try_from(row).unwrap_or(0)) * SPACING,
            );
        let step = f32::from(u16::try_from(index).unwrap_or(0));
        commands.spawn((
            SceneRoot(orc.clone()),
            FitTo {
                height: HEIGHT,
                position,
            },
            // Соседи по сетке идут подряд, шаг 0.618 разводит их по циклу.
            AnimationPhase((step * 0.618_034).fract()),
            Transform::default(),
        ));
    }
}

/// Фонтан и пятно на полу. `at` — куда поставить: в `fx` в начало координат,
/// в витрине — перед камерой, чтобы не пряталось в моделях.
fn spawn_effects(commands: &mut Commands, images: &mut Assets<Image>, at: Vec3) {
    let spark = images.add_named("sandbox://fx/spark", soft_disc(64));
    let stamp = images.add_named("sandbox://fx/stamp", ring(64));
    let mut emitter = ParticleEmitter::new(spark);
    emitter.rate = 36.0;
    emitter.lifetime = 1.3;
    emitter.speed = 2.2;
    emitter.size = 0.28;
    // Выше 1: после тональной компрессии они ярче освещённого пола, иначе тонут в нём.
    emitter.color = [6.0, 2.2, 0.5, 1.0];
    emitter.gravity = 3.2;
    emitter.max = 160;
    commands.spawn((emitter, Transform::from_translation(at + Vec3::Y * 0.15)));
    // Локальный Z смотрит вверх: XY ложатся на пол, толщина — по вертикали.
    let mut decal = Decal::new(stamp);
    decal.color = [4.0, 0.25, 0.15, 1.0];
    let place = at + Vec3::new(0.55, 0.0, 0.15);
    commands.spawn((
        decal,
        Transform::from_translation(place)
            .looking_at(place + Vec3::NEG_Y, Vec3::Z)
            .with_scale(Vec3::new(1.1, 1.1, 0.3)),
    ));
}

fn soft_disc(size: u32) -> Image {
    let mut data = vec![0u8; (size * size * 4) as usize];
    let center = f32::from(u16::try_from(size / 2).unwrap_or(0));
    let radius = center.max(1.0);
    for y in 0..size {
        for x in 0..size {
            let dx = f32::from(u16::try_from(x).unwrap_or(0)) - center;
            let dy = f32::from(u16::try_from(y).unwrap_or(0)) - center;
            let t = 1.0 - (dx * dx + dy * dy).sqrt() / radius;
            let alpha = byte(t.max(0.0) * t.max(0.0));
            let offset = ((y * size + x) * 4) as usize;
            data[offset..offset + 4].copy_from_slice(&[255, 255, 255, alpha]);
        }
    }
    image_with_mips(size, data)
}

fn ring(size: u32) -> Image {
    let mut data = vec![0u8; (size * size * 4) as usize];
    let center = f32::from(u16::try_from(size / 2).unwrap_or(0));
    let radius = center.max(1.0);
    for y in 0..size {
        for x in 0..size {
            let dx = f32::from(u16::try_from(x).unwrap_or(0)) - center;
            let dy = f32::from(u16::try_from(y).unwrap_or(0)) - center;
            let t = (dx * dx + dy * dy).sqrt() / radius;
            let alpha = if (0.2..0.8).contains(&t) { 255 } else { 0 };
            let offset = ((y * size + x) * 4) as usize;
            data[offset..offset + 4].copy_from_slice(&[255, 255, 255, alpha]);
        }
    }
    // Без мипов: дальние уровни усредняют кольцо в пятно.
    Image {
        width: size,
        height: size,
        color_space: ColorSpace::Srgb,
        data,
        mips: Vec::new(),
        basis: Vec::new(),
    }
}

fn image_with_mips(size: u32, data: Vec<u8>) -> Image {
    let mut image = Image {
        width: size,
        height: size,
        color_space: ColorSpace::Srgb,
        data,
        mips: Vec::new(),
        basis: Vec::new(),
    };
    image.generate_mips();
    image
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn byte(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Плоская сцена: атлас 2×2, четыре кадра и один спрайт, который их перебирает.
fn spawn_sprites(
    commands: &mut Commands,
    images: &mut Assets<Image>,
    atlases: &mut Assets<Atlas>,
    clips: &mut Assets<SpriteClip>,
) {
    let image = images.add_named("sandbox://sprites", quadrants(64));
    let mut white_image = Image {
        width: 1,
        height: 1,
        color_space: ColorSpace::Srgb,
        data: vec![255, 255, 255, 255],
        mips: Vec::new(),
        basis: Vec::new(),
    };
    white_image.generate_mips();
    let white = images.add(white_image);
    let atlas = Atlas::grid(image.clone(), 2, 2);
    // Фон закрывает вид камеры высотой 10 при широком окне
    commands.spawn((
        Sprite::new(white, Vec2::new(40.0, 24.0))
            .with_color([0.15, 0.16, 0.2, 1.0])
            .with_layer(-10),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
    let places = [(-2.2, 1.6), (2.2, 1.6), (-2.2, -1.6), (2.2, -1.6)];
    for (frame, (x, y)) in places.into_iter().enumerate() {
        let sprite = atlas
            .sprite(frame, Vec2::splat(1.5))
            .expect("кадр сетки 2×2");
        commands.spawn((sprite, Transform::from_xyz(x, y, 0.0)));
    }
    let clip = clips.add(SpriteClip::grid(2, 2, 0.35));
    commands.spawn((
        Sprite::new(image, Vec2::splat(2.2))
            .with_uv(atlas.frames[0])
            .with_layer(1),
        SpritePlayer::new(clip),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
    atlases.add_named("sandbox://sprites.atlas", atlas);
    commands.spawn((
        Camera2d {
            height: 10.0,
            ..Camera2d::default()
        },
        Transform::from_xyz(0.0, 0.0, 10.0),
    ));
}

/// Четыре цветных квадрата: слева направо, сверху вниз.
fn quadrants(size: u32) -> Image {
    let mut data = vec![0u8; (size * size * 4) as usize];
    let half = size / 2;
    for y in 0..size {
        for x in 0..size {
            let color: [u8; 4] = match (x < half, y < half) {
                (true, true) => [220, 40, 40, 255],
                (false, true) => [40, 180, 60, 255],
                (true, false) => [40, 80, 220, 255],
                (false, false) => [230, 190, 40, 255],
            };
            let offset = ((y * size + x) * 4) as usize;
            data[offset..offset + 4].copy_from_slice(&color);
        }
    }
    let mut image = Image {
        width: size,
        height: size,
        color_space: ColorSpace::Srgb,
        data,
        mips: Vec::new(),
        basis: Vec::new(),
    };
    image.generate_mips();
    image
}

/// Вывеска в мире — её текст меняет команда консоли `sign`.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Default)]
pub struct Sign;

/// Команды консоли sandbox — примеры для своих команд.
pub fn console_commands() -> [ConsoleCommand; 2] {
    [
        ConsoleCommand::new(
            "sign",
            "Текст вывески в витрине",
            |world, args| {
                if args.is_empty() {
                    return Err("использование: sign <текст>".to_owned());
                }
                let content = args.join(" ");
                // [Rust] Запрос к миру напрямую (вне системы): `query_filtered` + `iter_mut`
                let mut signs = world.query_filtered::<&mut Text, With<Sign>>();
                let mut count = 0;
                for mut text in signs.iter_mut(world) {
                    text.content.clone_from(&content);
                    count += 1;
                }
                if count == 0 {
                    return Err("вывески нет (она только в сцене showcase)".to_owned());
                }
                Ok(format!("вывеска: {content}"))
            },
        )
        .with_usage("sign <текст>"),
        ConsoleCommand::new(
            "sun",
            "Яркость солнца (пример чит-команды)",
            |world, args| {
                let [value] = args else {
                    return Err("использование: sun <яркость>".to_owned());
                };
                let intensity: f32 = value.parse().map_err(|_| format!("«{value}» — не число"))?;
                let mut suns = world.query::<&mut DirectionalLight>();
                for mut sun in suns.iter_mut(world) {
                    sun.intensity = intensity;
                }
                Ok(format!("солнце: {intensity}"))
            },
        )
        .with_usage("sun <яркость>")
        .cheat(),
    ]
}

/// Текст витрины: вывеска в мире и подписи над моделями.
fn spawn_texts(commands: &mut Commands, server: &AssetServer) {
    // В мире: закрывается моделями, освещения нет, но проходит тональную компрессию
    let noto = server.load("fonts/NotoSans-Regular.ttf");
    commands.spawn((
        Sign,
        Text::new("Istok — текст в мире", 0.35)
            .with_font(noto)
            .with_color([1.0, 0.85, 0.4, 1.0]),
        Transform::from_xyz(0.0, 2.3, -1.0),
    ));
    // Подписи: размер в точках экрана, всегда лицом к камере, поверх сцены
    for (name, position) in [
        ("Шлем", Vec3::new(0.0, 1.8, 0.0)),
        ("Орк", Vec3::new(-2.2, 2.2, 0.0)),
        ("Лиса", Vec3::new(2.2, 1.2, 0.0)),
        ("Утка (из сети)", Vec3::new(1.2, 1.0, 1.6)),
    ] {
        commands.spawn((
            Text::label(name, 16.0).with_outline(0.1, [0.0, 0.0, 0.0, 1.0]),
            Transform::from_translation(position),
        ));
    }
}

/// Солнце, точечные источники и прожекторы витрины.
fn spawn_lights(commands: &mut Commands, clips: &mut Assets<PropertyClip>) {
    commands.spawn((
        DirectionalLight::default(),
        Transform::from_xyz(4.0, 8.0, 5.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    // Тёплый точечный с тенями (6 граней куба) и холодный без
    commands.spawn((
        PointLight {
            color: [1.0, 0.6, 0.3],
            intensity: 30.0,
            range: 12.0,
            shadows: true,
        },
        Transform::from_xyz(-3.0, 2.5, 2.0),
    ));
    // Холодный свет дышит: кривая пишет в поле компонента (R2-06)
    let pulse = clips.add(PropertyClip::new(
        "pulse",
        vec![PropertyTrack::new(
            "PointLight",
            "intensity",
            Curve::Float(Keys::linear(vec![0.0, 1.0, 2.0], vec![8.0, 40.0, 8.0])),
        )],
    ));
    commands.spawn((
        PointLight {
            color: [0.3, 0.6, 1.0],
            intensity: 8.0,
            range: 12.0,
            ..PointLight::default()
        },
        PropertyPlayer::new(pulse),
        Transform::from_xyz(3.0, 2.0, 3.0),
    ));
    // Прожекторы сверху на шлем и на лису
    commands.spawn((
        SpotLight {
            color: [1.0, 0.95, 0.8],
            intensity: 60.0,
            range: 15.0,
            ..SpotLight::default()
        },
        Transform::from_xyz(0.5, 5.0, 2.5).looking_at(Vec3::new(0.0, 0.5, 0.0), Vec3::Y),
    ));
    commands.spawn((
        SpotLight {
            color: [0.4, 1.0, 0.5],
            intensity: 40.0,
            range: 12.0,
            inner_angle: 0.15,
            outer_angle: 0.3,
            ..SpotLight::default()
        },
        Transform::from_xyz(3.5, 4.0, 1.5).looking_at(Vec3::new(2.2, 0.0, 0.0), Vec3::Y),
    ));
}

/// Когда сцена модели развернулась, считает её границы и масштабирует корень так,
/// чтобы модель стояла основанием на точке `position` и имела высоту `height`.
pub fn fit_models(
    mut commands: Commands,
    mut roots: Query<(Entity, &FitTo, &mut Transform, Ref<SceneSpawned>)>,
    children: Query<&Children>,
    objects: Query<(&Mesh3d, &GlobalTransform)>,
    meshes: Res<Assets<Mesh>>,
) {
    for (root, fit, mut transform, spawned) in &mut roots {
        // [Rust] `Ref<T>` — доступ на чтение с флагами изменений. В кадре, когда сцену
        // развернули, мировые трансформы потомков ещё не посчитаны — ждём следующий.
        if spawned.is_added() {
            continue;
        }
        // Границы в мире при текущем (единичном) трансформе корня
        let mut bounds = Aabb::EMPTY;
        let mut stack = vec![root];
        while let Some(entity) = stack.pop() {
            if let Ok((mesh3d, global)) = objects.get(entity)
                && let Some(mesh) = meshes.get_loaded(&mesh3d.mesh)
            {
                for primitive in &mesh.primitives {
                    bounds = bounds.union(&primitive.bounds.transformed(&global.0));
                }
            }
            if let Ok(kids) = children.get(entity) {
                stack.extend(kids.iter());
            }
        }
        if bounds.is_empty() {
            continue;
        }
        let size = bounds.max - bounds.min;
        let scale = fit.height / size.y.max(1e-4);
        let base = Vec3::new(bounds.center().x, bounds.min.y, bounds.center().z);
        transform.scale = Vec3::splat(scale);
        transform.translation = fit.position - base * scale;
        engine::log::debug!(
            "Модель вписана: размер {:.2?} → масштаб {scale:.4}",
            size.to_array()
        );
        commands.entity(root).remove::<FitTo>();
    }
}

/// Развёрнутые сцены, у которых ещё нет проигрывателя.
// [Rust] Псевдоним типа для длинного фильтра запроса — как `using` в C++
type WithoutPlayer = (With<SceneNodes>, Without<AnimationPlayer>);

/// Как только сцена с анимациями развернулась — запускаем первый клип.
pub fn autoplay(
    mut commands: Commands,
    roots: Query<(Entity, &SceneRoot, Option<&AnimationPhase>), WithoutPlayer>,
    gltfs: Res<Assets<Gltf>>,
    clips: Res<Assets<AnimationClip>>,
) {
    for (entity, root, phase) in &roots {
        let Some(clip) = gltfs.get_loaded(&root.0).and_then(|g| g.animations.first()) else {
            continue;
        };
        let mut player = AnimationPlayer::default();
        player.play(clip.clone());
        if let Some(phase) = phase
            && let Some(asset) = clips.get_loaded(clip)
            && asset.duration > 0.0
        {
            player.seek(phase.0 * asset.duration);
        }
        commands.entity(entity).insert(player);
    }
}

/// N — плавный переход на следующий клип у всех моделей.
pub fn next_clip(
    input: Res<Input>,
    gltfs: Res<Assets<Gltf>>,
    clips: Res<Assets<AnimationClip>>,
    mut players: Query<(&SceneRoot, &mut AnimationPlayer, Option<&AnimationPhase>)>,
) {
    if !input.key_pressed(Key::N) {
        return;
    }
    for (root, mut player, phase) in &mut players {
        let Some(gltf) = gltfs.get_loaded(&root.0) else {
            continue;
        };
        let current = player
            .current_clip()
            .and_then(|c| gltf.animations.iter().position(|h| h == c))
            .unwrap_or(0);
        let next = gltf.animations[(current + 1) % gltf.animations.len()].clone();
        if let Some(clip) = clips.get_loaded(&next) {
            engine::log::info!("Анимация: {}", clip.name);
        }
        player.crossfade(next.clone(), 0.4);
        if let Some(phase) = phase
            && let Some(clip) = clips.get_loaded(&next)
            && clip.duration > 0.0
        {
            player.seek(phase.0 * clip.duration);
        }
    }
}
