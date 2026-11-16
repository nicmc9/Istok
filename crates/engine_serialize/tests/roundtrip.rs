//! Сохранение → удаление → загрузка: значения, иерархия, хэндлы, исключения, ошибки.

use engine_assets::formats::{Gltf, Mesh};
use engine_assets::{AssetPlugin, AssetServer, Assets};
use engine_core::App;
use engine_ecs::prelude::*;
use engine_math::{Quat, Vec3};
use engine_scene::{Mesh3d, PointLight, ScenePlugin, SceneRoot, Transform};
use engine_serialize::{SerializePlugin, load_scene, save_scene};

fn app() -> App {
    let mut app = App::new();
    app.add_plugin(engine_net::NetPlugin)
        .add_plugin(AssetPlugin)
        .add_plugin(ScenePlugin)
        .add_plugin(SerializePlugin);
    app.update();
    app
}

fn by_name(world: &mut World, name: &str) -> Entity {
    let mut query = world.query::<(Entity, &Name)>();
    query
        .iter(world)
        .find_map(|(e, n)| (n.as_str() == name).then_some(e))
        .unwrap_or_else(|| panic!("нет сущности «{name}»"))
}

fn count(world: &mut World) -> usize {
    world.query::<Entity>().iter(world).count()
}

fn despawn_all(world: &mut World) {
    let all: Vec<Entity> = world
        .query_filtered::<Entity, With<Transform>>()
        .iter(world)
        .collect();
    for entity in all {
        // Потомки могли уйти вместе с родителем
        if world.get_entity(entity).is_ok() {
            world.despawn(entity);
        }
    }
}

#[test]
fn values_and_hierarchy_survive_roundtrip() {
    let mut app = app();
    let world = app.world_mut();
    let transform = Transform {
        translation: Vec3::new(1.0, 2.0, 3.0),
        rotation: Quat::from_rotation_y(0.5),
        scale: Vec3::splat(2.0),
    };
    let parent = world
        .spawn((
            Name::new("Родитель"),
            transform,
            PointLight {
                intensity: 7.5,
                ..PointLight::default()
            },
        ))
        .id();
    world.spawn((
        Name::new("Ребёнок"),
        Transform::from_xyz(0.0, 1.0, 0.0),
        ChildOf(parent),
    ));

    let text = save_scene(world).unwrap();
    assert!(
        text.contains("ChildOf") && text.contains("intensity: 7.5"),
        "{text}"
    );
    assert_eq!(
        save_scene(world).unwrap(),
        text,
        "повторное сохранение — тот же текст"
    );

    despawn_all(world);
    let created = load_scene(world, &text).unwrap();
    assert_eq!(created.len(), 2);
    let parent = by_name(world, "Родитель");
    let child = by_name(world, "Ребёнок");
    assert_eq!(*world.get::<Transform>(parent).unwrap(), transform);
    assert!((world.get::<PointLight>(parent).unwrap().intensity - 7.5).abs() < 1e-6);
    assert_eq!(
        world.get::<ChildOf>(child).unwrap().parent(),
        parent,
        "ссылка на новую сущность"
    );
    assert!(world.get::<Children>(parent).unwrap().contains(&child));
    assert_eq!(
        save_scene(world).unwrap(),
        text,
        "после загрузки — тот же текст"
    );
}

#[test]
fn handles_are_saved_as_paths() {
    let mut app = app();
    let world = app.world_mut();
    let named = world
        .resource_mut::<Assets<Mesh>>()
        .add_named("test://cube", Mesh::cuboid(Vec3::ONE));
    let unnamed = world.resource_mut::<Assets<Mesh>>().add(Mesh::plane(1.0));
    world.spawn((
        Name::new("Именной"),
        Mesh3d {
            mesh: named.clone(),
            materials: vec![None],
        },
    ));
    world.spawn((
        Name::new("Безымянный"),
        Mesh3d {
            mesh: unnamed,
            materials: vec![],
        },
    ));

    let text = save_scene(world).unwrap();
    assert!(text.contains("Some(\"test://cube\")"), "{text}");
    assert!(text.contains("mesh: None"), "{text}");

    despawn_all(world);
    load_scene(world, &text).unwrap();
    let entity = by_name(world, "Именной");
    assert_eq!(
        world.get::<Mesh3d>(entity).unwrap().mesh.id(),
        named.id(),
        "путь привёл к тому же ресурсу"
    );
    let empty = by_name(world, "Безымянный");
    assert!(
        !world.get::<Mesh3d>(empty).unwrap().mesh.is_strong(),
        "пустой хэндл"
    );
}

#[test]
fn gltf_contents_are_not_saved() {
    let mut app = app();
    let world = app.world_mut();
    let gltf = world
        .resource::<AssetServer>()
        .load::<Gltf>("models/none.glb");
    let root = world.spawn((Name::new("Модель"), SceneRoot(gltf))).id();
    // Так glTF разворачивает узлы — потомками корня
    let node = world
        .spawn((Name::new("Узел"), Transform::default(), ChildOf(root)))
        .id();
    world.spawn((Name::new("Внук"), Transform::default(), ChildOf(node)));

    let text = save_scene(world).unwrap();
    assert!(
        text.contains("Модель") && text.contains("models/none.glb"),
        "{text}"
    );
    assert!(!text.contains("Узел") && !text.contains("Внук"), "{text}");
}

#[test]
fn unknown_component_is_skipped() {
    let mut app = app();
    let world = app.world_mut();
    let text = r#"(entities: [(id: 0, components: {
        "game::Health": (current: 5.0),
        "Name": "Выживший",
    })])"#;
    load_scene(world, text).unwrap();
    by_name(world, "Выживший"); // короткое имя типа тоже понимается
}

#[test]
fn broken_file_changes_nothing() {
    let mut app = app();
    let world = app.world_mut();
    let before = count(world);
    let text =
        r#"(entities: [(id: 0, components: {"Name": "А"}), (id: 1, components: {"Name": 5})])"#;
    assert!(load_scene(world, text).is_err());
    assert_eq!(count(world), before, "созданные заготовки удалены");
}

/// Сцена из репозитория (`assets/scenes/demo.scene`, сохранена sandbox) должна читаться и после
/// изменений формата. Компоненты незарегистрированных здесь плагинов пропускаются.
#[test]
fn committed_demo_scene_still_loads() {
    let mut app = app();
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../assets/scenes/demo.scene"
    );
    let text = std::fs::read_to_string(path).unwrap();
    let created = load_scene(app.world_mut(), &text).unwrap();
    assert!(created.len() > 40, "{}", created.len());
}
