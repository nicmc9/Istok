//! Сериализация сцен (R2-11): сущности с компонентами ↔ текст RON через рефлексию (решение D7).
//!
//! - [`save_scene`] — все сущности, у которых есть хотя бы один зарегистрированный компонент
//!   (`app.register_type::<T>()`). Потомки `SceneRoot` не сохраняются: их снова развернёт glTF.
//! - [`load_scene`] — создаёт сущности из текста; ссылки (`ChildOf`) переводятся на новые.
//! - Файл сцены — ресурс [`SceneFile`] (`.scene`, внутри RON); компонент [`LoadScene`]
//!   загружает его в мир через сервер ресурсов — на любой платформе.
//! - Хэндлы ресурсов пишутся путями. Ресурс, созданный кодом, сохраняется, только если он
//!   добавлен с именем (`Assets::add_named`) — иначе в файле `None` и предупреждение в логе.
//!
//! Ресурсы ECS (`AmbientLight`, `ClearColor`...) в сцену не входят — только сущности.

mod format;

use std::collections::{HashMap, HashSet};

use engine_assets::{
    Asset, AssetAppExt, AssetError, AssetServer, AssetStatus, Assets, Bytes, Handle, LoadContext,
};
use engine_core::tasks::BoxedFuture;
use engine_core::{App, Plugin, Stage};
use engine_ecs::prelude::*;
use engine_ecs::reflect::{AppTypeRegistry, ReflectComponent};
use engine_scene::SceneRoot;
use serde::de::DeserializeSeed;

use format::{LoadProcessor, SaveProcessor, SavedEntity, SceneSeed, SceneSer};

#[derive(Debug, thiserror::Error)]
pub enum SceneError {
    #[error("сцена не записана: {0}")]
    Save(String),
    #[error("сцена не прочитана: {0}")]
    Load(String),
}

/// Rust-тип компонента по его id в мире (`None` — компонент без Rust-типа).
fn component_type(world: &World, id: engine_ecs::reflect::ComponentId) -> Option<std::any::TypeId> {
    world.components().get_info(id)?.type_id()
}

/// Сущности сцены — то, что сохранит [`save_scene`]. Например, чтобы очистить мир перед
/// загрузкой другой сцены (потомки glTF удалятся вместе с корнями).
pub fn scene_entities(world: &mut World) -> Vec<Entity> {
    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();
    saved_entities(world, &registry)
}

/// Сущности, которые попадут в файл, — по порядку создания.
fn saved_entities(world: &mut World, registry: &engine_ecs::reflect::TypeRegistry) -> Vec<Entity> {
    // Содержимое glTF-сцен пересоздаётся при загрузке — исключаем потомков SceneRoot
    let mut excluded = HashSet::new();
    let mut roots = world.query_filtered::<Entity, With<SceneRoot>>();
    let mut stack: Vec<Entity> = Vec::new();
    for root in roots.iter(world) {
        if let Some(children) = world.get::<Children>(root) {
            stack.extend(children.iter());
        }
    }
    while let Some(entity) = stack.pop() {
        if excluded.insert(entity)
            && let Some(children) = world.get::<Children>(entity)
        {
            stack.extend(children.iter());
        }
    }
    let mut all = world.query::<EntityRef>();
    let mut entities: Vec<Entity> = all
        .iter(world)
        .filter(|entity| !excluded.contains(&entity.id()))
        .filter(|entity| {
            entity.archetype().components().iter().any(|id| {
                component_type(world, *id)
                    .and_then(|type_id| registry.get_type_data::<ReflectComponent>(type_id))
                    .is_some()
            })
        })
        .map(|entity| entity.id())
        .collect();
    entities.sort_by_key(|entity| entity.index());
    entities
}

/// Сохраняет сущности мира в текст RON.
pub fn save_scene(world: &mut World) -> Result<String, SceneError> {
    let _zone = tracing::trace_span!("save_scene").entered();
    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();
    let entities = saved_entities(world, &registry);
    let ids: HashMap<Entity, u32> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| (*e, u32::try_from(i).unwrap_or(u32::MAX)))
        .collect();

    let mut saved = Vec::with_capacity(entities.len());
    for &entity in &entities {
        let id = ids[&entity];
        let entity_ref = world.entity(entity);
        let mut components = Vec::new();
        for component_id in entity_ref.archetype().components() {
            let Some(type_id) = component_type(world, *component_id) else {
                continue;
            };
            let Some(reflect) = registry.get_type_data::<ReflectComponent>(type_id) else {
                continue;
            };
            // Родитель не попал в файл (например, сущность — потомок glTF) — связь не пишем
            if let Some(parent) = entity_ref.get::<ChildOf>()
                && type_id == std::any::TypeId::of::<ChildOf>()
                && !ids.contains_key(&parent.parent())
            {
                continue;
            }
            if let Some(value) = reflect.reflect(entity_ref) {
                components.push(value.as_partial_reflect());
            }
        }
        // Порядок компонентов в архетипе не определён — сортируем для стабильных диффов
        components.sort_by_key(|c| c.reflect_type_path());
        saved.push(SavedEntity { id, components });
    }

    let server = world.get_resource::<AssetServer>();
    let processor = SaveProcessor { ids: &ids, server };
    let scene = SceneSer {
        entities: &saved,
        registry: &registry,
        processor: &processor,
    };
    let config = ron::ser::PrettyConfig::new()
        .struct_names(false)
        .indentor("    ".to_owned());
    ron::ser::to_string_pretty(&scene, config).map_err(|e| SceneError::Save(e.to_string()))
}

/// Создаёт сущности из текста сцены. Возвращает созданные сущности.
/// При ошибке разбора мир не меняется (созданные заготовки удаляются).
pub fn load_scene(world: &mut World, text: &str) -> Result<Vec<Entity>, SceneError> {
    let _zone = tracing::trace_span!("load_scene").entered();
    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();
    let server = world.get_resource::<AssetServer>().cloned();
    let mut processor = LoadProcessor {
        world,
        server,
        ids: HashMap::new(),
    };
    let parsed = ron::Deserializer::from_str(text)
        .map_err(|e| e.to_string())
        .and_then(|mut de| {
            SceneSeed {
                registry: &registry,
                processor: &mut processor,
            }
            .deserialize(&mut de)
            .map_err(|e| e.to_string())
        });
    let ids = std::mem::take(&mut processor.ids);
    let entities = match parsed {
        Ok(entities) => entities,
        Err(error) => {
            for entity in ids.into_values() {
                world.despawn(entity);
            }
            return Err(SceneError::Load(error));
        }
    };
    let mut created = Vec::with_capacity(entities.len());
    for (entity, components) in entities {
        let mut entity_mut = world.entity_mut(entity);
        for component in &components {
            let reflect = component
                .get_represented_type_info()
                .and_then(|info| registry.get_type_data::<ReflectComponent>(info.type_id()));
            if let Some(reflect) = reflect {
                reflect.insert(&mut entity_mut, component.as_ref(), &registry);
            } else {
                tracing::warn!(
                    "Сцена: «{}» — не компонент, пропущен",
                    component.reflect_type_path()
                );
            }
        }
        created.push(entity);
    }
    Ok(created)
}

/// Файл сцены (`.scene`, RON) как ресурс. Разбирается при создании сущностей — для этого
/// нужен мир (реестр типов, сервер ресурсов), поэтому загрузчик только читает текст.
#[derive(Debug, Clone, TypePath)]
pub struct SceneFile {
    pub text: String,
}

impl Asset for SceneFile {}

#[derive(Debug)]
pub struct SceneFileLoader;

impl engine_assets::AssetLoader for SceneFileLoader {
    type Asset = SceneFile;

    fn extensions(&self) -> &[&'static str] {
        &["scene"]
    }

    fn load<'a>(
        &'a self,
        bytes: Bytes,
        ctx: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<SceneFile, AssetError>> {
        Box::pin(async move {
            let text = String::from_utf8(bytes.to_vec())
                .map_err(|_| AssetError::format(ctx.path(), "не UTF-8"))?;
            Ok(SceneFile { text })
        })
    }
}

/// Загрузить сцену из файла: `commands.spawn(LoadScene(server.load("scenes/demo.scene")))`.
/// Когда файл загрузится, сущности сцены создаются, а эта сущность удаляется.
#[derive(Component, Debug, Clone)]
pub struct LoadScene(pub Handle<SceneFile>);

fn spawn_loaded_scenes(world: &mut World) {
    let mut pending = world.query::<(Entity, &LoadScene)>();
    // `None` — файл не загрузился (ошибка уже в логе сервера ресурсов)
    let ready: Vec<(Entity, Option<String>)> = {
        let files = world.resource::<Assets<SceneFile>>();
        let server = world.resource::<AssetServer>();
        pending
            .iter(world)
            .filter_map(|(entity, load)| match server.status(load.0.untyped_id()) {
                AssetStatus::Loading => None,
                AssetStatus::Failed(_) => Some((entity, None)),
                AssetStatus::Ready => {
                    Some((entity, files.get_loaded(&load.0).map(|f| f.text.clone())))
                }
            })
            .collect()
    };
    for (entity, text) in ready {
        world.despawn(entity);
        let Some(text) = text else { continue };
        match load_scene(world, &text) {
            Ok(created) => tracing::info!("Сцена загружена: {} сущностей", created.len()),
            Err(error) => tracing::error!("{error}"),
        }
    }
}

/// Файлы сцен и [`LoadScene`]. Требует `AssetPlugin`.
#[derive(Debug, Default)]
pub struct SerializePlugin;

impl Plugin for SerializePlugin {
    fn build(&self, app: &mut App) {
        app.register_loader(SceneFileLoader)
            .add_systems(Stage::PreUpdate, spawn_loaded_scenes);
    }
}
