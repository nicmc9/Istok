//! Формат файла сцены (RON) поверх рефлексии.
//!
//! ```ron
//! (entities: [
//!     (id: 0, components: {
//!         "engine_scene::components::Transform": (translation: (x: 0.0, y: 2.0, z: 0.0), ...),
//!         "engine_physics::components::RigidBody": Dynamic,
//!     }),
//!     (id: 1, components: { "bevy_ecs::hierarchy::ChildOf": (0) }),
//! ])
//! ```
//!
//! - Ключ компонента — полный путь типа (`TypePath`), значение — данные через рефлексию.
//! - Ссылка на сущность — номер `id` в этом же файле (а не id в мире: при загрузке сущности
//!   получат новые).
//! - Хэндл ресурса — путь: `Some("models/helmet.glb")`; `None` — ресурс был без пути.
//!
//! [Rust] serde разделяет «что сериализовать» (трейт `Serialize`) и «во что» (формат: RON,
//! JSON...). Формат здесь подставляется один — RON, но код ниже от него не зависит.

use std::collections::HashMap;
use std::fmt;

use engine_assets::{AssetServer, ReflectHandle};
use engine_ecs::prelude::*;
use engine_ecs::reflect::{
    PartialReflect, ReflectDeserializerProcessor, ReflectSerializerProcessor, TypeRegistration,
    TypeRegistry, TypedReflectDeserializer, TypedReflectSerializer,
};
use serde::de::{
    DeserializeSeed, Deserializer, Error as _, IgnoredAny, MapAccess, SeqAccess, Visitor,
};
use serde::ser::{SerializeMap, SerializeSeq, SerializeStruct, Serializer};
use serde::{Deserialize, Serialize};

// ---------- сохранение ----------

/// Сущность для записи: номер в файле и её компоненты.
pub(crate) struct SavedEntity<'w> {
    pub id: u32,
    pub components: Vec<&'w dyn PartialReflect>,
}

/// Подмена при записи: хэндлы → пути, сущности → номера в файле.
pub(crate) struct SaveProcessor<'a> {
    pub ids: &'a HashMap<Entity, u32>,
    pub server: Option<&'a AssetServer>,
}

/// Номер для сущности, не попавшей в файл.
const NO_ENTITY: u32 = u32::MAX;

impl ReflectSerializerProcessor for SaveProcessor<'_> {
    fn try_serialize<S: Serializer>(
        &self,
        value: &dyn PartialReflect,
        registry: &TypeRegistry,
        serializer: S,
    ) -> Result<Result<S::Ok, S>, S::Error> {
        if let Some(entity) = value.try_downcast_ref::<Entity>() {
            let id = self.ids.get(entity).copied().unwrap_or(NO_ENTITY);
            return serializer.serialize_u32(id).map(Ok);
        }
        // У `ChildOf` своя serde-сериализация (фича bevy_ecs `serialize`): рефлексия записала бы
        // его целиком, с id сущности в мире, а не номером в файле. Поэтому — явно
        if let Some(child_of) = value.try_downcast_ref::<ChildOf>() {
            let id = self
                .ids
                .get(&child_of.parent())
                .copied()
                .unwrap_or(NO_ENTITY);
            return serializer.serialize_newtype_struct("ChildOf", &id).map(Ok);
        }
        let handle = value
            .get_represented_type_info()
            .and_then(|info| registry.get_type_data::<ReflectHandle>(info.type_id()));
        let Some(handle) = handle else {
            return Ok(Err(serializer)); // обычное значение — как есть
        };
        let path = self.server.and_then(|server| (handle.path)(value, server));
        if path.is_none() {
            tracing::warn!(
                "Сцена: ресурс {} без пути — сохранён пустым (используйте Assets::add_named)",
                value.reflect_type_path()
            );
        }
        path.serialize(serializer).map(Ok)
    }
}

/// Сцена для записи.
pub(crate) struct SceneSer<'a> {
    pub entities: &'a [SavedEntity<'a>],
    pub registry: &'a TypeRegistry,
    pub processor: &'a SaveProcessor<'a>,
}

impl Serialize for SceneSer<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut scene = serializer.serialize_struct("Scene", 1)?;
        scene.serialize_field("entities", &EntitiesSer(self))?;
        scene.end()
    }
}

struct EntitiesSer<'a>(&'a SceneSer<'a>);

impl Serialize for EntitiesSer<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let scene = self.0;
        let mut seq = serializer.serialize_seq(Some(scene.entities.len()))?;
        for entity in scene.entities {
            seq.serialize_element(&EntitySer { scene, entity })?;
        }
        seq.end()
    }
}

struct EntitySer<'a> {
    scene: &'a SceneSer<'a>,
    entity: &'a SavedEntity<'a>,
}

impl Serialize for EntitySer<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut entity = serializer.serialize_struct("Entity", 2)?;
        entity.serialize_field("id", &self.entity.id)?;
        entity.serialize_field("components", &ComponentsSer(self))?;
        entity.end()
    }
}

struct ComponentsSer<'a>(&'a EntitySer<'a>);

impl Serialize for ComponentsSer<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let scene = self.0.scene;
        let components = &self.0.entity.components;
        let mut map = serializer.serialize_map(Some(components.len()))?;
        for component in components {
            map.serialize_entry(
                component.reflect_type_path(),
                &TypedReflectSerializer::with_processor(
                    *component,
                    scene.registry,
                    scene.processor,
                ),
            )?;
        }
        map.end()
    }
}

// ---------- загрузка ----------

/// Подмена при чтении: пути → хэндлы (`server.load`), номера → новые сущности.
pub(crate) struct LoadProcessor<'a> {
    pub world: &'a mut World,
    pub server: Option<AssetServer>,
    /// Номер в файле → созданная сущность. Сущность создаётся при первом упоминании:
    /// ссылка может встретиться раньше, чем сама сущность (`ChildOf` на следующую).
    pub ids: HashMap<u32, Entity>,
}

impl LoadProcessor<'_> {
    pub fn entity(&mut self, id: u32) -> Entity {
        if id == NO_ENTITY {
            return Entity::PLACEHOLDER;
        }
        *self
            .ids
            .entry(id)
            .or_insert_with(|| self.world.spawn_empty().id())
    }
}

impl ReflectDeserializerProcessor for LoadProcessor<'_> {
    fn try_deserialize<'de, D: Deserializer<'de>>(
        &mut self,
        registration: &TypeRegistration,
        _registry: &TypeRegistry,
        deserializer: D,
    ) -> Result<Result<Box<dyn PartialReflect>, D>, D::Error> {
        if registration.type_id() == std::any::TypeId::of::<Entity>() {
            let id = u32::deserialize(deserializer)?;
            return Ok(Ok(Box::new(self.entity(id))));
        }
        if registration.type_id() == std::any::TypeId::of::<ChildOf>() {
            let id = deserializer.deserialize_newtype_struct("ChildOf", ParentId)?;
            return Ok(Ok(Box::new(ChildOf(self.entity(id)))));
        }
        let Some(handle) = registration.data::<ReflectHandle>() else {
            return Ok(Err(deserializer));
        };
        let path = Option::<String>::deserialize(deserializer)?;
        let value = match (path, &self.server) {
            (Some(path), Some(server)) => (handle.load)(server, &path),
            _ => (handle.missing)(),
        };
        Ok(Ok(value))
    }
}

/// `ChildOf(номер)` → номер родителя в файле.
struct ParentId;

impl<'de> Visitor<'de> for ParentId {
    type Value = u32;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("номер родителя: ChildOf(N)")
    }

    fn visit_newtype_struct<D: Deserializer<'de>>(self, deserializer: D) -> Result<u32, D::Error> {
        u32::deserialize(deserializer)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<u32, A::Error> {
        seq.next_element()?
            .ok_or_else(|| A::Error::invalid_length(0, &self))
    }
}

/// Прочитанная сущность: созданная сущность мира и значения компонентов.
pub(crate) type LoadedEntity = (Entity, Vec<Box<dyn PartialReflect>>);

/// Чтение сцены: `(entities: [...])`.
pub(crate) struct SceneSeed<'a, 'w> {
    pub registry: &'a TypeRegistry,
    pub processor: &'a mut LoadProcessor<'w>,
}

impl<'de> DeserializeSeed<'de> for SceneSeed<'_, '_> {
    type Value = Vec<LoadedEntity>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_struct("Scene", &["entities"], self)
    }
}

impl<'de> Visitor<'de> for SceneSeed<'_, '_> {
    type Value = Vec<LoadedEntity>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("сцена (entities: [...])")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut entities = None;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "entities" => {
                    entities = Some(map.next_value_seed(EntitiesSeed {
                        registry: self.registry,
                        processor: &mut *self.processor,
                    })?);
                }
                other => return Err(A::Error::unknown_field(other, &["entities"])),
            }
        }
        entities.ok_or_else(|| A::Error::missing_field("entities"))
    }
}

struct EntitiesSeed<'a, 'w> {
    registry: &'a TypeRegistry,
    processor: &'a mut LoadProcessor<'w>,
}

impl<'de> DeserializeSeed<'de> for EntitiesSeed<'_, '_> {
    type Value = Vec<LoadedEntity>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_seq(self)
    }
}

impl<'de> Visitor<'de> for EntitiesSeed<'_, '_> {
    type Value = Vec<LoadedEntity>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("список сущностей")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
        let mut entities = Vec::new();
        while let Some(entity) = seq.next_element_seed(EntitySeed {
            registry: self.registry,
            processor: &mut *self.processor,
        })? {
            entities.push(entity);
        }
        Ok(entities)
    }
}

struct EntitySeed<'a, 'w> {
    registry: &'a TypeRegistry,
    processor: &'a mut LoadProcessor<'w>,
}

impl<'de> DeserializeSeed<'de> for EntitySeed<'_, '_> {
    type Value = LoadedEntity;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_struct("Entity", &["id", "components"], self)
    }
}

impl<'de> Visitor<'de> for EntitySeed<'_, '_> {
    type Value = LoadedEntity;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("сущность (id: N, components: {...})")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut id = None;
        let mut components = None;
        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "id" => id = Some(map.next_value::<u32>()?),
                "components" => {
                    components = Some(map.next_value_seed(ComponentsSeed {
                        registry: self.registry,
                        processor: &mut *self.processor,
                    })?);
                }
                other => return Err(A::Error::unknown_field(other, &["id", "components"])),
            }
        }
        let id = id.ok_or_else(|| A::Error::missing_field("id"))?;
        Ok((self.processor.entity(id), components.unwrap_or_default()))
    }
}

struct ComponentsSeed<'a, 'w> {
    registry: &'a TypeRegistry,
    processor: &'a mut LoadProcessor<'w>,
}

impl<'de> DeserializeSeed<'de> for ComponentsSeed<'_, '_> {
    type Value = Vec<Box<dyn PartialReflect>>;

    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_map(self)
    }
}

impl<'de> Visitor<'de> for ComponentsSeed<'_, '_> {
    type Value = Vec<Box<dyn PartialReflect>>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("компоненты { \"путь::Типа\": значение }")
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
        let mut components = Vec::new();
        while let Some(type_path) = map.next_key::<String>()? {
            // Полный путь; короткое имя — на случай ручной правки файла
            let registration = self
                .registry
                .get_with_type_path(&type_path)
                .or_else(|| self.registry.get_with_short_type_path(&type_path));
            let Some(registration) = registration else {
                // Плагин компонента не подключён — остальное всё равно загрузим
                tracing::warn!("Сцена: неизвестный компонент «{type_path}» пропущен");
                map.next_value::<IgnoredAny>()?;
                continue;
            };
            let value = map.next_value_seed(TypedReflectDeserializer::with_processor(
                registration,
                self.registry,
                &mut *self.processor,
            ))?;
            components.push(value);
        }
        Ok(components)
    }
}
