use std::any::TypeId;
use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::sync::Arc;

use crossbeam_channel::Sender;
// [Rust] `bevy_reflect` — имя, на которое ссылается код, сгенерированный `#[derive(Reflect)]`
use engine_ecs::prelude::bevy_reflect;

/// Идентификатор ресурса без типа: индекс + поколение + тип.
/// Поколение растёт при повторном использовании индекса — старый id не укажет на чужой ресурс.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UntypedAssetId {
    pub(crate) type_id: TypeId,
    pub(crate) index: u32,
    pub(crate) generation: u32,
}

/// Типизированный идентификатор. Копируется, не держит ресурс в памяти.
// [Rust] `PhantomData<fn() -> T>` — «использование» T без хранения T: тип помнит,
// на что указывает, но остаётся Copy/Send/Sync независимо от T.
pub struct AssetId<T> {
    pub(crate) index: u32,
    pub(crate) generation: u32,
    marker: PhantomData<fn() -> T>,
}

impl<T: 'static> AssetId<T> {
    pub(crate) fn new(index: u32, generation: u32) -> Self {
        Self {
            index,
            generation,
            marker: PhantomData,
        }
    }

    pub fn untyped(self) -> UntypedAssetId {
        UntypedAssetId {
            type_id: TypeId::of::<T>(),
            index: self.index,
            generation: self.generation,
        }
    }
}

// [Rust] derive(Clone, Copy, ...) потребовал бы `T: Clone`, хотя T не хранится. Поэтому вручную.
impl<T> Clone for AssetId<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for AssetId<T> {}
impl<T> PartialEq for AssetId<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}
impl<T> Eq for AssetId<T> {}
impl<T> Ord for AssetId<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.index
            .cmp(&other.index)
            .then(self.generation.cmp(&other.generation))
    }
}
impl<T> PartialOrd for AssetId<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T> Hash for AssetId<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}
impl<T> fmt::Debug for AssetId<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AssetId({}v{})", self.index, self.generation)
    }
}

/// Владение ресурсом. Когда уничтожается последняя копия — сервер узнаёт об этом.
pub(crate) struct StrongRef {
    pub id: UntypedAssetId,
    pub drop_sender: Sender<UntypedAssetId>,
}

impl Drop for StrongRef {
    // [Rust] Drop — аналог деструктора. Вызывается автоматически, когда значение
    // выходит из области видимости; здесь — когда исчез последний Arc.
    fn drop(&mut self) {
        let _ = self.drop_sender.send(self.id);
    }
}

/// Хэндл ресурса.
///
/// - **Сильный** (из `AssetServer::load`) держит ресурс загруженным. Копии через `clone()`
///   разделяют владение, как `shared_ptr`.
/// - **Слабый** (`weak()`) только указывает на ресурс, как `weak_ptr`.
///
/// Для рефлексии хэндл «непрозрачный» (`opaque`): внутренние поля не видны, в сохранённой
/// сцене он записывается путём к ресурсу ([`ReflectHandle`]).
#[derive(engine_ecs::prelude::Reflect)]
#[reflect(opaque, Clone, Handle)]
#[reflect(where T: crate::Asset)]
pub struct Handle<T> {
    id: AssetId<T>,
    strong: Option<Arc<StrongRef>>,
}

impl<T: 'static> Handle<T> {
    pub(crate) fn strong(id: AssetId<T>, strong: Arc<StrongRef>) -> Self {
        Self {
            id,
            strong: Some(strong),
        }
    }

    pub fn id(&self) -> AssetId<T> {
        self.id
    }

    pub fn untyped_id(&self) -> UntypedAssetId {
        self.id.untyped()
    }

    pub fn is_strong(&self) -> bool {
        self.strong.is_some()
    }

    /// Слабая копия: не держит ресурс в памяти.
    #[must_use]
    pub fn weak(&self) -> Self {
        Self {
            id: self.id,
            strong: None,
        }
    }
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            strong: self.strong.clone(),
        }
    }
}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl<T> Eq for Handle<T> {}

impl<T> fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = if self.strong.is_some() {
            "strong"
        } else {
            "weak"
        };
        write!(
            f,
            "Handle<{}>({:?}, {kind})",
            short_type_name::<T>(),
            self.id
        )
    }
}

pub(crate) fn short_type_name<T: ?Sized>() -> &'static str {
    let name = std::any::type_name::<T>();
    name.rsplit("::").next().unwrap_or(name)
}

/// Рефлексия хэндла (данные типа `Handle<T>` в реестре): путь ресурса для сохранения
/// и загрузка по пути при чтении сцены. Сериализатор сцен находит хэндлы по этим данным —
/// не зная конкретного типа ресурса.
#[derive(Clone, Debug)]
pub struct ReflectHandle {
    /// Путь ресурса, на который указывает хэндл (`None` — ресурс создан кодом без имени).
    pub path: fn(&dyn engine_ecs::reflect::PartialReflect, &crate::AssetServer) -> Option<String>,
    /// Хэндл нужного типа по пути (`server.load`), упакованный для рефлексии.
    pub load: fn(&crate::AssetServer, &str) -> Box<dyn engine_ecs::reflect::PartialReflect>,
    /// Пустой хэндл (никуда не указывает) — для ресурса, сохранённого без пути.
    pub missing: fn() -> Box<dyn engine_ecs::reflect::PartialReflect>,
}

impl<T: crate::Asset> engine_ecs::reflect::CreateTypeData<Handle<T>> for ReflectHandle {
    fn create_type_data(_input: ()) -> Self {
        Self {
            path: |value, server| {
                let handle = value.try_downcast_ref::<Handle<T>>()?;
                server.path(handle.untyped_id())
            },
            load: |server, path| Box::new(server.load::<T>(path)),
            missing: || {
                Box::new(Handle::<T> {
                    id: AssetId::new(u32::MAX, u32::MAX),
                    strong: None,
                })
            },
        }
    }
}
