//! Ресурсы (`docs/acrhitecture.md`, раздел 7).
//!
//! ```ignore
//! fn startup(server: Res<AssetServer>, mut commands: Commands) {
//!     let helmet: Handle<Gltf> = server.load("models/khronos/DamagedHelmet/DamagedHelmet.glb");
//!     commands.insert_resource(Scene { helmet });
//! }
//!
//! fn update(scene: Res<Scene>, gltfs: Res<Assets<Gltf>>) {
//!     if let Some(resolved) = gltfs.get(&scene.helmet) { /* resolved.asset, resolved.status */ }
//! }
//! ```
//!
//! - Хэндл выдаётся сразу, загрузка идёт в задаче, результат применяется в начале кадра.
//! - Пока ресурс грузится или если загрузка упала — подставляется заглушка этого типа
//!   (если она есть), а статус говорит, что произошло.
//! - Когда исчезает последний сильный хэндл, ресурс выгружается.

mod error;
mod handle;
mod loader;
mod path;
mod server;
mod source;
mod storage;
#[cfg(not(any(target_arch = "wasm32", target_os = "android", target_os = "ios")))]
mod watch;

pub mod formats;

use engine_core::{App, CVar, CVarFlags, CVarString, Plugin, Stage};

pub use error::AssetError;
pub use handle::{AssetId, Handle, ReflectHandle, UntypedAssetId};
pub use loader::{AssetLoader, LoadContext};
pub use path::AssetPath;
pub use server::AssetServer;
pub use source::Bytes;
pub use storage::{AssetEvent, AssetStatus, Assets, Resolved};

/// Тип ресурса. Заглушки — что показывать, пока ресурс грузится и если загрузка упала.
///
/// [Rust] Методы с телом — реализация по умолчанию: тип без заглушек просто пишет
/// `impl Asset for MyType {}`. `TypePath` (`#[derive(TypePath)]`) — имя типа для рефлексии:
/// хэндлы ресурсов попадают в сохранённые сцены.
pub trait Asset: engine_ecs::prelude::TypePath + Send + Sync + 'static {
    fn loading_fallback() -> Option<Self>
    where
        Self: Sized,
    {
        None
    }
    fn error_fallback() -> Option<Self>
    where
        Self: Sized,
    {
        None
    }

    /// В какой бюджет памяти входит ресурс. По умолчанию — «прочие».
    fn memory_category(&self) -> MemoryCategory {
        MemoryCategory::Other
    }

    /// Сколько байт занимает значение на CPU. 0 — в кеше не держать: при потере
    /// последнего хэндла ресурс выгружается сразу (так устроен glTF: тяжёлые данные
    /// лежат в отдельных мешах и текстурах).
    fn memory_size(&self) -> u64 {
        0
    }
}

/// Группа бюджета памяти (R2-12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryCategory {
    Image,
    Mesh,
    Other,
}

/// Сколько байт сейчас занимают загруженные ресурсы, по группам.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemoryUsage {
    pub image: u64,
    pub mesh: u64,
    pub other: u64,
}

impl MemoryUsage {
    pub fn total(self) -> u64 {
        self.image + self.mesh + self.other
    }

    fn add(&mut self, category: MemoryCategory, bytes: u64) {
        match category {
            MemoryCategory::Image => self.image += bytes,
            MemoryCategory::Mesh => self.mesh += bytes,
            MemoryCategory::Other => self.other += bytes,
        }
    }

    pub(crate) fn of(self, category: MemoryCategory) -> u64 {
        match category {
            MemoryCategory::Image => self.image,
            MemoryCategory::Mesh => self.mesh,
            MemoryCategory::Other => self.other,
        }
    }
}

pub static ASSETS_ROOT: CVar<CVarString> = CVar::new(
    "assets.root",
    std::borrow::Cow::Borrowed("assets"),
    "Корень ресурсов: папка на desktop, префикс URL на Web",
)
.with_flags(CVarFlags::STARTUP);

pub static ASSETS_ENGINE_DIR: CVar<CVarString> = CVar::new(
    "assets.engine_dir",
    std::borrow::Cow::Borrowed(""),
    "Папка вместо встроенных файлов `engine://` (разработка: правки шейдеров движка на лету)",
)
.with_flags(CVarFlags::STARTUP);

pub static ASSETS_WATCH: CVar<bool> = CVar::new(
    "assets.watch",
    true,
    "Перезагружать изменённые файлы ресурсов (только desktop)",
)
.with_flags(CVarFlags::STARTUP);

pub static ASSETS_STRICT: CVar<bool> = CVar::new(
    "assets.strict",
    false,
    "Ошибка загрузки ресурса — паника (для CI и тестов)",
);

/// 0 — неиспользуемый ресурс выгружается сразу. Иначе он остаётся, пока группа
/// не превысит бюджет, и тогда уходит самый давно не использованный.
pub static ASSETS_BUDGET_IMAGE_MB: CVar<u64> = CVar::new(
    "assets.budget_image_mb",
    0,
    "Бюджет текстур в памяти, МБ (0 — не держать неиспользуемые)",
);
pub static ASSETS_BUDGET_MESH_MB: CVar<u64> = CVar::new(
    "assets.budget_mesh_mb",
    0,
    "Бюджет мешей в памяти, МБ (0 — не держать неиспользуемые)",
);
pub static ASSETS_BUDGET_OTHER_MB: CVar<u64> = CVar::new(
    "assets.budget_other_mb",
    0,
    "Бюджет остальных ресурсов, МБ (0 — не держать неиспользуемые)",
);

/// Подключает систему ресурсов и встроенные форматы (изображения, glTF, WGSL).
/// Требует `NetPlugin` — для загрузки по HTTP и на Web.
#[derive(Debug, Default)]
pub struct AssetPlugin;

impl Plugin for AssetPlugin {
    fn build(&self, app: &mut App) {
        assert!(
            app.is_plugin_added::<engine_net::NetPlugin>(),
            "AssetPlugin требует NetPlugin: добавьте его раньше"
        );
        app.register_cvar(&ASSETS_ROOT)
            .register_cvar(&ASSETS_STRICT)
            .register_cvar(&ASSETS_WATCH)
            .register_cvar(&ASSETS_ENGINE_DIR)
            .register_cvar(&ASSETS_BUDGET_IMAGE_MB)
            .register_cvar(&ASSETS_BUDGET_MESH_MB)
            .register_cvar(&ASSETS_BUDGET_OTHER_MB)
            .insert_resource(AssetServer::new())
            .add_systems(Stage::First, server::apply_loads);
        formats::register(app);
    }

    fn finish(&self, app: &mut App) {
        // Здесь уже применены CVar (файл, аргументы) и создан клиент сети (NetPlugin::finish
        // вызывается раньше — плагины завершаются в порядке добавления)
        let net = app.world().get_resource::<engine_net::Net>().cloned();
        let server = app.world().resource::<AssetServer>();
        server.configure(net, ASSETS_ROOT.get().into_owned());
        let engine_dir = ASSETS_ENGINE_DIR.get();
        if !engine_dir.is_empty() {
            server.set_engine_dir(std::path::PathBuf::from(engine_dir.as_ref()));
        }
        #[cfg(not(any(target_arch = "wasm32", target_os = "android", target_os = "ios")))]
        if ASSETS_WATCH.get() {
            server.start_watching(std::path::Path::new(ASSETS_ROOT.get().as_ref()), "");
            if !engine_dir.is_empty() {
                server.start_watching(std::path::Path::new(engine_dir.as_ref()), "engine://");
            }
        }
    }
}

/// Методы регистрации типов ресурсов и загрузчиков в [`App`].
// [Rust] Трейт-расширение: добавляет методы чужому типу App (своего App мы тут не владеем).
// Аналог свободных функций, но вызываются через точку: `app.init_asset::<Image>()`.
pub trait AssetAppExt {
    fn init_asset<T: Asset>(&mut self) -> &mut Self;
    fn register_loader<L: AssetLoader>(&mut self, loader: L) -> &mut Self;
}

impl AssetAppExt for App {
    fn init_asset<T: Asset>(&mut self) -> &mut Self {
        if self.world().contains_resource::<Assets<T>>() {
            return self;
        }
        let server = self.world().resource::<AssetServer>().clone();
        server.register_type::<T>();
        let mut assets = Assets::<T>::new();
        assets.set_server(server);
        self.insert_resource(assets)
            .add_message::<AssetEvent<T>>()
            .add_systems(Stage::First, storage::flush_added::<T>);
        self
    }

    fn register_loader<L: AssetLoader>(&mut self, loader: L) -> &mut Self {
        self.init_asset::<L::Asset>();
        self.world()
            .resource::<AssetServer>()
            .register_loader(loader);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::{Gltf, Mesh};
    use engine_core::CVarSource;
    use std::sync::Mutex;

    /// Бюджеты — процессные CVar. Тесты, которым нужно разное значение, идут по очереди.
    static BUDGET_LOCK: Mutex<()> = Mutex::new(());

    fn lock_budgets() -> std::sync::MutexGuard<'static, ()> {
        BUDGET_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[test]
    fn load_dedup_and_unload_with_labeled() {
        let _budgets = lock_budgets();
        // Соседний тест мог оставить ненулевой бюджет — здесь нужна прежняя выгрузка сразу
        ASSETS_BUDGET_IMAGE_MB.reset();
        ASSETS_BUDGET_MESH_MB.reset();
        ASSETS_BUDGET_OTHER_MB.reset();
        let mut app = App::new();
        // Тест запускается из папки крейта — корень ресурсов на два уровня выше
        app.world_mut()
            .resource_mut::<engine_core::CVars>()
            .set("assets.root", "../../assets", CVarSource::LaunchArguments)
            .unwrap();
        app.add_plugin(engine_net::NetPlugin)
            .add_plugin(AssetPlugin);
        app.update(); // finish плагинов, Startup

        let server = app.world().resource::<AssetServer>().clone();
        let a: Handle<Gltf> = server.load("models/khronos/Box/Box.glb");
        let b: Handle<Gltf> = server.load("models/khronos/Box/Box.glb");
        assert_eq!(a, b, "один путь — один ресурс");

        // Ждём загрузку (в тесте можно покрутить кадры)
        for _ in 0..500 {
            app.update();
            if matches!(server.is_loaded_with_dependencies(a.untyped_id()), Ok(true)) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let mesh = {
            let gltfs = app.world().resource::<Assets<Gltf>>();
            let gltf = gltfs.get_loaded(&a).expect("Box.glb загружен");
            gltf.meshes[0].weak()
        };
        assert!(
            app.world()
                .resource::<Assets<Mesh>>()
                .get_loaded(&mesh)
                .is_some()
        );

        // Последние сильные хэндлы уничтожены → в следующем кадре выгрузка, каскадом и меш
        drop(a);
        drop(b);
        app.update();
        app.update();
        assert!(app.world().resource::<Assets<Gltf>>().is_empty());
        assert!(
            app.world()
                .resource::<Assets<Mesh>>()
                .get_loaded(&mesh)
                .is_none()
        );
    }

    /// Ресурс, добавленный кодом (`Assets::add`), тоже даёт событие `Loaded` — иначе рендер
    /// о нём не узнает (так пол в sandbox не рисовался).
    #[test]
    fn added_asset_sends_loaded_event() {
        use engine_ecs::prelude::*;

        #[derive(Resource, Default)]
        struct Seen(Vec<AssetId<Mesh>>);

        let mut app = App::new();
        app.add_plugin(engine_net::NetPlugin)
            .add_plugin(AssetPlugin)
            .init_resource::<Seen>()
            .add_systems(
                Stage::Update,
                |mut events: MessageReader<AssetEvent<Mesh>>, mut seen: ResMut<Seen>| {
                    for event in events.read() {
                        if let AssetEvent::Loaded(id) = event {
                            seen.0.push(*id);
                        }
                    }
                },
            );
        app.update();
        let handle = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(Mesh::plane(1.0));
        app.update();
        assert_eq!(app.world().resource::<Seen>().0, [handle.id()]);
    }

    /// Бюджет 1 МБ: маленький меш без хэндла остаётся, лишний толстый — выгружается,
    /// а тот, что ещё держат, остаётся даже сверх бюджета.
    #[test]
    fn lru_keeps_unused_until_budget() {
        use crate::formats::Primitive;
        use engine_math::Aabb;

        let _budgets = lock_budgets();
        let mut app = App::new();
        app.world_mut()
            .resource_mut::<engine_core::CVars>()
            .set(
                "assets.budget_mesh_mb",
                "1",
                engine_core::CVarSource::LaunchArguments,
            )
            .unwrap();
        app.add_plugin(engine_net::NetPlugin)
            .add_plugin(AssetPlugin);
        app.update();

        let small = app
            .world_mut()
            .resource_mut::<Assets<Mesh>>()
            .add(Mesh::plane(1.0));
        let small_id = small.weak();
        drop(small);
        app.update();
        assert!(
            app.world()
                .resource::<Assets<Mesh>>()
                .get_loaded(&small_id)
                .is_some(),
            "ниже бюджета неиспользуемый меш остаётся"
        );

        let fat = || Mesh {
            name: String::new(),
            primitives: vec![Primitive {
                positions: vec![[0.0; 3]; 100_000],
                normals: Vec::new(),
                tangents: Vec::new(),
                uvs: Vec::new(),
                joints: Vec::new(),
                weights: Vec::new(),
                indices: Vec::new(),
                material: None,
                bounds: Aabb::EMPTY,
            }],
        };
        let older = app.world_mut().resource_mut::<Assets<Mesh>>().add(fat());
        let newer = app.world_mut().resource_mut::<Assets<Mesh>>().add(fat());
        let older_id = older.weak();
        app.update();
        assert!(
            app.world()
                .resource::<Assets<Mesh>>()
                .get_loaded(&small_id)
                .is_none(),
            "самый старый неиспользуемый уходит первым"
        );
        drop(older);
        app.update();
        let meshes = app.world().resource::<Assets<Mesh>>();
        assert!(meshes.get_loaded(&older_id).is_none());
        assert!(meshes.get_loaded(&newer).is_some());
        ASSETS_BUDGET_MESH_MB.reset();
    }
}
