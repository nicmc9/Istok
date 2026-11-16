//! Команды консоли `save` и `load`: сцена в файл `assets/scenes/<имя>.scene` и обратно.
//!
//! Только desktop: на мобильных и Web папки ресурсов не записать. Загрузка сцены из файла на
//! любой платформе — через `--cvar sandbox.scene=scenes/<имя>.scene` (сервер ресурсов).

use engine::prelude::*;

// [Rust] Модуль целиком под условием компиляции: на других платформах его нет вовсе —
// ни функций, ни импортов (иначе компилятор предупредит о неиспользуемом)
#[cfg(not(any(target_arch = "wasm32", target_os = "android", target_os = "ios")))]
mod files {
    use engine::prelude::*;
    use engine::serialize::{load_scene, save_scene, scene_entities};

    /// Файл сцены по имени из консоли.
    fn path(args: &[String]) -> Result<String, String> {
        let name = match args {
            [] => "quick",
            [name] => name.as_str(),
            _ => return Err("использование: save|load [имя]".to_owned()),
        };
        if name.contains(['/', '\\', '.']) {
            return Err("имя без пути и расширения: save demo".to_owned());
        }
        Ok(format!(
            "{}/scenes/{name}.scene",
            engine::assets::ASSETS_ROOT.get()
        ))
    }

    pub fn save(world: &mut World, args: &[String]) -> CommandResult {
        let path = path(args)?;
        let count = scene_entities(world).len();
        let text = save_scene(world).map_err(|e| e.to_string())?;
        if let Some(dir) = std::path::Path::new(&path).parent() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{path}: {e}"))?;
        }
        std::fs::write(&path, &text).map_err(|e| format!("{path}: {e}"))?;
        Ok(format!("сохранено {count} сущностей → {path}"))
    }

    pub fn load(world: &mut World, args: &[String]) -> CommandResult {
        let path = path(args)?;
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
        // Сначала проверяем, что файл читается, — и только потом заменяем текущую сцену
        let old = scene_entities(world);
        let created = load_scene(world, &text).map_err(|e| e.to_string())?;
        for entity in old {
            // Потомки уже удалены вместе с родителем
            if world.get_entity(entity).is_ok() {
                world.despawn(entity);
            }
        }
        Ok(format!("загружено {} сущностей ← {path}", created.len()))
    }
}

#[cfg(any(target_arch = "wasm32", target_os = "android", target_os = "ios"))]
mod files {
    use engine::prelude::*;

    pub fn save(_: &mut World, _: &[String]) -> CommandResult {
        Err("сохранение в файл — только на desktop".to_owned())
    }

    pub fn load(_: &mut World, _: &[String]) -> CommandResult {
        Err("на этой платформе: --cvar sandbox.scene=scenes/<имя>.scene".to_owned())
    }
}

pub fn console_commands() -> [ConsoleCommand; 2] {
    [
        ConsoleCommand::new(
            "save",
            "Сохранить сцену в assets/scenes/<имя>.scene",
            files::save,
        )
        .with_usage("save [имя]"),
        ConsoleCommand::new("load", "Заменить сцену на сохранённую", files::load)
            .with_usage("load [имя]"),
    ]
}
