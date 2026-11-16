//! Встроенные команды: help, find, reset, toggle, clear, history.

use std::fmt::Write as _;

use engine_core::{CVarSource, CVars};
use engine_ecs::prelude::*;

use crate::console::{CommandResult, Console, ConsoleCommand, describe, unknown_name};

pub(crate) fn commands() -> [ConsoleCommand; 6] {
    [
        ConsoleCommand::new("help", "Справка: список команд или описание одной", help)
            .with_usage("help [имя]"),
        ConsoleCommand::new(
            "find",
            "Поиск команд и переменных по имени и описанию",
            find,
        )
        .with_usage("find <текст>"),
        ConsoleCommand::new("reset", "Вернуть переменной значение по умолчанию", reset)
            .with_usage("reset <переменная>"),
        ConsoleCommand::new("toggle", "Переключить логическую переменную", toggle)
            .with_usage("toggle <переменная>"),
        ConsoleCommand::new("clear", "Очистить журнал", |world, _| {
            world.resource_mut::<Console>().clear();
            Ok(String::new())
        }),
        ConsoleCommand::new(
            "history",
            "Введённые команды",
            |world, _| {
                let console = world.resource::<Console>();
                let mut out = String::new();
                for (i, line) in console.history().iter().enumerate() {
                    let _ = writeln!(out, "{:>3}  {line}", i + 1); // запись в String не падает
                }
                Ok(out)
            },
        ),
    ]
}

/// Ровно один аргумент — имя.
fn single_name<'a>(args: &'a [String], usage: &str) -> Result<&'a str, String> {
    match args {
        [name] => Ok(name),
        _ => Err(format!("использование: {usage}")),
    }
}

fn help(world: &mut World, args: &[String]) -> CommandResult {
    let console = world.resource::<Console>();
    if let [name] = args {
        if let Some(command) = console.command(name) {
            let cheat = if command.is_cheat() { " (чит)" } else { "" };
            return Ok(format!(
                "{}{cheat}\n  {}",
                command.usage(),
                command.description()
            ));
        }
        return match world.resource::<CVars>().get(name) {
            Some(cvar) => Ok(describe(cvar)),
            None => Err(unknown_name(world, name)),
        };
    }
    let mut out = String::from("Команды:\n");
    for command in console.commands() {
        let cheat = if command.is_cheat() { " (чит)" } else { "" };
        let _ = writeln!(
            out,
            "  {:<24} {}{cheat}",
            command.usage(),
            command.description()
        );
    }
    out.push_str(
        "Переменные: «имя» — показать, «имя значение» — изменить, find — поиск.\n\
         Tab — дополнение, стрелки вверх/вниз — история.",
    );
    Ok(out)
}

fn find(world: &mut World, args: &[String]) -> CommandResult {
    let query = single_name(args, "find <текст>")?.to_lowercase();
    let matches = |text: &str| text.to_lowercase().contains(&query);
    let mut out = String::new();
    for command in world.resource::<Console>().commands() {
        if matches(command.name()) || matches(command.description()) {
            let _ = writeln!(out, "  {:<28} {}", command.usage(), command.description());
        }
    }
    for cvar in world.resource::<CVars>().iter() {
        if matches(cvar.name()) || matches(cvar.description()) {
            let _ = writeln!(
                out,
                "  {:<28} = {:<10} {}",
                cvar.name(),
                cvar.value_text(),
                cvar.description()
            );
        }
    }
    if out.is_empty() {
        return Err(format!("ничего не найдено по «{query}»"));
    }
    Ok(out)
}

fn reset(world: &mut World, args: &[String]) -> CommandResult {
    let name = single_name(args, "reset <переменная>")?;
    let mut cvars = world.resource_mut::<CVars>();
    let Some(cvar) = cvars.get(name) else {
        return Err(unknown_name(world, name));
    };
    // Через `set`, а не `cvar.reset()`: так проверяются флаги STARTUP и CHEAT
    cvars
        .set(name, &cvar.default_text(), CVarSource::Console)
        .map_err(|error| error.to_string())?;
    Ok(format!("{name} = {}", cvar.value_text()))
}

fn toggle(world: &mut World, args: &[String]) -> CommandResult {
    let name = single_name(args, "toggle <переменная>")?;
    let mut cvars = world.resource_mut::<CVars>();
    let Some(cvar) = cvars.get(name) else {
        return Err(unknown_name(world, name));
    };
    if cvar.type_name() != "bool" {
        return Err(format!("«{name}» не логическая ({})", cvar.type_name()));
    }
    let value = cvar.value_text() == "true";
    cvars
        .set(
            name,
            if value { "false" } else { "true" },
            CVarSource::Console,
        )
        .map_err(|error| error.to_string())?;
    Ok(format!("{name} = {}", cvar.value_text()))
}

#[cfg(test)]
mod tests {
    use engine_core::CVar;

    use super::*;
    use crate::console::{LineKind, execute};

    fn world() -> World {
        let mut world = World::new();
        world.init_resource::<CVars>();
        let mut console = Console::default();
        for command in commands() {
            console.add_command(command);
        }
        world.insert_resource(console);
        world
    }

    #[test]
    fn toggle_and_reset() {
        static FLAG: CVar<bool> = CVar::new("test.builtin.flag", false, "");
        static COUNT: CVar<u32> = CVar::new("test.builtin.count", 3, "");
        let mut world = world();
        world.resource_mut::<CVars>().register(&FLAG);
        world.resource_mut::<CVars>().register(&COUNT);
        execute(&mut world, "toggle test.builtin.flag");
        assert!(FLAG.get());
        execute(&mut world, "toggle test.builtin.count");
        let console = world.resource::<Console>();
        assert_eq!(console.lines().back().unwrap().kind, LineKind::Error);
        execute(&mut world, "test.builtin.count 10");
        execute(&mut world, "reset test.builtin.count");
        assert_eq!(COUNT.get(), 3);
    }

    #[test]
    fn find_and_help() {
        static FOG: CVar<f32> = CVar::new("test.builtin.fog", 0.5, "Плотность тумана");
        let mut world = world();
        world.resource_mut::<CVars>().register(&FOG);
        execute(&mut world, "find туман");
        execute(&mut world, "help");
        execute(&mut world, "help toggle");
        let text: Vec<String> = world
            .resource::<Console>()
            .lines()
            .iter()
            .map(|l| l.text.clone())
            .collect();
        assert!(text.iter().any(|t| t.contains("test.builtin.fog")));
        assert!(text.iter().any(|t| t.contains("Команды:")));
        assert!(text.iter().any(|t| t.contains("toggle <переменная>")));
        execute(&mut world, "clear");
        assert!(world.resource::<Console>().lines().is_empty());
    }
}
