//! Чит-консоль (R2-10).
//!
//! - `имя` — показать переменную (CVar), `имя значение` — изменить. Флаги проверяются как
//!   у любого изменения из консоли: `STARTUP` — нельзя после запуска, `CHEAT` — только
//!   при `cheats = true`.
//! - Команды регистрирует любой плагин: [`ConsoleAppExt::add_console_command`]. Обработчик
//!   получает `&mut World` и аргументы. Встроенные: `help`, `find`, `reset`, `toggle`,
//!   `clear`, `history`.
//! - Tab — дополнение имени, ↑/↓ — история, подсказки со значениями — под строкой ввода.
//! - Открыть: F5, кнопка в окне «Статистика» или `--cvar debug.console=true`.
//! - Из кода: `console.submit("render.bloom false")` — выполнится в стадии `Update`.

mod builtin;
mod console;
mod ui;

use engine_core::{App, CVar, CVarFlags, CVarString, Plugin, Stage};
use engine_ecs::prelude::*;
use engine_input::{Input, Key};

pub use console::{
    CommandResult, Completion, Console, ConsoleCommand, ConsoleLine, LineKind, execute,
    split_commands, tokenize,
};

pub static DEBUG_CONSOLE: CVar<bool> = CVar::new("debug.console", false, "Показывать консоль");

/// Команды при запуске — для тестов и CI: `--cvar "console.exec=cheats true; sun 0.5"`.
pub static CONSOLE_EXEC: CVar<CVarString> = CVar::new(
    "console.exec",
    std::borrow::Cow::Borrowed(""),
    "Команды консоли при запуске (через ;)",
)
.with_flags(CVarFlags::STARTUP);

/// Подключает консоль. Требует `RenderPlugin` (окно на egui) и `InputPlugin`.
#[derive(Debug, Default)]
pub struct ConsolePlugin;

impl Plugin for ConsolePlugin {
    fn build(&self, app: &mut App) {
        app.register_cvar(&DEBUG_CONSOLE)
            .register_cvar(&CONSOLE_EXEC)
            .add_systems(Stage::Startup, exec_on_startup);
        for command in builtin::commands() {
            app.add_console_command(command);
        }
        app.add_systems(
            Stage::Update,
            // [Rust] `.chain()` — строго по порядку: введённое в окне выполняется в том же кадре
            (hotkey, ui::console_window, ui::run_pending).chain(),
        );
    }
}

/// `console.exec` → в очередь; выполнится в первом `Update`, когда все плагины готовы.
fn exec_on_startup(mut console: ResMut<Console>) {
    let commands = CONSOLE_EXEC.get();
    if !commands.trim().is_empty() {
        console.submit(commands.as_ref());
    }
}

/// F5 — открыть / закрыть. Работает и при фокусе в консоли: F-клавиши текст не вводят.
fn hotkey(input: Res<Input>) {
    if input.key_pressed(Key::F5) {
        DEBUG_CONSOLE.set(!DEBUG_CONSOLE.get());
    }
}

/// Регистрация команд консоли из плагинов.
// [Rust] «Трейт-расширение»: добавляет метод чужому типу `App`, как `AssetAppExt`.
pub trait ConsoleAppExt {
    fn add_console_command(&mut self, command: ConsoleCommand) -> &mut Self;
}

impl ConsoleAppExt for App {
    fn add_console_command(&mut self, command: ConsoleCommand) -> &mut Self {
        // Плагин может добавить команду раньше, чем подключена консоль, — ресурс создаётся
        // при первой регистрации
        self.world_mut()
            .get_resource_or_init::<Console>()
            .add_command(command);
        self
    }
}
