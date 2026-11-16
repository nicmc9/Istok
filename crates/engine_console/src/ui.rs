//! Окно консоли на egui: журнал, строка ввода, подсказки, история.

use engine_core::CVars;
use engine_ecs::prelude::*;
use engine_render::{Ui, egui};

use crate::DEBUG_CONSOLE;
use crate::console::{Console, LineKind, execute};

/// Сколько подсказок показывать под строкой ввода.
const MAX_HINTS: usize = 8;

/// Состояние окна между кадрами.
#[derive(Default)]
pub(crate) struct ConsoleUiState {
    input: String,
    /// Позиция в истории при листании ↑/↓; `None` — редактируется новая строка.
    history_pos: Option<usize>,
    /// Новая строка, сохранённая на время листания истории.
    draft: String,
    was_open: bool,
    /// Вернуть фокус строке ввода (после открытия и после Enter).
    focus: bool,
}

/// Клавиши, которые консоль перехватывает у текстового поля (иначе Tab и стрелки
/// переводят фокус на другие элементы).
#[derive(Default)]
struct Keys {
    tab: bool,
    up: bool,
    down: bool,
}

pub(crate) fn console_window(
    ui: Res<Ui>,
    cvars: Res<CVars>,
    mut console: ResMut<Console>,
    mut state: Local<ConsoleUiState>,
) {
    if !DEBUG_CONSOLE.get() {
        state.was_open = false;
        return;
    }
    if !state.was_open {
        state.was_open = true;
        state.focus = true;
    }
    let mut open = true;
    egui::Window::new("Консоль")
        .open(&mut open)
        .default_pos([340.0, 12.0])
        .default_size([620.0, 360.0])
        .resizable(true)
        .show(ui.ctx(), |ui| {
            let hints_height = ui.text_style_height(&egui::TextStyle::Monospace) * 2.0;
            let input_height = ui.spacing().interact_size.y + hints_height + 16.0;
            egui::ScrollArea::vertical()
                .auto_shrink(false)
                .stick_to_bottom(true)
                .max_height((ui.available_height() - input_height).max(60.0))
                .show(ui, |ui| {
                    for line in console.lines() {
                        let color = match line.kind {
                            LineKind::Input => egui::Color32::from_rgb(140, 180, 255),
                            LineKind::Output => ui.visuals().text_color(),
                            LineKind::Error => egui::Color32::LIGHT_RED,
                        };
                        ui.label(egui::RichText::new(&line.text).monospace().color(color));
                    }
                });
            ui.separator();
            input_line(ui, &cvars, &mut console, &mut state);
        });
    if !open {
        DEBUG_CONSOLE.set(false);
    }
}

fn input_line(ui: &mut egui::Ui, cvars: &CVars, console: &mut Console, state: &mut ConsoleUiState) {
    let id = ui.make_persistent_id("console_input");
    let keys = if ui.memory(|m| m.has_focus(id)) {
        ui.input_mut(|i| Keys {
            tab: i.consume_key(egui::Modifiers::NONE, egui::Key::Tab),
            up: i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp),
            down: i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown),
        })
    } else {
        Keys::default()
    };
    let mut moved = false;
    if keys.tab
        && let Some(text) = console.complete(cvars, &state.input).text
    {
        state.input = text;
        moved = true;
    }
    if keys.up || keys.down {
        moved |= browse_history(state, console.history(), keys.up);
    }
    if moved {
        // Курсор — в конец подставленного текста
        let mut edit = egui::text_edit::TextEditState::load(ui.ctx(), id).unwrap_or_default();
        let end = egui::text::CCursor::new(state.input.chars().count());
        edit.cursor
            .set_char_range(Some(egui::text::CCursorRange::one(end)));
        edit.store(ui.ctx(), id);
    }

    let response = ui.add(
        egui::TextEdit::singleline(&mut state.input)
            .id(id)
            .font(egui::TextStyle::Monospace)
            .desired_width(f32::INFINITY)
            .hint_text("команда или переменная; help — справка, Tab — дополнение, стрелки вверх/вниз — история"),
    );
    if response.changed() {
        state.history_pos = None;
    }
    if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
        let line = std::mem::take(&mut state.input);
        if !line.trim().is_empty() {
            console.submit(line);
        }
        state.history_pos = None;
        state.focus = true;
    }
    if std::mem::take(&mut state.focus) {
        response.request_focus();
    }

    // Подсказки к последнему слову: имена и текущие значения переменных
    let word = state.input.split_whitespace().last().unwrap_or("");
    let ends_with_space = state.input.ends_with(char::is_whitespace);
    if !word.is_empty() && !ends_with_space {
        let candidates = console.complete(cvars, &state.input).candidates;
        ui.horizontal_wrapped(|ui| {
            for name in candidates.iter().take(MAX_HINTS) {
                let text = match cvars.get(name) {
                    Some(cvar) => format!("{name} = {}", cvar.value_text()),
                    None => (*name).to_owned(),
                };
                ui.label(egui::RichText::new(text).monospace().weak());
            }
            if candidates.len() > MAX_HINTS {
                ui.label(
                    egui::RichText::new(format!("… ещё {}", candidates.len() - MAX_HINTS)).weak(),
                );
            }
        });
    }
}

/// ↑ — старее, ↓ — новее; ниже последней — возврат к недописанной строке.
fn browse_history(state: &mut ConsoleUiState, history: &[String], up: bool) -> bool {
    if history.is_empty() {
        return false;
    }
    let pos = match (state.history_pos, up) {
        (None, true) => {
            state.draft = state.input.clone();
            Some(history.len() - 1)
        }
        (None, false) => return false,
        (Some(pos), true) => Some(pos.saturating_sub(1)),
        (Some(pos), false) if pos + 1 < history.len() => Some(pos + 1),
        (Some(_), false) => None,
    };
    state.history_pos = pos;
    state.input = match pos {
        Some(pos) => history[pos].clone(),
        None => std::mem::take(&mut state.draft),
    };
    true
}

/// Выполняет строки, накопившиеся за кадр (из окна и из кода через `Console::submit`).
// [Rust] Система с `&mut World` — «эксклюзивная»: выполняется одна, без параллельных систем.
// Командам нужен доступ ко всему миру, поэтому иначе нельзя.
pub(crate) fn run_pending(world: &mut World) {
    let lines = world.resource_mut::<Console>().take_pending();
    for line in lines {
        execute(world, &line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_browsing_restores_draft() {
        let history = ["a".to_owned(), "b".to_owned()];
        let mut state = ConsoleUiState {
            input: "черновик".to_owned(),
            ..ConsoleUiState::default()
        };
        browse_history(&mut state, &history, true);
        assert_eq!(state.input, "b");
        browse_history(&mut state, &history, true);
        browse_history(&mut state, &history, true);
        assert_eq!(state.input, "a", "дальше начала не уходит");
        browse_history(&mut state, &history, false);
        assert_eq!(state.input, "b");
        browse_history(&mut state, &history, false);
        assert_eq!(state.input, "черновик");
        assert!(!browse_history(&mut state, &history, false));
    }
}
