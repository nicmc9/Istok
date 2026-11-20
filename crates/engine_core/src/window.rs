use std::borrow::Cow;

use engine_ecs::prelude::*;
use engine_platform::{SafeArea, WindowSize};

use crate::{CVar, CVarString};

/// Состояние окна. Доступно системам как `Res<Window>`.
#[derive(Resource, Debug, Default)]
pub struct Window {
    pub size: WindowSize,
    pub scale_factor: f64,
    /// Отступы безопасной зоны, физические пиксели. Обновляются в начале кадра.
    /// Экранный текст и интерфейс держатся внутри.
    pub safe_area: SafeArea,
}

/// Отладка: своя безопасная зона, чтобы проверить интерфейс на desktop без телефона.
pub static WINDOW_SAFE_AREA: CVar<CVarString> = CVar::new(
    "window.safe_area",
    Cow::Borrowed(""),
    "Отладка: отступы «слева сверху справа снизу» в точках вместо отступов ОС. Пусто — от ОС",
);

/// Отступы из CVar `window.safe_area`: четыре числа в точках. `None` — пусто или ошибка.
fn debug_safe_area(text: &str, scale_factor: f64) -> Option<SafeArea> {
    let values: Vec<f64> = text
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<_, _>>()
        .ok()?;
    let [left, top, right, bottom] = values[..] else {
        return None;
    };
    // [Rust] `as` у float → u32 обрезает дробь и зажимает в [0, u32::MAX] — то, что нужно
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let px = |points: f64| (points * scale_factor).round().max(0.0) as u32;
    Some(SafeArea {
        left: px(left),
        top: px(top),
        right: px(right),
        bottom: px(bottom),
    })
}

/// Отступы безопасной зоны → `Window`. Опрос каждый кадр: на Android ОС не сообщает о
/// смене отступов, а на iOS они меняются при повороте вместе с размером.
///
/// Только из главного потока и не из системы ECS: на iOS winit выполняет запрос к окну
/// синхронно в главном потоке. Система на рабочем потоке ждала бы главный, а главный —
/// конца кадра ECS: взаимная блокировка.
pub(crate) fn refresh_safe_area(world: &mut World, warned: &mut u64) {
    let custom = WINDOW_SAFE_AREA.get();
    let scale_factor = world.resource::<Window>().scale_factor;
    let area = if custom.is_empty() {
        world
            .get_resource::<MainWindow>()
            .map(|main| main.0.safe_area())
            .unwrap_or_default()
    } else {
        debug_safe_area(&custom, scale_factor.max(0.5)).unwrap_or_else(|| {
            if *warned != WINDOW_SAFE_AREA.version() {
                *warned = WINDOW_SAFE_AREA.version();
                tracing::warn!("window.safe_area = «{custom}»: нужно четыре числа");
            }
            SafeArea::default()
        })
    };
    // Сравнение до записи: иначе Window считался бы изменённым каждый кадр
    if world.resource::<Window>().safe_area != area {
        tracing::info!(
            "Безопасная зона: слева {}, сверху {}, справа {}, снизу {} px",
            area.left,
            area.top,
            area.right,
            area.bottom
        );
        world.resource_mut::<Window>().safe_area = area;
    }
}

/// Окно изменило размер (физические пиксели).
#[derive(Message, Debug, Clone, Copy)]
pub struct WindowResized {
    pub size: WindowSize,
    pub scale_factor: f64,
}

/// Смена состояния приложения.
#[derive(Message, Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppLifecycle {
    /// Приложение активно (в том числе после возврата из фона).
    Resumed,
    /// Приложение ушло в фон.
    Suspended,
}

/// Главное окно — нужно рендеру для создания GPU-surface.
/// Появляется в мире после первого `resumed`.
#[derive(Resource, Debug, Clone)]
pub struct MainWindow(pub engine_platform::PlatformWindow);

/// Синхронное событие платформы для плагинов, которым нельзя ждать следующего кадра.
///
/// В отличие от сообщения [`AppLifecycle`], хук вызывается сразу, внутри обработчика ОС.
/// Пример: на Android после `Suspended` окно уничтожается немедленно, и рендер должен
/// отпустить surface до возврата из хука.
#[derive(Debug, Clone, Copy)]
pub enum LifecycleHook<'a> {
    Resumed(&'a engine_platform::PlatformWindow),
    Suspended,
    Resized { size: WindowSize, scale_factor: f64 },
    Exiting,
}

/// Событие ввода от платформы. Читается через `MessageReader<PlatformInput>`;
/// удобное состояние (нажатые клавиши и т.д.) собирает `engine_input`.
///
/// [Rust] Обёртка нужна из-за правила сироты (orphan rule): трейт `Message` (из `bevy_ecs`)
/// нельзя реализовать для `InputEvent` (из `engine_platform`) в третьем крейте —
/// хотя бы трейт или тип должен быть «своим». Своя обёртка решает это.
#[derive(Message, Debug, Clone, Copy)]
pub struct PlatformInput(pub engine_platform::InputEvent);

/// Управление приложением из систем.
#[derive(Resource, Debug, Default)]
pub struct AppControl {
    exit_requested: bool,
}

impl AppControl {
    /// Завершить приложение после текущего кадра.
    pub fn request_exit(&mut self) {
        self.exit_requested = true;
    }

    pub fn exit_requested(&self) -> bool {
        self.exit_requested
    }
}

#[cfg(test)]
mod tests {
    use super::debug_safe_area;
    use engine_platform::SafeArea;

    #[test]
    fn debug_area_is_in_points() {
        assert_eq!(
            debug_safe_area("0 59 0 34", 2.0),
            Some(SafeArea {
                left: 0,
                top: 118,
                right: 0,
                bottom: 68
            })
        );
        assert_eq!(debug_safe_area("1 2 3", 1.0), None);
        assert_eq!(debug_safe_area("a b c d", 1.0), None);
    }
}
