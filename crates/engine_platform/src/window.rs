use std::sync::Arc;

use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, WindowHandle,
};
use winit::window::Window;

/// Физический размер окна в пикселях.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WindowSize {
    pub width: u32,
    pub height: u32,
}

/// Безопасная зона: отступы от краёв окна в физических пикселях. Под ними вырез экрана,
/// часы, полоска «домой» (iOS), системные панели (Android). На desktop и Web — нули.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SafeArea {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
}

// На desktop и Web отступов нет, функция нужна только iOS, Android и тестам
#[cfg_attr(not(any(target_os = "ios", target_os = "android")), allow(dead_code))]
impl SafeArea {
    /// Отступы между внешним прямоугольником и внутренним, оба — от левого верхнего угла.
    fn between(outer: (i32, i32, u32, u32), inner: (i32, i32, u32, u32)) -> Self {
        let (ox, oy, ow, oh) = outer;
        let (ix, iy, iw, ih) = inner;
        let left = u32::try_from(ix - ox).unwrap_or(0);
        let top = u32::try_from(iy - oy).unwrap_or(0);
        Self {
            left,
            top,
            right: ow.saturating_sub(left.saturating_add(iw)),
            bottom: oh.saturating_sub(top.saturating_add(ih)),
        }
    }
}

/// Окно приложения. Дёшево копируется: все копии указывают на одно окно.
///
/// Реализует `HasWindowHandle`/`HasDisplayHandle` — по ним рендер создаёт GPU-surface,
/// не зная, что внутри `winit`.
// [Rust] `Arc` — атомарный счётчик ссылок, аналог `std::shared_ptr`. `clone()` только
// увеличивает счётчик. Arc нужен, потому что окно будет использовать и поток рендера.
#[derive(Debug, Clone)]
pub struct PlatformWindow {
    inner: Arc<Window>,
}

impl PlatformWindow {
    pub(crate) fn new(window: Window) -> Self {
        Self {
            inner: Arc::new(window),
        }
    }

    /// Размер области, в которую рисует surface.
    pub fn size(&self) -> WindowSize {
        // На iOS `inner_size` — безопасная зона, а слой рисования (и событие Resized) —
        // весь экран. Возьми мы inner, surface настраивался бы меньше, чем он есть.
        #[cfg(target_os = "ios")]
        let size = self.inner.outer_size();
        #[cfg(not(target_os = "ios"))]
        let size = self.inner.inner_size();
        WindowSize {
            width: size.width,
            height: size.height,
        }
    }

    /// Отступы безопасной зоны. Могут меняться при повороте экрана; на Android ОС не
    /// присылает об этом событие, поэтому движок опрашивает их каждый кадр.
    pub fn safe_area(&self) -> SafeArea {
        platform_safe_area(&self.inner)
    }

    pub fn scale_factor(&self) -> f64 {
        self.inner.scale_factor()
    }

    /// Попросить ОС прислать событие перерисовки (на Web — в следующем requestAnimationFrame).
    pub fn request_redraw(&self) {
        self.inner.request_redraw();
    }
}

// [Rust] Реализация чужого трейта для своего типа — так тип «подключается» к внешнему API.
// Аналог наследования от интерфейса, но может быть добавлена отдельно от объявления типа.
impl HasWindowHandle for PlatformWindow {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        self.inner.window_handle()
    }
}

impl HasDisplayHandle for PlatformWindow {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        self.inner.display_handle()
    }
}

/// iOS: `inner_*` у winit — безопасная зона, `outer_*` — весь экран.
#[cfg(target_os = "ios")]
fn platform_safe_area(window: &Window) -> SafeArea {
    let (Ok(inner), Ok(outer)) = (window.inner_position(), window.outer_position()) else {
        return SafeArea::default();
    };
    let (inner_size, outer_size) = (window.inner_size(), window.outer_size());
    SafeArea::between(
        (outer.x, outer.y, outer_size.width, outer_size.height),
        (inner.x, inner.y, inner_size.width, inner_size.height),
    )
}

/// Android: прямоугольник содержимого без системных панелей, в пикселях окна.
#[cfg(target_os = "android")]
fn platform_safe_area(window: &Window) -> SafeArea {
    use winit::platform::android::WindowExtAndroid;
    let rect = window.content_rect();
    let size = window.inner_size();
    // До первого сообщения от ОС прямоугольник пустой — отступов не знаем
    if rect.right <= rect.left || rect.bottom <= rect.top {
        return SafeArea::default();
    }
    let width = u32::try_from(rect.right - rect.left).unwrap_or(0);
    let height = u32::try_from(rect.bottom - rect.top).unwrap_or(0);
    SafeArea::between(
        (0, 0, size.width, size.height),
        (rect.left, rect.top, width, height),
    )
}

#[cfg(not(any(target_os = "ios", target_os = "android")))]
fn platform_safe_area(_window: &Window) -> SafeArea {
    SafeArea::default()
}

#[cfg(test)]
mod tests {
    use super::SafeArea;

    #[test]
    fn insets_from_outer_and_inner_rects() {
        // iPhone 15 в портрете: экран 1179×2556, сверху 177, снизу 102
        let area = SafeArea::between((0, 0, 1179, 2556), (0, 177, 1179, 2277));
        assert_eq!(
            area,
            SafeArea {
                left: 0,
                top: 177,
                right: 0,
                bottom: 102
            }
        );
    }

    #[test]
    fn inner_outside_outer_gives_zero() {
        let area = SafeArea::between((10, 10, 100, 100), (0, 0, 200, 200));
        assert_eq!(area, SafeArea::default());
    }
}
