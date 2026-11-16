//! Компоненты звука: источник [`AudioPlayer`] и слушатель [`AudioListener`].

use engine_assets::Handle;
use engine_ecs::prelude::*;

use crate::AudioClip;

/// Шина микшера: у каждой своя громкость (CVar `audio.music_volume`, `audio.sfx_volume`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Reflect)]
#[reflect(Default)]
pub enum Bus {
    #[default]
    Effects,
    Music,
}

/// Звук в пространстве: громкость падает с расстоянием до слушателя, панорама — по тому,
/// слева или справа источник.
#[derive(Debug, Clone, Copy, PartialEq, Reflect)]
#[reflect(Default)]
pub struct Spatial {
    /// Ближе этого расстояния (м) громкость полная.
    pub min_distance: f32,
    /// Дальше — тишина.
    pub max_distance: f32,
}

impl Default for Spatial {
    fn default() -> Self {
        Self {
            min_distance: 1.0,
            max_distance: 30.0,
        }
    }
}

/// Источник звука. Проигрывается, как только клип загружен (и звук разблокирован — на Web
/// это первое касание, клик или клавиша).
///
/// ```ignore
/// // Музыка: по кругу, шина Music, потоковое декодирование
/// commands.spawn(AudioPlayer::music(server.load("audio/theme.ogg")));
/// // Разовый звук в точке мира: сущность удалится, когда звук закончится
/// commands.spawn((AudioPlayer::once(hit.clone()).spatial(), Transform::from_translation(p)));
/// ```
///
/// Изменения полей применяются на лету: громкость, скорость, пауза. Новый `clip` или повторная
/// вставка компонента — проиграть сначала. Удаление компонента или сущности — остановить.
// Флаги независимы (повтор, пауза, поток, удаление) — перечисление их бы только запутало
#[allow(clippy::struct_excessive_bools)]
#[derive(Component, Debug, Clone, PartialEq, Reflect)]
#[reflect(Component)]
pub struct AudioPlayer {
    pub clip: Handle<AudioClip>,
    /// Громкость, 0..1 (линейная амплитуда; можно больше 1 — усиление).
    pub volume: f32,
    /// Скорость и высота тона: 2 — вдвое быстрее и на октаву выше.
    pub speed: f32,
    pub looped: bool,
    pub paused: bool,
    pub bus: Bus,
    /// `Some` — 3D-звук из позиции сущности (`GlobalTransform`); `None` — «в голове» слушателя.
    pub spatial: Option<Spatial>,
    /// Декодировать по ходу воспроизведения, а не целиком: для длинных треков (меньше памяти,
    /// нет паузы на декодирование). На Web недоступно — там клип декодируется целиком.
    pub stream: bool,
    /// Удалить сущность, когда звук закончится (разовые звуки).
    pub despawn_on_finish: bool,
}

impl AudioPlayer {
    /// Звук-эффект: один раз, шина `Effects`, без пространства.
    pub fn new(clip: Handle<AudioClip>) -> Self {
        Self {
            clip,
            volume: 1.0,
            speed: 1.0,
            looped: false,
            paused: false,
            bus: Bus::Effects,
            spatial: None,
            stream: false,
            despawn_on_finish: false,
        }
    }

    /// Разовый звук: сущность удаляется по окончании.
    pub fn once(clip: Handle<AudioClip>) -> Self {
        Self {
            despawn_on_finish: true,
            ..Self::new(clip)
        }
    }

    /// Музыка: по кругу, шина `Music`, потоковое декодирование.
    pub fn music(clip: Handle<AudioClip>) -> Self {
        Self {
            looped: true,
            bus: Bus::Music,
            stream: true,
            ..Self::new(clip)
        }
    }

    /// 3D-звук с расстояниями по умолчанию (1–30 м).
    #[must_use]
    pub fn spatial(mut self) -> Self {
        self.spatial = Some(Spatial::default());
        self
    }

    #[must_use]
    pub fn with_volume(mut self, volume: f32) -> Self {
        self.volume = volume;
        self
    }

    #[must_use]
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = speed;
        self
    }

    #[must_use]
    pub fn looped(mut self) -> Self {
        self.looped = true;
        self
    }
}

/// Слушатель — «уши» для 3D-звука, обычно на камере. Если слушателя нет, им считается камера.
#[derive(Component, Debug, Clone, Copy, Default, Reflect)]
#[reflect(Component, Default)]
pub struct AudioListener;
