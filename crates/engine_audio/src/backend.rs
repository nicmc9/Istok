//! Трейт бэкенда аудио и [`AudioOutput`] — бэкенд, создаваемый лениво.
//!
//! Бэкенд только проигрывает: громкость и панораму (в том числе для 3D-звука) считает фасад.
//! Так новый бэкенд — это немного кода: открыть устройство, проиграть клип, менять параметры.

use std::collections::HashMap;

use engine_assets::AssetId;
use engine_ecs::prelude::*;

use crate::AudioClip;

/// Идентификатор проигрываемого звука, выдаёт бэкенд.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SoundId(pub u64);

/// Параметры звука, которые можно менять на ходу.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoundParams {
    /// Линейная амплитуда (0 — тишина, 1 — как в файле).
    pub volume: f32,
    /// −1 — только левый канал, 0 — центр, 1 — только правый.
    pub pan: f32,
    /// Скорость воспроизведения (и высота тона).
    pub speed: f32,
}

/// Как проигрывать: задаётся при запуске и потом не меняется.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayMode {
    pub looped: bool,
    pub stream: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoundState {
    Playing,
    Paused,
    /// Доиграл, остановлен или неизвестен бэкенду.
    Finished,
}

/// Бэкенд аудио.
///
/// [Rust] Без `Send`: на Web объекты `WebAudio` привязаны к главному потоку. Поэтому бэкенд
/// хранится в ресурсе «только для главного потока» (`NonSend`), и системы звука выполняются там.
pub trait AudioBackend: 'static {
    /// Описание устройства — для лога.
    fn describe(&self) -> String;
    fn play(
        &mut self,
        clip: AssetId<AudioClip>,
        data: &AudioClip,
        mode: PlayMode,
        params: SoundParams,
    ) -> Result<SoundId, String>;
    fn set_params(&mut self, sound: SoundId, params: SoundParams);
    fn set_paused(&mut self, sound: SoundId, paused: bool);
    /// Остановить и забыть звук.
    fn stop(&mut self, sound: SoundId);
    fn state(&self, sound: SoundId) -> SoundState;
    /// Клип изменился или выгружен — выбросить его декодированную копию, если есть.
    fn forget_clip(&mut self, clip: AssetId<AudioClip>);
}

/// Создание бэкенда: вызывается при разблокировке звука.
pub type BackendFactory = Box<dyn Fn() -> Result<Box<dyn AudioBackend>, String>>;

/// Звучащий источник: что запущено для сущности с `AudioPlayer`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Voice {
    pub sound: SoundId,
    pub clip: AssetId<AudioClip>,
    pub params: SoundParams,
    pub paused: bool,
    /// Доиграл (не по кругу) — больше не запускать, пока не сменят клип.
    pub finished: bool,
}

/// Аудиовыход (ресурс главного потока). Бэкенд создаётся при первом кадре на native и после
/// первого жеста пользователя на Web: браузер не даёт запускать звук раньше.
pub struct AudioOutput {
    factory: BackendFactory,
    pub(crate) backend: Option<Box<dyn AudioBackend>>,
    /// Создание бэкенда не удалось (нет устройства) — больше не пытаться.
    pub(crate) failed: bool,
    pub(crate) voices: HashMap<Entity, Voice>,
}

impl AudioOutput {
    pub fn new(factory: BackendFactory) -> Self {
        Self {
            factory,
            backend: None,
            failed: false,
            voices: HashMap::new(),
        }
    }

    /// Создаёт бэкенд, если его ещё нет. Ошибка пишется в лог один раз.
    pub(crate) fn start(&mut self) {
        if self.backend.is_some() || self.failed {
            return;
        }
        match (self.factory)() {
            Ok(backend) => {
                tracing::info!("Аудио: {}", backend.describe());
                self.backend = Some(backend);
            }
            Err(error) => {
                tracing::error!("Аудио недоступно, звука не будет: {error}");
                self.failed = true;
            }
        }
    }
}

impl std::fmt::Debug for AudioOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioOutput")
            .field("started", &self.backend.is_some())
            .field("failed", &self.failed)
            .field("voices", &self.voices.len())
            .finish_non_exhaustive()
    }
}
