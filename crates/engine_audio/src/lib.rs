//! Аудио (R2-08): фасад. Бэкенд подключается отдельным крейтом (`engine_audio_kira`).
//!
//! - [`AudioClip`] — звуковой файл как ресурс (`.wav`, `.ogg`, `.mp3`).
//! - [`AudioPlayer`] — источник на сущности: громкость, скорость, повтор, пауза, шина,
//!   3D-звук ([`Spatial`]), потоковое декодирование для музыки. Меняется на лету.
//! - [`AudioListener`] — «уши» для 3D-звука (если его нет — камера). Громкость и панораму
//!   3D-источников считает фасад, бэкенду уходят готовые значения.
//! - CVar: `audio.enabled` — открыть устройство (по умолчанию нет), `audio.volume`,
//!   `audio.music_volume`, `audio.sfx_volume`, `audio.mute`.
//! - На Web звук запускается после первого жеста пользователя (политика браузеров);
//!   без звукового устройства — ошибка в лог, остальное работает.

mod backend;
mod clip;
mod components;
mod systems;

use engine_assets::AssetAppExt;
use engine_core::{App, CVar, CVarFlags, Plugin, Stage};
use engine_ecs::prelude::*;

pub use backend::{
    AudioBackend, AudioOutput, BackendFactory, PlayMode, SoundId, SoundParams, SoundState,
};
pub use clip::{AudioClip, AudioClipLoader};
pub use components::{AudioListener, AudioPlayer, Bus, Spatial};

pub static AUDIO_ENABLED: CVar<bool> = CVar::new(
    "audio.enabled",
    false,
    "Открыть звуковое устройство. Выкл — бэкенд не запускается",
)
.with_flags(CVarFlags::STARTUP);
pub static AUDIO_VOLUME: CVar<f32> = CVar::new("audio.volume", 1.0, "Общая громкость, 0..1");
pub static AUDIO_MUSIC_VOLUME: CVar<f32> =
    CVar::new("audio.music_volume", 0.7, "Громкость музыки, 0..1");
pub static AUDIO_SFX_VOLUME: CVar<f32> =
    CVar::new("audio.sfx_volume", 1.0, "Громкость звуковых эффектов, 0..1");
pub static AUDIO_MUTE: CVar<bool> = CVar::new("audio.mute", false, "Выключить звук");

/// Состояние звука для статистики и отладки.
#[derive(Resource, Debug, Default, Clone, Copy)]
pub struct AudioStats {
    /// Устройство открыто (на Web — после первого жеста).
    pub started: bool,
    /// Звучащих источников.
    pub playing: usize,
}

/// Системы звука. Добавляется плагином бэкенда, который кладёт [`AudioOutput`] со своей
/// фабрикой. Требует `AssetPlugin`.
#[derive(Debug, Default)]
pub struct AudioPlugin;

impl Plugin for AudioPlugin {
    fn build(&self, app: &mut App) {
        assert!(
            app.world().contains_non_send::<AudioOutput>(),
            "AudioPlugin требует бэкенд: добавьте плагин бэкенда (например, KiraAudioPlugin)"
        );
        app.register_type::<AudioPlayer>()
            .register_type::<AudioListener>()
            .register_cvar(&AUDIO_VOLUME)
            .register_cvar(&AUDIO_MUSIC_VOLUME)
            .register_cvar(&AUDIO_SFX_VOLUME)
            .register_cvar(&AUDIO_MUTE)
            .register_loader(AudioClipLoader)
            .init_resource::<AudioStats>()
            .add_systems(
                Stage::PostUpdate,
                (
                    systems::unlock,
                    systems::forget_clips,
                    systems::update_players,
                )
                    .chain()
                    // 3D-звук читает GlobalTransform текущего кадра
                    .after(engine_scene::propagate_transforms),
            );
    }
}
