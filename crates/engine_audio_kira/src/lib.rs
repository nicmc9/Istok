//! Бэкенд аудио на `kira` (R2-08). Вывод — `cpal`: `CoreAudio` (macOS, iOS), `AAudio` (Android),
//! ALSA (Linux), `WebAudio` (Web).
//!
//! ```ignore
//! App::new().add_plugin(AssetPlugin).add_plugin(KiraAudioPlugin) // + компоненты engine_audio
//! ```
//!
//! - Короткие звуки декодируются целиком при первом проигрывании и кешируются (копия звука —
//!   счётчик ссылок на те же отсчёты).
//! - Музыка (`stream`) декодируется по ходу воспроизведения в потоке kira. На Web потоков нет —
//!   там клип декодируется целиком.
//! - Типы kira наружу не выходят.

use std::collections::HashMap;
use std::io::Cursor;
use std::time::Duration;

use engine_assets::AssetId;
use engine_audio::{
    AUDIO_ENABLED, AudioBackend, AudioClip, AudioOutput, AudioPlugin, PlayMode, SoundId,
    SoundParams, SoundState,
};
use engine_core::{App, Plugin};
use kira::sound::PlaybackState;
use kira::sound::static_sound::{StaticSoundData, StaticSoundHandle};
use kira::{AudioManager, AudioManagerSettings, Decibels, Tween};

/// Подключает звук с бэкендом kira. Требует `AssetPlugin`.
#[derive(Debug, Default)]
pub struct KiraAudioPlugin;

impl Plugin for KiraAudioPlugin {
    fn build(&self, app: &mut App) {
        app.register_cvar(&AUDIO_ENABLED);
        if !AUDIO_ENABLED.get() {
            tracing::info!("Звук выключен (audio.enabled)");
            return;
        }
        app.world_mut()
            .insert_non_send(AudioOutput::new(Box::new(|| {
                KiraBackend::<kira::DefaultBackend>::new()
                    .map(|backend| Box::new(backend) as Box<dyn AudioBackend>)
            })));
        app.add_plugin(AudioPlugin);
    }
}

/// Сколько звуков может звучать одновременно.
const MAX_SOUNDS: usize = 256;
/// Короткая плавность для остановки и паузы: резкий обрыв волны слышен как щелчок.
const DECLICK: Tween = Tween {
    start_time: kira::StartTime::Immediate,
    duration: Duration::from_millis(15),
    easing: kira::Easing::Linear,
};

enum Handle {
    Static(StaticSoundHandle),
    #[cfg(not(target_arch = "wasm32"))]
    Stream(kira::sound::streaming::StreamingSoundHandle<kira::sound::FromFileError>),
}

// [Rust] Макрос вместо повторения `match` для двух видов хэндлов с одинаковыми методами
// (общего трейта у них в kira нет)
macro_rules! with_handle {
    ($handle:expr, $h:ident => $body:expr) => {
        match $handle {
            Handle::Static($h) => $body,
            #[cfg(not(target_arch = "wasm32"))]
            Handle::Stream($h) => $body,
        }
    };
}

/// Бэкенд на kira. Параметр `B` — вывод kira: `cpal` в приложении, `MockBackend` в тестах.
pub struct KiraBackend<B: kira::backend::Backend> {
    manager: AudioManager<B>,
    /// Декодированные клипы.
    decoded: HashMap<AssetId<AudioClip>, StaticSoundData>,
    sounds: HashMap<SoundId, Handle>,
    next_id: u64,
}

impl<B: kira::backend::Backend> KiraBackend<B>
where
    B::Error: std::fmt::Debug,
{
    /// Открывает устройство вывода (настройки вывода kira — по умолчанию).
    pub fn new() -> Result<Self, String>
    where
        B::Settings: Default,
    {
        let settings = AudioManagerSettings::<B> {
            main_track_builder: kira::track::MainTrackBuilder::new().sound_capacity(MAX_SOUNDS),
            ..AudioManagerSettings::default()
        };
        let manager =
            AudioManager::<B>::new(settings).map_err(|error: B::Error| format!("{error:?}"))?;
        Ok(Self {
            manager,
            decoded: HashMap::new(),
            sounds: HashMap::new(),
            next_id: 0,
        })
    }

    fn decoded(
        &mut self,
        clip: AssetId<AudioClip>,
        data: &AudioClip,
    ) -> Result<StaticSoundData, String> {
        if let Some(sound) = self.decoded.get(&clip) {
            return Ok(sound.clone());
        }
        let _zone = tracing::trace_span!("audio_decode").entered();
        let sound = StaticSoundData::from_cursor(Cursor::new(data.data.clone()))
            .map_err(|error| format!("не декодирован: {error}"))?;
        self.decoded.insert(clip, sound.clone());
        Ok(sound)
    }
}

impl<B: kira::backend::Backend> KiraBackend<B> {
    /// Запомнить хэндл звука под новым id.
    fn register(&mut self, handle: Handle) -> SoundId {
        let id = SoundId(self.next_id);
        self.next_id += 1;
        self.sounds.insert(id, handle);
        id
    }
}

/// Линейная амплитуда → децибелы. Ниже −60 дБ kira считает тишиной.
fn decibels(volume: f32) -> Decibels {
    if volume <= 0.001 {
        Decibels::SILENCE
    } else {
        Decibels(20.0 * volume.log10())
    }
}

impl<B: kira::backend::Backend + 'static> AudioBackend for KiraBackend<B>
where
    B::Error: std::fmt::Debug,
{
    fn describe(&self) -> String {
        format!("kira, до {MAX_SOUNDS} звуков одновременно")
    }

    fn play(
        &mut self,
        clip: AssetId<AudioClip>,
        data: &AudioClip,
        mode: PlayMode,
        params: SoundParams,
    ) -> Result<SoundId, String> {
        // На Web потокового воспроизведения нет — ветка не компилируется, клип декодируется целиком
        #[cfg(not(target_arch = "wasm32"))]
        if mode.stream {
            let handle = self.play_stream(data, mode, params)?;
            return Ok(self.register(handle));
        }
        let mut sound = self
            .decoded(clip, data)?
            .volume(decibels(params.volume))
            .panning(params.pan)
            .playback_rate(f64::from(params.speed));
        if mode.looped {
            sound = sound.loop_region(..);
        }
        let handle = self
            .manager
            .play(sound)
            .map_err(|error| error.to_string())?;
        Ok(self.register(Handle::Static(handle)))
    }

    fn set_params(&mut self, sound: SoundId, params: SoundParams) {
        if let Some(handle) = self.sounds.get_mut(&sound) {
            with_handle!(handle, h => {
                h.set_volume(decibels(params.volume), Tween::default());
                h.set_panning(params.pan, Tween::default());
                h.set_playback_rate(f64::from(params.speed), Tween::default());
            });
        }
    }

    fn set_paused(&mut self, sound: SoundId, paused: bool) {
        if let Some(handle) = self.sounds.get_mut(&sound) {
            with_handle!(handle, h => if paused { h.pause(DECLICK) } else { h.resume(DECLICK) });
        }
    }

    fn stop(&mut self, sound: SoundId) {
        if let Some(mut handle) = self.sounds.remove(&sound) {
            with_handle!(&mut handle, h => h.stop(DECLICK));
        }
    }

    fn state(&self, sound: SoundId) -> SoundState {
        let Some(handle) = self.sounds.get(&sound) else {
            return SoundState::Finished;
        };
        match with_handle!(handle, h => h.state()) {
            PlaybackState::Stopped | PlaybackState::Stopping => SoundState::Finished,
            PlaybackState::Paused | PlaybackState::Pausing | PlaybackState::WaitingToResume => {
                SoundState::Paused
            }
            _ => SoundState::Playing,
        }
    }

    fn forget_clip(&mut self, clip: AssetId<AudioClip>) {
        self.decoded.remove(&clip);
    }
}

impl<B: kira::backend::Backend> KiraBackend<B>
where
    B::Error: std::fmt::Debug,
{
    #[cfg(not(target_arch = "wasm32"))]
    fn play_stream(
        &mut self,
        data: &AudioClip,
        mode: PlayMode,
        params: SoundParams,
    ) -> Result<Handle, String> {
        use kira::sound::streaming::StreamingSoundData;
        // Байты клипа — `Arc<[u8]>`: поток читает их без копии
        let mut sound = StreamingSoundData::from_cursor(Cursor::new(data.data.clone()))
            .map_err(|error| format!("не декодирован: {error}"))?
            .volume(decibels(params.volume))
            .panning(params.pan)
            .playback_rate(f64::from(params.speed));
        if mode.looped {
            sound = sound.loop_region(..);
        }
        let handle = self
            .manager
            .play(sound)
            .map_err(|error| error.to_string())?;
        Ok(Handle::Stream(handle))
    }
}

impl<B: kira::backend::Backend> std::fmt::Debug for KiraBackend<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KiraBackend")
            .field("sounds", &self.sounds.len())
            .field("decoded", &self.decoded.len())
            .finish_non_exhaustive()
    }
}
