//! Системы звука (стадия `PostUpdate`, после расчёта `GlobalTransform`):
//! разблокировка, запуск, изменение параметров, остановка, 3D-звук.

use engine_assets::{AssetEvent, Assets};
use engine_core::PlatformInput;
use engine_ecs::prelude::*;
use engine_math::{Mat4, Vec3};
use engine_platform::{InputEvent, TouchPhase};
use engine_scene::{Camera, GlobalTransform};

use crate::backend::{AudioOutput, PlayMode, SoundParams, SoundState, Voice};
use crate::components::{AudioListener, AudioPlayer, Bus, Spatial};
use crate::{
    AUDIO_MUSIC_VOLUME, AUDIO_MUTE, AUDIO_SFX_VOLUME, AUDIO_VOLUME, AudioClip, AudioStats,
};

/// Нужен ли жест пользователя перед запуском звука (политика автовоспроизведения браузеров).
const NEEDS_GESTURE: bool = cfg!(target_arch = "wasm32");

/// Громкость и панорама 3D-источника относительно слушателя.
///
/// Громкость — обратно пропорциональна расстоянию (как в жизни: вдвое дальше — вдвое тише)
/// и плавно уходит в ноль к `max_distance`, чтобы далёкие звуки не обрывались резко.
pub(crate) fn spatialize(listener: &Mat4, source: Vec3, spatial: Spatial) -> (f32, f32) {
    let offset = source - listener.w_axis.truncate();
    let distance = offset.length();
    let min = spatial.min_distance.max(0.01);
    let max = spatial.max_distance.max(min);
    let falloff = (min / distance.max(min)).min(1.0);
    let fade = 1.0 - ((distance - max * 0.75) / (max * 0.25)).clamp(0.0, 1.0);
    // Панорама: проекция направления на правую ось слушателя. Вблизи — к центру,
    // иначе звук «в упор» прыгал бы из уха в ухо
    let right = listener.x_axis.truncate().normalize_or_zero();
    let pan = offset.normalize_or_zero().dot(right) * (distance / min).min(1.0);
    (falloff * fade, pan.clamp(-1.0, 1.0))
}

/// Жест пользователя в кадре: нажатие клавиши, кнопки мыши, начало касания.
fn is_gesture(event: &InputEvent) -> bool {
    matches!(
        event,
        InputEvent::Key { pressed: true, .. }
            | InputEvent::MouseButton { pressed: true, .. }
            | InputEvent::Touch {
                phase: TouchPhase::Started,
                ..
            }
    )
}

pub(crate) fn unlock(mut output: NonSendMut<AudioOutput>, mut input: MessageReader<PlatformInput>) {
    let gesture = input.read().any(|event| is_gesture(&event.0));
    if output.backend.is_none() && (!NEEDS_GESTURE || gesture) {
        output.start();
    }
}

/// Изменённые и выгруженные клипы: бэкенд выбрасывает их декодированные копии.
pub(crate) fn forget_clips(
    mut output: NonSendMut<AudioOutput>,
    mut events: MessageReader<AssetEvent<AudioClip>>,
) {
    for event in events.read() {
        let (AssetEvent::Loaded(id) | AssetEvent::Unloaded(id)) = event else {
            continue;
        };
        if let Some(backend) = &mut output.backend {
            backend.forget_clip(*id);
        }
    }
}

fn bus_volume(bus: Bus) -> f32 {
    if AUDIO_MUTE.get() {
        return 0.0;
    }
    let bus = match bus {
        Bus::Effects => AUDIO_SFX_VOLUME.get(),
        Bus::Music => AUDIO_MUSIC_VOLUME.get(),
    };
    AUDIO_VOLUME.get().max(0.0) * bus.max(0.0)
}

/// Параметры звука для текущего кадра.
fn params(player: &AudioPlayer, position: Option<Vec3>, listener: Option<&Mat4>) -> SoundParams {
    let (gain, pan) = match (player.spatial, position, listener) {
        (Some(spatial), Some(position), Some(listener)) => spatialize(listener, position, spatial),
        _ => (1.0, 0.0),
    };
    SoundParams {
        volume: player.volume.max(0.0) * gain * bus_volume(player.bus),
        pan,
        speed: player.speed.max(0.01),
    }
}

/// Заметно ли изменение: на каждый кадр бэкенду уходят только реальные изменения.
fn differs(a: SoundParams, b: SoundParams) -> bool {
    (a.volume - b.volume).abs() > 1e-3
        || (a.pan - b.pan).abs() > 1e-3
        || (a.speed - b.speed).abs() > 1e-4
}

#[allow(clippy::too_many_arguments)] // система ECS: параметры — запрашиваемые данные
pub(crate) fn update_players(
    mut commands: Commands,
    mut output: NonSendMut<AudioOutput>,
    clips: Res<Assets<AudioClip>>,
    players: Query<(Entity, Ref<AudioPlayer>, Option<&GlobalTransform>)>,
    listeners: Query<&GlobalTransform, With<AudioListener>>,
    cameras: Query<&GlobalTransform, With<Camera>>,
    mut removed: RemovedComponents<AudioPlayer>,
    mut stats: ResMut<AudioStats>,
) {
    let output = &mut *output;
    // Источник удалён — звук останавливается, даже если бэкенда нет (тогда и звука нет)
    for entity in removed.read() {
        if let (Some(voice), Some(backend)) = (output.voices.remove(&entity), &mut output.backend) {
            backend.stop(voice.sound);
        }
    }
    let Some(backend) = &mut output.backend else {
        stats.playing = 0;
        stats.started = false;
        return;
    };
    let _zone = tracing::trace_span!("audio_update").entered();
    let listener = listeners
        .iter()
        .next()
        .or_else(|| cameras.iter().next())
        .map(|global| global.0);

    for (entity, player, global) in &players {
        let params = params(
            &player,
            global.map(GlobalTransform::translation),
            listener.as_ref(),
        );
        let restart = output
            .voices
            .get(&entity)
            .is_some_and(|voice| player.is_added() || voice.clip != player.clip.id());
        if restart && let Some(voice) = output.voices.remove(&entity) {
            backend.stop(voice.sound);
        }
        match output.voices.get_mut(&entity) {
            None => {
                // Ещё грузится — запустим в следующих кадрах
                let Some(data) = clips.get_loaded(&player.clip) else {
                    continue;
                };
                let mode = PlayMode {
                    looped: player.looped,
                    stream: player.stream,
                };
                match backend.play(player.clip.id(), data, mode, params) {
                    Ok(sound) => {
                        if player.paused {
                            backend.set_paused(sound, true);
                        }
                        output.voices.insert(
                            entity,
                            Voice {
                                sound,
                                clip: player.clip.id(),
                                params,
                                paused: player.paused,
                                finished: false,
                            },
                        );
                    }
                    Err(error) => {
                        tracing::warn!("Звук не запущен: {error}");
                        // Не повторять каждый кадр: считаем доигравшим
                        output.voices.insert(
                            entity,
                            Voice {
                                sound: crate::backend::SoundId(u64::MAX),
                                clip: player.clip.id(),
                                params,
                                paused: false,
                                finished: true,
                            },
                        );
                    }
                }
            }
            Some(voice) if voice.finished => {}
            Some(voice) => {
                if backend.state(voice.sound) == SoundState::Finished {
                    voice.finished = true;
                    backend.stop(voice.sound);
                    if player.despawn_on_finish {
                        commands.entity(entity).despawn();
                    }
                    continue;
                }
                if differs(params, voice.params) {
                    backend.set_params(voice.sound, params);
                    voice.params = params;
                }
                if player.paused != voice.paused {
                    backend.set_paused(voice.sound, player.paused);
                    voice.paused = player.paused;
                }
            }
        }
    }
    stats.started = true;
    stats.playing = output.voices.values().filter(|v| !v.finished).count();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn listener_at(position: Vec3) -> Mat4 {
        // Смотрит вдоль −Z, правая ось +X
        Mat4::from_translation(position)
    }

    #[test]
    fn spatial_falloff_and_pan() {
        let spatial = Spatial {
            min_distance: 1.0,
            max_distance: 40.0,
        };
        let listener = listener_at(Vec3::ZERO);
        let (near, pan_near) = spatialize(&listener, Vec3::new(0.0, 0.0, -0.5), spatial);
        assert!(
            (near - 1.0).abs() < 1e-6 && pan_near.abs() < 1e-6,
            "вплотную впереди"
        );
        let (gain_2, _) = spatialize(&listener, Vec3::new(0.0, 0.0, -2.0), spatial);
        let (gain_4, _) = spatialize(&listener, Vec3::new(0.0, 0.0, -4.0), spatial);
        assert!(
            (gain_2 - 0.5).abs() < 1e-6 && (gain_4 - 0.25).abs() < 1e-6,
            "вдвое дальше — вдвое тише"
        );
        let (_, right) = spatialize(&listener, Vec3::new(5.0, 0.0, 0.0), spatial);
        let (_, left) = spatialize(&listener, Vec3::new(-5.0, 0.0, 0.0), spatial);
        assert!((right - 1.0).abs() < 1e-6 && (left + 1.0).abs() < 1e-6);
        let (far, _) = spatialize(&listener, Vec3::new(0.0, 0.0, -45.0), spatial);
        assert!(far.abs() < 1e-6, "за max_distance — тишина");
    }

    #[test]
    fn listener_rotation_changes_pan() {
        // Слушатель повернулся на 180°: источник справа оказался слева
        let listener = Mat4::from_rotation_y(std::f32::consts::PI);
        let (_, pan) = spatialize(&listener, Vec3::new(5.0, 0.0, 0.0), Spatial::default());
        assert!((pan + 1.0).abs() < 1e-5, "{pan}");
    }
}
