//! Логика фасада на поддельном бэкенде: что и когда уходит бэкенду.

use std::cell::RefCell;
use std::rc::Rc;

use engine_assets::{AssetId, AssetPlugin, Assets};
use engine_audio::{
    AUDIO_SFX_VOLUME, AudioBackend, AudioClip, AudioOutput, AudioPlayer, AudioPlugin, AudioStats,
    PlayMode, SoundId, SoundParams, SoundState, Spatial,
};
use engine_core::App;
use engine_math::Vec3;
use engine_scene::{Camera, ScenePlugin, Transform};

/// Что происходило с бэкендом. `Rc<RefCell>` — бэкенд не обязан быть `Send`.
#[derive(Default)]
struct Log {
    played: Vec<(PlayMode, SoundParams)>,
    params: Vec<(SoundId, SoundParams)>,
    paused: Vec<(SoundId, bool)>,
    stopped: Vec<SoundId>,
    /// Звуки, которые «доиграли».
    finished: Vec<SoundId>,
}

struct Fake(Rc<RefCell<Log>>);

impl AudioBackend for Fake {
    fn describe(&self) -> String {
        "fake".to_owned()
    }
    fn play(
        &mut self,
        _: AssetId<AudioClip>,
        _: &AudioClip,
        mode: PlayMode,
        params: SoundParams,
    ) -> Result<SoundId, String> {
        let mut log = self.0.borrow_mut();
        log.played.push((mode, params));
        Ok(SoundId(log.played.len() as u64 - 1))
    }
    fn set_params(&mut self, sound: SoundId, params: SoundParams) {
        self.0.borrow_mut().params.push((sound, params));
    }
    fn set_paused(&mut self, sound: SoundId, paused: bool) {
        self.0.borrow_mut().paused.push((sound, paused));
    }
    fn stop(&mut self, sound: SoundId) {
        self.0.borrow_mut().stopped.push(sound);
    }
    fn state(&self, sound: SoundId) -> SoundState {
        if self.0.borrow().finished.contains(&sound) {
            SoundState::Finished
        } else {
            SoundState::Playing
        }
    }
    fn forget_clip(&mut self, _: AssetId<AudioClip>) {}
}

fn app(device: bool) -> (App, Rc<RefCell<Log>>) {
    let log = Rc::new(RefCell::new(Log::default()));
    let backend_log = Rc::clone(&log);
    let mut app = App::new();
    app.add_plugin(engine_net::NetPlugin)
        .add_plugin(AssetPlugin)
        .add_plugin(ScenePlugin);
    app.world_mut()
        .insert_non_send(AudioOutput::new(Box::new(move || {
            if device {
                Ok(Box::new(Fake(Rc::clone(&backend_log))) as Box<dyn AudioBackend>)
            } else {
                Err("нет устройства".to_owned())
            }
        })));
    app.add_plugin(AudioPlugin);
    app.update();
    (app, log)
}

fn clip(app: &mut App) -> engine_assets::Handle<AudioClip> {
    let data: engine_assets::Bytes = b"RIFF\0\0\0\0WAVE".as_slice().into();
    app.world_mut()
        .resource_mut::<Assets<AudioClip>>()
        .add(AudioClip { data })
}

#[test]
fn plays_updates_pauses_and_stops() {
    let (mut app, log) = app(true);
    let clip = clip(&mut app);
    let entity = app
        .world_mut()
        .spawn(AudioPlayer::new(clip).with_volume(0.5))
        .id();
    app.update(); // клип добавлен в этом кадре — Loaded, затем запуск
    app.update();
    assert_eq!(log.borrow().played.len(), 1);
    assert!((log.borrow().played[0].1.volume - 0.5 * AUDIO_SFX_VOLUME.get()).abs() < 1e-6);
    assert_eq!(app.world().resource::<AudioStats>().playing, 1);

    // Без изменений бэкенду ничего не уходит
    app.update();
    assert_eq!(log.borrow().params.len(), 0, "без изменений — без вызовов");

    app.world_mut()
        .get_mut::<AudioPlayer>(entity)
        .unwrap()
        .speed = 2.0;
    app.world_mut()
        .get_mut::<AudioPlayer>(entity)
        .unwrap()
        .paused = true;
    app.update();
    assert_eq!(log.borrow().params.last().unwrap().1.speed, 2.0);
    assert_eq!(log.borrow().paused, [(SoundId(0), true)]);

    app.world_mut().despawn(entity);
    app.update();
    assert_eq!(log.borrow().stopped, [SoundId(0)]);
    assert_eq!(app.world().resource::<AudioStats>().playing, 0);
}

#[test]
fn finished_once_sound_despawns_entity() {
    let (mut app, log) = app(true);
    let clip = clip(&mut app);
    let entity = app.world_mut().spawn(AudioPlayer::once(clip.clone())).id();
    let kept = app.world_mut().spawn(AudioPlayer::new(clip)).id();
    app.update();
    app.update();
    log.borrow_mut().finished.extend([SoundId(0), SoundId(1)]);
    app.update();
    app.update();
    assert!(
        app.world().get_entity(entity).is_err(),
        "разовый звук удалил сущность"
    );
    assert!(app.world().get_entity(kept).is_ok(), "обычный — остался");
    app.update();
    assert_eq!(
        log.borrow().played.len(),
        2,
        "доигравший не запускается снова"
    );
}

#[test]
fn spatial_sound_follows_listener() {
    let (mut app, log) = app(true);
    let clip = clip(&mut app);
    app.world_mut()
        .spawn((Camera::default(), Transform::default()));
    let mut player = AudioPlayer::new(clip).looped();
    player.spatial = Some(Spatial {
        min_distance: 1.0,
        max_distance: 100.0,
    });
    let source = app
        .world_mut()
        .spawn((
            player,
            Transform::from_translation(Vec3::new(4.0, 0.0, 0.0)),
        ))
        .id();
    app.update();
    app.update();
    let first = log.borrow().played[0].1;
    assert!(
        (first.volume - 0.25 * AUDIO_SFX_VOLUME.get()).abs() < 1e-4,
        "{first:?}"
    );
    assert!(first.pan > 0.99, "справа: {first:?}");

    // Источник перешёл налево и ближе
    app.world_mut()
        .get_mut::<Transform>(source)
        .unwrap()
        .translation = Vec3::new(-2.0, 0.0, 0.0);
    app.update();
    let last = log.borrow().params.last().unwrap().1;
    assert!(last.pan < -0.99 && last.volume > first.volume, "{last:?}");
}

#[test]
fn without_device_everything_still_works() {
    let (mut app, _) = app(false);
    let clip = clip(&mut app);
    app.world_mut().spawn(AudioPlayer::music(clip));
    for _ in 0..3 {
        app.update();
    }
    let stats = *app.world().resource::<AudioStats>();
    assert!(!stats.started && stats.playing == 0);
}
