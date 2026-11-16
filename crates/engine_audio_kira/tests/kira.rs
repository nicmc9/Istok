//! kira без звукового устройства (`MockBackend`): декодирование настоящих файлов, запуск,
//! параметры, остановка. Работает и в CI без звуковой карты.

use engine_assets::{AssetAppExt, AssetId, AssetPlugin, Assets, Bytes, Handle};
use engine_audio::{AudioBackend, AudioClip, AudioOutput, PlayMode, SoundParams, SoundState};
use engine_audio_kira::{KiraAudioPlugin, KiraBackend};
use engine_core::App;
use kira::backend::mock::MockBackend;

fn clip(name: &str) -> AudioClip {
    let path = format!("{}/../../assets/audio/{name}", env!("CARGO_MANIFEST_DIR"));
    let data: Bytes = std::fs::read(path).unwrap().into();
    AudioClip { data }
}

/// Настоящие id ресурсов: бэкенд кеширует декодированное по ним. Хэндлы держим, пока идёт тест.
fn ids(count: usize) -> (App, Vec<Handle<AudioClip>>) {
    let mut app = App::new();
    app.add_plugin(engine_net::NetPlugin)
        .add_plugin(AssetPlugin)
        .init_asset::<AudioClip>();
    let handles = (0..count)
        .map(|_| {
            let data: Bytes = Vec::new().into();
            app.world_mut()
                .resource_mut::<Assets<AudioClip>>()
                .add(AudioClip { data })
        })
        .collect();
    (app, handles)
}

#[test]
fn plugin_stays_off_without_cvar() {
    let mut app = App::new();
    app.add_plugin(KiraAudioPlugin);
    assert!(!app.world().contains_non_send::<AudioOutput>());
}

const PARAMS: SoundParams = SoundParams {
    volume: 0.5,
    pan: -0.3,
    speed: 1.2,
};

#[test]
fn plays_static_and_streamed_clips() {
    let (_app, handles) = ids(2);
    let mut backend = KiraBackend::<MockBackend>::new().unwrap();
    let bounce = clip("bounce.wav");
    let id: AssetId<AudioClip> = handles[0].id();
    let once = PlayMode {
        looped: false,
        stream: false,
    };
    let a = backend.play(id, &bounce, once, PARAMS).unwrap();
    // Второй раз — из кеша декодированных
    let b = backend.play(id, &bounce, once, PARAMS).unwrap();
    assert_ne!(a, b);
    assert_eq!(backend.state(a), SoundState::Playing);

    let music = PlayMode {
        looped: true,
        stream: true,
    };
    let m = backend
        .play(handles[1].id(), &clip("ambient_loop.wav"), music, PARAMS)
        .unwrap();
    backend.set_params(
        m,
        SoundParams {
            volume: 0.0,
            ..PARAMS
        },
    );
    backend.set_paused(m, true);

    backend.stop(a);
    assert_eq!(
        backend.state(a),
        SoundState::Finished,
        "остановленный забыт"
    );
    backend.forget_clip(id);
}

#[test]
fn broken_file_is_an_error() {
    let (_app, handles) = ids(1);
    let mut backend = KiraBackend::<MockBackend>::new().unwrap();
    let broken = AudioClip {
        data: b"RIFF\0\0\0\0WAVEgarbage".as_slice().into(),
    };
    let mode = PlayMode {
        looped: false,
        stream: false,
    };
    assert!(
        backend
            .play(handles[0].id(), &broken, mode, PARAMS)
            .is_err()
    );
}
