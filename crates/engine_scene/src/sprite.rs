//! Покадровая анимация (R2-06): клип задаёт кадры атласа, проигрыватель — текущий.
//!
//! Рисование спрайта — плагин 2D (R2-13). Здесь только номер кадра и прямоугольник UV.

use engine_assets::{Asset, Assets, Handle};
use engine_core::FixedTime;
use engine_ecs::prelude::*;

use crate::animation::advance;

/// Прямоугольник в текстуре: лево, верх, право, низ. `(0, 0)` — левый верхний угол,
/// `(1, 1)` — правый нижний (как пиксели картинки и координаты текстуры в wgpu).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpriteFrame {
    pub uv: [f32; 4],
}

/// Последовательность кадров. Ресурс: создаётся кодом, `Assets::add`.
#[derive(Debug, Clone, TypePath)]
pub struct SpriteClip {
    pub frames: Vec<SpriteFrame>,
    /// Если длина совпадает с `frames` — у каждого кадра своя длительность.
    /// Иначе все кадры длятся `frame_duration`.
    pub durations: Vec<f32>,
    pub frame_duration: f32,
}

impl SpriteClip {
    pub fn new(frames: Vec<SpriteFrame>, frame_duration: f32) -> Self {
        Self {
            frames,
            durations: Vec::new(),
            frame_duration,
        }
    }

    /// Своя длительность каждого кадра. Длина должна совпадать с числом кадров.
    #[must_use]
    pub fn with_durations(mut self, durations: Vec<f32>) -> Self {
        debug_assert_eq!(
            durations.len(),
            self.frames.len(),
            "длительностей должно быть столько же, сколько кадров"
        );
        self.durations = durations;
        self
    }

    /// Сетка атласа: слева направо, сверху вниз.
    pub fn grid(columns: u32, rows: u32, frame_duration: f32) -> Self {
        assert!(columns > 0 && rows > 0, "в сетке атласа нет ячеек");
        let du = 1.0 / unit(columns);
        let dv = 1.0 / unit(rows);
        let mut frames = Vec::new();
        for row in 0..rows {
            for column in 0..columns {
                let x = unit(column) * du;
                let y = unit(row) * dv;
                frames.push(SpriteFrame {
                    uv: [x, y, x + du, y + dv],
                });
            }
        }
        Self::new(frames, frame_duration)
    }

    pub fn duration(&self) -> f32 {
        if self.frames.is_empty() {
            return 0.0;
        }
        if self.durations.len() == self.frames.len() {
            self.durations.iter().sum()
        } else {
            self.frame_duration * frames_as_f32(self.frames.len())
        }
    }

    /// Кадр в момент `time`. За последним кадром остаётся последний.
    pub fn frame_at(&self, time: f32) -> Option<(usize, [f32; 4])> {
        let last = self.frames.len().checked_sub(1)?;
        let own = self.durations.len() == self.frames.len();
        let mut cursor = 0.0;
        for index in 0..=last {
            let duration = if own {
                self.durations[index]
            } else {
                self.frame_duration
            };
            cursor += duration;
            if time < cursor || index == last {
                return Some((index, self.frames[index].uv));
            }
        }
        None
    }
}

/// u16 → f32 без потери. Сетка больше 65535 ячеек по стороне не нужна.
fn unit(value: u32) -> f32 {
    f32::from(u16::try_from(value).unwrap_or(u16::MAX))
}

fn frames_as_f32(count: usize) -> f32 {
    unit(u32::try_from(count).unwrap_or(u32::MAX))
}

impl Asset for SpriteClip {
    fn memory_size(&self) -> u64 {
        bytes_of(&self.frames) + bytes_of(&self.durations)
    }
}

fn bytes_of<T>(items: &[T]) -> u64 {
    u64::try_from(std::mem::size_of_val(items)).unwrap_or(u64::MAX)
}

/// Проигрыватель кадров на сущности. Номер и UV обновляет система на фиксированном шаге.
/// В сцену не пишется: это состояние проигрывания, как у [`crate::AnimationPlayer`].
#[derive(Component, Debug, Clone)]
pub struct SpritePlayer {
    clip: Handle<SpriteClip>,
    time: f32,
    frame: usize,
    uv: [f32; 4],
    /// Кадр уже выбран системой. До этого UV — нули, спрайт держит свой прямоугольник.
    sampled: bool,
    pub speed: f32,
    pub paused: bool,
    pub looping: bool,
}

impl SpritePlayer {
    pub fn new(clip: Handle<SpriteClip>) -> Self {
        Self {
            clip,
            time: 0.0,
            frame: 0,
            uv: [0.0; 4],
            sampled: false,
            speed: 1.0,
            paused: false,
            looping: true,
        }
    }

    pub fn play(&mut self, clip: Handle<SpriteClip>) -> &mut Self {
        self.clip = clip;
        self.time = 0.0;
        self.sampled = false;
        self
    }

    /// UV текущего кадра, когда проигрыватель уже успел его выбрать.
    pub fn sampled_uv(&self) -> Option<[f32; 4]> {
        self.sampled.then_some(self.uv)
    }

    pub fn frame(&self) -> usize {
        self.frame
    }

    pub fn uv(&self) -> [f32; 4] {
        self.uv
    }

    pub fn time(&self) -> f32 {
        self.time
    }
}

pub(crate) fn animate_sprites(
    fixed: Res<FixedTime>,
    clips: Res<Assets<SpriteClip>>,
    mut players: Query<&mut SpritePlayer>,
) {
    let _zone = tracing::trace_span!("animate_sprites").entered();
    for mut player in &mut players {
        let Some(clip) = clips.get_loaded(&player.clip) else {
            continue;
        };
        if !player.paused {
            let dt = fixed.step_secs() * player.speed;
            player.time = advance(player.time, dt, clip.duration(), player.looping);
        }
        if let Some((frame, uv)) = clip.frame_at(player.time) {
            player.frame = frame;
            player.uv = uv;
            player.sampled = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_goes_left_to_right_then_down() {
        let clip = SpriteClip::grid(2, 2, 0.25);
        assert!((clip.duration() - 1.0).abs() < 1e-4);
        assert_eq!(clip.frame_at(0.0).unwrap().0, 0);
        assert_eq!(clip.frame_at(0.0).unwrap().1, [0.0, 0.0, 0.5, 0.5]);
        assert_eq!(clip.frame_at(0.25).unwrap().0, 1);
        assert_eq!(clip.frame_at(0.5).unwrap().1, [0.0, 0.5, 0.5, 1.0]);
        assert_eq!(clip.frame_at(0.99).unwrap().0, 3);
        assert_eq!(
            clip.frame_at(5.0).unwrap().0,
            3,
            "после конца — последний кадр"
        );
    }

    #[test]
    fn per_frame_durations() {
        let clip = SpriteClip::grid(1, 2, 1.0).with_durations(vec![0.1, 0.9]);
        assert!((clip.duration() - 1.0).abs() < 1e-4);
        assert_eq!(clip.frame_at(0.05).unwrap().0, 0);
        assert_eq!(clip.frame_at(0.1).unwrap().0, 1);
    }

    #[test]
    fn player_advances_on_fixed_step() {
        use crate::ScenePlugin;
        use engine_assets::AssetPlugin;
        use engine_core::{App, Duration, FixedTime, Instant};

        let mut app = App::new();
        app.add_plugin(engine_net::NetPlugin)
            .add_plugin(AssetPlugin)
            .add_plugin(ScenePlugin);
        let clip = app
            .world_mut()
            .resource_mut::<Assets<SpriteClip>>()
            .add(SpriteClip::grid(2, 2, 0.25));
        let entity = app.world_mut().spawn(SpritePlayer::new(clip)).id();
        app.world_mut()
            .insert_resource(FixedTime::new(Duration::from_secs_f64(0.5), 4));
        let start = Instant::now();
        app.update_at(start);
        app.update_at(start + Duration::from_secs_f64(0.5));
        let player = app.world().get::<SpritePlayer>(entity).unwrap();
        assert_eq!(player.frame(), 2);
        assert!((player.uv()[1] - 0.5).abs() < 1e-4);
    }
}
