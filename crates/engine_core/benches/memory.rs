//! Кадровая арена против нового `Vec` на каждый проход и оборот слотов хэндлов.

use criterion::{Criterion, criterion_group, criterion_main};
use engine_core::{FrameArena, Slots};

fn frame_arena(c: &mut Criterion) {
    let mut arena = FrameArena::new();
    c.bench_function("арена: 10k чисел и сброс кадра", |b| {
        b.iter(|| {
            for value in 0..10_000 {
                std::hint::black_box(arena.alloc(value));
            }
            arena.reset();
        });
    });
}

fn fresh_vec(c: &mut Criterion) {
    c.bench_function("Vec: 10k чисел новым вектором", |b| {
        b.iter(|| {
            let mut values = Vec::new();
            for value in 0..10_000 {
                values.push(value);
            }
            std::hint::black_box(values.len());
        });
    });
}

fn slot_churn(c: &mut Criterion) {
    let mut slots = Slots::default();
    c.bench_function(
        "слоты: выдать и вернуть 10k ключей",
        |b| {
            b.iter(|| {
                let keys: Vec<_> = (0..10_000).map(|_| slots.allocate()).collect();
                for key in keys {
                    slots.free(key);
                }
            });
        },
    );
}

criterion_group!(benches, frame_arena, fresh_vec, slot_churn);
criterion_main!(benches);
