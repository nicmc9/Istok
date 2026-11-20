//! Кадровая арена и слоты хэндлов.
//!
//! Арена живёт один кадр: системы берут из неё временные значения, в начале следующего
//! кадра она сбрасывается. Сброс оставляет самый большой кусок, его хватает на
//! следующий кадр того же размера. Слоты — индекс и поколение:
//! освобождённый индекс можно выдать снова, старый ключ его уже не откроет.

use std::fmt;

use bumpalo::Bump;

/// Память на один кадр. Только главный поток: `Bump` нельзя делить между потоками.
///
/// Система берёт её как `NonSend<FrameArena>`. Ссылка из [`Self::alloc`] живёт,
/// пока система держит арену, и пропадает на следующем кадре.
pub struct FrameArena {
    bump: Bump,
}

impl FrameArena {
    pub fn new() -> Self {
        Self::default()
    }

    /// Кладёт значение в арену и возвращает ссылку на него.
    /// `Drop` не вызывается: для данных с кучей это утечка.
    pub fn alloc<T>(&self, value: T) -> &mut T {
        self.bump.alloc(value)
    }

    /// Копия среза. Элементы должны быть `Copy`: арена не вызывает деструкторы до сброса.
    pub fn alloc_slice_copy<T: Copy>(&self, values: &[T]) -> &mut [T] {
        self.bump.alloc_slice_copy(values)
    }

    /// Копия строки.
    pub fn alloc_str(&self, text: &str) -> &str {
        self.bump.alloc_str(text)
    }

    /// Начало кадра. Самый большой кусок остаётся, остальные возвращаются системе.
    pub fn reset(&mut self) {
        self.bump.reset();
    }

    /// Сколько байт сейчас держат куски арены.
    pub fn reserved_bytes(&self) -> usize {
        self.bump.allocated_bytes()
    }
}

impl Default for FrameArena {
    fn default() -> Self {
        Self { bump: Bump::new() }
    }
}

impl fmt::Debug for FrameArena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrameArena")
            .field("reserved_bytes", &self.reserved_bytes())
            .finish()
    }
}

/// Индекс слота и поколение. После [`Slots::free`] тот же индекс может достаться
/// новому жильцу, но с другим поколением.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SlotKey {
    pub index: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    generation: u32,
    occupied: bool,
}

/// Индексы с повторным использованием. Значение хранит тот, кто выдал ключ:
/// здесь только занятость и поколение.
#[derive(Clone, Debug, Default)]
pub struct Slots {
    items: Vec<Slot>,
    free: Vec<u32>,
    live: usize,
}

fn slot_index(index: u32) -> usize {
    usize::try_from(index).expect("индекс слота")
}

impl Slots {
    /// Свободный индекс. Поколение отличает новый ключ от уже освобождённого.
    pub fn allocate(&mut self) -> SlotKey {
        if let Some(index) = self.free.pop() {
            let slot = &mut self.items[slot_index(index)];
            slot.occupied = true;
            self.live += 1;
            return SlotKey {
                index,
                generation: slot.generation,
            };
        }
        let index = u32::try_from(self.items.len()).expect("слишком много слотов");
        self.items.push(Slot {
            generation: 0,
            occupied: true,
        });
        self.live += 1;
        SlotKey {
            index,
            generation: 0,
        }
    }

    /// Освобождает слот. `false` — ключ уже старый или слот и так свободен.
    pub fn free(&mut self, key: SlotKey) -> bool {
        let Some(slot) = self.items.get_mut(slot_index(key.index)) else {
            return false;
        };
        if !slot.occupied || slot.generation != key.generation {
            return false;
        }
        slot.occupied = false;
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(key.index);
        self.live -= 1;
        true
    }

    /// Ключ указывает на занятый слот своего поколения.
    pub fn contains(&self, key: SlotKey) -> bool {
        self.items
            .get(slot_index(key.index))
            .is_some_and(|slot| slot.occupied && slot.generation == key.generation)
    }

    pub fn len(&self) -> usize {
        self.live
    }

    pub fn is_empty(&self) -> bool {
        self.live == 0
    }
}

#[cfg(test)]
mod tests {
    use super::{FrameArena, Slots};

    #[test]
    fn arena_reset_does_not_grow_the_reserve() {
        let mut arena = FrameArena::new();
        for value in 0..1_000 {
            std::hint::black_box(arena.alloc(value));
        }
        let peak = arena.reserved_bytes();
        assert!(peak > 0);
        arena.reset();
        assert!(arena.reserved_bytes() > 0);
        for value in 0..1_000 {
            std::hint::black_box(arena.alloc(value));
        }
        assert!(arena.reserved_bytes() <= peak);
    }

    #[test]
    fn freed_slot_key_does_not_open_the_next_tenant() {
        let mut slots = Slots::default();
        let first = slots.allocate();
        assert!(slots.free(first));
        assert!(!slots.contains(first));
        assert!(!slots.free(first));
        let second = slots.allocate();
        assert_eq!(second.index, first.index);
        assert_ne!(second.generation, first.generation);
        assert!(slots.contains(second));
        assert!(!slots.contains(first));
        assert_eq!(slots.len(), 1);
    }
}
