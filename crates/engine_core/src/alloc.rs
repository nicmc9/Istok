//! Глобальный аллокатор. По умолчанию — системный. Feature `mimalloc` подменяет его
//! на native; на Web фича ничего не делает, там аллокатор браузера.

#[cfg(all(feature = "mimalloc", not(target_arch = "wasm32")))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
