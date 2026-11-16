//! Возможности GPU.
//!
//! Контекст GL — только GLES 3.2: compute-шейдеры есть, отдельного пути без них нет.
//! Возможности адаптера (compute, HDR, метки времени) читаются один раз и не
//! подменяются. Compute-проход спрашивает [`GpuCaps::compute`].

/// Что показывает статистика. `Unknown` — GPU ещё не создан.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuTier {
    #[default]
    Unknown,
    Compute,
    NoCompute,
}

impl GpuTier {
    pub(crate) fn from_code(code: u8) -> Self {
        match code {
            1 => Self::Compute,
            2 => Self::NoCompute,
            _ => Self::Unknown,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Unknown => "ещё нет GPU",
            Self::Compute => "compute есть",
            Self::NoCompute => "compute нет",
        }
    }
}

/// Биты возможностей. Отдельные `bool` в структуре clippy не пропускает.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Features(u8);

impl Features {
    const COMPUTE: u8 = 1;
    const VERTEX_STORAGE: u8 = 1 << 1;
    const INDIRECT: u8 = 1 << 2;
    const HDR: u8 = 1 << 3;
    const TIMESTAMPS: u8 = 1 << 4;
    const SURFACE_VIEW: u8 = 1 << 5;

    fn has(self, bit: u8) -> bool {
        self.0 & bit != 0
    }

    fn set(mut self, bit: u8, on: bool) -> Self {
        if on {
            self.0 |= bit;
        } else {
            self.0 &= !bit;
        }
        self
    }
}

/// Снимок возможностей адаптера. Собирается один раз в [`GpuCaps::from_probe`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Probe {
    features: Features,
}

impl Probe {
    pub(crate) fn from_adapter(adapter: &wgpu::Adapter) -> Self {
        let flags = adapter.get_downlevel_capabilities().flags;
        let features = Features::default()
            .set(
                Features::COMPUTE,
                flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS),
            )
            .set(
                Features::VERTEX_STORAGE,
                flags.contains(wgpu::DownlevelFlags::VERTEX_STORAGE),
            )
            .set(
                Features::INDIRECT,
                flags.contains(wgpu::DownlevelFlags::INDIRECT_EXECUTION),
            )
            .set(Features::HDR, crate::post::hdr_supported(adapter))
            .set(
                Features::TIMESTAMPS,
                adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY),
            )
            .set(
                Features::SURFACE_VIEW,
                flags.contains(wgpu::DownlevelFlags::SURFACE_VIEW_FORMATS),
            );
        Self { features }
    }
}

/// Возможности этого процесса. HDR и метки времени не зависят от compute:
/// у них свои проверки адаптера.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct GpuCaps {
    features: Features,
}

impl GpuCaps {
    pub(crate) fn from_probe(probe: Probe) -> Self {
        Self {
            features: probe.features,
        }
    }

    pub(crate) fn tier_code(self) -> u8 {
        if self.compute() { 1 } else { 2 }
    }

    pub(crate) fn compute(self) -> bool {
        self.features.has(Features::COMPUTE)
    }

    pub(crate) fn vertex_storage(self) -> bool {
        self.features.has(Features::VERTEX_STORAGE)
    }

    pub(crate) fn indirect(self) -> bool {
        self.features.has(Features::INDIRECT)
    }

    pub(crate) fn hdr(self) -> bool {
        self.features.has(Features::HDR)
    }

    pub(crate) fn timestamps(self) -> bool {
        self.features.has(Features::TIMESTAMPS)
    }

    pub(crate) fn surface_view_formats(self) -> bool {
        self.features.has(Features::SURFACE_VIEW)
    }

    pub(crate) fn report(self) -> String {
        let yes = |on| if on { "есть" } else { "нет" };
        format!(
            "Рендер: compute {}, vertex storage {}, indirect {}",
            yes(self.compute()),
            yes(self.vertex_storage()),
            yes(self.indirect())
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{Features, GpuCaps, GpuTier, Probe};

    fn probe(compute: bool, rest: bool) -> Probe {
        Probe {
            features: Features::default()
                .set(Features::COMPUTE, compute)
                .set(Features::VERTEX_STORAGE, rest)
                .set(Features::INDIRECT, rest)
                .set(Features::HDR, rest)
                .set(Features::TIMESTAMPS, rest)
                .set(Features::SURFACE_VIEW, rest),
        }
    }

    #[test]
    fn probe_keeps_compute_when_the_gpu_has_it() {
        let caps = GpuCaps::from_probe(probe(true, true));
        assert!(caps.compute() && caps.vertex_storage() && caps.indirect() && caps.hdr());
        assert_eq!(caps.tier_code(), 1);
        assert_eq!(GpuTier::from_code(caps.tier_code()), GpuTier::Compute);
    }

    #[test]
    fn missing_compute_does_not_clear_other_features() {
        let caps = GpuCaps::from_probe(probe(false, true));
        assert!(!caps.compute());
        assert!(caps.vertex_storage() && caps.indirect() && caps.hdr());
        assert_eq!(GpuTier::from_code(caps.tier_code()), GpuTier::NoCompute);
    }
}
