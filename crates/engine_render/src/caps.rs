//! Какой путь рендера доступен этому GPU.
//!
//! GLES3 не умеет compute-шейдеры и часто не умеет storage-буферы в вершинном
//! шейдере. Запасной путь — тот, которым движок рисует и без compute: фрагментные
//! шейдеры, скиннинг через uniform, мипы на CPU. Будущий compute-проход сначала
//! спрашивает [`GpuCaps::compute`].

use engine_core::{CVar, CVarFlags, CVarValue};

/// Что спросили при старте. `gles` включает запасной путь даже на Metal и Vulkan,
/// чтобы проверить его без Android-устройства.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TierRequest {
    #[default]
    Auto,
    Full,
    Gles,
}

impl CVarValue for TierRequest {
    const TYPE_NAME: &'static str = "auto | full | gles";
    fn parse(text: &str) -> Result<Self, String> {
        match text.trim() {
            "auto" => Ok(Self::Auto),
            "full" => Ok(Self::Full),
            "gles" => Ok(Self::Gles),
            other => Err(format!("ожидается auto, full или gles, получено «{other}»")),
        }
    }
    fn format(&self) -> String {
        match self {
            Self::Auto => "auto",
            Self::Full => "full",
            Self::Gles => "gles",
        }
        .to_owned()
    }
}

/// Путь рендера. Читается при создании GPU.
pub static RENDER_TIER: CVar<TierRequest> = CVar::new(
    "render.tier",
    TierRequest::Auto,
    "Путь рендера: auto (по GPU), full (compute, если GPU умеет), gles (запасной, без compute)",
)
.with_flags(CVarFlags::STARTUP);

/// Что показывает статистика. `Unknown` — GPU ещё не создан.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuTier {
    #[default]
    Unknown,
    Full,
    Gles,
}

impl GpuTier {
    pub(crate) fn from_code(code: u8) -> Self {
        match code {
            1 => Self::Full,
            2 => Self::Gles,
            _ => Self::Unknown,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Unknown => "ещё нет GPU",
            Self::Full => "полный, compute есть",
            Self::Gles => "запасной GLES3, без compute",
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

/// Снимок возможностей адаптера. Собирается один раз, решение считает [`GpuCaps::choose`].
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

    fn has(self, bit: u8) -> bool {
        self.features.has(bit)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tier {
    Full,
    Gles,
}

impl Tier {
    pub(crate) fn code(self) -> u8 {
        match self {
            Self::Full => 1,
            Self::Gles => 2,
        }
    }
}

/// Откуда взялся тир: устройство само, принудительный `gles` или урезанный `full`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum PathNote {
    #[default]
    Device,
    Forced,
    Clamped,
}

/// Решение на этот процесс. HDR и метки времени не зависят от compute:
/// у них свои проверки адаптера.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GpuCaps {
    pub tier: Tier,
    features: Features,
    note: PathNote,
}

impl Default for GpuCaps {
    fn default() -> Self {
        Self::choose(Probe::default(), TierRequest::Auto)
    }
}

impl GpuCaps {
    pub(crate) fn choose(probe: Probe, request: TierRequest) -> Self {
        let compute = probe.has(Features::COMPUTE);
        let (tier, note) = match request {
            TierRequest::Gles => (
                Tier::Gles,
                if compute {
                    PathNote::Forced
                } else {
                    PathNote::Device
                },
            ),
            TierRequest::Auto => {
                if compute {
                    (Tier::Full, PathNote::Device)
                } else {
                    (Tier::Gles, PathNote::Device)
                }
            }
            TierRequest::Full => {
                if compute {
                    (Tier::Full, PathNote::Device)
                } else {
                    (Tier::Gles, PathNote::Clamped)
                }
            }
        };
        let limited = tier == Tier::Gles;
        let features = probe
            .features
            .set(Features::COMPUTE, compute && !limited)
            .set(
                Features::VERTEX_STORAGE,
                probe.has(Features::VERTEX_STORAGE) && !limited,
            )
            .set(
                Features::INDIRECT,
                probe.has(Features::INDIRECT) && !limited,
            );
        Self {
            tier,
            features,
            note,
        }
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
        let why = match self.note {
            PathNote::Forced => "принудительно, render.tier=gles",
            PathNote::Clamped => "render.tier=full недоступен на этом GPU",
            PathNote::Device => "по возможностям GPU",
        };
        let yes = |on| if on { "есть" } else { "нет" };
        match self.tier {
            Tier::Full => format!(
                "Рендер: полный путь ({why}), compute {}, vertex storage {}, indirect {}",
                yes(self.compute()),
                yes(self.vertex_storage()),
                yes(self.indirect())
            ),
            Tier::Gles => format!(
                "Рендер: запасной путь GLES3 ({why}), compute {}, vertex storage {}, indirect {}; скиннинг через uniform, мипы на CPU",
                yes(self.compute()),
                yes(self.vertex_storage()),
                yes(self.indirect())
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use engine_core::CVarValue;

    use super::{Features, GpuCaps, Probe, Tier, TierRequest};

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
    fn auto_keeps_compute_on_a_full_gpu() {
        let caps = GpuCaps::choose(probe(true, true), TierRequest::Auto);
        assert_eq!(caps.tier, Tier::Full);
        assert!(caps.compute() && caps.vertex_storage() && caps.indirect() && caps.hdr());
    }

    #[test]
    fn auto_falls_back_without_compute() {
        let caps = GpuCaps::choose(probe(false, false), TierRequest::Auto);
        assert_eq!(caps.tier, Tier::Gles);
        assert!(!caps.compute());
        assert!(!caps.vertex_storage());
        assert!(!caps.indirect());
    }

    #[test]
    fn gles_request_disables_compute_even_on_desktop() {
        let caps = GpuCaps::choose(probe(true, true), TierRequest::Gles);
        assert_eq!(caps.tier, Tier::Gles);
        assert!(!caps.compute());
        assert!(!caps.vertex_storage());
        assert!(!caps.indirect());
        // HDR — отдельная проверка формата, запасной путь её не выключает.
        assert!(caps.hdr());
        assert!(caps.report().contains("принудительно"));
    }

    #[test]
    fn full_request_is_clamped_when_the_gpu_has_no_compute() {
        let caps = GpuCaps::choose(probe(false, false), TierRequest::Full);
        assert_eq!(caps.tier, Tier::Gles);
        assert!(caps.report().contains("недоступен"));
    }

    #[test]
    fn tier_names_round_trip() {
        assert_eq!(TierRequest::parse("gles").unwrap().format(), "gles");
        assert!(TierRequest::parse("vulkan").is_err());
    }
}
