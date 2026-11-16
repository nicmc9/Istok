use engine_core::tasks::BoxedFuture;

use crate::{Asset, AssetError, AssetLoader, Bytes, LoadContext};
use engine_ecs::prelude::{TypePath, bevy_reflect};

/// Исходный текст шейдера WGSL. Компиляция — в рендере.
#[derive(Debug, Clone, TypePath)]
pub struct Shader {
    pub source: String,
}

impl Asset for Shader {
    fn memory_size(&self) -> u64 {
        u64::try_from(self.source.len()).unwrap_or(u64::MAX)
    }
}

#[derive(Debug)]
pub struct ShaderLoader;

impl AssetLoader for ShaderLoader {
    type Asset = Shader;

    fn extensions(&self) -> &[&'static str] {
        &["wgsl"]
    }

    fn load<'a>(
        &'a self,
        bytes: Bytes,
        ctx: &'a mut LoadContext,
    ) -> BoxedFuture<'a, Result<Shader, AssetError>> {
        Box::pin(async move {
            let source = std::str::from_utf8(&bytes)
                .map_err(|e| AssetError::format(ctx.path(), e))?
                .to_owned();
            Ok(Shader { source })
        })
    }
}
