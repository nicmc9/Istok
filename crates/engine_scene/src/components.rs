use engine_assets::Handle;
use engine_assets::formats::{Material, Mesh};
use engine_ecs::prelude::*;
use engine_math::{Mat4, Quat, Vec3, camera};

/// Положение, поворот и масштаб относительно родителя (или мира, если родителя нет).
#[derive(Component, Debug, Clone, Copy, PartialEq, Reflect)]
#[reflect(Component, Default)]
// [Rust] `#[require(...)]` — при добавлении Transform ECS сам добавит GlobalTransform и
// Visibility со значениями по умолчанию. Не нужно помнить о «парных» компонентах.
#[require(GlobalTransform, Visibility)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };

    pub fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            ..Self::IDENTITY
        }
    }

    pub fn from_xyz(x: f32, y: f32, z: f32) -> Self {
        Self::from_translation(Vec3::new(x, y, z))
    }

    /// Повернуть так, чтобы «вперёд» (−Z) смотрело на `target`.
    #[must_use]
    pub fn looking_at(mut self, target: Vec3, up: Vec3) -> Self {
        // Матрица вида переводит мир в камеру; поворот объекта — обратный к ней
        let view = camera::look_at(self.translation, target, up);
        self.rotation = Quat::from_mat4(&view.inverse());
        self
    }

    #[must_use]
    pub fn with_scale(mut self, scale: Vec3) -> Self {
        self.scale = scale;
        self
    }

    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    /// Направление «вперёд» (−Z), как у камеры и направленного света.
    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Итоговая матрица в мировых координатах. Считается системой — не меняйте вручную.
#[derive(Component, Debug, Clone, Copy, PartialEq, Default)]
pub struct GlobalTransform(pub Mat4);

impl GlobalTransform {
    pub fn translation(&self) -> Vec3 {
        self.0.w_axis.truncate()
    }
    pub fn forward(&self) -> Vec3 {
        (-self.0.z_axis.truncate()).normalize_or_zero()
    }
}

/// Видимость. Скрытый объект не рисуется вместе с потомками.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[reflect(Component, Default)]
pub struct Visibility(pub bool);

impl Default for Visibility {
    fn default() -> Self {
        Self(true)
    }
}

/// Меш с материалами — по одному на примитив (часть меша). `None` — материал по умолчанию.
#[derive(Component, Debug, Clone, Reflect)]
#[reflect(Component)]
#[require(Transform)]
pub struct Mesh3d {
    pub mesh: Handle<Mesh>,
    pub materials: Vec<Option<Handle<Material>>>,
}

/// По какой стороне экрана камера держит заданный обзор (угол у 3D-камеры, размер у 2D).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Reflect)]
#[reflect(Default)]
pub enum FitAxis {
    /// По вертикали. В портрете по бокам видно мало: на телефоне 9:19.5 горизонтальный угол
    /// вдвое меньше вертикального.
    Vertical,
    /// По горизонтали.
    Horizontal,
    /// По меньшей стороне: в ландшафте как `Vertical`, в портрете как `Horizontal`.
    /// Что видно в квадрате посередине, видно при любой ориентации.
    #[default]
    Shorter,
}

impl FitAxis {
    /// Обзор задан по ширине при соотношении сторон `aspect` (ширина / высота).
    fn by_width(self, aspect: f32) -> bool {
        match self {
            Self::Vertical => false,
            Self::Horizontal => true,
            Self::Shorter => aspect < 1.0,
        }
    }

    /// Вертикальный угол, при котором `fov` приходится на свою сторону.
    pub fn fov_y(self, fov: f32, aspect: f32) -> f32 {
        if self.by_width(aspect) {
            2.0 * ((fov * 0.5).tan() / aspect.max(1e-4)).atan()
        } else {
            fov
        }
    }

    /// Видимая высота, при которой `size` приходится на свою сторону.
    pub fn height(self, size: f32, aspect: f32) -> f32 {
        if self.by_width(aspect) {
            size / aspect.max(1e-4)
        } else {
            size
        }
    }
}

/// Перспективная камера. Смотрит вдоль −Z своего `Transform`.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(Component, Default)]
#[require(Transform)]
pub struct Camera {
    /// Угол обзора, радианы. По какой стороне — [`Self::fit`].
    pub fov: f32,
    /// По умолчанию по меньшей стороне: в портрете на телефоне сцена не уходит за края.
    #[reflect(default)]
    pub fit: FitAxis,
    pub near: f32,
    pub far: f32,
    /// Если камер несколько, рисует камера с меньшим номером (несколько вьюпортов — Б-09).
    pub order: i32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            fov: 50f32.to_radians(),
            fit: FitAxis::default(),
            near: 0.05,
            far: 2000.0,
            order: 0,
        }
    }
}

impl Camera {
    /// Вертикальный угол при соотношении сторон `aspect` (ширина / высота).
    pub fn fov_y(&self, aspect: f32) -> f32 {
        self.fit.fov_y(self.fov, aspect)
    }

    pub fn projection(&self, aspect: f32) -> Mat4 {
        camera::perspective(self.fov_y(aspect), aspect.max(1e-4), self.near, self.far)
    }
}

/// Направленный свет (солнце). Светит вдоль −Z своего `Transform`.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(Component, Default)]
#[require(Transform)]
pub struct DirectionalLight {
    /// Линейный цвет.
    pub color: [f32; 3],
    /// Освещённость (условные единицы: 1 — «пасмурный день», 3–5 — яркое солнце).
    pub intensity: f32,
    /// Отбрасывать тени (каскадные карты теней; общие настройки — CVar `render.shadow*`).
    pub shadows: bool,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        Self {
            color: [1.0, 0.98, 0.95],
            intensity: 3.0,
            shadows: true,
        }
    }
}

/// Точечный свет.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(Component, Default)]
#[require(Transform)]
pub struct PointLight {
    pub color: [f32; 3],
    pub intensity: f32,
    /// Дальше этого расстояния свет не действует.
    pub range: f32,
    /// Отбрасывать тени. Дорого: 6 проходов (граней куба) и 6 из 16 слоёв карты теней
    /// локальных источников — поэтому по умолчанию выключено.
    pub shadows: bool,
}

impl Default for PointLight {
    fn default() -> Self {
        Self {
            color: [1.0; 3],
            intensity: 10.0,
            range: 20.0,
            shadows: false,
        }
    }
}

/// Прожектор: конус света вдоль −Z своего `Transform` (как камера).
/// Внутри `inner_angle` — полная яркость, к `outer_angle` плавно гаснет.
#[derive(Component, Debug, Clone, Copy, Reflect)]
#[reflect(Component, Default)]
#[require(Transform)]
pub struct SpotLight {
    pub color: [f32; 3],
    pub intensity: f32,
    pub range: f32,
    /// Половина угла раствора, радианы.
    pub inner_angle: f32,
    pub outer_angle: f32,
    /// Отбрасывать тени (1 проход и 1 слой карты — дёшево, включено по умолчанию).
    pub shadows: bool,
}

impl Default for SpotLight {
    fn default() -> Self {
        Self {
            color: [1.0; 3],
            intensity: 20.0,
            range: 20.0,
            inner_angle: 0.3,
            outer_angle: 0.45,
            shadows: true,
        }
    }
}

/// Рассеянный свет — одинаковый со всех сторон.
#[derive(Resource, Debug, Clone, Copy)]
pub struct AmbientLight {
    pub color: [f32; 3],
    pub intensity: f32,
}

impl Default for AmbientLight {
    fn default() -> Self {
        Self {
            color: [0.6, 0.7, 1.0],
            intensity: 0.15,
        }
    }
}

#[cfg(test)]
mod fit_tests {
    use super::FitAxis;

    #[test]
    fn shorter_keeps_vertical_angle_in_landscape() {
        let fov = 50f32.to_radians();
        assert!((FitAxis::Shorter.fov_y(fov, 16.0 / 9.0) - fov).abs() < 1e-6);
    }

    #[test]
    fn shorter_keeps_horizontal_angle_in_portrait() {
        let fov = 50f32.to_radians();
        let aspect = 9.0 / 19.5;
        let fov_y = FitAxis::Shorter.fov_y(fov, aspect);
        // Горизонтальный угол обратно из вертикального — тот же, что задан
        let fov_x = 2.0 * ((fov_y * 0.5).tan() * aspect).atan();
        assert!((fov_x - fov).abs() < 1e-5);
        assert!(fov_y > fov);
    }

    #[test]
    fn vertical_ignores_aspect() {
        let fov = 1.0;
        assert!((FitAxis::Vertical.fov_y(fov, 0.4) - fov).abs() < 1e-6);
    }

    #[test]
    fn shorter_height_keeps_width_in_portrait() {
        let height = FitAxis::Shorter.height(10.0, 0.5);
        assert!((height * 0.5 - 10.0).abs() < 1e-4);
        assert!((FitAxis::Shorter.height(10.0, 2.0) - 10.0).abs() < 1e-4);
    }
}
