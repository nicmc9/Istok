//! Поле расстояний со знаком (SDF, signed distance field) из маски глифа.
//!
//! В атлас кладётся не яркость пикселя, а расстояние до контура буквы: 0.5 — на контуре,
//! больше — внутри, меньше — снаружи. Шейдер рисует край по уровню 0.5 при любом
//! масштабе: одна растеризация глифа годится и для мелкого, и для крупного текста, а края
//! остаются чёткими (в отличие от растянутой картинки). Заодно даром получаются обводка
//! (уровень ниже 0.5) и тень.
//!
//! Расстояния — точное евклидово преобразование (Felzenszwalb & Huttenlocher, «Distance
//! Transforms of Sampled Functions», 2012): два прохода одномерного преобразования, O(n).

/// «Бесконечность» для преобразования: больше любого квадрата расстояния в глифе.
const FAR: f32 = 1e10;

/// SDF из маски `width × height` (0..255). Результат больше на `spread` с каждой стороны:
/// поле должно выходить за контур, иначе обводке некуда расти.
/// `spread` — расстояние в пикселях, которое укладывается в диапазон 0..0.5 значения.
pub(crate) fn build(mask: &[u8], width: usize, height: usize, spread: usize) -> Vec<u8> {
    let (w, h) = (width + 2 * spread, height + 2 * spread);
    // Внутри ли пиксель (с полями вокруг маски — снаружи)
    let inside = |x: usize, y: usize| -> bool {
        x >= spread
            && y >= spread
            && x - spread < width
            && y - spread < height
            && mask[(y - spread) * width + (x - spread)] >= 128
    };
    // Квадрат расстояния до ближайшего пикселя «снаружи» и до ближайшего «внутри»
    let mut to_outside = vec![0.0; w * h];
    let mut to_inside = vec![0.0; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if inside(x, y) {
                to_outside[i] = FAR;
            } else {
                to_inside[i] = FAR;
            }
        }
    }
    transform_2d(&mut to_outside, w, h);
    transform_2d(&mut to_inside, w, h);

    #[allow(clippy::cast_precision_loss)]
    let spread = spread as f32;
    to_outside
        .iter()
        .zip(&to_inside)
        .map(|(&out, &inn)| {
            // Расстояния между центрами пикселей; −0.5 — контур проходит между ними
            let signed = if out > 0.0 {
                out.sqrt() - 0.5 // внутри: положительное
            } else {
                -(inn.sqrt() - 0.5)
            };
            let value = (0.5 + signed / (2.0 * spread)).clamp(0.0, 1.0);
            // [Rust] `as u8` из f32 насыщает (а не обрезает биты, как в C++)
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let byte = (value * 255.0).round() as u8;
            byte
        })
        .collect()
}

/// Двумерное преобразование: по столбцам, затем по строкам.
// Имена f, d, v, z — как в статье: так код проще сверять с её псевдокодом
#[allow(clippy::many_single_char_names)]
fn transform_2d(grid: &mut [f32], w: usize, h: usize) {
    let n = w.max(h);
    let mut f = vec![0.0; n];
    let mut d = vec![0.0; n];
    let mut v = vec![0usize; n];
    let mut z = vec![0.0; n + 1];
    for x in 0..w {
        for y in 0..h {
            f[y] = grid[y * w + x];
        }
        transform_1d(&f[..h], &mut d[..h], &mut v, &mut z);
        for y in 0..h {
            grid[y * w + x] = d[y];
        }
    }
    for y in 0..h {
        f[..w].copy_from_slice(&grid[y * w..(y + 1) * w]);
        transform_1d(&f[..w], &mut d[..w], &mut v, &mut z);
        grid[y * w..(y + 1) * w].copy_from_slice(&d[..w]);
    }
}

/// Одномерное: нижняя огибающая парабол `(q − p)² + f(p)`.
#[allow(clippy::cast_precision_loss, clippy::many_single_char_names)]
fn transform_1d(f: &[f32], d: &mut [f32], v: &mut [usize], z: &mut [f32]) {
    let n = f.len();
    let mut k = 0;
    v[0] = 0;
    z[0] = -FAR;
    z[1] = FAR;
    // Точка пересечения параболы из q с параболой из v[k]
    let intersect = |q: usize, p: usize| {
        let (qf, pf) = (q as f32, p as f32);
        ((f[q] + qf * qf) - (f[p] + pf * pf)) / (2.0 * qf - 2.0 * pf)
    };
    for q in 1..n {
        let mut s = intersect(q, v[k]);
        while s <= z[k] {
            k -= 1;
            s = intersect(q, v[k]);
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = FAR;
    }
    k = 0;
    for (q, out) in d.iter_mut().enumerate().take(n) {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let dq = q as f32 - v[k] as f32;
        *out = dq * dq + f[v[k]];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_has_contour_at_half() {
        // Квадрат 8×8 в маске 8×8: всё внутри; поля 4 пикселя снаружи
        let mask = vec![255u8; 64];
        let sdf = build(&mask, 8, 8, 4);
        let w = 16;
        let at = |x: usize, y: usize| sdf[y * w + x];
        assert!(at(8, 8) > 200, "центр глубоко внутри: {}", at(8, 8));
        assert!(at(0, 0) < 20, "угол далеко снаружи: {}", at(0, 0));
        // Пиксели по обе стороны контура (x = 3 снаружи, x = 4 внутри) — около 0.5
        let (out, inn) = (at(3, 8), at(4, 8));
        assert!(out < 128 && inn >= 128, "{out} {inn}");
        assert!(inn - out < 40, "плавный переход: {out} → {inn}");
    }

    #[test]
    fn empty_mask_is_all_outside() {
        let sdf = build(&[0u8; 16], 4, 4, 2);
        assert!(sdf.iter().all(|&v| v < 128));
    }
}
