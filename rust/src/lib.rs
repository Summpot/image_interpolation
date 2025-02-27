use pyo3::prelude::*;

use ndarray::{Array3, ArrayView3};
use numpy::{PyArray3, PyReadonlyArray3, ToPyArray};

#[pymodule]
fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Nearest Neighbor Interpolation
    fn nearest_neighbor(image: ArrayView3<f64>, scale_factor: f64) -> Array3<f64> {
        let (height, width, channels) = image.dim();
        let new_height = (height as f64 * scale_factor) as usize;
        let new_width = (width as f64 * scale_factor) as usize;
        let mut new_image = Array3::zeros((new_height, new_width, channels));

        for y in 0..new_height {
            for x in 0..new_width {
                let original_x = (x as f64 / scale_factor).round() as usize;
                let original_y = (y as f64 / scale_factor).round() as usize;
                if original_x < width && original_y < height {
                    for channel in 0..channels {
                        new_image[[y, x, channel]] = image[[original_y, original_x, channel]];
                    }
                }
            }
        }
        new_image
    }

    // Bilinear Interpolation
    fn bilinear(image: ArrayView3<f64>, scale_factor: f64) -> Array3<f64> {
        let (height, width, channels) = image.dim();
        let new_height = (height as f64 * scale_factor) as usize;
        let new_width = (width as f64 * scale_factor) as usize;
        let mut new_image = Array3::zeros((new_height, new_width, channels));

        for y in 0..new_height {
            for x in 0..new_width {
                let fx = x as f64 / scale_factor;
                let fy = y as f64 / scale_factor;
                let x0 = fx.floor() as usize;
                let y0 = fy.floor() as usize;
                let x1 = (x0 + 1).min(width - 1);
                let y1 = (y0 + 1).min(height - 1);

                let dx = fx - x0 as f64;
                let dy = fy - y0 as f64;

                for channel in 0..channels {
                    let top = image[[y0, x0, channel]] * (1.0 - dx) + image[[y0, x1, channel]] * dx;
                    let bottom =
                        image[[y1, x0, channel]] * (1.0 - dx) + image[[y1, x1, channel]] * dx;
                    new_image[[y, x, channel]] = top * (1.0 - dy) + bottom * dy;
                }
            }
        }
        new_image
    }

    // Bicubic Interpolation
    fn bicubic_weight(t: f64) -> f64 {
        let a = -0.5; // Common parameter for bicubic interpolation
        let t = t.abs();
        if t <= 1.0 {
            (a + 2.0) * t.powi(3) - (a + 3.0) * t.powi(2) + 1.0
        } else if t < 2.0 {
            a * t.powi(3) - 5.0 * a * t.powi(2) + 8.0 * a * t - 4.0 * a
        } else {
            0.0
        }
    }

    fn bicubic(image: ArrayView3<f64>, scale_factor: f64) -> Array3<f64> {
        let (height, width, channels) = image.dim();
        let new_height = (height as f64 * scale_factor) as usize;
        let new_width = (width as f64 * scale_factor) as usize;
        let mut new_image = Array3::zeros((new_height, new_width, channels));

        for y in 0..new_height {
            for x in 0..new_width {
                let fx = x as f64 / scale_factor;
                let fy = y as f64 / scale_factor;
                let x0 = fx.floor() as isize;
                let y0 = fy.floor() as isize;

                for channel in 0..channels {
                    let mut value = 0.0;
                    for dy in -1..=2 {
                        for dx in -1..=2 {
                            let px = (x0 + dx).max(0).min(width as isize - 1) as usize;
                            let py = (y0 + dy).max(0).min(height as isize - 1) as usize;
                            let wx = bicubic_weight(fx - (x0 + dx) as f64);
                            let wy = bicubic_weight(fy - (y0 + dy) as f64);
                            value += image[[py, px, channel]] * wx * wy;
                        }
                    }
                    new_image[[y, x, channel]] = value.max(0.0).min(1.0); // Clamp to [0, 1]
                }
            }
        }
        new_image
    }

    // Lanczos Interpolation (using a=3 for 3-lobe Lanczos)
    fn lanczos_weight(t: f64, a: f64) -> f64 {
        if t.abs() < f64::EPSILON {
            1.0
        } else if t.abs() < a {
            let pi_t = std::f64::consts::PI * t;
            a * (pi_t.sin() / pi_t) * ((pi_t / a).sin() / (pi_t / a))
        } else {
            0.0
        }
    }

    fn lanczos(image: ArrayView3<f64>, scale_factor: f64) -> Array3<f64> {
        let (height, width, channels) = image.dim();
        let new_height = (height as f64 * scale_factor) as usize;
        let new_width = (width as f64 * scale_factor) as usize;
        let mut new_image = Array3::zeros((new_height, new_width, channels));
        let a = 3.0; // Lanczos window size (3-lobe)

        for y in 0..new_height {
            for x in 0..new_width {
                let fx = x as f64 / scale_factor;
                let fy = y as f64 / scale_factor;
                let x0 = fx.floor() as isize;
                let y0 = fy.floor() as isize;

                for channel in 0..channels {
                    let mut value = 0.0;
                    let mut weight_sum = 0.0;
                    for dy in -2..=2 {
                        // 5x5 neighborhood for a=3
                        for dx in -2..=2 {
                            let px = (x0 + dx).max(0).min(width as isize - 1) as usize;
                            let py = (y0 + dy).max(0).min(height as isize - 1) as usize;
                            let wx = lanczos_weight(fx - (x0 + dx) as f64, a);
                            let wy = lanczos_weight(fy - (y0 + dy) as f64, a);
                            let weight = wx * wy;
                            value += image[[py, px, channel]] * weight;
                            weight_sum += weight;
                        }
                    }
                    new_image[[y, x, channel]] = (value / weight_sum).max(0.0).min(1.0);
                    // Normalize and clamp
                }
            }
        }
        new_image
    }

    // Python bindings for Nearest Neighbor
    #[pyfn(m)]
    #[pyo3(name = "nearest_neighbor", signature=(image, scale_factor))]
    fn py_nearest_neighbor<'py>(
        py: Python<'py>,
        image: PyReadonlyArray3<'py, f64>,
        scale_factor: f64,
    ) -> Bound<'py, PyArray3<f64>> {
        let image_array = image.as_array();
        let interpolated_array = nearest_neighbor(image_array, scale_factor);
        interpolated_array.to_pyarray(py)
    }

    // Python bindings for Bilinear
    #[pyfn(m)]
    #[pyo3(name = "bilinear", signature=(image, scale_factor))]
    fn py_bilinear<'py>(
        py: Python<'py>,
        image: PyReadonlyArray3<'py, f64>,
        scale_factor: f64,
    ) -> Bound<'py, PyArray3<f64>> {
        let image_array = image.as_array();
        let interpolated_array = bilinear(image_array, scale_factor);
        interpolated_array.to_pyarray(py)
    }

    // Python bindings for Bicubic
    #[pyfn(m)]
    #[pyo3(name = "bicubic", signature=(image, scale_factor))]
    fn py_bicubic<'py>(
        py: Python<'py>,
        image: PyReadonlyArray3<'py, f64>,
        scale_factor: f64,
    ) -> Bound<'py, PyArray3<f64>> {
        let image_array = image.as_array();
        let interpolated_array = bicubic(image_array, scale_factor);
        interpolated_array.to_pyarray(py)
    }

    // Python bindings for Lanczos
    #[pyfn(m)]
    #[pyo3(name = "lanczos", signature=(image, scale_factor))]
    fn py_lanczos<'py>(
        py: Python<'py>,
        image: PyReadonlyArray3<'py, f64>,
        scale_factor: f64,
    ) -> Bound<'py, PyArray3<f64>> {
        let image_array = image.as_array();
        let interpolated_array = lanczos(image_array, scale_factor);
        interpolated_array.to_pyarray(py)
    }
    Ok(())
}
