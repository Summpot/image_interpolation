use pyo3::prelude::*;

use ndarray::{Array3, ArrayView3};
use numpy::{PyArray3, PyReadonlyArray3, ToPyArray};

#[pymodule]
fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    fn nearest_neighbor(image: ArrayView3<u8>, scale_factor: f64) -> Array3<u8> {
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

    fn bilinear(image: ArrayView3<u8>, scale_factor: f64) -> Array3<u8> {
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
                    // 将 u8 转换为 f64 进行计算
                    let top_f64 = (image[[y0, x0, channel]] as f64) * (1.0 - dx) + (image[[y0, x1, channel]] as f64) * dx;
                    let bottom_f64 = (image[[y1, x0, channel]] as f64) * (1.0 - dx) + (image[[y1, x1, channel]] as f64) * dx;
                    let value_f64 = top_f64 * (1.0 - dy) + bottom_f64 * dy;
                    // 钳制到 0-255 范围并转换为 u8
                    new_image[[y, x, channel]] = value_f64.round().max(0.0).min(255.0) as u8;
                }
            }
        }
        new_image
    }

    fn bicubic_weight(t: f64) -> f64 {
        let a = -0.5;
        let t = t.abs();
        if t <= 1.0 {
            (a + 2.0) * t.powi(3) - (a + 3.0) * t.powi(2) + 1.0
        } else if t < 2.0 {
            a * t.powi(3) - 5.0 * a * t.powi(2) + 8.0 * a * t - 4.0 * a
        } else {
            0.0
        }
    }

    fn bicubic(image: ArrayView3<u8>, scale_factor: f64) -> Array3<u8> {
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
                    let mut value_f64 = 0.0;
                    for dy in -1..=2 {
                        for dx in -1..=2 {
                            let px = (x0 + dx).max(0).min(width as isize - 1) as usize;
                            let py = (y0 + dy).max(0).min(height as isize - 1) as usize;
                            let wx = bicubic_weight(fx - (x0 + dx) as f64);
                            let wy = bicubic_weight(fy - (y0 + dy) as f64);
                            // 转换为 f64 进行计算
                            value_f64 += (image[[py, px, channel]] as f64) * wx * wy;
                        }
                    }
                    // 钳制到 0-255 范围并转换为 u8
                    new_image[[y, x, channel]] = value_f64.round().max(0.0).min(255.0) as u8;
                }
            }
        }
        new_image
    }

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

    fn lanczos(image: ArrayView3<u8>, scale_factor: f64) -> Array3<u8> {
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
                    let mut value_f64 = 0.0;
                    let mut weight_sum = 0.0;
                    for dy in -2..=2 { // 5x5 neighborhood for a=3
                        for dx in -2..=2 {
                            let px = (x0 + dx).max(0).min(width as isize - 1) as usize;
                            let py = (y0 + dy).max(0).min(height as isize - 1) as usize;
                            let wx = lanczos_weight(fx - (x0 + dx) as f64, a);
                            let wy = lanczos_weight(fy - (y0 + dy) as f64, a);
                            let weight = wx * wy;
                            // 转换为 f64 进行计算
                            value_f64 += (image[[py, px, channel]] as f64) * weight;
                            weight_sum += weight;
                        }
                    }
                    // 归一化、钳制到 0-255 范围并转换为 u8
                    new_image[[y, x, channel]] = (value_f64 / weight_sum).round().max(0.0).min(255.0) as u8;
                }
            }
        }
        new_image
    }

    #[pyfn(m)]
    #[pyo3(name = "nearest_neighbor", signature=(image, scale_factor))]
    fn py_nearest_neighbor<'py>(
        py: Python<'py>,
        image: PyReadonlyArray3<'py, u8>, 
        scale_factor: f64,
    ) -> Bound<'py, PyArray3<u8>> { 
        let image_array = image.as_array();
        let interpolated_array = nearest_neighbor(image_array, scale_factor);
        interpolated_array.to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "bilinear", signature=(image, scale_factor))]
    fn py_bilinear<'py>(
        py: Python<'py>,
        image: PyReadonlyArray3<'py, u8>, 
        scale_factor: f64,
    ) -> Bound<'py, PyArray3<u8>> { 
        let image_array = image.as_array();
        let interpolated_array = bilinear(image_array, scale_factor);
        interpolated_array.to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "bicubic", signature=(image, scale_factor))]
    fn py_bicubic<'py>(
        py: Python<'py>,
        image: PyReadonlyArray3<'py, u8>, 
        scale_factor: f64,
    ) -> Bound<'py, PyArray3<u8>> { 
        let image_array = image.as_array();
        let interpolated_array = bicubic(image_array, scale_factor);
        interpolated_array.to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "lanczos", signature=(image, scale_factor))]
    fn py_lanczos<'py>(
        py: Python<'py>,
        image: PyReadonlyArray3<'py, u8>, 
        scale_factor: f64,
    ) -> Bound<'py, PyArray3<u8>> { 
        let image_array = image.as_array();
        let interpolated_array = lanczos(image_array, scale_factor);
        interpolated_array.to_pyarray(py)
    }

    Ok(())
}