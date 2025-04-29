use pyo3::prelude::*;

use ndarray::{Array3, ArrayView3};
use numpy::{PyArray3, PyReadonlyArray3, ToPyArray};

#[pymodule]
fn rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    fn nearest_neighbor(image: ArrayView3<u8>, scale_factor: f64) -> Array3<u8> {
        if scale_factor <= 0.0 {
            panic!("Scale factor must be positive");
        }
        let (height, width, channels) = image.dim();
        if height == 0 || width == 0 || channels == 0 {
            panic!(
                "Invalid image dimensions: {}x{}x{}",
                height, width, channels
            );
        }
        let new_height = (height as f64 * scale_factor).round() as usize;
        let new_width = (width as f64 * scale_factor).round() as usize;
        if new_height == 0 || new_width == 0 {
            panic!("Output dimensions too small: {}x{}", new_height, new_width);
        }
        let mut new_image = Array3::zeros((new_height, new_width, channels));

        for y in 0..new_height {
            for x in 0..new_width {
                let original_x = ((x as f64 / scale_factor).round() as usize).min(width - 1);
                let original_y = ((y as f64 / scale_factor).round() as usize).min(height - 1);
                for channel in 0..channels {
                    new_image[[y, x, channel]] = image[[original_y, original_x, channel]];
                }
            }
        }
        new_image
    }

    fn bilinear(image: ArrayView3<u8>, scale_factor: f64) -> Array3<u8> {
        if scale_factor <= 0.0 {
            panic!("Scale factor must be positive");
        }
        let (height, width, channels) = image.dim();
        if height == 0 || width == 0 || channels == 0 {
            panic!(
                "Invalid image dimensions: {}x{}x{}",
                height, width, channels
            );
        }
        let new_height = (height as f64 * scale_factor).round() as usize;
        let new_width = (width as f64 * scale_factor).round() as usize;
        if new_height == 0 || new_width == 0 {
            panic!("Output dimensions too small: {}x{}", new_height, new_width);
        }
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
                    let top_f64 = (image[[y0, x0, channel]] as f64) * (1.0 - dx)
                        + (image[[y0, x1, channel]] as f64) * dx;
                    let bottom_f64 = (image[[y1, x0, channel]] as f64) * (1.0 - dx)
                        + (image[[y1, x1, channel]] as f64) * dx;
                    let value_f64 = top_f64 * (1.0 - dy) + bottom_f64 * dy;
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
        if scale_factor <= 0.0 {
            panic!("Scale factor must be positive");
        }
        let (height, width, channels) = image.dim();
        if height == 0 || width == 0 || channels == 0 {
            panic!(
                "Invalid image dimensions: {}x{}x{}",
                height, width, channels
            );
        }
        let new_height = (height as f64 * scale_factor).round() as usize;
        let new_width = (width as f64 * scale_factor).round() as usize;
        if new_height == 0 || new_width == 0 {
            panic!("Output dimensions too small: {}x{}", new_height, new_width);
        }
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
                            value_f64 += (image[[py, px, channel]] as f64) * wx * wy;
                        }
                    }
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
        if scale_factor <= 0.0 {
            panic!("Scale factor must be positive");
        }
        let (height, width, channels) = image.dim();
        if height == 0 || width == 0 || channels == 0 {
            panic!(
                "Invalid image dimensions: {}x{}x{}",
                height, width, channels
            );
        }
        let new_height = (height as f64 * scale_factor).round() as usize;
        let new_width = (width as f64 * scale_factor).round() as usize;
        if new_height == 0 || new_width == 0 {
            panic!("Output dimensions too small: {}x{}", new_height, new_width);
        }
        let mut new_image = Array3::zeros((new_height, new_width, channels));
        let a = 3.0;

        for y in 0..new_height {
            for x in 0..new_width {
                let fx = x as f64 / scale_factor;
                let fy = y as f64 / scale_factor;
                let x0 = fx.floor() as isize;
                let y0 = fy.floor() as isize;

                for channel in 0..channels {
                    let mut value_f64 = 0.0;
                    let mut weight_sum = 0.0;
                    for dy in -2..=2 {
                        for dx in -2..=2 {
                            let px = (x0 + dx).max(0).min(width as isize - 1) as usize;
                            let py = (y0 + dy).max(0).min(height as isize - 1) as usize;
                            let wx = lanczos_weight(fx - (x0 + dx) as f64, a);
                            let wy = lanczos_weight(fy - (y0 + dy) as f64, a);
                            let weight = wx * wy;
                            value_f64 += (image[[py, px, channel]] as f64) * weight;
                            weight_sum += weight;
                        }
                    }
                    new_image[[y, x, channel]] = if weight_sum > 0.0 {
                        (value_f64 / weight_sum).round().max(0.0).min(255.0) as u8
                    } else {
                        0
                    };
                }
            }
        }
        new_image
    }

    fn edge_preserving(image: ArrayView3<u8>, scale_factor: f64) -> Array3<u8> {
        if scale_factor <= 0.0 {
            panic!("Scale factor must be positive");
        }
        let (height, width, channels) = image.dim();
        if height == 0 || width == 0 || channels == 0 {
            panic!(
                "Invalid image dimensions: {}x{}x{}",
                height, width, channels
            );
        }
        let new_height = (height as f64 * scale_factor).round() as usize;
        let new_width = (width as f64 * scale_factor).round() as usize;
        if new_height == 0 || new_width == 0 {
            panic!("Output dimensions too small: {}x{}", new_height, new_width);
        }
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
                    let grad_x = if x1 > x0 {
                        (image[[y0, x1, channel]] as f64 - image[[y0, x0, channel]] as f64).abs()
                    } else {
                        0.0
                    };
                    let grad_y = if y1 > y0 {
                        (image[[y1, x0, channel]] as f64 - image[[y0, x0, channel]] as f64).abs()
                    } else {
                        0.0
                    };

                    let total_grad = grad_x + grad_y + 1e-6;
                    let wx = grad_y / total_grad;
                    let wy = grad_x / total_grad;

                    let top = (image[[y0, x0, channel]] as f64) * (1.0 - dx)
                        + (image[[y0, x1, channel]] as f64) * dx;
                    let bottom = (image[[y1, x0, channel]] as f64) * (1.0 - dx)
                        + (image[[y1, x1, channel]] as f64) * dx;
                    let value_f64 = (top * (1.0 - dy) + bottom * dy) * wx
                        + (top * wy + bottom * (1.0 - wy)) * dy;
                    new_image[[y, x, channel]] = value_f64.round().max(0.0).min(255.0) as u8;
                }
            }
        }
        new_image
    }

    fn haar_wavelet_decompose(image: ArrayView3<u8>, channel: usize) -> (Array3<f64>, Array3<f64>) {
        let (height, width, _) = image.dim();
        // 确保最小尺寸为 4x4，避免分解后子带过小
        if height < 4 || width < 4 {
            panic!("Image too small for wavelet decomposition: {}x{}", height, width);
        }
        // 调整为偶数尺寸
        let h = height - (height % 2);
        let w = width - (width % 2);
        let out_height = h / 2;
        let out_width = w / 2;
        if out_height == 0 || out_width == 0 {
            panic!("Output subband dimensions too small: {}x{}", out_height, out_width);
        }
        let mut low_freq = Array3::zeros((out_height, out_width, 1));
        let mut high_freq = Array3::zeros((out_height, out_width, 3));
    
        for y in (0..h).step_by(2) {
            for x in (0..w).step_by(2) {
                let p00 = image[[y, x, channel]] as f64;
                let p01 = image[[y, x + 1, channel]] as f64;
                let p10 = image[[y + 1, x, channel]] as f64;
                let p11 = image[[y + 1, x + 1, channel]] as f64;
    
                low_freq[[y / 2, x / 2, 0]] = (p00 + p01 + p10 + p11) / 4.0;
                high_freq[[y / 2, x / 2, 0]] = p00 - low_freq[[y / 2, x / 2, 0]];
                high_freq[[y / 2, x / 2, 1]] = p01 - low_freq[[y / 2, x / 2, 0]];
                high_freq[[y / 2, x / 2, 2]] = p11 - low_freq[[y / 2, x / 2, 0]];
            }
        }
        (low_freq, high_freq)
    }
    
    fn haar_wavelet_reconstruct(low_freq: ArrayView3<f64>, high_freq: ArrayView3<f64>, output_shape: (usize, usize)) -> Array3<u8> {
        let (new_height, new_width) = output_shape;
        // 确保最小输出尺寸
        if new_height < 4 || new_width < 4 {
            panic!("Output dimensions too small for wavelet reconstruction: {}x{}", new_height, new_width);
        }
        // 调整为偶数尺寸
        let out_height = new_height - (new_height % 2);
        let out_width = new_width - (new_width % 2);
        let max_y = (out_height / 2).min(low_freq.dim().0);
        let max_x = (out_width / 2).min(low_freq.dim().1);
        if max_y == 0 || max_x == 0 {
            panic!("Reconstruction subband dimensions too small: {}x{}", max_y, max_x);
        }
        let mut reconstructed = Array3::zeros((new_height, new_width, 1));
    
        for y in 0..max_y {
            for x in 0..max_x {
                let low = low_freq[[y, x, 0]];
                let h0 = if y < high_freq.dim().0 && x < high_freq.dim().1 {
                    high_freq[[y, x, 0]]
                } else {
                    0.0
                };
                let h1 = if y < high_freq.dim().0 && x < high_freq.dim().1 {
                    high_freq[[y, x, 1]]
                } else {
                    0.0
                };
                let h2 = if y < high_freq.dim().0 && x < high_freq.dim().1 {
                    high_freq[[y, x, 2]]
                } else {
                    0.0
                };
    
                if y * 2 < new_height && x * 2 < new_width {
                    reconstructed[[y * 2, x * 2, 0]] = (low + h0).round().max(0.0).min(255.0) as u8;
                }
                if y * 2 < new_height && x * 2 + 1 < new_width {
                    reconstructed[[y * 2, x * 2 + 1, 0]] = (low + h1).round().max(0.0).min(255.0) as u8;
                }
                if y * 2 + 1 < new_height && x * 2 < new_width {
                    reconstructed[[y * 2 + 1, x * 2, 0]] = (low + h2).round().max(0.0).min(255.0) as u8;
                }
                if y * 2 + 1 < new_height && x * 2 + 1 < new_width {
                    reconstructed[[y * 2 + 1, x * 2 + 1, 0]] = (low + h2).round().max(0.0).min(255.0) as u8;
                }
            }
        }
        reconstructed
    }
    
    fn wavelet_based(image: ArrayView3<u8>, scale_factor: f64) -> Array3<u8> {
        if scale_factor <= 0.0 {
            panic!("Scale factor must be positive");
        }
        let (height, width, channels) = image.dim();
        if height < 4 || width < 4 || channels == 0 {
            panic!("Image too small or invalid: {}x{}x{}", height, width, channels);
        }
        let new_height = (height as f64 * scale_factor).round() as usize;
        let new_width = (width as f64 * scale_factor).round() as usize;
        if new_height < 4 || new_width < 4 {
            panic!("Output dimensions too small: {}x{}", new_height, new_width);
        }
        let mut new_image = Array3::zeros((new_height, new_width, channels));
    
        for channel in 0..channels {
            let (low_freq, high_freq) = haar_wavelet_decompose(image, channel);
            let low_freq_view = low_freq.view();
            let interpolated_low = bilinear(low_freq_view.mapv(|v| v as u8).view(), scale_factor);
            let reconstructed = haar_wavelet_reconstruct(
                interpolated_low.mapv(|v| v as f64).view(),
                high_freq.view(),
                (new_height, new_width),
            );
    
            for y in 0..new_height {
                for x in 0..new_width {
                    new_image[[y, x, channel]] = reconstructed[[y, x, 0]];
                }
            }
        }
        new_image
    }

    fn nedi(image: ArrayView3<u8>, scale_factor: f64) -> Array3<u8> {
        if scale_factor <= 0.0 {
            panic!("Scale factor must be positive");
        }
        let (height, width, channels) = image.dim();
        if height == 0 || width == 0 || channels == 0 {
            panic!(
                "Invalid image dimensions: {}x{}x{}",
                height, width, channels
            );
        }
        let new_height = (height as f64 * scale_factor).round() as usize;
        let new_width = (width as f64 * scale_factor).round() as usize;
        if new_height == 0 || new_width == 0 {
            panic!("Output dimensions too small: {}x{}", new_height, new_width);
        }
        let mut new_image = Array3::zeros((new_height, new_width, channels));

        for y in 0..new_height {
            for x in 0..new_width {
                let fx = x as f64 / scale_factor;
                let fy = y as f64 / scale_factor;
                let x0 = fx.floor() as isize;
                let y0 = fy.floor() as isize;

                for channel in 0..channels {
                    let mut value_f64 = 0.0;
                    let mut weight_sum = 0.0;

                    for dy in -1..=2 {
                        for dx in -1..=2 {
                            let px = (x0 + dx).max(0).min(width as isize - 1) as usize;
                            let py = (y0 + dy).max(0).min(height as isize - 1) as usize;
                            let dist_x = (fx - (x0 + dx) as f64).abs();
                            let dist_y = (fy - (y0 + dy) as f64).abs();
                            let weight = 1.0 / (dist_x * dist_y + 1e-6);
                            value_f64 += (image[[py, px, channel]] as f64) * weight;
                            weight_sum += weight;
                        }
                    }
                    new_image[[y, x, channel]] = if weight_sum > 0.0 {
                        (value_f64 / weight_sum).round().max(0.0).min(255.0) as u8
                    } else {
                        0
                    };
                }
            }
        }
        new_image
    }

    fn dcci(image: ArrayView3<u8>, scale_factor: f64) -> Array3<u8> {
        if scale_factor <= 0.0 {
            panic!("Scale factor must be positive");
        }
        let (height, width, channels) = image.dim();
        if height == 0 || width == 0 || channels == 0 {
            panic!(
                "Invalid image dimensions: {}x{}x{}",
                height, width, channels
            );
        }
        let new_height = (height as f64 * scale_factor).round() as usize;
        let new_width = (width as f64 * scale_factor).round() as usize;
        if new_height == 0 || new_width == 0 {
            panic!("Output dimensions too small: {}x{}", new_height, new_width);
        }
        let mut new_image = Array3::zeros((new_height, new_width, channels));

        for y in 0..new_height {
            for x in 0..new_width {
                let fx = x as f64 / scale_factor;
                let fy = y as f64 / scale_factor;
                let x0 = fx.floor() as isize;
                let y0 = fy.floor() as isize;

                for channel in 0..channels {
                    let grad_x = if x0 > 0 && x0 < width as isize - 1 {
                        (image[[y0 as usize, (x0 + 1) as usize, channel]] as f64
                            - image[[y0 as usize, (x0 - 1) as usize, channel]] as f64)
                            .abs()
                    } else {
                        0.0
                    };
                    let grad_y = if y0 > 0 && y0 < height as isize - 1 {
                        (image[[(y0 + 1) as usize, x0 as usize, channel]] as f64
                            - image[[(y0 - 1) as usize, x0 as usize, channel]] as f64)
                            .abs()
                    } else {
                        0.0
                    };
                    let direction = grad_y.atan2(grad_x);

                    let mut value_f64 = 0.0;
                    for dy in -1..=2 {
                        for dx in -1..=2 {
                            let px = (x0 + dx).max(0).min(width as isize - 1) as usize;
                            let py = (y0 + dy).max(0).min(height as isize - 1) as usize;
                            let dist_x = fx - (x0 + dx) as f64;
                            let dist_y = fy - (y0 + dy) as f64;

                            let rot_x = dist_x * direction.cos() + dist_y * direction.sin();
                            let rot_y = -dist_x * direction.sin() + dist_y * direction.cos();
                            let wx = bicubic_weight(rot_x);
                            let wy = bicubic_weight(rot_y);

                            value_f64 += (image[[py, px, channel]] as f64) * wx * wy;
                        }
                    }
                    new_image[[y, x, channel]] = value_f64.round().max(0.0).min(255.0) as u8;
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

    #[pyfn(m)]
    #[pyo3(name = "edge_preserving", signature=(image, scale_factor))]
    fn py_edge_preserving<'py>(
        py: Python<'py>,
        image: PyReadonlyArray3<'py, u8>,
        scale_factor: f64,
    ) -> Bound<'py, PyArray3<u8>> {
        let image_array = image.as_array();
        let interpolated_array = edge_preserving(image_array, scale_factor);
        interpolated_array.to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "wavelet_based", signature=(image, scale_factor))]
    fn py_wavelet_based<'py>(
        py: Python<'py>,
        image: PyReadonlyArray3<'py, u8>,       
        scale_factor: f64,
    ) -> Bound<'py, PyArray3<u8>> {
        let image_array = image.as_array();
        let interpolated_array = wavelet_based(image_array, scale_factor);
        interpolated_array.to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "nedi", signature=(image, scale_factor))]
    fn py_nedi<'py>(
        py: Python<'py>,
        image: PyReadonlyArray3<'py, u8>,
        scale_factor: f64,
    ) -> Bound<'py, PyArray3<u8>> {
        let image_array = image.as_array();
        let interpolated_array = nedi(image_array, scale_factor);
        interpolated_array.to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "dcci", signature=(image, scale_factor))]
    fn py_dcci<'py>(
        py: Python<'py>,
        image: PyReadonlyArray3<'py, u8>,
        scale_factor: f64,
    ) -> Bound<'py, PyArray3<u8>> {
        let image_array = image.as_array();
        let interpolated_array = dcci(image_array, scale_factor);
        interpolated_array.to_pyarray(py)
    }
    Ok(())
}
