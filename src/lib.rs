use pyo3::prelude::*;

use ndarray::{Array3, ArrayView3};
use numpy::{PyArray3, PyReadonlyArray3, ToPyArray};

#[pymodule]
fn image_interpolation(m: &Bound<'_, PyModule>) -> PyResult<()> {
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

    #[pyfn(m)]
    #[pyo3(name = "nearest_neighbor",signature=(image, scale_factor))]
    fn py_nearest_neighbor<'py>(
        py: Python<'py>,
        image: PyReadonlyArray3<'py, f64>,
        scale_factor: f64,
    ) -> Bound<'py, PyArray3<f64>> {
        let image_array = image.as_array();
        let interpolated_array = nearest_neighbor(image_array, scale_factor);
        interpolated_array.to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "bilinear",signature=(image, scale_factor))]
    fn py_bilinear<'py>(
        py: Python<'py>,
        image: PyReadonlyArray3<'py, f64>,
        scale_factor: f64,
    ) -> Bound<'py, PyArray3<f64>> {
        let image_array = image.as_array();
        let interpolated_array = bilinear(image_array, scale_factor);
        interpolated_array.to_pyarray(py)
    }
    Ok(())
}
