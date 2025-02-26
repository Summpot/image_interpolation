use pyo3::prelude::*;

use ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray, PyArray2, PyArrayDyn, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;

#[pymodule]
fn image_interpolation(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Nearest Neighbor Interpolation
    fn nearest_neighbor(image: ArrayView2<f64>, scale_factor: f64) -> Array2<f64> {
        let (height, width) = image.dim();
        let new_height = (height as f64 * scale_factor) as usize;
        let new_width = (width as f64 * scale_factor) as usize;
        let mut new_image = Array2::zeros((new_height, new_width));

        for y in 0..new_height {
            for x in 0..new_width {
                let original_x = (x as f64 / scale_factor) as usize;
                let original_y = (y as f64 / scale_factor) as usize;
                if original_x < width && original_y < height {
                    new_image[[y, x]] = image[[original_y, original_x]];
                }
            }
        }
        new_image
    }

    // Bilinear Interpolation
    fn bilinear(image: ArrayView2<f64>, scale_factor: f64) -> Array2<f64> {
        let (height, width) = image.dim();
        let new_height = (height as f64 * scale_factor) as usize;
        let new_width = (width as f64 * scale_factor) as usize;
        let mut new_image = Array2::zeros((new_height, new_width));

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

                let top = image[[y0, x0]] * (1.0 - dx) + image[[y0, x1]] * dx;
                let bottom = image[[y1, x0]] * (1.0 - dx) + image[[y1, x1]] * dx;
                new_image[[y, x]] = top * (1.0 - dy) + bottom * dy;
            }
        }
        new_image
    }

    #[pyfn(m)]
    #[pyo3(name = "nearest_neighbor")]
    fn py_nearest_neighbor<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f64>,
        scale_factor: f64,
    ) -> Bound<'py, PyArray2<f64>> {
        let image_array = image.as_array();
        let interpolated_array = nearest_neighbor(image_array, scale_factor);
        interpolated_array.to_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "bilinear")]
    fn py_bilinear<'py>(
        py: Python<'py>,
        image: PyReadonlyArray2<'py, f64>,
        scale_factor: f64,
    ) -> Bound<'py, PyArray2<f64>> {
        let image_array = image.as_array();
        let interpolated_array = bilinear(image_array, scale_factor);
        interpolated_array.to_pyarray(py)
    }
    Ok(())
}
