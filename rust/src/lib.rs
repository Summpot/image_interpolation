use pyo3::prelude::*;

#[pyfunction]
fn fibonacci(n: usize) -> u64 {
    if n <= 1 {
        return n as u64;
    }

    let (mut a, mut b) = (0, 1);
    for _ in 2..=n {
        let next = a + b;
        a = b;
        b = next;
    }

    b
}

#[pymodule]
fn image_interpolation(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fibonacci, m)?)?;
    Ok(())
}
