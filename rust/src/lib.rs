use pyo3::prelude::*;

#[pyfunction]
fn fibonacci(n: usize) -> u64 {
    if n <= 1 {
        return n as u64;
    }

    let mut dp = vec![0; n + 1];
    dp[1] = 1;

    for i in 2..=n {
        dp[i] = dp[i - 1] + dp[i - 2];
    }

    dp[n]
}

#[pymodule]
fn image_interpolation(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fibonacci, m)?)?;
    Ok(())
}
