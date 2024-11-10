import pytest
import image_interpolation

def fibonacci_pure_python(n, computed={0: 0, 1: 1}):
    if n not in computed:
        computed[n] = fibonacci_pure_python(n - 1, computed) + fibonacci_pure_python(n - 2, computed)
    return computed[n]

@pytest.mark.parametrize("n", [10, 20, 100, 500, 1000])
def test_fibonacci_pure_python(benchmark, n):
    result = benchmark(fibonacci_pure_python, n)
    assert result >= 0

@pytest.mark.parametrize("n", [10, 20, 100, 500, 1000])
def test_fibonacci_pyo3(benchmark, n):
    result = benchmark(image_interpolation.fibonacci, n)
    assert result >= 0