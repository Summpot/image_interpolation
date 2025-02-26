import pytest
import image_interpolation


def test_sum_as_string():
    assert image_interpolation.sum_as_string(1, 1) == "2"
