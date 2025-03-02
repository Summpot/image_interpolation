import inspect

import cv2
import numpy as np


def get_public_functions(module):
    return [
        member
        for name, member in inspect.getmembers(module)
        if inspect.isroutine(member)
        and not name.startswith("_")
        and (
            getattr(member, "__module__") == module.__name__
            or inspect.isbuiltin(member)  # for pyo3
        )
    ]


def downsample_image(image, target_shape):
    return cv2.resize(
        image, dsize=(target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA
    )
