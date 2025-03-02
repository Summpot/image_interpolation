import cv2
import numpy as np


def _interpolate(image, scale_factor, interpolation_method):
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    return cv2.resize(
        image, (new_width, new_height), interpolation=interpolation_method
    )


def nearest_neighbor(image, scale_factor):
    return _interpolate(image, scale_factor, cv2.INTER_NEAREST_EXACT)


def bilinear(image, scale_factor):
    return _interpolate(image, scale_factor, cv2.INTER_LINEAR_EXACT)


def bicubic(image, scale_factor):
    return (_interpolate(image, scale_factor, cv2.INTER_CUBIC) * 255).astype(np.uint8)


def lanczos(image, scale_factor):
    return _interpolate(image, scale_factor, cv2.INTER_LANCZOS4)
