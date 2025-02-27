import itertools
import numpy as np
import pyperf
import opencv
import skimage
import rust
import inspect


def prepare_image(height, width):
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


def get_public_functions(module):
    return [
        obj
        for name, obj in inspect.getmembers(module)
        if inspect.isfunction(obj) and not name.startswith("_")
    ]


if __name__ == "__main__":
    modules = [opencv, skimage, rust]
    functions = list(
        itertools.chain.from_iterable(
            get_public_functions(module) for module in modules
        )
    )
    image_sizes = [(2**i, 2**i) for i in range(4, 11)]
    scale_factors = [2**i for i in range(1, 5)]

    runner = pyperf.Runner()

    for func, image_size, scale_factor in itertools.product(
        functions, image_sizes, scale_factors
    ):
        image = prepare_image(*image_size)
        runner.bench_func(
            f"{func.__module__} {func.__name__} {image_size[0]}*{image_size[1]} x{scale_factor}",
            func,
            image,
            scale_factor,
            metadata=dict(module=func.__module__,func=func.__name__,image_size=f"{image_size[0]}*{image_size[1]}", scale_factor=scale_factor),
        )
