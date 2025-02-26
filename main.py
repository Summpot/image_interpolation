import numpy as np
import image_interpolation
import pyperf
import polars as pl
from datasets import load_dataset


def bench_time(runner, func, dataset, label, image1, scale_factor):
    runner.bench_func(
        f"{dataset}/{label}/{func.__name__}",
        func,
        image1,
        scale_factor,
    )


def bench_other():
    pass


def run_benchmarks(dataset, label, image, func):
    runner = pyperf.Runner()
    bench_time(runner, func, dataset, label, image, scale_factor)
    runner.metadata["image_size"] = str(image.shape)
    runner.metadata["scale_factor"] = str(scale_factor)
    runner.metadata["interpolation_method"] = func.__name__


if __name__ == "__main__":
    dataset = load_dataset("blanchon/UC_Merced", split="train", streaming=True)
    funcitions = [
        image_interpolation.nearest_neighbor,
        image_interpolation.bilinear,
    ]
    scale_factor = 2.0
    for row in dataset:
        image = np.array(row["image"], dtype=np.float64, ndmin=2)
        label = row["label"]
        for func in funcitions:
            run_benchmarks(dataset, label, image, func)
