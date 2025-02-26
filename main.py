import numpy as np
import image_interpolation
import pyperf
import polars as pl
from datasets import load_dataset, Image


def bench_time(runner, func, dataset, label, index, image, scale_factor):
    runner.bench_func(
        f"{dataset}/{label}/{index}/{func.__name__}",
        func,
        image,
        scale_factor,
    )


def bench_other():
    pass


def run_benchmarks(runner, dataset, label, index, image, func, scale_factor):
    bench_time(runner, func, dataset, label, index, image, scale_factor)
    runner.metadata["image_size"] = str(image.shape)
    runner.metadata["scale_factor"] = str(scale_factor)
    runner.metadata["interpolation_method"] = func.__name__


def process(item):
    pass


def add_index(data, index):
    data["index"] = index
    return data


if __name__ == "__main__":
    runner = pyperf.Runner()
    dataset = load_dataset("blanchon/UC_Merced", split="train")
    dataset = dataset.map(add_index, with_indices=True)
    label_names = dataset.features["label"].names
    funcitions = [
        image_interpolation.nearest_neighbor,
        image_interpolation.bilinear,
    ]
    scale_factor = 2.0
    for data in dataset.to_iterable_dataset():
        image = np.array(data["image"], dtype=np.float64, ndmin=3)
        label = data["label"]
        index = data["index"]
        for func in funcitions:
            run_benchmarks(runner, "uc_merced", label, index, image, func, scale_factor)
