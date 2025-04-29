import itertools
import cv2
import os
import random
from datasets import load_dataset
import numpy as np
from src import rust
from PIL import Image
import polars as pl
from rich.progress import track
from utils import get_public_functions


def downsample_image(image, scale_factor):
    """Downsample an image using INTER_AREA."""
    new_height = int(image.shape[0] / scale_factor)
    new_width = int(image.shape[1] / scale_factor)
    if new_height == 0 or new_width == 0:
        raise ValueError("Downsampled dimensions are too small.")
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def convert_to_rgb(image):
    """Convert image to RGB format."""
    image_np = np.array(image)
    if image_np.ndim == 2:
        # Grayscale to RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[-1] == 1:
        # Single-channel to RGB
        image_np = np.squeeze(image_np, axis=-1)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[-1] == 3:
        # Ensure RGB
        if image_np.dtype in (np.float32, np.float64, np.float16):
            image_np = (image_np * 255).astype(np.uint8)
    return image_np


def save_image(image, path):
    """Save image to the specified path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image.save(path)


if __name__ == "__main__":
    datasets_config = [
        {
            "name": "uoft-cs/cifar10",
            "image_col": "img",
            "label_col": "label",
            "num_images": 8,  # Low-resolution, select 8 images
            "is_high_res": False,
            "subset": None,
        },
        {
            "name": "ylecun/mnist",
            "image_col": "image",
            "label_col": "label",
            "num_images": 8,  # Low-resolution, select 8 images
            "is_high_res": False,
            "subset": None,
        },
        {
            "name": "AI-Lab-Makerere/beans",
            "image_col": "image",
            "label_col": "labels",
            "num_images": 4,  # High-resolution, select 4 images
            "is_high_res": True,
            "subset": None,
        },
        {
            "name": "blanchon/UC_Merced",
            "image_col": "image",
            "label_col": "label",
            "num_images": 4,  # High-resolution, select 4 images
            "is_high_res": True,
            "subset": None,
        },
        {
            "name": "keremberke/chest-xray-classification",
            "image_col": "image",
            "label_col": "labels",
            "num_images": 4,  # High-resolution, select 4 images
            "is_high_res": True,
            "subset": "full",
        },
    ]
    modules = [rust]
    functions = list(
        itertools.chain.from_iterable(
            (get_public_functions(module) for module in modules)
        )
    )
    functions = [(f"{func.__name__}", func) for func in functions]
    scale_factors = [2, 4]
    image_records = []

    for dataset_info in datasets_config:
        dataset_name = dataset_info["name"]
        image_col = dataset_info["image_col"]
        label_col = dataset_info["label_col"]
        num_images = dataset_info["num_images"]
        is_high_res = dataset_info["is_high_res"]
        subset = dataset_info["subset"]

        try:
            dataset = load_dataset(dataset_name, subset, split="train")
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            continue

        # Randomly select images
        total_images = len(dataset)
        selected_indices = random.sample(
            range(total_images), min(num_images, total_images)
        )

        for idx, example_idx in track(
            enumerate(selected_indices), description=f"Processing {dataset_name}"
        ):
            example = dataset[example_idx]
            original_image = example[image_col]
            if not isinstance(original_image, Image.Image):
                original_image = Image.fromarray(original_image)
            original_image_np = convert_to_rgb(original_image)
            label_name = (
                dataset.features[label_col].names[example[label_col]]
                if label_col in example
                and hasattr(dataset.features[label_col], "names")
                else "unknown"
            )

            # Save original image
            original_path = (
                f"images/{dataset_name.replace('/', '_')}/original_{idx}.png"
            )
            save_image(original_image_np, original_path)

            # Prepare input image
            if is_high_res:
                # Downsample high-resolution images
                input_image_np = downsample_image(original_image_np, scale_factor=2)
                # 新增：保存降采样图像
                downsampled_path = (
                    f"images/{dataset_name.replace('/', '_')}/downsampled_2x_{idx}.png"
                )
                save_image(input_image_np, downsampled_path)
            else:
                # Use original image for low-resolution datasets
                input_image_np = original_image_np
                downsampled_path = None  # 低分辨率数据集无降采样图像

            for interp_name, interp_func in functions:
                for scale_factor in scale_factors:
                    # Perform interpolation
                    interpolated_image_np = interp_func(input_image_np, scale_factor)
                    if (
                        interpolated_image_np.ndim == 3
                        and interpolated_image_np.shape[-1] == 1
                    ):
                        interpolated_image_np = np.squeeze(
                            interpolated_image_np, axis=-1
                        )
                        interpolated_image_np = cv2.cvtColor(
                            interpolated_image_np, cv2.COLOR_GRAY2RGB
                        )

                    # Save interpolated image
                    interp_path = (
                        f"images/{dataset_name.replace('/', '_')}/"
                        f"{interp_name.replace('.', '_')}_{scale_factor}x_{idx}.png"
                    )
                    save_image(interpolated_image_np, interp_path)

                    # Record image information
                    record = {
                        "dataset_name": dataset_name,
                        "label_name": label_name,
                        "image_index": idx,
                        "original_path": original_path,
                        "interp_algorithm": interp_name,
                        "scale_factor": scale_factor,
                        "interp_path": interp_path,
                    }
                    # 新增：记录降采样路径（仅高分辨率数据集）
                    if is_high_res:
                        record["downsampled_path"] = downsampled_path
                    image_records.append(record)

    df_images = pl.DataFrame(image_records)
    df_images.write_json("subjective_images.json")
    print("Subjective image data saved to subjective_images.json")
