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
from utils import downsample_image, get_public_functions

def convert_to_grayscale(image):
    if image.ndim == 3:
        if image.shape[-1] == 1:
            image = np.squeeze(image, axis=-1)
        elif image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.dtype in (np.float32, np.float64, np.float16):
        image = (image * 255).astype(np.uint8)
    return image

def save_image(image, path):
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
            "num_images": 8  # 低分辨率，选8张
        },
        {
            "name": "AI-Lab-Makerere/beans",
            "image_col": "image",
            "label_col": "labels",
            "num_images": 4  # 高分辨率，选4张
        },
        {
            "name": "ylecun/mnist",
            "image_col": "image",
            "label_col": "label",
            "num_images": 8  # 低分辨率，选8张
        },
        {
            "name": "blanchon/UC_Merced",
            "image_col": "image",
            "label_col": "label",
            "num_images": 4  # 高分辨率，选4张
        },
    ]
    modules = [rust]
    functions = list(itertools.chain.from_iterable(
        (get_public_functions(module) for module in modules)
    ))
    functions = [
        (f"{func.__module__.split('.')[1]}.{func.__name__}", func) for func in functions
    ]
    scale_factors = [2, 4]
    image_records = []

    for dataset_info in datasets_config:
        dataset_name = dataset_info["name"]
        image_col = dataset_info["image_col"]
        label_col = dataset_info["label_col"]
        num_images = dataset_info["num_images"]

        try:
            dataset = load_dataset(dataset_name, split="train")
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            continue

        # 随机选择图片
        total_images = len(dataset)
        selected_indices = random.sample(range(total_images), min(num_images, total_images))
        
        for idx, example_idx in track(
            enumerate(selected_indices), description=f"Processing {dataset_name}"
        ):
            example = dataset[example_idx]
            original_image = example[image_col]
            if not isinstance(original_image, Image.Image):
                original_image = Image.fromarray(original_image)
            original_image_np = np.array(original_image)
            if original_image_np.ndim == 2:
                original_image_np = np.expand_dims(original_image_np, axis=2)
            label_name = (
                dataset.features[label_col].names[example[label_col]]
                if label_col in example else "unknown"
            )

            # 保存原始图片
            original_path = f"images/{dataset_name.replace('/', '_')}/original_{idx}.png"
            save_image(original_image, original_path)

            for interp_name, interp_func in functions:
                for scale_factor in scale_factors:
                    # 插值
                    interpolated_image_np = interp_func(original_image_np, scale_factor)
                    if interpolated_image_np.ndim == 3 and interpolated_image_np.shape[-1] == 1:
                        interpolated_image_np = np.squeeze(interpolated_image_np, axis=2)
                    interpolated_image = Image.fromarray(interpolated_image_np)

                    # 下采样回原始分辨率
                    downsampled_image_np = downsample_image(
                        interpolated_image_np, original_image_np.shape
                    )
                    downsampled_image = Image.fromarray(downsampled_image_np)

                    # 保存插值后图片
                    interp_path = (
                        f"images/{dataset_name.replace('/', '_')}/"
                        f"{interp_name.replace('.', '_')}_{scale_factor}x_{idx}.png"
                    )
                    save_image(downsampled_image, interp_path)

                    # 记录
                    record = {
                        "dataset_name": dataset_name,
                        "label_name": label_name,
                        "image_index": idx,
                        "original_path": original_path,
                        "interp_algorithm": interp_name,
                        "scale_factor": scale_factor,
                        "interp_path": interp_path,
                    }
                    image_records.append(record)

    df_images = pl.DataFrame(image_records)
    df_images.write_json("subjective_images.json")
    print("Subjective image data saved to subjective_images.json")