import itertools
import cv2
from datasets import load_dataset
import numpy as np
from src import rust
from skimage.metrics import (
    structural_similarity,
    peak_signal_noise_ratio,
    mean_squared_error,
)
from PIL import Image
import imagehash
import polars as pl
from rich.progress import track


from utils import get_public_functions


def convert_to_grayscale(image):
    if image.ndim == 3:
        if image.shape[-1] == 1:
            image = np.squeeze(image, axis=-1)
        elif image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if (
        image.dtype == np.float32
        or image.dtype == np.float64
        or image.dtype == np.float16
    ):
        image = (image * 255).astype(np.uint8)
    return image


def calculate_metrics(original_image, interpolated_image):
    original_image_np = np.array(original_image)
    interpolated_image_np = np.array(interpolated_image)

    # Ensure interpolated image matches original image dimensions
    if interpolated_image_np.shape != original_image_np.shape:
        interpolated_image_np = cv2.resize(
            interpolated_image_np,
            (original_image_np.shape[1], original_image_np.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    original_gray = convert_to_grayscale(original_image_np)
    interpolated_gray = convert_to_grayscale(interpolated_image_np)

    mse = mean_squared_error(original_image_np, interpolated_image_np)
    psnr = peak_signal_noise_ratio(original_image_np, interpolated_image_np)
    ssim = structural_similarity(original_gray, interpolated_gray)

    hash_a = imagehash.average_hash(original_image) - imagehash.average_hash(
        interpolated_image
    )
    hash_p = imagehash.phash(original_image) - imagehash.phash(interpolated_image)
    hash_d = imagehash.dhash(original_image) - imagehash.dhash(interpolated_image)
    hash_w = imagehash.whash(original_image) - imagehash.whash(interpolated_image)

    metrics_dict = {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "average_hash_diff": int(hash_a),
        "phash_diff": int(hash_p),
        "dhash_diff": int(hash_d),
        "whash_diff": int(hash_w),
    }
    return metrics_dict


def downsample_image(image, scale_factor):
    """Downsample an image using INTER_AREA."""
    new_height = int(image.shape[0] / scale_factor)
    new_width = int(image.shape[1] / scale_factor)
    if new_height == 0 or new_width == 0:
        raise ValueError("Downsampled dimensions are too small.")
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    datasets_config = [
        {
            "name": "AI-Lab-Makerere/beans",
            "image_col": "image",
            "label_col": "labels",
        },
        {
            "name": "blanchon/UC_Merced",
            "image_col": "image",
            "label_col": "label",
        },
        {
            "name": "keremberke/chest-xray-classification",
            "image_col": "image",
            "label_col": "labels",
        },
    ]
    modules = [rust]
    functions = list(
        itertools.chain.from_iterable(
            (get_public_functions(module) for module in modules)
        )
    )
    functions = [(func.__name__, func) for func in functions]
    scale_factors = [2**i for i in range(1, 3)]

    test_data_records = []

    for dataset_info in datasets_config:
        dataset_name = dataset_info["name"]
        image_col = dataset_info["image_col"]
        label_col = dataset_info["label_col"]

        try:
            dataset = load_dataset(dataset_name, split="train")
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            continue

        # Randomly select 1000 indices (or all if dataset is smaller)
        num_samples = min(len(dataset), 1000)
        if num_samples == 0:
            print(f"Dataset {dataset_name} is empty, skipping.")
            continue
        indices = np.random.choice(len(dataset), size=num_samples, replace=False)

        label_names = dataset.features[label_col].names
        for idx in track(indices, description=dataset_name):
            example = dataset[int(idx)]  # Convert idx to int for dataset access
            original_image = example[image_col]
            if not isinstance(original_image, Image.Image):
                original_image = Image.fromarray(original_image)
            original_image_np = np.array(original_image)

            # Handle grayscale images
            if original_image_np.ndim == 2:
                original_image_np = np.expand_dims(original_image_np, axis=2)

            label_name = (
                label_names[example[label_col]] if label_col in example else "unknown"
            )

            for interp_name, interp_func in functions:
                for scale_factor in scale_factors:
                    try:
                        # Downsample the original image
                        downsampled_image_np = downsample_image(
                            original_image_np, scale_factor
                        )

                        # Apply interpolation to upsample back to original size
                        interpolated_image_np = interp_func(
                            downsampled_image_np, scale_factor
                        )

                        # Ensure interpolated image is in correct format
                        if (
                            interpolated_image_np.ndim == 3
                            and interpolated_image_np.shape[-1] == 1
                        ):
                            interpolated_image_np = np.squeeze(
                                interpolated_image_np, axis=2
                            )

                        interpolated_image = Image.fromarray(interpolated_image_np)
                        metrics = calculate_metrics(original_image, interpolated_image)

                        record = {
                            "dataset_name": dataset_name,
                            "label_name": label_name,
                            "interp_algorithm": interp_name,
                            "scale_factor": scale_factor,
                            "mse": metrics["mse"],
                            "psnr": metrics["psnr"],
                            "ssim": metrics["ssim"],
                            "average_hash_diff": metrics["average_hash_diff"],
                            "phash_diff": metrics["phash_diff"],
                            "dhash_diff": metrics["dhash_diff"],
                            "whash_diff": metrics["whash_diff"],
                        }

                        test_data_records.append(record)
                    except Exception as e:
                        print(
                            f"Error processing {interp_name} with scale {scale_factor}: {e}"
                        )
                        continue

    df_results = pl.DataFrame(test_data_records)
    df_results.write_json("quality.json")
