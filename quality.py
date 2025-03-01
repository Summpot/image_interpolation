import itertools
from datasets import load_dataset
import numpy as np
from src import opencv
from src import skimage
from src import rust
from skimage.metrics import (
    structural_similarity,
    peak_signal_noise_ratio,
    mean_squared_error,
)
from PIL import Image
import imagehash
import polars as pl


from utils import downsample_image, get_public_functions


def mean_absolute_error(image1, image2):
    """
    计算两幅图像的平均绝对误差 (MAE)。

    Args:
        image1 (ndarray): 第一幅图像，NumPy 数组。
        image2 (ndarray): 第二幅图像，NumPy 数组。

    Returns:
        float: 两幅图像的 MAE 值。
    """
    # 1. 计算像素差值的绝对值
    abs_error = np.abs(image1 - image2)

    # 2. 计算绝对误差的平均值
    mae = np.mean(abs_error)
    return mae


def calculate_metrics(original_image, interpolated_image):
    downsampled_image = downsample_image(interpolated_image, original_image.shape)
    print(original_image.shape, interpolated_image.shape, downsampled_image.shape)
    original_gray = np.array(Image.fromarray(original_image).convert("L"))
    interpolated_gray = np.array(Image.fromarray(downsampled_image).convert("L"))
    mse = mean_squared_error(original_image, downsampled_image)
    psnr = peak_signal_noise_ratio(original_image, downsampled_image)
    ssim = structural_similarity(
        original_gray, interpolated_gray
    )  # SSIM 通常在灰度图上计算

    # 计算各种 ImageHash
    hash_a = imagehash.average_hash(
        Image.fromarray(original_image)
    ) - imagehash.average_hash(Image.fromarray(interpolated_image))
    hash_p = imagehash.phash(Image.fromarray(original_image)) - imagehash.phash(
        Image.fromarray(interpolated_image)
    )
    hash_d = imagehash.dhash(Image.fromarray(original_image)) - imagehash.dhash(
        Image.fromarray(interpolated_image)
    )
    hash_w = imagehash.whash(Image.fromarray(original_image)) - imagehash.whash(
        Image.fromarray(interpolated_image)
    )

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


if __name__ == "__main__":
    datasets_config = [
        {
            "name": "uoft-cs/cifar10",
            "image_col": "img",
            "label_col": "label",
        },  # 添加 label_col
        {
            "name": "ylecun/mnist",
            "image_col": "image",
            "label_col": "label",
        },  # 添加 label_col
        {
            "name": "AI-Lab-Makerere/beans",
            "image_col": "image",
            "label_col": "labels",  #  beans 数据集的标签列名是 labels
        },
        {
            "name": "blanchon/UC_Merced",
            "image_col": "image",
            "label_col": "label",
        },  # 添加 label_col
    ]
    modules = [rust]
    functions = list(
        itertools.chain.from_iterable(
            (get_public_functions(module) for module in modules)
        )
    )
    functions = [
        (f"{func.__module__.split('.')[1]}.{func.__name__}", func) for func in functions
    ]
    scale_factors = [2**i for i in range(1, 2)]

    test_data_records = []  # 存储测试数据的列表

    for dataset_info in datasets_config:
        dataset_name = dataset_info["name"]
        image_col = dataset_info["image_col"]
        label_col = dataset_info["label_col"]

        try:
            dataset = load_dataset(dataset_name, split="train")
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            continue  # 如果数据集加载失败，则跳过

        print(f"Processing dataset: {dataset_name}")
        label_names = dataset.features["label"].names
        for image_index, example in enumerate(dataset):
            original_image = example[image_col]  # 从 datasets 中获取的是 PIL Image 对象
            if not isinstance(original_image, Image.Image):  # 确保是 PIL Image 对象
                original_image = Image.fromarray(
                    original_image
                )  # 如果是 NumPy 数组或其他格式，尝试转换为 PIL Image
            original_image_np = np.array(original_image)  # 转换为 NumPy 数组用于计算

            label_name = (
                label_names[example[label_col]] if label_col in example else "unknown"
            )

            for interp_name, interp_func in functions:
                for scale_factor in scale_factors:
                    interpolated_image_np = interp_func(original_image_np, scale_factor)
                    print(
                        interp_name,
                        original_image_np.dtype,
                        interpolated_image_np.dtype,
                    )
                    metrics = calculate_metrics(
                        original_image_np, interpolated_image_np
                    )  # 计算指标

                    record = {
                        "dataset_name": dataset_name,
                        "label_name": label_name,
                        "image_index": image_index,
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
                    print(
                        f"  - Image {image_index}, Interp: {interp_name},Scale: {scale_factor}, MSE: {metrics['mse']:.4f}, PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}"
                    )  # 打印部分结果

        df_results = pl.DataFrame(test_data_records)
        print("\n--- 测试结果 (部分) ---")
        print(df_results.head())
        df_results.write_json("quality.json")
