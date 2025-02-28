import functools
import itertools
from datasets import load_dataset
from PIL import Image
from PIL import ImageFilter
import numpy as np
from src import opencv
from src import skimage
from src import rust
from skimage.metrics import (
    structural_similarity,
    peak_signal_noise_ratio,
    mean_squared_error,
)
import imagehash
import matplotlib.pyplot as plt
import pandas as pd

from utils import get_public_functions


def add_index(data, index):
    data["index"] = index
    return data


def interpolate(data, image_column):
    image = data["image"]


def calculate_metrics(original_image, interpolated_image):
    original_array = np.array(original_image)
    interpolated_array = np.array(interpolated_image)

    # 确保图像是灰度图，如果不是则转换为灰度图以计算 imagehash
    original_gray = original_image.convert("L")
    interpolated_gray = interpolated_image.convert("L")

    mse = mean_squared_error(original_array, interpolated_array)
    psnr = peak_signal_noise_ratio(original_array, interpolated_array)
    ssim = structural_similarity(
        original_array,
        interpolated_array,
        channel_axis=-1,
        data_range=original_array.max() - original_array.min(),
    )  # 通道轴通常是最后一个轴
    hash_original = imagehash.average_hash(original_gray)
    hash_interpolated = imagehash.average_hash(interpolated_gray)
    hamming_distance = hash_original - hash_interpolated  # 计算汉明距离

    return {
        "MSE": mse,
        "PSNR": psnr,
        "SSIM": ssim,
        "ImageHash Hamming Distance": hamming_distance,
    }


def interpolate_and_evaluate(
    dataset_name, image_column, interpolation_algorithms, scaling_factors, num_images=5
):
    for data in dataset.to_iterable_dataset():
        example = data["index"]
        original_image = example[image_column]
        if not isinstance(original_image, Image.Image):  # 确保是 PIL Image 对象
            original_image = Image.fromarray(original_image)  # 如果是 NumPy 数组则转换

        for algorithm in interpolation_algorithms:
            for scale_factor in scaling_factors:
                width, height = original_image.size
                new_size = (int(width * scale_factor), int(height * scale_factor))

                # 先缩小再放大以模拟先降低分辨率再恢复的情况，或者直接放大来模拟超分辨率
                if scale_factor < 1:
                    # 缩小然后放大
                    temp_image = original_image.resize(new_size, resample=algorithm)
                    interpolated_image = temp_image.resize(
                        original_image.size, resample=algorithm
                    )  # 放大回原始尺寸
                else:
                    # 直接放大
                    interpolated_image = original_image.resize(
                        new_size, resample=algorithm
                    )
                    interpolated_image = interpolated_image.resize(
                        original_image.size, resample=algorithm
                    )  # 放大后也resize回原尺寸进行比较

                metrics = calculate_metrics(original_image, interpolated_image)
                results_list.append(
                    {
                        "Dataset": dataset_name,
                        "Image Index": idx,
                        "Algorithm": algorithm.name
                        if hasattr(algorithm, "name")
                        else str(algorithm),  # 获取算法名称或字符串表示
                        "Scaling Factor": scale_factor,
                        "Original Size": original_image.size,
                        "Interpolated Size": interpolated_image.size,
                        **metrics,
                    }
                )

    results_df = pd.DataFrame(results_list)
    return results_df


if __name__ == "__main__":
    datasets_to_test = [
        {"name": "uoft-cs/cifar10", "image_col": "img"},
        {"name": "ylecun/mnist", "image_col": "image"},
        {
            "name": "AI-Lab-Makerere/beans",
            "image_col": "image",
        },
        {"name": "blanchon/UC_Merced", "image_col": "image"},
    ]
    modules = [rust]
    functions = list(
        itertools.chain.from_iterable(
            get_public_functions(module) for module in modules
        )
    )
    image_sizes = [(2**i, 2**i) for i in range(4, 6)]
    scale_factors = [2**i for i in range(1, 2)]

    all_results = []
    for dataset_info in datasets_to_test:
        dataset_name = dataset_info["name"]
        image_column = dataset_info["image_col"]
        dataset = (
            load_dataset(dataset_name, split="train")
            .map(add_index, with_indices=True)
            .map(functools.partial(interpolate, image_column=image_column))
        )
        results = interpolate_and_evaluate(
            dataset_name,
            image_column,
            interpolation_algorithms_to_test,
            scaling_factors_to_test,
            num_images=5,
        )  # 减少num_images加快测试
        all_results.append(results)

    # 合并所有数据集的结果
    final_results_df = pd.concat(all_results, ignore_index=True)

    # 打印结果 DataFrame
    print(final_results_df)

    # 可选: 保存结果到 CSV 文件
    # final_results_df.to_csv("interpolation_results.csv", index=False)

    # 可选: 结果可视化 (例如，针对不同算法和缩放因子的性能比较)
    # 可以使用 matplotlib 或 seaborn 进行数据可视化
    # 例如，绘制箱线图来比较不同算法在不同指标上的表现
    # 例如，绘制柱状图来比较不同缩放因子对性能的影响
