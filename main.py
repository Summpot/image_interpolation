import numpy as np
import image_interpolation
import pyperf


def prepare_image(size=(64, 64)):
    """准备一个用于基准测试的随机图像数组"""
    return np.random.rand(*size).astype(np.float64)


def bench_nearest_neighbor(runner, image, scale_factor):
    """基准测试最邻近插值"""
    runner.bench_func(
        "nearest_neighbor_interpolate",
        image_interpolation.py_nearest_neighbor_interpolate,
        args=(image, scale_factor),
    )


def bench_bilinear(runner, image, scale_factor):
    """基准测试双线性插值"""
    runner.bench_func(
        "bilinear_interpolate",
        image_interpolation.py_bilinear_interpolate,
        args=(image, scale_factor),
    )


if __name__ == "__main__":
    runner = pyperf.Runner()
    image = prepare_image()
    scale_factor = 2.0

    print(f"基准测试图像大小: {image.shape}, 缩放倍率: {scale_factor}")

    bench_nearest_neighbor(runner, image, scale_factor)
    bench_bilinear(runner, image, scale_factor)

    # 你也可以添加更多配置和信息到基准测试结果中，例如：
    runner.metadata["image_size"] = str(image.shape)
    runner.metadata["scale_factor"] = str(scale_factor)
    runner.metadata["interpolation_methods"] = "nearest_neighbor, bilinear"

    runner.run_cli()
