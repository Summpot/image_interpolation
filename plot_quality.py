import json
from collections import defaultdict
import numpy as np

# 读取quality.json文件
try:
    with open('quality.json', 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print("Error: quality.json file not found.")
    exit(1)
except json.JSONDecodeError:
    print("Error: quality.json is not a valid JSON file.")
    exit(1)

# 定义所有指标
metrics = [
    'mse',
    'psnr',
    'ssim',
    'average_hash_diff',
    'phash_diff',
    'dhash_diff',
    'whash_diff'
]

# 过滤无效数据：剔除mse为0或psnr为null的记录
filtered_data = [
    entry for entry in data
    if entry['mse'] != 0 and entry['psnr'] is not None
]

if not filtered_data:
    print("Error: No valid data after filtering (all entries have mse=0 or psnr=null).")
    exit(1)

# 数据分组：按dataset_name、interp_algorithm和scale_factor收集所有指标值
grouped_data = defaultdict(lambda: defaultdict(list))
for entry in filtered_data:
    key = (entry['dataset_name'], entry['interp_algorithm'], entry['scale_factor'])
    for metric in metrics:
        grouped_data[key][metric].append(entry[metric])

# 构建JSON数据
chart_data = {}
datasets = sorted(set(entry['dataset_name'] for entry in filtered_data))

for dataset in datasets:
    labels = []
    metric_values = {metric: [] for metric in metrics}
    y_axis_ranges = {metric: {} for metric in metrics}  # 存储Y轴范围建议
    
    # 收集数据
    for algo in sorted(set(entry['interp_algorithm'] for entry in filtered_data)):
        for scale in sorted(set(entry['scale_factor'] for entry in filtered_data)):
            key = (dataset, algo, scale)
            if key in grouped_data:
                labels.append(f"{algo}@{scale}x")
                for metric in metrics:
                    metric_values[metric].append(grouped_data[key][metric])
    
    # 计算Y轴范围建议
    for metric in metrics:
        all_values = []
        for values in metric_values[metric]:
            all_values.extend(values)
        if all_values:  
            min_val = min(all_values)
            max_val = max(all_values)
            # 添加缓冲区（10%范围扩展）以改善可视化
            range_buffer = (max_val - min_val) * 0.1 if max_val > min_val else 0.1
            y_axis_ranges[metric] = {
                "min": min_val - range_buffer,
                "max": max_val + range_buffer
            }
    
    # 如果有数据，添加到chart_data
    if labels:
        chart_data[dataset] = {
            "labels": labels,
            **{f"{metric}_values": metric_values[metric] for metric in metrics},
            "y_axis_ranges": y_axis_ranges
        }

# 保存为chart_data.json
with open('chart_data.json', 'w') as f:
    json.dump(chart_data, f, indent=2)

print("Chart data for boxplots with original metrics and Y-axis ranges has been saved to chart_data.json")