import json
from collections import defaultdict

# 读取time.json和memory.json
try:
    with open('time.json', 'r') as f:
        time_data = json.load(f)
    with open('memory.json', 'r') as f:
        memory_data = json.load(f)
except FileNotFoundError as e:
    print(f"Error: {e.filename} not found.")
    exit(1)
except json.JSONDecodeError:
    print("Error: Invalid JSON format in input file.")
    exit(1)

# 提取时间和内存数据，仅处理包含values的run
def extract_data(data, is_time=True):
    grouped_data = defaultdict(lambda: defaultdict(list))
    for benchmark in data['benchmarks']:
        func = benchmark['metadata']['func']
        image_size = benchmark['metadata']['image_size']
        key = (func, image_size)
        for run in benchmark['runs']:
            # 仅收集包含values且非空的run
            if 'values' in run and run['values']:
                if is_time:
                    # 运行时间（毫秒）
                    grouped_data[key]['time'].extend(run['values'])
                else:
                    # 内存消耗（字节）
                    grouped_data[key]['memory'].extend(run['values'])
    return grouped_data

# 处理时间和内存数据
time_grouped = extract_data(time_data, is_time=True)
memory_grouped = extract_data(memory_data, is_time=False)

# 构建JSON数据
chart_data = {}
metrics = ['time', 'memory']

# 获取所有func和image_size组合
funcs = sorted(set(func for (func, _) in time_grouped.keys() | memory_grouped.keys()))
image_sizes = sorted(set(size for (_, size) in time_grouped.keys() | memory_grouped.keys()))

# 为每个func生成数据
for func in funcs:
    labels = []
    metric_values = {metric: [] for metric in metrics}
    
    # 收集数据
    for size in image_sizes:
        key = (func, size)
        label = f"{func}@{size}"
        labels.append(label)
        
        # 时间数据：仅有效数据点
        time_values = time_grouped[key]['time'] if key in time_grouped else []
        metric_values['time'].append(time_values)
        
        # 内存数据：仅有效数据点
        memory_values = memory_grouped[key]['memory'] if key in memory_grouped else []
        metric_values['memory'].append(memory_values)
    
    # 添加到chart_data（仅当有数据时）
    if labels and (any(metric_values['time']) or any(metric_values['memory'])):
        chart_data[func] = {
            "labels": labels,
            **{f"{metric}_values": metric_values[metric] for metric in metrics}
        }

# 保存为chart_data_performance.json
with open('chart_data_performance.json', 'w') as f:
    json.dump(chart_data, f, indent=2)

print("Chart data for time and memory boxplots (using values for memory, skipping runs without values) has been saved to chart_data_performance.json")