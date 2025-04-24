import json
import matplotlib.pyplot as plt
import pandas as pd

# 读取 JSON 文件
with open('time.json', 'r') as file:
    data = json.load(file)

# 提取 benchmarks 数据
benchmarks = data['benchmarks']

# 创建一个空的 DataFrame 来存储结果
results = []

# 遍历每个 benchmark
for benchmark in benchmarks:
    metadata = benchmark['metadata']
    runs = benchmark['runs']

    # 提取 benchmark 的元数据
    func = metadata['func']
    image_size = metadata['image_size']
    module = metadata['module']
    name = metadata['name']

    # 遍历每个 run
    for run in runs:
        run_metadata = run['metadata']

        # 检查 run 是否包含 values
        if 'values' in run:
            values = run['values']

            # 提取 run 的元数据
            date = run_metadata['date']
            duration = run_metadata['duration']

            # 将结果添加到 DataFrame
            for value in values:
                results.append({
                    'func': func,
                    'image_size': image_size,
                    'module': module,
                    'name': name,
                    'date': date,
                    'duration': duration,
                    'value': value
                })

# 将结果转换为 DataFrame
df = pd.DataFrame(results)

# 绘制图表
plt.figure(figsize=(12, 6))

# 按模块和函数分组，绘制箱线图
df.boxplot(column='value', by=['module', 'func'], grid=False)

# 设置图表标题和标签
plt.title('Benchmark Results')
plt.suptitle('')
plt.xlabel('Module and Function')
plt.ylabel('Value')

# 保存图表为文件
plt.savefig('benchmark_results.png')

# 关闭图表
plt.close()
