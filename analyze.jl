using JSON

"""
导入 quality.json 文件，按 dataset_name 分组，
并将每个数据集的数据转换为包含 "x" (组合字符串数组)
和 "data" (mse 值数组) 的字典。

Args:
    filename::String: quality.json 文件的路径。

Returns:
    Dict{String, Dict{String, Vector{Any}}}: 转换后的字典。
    键为 dataset_name，值为一个字典，该字典包含键 "x" 和 "data"，
    它们的值分别是组合字符串数组和对应的 mse 值数组。
"""
function process_quality_data(filename::String)
    # 检查文件是否存在
    if !isfile(filename)
        error("错误：文件未找到： $filename")
    end

    # 读取并解析 JSON 文件
    try
        data = JSON.parsefile(filename)

        # 确保数据是数组且数组元素是字典
        if !isa(data, AbstractArray) || !all(isa.(data, AbstractDict))
             error("错误：JSON 数据格式不符合预期，应为对象数组。")
        end

        # 用于存储最终结果的字典
        # 外层 Dict: key=dataset_name, value=内层 Dict
        # 内层 Dict: key="x" 或 "data", value=Vector{Any}
        output_dict = Dict{String, Dict{String, Vector{Any}}}()

        # 遍历原始数据中的每一项
        for item in data
            # 提取所需的字段
            dataset_name = get(item, "dataset_name", nothing)
            interp_algorithm = get(item, "interp_algorithm", nothing)
            scale_factor = get(item, "scale_factor", nothing)
            mse_value = get(item, "mse", nothing) # MSE值可能是浮点数或其他类型，使用 Any

            # 检查必需字段是否存在且非空
            if isnothing(dataset_name) || isnothing(interp_algorithm) || isnothing(scale_factor) || isnothing(mse_value)
                 @warn "跳过包含缺失必需字段的条目: $item"
                 continue
            end

            # 构建 "x" 的值 (interp_algorithm_scalescale_factor)
            x_value = string(interp_algorithm, "_scale", scale_factor)

            # 获取或创建当前 dataset_name 的内层字典
            if !haskey(output_dict, dataset_name)
                # 如果 dataset_name 是新的，初始化内层字典及其 "x" 和 "data" 数组
                output_dict[dataset_name] = Dict{String, Vector{Any}}(
                    "x" => [],
                    "data" => []
                )
            end

            # 将当前的 x_value 和 mse_value 添加到对应 dataset_name 的数组中
            push!(output_dict[dataset_name]["x"], x_value)
            push!(output_dict[dataset_name]["data"], mse_value)
        end

        return output_dict

    catch e
        error("处理 JSON 文件时发生错误： $e")
    end
end

# 调用函数处理文件
processed_data = process_quality_data("quality.json")

# 指定输出文件名为 mse.json
output_filename = "mse.json"

# 将处理后的数据保存到 mse.json 文件
try
    open(output_filename, "w") do f
        JSON.print(f, processed_data, 2) # 使用 2 进行漂亮的缩进
    end
    println("数据已成功保存到 $output_filename")
catch e
    println("保存文件时发生错误： $e")
end