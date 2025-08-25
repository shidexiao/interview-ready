def flatten_dict(nested_dict, parent_key='', sep='_'):
    """
    递归地将嵌套字典转换为单层字典，嵌套的键用 '_' 连接。
    :param nested_dict: 需要扁平化的嵌套字典
    :param parent_key: 父键的前缀（用于递归时传递）
    :param sep: 键名之间的分隔符
    :return: 扁平化后的字典
    """
    flattened = {}
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # 递归调用 flatten_dict 获取子字典的扁平化结果
            nested_flat = flatten_dict(v, new_key, sep=sep)
            for nk, nv in nested_flat.items():
                # 手动逐个添加到扁平化字典中
                flattened[nk] = nv
        else:
            # 如果不是字典类型，直接赋值到扁平化字典中
            flattened[new_key] = v
    return flattened


# 示例字典
data = {
    "code": "00",
    "swift_number": "3011108_20240912101315_4199276BA",
    "ApplyLoanStr": {
        "d7": {
            "id": {
                "caoff": {
                    "orgnum": "1",
                    "allnum": "1"
                },
                "oth": {
                    "orgnum": "1",
                    "allnum": "1"
                },
                "bank": {
                    "week_allnum": "5",
                    "selfnum": "1",
                    "orgnum": "2",
                    "night_allnum": "5",
                    "ret_allnum": "1",
                    "week_orgnum": "6",
                    "tra_allnum": "1",
                    "night_orgnum": "6",
                    "tra_orgnum": "3",
                    "ret_orgnum": "4",
                    "allnum": "2"
                }
            }
        }
    }
}

# 扁平化字典
flattened_data = flatten_dict(data['ApplyLoanStr'],"ApplyLoanStr")
print(flattened_data)
