def calculate_bowling_score(rolls):
    """
    计算十瓶保龄球比赛总得分
    :param rolls: 击球序列，如[10, 7, 3, 9, 0, ...]
    :return: 总得分
    """
    total = 0
    roll_index = 0  # 当前击球位置

    # 处理前9个球局
    for frame in range(10):
        if frame < 9:  # 前9局
            if rolls[roll_index] == 10:  # 全中(strike)
                total += 10 + rolls[roll_index + 1] + rolls[roll_index + 2]
                roll_index += 1
            else:
                if rolls[roll_index] + rolls[roll_index + 1] == 10:  # 补中(spare)
                    total += 10 + rolls[roll_index + 2]
                else:  # 常规
                    total += rolls[roll_index] + rolls[roll_index + 1]
                roll_index += 2
        else:  # 第10局特殊处理
            if rolls[roll_index] == 10:  # 第一球全中
                total += 10 + rolls[roll_index + 1] + rolls[roll_index + 2]
            elif rolls[roll_index] + rolls[roll_index + 1] == 10:  # 补中
                total += 10 + rolls[roll_index + 2]
            else:  # 常规
                total += rolls[roll_index] + rolls[roll_index + 1]

    return total


# 测试案例
if __name__ == "__main__":
    # 测试案例集 (击球序列, 预期得分)
    test_cases = [
        # 全零局
        ([0] * 20, 0),
        # 全补中 (每次第一球9，第二球1)
        ([9, 1] * 10 + [9], 190),  # 10局补中+奖励球
        # 全全中
        ([10] * 12, 300),  # 12连全中(10局×30分)
        # 混合案例1 (参考问题描述中的例子)
        ([10, 7, 3, 9, 0, 10, 0, 8, 8, 2, 0, 6, 10, 10, 10, 8, 1], 167),
        # 混合案例2
        ([5, 5, 3, 7, 10, 10, 2, 8, 5, 4, 9, 0, 8, 2, 7, 3, 10, 10, 10], 183),
        # 第10局常规
        ([3, 4] * 10, 70),
    ]

    for i, (rolls, expected) in enumerate(test_cases):
        result = calculate_bowling_score(rolls)
        print(f"测试案例 {i + 1}: {'通过' if result == expected else '失败'}")
        print(f"击球序列: {rolls}")
        print(f"计算得分: {result} (预期: {expected})\n")