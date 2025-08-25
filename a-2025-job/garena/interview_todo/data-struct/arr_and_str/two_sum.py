# 示例：两数之和
'''
给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出和为目标值的那 两个整数，并返回它们的数组下标。
你可以假设每种输入只会对应一个答案，但是同一个元素不能使用两次。

哈希表优化（时间复杂度 O(n)）
核心思想是：
在遍历数组时，用哈希表记录“之前出现过的数字及其下标”
对于当前数字 x，判断 target - x 是否在哈希表中出现过
如果有，说明找到一对符合条件的数
'''


def twoSum(nums, target):
    hashmap = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in hashmap:
            return [hashmap[complement], i]
        hashmap[num] = i
    return []

class Model():
    a: str
    b: str
    c='a'
    def __init__(self):
        self.x = 'x'