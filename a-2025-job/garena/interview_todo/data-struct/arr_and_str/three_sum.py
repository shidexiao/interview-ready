'''
给你一个整数数组 nums，请你找出所有和为 0 的三元组（a + b + c == 0），并且不能包含重复的三元组。
你需要返回所有不重复的三元组列表，三元组中的元素需要是升序排列的（即满足 a <= b <= c）。

输入: nums = [-1, 0, 1, 2, -1, -4]
输出: [[-1, -1, 2], [-1, 0, 1]]

解题思路（排序 + 双指针）
总体思路：
将数组排序（时间复杂度 O(n log n)）。
遍历数组，每次固定一个数 nums[i]，问题转化为在 nums[i+1:] 中找两个数，使得它们的和是 -nums[i]。
用双指针从两端向中间夹逼查找。
注意跳过重复的数字，以避免重复结果。

时间复杂度分析：
排序：O(n log n)
双指针遍历：
外层循环 n 次，每次双指针最多移动 n 步
所以整体是 O(n²)
✅ 总时间复杂度：O(n²)

空间复杂度分析：
除了存储结果的列表，排序使用 O(log n) 的栈空间（Python 内建排序是 Timsort）。
使用了常数级指针变量。
✅ 总空间复杂度：O(1)（不计输出）

Python 的内建排序（如 list.sort() 和 sorted()）使用的是一种名为 Timsort 的排序算法。
它是一个 混合排序算法，结合了：
归并排序（Merge Sort） 的稳定性和效率
插入排序（Insertion Sort） 在小数组上的高效表现

Timsort 的核心思想
1. 对数组进行划分成“有序块”
Timsort 会先在列表中识别出天然有序的**“子数组”**（称为 runs）

然后利用归并排序将这些 runs 合并起来

2. 插入排序用于小规模的 run
如果某个 run 的大小小于一定阈值（通常为 32），就使用插入排序处理它
⏱ 插入排序对小规模数据非常快（常数时间优）

3. 归并排序用于合并 runs
将多个 runs 按一定策略合并起来，这一步稳定高效

项目	内容
算法名	Timsort
基础算法	归并排序 + 插入排序
复杂度	最优 O(n)，最坏 O(n log n)
是否稳定	✅ 是
使用范围	Python 的 list.sort()、sorted()
优势	适应性强，性能优秀，适合工程场景

'''
from typing import List


def three_sum(nums: List[int]) -> List[List[int]]:
    nums.sort()
    result: List[List[int]] = []
    n = len(nums)
    for i in range(n):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        left, right = i + 1, n - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            if s == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1
                left += 1
                right -= 1
            elif s < 0:
                left += 1
            else:
                right -= 1
    return result


if __name__ == '__main__':
    nums = [-1, 0, 1, 2, -1, -4]
    res = three_sum(nums)
    print(res)


