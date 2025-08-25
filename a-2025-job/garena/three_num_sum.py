from typing import List

'''
输入: nums = [-1, 0, 1, 2, -1, -4]
输出: [[-1, -1, 2], [-1, 0, 1]]

解题思路（排序 + 双指针）
总体思路：
将数组排序（时间复杂度 O(n log n)）。
遍历数组，每次固定一个数 nums[i]，问题转化为在 nums[i+1:] 中找两个数，使得它们的和是 -nums[i]。
用双指针从两端向中间夹逼查找。
注意跳过重复的数字，以避免重复结果。
'''
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