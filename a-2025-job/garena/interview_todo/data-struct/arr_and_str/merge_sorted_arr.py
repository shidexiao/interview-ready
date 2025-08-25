from typing import List
'''
解题思路（从后往前双指针 ✅ 推荐）
为什么从后往前合并？
因为 nums1 后面有空位，如果从前往后合并，会导致覆盖未处理的数据。
从末尾开始合并，可以直接把最大的数字放到 nums1 的末尾，不会覆盖。

步骤：
用三个指针：
p1 指向 nums1 中最后一个有效元素（m - 1）
p2 指向 nums2 最后一个元素（n - 1）
p 指向 nums1 的最后一个位置（m + n - 1）
比较 nums1[p1] 和 nums2[p2]，谁大就把谁放到 nums1[p]
继续向前移动指针
最后，如果 nums2 还有剩，全部拷贝进 nums1


'''

def merge(nums1: List[int], m: int, nums2: List[int], n: int) -> None:
    """
    合并 nums2 到 nums1，使 nums1 成为一个有序数组。
    注意：此函数为就地修改 nums1，不返回值。
    """
    p1 = m - 1  # nums1 有效部分的末尾
    p2 = n - 1  # nums2 的末尾
    p = m + n - 1  # nums1 的最后一位（总长度）

    while p1 >= 0 and p2 >= 0:
        if nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1

    # 如果 nums2 还有剩余，填充到 nums1 前面
    while p2 >= 0:
        nums1[p] = nums2[p2]
        p2 -= 1
        p -= 1
