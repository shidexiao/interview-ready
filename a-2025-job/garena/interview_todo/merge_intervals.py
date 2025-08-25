from typing import List

'''

合并区间
复杂度分析

时间复杂度：O(nlogn)，其中 n 为区间的数量。
除去排序的开销，我们只需要一次线性扫描，所以主要的时间开销是排序的 O(nlogn)。

空间复杂度：O(logn)，其中 n 为区间的数量。
这里计算的是存储答案之外，使用的额外空间。O(logn) 即为排序所需要的空间复杂度。



'''
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals:
            return []
        intervals.sort()
        res=[intervals[0]]
        for x,y in intervals[1:]:
            if x>res[-1][-1]:
                res.append([x,y])
            else:
                y=max(y,res[-1][-1])
                res[-1][-1]=y
        return res


if __name__ == "__main__":
    res = Solution().merge([[1,3],[2,6],[8,10],[15,18]])
    print(res)