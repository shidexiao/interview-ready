# 方法1: 排序法
def find_kth_largest_sort(nums, k):
    nums.sort()
    return nums[-k]

# 方法2: 快速选择
def find_kth_largest_quickselect(nums, k):
    def partition(left, right, pivot_index):
        pivot = nums[pivot_index]
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
        store_index = left
        for i in range(left, right):
            if nums[i] < pivot:
                nums[store_index], nums[i] = nums[i], nums[store_index]
                store_index += 1
        nums[right], nums[store_index] = nums[store_index], nums[right]
        return store_index

    def select(left, right, k_smallest):
        if left == right:
            return nums[left]
        pivot_index = partition(left, right, (left + right) // 2)
        if k_smallest == pivot_index:
            return nums[k_smallest]
        elif k_smallest < pivot_index:
            return select(left, pivot_index - 1, k_smallest)
        else:
            return select(pivot_index + 1, right, k_smallest)

    return select(0, len(nums) - 1, len(nums) - k)

# 方法3: 堆排序法
import heapq

def find_kth_largest_heap(nums, k):
    return heapq.nlargest(k, nums)[-1]


if __name__ == '__main__':
    # 测试
    nums = [3, 2, 1, 5, 6, 4]
    k = 2
    print(find_kth_largest_sort(nums.copy(), k))        # 输出: 5
    print(find_kth_largest_quickselect(nums.copy(), k)) # 输出: 5
    print(find_kth_largest_heap(nums.copy(), k))        # 输出: 5