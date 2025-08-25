from typing import List, TypeVar
T = TypeVar("T")

def quick_sort(arr: List[T]) ->List[T]:
    if len(arr)<=1:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x<pivot]
    mid = [x for x in arr if x==pivot]
    right = [x for x in arr if x>pivot]
    return quick_sort(left)+mid+quick_sort(right)


def quick_sort_inplace(arr:List[T], low:int, high:int):
    if low>=high:
        return
    pivot = arr[(low+high)//2]
    i,j = low,high
    while i<=j:
        while arr[i]<pivot:
            i +=1
        while arr[j]>pivot:
            j -= 1
        if i<=j:
            arr[i],arr[j] = arr[j],arr[i]
            i,j=i+1,j-1
    if low<j:
        quick_sort_inplace(arr,low,j)
    if i<high:
        quick_sort_inplace(arr,i,high)


if __name__ == '__main__':
    arr = [3, 6, 8, 10, 1, 2, 1]
    # quick_sort_inplace(arr, 0, len(arr) - 1)
    quick_sort(arr)
    print(arr)