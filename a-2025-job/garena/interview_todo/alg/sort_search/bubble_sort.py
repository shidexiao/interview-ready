from typing import List

arr = [10,17,50,7,30,24,27,45,15,5,38,21]

def bubble_sort(arr:List[int]):
    if len(arr)<=1:
        return arr
    for i in range(len(arr)):
        for j in range(len(arr)-i-1):
            if arr[j]>arr[j+1]:
                arr[j],arr[j+1] = arr[j+1],arr[j]
                swap = True
        if not swap:
            break
    return arr



