# 求数组中两两相加等于K的组合（Python实现）
""""""
"""例如：求数组中两两相加等于20的组合。
给定一个数组[1, 7, 17, 2, 6, 3, 14]
这个数组中满足条件的有两对：17+3=20, 6+14=20"""

"""分为两个步骤：
先采用快速排序对数组进行排序，时间复杂度为O(nlogn)。然后对排序的数组分别从前到后和从后到前进行遍历, 时间复杂度为O(n)。
假设从前到后遍历的下标为begin，从后到前遍历的下标为end。

当arr[begin] + arr[end] < k时，满足条件的数一定在[begin+1, end]之间；
当arr[begin] + arr[end] > k时，满足条件的数一定在[begin, end-1]之间；
当arr[begin] + arr[end] = k时，找到一组符合条件的数，剩下的组合一定在[begin-1, end-1]之间。
整个算法的时间复杂度为 O(nlogn)"""
def quick_sort(array, start, end):
    if start >= end:
        return
    left = start
    right = end
    mid = array[left]
    while left < right:
        while left < right and array[right] >= mid:
            right -= 1
        array[left] = array[right]

        while left < right and array[left] < mid:
            left +=1
        array[right] = array[left]

    array[left] = mid    # 当left=right时while结束，结束后，把mid放到中间位置，left=right
    quick_sort(array, start, left-1)
    quick_sort(array, left+1, end)
    return array


def find_sum(array, int_k):
    if int_k <= 0:
        return
    new_array = quick_sort(array=array, start=0, end=len(array)-1)
    result = []
    left = 0
    right = len(new_array)-1
    while left < right:
        if new_array[left] + new_array[right] < int_k:
            left += 1
        elif new_array[left] + new_array[right] > int_k:
            right -= 1
        else:
            list = []
            list.append(new_array[left])
            list.append(new_array[right])
            list_to_tuple = tuple(list)
            result.append(list_to_tuple)
            left += 1
            right -= 1
    return result


ls = [1, 7, 17, 2, 6, 3, 14]
new_li = quick_sort(ls, 0, 6)
print(new_li)

int_k = 20
result = find_sum(array=ls, int_k=int_k)
print(result)