# python实现快速排序
# 稳定性：所有相等的数经过某种排序后，仍能保持它们在排序之前的相对次序，就称这种排序方法是稳定的。快排是不稳定的
# 快排的时间复杂度：最优是O(nlogn)，最坏是O(n^2)

# 快速排序，从无序队列中挑取一个元素，把无序队列分割成独立的两部分，其中一部分的所有数据都比另外一部分的所有数据都要小，
# 然后再按此方法对这两部分数据分别进行快速排序，整个排序过程可以递归进行，以此达到整个数据变成有序序列。
# 简单来说：挑元素、划分组、分组重复前两步

def quick_sort(start, end, array):
    if start >= end:    # 直接返回，不用执行下面了
        return
    # 定义两个游标，分别指向开始和末尾位置
    left = start
    right = end
    # 把0位置的数据认为是中间值
    mid = array[left]
    """中间值的选择对快排的影响至关重要，当选取的中间是最大或最小的值时，此时是最坏的时间复杂度O(n^2)"""
    while left < right:
        # 让右边的游标向左移动，目的是找到小于mid的值，放在left游标位置
        while left < right and array[right] >= mid:
            right -= 1
        array[left] = array[right]
        # 让左边的游标向右移动，目的是找到大于mid的值，放在right游标的位置
        while left < right and array[left] < mid:
            left += 1
        array[right] = array[left]

    array[left] = mid    # 当left=right时while结束，结束后，把mid放到中间位置，left=right

    quick_sort(start, left-1, array)    # 递归处理左边数据
    quick_sort(left+1, end, array)      # 递归处理右边数据
    return array


ls = [56, 26, 44, 17, 77, 31, 93, 55]
start = 0
end = len(ls)-1
new_list = quick_sort(start, end, ls)
print(new_list)