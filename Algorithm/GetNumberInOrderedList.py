# python使用二分法实现在一个有序列表中查找指定的元素
# 二分法是一种快速查找的方法，时间复杂度低，逻辑简单易懂，总的来说就是不断的除以2除以2...
""""""
"""例如需要查找 有序list 里面的某个关键字key的位置，那么首先确认list的中位数mid，下面分为三种情况：
如果 list[mid] < key,说明key 在中位数的 右边；
如果 list[mid] > key,说明key 在中位数的 左边；
如果 list[mid] = key,说明key 在中位数的中间；
范围每次缩小一半，写个while的死循环直到找到为止。
二分法查找非常快且常用，但是唯一要求是要求数组是有序的"""
def get_number_in_ordered_list(array, key):    # 这个list必须是有序的
    if not isinstance(array, list):
        return 0
    start = 0
    end = len(array) - 1
    if key in array:
        while True:     # 构建一个死循环，直至找到key值
            mid = int((start + end) / 2)
            if array[mid] < key:
                start = mid + 1
            elif array[mid] > key:
                end = mid -1
            else:
                print(key, '在array中的位置的下标是', mid)
                return mid
    else:
        print('key不在array中')


ls = [3, 8, 20, 34, 45, 56, 67, 78, 92, 100]
key_a = 56
mid = get_number_in_ordered_list(ls, key_a)
print(mid)