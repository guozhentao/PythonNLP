# 找出数组中重复的数
""""""
"""在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，
但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字"""
# 例如，如果输入长度为7的数组 [ 2, 3, 1, 0, 2, 5, 3 ]，那么对应的输出是重复的数字2或者3
# 采用交换原数组中的元素的方法，时间 O(n) 空间 O(1)
def find_duplicate_num(array):
    if not isinstance(array, list):    # 如果array的类型不是list，就返回-1
        return -1
    for i in range(len(array)):     # 在遍历的过程中如果数组的元素不是int型，或者小于0，或者大于数组长度，就直接返回
        if not isinstance(array[i], int):
            return -2
        if array[i] < 0 | array[i] > len(array):
            return -3
        m = array[i]
        while i != m:
            array[i], array[m] = array[m], array[i]
            if m == array[m]:
                return m
    return -4


lis = [ 2, 3, 1, 0, 2, 5, 3 ]
m = find_duplicate_num(lis)
print(m)

