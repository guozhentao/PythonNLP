# 找出数组中重复的数
""""""
"""在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，
但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字"""
# 例如，如果输入长度为7的数组 [ 2, 3, 1, 0, 2, 5, 3 ]，那么对应的输出是重复的数字2或者3
# 采用交换原数组中的元素的方法，时间 O(n) 空间 O(1)
def find_duplicate_num(array):
    if array is None or not isinstance(array, list):
        return False
    for i in range(len(array)):
        while i != array[i]:
            if array[i] == array[array[i]]:
                print("重复数字: ", array[i])
                return True
            else:
                tmp = array[i]
                array[i] = array[tmp]
                array[tmp] = tmp
    return False

# 当然这个算法可以先排序，排完序后遍历，那么相邻数字相同的就是重复的


lis = [2, 3, 1, 0, 3, 5, 3]
m = find_duplicate_num(lis)
print(m)

