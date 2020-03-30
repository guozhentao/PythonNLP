# 给一个数组，找出数组中第K大的数
""""""
"""已知numpy函数库中max和argmax函数可以得出一个数组中最大的成员以及最大成员所在位置，
比如："""
# 如果可以引入库函数话，那么使用numpy
import numpy as np
arr = [2, 3, 4, 1, 7, 6, 5]
print('arr中最大元素是{}，最大元素的下标索引是{}'.format(np.max(arr), np.argmax(arr)))
print('arr中最大元素是', np.max(arr),'，其下标索引是', np.argmax(arr))

# 尝试找出第2大的数，仍然使用numpy
"""思路是：我们用数组中最小的元素替换掉最大的元素，然后再用max和argmax"""
new_arr = arr
new_arr[np.argmax(arr)] = np.min(arr)
print('第2大的数是', np.max(new_arr), '，其下标索引是', np.argmax(new_arr))


# 进行扩展，尝试找出第K大元素
def find_Kth_large_num(array, k):
    if k <= 0 or k > len(array):
        return
    arr_ = array
    for i in range(k-1):
        arr_[np.argmax(array)] = np.min(array)
    num = np.max(arr_)
    num_index = np.argmax(arr_)
    print('第', k, '大的数是', num, '，其下标索引是', num_index)
    return num, num_index


li = [2, 3, 4, 1, 7, 6, 5]
num, num_index = find_Kth_large_num(li, 2)
print(num, num_index)

"""以上是通过引入numpy库实现的"""
# 下面是不引入库来找出第二大的数，只需要一次for遍历
def find_second_large_num(array):
    # one 存储最大值，two 存储第二大的值，遍历一次数组即可，
    # 先判断是否大于 one，若大于将 one 的值给 two，将 num_list[i] 的值给 one，
    # 否则比较是否大于two，若大于直接将 num_list[i] 的值给two，否则pass
    one = array[0]
    two = array[0]
    for num in array[1:]:
        if num >= one:
            two = one
            one = num
        elif two < num < one:
            two = num
        else:
            pass
    return two
