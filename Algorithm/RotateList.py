# 旋转数组，即使数组右移K位
# 给定一个数组，将数组中的元素向右移动 k 个位置，其中 k 是非负数.
# 输入: [1,2,3,4,5,6,7] 和 k = 3
# 输出: [5,6,7,1,2,3,4]
# 利用切片
""""""
"""思路：将后k个看做一个新数组
利用数组的切片"""
def rotate_list(array, int_k):
    new_array = array[-int_k:]
    array1 = array[:len(array)-int_k]
    new_array.extend(array1)
    return new_array


ls = [1, 2, 3, 4, 5, 6, 78]
k = 3
new_list = rotate_list(ls, k)
print(new_list)
