# 题目：一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。
# 思路：列表中只有两种元素，一种为出现次数是两次的，一种为出现次数是1次的。因此，可以利用下面的方法来推断出最终的结果。
def get_nums_appear_only_once(array):
    if len(array) < 2:
        return '数组不符合题目要求~'
    arr = []
    for element in array:
        if element in arr:
            arr.remove(element)
        else:
            arr.append(element)
    return arr


array = [1, 2, 3, 4, 1, 2, 3, 5]
arr = get_nums_appear_only_once(array)
print(arr)