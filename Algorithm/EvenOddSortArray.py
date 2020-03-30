# 给定一个非负整数数组 A，返回一个由 A 的所有偶数元素组成的数组，后面跟 A 的所有奇数元素。
# 你可以返回满足此条件的任何数组作为答案，即只要把数组的偶数都弄到前面，奇数弄到后面就可以。
def even_odd_sort_list(array):
    even = []
    odd_number = []
    for element in array:
        if element % 2 ==0:
            even.append(element)
        else:
            odd_number.append(element)
    return even + odd_number


arr = [3, 1, 2, 4, 7, 9, 10,11]
list = even_odd_sort_list(arr)
print(list)