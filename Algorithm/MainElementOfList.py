# 找出数组的主元素
# 问题描述：
# 大小为Ｎ的数组，其主元素是一个出现超过Ｎ／２次的元素（从而这样的元素最多有一个），
# 怎么用线性时间算法得到一个数组的主元素（如果有的话）。

# 借助一个字典
def find_main_element(array):
    dict = {}
    for num in array:
        if num not in dict.keys():
            dict[num] = 1
        else:
            dict[num] +=1
        if dict[num] > len(array)/2:
            return num
    return -1


li = [1, 2, 3, 4, 5, 5, 5, 5, 5]
num = find_main_element(li)
print(num)