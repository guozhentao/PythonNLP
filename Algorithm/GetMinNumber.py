# 输入一个整数数组，把数组中的所有数字拼接成一个数，打印能拼接出所有数字最小的一个。
# 例如输入数组[3,32,321],则打印出这3个数字能排列的最小数字321323
""""""
"""思路：（1）可以先考虑只有两个数字的情况： [3,32] ，可以看出来 332>323 因此需要把数组改变为 [32,3] ；
（2）对于有三个数字的情况： [3,32,321] 我们两两进行比较， 332>323 于是，将 3 与 32 交换位置变成 [32,3,321] 
而 3321>3213 于是将 3 与 321 继续交换位置到 [32,321,3] ；
接着我们继续使用 32 进行比较，由于 32321>32132 将 32与321 进行位置交换为 [321,32,3] 此时，将数组链接起来变成 321323 即为最小的数
考虑到组合后大数问题以及字符串组合可以相加，需先将整数数组转为字符串数组，然后构建比较函数，当str1+str2>str2+str1时，str1>str2，
之后将字符串列表安装比较函数进行排序"""
# 或者如下：
"""定义一种排序规则，数字m 和 n 能拼成数字mn和nm，比较mn和nm的大小，由于位数相同的，比较大小可以按照字符串大小的比较规则。
如果拼接后的数字mn 大于 nm，则m>n; 否则m < n.
按照这种排序规则，将字符串数组从小到大排序，排序后的就是最小数字。"""
def get_min_number(array):
    if not array or len(array) == 0:
        return 0
    str_num_list = [str(num) for num in array]    # 将int型列表array转化成字符串list
    res = string_sort(str_num_list)
    return int(''.join(res))

def string_sort(mylist):     # mylist是一个列表，里面的元素是字符串
    if len(mylist) < 2:
        return 0
    else:
        less, greater = [], []
        midValue = mylist[0]
        for elem in mylist[1:]:
            if midValue + elem > elem + midValue:
                less.append(elem)
            else:
                greater.append(elem)
    return string_sort(less) + [midValue] + string_sort(greater)


li = [3, 32, 321]
num = get_min_number(li)
print(num)
