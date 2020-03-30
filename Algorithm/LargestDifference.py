"""数对之差的最大值"""
# 【问题描述】在数组中，数字减去它右边的数字得到一个数对之差。求所有数对之差的最大值。
# 例如，在数组[2, 4, 1, 16, 7, 5, 11, 9]中，数对之差的最大值是11，是16减去5的结果。
"""我们讲一个比较巧妙的方法：思想是：
如果输入的数组numbers长度为n，我们先构建一个长度为n-1的辅助数组diff，并且diff[i]等于numbers[i]-numbers[i+1]（0<=i<n-1）。
如果我们从数组diff中的第i个数字一直累加到第j个数字（j > i），
也就是diff[i]+diff[i+1]+…+diff[j]=(numbers[i]-numbers[i+1])+(numbers[i+1]-numbers[i+2])+...+(numbers[j]–numbers[j+1]) 
= numbers[i] – numbers[j + 1]。
分析到这里，我们发现原始数组中最大的数对之差（即numbers[i] – numbers[j + 1]）其实是辅助数组diff中最大的连续子数组之和。
但是这种方法需要额外的辅助空间"""
# 下面用动态规划的思想做这个算法。  a-b=c，a是被减数，b是减数，c是差
"""我们定义diff[i]是以数组中第i个数字为减数的所有 数对之差的最大值。
也就是说对于任意h(h<i)，diff[i]≥number[h]-number[i]。  diff[i]（0≤i<n）的最大值就是整个数组数对之差的最大值。
假设我们已经求得了diff[i],我们该怎么求得diff[i+1]呢？对于diff[i],肯定存在一个h(h<i),满足number[h]减去number[i]之差是最大的,
也就是number[h]应该是number[i]之前的所有数字的最大值。当我们求diff[i+1]的时候，我们需要找到第i+1个数字之前的最大值。
第i+1个数字之前的最大值有两种可能：这个最大值可能是第i个数字之前的最大值，也有可能就是第i个数字本身。
第i+1个数字之前的最大值肯定是这两者中的较大者。我们只要拿第i+1个数字之前的最大值减去number[i+1]，就得到了diff[i+1]。"""


def find_max_diff(array):
    if len(array) < 2 or not isinstance(array, list):
        return 0
    max = array[0]  # 用max表示第i个数字之前的最大值
    max_diff = max - array[1]
    for i in range(2, len(array)):
        if array[i - 1] > max:
            max = array[i - 1]
        current_diff = max - array[i]  # current_diff表示diff[i](0≤i<n), diff[i]的最大值就是代码中max_diff
        if current_diff > max_diff:
            max_diff = current_diff
    return max_diff


li = [2, 4, 1, 16, 7, 5, 11, 9]
max_diff = find_max_diff(li)
print(max_diff)
