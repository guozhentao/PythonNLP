# 二维数组中的查找
import numpy as np
"""在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，
每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，
判断数组中是否含有该整数"""
# 从左下角开始查找
def list_find(matrix, int_a):    # array是一个二维矩阵，int_a是一个int型整数
    rows = len(matrix) - 1      # len(矩阵)是这个矩阵的行数
    cols = len(matrix[0]) - 1
    i = rows
    j = 0
    while j <= cols and i >= 0:
        if matrix[i][j] < int_a:
            j += 1
        elif matrix[i][j] > int_a:
            i -= 1
        else:
            return True
    return False


array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(len(array))
true_or_false = list_find(array, int_a=1)
print(true_or_false)