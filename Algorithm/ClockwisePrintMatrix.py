# 顺时针打印矩阵的元素
"""题目描述：输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，
例如，如果输入如下矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 
则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10."""
# 思路
"""思路：先取矩阵的第一行，接着将剩下作为新矩阵进行一个逆时针90度的翻转，接着再获取第一行，直到矩阵为空。
需要注意的点pop() 越界，翻转矩阵的时候相当于将列数据变成行数据，可以一列一列获取，最后注意顺序"""
def clock_print_matrix(matrix):
    result = []
    while matrix:
        result += matrix.pop(0)    # 这个matrix必须是二维数组，就是matrix=[[]]的形式
                                   # 如果是np.array([[]])这样构建的话，就没有pop()方法
        if not matrix or not matrix[0]:
            break
        matrix = turn_matrix(matrix)
    return result


def turn_matrix(matrix):    # 把矩阵逆时针旋转90°
    nrows = len(matrix)
    ncols = len(matrix[0])
    new_matrix = []
    for i in range(ncols):
        list = []
        for j in range(nrows):
            list.append(matrix[j][i])
        new_matrix.append(list)
    new_matrix.reverse()
    return new_matrix


matrix = [[1, 2, 3, 10], [4, 5, 6, 11], [7, 8, 9, 12]]
result = clock_print_matrix(matrix)
print(result)