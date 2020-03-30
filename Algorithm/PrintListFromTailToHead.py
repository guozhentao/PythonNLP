# 从尾到头打印一个list
# python中有三种方法可以倒序输出一个list
ls = [1, 2, 3, 4, 5, 6, 78]
print(ls[::-1])     # list[::-1]就是倒序
# 如果是某个切片的倒序，可以这样list[start:end][::-1]
print(ls[0:4][::-1])
print(ls[0:4][::-2])    # 倒序后步长为2的取值返回


# list的[]中有三个参数，用冒号分割
# list[param1:param2:param3]，param1，相当于start_index，可以为空，默认是0，
# param2，相当于end_index，可以为空，默认是list.size
# param3，步长，默认为1。步长为-1时，返回倒序原序列
print('**************')
ls.reverse()    # 把list翻转后输出
print(ls)
print('**************')

# 函数reversed 返回一个迭代对象，需要list化
print(list(reversed(ls)))

