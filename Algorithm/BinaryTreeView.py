"""二叉树的左右视图"""
class Node:
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


# 二叉树的右视图
"""思想：
从上至下一层一层的遍历二叉树，每层从左至右遍历当前层所有节点，
将每个节点的子节点储存起来作为下一层要遍历的节点，将当前层的最后一个节点的值添加至最终结果，直到所有层遍历完毕"""
def right_view(root):
    if root is None:
        return 0
    array = []
    li = [root]
    while li != []:
        p = []
        array.append(li[-1].value)
        for node in li:
            if node.left:
                p.append(node.left)
            if node.right:
                p.append(node.right)
        li = p
    return array


# 二叉树的左视图
def left_view(root):
    if root is None:
        return 0
    list = []
    li = [root]
    while li != []:
        p = []
        list.append(li[0].value)
        for node in li:
            if node.left:
                p.append(node.left)
            if node.right:
                p.append(node.right)
        li = p
    return list


root = Node(1,
            Node(2, Node(4), Node(5)),
            Node(3, left=Node(6))
            )
print('右视图')
array = right_view(root)
print(array)
print('左视图')
lis = left_view(root)
print(lis)
