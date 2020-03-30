# 首先定义一颗二叉树
class Node:
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


# 求二叉树的节点数，递归求解
def get_node_nums(root):
    if root is None:
        return
    num1 = get_node_nums(root.left)
    num2 = get_node_nums(root.right)
    num3 = num1 + num2
    num = num3 + 1
    return num    # 加1是因为得算上根节点


# 求二叉树的叶子结点个数
def get_leaf_node_nums(root):
    if root is Node:
        return 0
    if root.left == None and root.right == None:
        return 1
    return get_leaf_node_nums(root.left)+get_leaf_node_nums(root.right)


# 求二叉树的最大深度，递归
def get_binary_tree_depth(root):
    if root is None:
        return
    left_depth = get_binary_tree_depth(root.left)
    right_depth = get_binary_tree_depth(root.right)
    lagest_depth = max(left_depth, right_depth) + 1
    return lagest_depth


if __name__=='__main__':
    root = Node('A',
                Node('B',
                     Node('D'), Node('E')),
                Node('C',
                     Node('F'), Node('G',
                                     right=Node('H')
                                     )
                     )
                )