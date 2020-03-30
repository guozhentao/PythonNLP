class Node(object):
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

# 判断两棵二叉树是否是相同的树
# 递归实现：
# 如果两棵二叉树都为空，返回真
# 如果两棵二叉树一棵为空，另外一棵不为空，返回假
# 如果两棵二叉树都不为空，如果对应的左子树和右子树都同构返回真，其他返回假
def judge_same_binary_tree(root1, root2):
    if root1 is None and root2 is None:     # 都为空，返回真
        return True
    if root1 is None or root2 is None:      # 有一个为空，另一个不为空，返回False
        return False
    if root1.value != root2.value:
        return False
    return judge_same_binary_tree(root1.left, root2.left) and judge_same_binary_tree(root1.right, root2.right)


# 判断二叉树是不是平衡二叉树。若左右子树深度差不超过1则为平衡二叉树
# 递归实现：
# 先获取二叉树深度
# 如果二叉树为空， 返回真
# 如果二叉树不为空，如果左子树和右子树都是AVL树并且左子树和右子树高度相差不大于1，返回真，其他返回假
def get_tree_depth(root):
    if root is None:
        return 0
    left_depth = get_tree_depth(root.left)
    right_depth = get_tree_depth(root.right)
    return max(left_depth, right_depth) + 1


def is_balance_binary_tree(proot):
    if proot is None:
        return 0
    l_depth = get_tree_depth(proot.left)
    r_depth = get_tree_depth(proot.right)
    if abs(l_depth - r_depth) > 1:
        return False
    return is_balance_binary_tree(proot.left) and is_balance_binary_tree(proot.right)


# 判断一颗二叉树是否是镜像对称的
# 思想：用递归做比较简单：一棵树是对称的等价于它的左子树和右子树两棵子树是对称的，问题就转变为判断两棵树是否对称
def judge_symmetric_binary_tree(root):
    if root is None:
        return 0
    return is_symmetric(root.left) and is_symmetric(root.right)
# 判断根节点为root1和root2的两棵树是否是对称的
def is_symmetric(root1, root2):     # 该方法可用于判断两颗二叉树是否互相镜像，镜像即对称
    if root1 is None and root2 is None:
        return True
    if root1 is None or root2 is None:
        return False
    # 这两棵树是对称需要满足的条件：
    # 1.俩根节点相等。 2.树1的左子树和树2的右子树，树2的左子树和树1的右子树都得是对称的
    return root1.value == root2.value and is_symmetric(root1.left, root2.right) and is_symmetric(root1.right, root2.left)


# 求二叉树的镜像
# 递归实现：破坏原来的树，把原来的树改成其镜像
# 如果二叉树为空，返回空
# 如果二叉树不为空，求左子树和右子树的镜像，然后交换左右子树
def get_mirror_binary_tree_one(root):
    if root is None:
        return 0
    left = get_mirror_binary_tree_one(root.right)        # 递归镜像左右子树
    right = get_mirror_binary_tree_one(root.left)
    root.left = left    # 更新根节点的左右子树为镜像后的树
    root.right = right
    return root


# 给定一棵二叉树，要求输出其左右翻转后二叉树的中序遍历。
# 两个步骤：
# 镜像翻转：只需要遍历二叉树，每次访问一个结点时，交换其左右孩子。
# 中序遍历
def mirror_flip(root):
    """翻转镜像"""
    if not root:
        return
    root.left, root.right = root.right, root.left
    mirror_flip(root.left)
    mirror_flip(root.right)
def midorder_traversal_recursive(root):
    """中序遍历"""
    if root is None:
        return
    midorder_traversal_recursive(root.left)
    print(root.value)
    midorder_traversal_recursive(root.right)


# 判断是否为二分查找树BST
# 二分查找树的定义为：
# 1.若它的左子树不为空，则左子树上所有结点的值均小于等于根结点的值；
# 2.若它的右子树不为空，则右子树上所有结点的值均大于等于根结点的值；
# 3.它的左右子树均为二分查找树。
# 递归解法：中序遍历的结果应该是递增的

def is_BST(root):
    array = zhong_xu_bian_li(root)
    bool_val = panduan_di_zeng(array)
    return bool_val


array = []
def zhong_xu_bian_li(root):
    if root is None:
        return 0
    zhong_xu_bian_li(root.left)
    array.append(root.value)
    zhong_xu_bian_li(root.right)
    return array
def panduan_di_zeng(array):
    for i in range(len(array)-2):
        if array[i+1] >= array[i]:
            pass
        else:
            return '该中序遍历结果不是递增的！'
    return True


if __name__ == '__main__':
    root = Node(1,
                Node(2,
                     Node(4), Node(5)),
                Node(3)
                )

    mirror_flip(root)    # 翻转二叉树
    midorder_traversal_recursive(root)    # 中序遍历
    b = is_BST(root)
    print(b)
