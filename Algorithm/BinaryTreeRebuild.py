# 根据二叉树的前序遍历和中序遍历的结果，重建出该二叉树。
# 假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
# 例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

# 在二叉树的前序遍历序列中，第一个数字总是树的根结点的值。但在中序遍历中，根结点的值在序列的中间，
# 左子树的节点的值位于根结点的值的左边，而右子树的节点的值位于根结点的值的右边。
# 采用递归方法
class Node:
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

# 返回构造的二叉树的根节点
# 在进行本函数的递归调用时，需要在本函数名前面加上self.。
def rebuild_binary_tree(self, preorder, inorder):     # 前序、中序遍历序列分别用preorder、inorder来表示，都是list类型
    if not preorder or not inorder or len(preorder) != len(inorder):
        return None
    root = Node(preorder[0])    # 将根定义成节点形式
    i = inorder.index(preorder[0])      # 找出preorder[0]在inorder中的索引
    root.left = self.rebuild_binary_tree(preorder[1:1+i], inorder[:i])  # 在进行本函数的递归调用时，需要在本函数名前面加上self.。
    root.right = self.rebuild_binary_tree(preorder[1+i:], inorder[i+1:])
    return root