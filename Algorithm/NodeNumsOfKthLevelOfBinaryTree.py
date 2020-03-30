# 求二叉树第K层的节点个数
# 递归解法： O(n)
# 思路：求以root为根的第k层节点数目，等价于求以root左孩子为根的第k-1层（因为少了root）节点数目
# 加上以root右孩子为根的第k-1层（因为 少了root）节点数目。即：
# 如果二叉树为空或者k<1，返回0
# 如果二叉树不为空并且k==1，返回1
# 如果二叉树不为空且k>1，返回root左子树中第k-1层的节点个数与root右子树第k-1层节点个数之和

def get_node_nums_kth_layer(root, int_k):
    if root is None or int_k < 1:
        return 0
    if root is not None and int_k == 1:
        return 1
    return get_node_nums_kth_layer(root.left, int_k-1) + get_node_nums_kth_layer(root.right, int_k-1)
