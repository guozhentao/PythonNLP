""""""
"""二叉树中和为某一值的路径(递归)"""
# 输入一颗二叉树的根节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。
# 路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)
"""在题目要求中，要注意两点，第一是从根节点开始到叶子结点结束，第二是所有路径。
所以，我们可以利用递归，用带记忆的深度遍历法来进行"""
class Node:
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def find_path(root, int_num):
    if not root:
        return
    path = []   # 记录路径
    sum = 0     # 路径中节点值之和
    find_one_path(root, int_num, path, sum)

def find_one_path(node, int_num, path, sum):
    sum += node.value       # 将当前的根节点加入path的尾部，并且此时路径之和为原来和加上当前根节点的值
    path.append(node.value)

    # 路径之和等于输入的整数，且此节点为叶节点点(既无左子节点又无右子节点)
    if sum == int_num and not node.left and not node.right:
        # 那么此时应该从前到后打印path中的元素，这就是我们走过的路径。
        print('发现一条路径！')
        for i in path:
            print("{}\t".format(i))
        print('\n')

    # 如果不是叶节点，则遍历它的子节点
    if node.left:
        find_one_path(node.left, int_num, path, sum)
    if node.right:
        find_one_path(node.right, int_num, path, sum)

    # 如果已经是叶节点了，但是路径之和不等于我们期望的值，那么往后退一步，返回父节点
    path.pop()


# 变形：根到叶子结点组成的数之和
"""给定一个仅包含数字从0到9的二叉树，每个根到叶子的路径都可以表示一个数字。
比如从根到叶的路径1-> 2-> 3，它表示数字123。
找到所有从根到叶的数字的总和。"""
# 从上到下遍历二叉树，逐个节点进行num*10+root.val操作
class Solution(object):
    def sum_numbers(self, root):
        if not root:
            return 0
        self.res = 0

        def dfs(node, num):
            if not node.left and not node.right:

                self.res += num*10+node.value
            if node.left:
                dfs(node.left, num*10+node.value)
            if node.right:
                dfs(node.right, num*10+node.value)

        dfs(root, 0)
        print(self.res)
        return self.res


if __name__ == '__main__':
    root = Node(1,
                Node(2,
                     Node(4), Node(5)),
                Node(3)
                )

    find_path(root, 7)

    solution = Solution()
    solution.sum_numbers(root)