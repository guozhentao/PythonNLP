# 前序遍历二叉树的顺序是：根节点、左节点、右节点
# 中序遍历二叉树的顺序是：左节点、根节点、右节点
# 后序遍历二叉树的顺序是：左节点、右节点、根节点
# 层次遍历二叉树的顺序是：从上到下、从左到右按层遍历

# 前序、中序、后续遍历都是深度优先遍历
# 层次遍历是广度优先遍历

# python中可以使用list来代替栈结构，list.append()和list.pop()
"""首先，定义一个二叉树"""
class Node:
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


"""前序遍历，递归"""
def preorder_traversal_recursive(root):     # recursive: adj.递归的
    if root is None:
        return None
    print(root.value)
    preorder_traversal_recursive(root.left)
    preorder_traversal_recursive(root.right)


"""前序遍历，非递归"""
def preorder_traversal_non_recursive(root):
    """借助栈实现前序遍历
    """
    if root == None:
        return
    stack = []
    while root or len(stack) > 0:
        if root:
            stack.append(root)
            print(root.value)
            root = root.left
        else:
            root = stack[-1]
            stack.pop()
            root = root.right


"""中序遍历，递归"""
def midorder_traversal_recursive(root):
    if root is None:
        return
    midorder_traversal_recursive(root.left)
    print(root.value)
    midorder_traversal_recursive(root.right)


"""中序遍历，非递归"""
def midorder_traversal_non_recursive(root):
    node_list = []
    node = root
    while node is not None or len(node_list)>0:
        if node is not None:
            node_list.append(node)
            node = node.left
        else:
            node = node_list.pop()
            print(node.value)
            node = node.right


"""后序遍历，递归"""
def postorder_traversal_recursive(root):
    if root is None:
        return
    postorder_traversal_recursive(root.left)
    postorder_traversal_recursive(root.right)
    print(root.value)


"""后序遍历，非递归"""
# 使用两个栈结构
# 第一个栈进栈顺序：左节点->右节点->根节点
# 第一个栈弹出顺序： 根节点->右节点->左节点
# 第二个栈为第一个栈的每个弹出依次进栈
# 最后第二个栈依次出栈
def postorder_traversal_non_recursive(root):
    node_list1 = [root]
    node_list2 = []
    while len(node_list1)>0:
        node = node_list1.pop()
        node_list2.append(node)
        if root.left is not None:
            node_list1.append(root.left)
        if root.right is not None:
            node_list2.append(root.right)
    while len(node_list2)>0:
        print(node_list2.pop().value)


"""层次遍历，从上往下、从左到右按层遍历"""
"""有时所说的广度优先遍历就是层次遍历"""
def layer_traversal(root):
    if root is None:
        return
    import queue
    que = queue.Queue()     # 创建先进先出队列
    que.put(root)
    while not que.empty():
        head = que.get()    # 弹出第一个元素并打印
        print(head.value)
        if head.left is not None:      # 若该节点存在左子节点,则加入队列（先put左节点）
            que.put(head.left)
        if head.right is not None:     # 若该节点存在右子节点,则加入队列（再put右节点）
            que.put(head.right)


if __name__=='__main__':
    root = Node('D',
                Node('B',
                     Node('A'), Node('C')),
                Node('E',
                     right=Node('G',
                                Node('F')
                                )
                     )
                )

    print('前序遍历：')
    preorder_traversal_recursive(root)
    print('\n')
    print('中序遍历：')
    midorder_traversal_recursive(root)
    print('\n')
    print('后序遍历：')
    postorder_traversal_recursive(root)
    print('\n')
    print('层次遍历')
    layer_traversal(root)
    print('\n')
