import pickle
import os

class BPlusTreeNode:
    def __init__(self, is_leaf=False):
        self.is_leaf = is_leaf
        self.keys = []
        self.children = []
        self.next = None  # 用于叶子节点链表连接

class BPlusTree:
    def __init__(self, order=4):
        self.root = BPlusTreeNode(is_leaf=True)
        self.order = order

    def insert(self, key, value):
        node = self.root
        if len(node.keys) == (self.order - 1):
            new_root = BPlusTreeNode()
            new_root.children.append(f'node_{id(self.root)}.pkl')
            self._split_child(new_root, 0)
            self.root = new_root
        self._insert_non_full(self.root, key, value)

    def _insert_non_full(self, node, key, value):
        if node.is_leaf:
            i = 0
            while i < len(node.keys) and key > node.keys[i]:
                i += 1
            node.keys.insert(i, key)
            node.children.insert(i, value)
            save_node_to_disk(node, f'node_{id(node)}.pkl')
        else:
            i = len(node.keys) - 1
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1

            # Debug output to trace the issue
            print(f"Inserting key: {key} at position {i} in node with keys {node.keys}")
            print(f"Node children before inserting: {node.children}")

            if i >= len(node.children):
                raise IndexError("Attempted to access out-of-bounds index in node.children")

            child_node = load_node_from_disk(node.children[i])
            if len(child_node.keys) == (self.order - 1):
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1
            self._insert_non_full(load_node_from_disk(node.children[i]), key, value)

    # def _insert_non_full(self, node, key, value):
    #     if node.is_leaf:
    #         i = 0
    #         while i < len(node.keys) and key > node.keys[i]:
    #             i += 1
    #         node.keys.insert(i, key)
    #         node.children.insert(i, value)
    #         save_node_to_disk(node, f'node_{id(node)}.pkl')
    #     else:
    #         i = len(node.keys) - 1
    #         while i >= 0 and key < node.keys[i]:
    #             i -= 1
    #         i += 1
    #         child_node = load_node_from_disk(node.children[i])
    #         if len(child_node.keys) == (self.order - 1):
    #             self._split_child(node, i)
    #             if key > node.keys[i]:
    #                 i += 1
    #         self._insert_non_full(load_node_from_disk(node.children[i]), key, value)

    # def _split_child(self, parent, index):
    #     node_to_split = load_node_from_disk(parent.children[index])
    #     new_node = BPlusTreeNode(is_leaf=node_to_split.is_leaf)
    #     mid = len(node_to_split.keys) // 2
    #
    #     parent.keys.insert(index, node_to_split.keys[mid])
    #     parent.children.insert(index + 1, f'node_{id(new_node)}.pkl')
    #
    #     new_node.keys = node_to_split.keys[mid + 1:]
    #     new_node.children = node_to_split.children[mid + 1:]
    #
    #     node_to_split.keys = node_to_split.keys[:mid]
    #     node_to_split.children = node_to_split.children[:mid + 1]
    #
    #     if node_to_split.is_leaf:
    #         new_node.next = node_to_split.next
    #         node_to_split.next = new_node
    #
    #     save_node_to_disk(node_to_split, parent.children[index])
    #     save_node_to_disk(new_node, parent.children[index + 1])
    #
    #     # 更新parent的子节点为字符串形式
    #     parent.children[index] = f'node_{id(node_to_split)}.pkl'
    #     parent.children[index + 1] = f'node_{id(new_node)}.pkl'

    def _split_child(self, parent, index):
        node_to_split = load_node_from_disk(parent.children[index])
        new_node = BPlusTreeNode(is_leaf=node_to_split.is_leaf)
        mid = len(node_to_split.keys) // 2

        parent.keys.insert(index, node_to_split.keys[mid])
        parent.children.insert(index + 1, f'node_{id(new_node)}.pkl')

        new_node.keys = node_to_split.keys[mid + 1:]
        new_node.children = node_to_split.children[mid + 1:]

        node_to_split.keys = node_to_split.keys[:mid]
        node_to_split.children = node_to_split.children[:mid + 1]

        if node_to_split.is_leaf:
            new_node.next = node_to_split.next
            node_to_split.next = new_node

        # Save nodes after splitting
        save_node_to_disk(node_to_split, f'node_{id(node_to_split)}.pkl')
        save_node_to_disk(new_node, f'node_{id(new_node)}.pkl')

        # Also, save the updated parent node
        save_node_to_disk(parent, f'node_{id(parent)}.pkl')

    def search(self, key):
        node = self.root
        while not node.is_leaf:
            index = self._find_child_index(node, key)
            node = load_node_from_disk(node.children[index])
        for i, item in enumerate(node.keys):
            if item == key:
                return node.children[i]
        return None

    def _find_child_index(self, node, key):
        for i, k in enumerate(node.keys):
            if key < k:
                return i
        return len(node.keys)

def save_node_to_disk(node, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(node, f)

def load_node_from_disk(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    else:
        return BPlusTreeNode()

if __name__ == '__main__':
    # 初始化B+树
    bptree = BPlusTree(order=4)

    # 插入一些键值对
    bptree.insert(10, "Value for 10")
    bptree.insert(20, "Value for 20")
    bptree.insert(5, "Value for 5")
    bptree.insert(6, "Value for 6")
    bptree.insert(12, "Value for 12")
    bptree.insert(30, "Value for 30")
    bptree.insert(7, "Value for 7")
    bptree.insert(17, "Value for 17")

    # 从磁盘加载并查找一个键
    result = bptree.search(6)
    print(f"Search result for key 6: {result}")

    result = bptree.search(20)
    print(f"Search result for key 20: {result}")

    result = bptree.search(50)
    print(f"Search result for key 50 (not present): {result}")
