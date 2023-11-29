"""
Project 5
CSE 331 S21 (Onsay)
Your Name
AVLTree.py
"""
import math
import queue
from typing import TypeVar, Generator, List, Tuple

# for more information on typehinting, check out https://docs.python.org/3/library/typing.html
T = TypeVar("T")  # represents generic type
Node = TypeVar("Node")  # represents a Node object (forward-declare to use in Node __init__)
AVLWrappedDictionary = TypeVar("AVLWrappedDictionary")  # represents a custom type used in application


####################################################################################################


class Node:
    """
    Implementation of an AVL tree node.
    Do not modify.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["value", "parent", "left", "right", "height"]

    def __init__(self, value: T, parent: Node = None,
                 left: Node = None, right: Node = None) -> None:
        """
        Construct an AVL tree node.

        :param value: value held by the node object
        :param parent: ref to parent node of which this node is a child
        :param left: ref to left child node of this node
        :param right: ref to right child node of this node
        """
        self.value = value
        self.parent, self.left, self.right = parent, left, right
        self.height = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return f"<{str(self.value)}>"

    def __str__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return f"<{str(self.value)}>"


####################################################################################################


class AVLTree:
    """
    Implementation of an AVL tree.
    Modify only below indicated line.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty AVL tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree as a string. Inspired by Anna De Biasi (Fall'20 Lead TA).

        :return: string representation of the AVL tree
        """
        if self.origin is None:
            return "Empty AVL Tree"

        # initialize helpers for tree traversal
        root = self.origin
        result = ""
        q = queue.SimpleQueue()
        levels = {}
        q.put((root, 0, root.parent))
        for i in range(self.origin.height + 1):
            levels[i] = []

        # traverse tree to get node representations
        while not q.empty():
            node, level, parent = q.get()
            if level > self.origin.height:
                break
            levels[level].append((node, level, parent))

            if node is None:
                q.put((None, level + 1, None))
                q.put((None, level + 1, None))
                continue

            if node.left:
                q.put((node.left, level + 1, node))
            else:
                q.put((None, level + 1, None))

            if node.right:
                q.put((node.right, level + 1, node))
            else:
                q.put((None, level + 1, None))

        # construct tree using traversal
        spaces = pow(2, self.origin.height) * 12
        result += "\n"
        result += f"AVL Tree: size = {self.size}, height = {self.origin.height}".center(spaces)
        result += "\n\n"
        for i in range(self.origin.height + 1):
            result += f"Level {i}: "
            for node, level, parent in levels[i]:
                level = pow(2, i)
                space = int(round(spaces / level))
                if node is None:
                    result += " " * space
                    continue
                if not isinstance(self.origin.value, AVLWrappedDictionary):
                    result += f"{node} ({parent} {node.height})".center(space, " ")
                else:
                    result += f"{node}".center(space, " ")
            result += "\n"
        return result

    def __str__(self) -> str:
        """
        Represent the AVL tree as a string. Inspired by Anna De Biasi (Fall'20 Lead TA).

        :return: string representation of the AVL tree
        """
        return repr(self)

    def height(self, root: Node) -> int:
        """
        Return height of a subtree in the AVL tree, properly handling the case of root = None.
        Recall that the height of an empty subtree is -1.

        :param root: root node of subtree to be measured
        :return: height of subtree rooted at `root` parameter
        """
        return root.height if root is not None else -1

    def left_rotate(self, root: Node) -> Node:
        """
        Perform a left rotation on the subtree rooted at `root`. Return new subtree root.

        :param root: root node of unbalanced subtree to be rotated.
        :return: new root node of subtree following rotation.
        """
        if root is None:
            return None

        # pull right child up and shift right-left child across tree, update parent
        new_root, rl_child = root.right, root.right.left
        root.right = rl_child
        if rl_child is not None:
            rl_child.parent = root

        # right child has been pulled up to new root -> push old root down left, update parent
        new_root.left = root
        new_root.parent = root.parent
        if root.parent is not None:
            if root is root.parent.left:
                root.parent.left = new_root
            else:
                root.parent.right = new_root
        root.parent = new_root

        # handle tree origin case
        if root is self.origin:
            self.origin = new_root

        # update heights and return new root of subtree
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        new_root.height = 1 + max(self.height(new_root.left), self.height(new_root.right))
        return new_root

    ########################################
    # Implement functions below this line. #
    ########################################

    def right_rotate(self, root: Node) -> Node:
        """
        Perform a right rotation on the subtree rooted at `root`. Return new subtree root.

        :param root: root node of unbalanced subtree to be rotated.
        :return: new root node of subtree following rotation.
        """
        if root is None:
            return None

        # pull right child up and shift right-left child across tree, update parent
        new_root, lr_child = root.left, root.left.right
        root.left = lr_child
        if lr_child is not None:
            lr_child.parent = root

        # right child has been pulled up to new root -> push old root down left, update parent
        new_root.right = root
        new_root.parent = root.parent
        if root.parent is not None:
            if root is root.parent.right:
                root.parent.right = new_root
            else:
                root.parent.left = new_root
        root.parent = new_root

        # handle tree origin case
        if root is self.origin:
            self.origin = new_root

        # update heights and return new root of subtree
        root.height = 1 + max(self.height(root.right), self.height(root.left))
        new_root.height = 1 + max(self.height(new_root.right), self.height(new_root.left))
        return new_root

    def balance_factor(self, root: Node) -> int:
        """
        Determine the balance factor of a node in a AVL
        :param root: Root to find the balance factor of
        :returns: Integer balance factor
        """
        if root is None:
            return 0

        l_child, r_child = root.left, root.right

        if l_child is None and r_child is None:
            return 0
        elif l_child is None:
            return -r_child.height - 1
        elif r_child is None:
            return l_child.height + 1
        return l_child.height - r_child.height

    def rebalance(self, root: Node) -> Node:
        """
        Rebalance the tree after adding or taking off a node
        :param root: The root node to check
        :returns: New root node
        """
        bal_factor = self.balance_factor(root)

        if bal_factor <= -2:
            if self.balance_factor(root.right) == 1:
                self.right_rotate(root.right)
            self.left_rotate(root)
            return root.right

        elif bal_factor >= 2:
            if self.balance_factor(root.left) == -1:
                self.left_rotate(root.left)
            self.right_rotate(root)
            return root.left

        return root

    def insert(self, root: Node, val: T) -> Node:
        """
        Insert a node into the AVL
        :param root: Root of the AVL
        :param val: Value to be inserted
        :returns: Node that was just inserted
        """

        def insert_inner(cur: Node, val: T):
            """
            Check if node can be inserted in AVL at cur
            :param cur: Current node of the AVL
            :param val: Value to be inserted
            """
            if cur.value > val:
                if cur.left is None:
                    cur.left = Node(val)
                    cur.left.parent = cur
                    cur.height = 1 + max(self.height(cur.right), self.height(cur.left))
                    self.rebalance(cur)

                else:
                    insert_inner(cur.left, val)
                    cur.height = 1 + max(self.height(cur.right), self.height(cur.left))
                    self.rebalance(cur)

            elif cur.value < val:
                if cur.right is None:
                    cur.right = Node(val)
                    cur.right.parent = cur
                    cur.height = 1 + max(self.height(cur.right), self.height(cur.left))
                    self.rebalance(cur)

                else:
                    insert_inner(cur.right, val)
                    cur.height = 1 + max(self.height(cur.right), self.height(cur.left))
                    self.rebalance(cur)

        if root is None:
            self.origin = Node(val)
            self.size = 1
            return self.origin

        insert_inner(root, val)
        self.size += 1
        return self.origin

    def min(self, root: Node) -> Node:
        """
        Finds minimum in the AVL
        :param root: Root of the AVL to check
        :returns: Node of maximum value
        """
        if root is None:
            return None

        elif root.left is not None:
            return self.min(root.left)
        return root

    def max(self, root: Node) -> Node:
        """
        Finds maximum in the AVL
        :param root: Root of the AVL to check
        :returns: Node of maximum value
        """
        if root is None:
            return None

        elif root.right is not None:
            return self.max(root.right)
        return root

    def search(self, root: Node, val: T) -> Node:
        """
        Finds root with value given in input
        :param root: Root of AVL to check
        :param val: Value to search for
        :returns: Node with value of input, or where it would be
        """
        if self.origin is None:
            return None

        if root.left is not None and val < root.value:
            return self.search(root.left, val)
        elif root.right is not None and val > root.value:
            return self.search(root.right, val)

        else:
            return root

    def inorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Gets the AVL in number order sequentially
        :param root: Root of AVL to check
        :returns: Next attribute in the order
        """
        if root is not None:
            yield from self.inorder(root.left)
            yield root
            yield from self.inorder(root.right)

    def preorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Gets the AVL in preorder order sequentially
        :param root: Root of AVL to check
        :returns: Next attribute in the order
        """
        if root is not None:
            yield root
            yield from self.preorder(root.left)
            yield from self.preorder(root.right)

    def postorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Gets the AVL in postorder order sequentially
        :param root: Root of AVL to check
        :returns: Next attribute in the order
        """
        if root is not None:
            yield from self.postorder(root.left)
            yield from self.postorder(root.right)
            yield root

    def levelorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Gets the AVL in level order sequentially
        :param root: Root of AVL to check
        :returns: Next attribute in the order
        """
        if root is None:
            return

        nodes = queue.Queue()
        nodes.put(root)

        while not nodes.empty():
            res = nodes.get()
            yield res
            if res.left is not None:
                nodes.put(res.left)
            if res.right is not None:
                nodes.put(res.right)
        return

    def remove(self, root: Node, val: T) -> Node:
        """
        Removes given value from AVL
        :param root: Root of AVL node to check
        :returns: New origin of AVL node
        """
        if self.origin is None:
            return None
        elif self.size == 1:
            self.origin = None
            self.size = 0
            return None

        if root.value < val:
            self.remove(root.right, val)
            root.height = 1 + max(self.height(root.left), self.height(root.right))
            return self.rebalance(root)
        elif root.value > val:
            self.remove(root.left, val)
            root.height = 1 + max(self.height(root.left), self.height(root.right))
            return self.rebalance(root)

        if root.left is not None and root.right is not None:
            predecessor = self.max(root.left)
            temp = predecessor.value
            predecessor.value = root.value
            root.value = temp
            return self.remove(root.left, val)

        elif root.left is not None:
            root.value = root.left.value
            root.left = None
        elif root.right is not None:
            root.value = root.right.value
            root.right = None
        else:
            if root.parent.left == root:
                root.parent.left = None
            elif root.parent.right == root:
                root.parent.right = None

        self.size -= 1
        root.height = 1 + max(self.height(root.left), self.height(root.right))
        return self.rebalance(root)


####################################################################################################


class AVLWrappedDictionary:
    """
    Implementation of a helper class which will be used as tree node values in the
    NearestNeighborClassifier implementation. Compares objects with keys less than
    1e-6 apart as equal.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["key", "dictionary"]

    def __init__(self, key: float) -> None:
        """
        Construct a AVLWrappedDictionary with a key to search/sort on and a dictionary to hold data.

        :param key: floating point key to be looked up by.
        """
        self.key = key
        self.dictionary = {}

    def __repr__(self) -> str:
        """
        Represent the AVLWrappedDictionary as a string.

        :return: string representation of the AVLWrappedDictionary.
        """
        return f"key: {self.key}, dict: {self.dictionary}"

    def __str__(self) -> str:
        """
        Represent the AVLWrappedDictionary as a string.

        :return: string representation of the AVLWrappedDictionary.
        """
        return f"key: {self.key}, dict: {self.dictionary}"

    def __eq__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement == operator to compare 2 AVLWrappedDictionaries by key only.
        Compares objects with keys less than 1e-6 apart as equal.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating whether keys of AVLWrappedDictionaries are equal
        """
        return abs(self.key - other.key) < 1e-6

    def __lt__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement < operator to compare 2 AVLWrappedDictionarys by key only.
        Compares objects with keys less than 1e-6 apart as equal.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating ordering of AVLWrappedDictionaries
        """
        return self.key < other.key and not abs(self.key - other.key) < 1e-6

    def __gt__(self, other: AVLWrappedDictionary) -> bool:
        """
        Implement > operator to compare 2 AVLWrappedDictionaries by key only.
        Compares objects with keys less than 1e-6 apart as equal.

        :param other: other AVLWrappedDictionary to compare with
        :return: boolean indicating ordering of AVLWrappedDictionaries
        """
        return self.key > other.key and not abs(self.key - other.key) < 1e-6


class NearestNeighborClassifier:
    """
    Implementation of a one-dimensional nearest-neighbor classifier with AVL tree lookups.
    Modify only below indicated line.
    """
    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["resolution", "tree"]

    def __init__(self, resolution: int) -> None:
        """
        Construct a one-dimensional nearest neighbor classifier with AVL tree lookups.
        Data are assumed to be floating point values in the closed interval [0, 1].

        :param resolution: number of decimal places the data will be rounded to, effectively
                           governing the capacity of the model - for example, with a resolution of
                           1, the classifier could maintain up to 11 nodes, spaced 0.1 apart - with
                           a resolution of 2, the classifier could maintain 101 nodes, spaced 0.01
                           apart, and so on - the maximum number of nodes is bounded by
                           10^(resolution) + 1.
        """
        self.tree = AVLTree()
        self.resolution = resolution

        # pre-construct lookup tree with AVLWrappedDictionary objects storing (key, dictionary)
        # pairs, but which compare with <, >, == on key only
        for i in range(10 ** resolution + 1):
            w_dict = AVLWrappedDictionary(key=(i / 10 ** resolution))
            self.tree.insert(self.tree.origin, w_dict)

    def __repr__(self) -> str:
        """
        Represent the NearestNeighborClassifier as a string.

        :return: string representation of the NearestNeighborClassifier.
        """
        return f"NNC(resolution={self.resolution}):\n{self.tree}"

    def __str__(self) -> str:
        """
        Represent the NearestNeighborClassifier as a string.

        :return: string representation of the NearestNeighborClassifier.
        """
        return f"NNC(resolution={self.resolution}):\n{self.tree}"

    def fit(self, data: List[Tuple[float, str]]) -> None:
        """
        Fits data to AVL tree in NNC
        :param data: Input data with a float and corresponding string value
        """
        for index in data:
            key = round(index[0], self.resolution)
            tmp = AVLWrappedDictionary(key)
            node = self.tree.search(self.tree.origin, tmp)
            if index[1] not in node.value.dictionary:
                node.value.dictionary[index[1]] = 1
            else:
                node.value.dictionary[index[1]] += 1

    def predict(self, x: float, delta: float) -> str:
        """
        Predict the value of given x value
        :param x: Float value to check
        :param delta: The range to check of key value
        :returns: The most likely corresponding value
        """
        res = {}

        rounded = round(x, self.resolution)
        increment = 10 ** -self.resolution
        index = round(rounded - delta, self.resolution)
        max_index = rounded + delta
        while index <= max_index:
            node = self.tree.search(self.tree.origin, AVLWrappedDictionary(index))
            if index == node.value.key:
                for key in node.value.dictionary:
                    if key in res:
                        res[key] += node.value.dictionary[key]
                    else:
                        res[key] = node.value.dictionary[key]
            index = round(index + increment, self.resolution)

        if len(res) == 0:
            return None

        res_key = ""
        max = 0
        for key in res.keys():
            if res[key] > max:
                max = res[key]
                res_key = key

        return res_key
