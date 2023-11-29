"""
Your Name Here
Project 5 - PriorityHeaps - Solution Code
CSE 331 Fall 2020
Dr. Sebnem Onsay
"""

from typing import List, Any
from Heaps.PriorityNode import PriorityNode, MaxNode, MinNode


class PriorityQueue:
    """
    Implementation of a priority queue - the highest/lowest priority elements
    are at the front (root). Can act as a min or max-heap.
    """

    #   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   DO NOT MODIFY the following attributes/functions
    #   Modify only below indicated line
    __slots__ = ["_data", "_is_min"]

    def __init__(self, is_min: bool = True):
        """
        Constructs the priority queue
        :param is_min: If the priority queue acts as a priority min or max-heap.
        """
        self._data = []
        self._is_min = is_min

    def __str__(self) -> str:
        """
        Represents the priority queue as a string
        :return: string representation of the heap
        """
        return F"PriorityQueue [{', '.join(str(item) for item in self._data)}]"

    __repr__ = __str__

    def to_tree_str(self) -> str:
        """
        Generates string representation of heap in Breadth First Ordering Format
        :return: String to print
        """
        string = ""

        # level spacing - init
        nodes_on_level = 0
        level_limit = 1
        spaces = 10 * int(1 + len(self))

        for i in range(len(self)):
            space = spaces // level_limit
            # determine spacing

            # add node to str and add spacing
            string += str(self._data[i]).center(space, ' ')

            # check if moving to next level
            nodes_on_level += 1
            if nodes_on_level == level_limit:
                string += '\n'
                level_limit *= 2
                nodes_on_level = 0
            i += 1

        return string

    def is_min_heap(self) -> bool:
        """
        Check if priority queue is a min or a max-heap
        :return: True if min-heap, False if max-heap
        """
        return self._is_min

    #   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   Modify below this line
    def __len__(self) -> int:
        """
        Get the number of nodes in the heap
        :return: The number of nodes in the heap
        """
        return len(self._data)

    def empty(self) -> bool:
        """
        Check if there are no nodes in the heap
        :return: True only if there are no nodes in the heap
        """
        return len(self._data) == 0

    def peek(self) -> PriorityNode:
        """
        Gets the root of the heap
        :return: Root of the heap, None if empty
        """
        if len(self._data) == 0:
            return None

        return self._data[0]

    def get_left_child_index(self, index: int) -> int:
        """
        Gets the left child of the given index
        :param index: Index to get the left child of
        :return: The index of the left child
        """
        index = 2 * index + 1
        if index >= len(self):
            return None

        return index

    def get_right_child_index(self, index: int) -> int:
        """
        Gets the right child of the given index
        :param index: Index to get the right child of
        :return: The index of the right child
        """
        index = 2 * index + 2
        if index >= len(self):
            return None

        return index

    def get_parent_index(self, index: int) -> int:
        """
        Gets the parent of the given index
        :param index: Index to get the parent of
        :return: The index of the parent
        """
        if index == 0:
            return None

        return int((index - 1) / 2)

    def push(self, priority: Any, val: Any) -> None:
        """
        Add an item into the heap
        :param priority: Priority of node to add
        :param val: Value of node to add
        :return:
        """
        if self._is_min:
            self._data.append(MinNode(priority, val))

        else:
            self._data.append(MaxNode(priority, val))

        self.percolate_up(len(self._data) - 1)

    def pop(self) -> PriorityNode:
        """
        Removes the top priority node from the heap
        :return: The root of the heap (MinNode or MaxNode)
        """
        if len(self._data) < 1:
            return None

        popped = self._data[0]
        if len(self._data) > 1:
            self._data[0] = self._data.pop()
            self.percolate_down(0)
        else:
            self._data.pop()

        return popped

    def get_minmax_child_index(self, index: int) -> int:
        """
        Gets the parent's minimum or maximum child
        :param index: Index to find min/max child of
        :return: Returns index of min/max child
        """
        if index < 0 or index >= len(self._data):
            return None

        left = self.get_left_child_index(index)
        right = self.get_right_child_index(index)
        if right is None and left is not None:
            return left
        elif left is None and right is not None:
            return right
        elif right is None and left is None:
            return None

        if self._is_min:
            if self._data[left] < self._data[right]:
                return left
            else:
                return right

        else:
            if self._data[left] > self._data[right]:
                return left
            else:
                return right

    def percolate_up(self, index: int) -> None:
        """
        Moves a node in the queue/heap up to its correct position (level in the tree)
        :param index: Index of node to be percolated up
        :return: None
        """
        parent = self.get_parent_index(index)
        if index > 0 and self._data[index] < self._data[parent]:
            self._data[index], self._data[parent] = \
                self._data[parent], self._data[index]
            self.percolate_up(parent)

    def percolate_down(self, index: int) -> None:
        """
        Moves a node in the queue/heap down to its correct position (level in the tree).
        :param index: Index of node to be percolated down
        :return: None
        """
        if self.get_left_child_index(index) is not None:
            left = self.get_left_child_index(index)
            small_child = left
            if self.get_right_child_index(index) is not None:
                right = self.get_right_child_index(index)
                if self._data[right] < self._data[left]:
                    small_child = right
            if self._data[small_child] < self._data[index]:
                self._data[index], self._data[small_child] = \
                    self._data[small_child], self._data[index]
                self.percolate_down(small_child)


class MaxHeap:
    """
    Implementation of a max-heap - the highest value is at the front (root).

    Initializes a PriorityQueue with is_min set to False.

    Uses the priority queue to satisfy the min heap properties by initializing
    the priority queue as a max-heap, and then using value as both the priority
    and value.
    """

    #   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   DO NOT MODIFY the following attributes/functions
    #   Modify only below indicated line

    __slots__ = ['_pqueue']

    def __init__(self):
        """
        Constructs a priority queue as a max-heap
        """
        self._pqueue = PriorityQueue(False)

    def __str__(self) -> str:
        """
        Represents the max-heap as a string
        :return: string representation of the heap
        """
        # NOTE: This hides implementation details
        return F"MaxHeap [{', '.join(item.value for item in self._pqueue._data)}]"

    __repr__ = __str__

    def to_tree_str(self) -> str:
        """
        Generates string representation of heap in Breadth First Ordering Format
        :return: String to print
        """
        return self._pqueue.to_tree_str()

    def __len__(self) -> int:
        """
        Determine the amount of nodes on the heap
        :return: Length of the data inside the heap
        """
        return len(self._pqueue)

    def empty(self) -> bool:
        """
        Checks if the heap is empty
        :returns: True if empty, else False
        """
        return self._pqueue.empty()

    #   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   Modify below this line
    def peek(self) -> Any:
        """
        Get the max element's value
        :return: None if heap is empty, else root's value
        """
        peeked = self._pqueue.peek()
        if peeked is None:
            return None
        return peeked.value

    def push(self, val: Any) -> None:
        """
        Inserts a node with the specified value onto the heap
        :param val: Value to push onto the heap
        :return: None
        """
        self._pqueue.push(val, val)

    def pop(self) -> Any:
        """
        Removes the max element from the heap
        :return: The value of the max element
        """
        if len(self._pqueue) == 0:
            return None

        return self._pqueue.pop().value


class MinHeap(MaxHeap):
    """
    Implementation of a max-heap - the highest value is at the front (root).

    Initializes a PriorityQueue with is_min set to True.

    Inherits from MaxHeap because it uses the same exact functions, but instead
    has a priority queue with a min-heap.
    """

    #   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #   DO NOT MODIFY the following attributes/functions
    __slots__ = []

    def __init__(self):
        """
        Constructs a priority queue as a min-heap
        """
        super().__init__()
        self._pqueue._is_min = True


def heap_sort(array: List[Any]) -> None:
    """
    Sorts an array in-place using a max-heap
    :param array: Input array to be sorted in place
    :return: Nothing
    """
    heap = MaxHeap()
    for i in array:
        heap.push(i)

    for i in range(len(array) - 1, -1, -1):
        # print(heap)
        array[i] = heap.pop()


def current_medians(array: List[int]) -> List[int]:
    """
    Finds the median of list while each index of array is added to it
    :param array: Input of data to be read in
    :return: List of medians as they are added
    """
    res = []
    medians = []
    below = MaxHeap()
    above = MinHeap()
    for i in array:
        # adding the number to the proper heap
        if below.empty() or i < below.peek():
            below.push(i)
        else:
            above.push(i)

        # shifting the heaps around to balance
        size_diff = len(below) - len(above)
        if size_diff >= 2:
            above.push(below.pop())
        elif size_diff <= -2:
            below.push(above.pop())

        # finally getting the medians
        size_diff = len(below) - len(above)
        if abs(size_diff) == 0:
            middle = (below.peek() + above.peek()) / 2

        elif size_diff < 0:
            middle = above.peek()

        else:
            middle = below.peek()

        medians.append(middle)
        # if len(res) % 2 == 0:
        #     middle = int(len(res) / 2) - 1
        #     medians.append((res[middle] + res[middle + 1]) / 2)
        # else:
        #     medians.append(float(res[int(len(res) / 2) - 1]))

    return medians
