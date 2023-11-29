"""
Project 1
CSE 331 S21 (Onsay)
Preston Harrell
DLL.py
"""

from typing import TypeVar, List, Tuple
import datetime

# for more information on typehinting, check out https://docs.python.org/3/library/typing.html
T = TypeVar("T")  # represents generic type
Node = TypeVar("Node")  # represents a Node object (forward-declare to use in Node __init__)


# pro tip: PyCharm auto-renders docstrings (the multiline strings under each function definition)
# in its "Documentation" view when written in the format we use here. Open the "Documentation"
# view to quickly see what a function does by placing your cursor on it and using CTRL + Q.
# https://www.jetbrains.com/help/pycharm/documentation-tool-window.html


class Node:
    """
    Implementation of a doubly linked list node.
    Do not modify.
    """
    __slots__ = ["value", "next", "prev"]

    def __init__(self, value: T, next: Node = None, prev: Node = None) -> None:
        """
        Construct a doubly linked list node.

        :param value: value held by the Node.
        :param next: reference to the next Node in the linked list.
        :param prev: reference to the previous Node in the linked list.
        :return: None.
        """
        self.next = next
        self.prev = prev
        self.value = value

    def __repr__(self) -> str:
        """
        Represents the Node as a string.

        :return: string representation of the Node.
        """
        return str(self.value)

    def __str__(self) -> str:
        """
        Represents the Node as a string.

        :return: string representation of the Node.
        """
        return str(self.value)


class DLL:
    """
    Implementation of a doubly linked list without padding nodes.
    Modify only below indicated line.
    """
    __slots__ = ["head", "tail", "size"]

    def __init__(self) -> None:
        """
        Construct an empty doubly linked list.

        :return: None.
        """
        self.head = self.tail = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the DLL as a string.

        :return: string representation of the DLL.
        """
        result = ""
        node = self.head
        while node is not None:
            result += str(node)
            if node.next is not None:
                result += " <-> "
            node = node.next
        return result

    def __str__(self) -> str:
        """
        Represent the DLL as a string.

        :return: string representation of the DLL.
        """
        return repr(self)

    # MODIFY BELOW #

    def empty(self) -> bool:
        """
        Return boolean indicating whether DLL is empty.

        Suggested time & space complexity (respectively): O(1) & O(1).

        :return: True if DLL is empty, else False.
        """
        if self.size == 0:
            return True
        return False

    def push(self, val: T, back: bool = True) -> None:
        """
        Create Node containing `val` and add to back (or front) of DLL. Increment size by one.

        Suggested time & space complexity (respectively): O(1) & O(1).
        :param val: value to be added to the DLL.
        :param back: if True, add Node containing value to back (tail-end) of DLL;
            if False, add to front (head-end).
        :return: None.
        """
        # Accounting for empty DLL
        if self.size == 0:
            self.head = Node(val, self.head)
            self.tail = self.head

        # Push to the back
        elif back:
            self.tail.next = Node(val, None, self.tail)
            self.tail = self.tail.next

        # Push to the front
        else:
            self.head.prev = Node(val, self.head)
            self.head = self.head.prev

        self.size += 1

    def pop(self, back: bool = True) -> None:
        """
        Remove Node from back (or front) of DLL. Decrement size by 1. If DLL is empty, do nothing.

        Suggested time & space complexity (respectively): O(1) & O(1).

        :param back: if True, remove Node from (tail-end) of DLL;
            if False, remove from front (head-end).
        :return: None.
        """
        # Circumvent everything if empty
        if self.size != 0:
            # If only 1 item, reset DLL
            if self.size == 1:
                self.head = None
                self.tail = None

            # Removing back item
            elif back:
                self.tail.prev.next = None
                self.tail = self.tail.prev

            # Removing front item
            else:
                self.head = self.head.next
                self.head.prev = None

            self.size -= 1

    def from_list(self, source: List[T]) -> None:
        """
        Construct DLL from a standard Python list.

        Suggested time & space complexity (respectively): O(n) & O(n).

        :param source: standard Python list from which to construct DLL.
        :return: None.
        """
        # Circumvent if empty list
        if len(source) != 0:
            self.size = len(source)

            # Establishing DLL beginning + index we're looking at
            index = Node(source[0])
            self.head = index

            if len(source) > 1:
                # Making node after index
                next = Node(source[1])
                index.next = next
                next.prev = index

                for i in range(2, len(source)):
                    index = index.next
                    next = Node(source[i])

                    next.prev = index
                    index.next = next

                self.tail = next

            # Special case for size of 1
            else:
                self.tail = Node(source[-1])

    def to_list(self) -> List[T]:
        """
        Construct standard Python list from DLL.

        Suggested time & space complexity (respectively): O(n) & O(n).

        :return: standard Python list containing values stored in DLL.
        """
        index = self.head
        res = []

        for i in range(self.size):
            res.append(index.value)
            index = index.next

        return res

    def find(self, val: T) -> None:
        """
        Find first instance of `val` in the DLL and return associated Node object.

        Suggested time & space complexity (respectively): O(n) & O(1).

        :param val: value to be found in DLL.
        :return: first Node object in DLL containing `val`.
            If `val` does not exist in DLL, return None.
        """
        index = self.head
        for i in range(self.size):
            if index.value == val:
                return index
            index = index.next

        return None

    def find_all(self, val: T) -> List[Node]:
        """
        Find all instances of `val` in DLL and return Node objects in standard Python list.

        Suggested time & space complexity (respectively): O(n) & O(n).

        :param val: value to be searched for in DLL.
        :return: Python list of all Node objects in DLL containing `val`.
            If `val` does not exist in DLL, return empty list.
        """
        res = []
        index = self.head
        for i in range(self.size):
            if index.value == val:
                res.append(index)
            index = index.next

        return res

    def delete(self, val: T) -> bool:
        """
        Delete first instance of `val` in the DLL.

        Suggested time & space complexity (respectively): O(n) & O(1).

        :param val: value to be deleted from DLL.
        :return: True if Node containing `val` was deleted from DLL; else, False.
        """
        index = self.head
        for i in range(self.size):
            if index.value == val:
                # Case w/ index in middle of list
                if 0 < i < self.size - 1:
                    index.prev.next = index.next
                    index.next.prev = index.prev

                # Case w/ list of size 1
                elif i == self.size - 1 == 0:
                    self.head = None
                    self.tail = None

                # Case w/ index at end of list
                elif i == self.size - 1:
                    self.tail = index.prev
                    index.prev.next = None

                # Case w/ index at beginning of list
                elif i == 0:
                    self.head = index.next
                    index.next.prev = None

                self.size -= 1
                return True

            # Iterating through the DLL
            index = index.next

        return False

    def delete_all(self, val: T) -> int:
        """
        Delete all instances of `val` in the DLL.

        Suggested time & space complexity (respectively): O(n) & O(1).

        :param val: value to be deleted from DLL.
        :return: integer indicating the number of Nodes containing `val` deleted from DLL;
                 if no Node containing `val` exists in DLL, return 0.
        """
        index = self.head
        orig_size = self.size
        count = 0
        for i in range(orig_size):
            if index.value == val:
                # Case w/ index in middle of list
                # "index != self.head" to account for changing head
                if 0 < i < orig_size - 1 and index != self.head:
                    index.prev.next = index.next
                    index.next.prev = index.prev

                # Case w/ list of size 1 at time of check
                elif self.size == 1:
                    self.head = None
                    self.tail = None

                # Case w/ index at end of list
                elif i == orig_size - 1:
                    self.tail = index.prev
                    index.prev.next = None

                # Case w/ index at beginning of list
                elif index == self.head:
                    self.head = index.next
                    self.head.prev = None

                self.size -= 1
                count += 1

            # Iterating through the DLL
            if index is not None:
                index = index.next

        return count

    def reverse(self) -> None:
        """
        Reverse DLL in-place by modifying all `next` and `prev` references of Nodes in the
        DLL and resetting the `head` and `tail` references.
        Must be implemented in-place for full credit. May not create new Node objects.

        Suggested time & space complexity (respectively): O(n) & O(1).

        :return: None.
        """
        # Only need process if > 1
        # If size is 0 or 1, returns list as is
        if self.size > 1:
            # Establish iterator
            index = self.tail

            # Make last index the first
            self.head = index

            # Swapping values for head
            index.next = index.prev
            index.prev = None

            for i in range(1, self.size - 1):
                # Iterating through list
                index = index.next

                # Swapping prev/next node values
                next = index.next
                index.next = index.prev
                index.prev = next

            # Swapping values for tail
            index = index.next
            self.tail = index
            index.prev = index.next
            index.next = None


class Stock:
    """
    Implementation of a stock price on a given day.
    Do not modify.
    """

    __slots__ = ["date", "price"]

    def __init__(self, date: datetime.date, price: float) -> None:
        """
        Construct a stock.

        :param date: date of stock.
        :param price: the price of the stock at the given date.
        """
        self.date = date
        self.price = price

    def __repr__(self) -> str:
        """
        Represents the Stock as a string.

        :return: string representation of the Stock.
        """
        return f"<{str(self.date)}, ${self.price}>"

    def __str__(self) -> str:
        """
        Represents the Stock as a string.

        :return: string representation of the Stock.
        """
        return repr(self)


def intellivest(stocks: DLL) -> Tuple[datetime.date, datetime.date, int]:
    """
    Given a DLL representing daily stock prices,
    find the optimal streak of days over which to invest.
    To be optimal, the streak of stock prices must:

        (1) Be strictly increasing, such that the price of the stock on day i+1
        is greater than the price of the stock on day i, and
        (2) Have the greatest total increase in stock price from
        the first day of the streak to the last.

    In other words, the optimal streak of days over which to invest is the one over which stock
    price increases by the greatest amount, without ever going down (or staying constant).

    Suggested time & space complexity (respectively): O(n) & O(1).

    :param stocks: DLL with Stock objects as node values, as defined above.
    :return: Tuple with the following elements:
        [0]: date: The date at which the optimal streak begins.
        [1]: date: The date at which the optimal streak ends.
        [2]: float: The (positive) change in stock price between the start and end
                dates of the streak.
    """
    # Special case for empty DLL
    if stocks.size == 0:
        return None, None, 0

    # Special case for DLL of size 1
    if stocks.size == 1:
        return stocks.head.value.date, stocks.head.value.date, 0

    # Initializing default values
    start = stocks.head
    tmp_start = None
    end = None

    diff = 0
    curr_diff = 0

    index = stocks.head
    next = index.next
    streak = False

    for i in range(1, stocks.size):
        # Storing the price difference between index and next
        curr_diff += next.value.price - index.value.price

        # Checks if it is increasing from current day to next
        if index.value.price < next.value.price:
            # Determining whether this is the start of middle of streak
            if not streak:
                tmp_start = index
                streak = True

            # Replacing the current values if this streak has a larger diff
            if curr_diff > diff:
                start = tmp_start
                end = next
                diff = curr_diff

        # Resetting for decrease in price from index to next
        else:
            streak = False
            curr_diff = 0

        # Iterating...
        index = index.next
        next = index.next

    # Special case for if there is no streak
    if end is None:
        return start.value.date, start.value.date, 0

    return start.value.date, end.value.date, diff
