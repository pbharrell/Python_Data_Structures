"""
Project 1
CSE 331 S21 (Onsay)
Preston Harrell
SLL.py
"""

from SLL.Node import Node  # Import `Node` class
from typing import TypeVar  # For use in type hinting
from copy import deepcopy  # For use in crafting

# Type Declarations
T = TypeVar('T')  # generic type
SLL = TypeVar('SLL')  # forward declared


class RecursiveSinglyLinkList:
    """
    Recursive implementation of an SLL
    """

    __slots__ = ['head']

    def __init__(self) -> None:
        """
        Initializes an `SLL`
        :return: None
        """
        self.head = None

    def __repr__(self) -> str:
        """
        Represents an `SLL` as a string
        """
        return self.to_string(self.head)

    def __str__(self) -> str:
        """
        Represents an `SLL` as a string
        """
        return self.to_string(self.head)

    def __eq__(self, other: SLL) -> bool:
        """
        Overloads `==` operator to compare SLLs
        :param other: right hand operand of `==`
        :return: `True` if equal, else `False`
        """
        comp = lambda n1, n2: n1 == n2 and (comp(n1.next, n2.next) if (n1 and n2) else True)
        return comp(self.head, other.head)

    # ============ Modify below ============ #

    def to_string(self, curr: Node) -> str:
        """
        String conversion of SLL list
        :param curr: head node of SLL
        :return: string representation of SLL
        Time Complexity: O(n^2)
        """
        res = ""

        if self.head is None:
            return "None"

        if curr.next is None:
            res += str(curr.val)
            return res


        res += str(curr.val) + " --> " + self.to_string(curr.next)
        return res

    def length(self, curr: Node) -> int:
        """
        Outputs length of SLL
        :param curr: head of SLL
        :return: length of SLL
        Time complexity: O(n)
        """
        if self.head is None:
            return 0

        elif curr.next is None:
            return 1

        return 1 + self.length(curr.next)

    def sum_list(self, curr: Node) -> T:
        """
        Creates total sum of all items in SLL
        :param curr: head of SLL
        :return: sum of all items in SLL
        Time complexity: O(n)
        """
        if self.head is None:
            return 0

        elif curr.next is None:
            return curr.val

        return curr.val + self.sum_list(curr.next)

    def push(self, value: T) -> None:
        """
        Adds value to end of SLL
        :param value: value to add to SLL
        :return: None
        Time complexity: O(n)
        """

        def push_inner(curr: Node) -> None:
            """
            Iterates through SLL and makes new node
            :param curr: node at head of SLL
            :return: None
            Time complexity: O(n)
            """
            if curr.next is None:
                curr.next = Node(value)

            else:
                push_inner(curr.next)

        if self.head is None:
            self.head = Node(value)

        else:
            push_inner(self.head)

    def remove(self, value: T):
        """
        Removes first instance of given value
        :param value: value to remove
        :return: None
        Time complexity: O(n)
        """

        def remove_inner(curr: Node) -> Node:
            """
            Iterates through list and removes node with given value
            :param curr: head of SLL
            :return: None
            Time complexity: O(n)
            """
            if curr is self.head and curr.val == value:
                self.head = curr.next
                return

            if curr.next is None:
                return

            if curr.next.val == value:
                curr.next = curr.next.next
                return

            remove_inner(curr.next)

        if self.head is None:
            return

        remove_inner(self.head)

    def remove_all(self, value: T) -> None:
        """
        Removes all instances of given value
        :param value: value to remove
        :return: None
        Time complexity: O(n)
        """

        def remove_all_inner(curr: Node) -> Node:
            """
            Iterates through list and removes all nodes with given value
            :param curr: head of SLL
            :return: None
            Time complexity: O(n)
            """
            if curr is self.head and curr.val == value:
                self.head = curr.next
                if self.head is not None and self.head.val == value:
                    self.head = self.head.next
                    remove_all_inner(self.head)

            if curr is None or curr.next is None:
                return

            if curr.next.val == value:
                curr.next = curr.next.next

            remove_all_inner(curr.next)

        if self.head is None:
            return

        remove_all_inner(self.head)

    def search(self, value: T) -> bool:
        """
        Looks for given value in the given SLL
        :param value: value to search for
        :return: true if value is found
        Time complexity: O(n)
        """

        def search_inner(curr: Node) -> bool:
            """
            Looks for value in SLL
            :param curr: node at head of SLL
            :return: true if value is found
            Time complexity: O(n)
            """
            if curr.val == value:
                return True

            if curr.next is None:
                return False

            return search_inner(curr.next)

        if self.head is None:
            return False

        return search_inner(self.head)

    def count(self, value: T) -> int:
        """
        Counts how many times a given value occurs in SLL
        :param value: value to search for
        :return: number of times a given value occurs in SLL
        Time complexity: O(n)
        """

        def count_inner(curr) -> int:
            """
            Iterates through SLL and counts how many times a given value occurs
            :param curr: head of SLL
            :return: None
            Time complexity: O(n)
            """
            if curr is None:
                return 0

            if curr.val == value:
                return count_inner(curr.next) + 1

            return count_inner(curr.next)

        if self.head is None:
            return 0

        return count_inner(self.head)

    def reverse(self, curr: Node) -> Node:
        """
        Reverses given SLL
        :param curr: head of the SLL
        :return: head of the reversed list
        Time complexity: O(n)
        """
        if curr is None:
            return None

        elif curr.next is None:
            self.head = Node(curr.val)
            return self.head

        temp = self.reverse(curr.next)
        temp.next = Node(curr.val)
        return temp.next


def crafting(recipe: RecursiveSinglyLinkList, pockets: RecursiveSinglyLinkList) -> bool:
    """
    Returns true if crafting is possible and false if not. If true,
    removes those items from pockets.
    :param recipe: SLL containing recipe
    :param pockets: SLL containing what is in pockets (inventory)
    Time complexity: O(rp)
    """

    def recipe_check(pockets_list: RecursiveSinglyLinkList, recipe_item: Node):
        """
        Removes available items from recipe from pockets
        :param pockets_list: SLL representing your pockets
        :param recipe_item: head of recipe SLL
        :return: None
        Time complexity: O(n)
        """
        pockets_list.remove(recipe_item.val)
        if recipe_item.next is not None:
            recipe_check(pockets_list, recipe_item.next)

    if recipe.length(recipe.head) == 0 or pockets.length(pockets.head) == 0:
        return False

    copy = RecursiveSinglyLinkList()
    copy.head = deepcopy(pockets.head)

    recipe_check(copy, recipe.head)

    if copy.length(copy.head) == pockets.length(pockets.head) - recipe.length(recipe.head):
        recipe_check(pockets, recipe.head)
        return True

    return False
