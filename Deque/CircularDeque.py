"""
Project 4
CSE 331 S21 (Onsay)
Name
CircularDeque.py
"""

from __future__ import annotations
from typing import TypeVar, List, Tuple, Union, Any

# from re import split as rsplit
# import re

# for more information on typehinting, check out https://docs.python.org/3/library/typing.html
T = TypeVar("T")  # represents generic type
CircularDeque = TypeVar("CircularDeque")  # represents a CircularDeque object


class CircularDeque:
    """
    Class representation of a Circular Deque
    """

    __slots__ = ['capacity', 'size', 'queue', 'front', 'back']

    def __init__(self, data: List[T] = [], capacity: int = 4):
        """
        Initializes an instance of a CircularDeque
        :param data: starting data to add to the deque, for testing purposes
        :param capacity: amount of space in the deque
        """
        self.capacity: int = capacity
        self.size: int = len(data)

        self.queue: list[T] = [None] * capacity
        self.front: int = None
        self.back: int = None

        for index, value in enumerate(data):
            self.queue[index] = value
            self.front = 0
            self.back = index

    def __str__(self) -> str:
        """
        Provides a string represenation of a CircularDeque
        :return: the instance as a string
        """
        if self.size == 0:
            return "CircularDeque <empty>"

        string = f"CircularDeque <{self.queue[self.front]}"
        current_index = self.front + 1 % self.capacity
        while current_index <= self.back:
            string += f", {self.queue[current_index]}"
            current_index = (current_index + 1) % self.capacity
        return string + ">"

    def __repr__(self) -> str:
        """
        Provides a string represenation of a CircularDeque
        :return: the instance as a string
        """
        return str(self)

    # ============ Modify below ============ #

    def __len__(self) -> int:
        """
        Gets the length of the circular dequeue
        :return: Size of dequeue as an int
        """
        return self.size

    def is_empty(self) -> bool:
        """
        Gets if the dequeue is empty
        :return: Boolean that is true if empty, false if not
        """
        return self.size == 0

    def front_element(self) -> T:
        """
        Gets the front element of the dequeue
        :return: Front element of the dequeue
        """
        if self.front is None:
            return None

        return self.queue[self.front]

    def back_element(self) -> T:
        """
        Gets the back element of the dequeue
        :return: Back element of the dequeue
        """
        if self.back is None:
            return None

        return self.queue[self.back]

    def front_enqueue(self, value: T) -> None:
        """
        Adds a value to the front of the dequeue
        :param value: Value to be added
        """
        if self.size == 0:
            self.front = 0
            self.back = 0
            self.queue[self.front] = value

        elif self.front == 0:
            self.front = self.capacity - 1
            self.queue[self.front] = value

        else:
            self.front -= 1
            self.queue[self.front] = value

        self.size += 1
        self.grow()

    def back_enqueue(self, value: T) -> None:
        """
        Adds an element to the back of the dequeue
        :param value: Value to be added to the dequeue
        """
        if self.size == 0:
            self.front = 0
            self.back = 0
            self.queue[self.back] = value

        elif self.back == self.capacity - 1:
            self.back = 0
            self.queue[self.back] = value

        else:
            self.back += 1
            self.queue[self.back] = value

        self.size += 1
        self.grow()

    def front_dequeue(self) -> T:
        """
        Removes item from the front of the dequeue
        :return: Item removed
        """
        res = self.queue[self.front]
        if self.front == self.capacity - 1:
            self.queue[self.front] = None
            self.front = 0

        elif self.size == 0:
            return res

        else:
            self.queue[self.front] = None
            self.front += 1

        self.size -= 1
        self.shrink()
        return res

    def back_dequeue(self) -> T:
        """
        Removes item from back of dequeue
        :return: Item that was removed
        """
        res = self.queue[self.back]
        if self.back == 0:
            self.queue[self.back] = None
            self.back = self.capacity - 1

        elif self.size == 0:
            return res

        else:
            self.queue[self.back] = None
            self.back -= 1

        self.size -= 1
        self.shrink()
        return res

    def grow(self) -> None:
        """
        Grows the capacity of the dequeue if necessary
        """
        if self.size == self.capacity:
            self.capacity *= 2

            if self.front == 0:
                self.queue.extend([None] * (self.capacity // 2))

            else:
                tmp = [None] * self.capacity
                for i in range(self.size):
                    index = (self.front + i) % self.size
                    tmp[i] = self.queue[index]

                self.front = 0
                self.back = 0 + self.size - 1
                self.queue = tmp

    def shrink(self) -> None:
        """
        Shrinks the size of the dequeue if necessary
        """
        if self.size <= self.capacity / 4 \
                and self.capacity // 2 >= 4:

            tmp = [None] * (self.capacity // 2)
            for i in range(self.size):
                index = (self.front + i) % self.capacity
                tmp[i] = self.queue[index]

            self.capacity //= 2
            self.front = 0
            self.back = 0 + self.size - 1
            self.queue = tmp


def LetsPassTrains102(infix: str) -> Tuple[Union[Union[int, float, str], Any], ...]:
    """
    Converts standard in-fix notation to post-fix notation
    :param infix: String of a standard in-fix notation problem
    :return: Result of calculation and string of post-fix problem
    """

    def is_number(num):
        """
        Determines if input is number, including decimals and negatives
        :param num: Input to check if number
        :return: True if is a number, false if not
        """
        try:
            float(num)
            return True
        except ValueError:
            return False

    ops = {'*': 3, '/': 3,  # key: operator, value: precedence
           '+': 2, '-': 2,
           '^': 4,
           '(': 0}  # '(' is lowest bc must be closed by ')'

    operators = CircularDeque()

    pieces = infix.split()
    res = ""
    for i, piece in enumerate(pieces):
        if piece[0] == '(':
            while piece[0] == '(':
                operators.back_enqueue(piece[0])
                piece = piece[1:]

        if piece[-1] == ')':
            count = 0
            while piece[-1] == ')':
                piece = piece[:-1]
                count += 1

            res += piece + ' '
            for i in range(count):
                while operators.back_element() != '(':
                    res += operators.back_dequeue() + ' '

                operators.back_dequeue()

        elif is_number(piece):
            res += piece + ' '

        elif piece in ops.keys():
            while not operators.is_empty() and ops[operators.back_element()] >= ops[piece]:
                res += operators.back_dequeue() + ' '

            operators.back_enqueue(piece)

    if len(operators) > 0:
        for i in range(len(operators) - 1):
            res += operators.back_dequeue() + ' '

        res += operators.back_dequeue()

    result = 0
    pieces = res.split()
    numbers = CircularDeque()

    if len(pieces) == 1:
        result = int(pieces[0])

    else:
        for i, piece in enumerate(pieces):
            if is_number(piece):
                numbers.back_enqueue(float(piece))

            elif piece in ops.keys():
                if piece == '+':
                    result = numbers.front_dequeue() + numbers.front_dequeue()
                    numbers.back_enqueue(result)

                elif piece == '-':
                    num2 = numbers.back_dequeue()
                    if numbers.is_empty():
                        neg = True
                        index = i + 1
                        while not (pieces[index].isdigit() or \
                                   (piece[0] == '-' and piece[1:].isdigit())):
                            if pieces[index] == '-':
                                neg = not neg

                            index += 1

                        if neg:
                            pieces[index] = str(-int(pieces[index]))

                        numbers.front_enqueue(num2)

                    else:
                        num1 = numbers.back_dequeue()
                        result = num1 - num2
                        numbers.back_enqueue(result)

                elif piece == '*':
                    num2 = numbers.back_dequeue()
                    num1 = numbers.back_dequeue()
                    result = num1 * num2
                    numbers.back_enqueue(result)

                elif piece == '/':
                    num2 = numbers.back_dequeue()
                    num1 = numbers.back_dequeue()
                    result = num1 / num2
                    numbers.back_enqueue(result)

                elif piece == '^':
                    num2 = numbers.back_dequeue()
                    num1 = numbers.back_dequeue()
                    result = num1 ** num2
                    numbers.back_enqueue(result)

    return tuple([result, res.rstrip()])
