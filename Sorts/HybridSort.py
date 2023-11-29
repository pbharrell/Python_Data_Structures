"""
Name:
Project 3 - Hybrid Sorting
Developed by Sean Nguyen and Andrew Haas
Based on work by Zosha Korzecke and Olivia Mikola
CSE 331 Spring 2021
Professor Sebnem Onsay
"""
from typing import TypeVar, List, Callable, Tuple
from copy import deepcopy
from math import sqrt

T = TypeVar("T")  # represents generic type


def merge_sort(data: List[T], threshold: int = 0,
               comparator: Callable[[T, T], bool] = lambda x, y: x <= y) -> int:
    """
    Merge sort implementation
    :param data: Input list of data
    :param threshold: When list size below this threshold, switch to insertion sort
    :param comparator: Comparator that dictates the way the list is sorted
    :return: Number of inversions
    """

    def merge(first_half, last_half, data, comparator: Callable[[T, T], bool]) -> int:
        """
        Inner merge function that compares and merges the split lists
        :param first_half: First half of data
        :param last_half: Second half of data
        :param data: merge_sort input list
        :param comparator: Lambda that dictates the way the list is sorted
        :return: Number of inversions
        """
        i = j = 0
        inversions = 0
        while i + j < len(data):
            if j == len(last_half) or (i < len(first_half) and comparator(first_half[i], last_half[j])):
                data[i + j] = first_half[i]
                i = i + 1
            else:
                data[i + j] = last_half[j]
                j = j + 1
                inversions += len(data) // 2 - i

        return inversions

    inversions = 0
    n = len(data)
    if n < 2:
        return inversions
    elif n <= threshold:
        insertion_sort(data, comparator)
        return inversions

    mid = n // 2
    first_half = data[0:mid]
    last_half = data[mid:n]
    inversions += merge_sort(first_half, threshold, comparator)
    inversions += merge_sort(last_half, threshold, comparator)
    inversions += merge(first_half, last_half, data, comparator)
    return inversions


def insertion_sort(data: List[T], comparator: Callable[[T, T], bool] = lambda x, y: x <= y) -> None:
    """
    Insertion sort implementation
    :param data: Input list of data
    :param comparator: Comparator that dictates the way the list is sorted
    """

    def exchange(data, index1, index2):
        """
        Exchanges two indices
        :param data: input data
        :param index1: first index to exchange
        :param index2: second index to exchange
        """
        data[index1], data[index2] = data[index2], data[index1]

    n = len(data)
    for j in range(1, n):
        i = j
        while (i > 0) and (comparator(data[i], data[i - 1])):
            exchange(data, i, i - 1)
            i -= 1


def hybrid_sort(data: List[T], threshold: int,
                comparator: Callable[[T, T], bool] = lambda x, y: x <= y) -> None:
    """
    Hybrid sort shell for merge sort
    :param data: Input list of data
    :param threshold: When list size below this threshold, switch to insertion sort
    :param comparator: Comparator that dictates the way the list is sorted
    """
    merge_sort(data, threshold, comparator)


def inversions_count(data: List[T]) -> int:
    """
    Gets number of inversions for a list
    :param data: Input list of data
    """
    copy = deepcopy(data)
    return merge_sort(copy)


def reverse_sort(data: List[T], threshold: int) -> None:
    """
    Sorts input list in reverse order
    :param data: Input list of data
    :param threshold: When list size below this threshold, switch to insertion sort
    """
    merge_sort(data, comparator=lambda x, y: y <= x)


def password_rate(password: str) -> float:
    """
    Calculates rate of a password
    :param password: Password to get rate
    :return: Float rate value of password
    """
    p = len(password)
    u_set = set()
    ascii_pass = []
    for i in password:
        u_set.add(i)
        ascii_pass.append(ord(i))

    u = len(u_set)
    c = inversions_count(ascii_pass)
    return sqrt(p) * sqrt(u) + c


def password_sort(data: List[str]) -> None:
    """
    Sorts input list of passwords by their rate
    :param data: Input list of passwords
    """
    data_tups = []
    for password in data:
        rate = password_rate(password)
        data_tups.append((rate, password))

    comp = lambda x, y: x > y
    merge_sort(data_tups, comparator=comp)

    res = []
    for i in range(len(data_tups)):
        data[i] = data_tups[i][1]