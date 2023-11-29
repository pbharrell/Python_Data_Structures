"""
Project 6
CSE 331 S21 (Onsay)
Your Name
hashtable.py
"""

from typing import TypeVar, List, Tuple

T = TypeVar("T")
HashNode = TypeVar("HashNode")
HashTable = TypeVar("HashTable")


class HashNode:
    """
    DO NOT EDIT
    """
    __slots__ = ["key", "value", "deleted"]

    def __init__(self, key: str, value: T, deleted: bool = False) -> None:
        self.key = key
        self.value = value
        self.deleted = deleted

    def __str__(self) -> str:
        return f"HashNode({self.key}, {self.value})"

    __repr__ = __str__

    def __eq__(self, other: HashNode) -> bool:
        return self.key == other.key and self.value == other.value

    def __iadd__(self, other: T) -> None:
        self.value += other


class HashTable:
    """
    Hash Table Class
    """
    __slots__ = ['capacity', 'size', 'table', 'prime_index']

    primes = (
        2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83,
        89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
        181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277,
        281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389,
        397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499,
        503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617,
        619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739,
        743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859,
        863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991,
        997)

    def __init__(self, capacity: int = 8) -> None:
        """
        DO NOT EDIT
        Initializes hash table
        :param capacity: capacity of the hash table
        """
        self.capacity = capacity
        self.size = 0
        self.table = [None] * capacity

        i = 0
        while HashTable.primes[i] <= self.capacity:
            i += 1
        self.prime_index = i - 1

    def __eq__(self, other: HashTable) -> bool:
        """
        DO NOT EDIT
        Equality operator
        :param other: other hash table we are comparing with this one
        :return: bool if equal or not
        """
        if self.capacity != other.capacity or self.size != other.size:
            return False
        for i in range(self.capacity):
            if self.table[i] != other.table[i]:
                return False
        return True

    def __str__(self) -> str:
        """
        DO NOT EDIT
        Represents the table as a string
        :return: string representation of the hash table
        """
        represent = ""
        bin_no = 0
        for item in self.table:
            represent += "[" + str(bin_no) + "]: " + str(item) + '\n'
            bin_no += 1
        return represent

    __repr__ = __str__

    def _hash_1(self, key: str) -> int:
        """
        ---DO NOT EDIT---
        Converts a string x into a bin number for our hash table
        :param key: key to be hashed
        :return: bin number to insert hash item at in our table, None if key is an empty string
        """
        if not key:
            return None
        hashed_value = 0

        for char in key:
            hashed_value = 181 * hashed_value + ord(char)
        return hashed_value % self.capacity

    def _hash_2(self, key: str) -> int:
        """
        ---DO NOT EDIT---
        Converts a string x into a hash
        :param key: key to be hashed
        :return: a hashed value
        """
        if not key:
            return None
        hashed_value = 0

        for char in key:
            hashed_value = 181 * hashed_value + ord(char)

        prime = HashTable.primes[self.prime_index]

        hashed_value = prime - (hashed_value % prime)
        if hashed_value % 2 == 0:
            hashed_value += 1
        return hashed_value

    def __len__(self) -> int:
        """
        Gets length of hashtable
        :returns: Size of hashtable
        """
        return self.size

    def __setitem__(self, key: str, value: T) -> None:
        """
        Sets adds or resets an item with key given
        :param key: Table key to that item
        :param value: Value to store at the key
        """
        self._insert(key, value)

    def __getitem__(self, key: str) -> T:
        """
        Gets item stored at that key
        :param key: Key to find value at
        :returns: Value stored at that key
        """
        temp = self._get(key)
        if temp is None:
            raise KeyError('Key Error: ' + repr(key))
        return temp.value

    def __delitem__(self, key: str) -> None:
        """
        Deletes value with associated key in hashtable
        :param key: Key to delete
        """
        temp = self._get(key)
        if temp is not None:
            self._delete(key)
        else:
            raise KeyError('Key Error: ' + repr(key))

    def __contains__(self, key: str) -> bool:
        """
        Checks if item at given key is in dictionary
        :param key: Key to search for
        """
        if self._get(key) is None:
            return False
        return True

    def hash(self, key: str, inserting: bool = False) -> int:
        """
        Converts key to index in hash table
        :param key: Key to convert to index in table
        :param inserting: Whether or not we are inserting
        :returns: Integer index of node in table
        """
        hashed = self._hash_1(key)
        hash_offset = self._hash_2(key)

        if inserting:
            while self.table[hashed] is not None and \
                    not self.table[hashed].deleted and \
                    not self.table[hashed].key == key:
                hashed = (hashed + hash_offset) % self.capacity
        else:
            while self.table[hashed] is not None and \
                    not self.table[hashed].key == key:
                hashed = (hashed + hash_offset) % self.capacity

        return hashed

    def _insert(self, key: str, value: T) -> None:
        """
        Insert a value into the hashtable
        :param key: Key for insertion
        :param value: Value to insert
        """
        new_load_factor = (self.size + 1) / self.capacity
        if new_load_factor >= .5:
            self._grow()

        index = self.hash(key, inserting=True)
        if self.table[index] is None:
            self.table[index] = HashNode(key, value)
            self.size += 1
            return

        self.table[index].key = key
        self.table[index].value = value

    def _get(self, key: str) -> HashNode:
        """
        Get the proper node when given the key
        :param key: Key to value
        :returns: The node corresponding to that key
        """
        return self.table[self.hash(key)]

    def _delete(self, key: str) -> None:
        """
        Deletes an item from a hash table with the given key
        :param key: Key of node to delete
        """
        node = self.table[self.hash(key)]
        node.deleted = True
        node.key = None
        node.value = None
        self.size -= 1

    def _grow(self) -> None:
        """
        Doubles and rehashes the list capacity
        """
        old = self.table
        self.table = [None] * (2 * len(old))
        self.size = 0
        self.capacity *= 2

        i = self.prime_index
        while HashTable.primes[i] <= self.capacity:
            i += 1
        self.prime_index = i - 1

        for i in old:
            if i is not None and not i.deleted:
                self._insert(i.key, i.value)

        i = self.prime_index
        while HashTable.primes[i] <= self.capacity:
            i += 1
        self.prime_index = i - 1

    def update(self, pairs: List[Tuple[str, T]] = []) -> None:
        """
        Add or update values with given list of keys and values
        :param pairs: Keys and corresponding values to insert
        """
        for i in pairs:
            self._insert(i[0], i[1])

    def keys(self) -> List[str]:
        """
        Makes a list of all the keys in the table
        :returns: A list of the keys in the hashtable
        """
        res = []
        for i in self.table:
            if i is not None:
                res.append(i.key)

        return res

    def values(self) -> List[T]:
        """
        Makes a list of all the values in the table
        :returns: A list of the values in the hashtable
        """
        res = []
        for i in self.table:
            if i is not None:
                res.append(i.value)

        return res

    def items(self) -> List[Tuple[str, T]]:
        """
        Gets all the keys and values in the table
        :returns: A list of tuples with all the keys and values
        """
        res = []
        for i in self.table:
            if i is not None:
                res.append((i.key, i.value))

        return res

    def clear(self) -> None:
        """
        Clear the hashtable
        """
        for i, node in enumerate(self.table):
            if node is not None:
                self.table[i] = None
        self.size = 0

class CataData:
    __slots__ = ["enter_table", "exit_table", "in_transit"]

    def __init__(self) -> None:
        """
        Constructs instantiation of CataData class
        """
        self.enter_table = HashTable()
        self.exit_table = HashTable()
        self.in_transit = HashTable()

    def enter(self, idx: str, origin: str, time: int) -> None:
        """
        Add data of a student entering the bus
        :param idx: Student name
        :param origin: Where student got on the bus
        :param time: Time of getting on bus
        """
        if idx in self.in_transit and \
                not self.in_transit[idx].deleted:
            old_origin = self.in_transit[idx].value
            self.enter_table[old_origin][idx].deleted = True
        else:
            self.in_transit[idx] = HashNode(idx, origin)

        if origin not in self.enter_table:
            self.enter_table[origin] = HashTable()
        self.enter_table[origin][idx] = HashNode(idx, time)

    def exit(self, idx: str, dest: str, time: int) -> None:
        """
        Add data of the student exiting the bus
        :param idx: Student name
        :param dest: Where student got off the bus
        :param time: Time of getting off bus
        """
        if idx not in self.in_transit or \
                self.in_transit[idx].deleted:
            return

        if dest not in self.exit_table:
            self.exit_table[dest] = HashTable()
        self.exit_table[dest][idx] = HashNode(idx, time)

    def get_average(self, origin: str, dest: str) -> float:
        """
        Average commute time of student from origin to destination
        :param origin: Origin of commute
        :param dest: Destination of commute
        :returns: Average commute time from origin to dest
        """
        if dest in self.exit_table:
            keys = self.exit_table[dest].keys()
        else:
            return 0

        res = []
        for key in keys:
            if key in self.enter_table[origin]:
                leave_time = self.enter_table[origin][key].value
                arrive_time = self.exit_table[dest][key].value
                diff = arrive_time - leave_time
                res.append(diff)

        if not res:
            return 0

        average = sum(res) / len(res)

        return average
