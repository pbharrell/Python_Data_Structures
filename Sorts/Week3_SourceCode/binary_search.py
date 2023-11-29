"""Return True if target is found in indicated portion of a Python list.
  The search only considers the portion from data[low] to data[high] inclusive.
  """


def binarySearch(data, value, low_index, high_index):
  """
  Finds the target value in a list
  :param high_index:
  :param low_index:
  :param data:
  :param value: the value to find
  :return: Boolean
  """

  if low_index > high_index:
    return False
  else:
    mid = (low_index + high_index) // 2
    if value == data[mid]:
      return True
    elif value < data[mid]:
      return binarySearch(data, value, low_index, mid - 1)
    else:
      return binarySearch(data, value, mid + 1, high_index)


# client code:
a_list = [5, 12, 18, 19, 26, 46, 64, 78, 90, 120]

low = 0
high = len(a_list) - 1
target = 35

if binarySearch(a_list, target, low, high):
    print("Target value found")
else:
    print("Target value not found")
