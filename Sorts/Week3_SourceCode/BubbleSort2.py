def bubble_sort(mylist):
    for i in range(len(mylist)):
        for k in range(len(mylist) - 1, i, -1):
            if mylist[k] < mylist[k - 1]:
                swap(mylist, k, k - 1)


def swap(A, x, y):
    tmp = A[x]
    A[x] = A[y]
    A[y] = tmp


A = [5, 4, 3, 2, 1, 0]
bubble_sort(A)
print(A)
