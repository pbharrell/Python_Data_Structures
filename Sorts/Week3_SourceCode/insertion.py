def exchange(a, i, j):
    a[i], a[j] = a[j], a[i]


def insertion_sort(a):
    n = len(a)
    for j in range(1, n):
        i = j
        while (i > 0) and (a[i] < a[i - 1]):
            exchange(a, i, i - 1)
            i -= 1


def main():
    a = [6,5,4,3,2,1,0]
    insertion_sort(a)
    print(a)
    b=[1,2,3,4,5,6]
    insertion_sort(b)
    print(b)

if __name__ == '__main__': main()
