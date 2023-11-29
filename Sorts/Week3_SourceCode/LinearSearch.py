


def linearSearch(mylist, target):

    for i in range(len(mylist)):
        if mylist[i] == target:
            return True
    return False










theList = [4, 2, 7, 5, 12, 54, 21, 64, 12, 32]
print(linearSearch(theList, 54))


# print('List has the items: ', theList)
# searchItem = int(input('Enter a number to search for: '))
# linearSearch(theList,searchItem)
#
# someother_Search(theList,searchItem)
#
