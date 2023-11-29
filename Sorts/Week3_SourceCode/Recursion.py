

def recurse(n):
    if n <3:
        return 80
    a=n*2
    val1=recurse(n/3)
    val2=recurse(n/3)
    return val1+val2




#print(recurse(5))



def recurse2(n):
    if n <3:
        return 80

    for i in range(n):
        print(i)

    val1=recurse(n/3)
    val2=recurse(n/3)
    return val1+val2



def recurse3(n):
    if n <3:
        return 80

    for i in range(n):
        print(i)

    val1=recurse(n/4)
    val2=recurse(n/4)
    val3=recurse3(n/4)

    return val1+val2*val3


print(recurse2(5))
