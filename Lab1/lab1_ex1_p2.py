def main():
    list1 = [1, 23, 34, 554, 2334, 3534, 34]
    list2 = [23, 34, 554, 2334, 3534, 34, 54]

    ex1(list1, list2)


def ex1(listone, listtwo):
    list3 = list()
    listodd = listone[1::2]
    listeven = listtwo[0::2]
    list3.extend(listodd)
    list3.extend(listeven)
    print(list3)


if __name__ == '__main__':
    main()
