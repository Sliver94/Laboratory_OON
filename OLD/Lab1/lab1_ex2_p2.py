def main():
    list = [1, 23, 34, 554, 2334, 3534, 34]

    ex2(list)


def ex2(lista):
    element = lista.pop(4)
    print(element)
    print(lista)
    lista.insert(1, element)
    print(lista)
    lista.append(element)
    print(lista)


if __name__ == '__main__':
    main()
