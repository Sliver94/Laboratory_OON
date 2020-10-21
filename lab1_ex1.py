def main():
    in1 = int(input('Primo numero: '))
    in2 = int(input('Secondo numero: '))
    print(ex1(in1, in2))
    return


def ex1(a, b):
    if a*b > 1000:
        return 'Il prodotto è maggiore di 1000 -> In1 + In2 = ' + str(a+b)
    else:
        return 'Il prodotto è minore di 1000 -> In1 * In2 = ' + str(a*b)


if __name__ == '__main__':
    main()

