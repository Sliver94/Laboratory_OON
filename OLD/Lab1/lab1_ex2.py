def main():
    x = int(input('Inserire il numero: '))
    ex2(x)


def ex2(n):
    for i in range(n):
        print(2*i+1)


if __name__ == '__main__':
    main()
