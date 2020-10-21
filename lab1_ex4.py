def main():
    x = []
    lung = int(input('Inserire lunghezza lista: '))
    for i in range(lung):
        x.append(int(input('Inserire l\'elemento '+str(i)+': ')))
    ex4(x)


def ex4(a):
    for i in range(len(a)-1):
        if a[i] % 5 == 0:
            print('L\'elemento ', i, ' è divisibile per 5 e vale: ', a[i])


if __name__ == '__main__':
    main()
