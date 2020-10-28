def main():
    x = []
    lung = int(input('Inserire lunghezza lista: '))
    for i in range(lung):
        x.append(int(input('Inserire l\'elemento '+str(i)+': ')))
    print('Il primo e l\'ultimo elemento sono uguali? ', ex3(x))


def ex3(a):
    if a[0] == a[-1]:
        return True
    else:
        return False


if __name__ == '__main__':
    main()
