def main():
    x = [1, 2134, 4, 89, 354, 7818]
    y = [854854, 545, 1210, 4, 8]
    print(ex6(x, y))


def ex6(in1, in2):
    out1 = []
    for i in range(len(in1)):
        if in1[i] % 2 == 1:
            out1.append(in1[i])
    for i in range(len(in2)):
        if in2[i] % 2 == 0:
            out1.append(in2[i])
    return out1


if __name__ == '__main__':
    main()
