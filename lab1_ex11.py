def main():
    s = 'English = 78 Science = 83 Math = 68 History = 65'
    print(ex11(s))


def ex11(in1):
    numbers = []
    words = in1.split()
    for i in range(len(words)):
        if words[i].isnumeric():
            numbers.append(int(words[i]))
    somma = sum(numbers)
    media = somma / len(numbers)
    out1 = [somma, media]
    return out1


if __name__ == '__main__':
    main()
