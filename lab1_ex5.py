def main():
    y = 'Emma is a good developer. Emma is also a writer. Good Emma'
    print(ex5(y))


def ex5(frase):
    count = 0
    for i in range(len(frase)-1):
        if frase[i:i+4] == 'Emma':
            count = count+1
    return count


if __name__ == '__main__':
    main()
