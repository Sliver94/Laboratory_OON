def main():
    input_str = ' pynativepynvepynative '
    print(ex12(input_str))


def ex12(in1):
    count_dict = dict()
    for char in in1:
        count = in1.count(char)
        count_dict[char] = count
    return count_dict


if __name__ == '__main__':
    main()
