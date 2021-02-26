def main():
    sampleList = [11, 45, 8, 11, 23, 45, 23, 45, 89]

    ex4(sampleList)


def ex4(sample_list):

    print('Original list ', sample_list)
    count_dict = dict()
    for element in sample_list:
        if element in count_dict:
            count_dict[element] += 1
        else:
            count_dict[element] = 1
    print(count_dict)


if __name__ == '__main__':
    main()
