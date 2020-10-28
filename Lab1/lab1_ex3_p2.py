def main():
    sampleList = [11, 45, 8, 23, 14, 12, 78, 45, 89]

    ex3(sampleList)


def ex3(sample_list):

    print('Original list ', sample_list)
    length = len(sample_list)
    chunk_size = int(length / 3)
    start = 0
    end = chunk_size
    for i in range(1, chunk_size + 1):
        indexes = slice(start, end, 1)
        list_chunk = sample_list[indexes]
        print('Chunk ', i, list_chunk)
        print('After reversing it ', list(reversed(list_chunk)))
        start = end
        if i < chunk_size:
            end += chunk_size
        else:
            end += length - chunk_size


if __name__ == '__main__':
    main()
