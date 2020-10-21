def main():
    s1 = 'mt'
    s2 = 'amm'
    s3 = ex7(s1, s2)
    print(s3)


def ex7(in1, in2):
    out1 = in1[0:int(len(in1)/2)]+in2+in1[int(len(in1)/2):int(len(in1))]
    return out1


if __name__ == '__main__':
    main()
