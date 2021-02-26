def main():
    s1 = 'mabaabm'
    s2 = 'moktko!'
    s3 = ex7(s1, s2)
    print(s3)


def ex7(in1, in2):
    out1 = in1[0]+in1[int(len(in1)/2)]+in1[int(len(in1)-1)]+in2[0]+in2[int(len(in2)/2)]+in2[int(len(in2)-1)]
    return out1


if __name__ == '__main__':
    main()
