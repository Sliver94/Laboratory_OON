def main():
    s = 'M@|\/||\/|T F0tt'
    print(ex9(s))


def ex9(in1):
    dict = {'lower_case': 0, 'upper_case': 0, 'digits': 0, 'special_cases': 0}
    for char in in1:
        if char.islower():
            dict['lower_case'] = dict['lower_case'] + 1
        elif char.isupper():
            dict['upper_case'] = dict['upper_case'] + 1
        elif char.isnumeric():
            dict['digits'] = dict['digits'] + 1
        else:
            dict['special_cases'] = dict['special_cases'] + 1
    return dict


if __name__ == '__main__':
    main()
