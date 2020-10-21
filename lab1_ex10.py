def main():
    input_string = 'Welcome to USA. Awesome usa , isn\'t it?'
    print(ex10(input_string))


def ex10(in1):
    substring = 'USA'
    temp_string = in1.upper()
    count = temp_string.count(substring)
    return 'The ' + substring + ' count is: ' + str(count)


if __name__ == '__main__':
    main()
