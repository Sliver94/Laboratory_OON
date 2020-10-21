def ex1(a, b):
    if a*b > 1000:
        return a+b
    else:
        return a*b


def main():
    a = input()
    b = input()
    z = ex1(a, b)
    print(z)
    print('ciao')
    return 0

