def main():
    triangle1 = Triangle(3, 4, 5)
    triangle2 = Triangle(30, 30, 30)
    print('Triangle 1 is: ', triangle1.a, triangle1.b, triangle1.c)
    print('Triangle 2 is: ', triangle2.a, triangle2.b, triangle2.c)
    print('Triangle 1 is equilateral - ', triangle1.is_equilateral())
    print('Triangle 2 is equilateral - ', triangle2.is_equilateral())
    print('Triangle 1 perimeter is: ', Triangle.perimeter(triangle1))
    print('Triangle 1 perimeter is: ', Triangle.perimeter(triangle2))


class Triangle:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def is_equilateral(self):   #Instance method:
        return (self.a == self.b) and (self.a == self.c)

    @staticmethod   #static method
    def perimeter(triangle):
        return triangle.a + triangle.b + triangle.c


if __name__ == '__main__':
    main()
