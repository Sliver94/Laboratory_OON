import numpy as np

print('EXERCISE 1')
array1 = np.empty([4, 2], dtype=np.uint16)
print('Printing array: ')
print(array1)
print('\nPrinting numpy array attributes')
print('1. Array Shape is ', array1.shape)
print('2. Array dimensions are ', array1.ndim)
print('3. Length of each element of array in bytes is ', array1.itemsize)

print('\n\nEXERCISE 2')
print('Creating 5X2 array using np. arange method ')
sample_array = np. arange(100, 200, 10) # start , stop , step
sample_array = sample_array . reshape(5, 2)
print( sample_array)

print('\n\nEXERCISE 3')
sample_array = np. array([[11, 22, 33], [44, 55, 66], [77, 88, 99]])
print('Printing Input Array ')
print(sample_array)
print('\nPrinting array of items in the third column from all rows ')
new_array = sample_array[:, 2]
print(new_array)


print('\n\nEXERCISE 4')
sample_array = np.array([[3 , 6, 9, 12] , [15 , 18, 21, 24], [27, 30, 33, 36] , [39 , 42, 45, 48] , [51 , 54, 57, 60]])
print('Printing Input Array ')
print( sample_array )
print('\nPrinting array of odd rows and even columns ')
new_array = sample_array[::2, 1::2]
print(new_array)



print('\n\nEXERCISE 5')
array1 = np. array([[5, 6, 9], [21, 18, 27]])
array2 = np. array([[15, 33, 24], [4, 7, 1]])
result_array = array1 + array2
print('The sum of the two arrays is ')
print(result_array)
for num in np.nditer(result_array, op_flags=['readwrite']):
    num[...] = np.sqrt(num)
print('\nThe result array after calculating the square root of all elements')
print(result_array)


print('\n\nEXERCISE 6')
sample_array = np.array([[34, 43, 73], [82, 22, 12], [53, 94, 66]])
print('Original array ')
print(sample_array)
print('Shape is: ', sample_array.shape)
print('\nOne - line array ')
line_array = sample_array.reshape(1, np.prod(sample_array.shape))
print(line_array)
print('\nOne - line sorted array ')
sort_line_array = np.sort(line_array)
print(sort_line_array)
print('\nResult array ')
sort_array = sort_line_array.reshape(sample_array.shape)
print(sort_array)


print('\n\nEXERCISE 7')
sample_array = np.array([[34, 43, 73], [82, 22, 12], [53, 94, 66]])
print('Printing Original array ')
print(sample_array)
min_of_axis1 = np.amin(sample_array, 1)
print('\nPrinting amin Of Axis 1')
print(min_of_axis1)
max_of_axis0 = np.amax(sample_array, 0)
print('Printing amax Of Axis 0')
print(max_of_axis0)
new_arraaaay = np.array([min_of_axis1, max_of_axis0])
print('New array is: \n', new_arraaaay)


print('\n\nEXERCISE 8')
sample_array = np.array([[34, 43, 73], [82, 22, 12], [53, 94, 66]])
print('Printing Original array ')
print(sample_array)
print('Array after deleting column 2 on axis 1')
sample_array = np.delete(sample_array, 1, axis=1)
print(sample_array)
arr = np.array([[10, 10, 10]])
print('Array after inserting column 2 on axis 1')
sample_array = np.insert(sample_array, 1, arr, axis=1)
print(sample_array)
