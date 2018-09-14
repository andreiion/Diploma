import numpy as np
import random
import time

N = 52
K = 2049
matrix = np.zeros([K, N])
total = np.zeros(N)

for k in range(0, K):
    for n in range(0, N):
        matrix[k][n] = k ** n + 3.1416 + random.random()


def half_wave_rectifier(x):
    return (x + abs(x)) / 2


start_time = time.time()
partial_sum = 0
for n in range(1, len(matrix[0])):
    for k in range(0, len(matrix), 5):
        partial_sum +=  half_wave_rectifier(abs(matrix[k][n]) - abs(matrix[k][n - 1]))
        #partial_sum += half_wave_rectifier(matrix[k][n] - matrix[k][n - 1])
        #half_wave_rectifier(abs(matrix[k][n]) - abs(matrix[k][n - 1]))
    total[n - 1] = partial_sum
    partial_sum = 0
print("--- Python: %s seconds ---" % round(time.time() - start_time, 3))

print(total)

def half_wave(array, n):
    #return (abs(array[n]) - abs(array[n - 1]) + abs(abs(array[n]) - abs(array[n - 1]))) / 2
    return (array[n] - array[n - 1] + abs(array[n] - array[n - 1])) / 2

#Test
#c = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]])

output = np.zeros(N)
start_time = time.time()
#count = 0
for n in range(1, N):
    output[n - 1] = np.sum(np.apply_along_axis(half_wave, 1, matrix, n))
    #np.apply_along_axis(half_wave, 1, matrix, n)

print("--- Numpy: %s seconds ---" % round(time.time() - start_time, 3))
print(output)
