from joblib import Parallel, delayed
import multiprocessing
import numpy as np
import time

N = 3
M = 5 # After 10**5 parallelism becomes more efficient
sums1 = np.zeros(N)
sums2 = np.zeros(N)
partial_sum = 0

# Function to parallelize
def find_square(i):
    return (abs(i ** 2) + abs(i ** 3)) / 2


def process_shit(n):
    partial_sump = 0
    for m in range(1, M):
        partial_sump += find_square(m)
    return partial_sump + n


# Serial
start_time = time.time()
for n in range(1, N):
    for m in range(1, M):
        partial_sum += find_square(m)
    sums1[n - 1] = partial_sum + n
    partial_sum = 0
print("--- Serial: %s seconds ---" % round(time.time() - start_time, 3))


# Parallel
start_time2 = time.time()
num_cores = multiprocessing.cpu_count()

sums2 = Parallel(n_jobs=num_cores)(delayed(process_shit)(n)for n in range(1, N))
print("--- Parallel: %s seconds ---" % round(time.time() - start_time2, 3))

print(sums1)
print(sums2)