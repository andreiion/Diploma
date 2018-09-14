import numpy as np

producer_count = 0
consumer_count = 0
BUF_SIZE = 16384  # condition is that BUF_SIZE has to be even disible with sys.maxsize + 1

data = 1

def process_data(data):
    return

shared_buffer = np.empty(BUF_SIZE, dtype=object)


# Producer side
while producer_count - consumer_count == BUF_SIZE:
    continue

shared_buffer[producer_count % BUF_SIZE] = data
producer_count = producer_count + 1


# Consumer side
while producer_count - consumer_count == 0:
    continue

process_data(shared_buffer[consumer_count % BUF_SIZE])
consumer_count = consumer_count + 1
