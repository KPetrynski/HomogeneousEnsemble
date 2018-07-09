import numpy as np

a = [11, 12, 13, 14, 15, 16, 17]
b = [21, 22, 23, 24, 25, 26, 27]
c = [31, 32, 33, 34, 35, 36, 37]

print("normal a: ", a)

a = np.asarray(a)
b = np.asarray(b)
c = np.asarray(c)

print("np a: ", a)
chunk_size = 500
range = np.arange(chunk_size, (len(a)+1)*chunk_size, chunk_size)
# print("range: ",range)
# print("range: ", range(len(a*chunk_size), chunk_size))

data = np.stack((range, a, b, c), axis=-1)
print(data)
