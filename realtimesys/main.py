import numpy as np
import pyaudio

# b = b'\x01\x02\x03\x04'
# print(len(b))
# print(b[1:3])
# n = np.frombuffer(b, dtype=np.int16)
# # n = int.from_bytes(b,byteorder='big',signed=False)
# f = np.array([2])
# f2 = np.array([1,2,3,4])
# f = np.concatenate((f,f2))
# print(f)
#
# print(f2.reshape(2, -1))

a = np.array([[1,2],[3,4]])
b = np.array([[1,2,3,4,5,6,7,8,9,10],[1,2,3,4,5,6,7,8,9,10]])
c = b[:,-2*3:]
print(c[:,2:2*2])
