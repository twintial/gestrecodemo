from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np
a = np.arange(10)
b = np.arange(10)
c = np.arange(10,20)
b = np.vstack((b,c,c))
# print(b)
# print(np.outer(a,b))


def t(x1, x2):
    print(x1 + x2)
    return x1 + x2

executor = ThreadPoolExecutor(max_workers=21)
s = 0
for R_ij in executor.map(t, [1,2,3,4,5],[2,3,4,5,6]):
    s += R_ij
print(s)