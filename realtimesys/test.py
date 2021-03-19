from concurrent.futures.thread import ThreadPoolExecutor

import numpy as np

a = np.array([[1,2,3],[4,6,5]])
i_a = np.argsort(a)[:, ::-1]
# print(i_a)
print(a[:, i_a])