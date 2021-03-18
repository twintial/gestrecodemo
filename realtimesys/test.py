import numpy as np
a = np.arange(10)
b = np.arange(10)
c = np.arange(10,20)
b = np.vstack((b,c,c))
print(b)
print(np.outer(a,b))