import numpy as np

a = [(725,), (84,)]
b = np.arange(84)
c = b[-8*12:]
print(c.reshape([12,8]))