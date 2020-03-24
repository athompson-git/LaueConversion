import matplotlib.pyplot as plt
import numpy as np

from laue import LaueCrystal

lc = LaueCrystal()


ma_array = np.linspace(0, 10, 1000)
p_array = np.zeros(ma_array.shape[0])

for i in range(0, ma_array.shape[0]):
    p_array[i] = lc.conversion_probability(1, lc.bragg, 20, ma_array[i])


plt.plot(ma_array, p_array)
plt.show()