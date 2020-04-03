import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

from laue import LaueCrystal

lc = LaueCrystal()


q_list = np.linspace(0, 10, 1000)
fg_list = np.empty_like(q_list)
fa_list = np.empty_like(q_list)
for i in range(0, q_list.shape[0]):
    fg_list[i] = lc.gamma_ff(q_list[i])
    fa_list[i] = lc.axion_ff(q_list[i], 20, 1)

plt.plot(q_list, fg_list, label="fg")
plt.plot(q_list, fa_list, label="fa")
plt.legend()
plt.yscale('log')
plt.show()