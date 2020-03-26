import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

from laue import LaueCrystal

lc = LaueCrystal()


q_list = np.linspace(0, 10)

plt.plot(q_list, lc.gamam_ff(q_list))
plt.plot(q_list, lc.atomic_ff(q_list, 20, 1))
plt.show()