import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, sqrt

from laue import LaueCrystal

lc = LaueCrystal()

kg = 20
ma = 2
ka = sqrt(kg**2 - ma**2)

angles = np.linspace(0, pi, 1000)
q_list = np.empty_like(angles)
qt_list = np.ones_like(angles) * lc.qt

gamma_p = kg * np.sin(angles)
axion_p = ka * np.sin(lc.theta_as(angles, kg, ma))

for i in range(0, angles.shape[0]):
    q_list[i] = lc.q(angles[i], lc.theta_as(angles[i], kg, ma), kg, ka)


plt.plot(angles, q_list, label="q")
plt.plot(angles, qt_list, label="qt")
plt.plot(angles, axion_p, label="q_a")
plt.plot(angles, gamma_p, label="q_g")
plt.legend()
plt.show()