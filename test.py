import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

from laue import LaueCrystal

lc = LaueCrystal() #9.81220


ma_array = np.linspace(0.0, 0.03, 5000)
theta_array = np.linspace(0.000001, pi/2, 5000)
p_ma_array = np.zeros(ma_array.shape[0])
p_th_array = np.zeros(theta_array.shape[0])

for i in range(0, p_ma_array.shape[0]):
    p_ma_array[i] = lc.conversion_probability(1e-3, lc.bragg, 20, ma_array[i])


plt.plot(1000*ma_array, p_ma_array)
#plt.xscale("log")
plt.xlabel(r"$m_a$ [eV]")
plt.ylabel(r"$P_{\gamma \to a}$")
plt.yscale("log")
plt.show()

"""
plt.clf()
for i in range(0, theta_array.shape[0]):
    p_th_array[i] = lc.conversion_probability(1e-3, theta_array[i], 20, 0.001)


plt.plot(theta_array, p_th_array)
plt.xscale("log")
plt.yscale("log")
plt.show()
"""
