import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

from primakoff import primakoff
from laue import LaueCrystal

lc = LaueCrystal() #9.81220
energy = 20
coupling = 1e-3


ma_array = np.linspace(0.0, 0.05, 5000)
theta_array = np.linspace(0.000001, pi/2, 5000)

prim_array = np.ones(ma_array.shape[0]) * primakoff(coupling, 6)

p_ma_array = np.zeros(ma_array.shape[0])
p_ma_array2 = np.zeros(ma_array.shape[0])
p_ma_array3 = np.zeros(ma_array.shape[0])
p_th_array = np.zeros(theta_array.shape[0])

for i in range(0, p_ma_array.shape[0]):
    p_ma_array[i] = lc.conversion_probability(coupling, 2*lc.bragg, 20, ma_array[i])
    p_ma_array2[i] = lc.conversion_probability(coupling, 0.5*lc.bragg, 20, ma_array[i])
    p_ma_array3[i] = lc.conversion_probability(coupling, lc.bragg, 20, ma_array[i])


plt.plot(1000*ma_array, p_ma_array, label="E = 17")
plt.plot(1000*ma_array, p_ma_array2, label="E = 20")
plt.plot(1000*ma_array, p_ma_array3, label="E = 25")
plt.plot(1000*ma_array, prim_array, label="Primakoff")
#plt.xscale("log")
plt.xlabel(r"$m_a$ [eV]")
plt.ylabel(r"$P_{\gamma \to a}$")
plt.yscale("log")
plt.legend()
plt.show()

plt.clf()
for i in range(0, theta_array.shape[0]):
    p_th_array[i] = lc.conversion_probability(1e-3, theta_array[i], 20, 0.001)


plt.plot(theta_array, p_th_array)
plt.yscale("log")
plt.show()

