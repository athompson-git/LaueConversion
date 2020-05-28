import matplotlib.pyplot as plt
import numpy as np
from numpy import pi

from primakoff import primakoff
from laue import LaueCrystal

from matplotlib.pylab import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

lc = LaueCrystal()
energy = 20
coupling = 1e-3  # GeV^-1
coupling *= 1e-6 # convert to keV^-1


ma_array = np.linspace(0.025, 0.05, 10000)
theta_array = np.linspace(0.000001, pi/2, 5000)
prim_array = np.ones(ma_array.shape[0]) * primakoff(coupling, 6)
p_ma_array = np.zeros(ma_array.shape[0])


for i in range(0, p_ma_array.shape[0]):
    p_ma_array[i] = lc.conversion_probability(coupling, lc.bragg, 20, ma_array[i])


plt.plot(1000*ma_array, p_ma_array, label="E = 20")
plt.title(r"$E_\gamma = 20$ keV, $g_{a\gamma\gamma} = 10^{-3}$ GeV$^{-1}$, Bragg condition", loc="right")
plt.xlabel(r"$m_a$ [eV]")
plt.ylabel(r"$P_{\gamma \to a}$")
plt.yscale("log")
plt.xlim((25, 50))
plt.ylim((1e-12, 1e-4))
plt.tight_layout()
plt.show()

