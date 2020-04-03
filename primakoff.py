# Primakoff scattering cross-sections
import numpy as np

alpha = 1/137

def primakoff(g, z):
    return (9 / 4) * alpha * (z * g)**2
