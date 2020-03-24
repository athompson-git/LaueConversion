import numpy as np
import matplotlib.pyplot as plt

from numpy import pi

from laue import LaueCrystal


class Photon:
    def __init__(self, energy, box_width, x0=0.0, y0=0.0, z0=0.0):
        self.energy = energy
        self.x = x0
        self.y = y0
        self.z = z0
        self.phi, self.theta = self.generate_decay()
        self.boxw = box_width
        
    def generate_decay(self):
        phi = np.random.uniform(0, 2*pi)
        theta = np.random.uniform(0, np.sqrt(2)/2)
        return phi, theta


def main():
    angles = []
    weights = []
    lc = LaueCrystal()
    for i in range(0,100000):
        gamma = Photon(energy=10, box_width=10)
        print(gamma.phi, gamma.theta)
        angles.append(gamma.theta)
        prob = lc.conversion_probability(1, gamma.theta, 20, 1.93)
        weights.append(np.random.binomial(1, prob))
    
    plt.hist(angles, weights=weights, bins=50)
    plt.show()
        



if __name__ == "__main__":
    main()