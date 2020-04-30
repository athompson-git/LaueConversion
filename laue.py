import numpy as np
from numpy import pi, sqrt, exp, sin, cos, arccos, absolute

# Constants
alpha = 1/137.035999084  # fine structure constant
keV_cm = 1.98e-8  # 1 cm^-1 = 1.98e-11 keV
me = 510.99895000  # keV
r_e = alpha / me # classical electron radius in keV^-1
bohr_radius = 1/(me * alpha)



class LaueCrystal:
    # Default is C(220)
    def __init__(self, thickness=1, proton_number=6, plasma_freq=38,
                 bragg_angle=0.2478368, reciprocal_lattice=9.812297, lattice=(2,2,0)):
        self.qt = reciprocal_lattice  # reciprocal lattice spacing, keV
        self.z = proton_number
        self.h = thickness / keV_cm  # total crystal thickness given in cm converted to keV^-1
        self.bragg = bragg_angle  # radians
        self.lattice = lattice  # tuplet of Miller indices, e.g. (2,2,0) for the Diamond lattice.
        self.m_gamma = plasma_freq/1000  # convert plasma_freq in eV to keV
        self.fc = self.get_fc(lattice)
        self.V = self.get_volume_from_plasma_freq()
        self.M = self.get_fc(lattice) / self.get_volume_from_plasma_freq()
    
    def get_volume_from_plasma_freq(self):
        return (4*pi*r_e*self.fc*self.z) / self.m_gamma**2  # ad hoc way of getting the cell volume
    
    def get_fc(self, lattice):
        m4 = lattice[0] + lattice[1] + lattice[2]
        if (lattice[0] % 2 == 0 and lattice[1] % 2 == 0 and lattice[2] % 2 == 0) \
            or (lattice[0] % 2 == 1 and lattice[1] % 2 == 1 and lattice[2] % 2 == 1):
                if m4 % 4 == 0:
                    return 8
                elif m4 % 4 == 3 or m4 % 4 == 1:
                    return 4*sqrt(2)
                else:
                    return 0
        else:
            return 0

    def q(self, theta_gamma, theta_a, k_gamma, k_a):
        return k_gamma * sin(theta_gamma) + k_a * sin(theta_a)
    
    def gamma_ff(self, q):
        parameters = get_ff_params(self.z)
        a1 = parameters[0]
        b1 = parameters[1]
        a2 = parameters[2]
        b2 = parameters[3]
        a3 = parameters[4]
        b3 = parameters[5]
        a4 = parameters[6]
        b4 = parameters[7]
        c = parameters[8]
        return a1*exp(-b1*(q/(4*pi))**2) + a2*exp(-b2*(q/(4*pi))**2) \
                + a3*exp(-b3*(q/(4*pi))**2) + a4*exp(-b4*(q/(4*pi))**2) + c

    def axion_ff(self, q, e, ma):
        ff0 = sqrt(4*pi*alpha) * self.z * (e**2 - ma**2) * (2.275*bohr_radius/6)
        return min(ff0, ((e**2 - ma**2)*sqrt(4*pi*alpha)*(self.z - self.gamma_ff(q))) / q**2)

    def eta(self, t, e, ma):
        q = self.qt
        return r_e * self.M * self.gamma_ff(q) / (e * cos(self.bragg))
    
    def eta0(self, e):
        return r_e * self.M * self.z / (e * cos(self.bragg))

    def dphi_gamma(self, t, e, ma):
        return -0.5 * (2*self.qt*(e * sin(t) - 0.5 * self.qt)) / (2 * e * cos(t))

    def dphi_a(self, t, e, ma):
        return 0.5 * (ma**2 - 2*self.qt*(e * sin(t) - 0.5 * self.qt)) / (2 * e * cos(t))

    def u(self, t, e, ma):
        return sqrt(self.eta(t, e, ma)**2 + self.dphi_gamma(t, e, ma)**2)

    def phi_as(self, t, e, ma):
        return sqrt(e**2 - ma**2 - (self.qt - e**sin(t))**2)

    def theta_as(self, t, e, ma):
        return arccos((self.phi_as(t, e, ma))/(sqrt(e**2 - ma**2)))
    
    def cos_as(self, t, e, ma):
        return sqrt(e**2 - ma**2 - (self.qt - e**sin(t))**2) / sqrt(e**2 - ma**2)
    
    def sin_as(self, t, e, ma):
        return (self.qt - e**sin(t)) / sqrt(e**2 - ma**2)
    
    def zeta(self, g, t, e, ma):
        q = self.qt
        sin_sum = sin(t)*self.cos_as(t, e, ma) + cos(t)*self.sin_as(t, e, ma)
        return g * self.M * self.axion_ff(q, e, ma) * sin_sum / ((4*pi*e*self.phi_as(t, e, ma)))
    
    def conversion_probability(self, g, t, e, ma):
        if e < ma:
            print("E < ma!...")
            return 1
        A = -0.5 * self.zeta(g, t, e, ma) * exp(-1j * self.h * self.phi_as(t, e, ma))
        B = self.dphi_gamma(t, e, ma) / self.u(t, e, ma)
        C1 = 1 - exp(1j * (self.eta0(e) - self.u(t, e, ma) + self.dphi_gamma(t, e, ma) - 2*self.dphi_a(t, e, ma))*self.h)
        C2 = 1 - exp(1j * (self.eta0(e) + self.u(t, e, ma) + self.dphi_gamma(t, e, ma) - 2*self.dphi_a(t, e, ma))*self.h)
        D1 = self.eta0(e) - self.u(t, e, ma) + self.dphi_gamma(t, e, ma) - 2*self.dphi_a(t, e, ma)
        D2 = self.eta0(e) + self.u(t, e, ma) + self.dphi_gamma(t, e, ma) - 2*self.dphi_a(t, e, ma)
        return np.power(absolute(A * ((1 + B) * (C1 / D1) + (1 - B) * (C2 / D2))), 2)



# X-ray form factor parameters from fits taken from
# http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
def get_ff_params(z):
    if (z not in (1, 32, 6, 55)):
        print("Element not found. You must enter it in by hand.")
        return 1
    if z == 1:
        return (0.489918,20.6593,0.262003,7.74039,0.196767,49.5519,0.049879,2.20159,0.001305)
    if z == 32:
        return (16.0816,2.8509,6.3747,0.2516,3.7068,11.4468,3.683,54.7625,2.1313)
    if z == 6:
        return (2.31,20.8439,1.02,10.2075,1.5886,0.5687,0.865,51.6512,0.2156)
    if z == 55:
        return (20.3892,3.569,19.1062,0.3107,10.662,24.3879,1.4953,213.904,3.3352)