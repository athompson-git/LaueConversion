import numpy as np
from numpy import pi, sqrt, exp, sin, cos, arccos, absolute

# Constants
r_e = 0.14  # classical electron radius in keV
alpha = 1/137  # fine structure constant



class LaueCrystal:
    def __init__(self, thickness=5e10, proton_number=32, planar_dist=2.83,
                 reciprocal_lattice=6.2, bragg_angle=0.16, lattice=(2,2,0)):
        self.r = planar_dist  # keV^-1
        self.qt = reciprocal_lattice  # keV
        self.z = proton_number
        self.h = thickness  # total crystal thickness in keV^-1
        self.bragg = bragg_angle  # radians
        self.lattice = lattice  # tuplet of basis vectors, e.g. (2,2,0) for the Ge lattice.
        self.N = self.h / self.r
        self.V = self.r ** 3
        self.fc = self.get_fc(lattice)
        self.M = self.fc * self.r / self.V
    
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
    
    def get_ff_params(self):
        # TODO: use self.z to do a lookup table
        return (16, 6.37, 3.7, 3.68, 2.85, 0.2516, 11.45, 54.76)

    def q(self, t, e):
        return 4*pi*e*sin(2*t)
    
    def gamam_ff(self, q):
        parameters=self.get_ff_params()
        a1 = parameters[0]
        a2 = parameters[1]
        a3 = parameters[2]
        a4 = parameters[3]
        b1 = parameters[4]
        b2 = parameters[5]
        b3 = parameters[6]
        b4 = parameters[7]
        c = self.z - a1 - a2 - a3 - a4
        return a1*exp(-b1*(q/(4*pi)**2)) + a2*exp(-b2*(q/(4*pi)**2)) \
                + a3*exp(-b3*(q/(4*pi)**2)) + a4*exp(-b4*(q/(4*pi)**2)) + c

    def atomic_ff(self, q, ma):
        return (2*ma*alpha*(self.z - self.gamam_ff(q))) / q

    def eta(self, t, e):
        q = self.q(t, e)
        return r_e * self.M * self.gamam_ff(q) / (e * cos(self.bragg))

    def dphi_gamma(self, t, e, ma):
        return -(self.r / 2) * (2*self.qt*(e * sin(t) - 0.5 * self.qt)) / (2 * e * cos(t))

    def dphi_a(self, t, e, ma):
        return (self.r / 2) * (ma**2 - 2*self.qt*(e * sin(t) - 0.5 * self.qt)) / (2 * e * cos(t))

    def u(self, t, e, ma):
        return sqrt(self.eta(t, e)**2 + self.dphi_gamma(t, e, ma)**2)

    def phi_as(self, t, e, ma):
        return self.r * sqrt(e**2 - ma**2 - (self.qt - e**sin(t)**2))

    def theta_as(self, t, e, ma):
        return arccos((self.phi_as(t, e, ma))/(self.r * sqrt(e**2 - ma**2)))
    
    def zeta(self, g, t, e, ma):
        q = self.q(t, e)
        return g * self.M * self.atomic_ff(q, ma) * self.r * sqrt(e**2 - ma**2) \
               * sin(t + self.theta_as(t, e, ma)) / (4*pi*e*self.phi_as(t, e, ma))
    
    def conversion_probability(self, g, t, e, ma):
        if e < ma:
            print("E < ma! You idiot...")
            return 1
        A = -0.5 * self.zeta(g, t, e, ma) * exp(-1j * self.N * self.phi_as(t, e, ma))
        B = self.dphi_gamma(t, e, ma) / self.u(t, e, ma)
        C1 = 1 - exp(1j * (self.eta(0, e) - self.u(t, e, ma) + self.dphi_gamma(t, e, ma) - 2*self.dphi_a(t, e, ma))*self.N)
        C2 = 1 - exp(1j * (self.eta(0, e) + self.u(t, e, ma) + self.dphi_gamma(t, e, ma) - 2*self.dphi_a(t, e, ma))*self.N)
        D1 = self.eta(0, e) - self.u(t, e, ma) + self.dphi_gamma(t, e, ma) - 2*self.dphi_a(t, e, ma)
        D2 = self.eta(0, e) + self.u(t, e, ma) + self.dphi_gamma(t, e, ma) - 2*self.dphi_a(t, e, ma)
        return sqrt(absolute(A * ((1 + B) * (C1 / D1) + (1 - B) * (C2 / D2))))
