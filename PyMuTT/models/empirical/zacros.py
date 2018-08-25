# -*- coding: utf-8 -*-
"""
PyMuTT.models.empirical.zacros

Operations related to the Zacros wrapper
"""

import numpy as np
from PyMuTT import constants as c
from PyMuTT import get_molecular_weight as mw
from PyMuTT.models.empirical_model import BaseThermo


class Zacros(BaseThermo):
    """Stores the information for an individual nasa specie
    Inherits from PyMuTT.models.empirical.BaseThermo
    """
    def __init__(self, A_st=None, atoms=None, symmetrynumber=None,
                 inertia=None, geometry=None, vib_energies=None,
                 potentialenergy=None, **kwargs):
        super().__init__(atoms=atoms, symmetrynumber=symmetrynumber,
                         geometry=geometry, vib_energies=vib_energies,
                         potentialenergy=potentialenergy, **kwargs)
        self.A_st = A_st
        self.atoms = atoms
        self.geometry = geometry
        self.symmetrynumber = symmetrynumber
        self.inertia = inertia
        self.etotal = potentialenergy
        self.vib_energies = vib_energies
        self.theta = np.array(vib_energies) / c.kb('eV/K')
        self.zpe = sum(np.array(vib_energies)/2.) *\
            c.convert_unit(from_='eV', to='kcal')*c.Na
        if np.sum(vib_energies) != 0:
            self.q_vib = np.product(np.divide(1, (1 - np.exp(-self.theta /
                                                             c.T0('K')))))
        if self.phase == 'G':
            if self.inertia is not None:
                self.I3 = self.inertia
            else:
                self.I3 = atoms.get_moments_of_inertia() *\
                        c.convert_unit(from_='A2', to='m2') *\
                        c.convert_unit(from_='amu', to='kg')
            self.T_I = c.h('J s')**2/(8*np.pi**2*c.kb('J/K'))
        if self.phase == 'G':
            Irot = np.max(self.I3)
            if self.geometry == 'nonlinear':
                self.q_rot = np.sqrt(np.pi*Irot)/self.symmetrynumber *\
                                    (c.T0('K')/self.T_I)**(3./2.)
            else:
                self.q_rot = (c.T0('K')*Irot/self.symmetrynumber)/self.T_I
        else:
            self.q_rot = 0.
        if self.A_st is not None:
            self.MW = mw(self.elements)*c.convert_unit(from_='g', to='kg')/c.Na
            self.q_trans2D = self.A_st * (2*np.pi*self.MW*c.kb('J/K') *
                                          c.T0('K'))/c.h('J s')**2
