# -*- coding: utf-8 -*-
""" Code for Harmonic approximation. """

import numpy as np
from ase import thermochemistry
from PyMuTT import constants as c
from PyMuTT.models.statmech.heat_capacity import get_CvoR_trans, \
    get_CvoR_vib, get_CvoR_rot

class HarmonicThermo:
    """Calculate enthalpy and entropy using harmonic approach.

    Depends on ASE thermochemistry module.
    """
    def __init__(self, vib_energies, pot_energy=0.0):
        """

        Args:
            vib_energies (numpy array): Vibrational energies in eV
            pot_energy (float): Potential energy in eV
        """
        self.model = thermochemistry.HarmonicThermo(vib_energies=vib_energies,
                                                    potentialenergy=pot_energy)

    def get_CpoR(self, Ts):
        """Calculate dimensionless heat capacity (Cp/R).

        Args:
            Ts (float or numpy array): Temperature(s) in K
        Returns:
            CpoR (float or numpy array): Dimensionless heat capacity (Cp/R)
        """
        return get_CvoR_vib(vib_energies=self.model.vib_energies, Ts=Ts)

    def get_HoRT(self, Ts, verbose = False):
        """Calculate dimensionless enthalpy at a given temperature.

        Args:
            Ts (float or numpy array): Temperature(s) in K
            verbose (bool): Flag to print each contribution to enthalpy
        Returns:
            HoRT (float or numpy array): Dimensionless heat capacity (H/RT)
        """
        try:
            iter(Ts)
        except TypeError:
            HoRT = self.model.get_internal_energy(Ts, verbose) / \
                   (c.kb('eV/K') * Ts)
        else:
            HoRT = np.zeros_like(Ts)
            for i, T in enumerate(Ts):
                HoRT[i] = self.model.get_internal_energy(T, verbose) / \
                          (c.kb('eV/K') * T)
        return HoRT

    def get_SoR(self, Ts, verbose=False):
        """Calculate dimensionless entropy at a given temperature.

        Args:
            Ts (float or numpy array): Temperature(s) in K
            verbose (bool): Flag to print each contribution to entropy
        Returns:
            SoR (float or numpy array): Dimensionless entropy (S/R)
        """
        try:
            iter(Ts)
        except TypeError:
            SoR = self.model.get_entropy(Ts, verbose)/c.R('eV/K')
        else:
            SoR = np.zeros_like(Ts)
            for i, T in enumerate(Ts):
                SoR[i] = self.model.get_entropy(T, verbose)/c.R('eV/K')
        return SoR

    def get_GoRT(self, Ts, verbose=False):
        """Calculate dimensionless Gibbs energy at a given temperature.

        Args:
            Ts (float or numpy array): Temperature(s) in K
            verbose (bool): Flag to print each contribution to enthalpy
        Returns:
            GoRT (float or numpy array): Dimensionless heat capacity (H/RT)
        """
        try:
            iter(Ts)
        except TypeError:
            GoRT = self.model.get_helmholtz_energy(Ts, verbose) / \
                   (c.kb('eV/K') * Ts)
        else:
            GoRT = np.zeros_like(Ts)
            for i, T in enumerate(Ts):
                GoRT[i] = self.model.get_helmholtz_energy(T, verbose) / \
                          (c.kb('eV/K') * T)
        return GoRT