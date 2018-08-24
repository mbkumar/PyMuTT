# -*- coding: utf-8 -*-
"""Operations for ideal gas models"""

import numpy as np
from ase import thermochemistry
from PyMuTT import constants as c
from PyMuTT.models.statmech.heat_capacity import get_CvoR_trans, get_CvoR_vib, get_CvoR_rot


class IdealGasThermo:
    """Ideal Gas Model to calculate enthalpy and entropy

     Parameters:
        vib_energies: ((N,) `numpy.ndarray`_) Vibrational energies in eV
        geometry: (str) Geometry of molecule.
                  Supported options: monatomic, linear, nonlinear
        potential_energy: (float) Potential energy in eV
        atoms: (`ase.Atoms`_) Atoms object required for calculating rotational modes
        symmetry_number:  (int) Symmetry number
                        Point group    symmetry number
                        ===========    ===============
                        C1             1
                        Cs             1
                        C2             2
                        C2v            2
                        C3v            3
                        Cinfv          1
                        D2h            4
                        D3h            6
                        D5h            10
                        Dinfh          2
                        D3d            6
                        Td             12
                        Oh             24
                        See DOI for more details: 10.1007/s00214-007-0328-0
        spin: (float) Holds the total electronic spin
               - 0 for molecules in which all electrons are paired
               - 0.5 for a free radical with a single unpaired electron
               - 1.0 for a triplet with two unpaired electrons, such as O_2
    """

    def __init__(self, vib_energies, geometry, potential_energy=0.0,
                 atoms=None, symmetry_number=None, spin=None, natoms=None):
        self.model = thermochemistry.IdealGasThermo(
            vib_energies, geometry, potentialenergy=potential_energy,
            atoms=atoms, symmetrynumber=symmetry_number, spin=spin,
            natoms=natoms)

    def get_CpoR(self, Ts):
        """Calculate dimensionless heat capacity (Cp/R).

        If you would like to use different behavior from the default, the
        thermo_model used must have the function 'get_CpoR'.

        Parameters:
            Ts : array-like or scalar
                Temperature(s) in K
        Returns:
            CpoR : float or (N,) `numpy.ndarray_`
                Dimensionless heat capacity (Cp/R)
        """
        CvoR_trans = get_CvoR_trans(degrees=3)
        CvoR_vib = get_CvoR_vib(vib_energies=self.model.vib_energies, Ts=Ts)
        CvoR_rot = get_CvoR_rot(geometry=self.model.geometry)
        CvoR_to_CpoR = 1.
        CpoR = CvoR_trans + CvoR_vib + CvoR_rot + CvoR_to_CpoR
        return CpoR

    def get_HoRT(self, Ts, verbose=False):
        """Calculate the dimensionless enthalpy.

        Parameters:
            Ts : float or (N,) `numpy.ndarray`_
                Temperature(s) in K
            verbose : bool
                Whether a table breaking down each contribution should be printed
        Returns
            HoRT : float or (N,) `numpy.ndarray`_
                Dimensionless heat capacity (H/RT)
        """
        try:
            iter(Ts)
        except TypeError:
            HoRT = self.model.get_enthalpy(Ts, verbose=verbose) / \
                   (c.kb('eV/K') * Ts)
        else:
            HoRT = np.zeros_like(Ts)
            for i, T in enumerate(Ts):
                HoRT[i] = self.model.get_enthalpy(T, verbose=verbose) / \
                          (c.kb('eV/K') * T)
        return HoRT

    def get_SoR(self, Ts, P=1., verbose=False):
        """Calculate the dimensionless entropy.

        Parameters:
            Ts : float or (N,) `numpy.ndarray`_
                Temperature(s) in K
            P : float, optional
                Pressure in atm. Default is 1 atm.
            verbose : bool
                Whether a table breaking down each contribution should be printed
        Returns:
            SoR : float or (N,) `numpy.ndarray`_
                Dimensionless entropy (S/R) at the specified temperature and pressure
        """
        try:
            iter(Ts)
        except TypeError:
            SoR = self.model.get_entropy(
                Ts, pressure=P*c.convert_unit(from_='atm', to='Pa'),
                verbose=verbose)/c.R('eV/K')
        else:
            SoR = np.zeros_like(Ts)
            for i, T in enumerate(Ts):
                SoR[i] = self.model.get_entropy(
                    T, pressure=P*c.convert_unit(from_='atm', to='Pa'),
                    verbose=verbose)/c.R('eV/K')
        return SoR

    def get_GoRT(self, Ts, P=1., verbose=False):
        """Returns the dimensionless Gibbs energy at a given temperature

        Parameters:
            Ts : float or (N,) `numpy.ndarray`_
                Temperature(s) in K
            P : float, optional
                Pressure in bar. Default is 1 atm.
            verbose : bool
                Whether a table breaking down each contribution should be printed
        Returns:
            GoRT : float or (N,) `numpy.ndarray`_
                Dimensionless heat capacity (G/RT) at the specified temperature
        """
        try:
            iter(Ts)
        except TypeError:
            GoRT = self.model.get_gibbs_energy(
                Ts, pressure=P*c.convert_unit(from_='bar', to='Pa'), verbose=verbose) / \
                   (c.kb('eV/K') * Ts)
        else:
            GoRT = np.zeros_like(Ts)
            for i, T in enumerate(Ts):
                GoRT[i] = self.model.get_gibbs_energy(
                    T, pressure=P*c.convert_unit(from_='bar', to='Pa'), verbose=verbose) / \
                          (c.kb('eV/K') * T)
        return GoRT
