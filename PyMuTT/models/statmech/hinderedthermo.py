# -*- coding: utf-8 -*-
"""
Hindered translator/hindered rotor model
"""

import numpy as np
from ase import thermochemistry
from PyMuTT import constants as c
from PyMuTT.models.statmech.heat_capacity import get_CvoR_trans, \
    get_CvoR_vib, get_CvoR_rot

class HinderedThermo:
    """Hindered translator/hindered rotor model.

    Used for surface species where there are 2 translational modes,
    1 rotational mode perpendicular to the surface, and 3N-3 vibrational modes.
    See further documentation from `ase.thermochemistry.HinderedThermo`_

    Parameters:
        vib_energies: ((N,) `numpy.ndarray`_) Vibrational energies in eV                                                               |
        trans_barrier_energy: (float) Translational energy barrier in eV                                                                       |
        rot_barrier_energy: (float) Rotational energy barrier in eV                                                                          |
        sitedensity: (float) Site density in cm^-2                                                                                    |
        rotationalminima: (int) Number of equivalent minima for an adsorbate
            full rotation e.g. 6 for an adsorbate on an fcc(111) top site |
        potentialenergy:  (float) Potential energy in eV                                                                                   |
        atoms: (`ase.Atoms`_) Atoms object required for calculating rotational modes                                            |
        symmetrynumber: (int) Symmetry number                                                                                            |
            For more details, see DOI:10.1007/s00214-007-0328-0                                                              |
            Some symmetry numbers are given below
            Point group    symmetry number                                                                                   |
            ===========    ===============                                                                                   |
            C1             1                                                                                                 |
            Cs             1                                                                                                 |
            C2             2                                                                                                 |
            C2v            2                                                                                                 |
            C3v            3                                                                                                 |
            Cinfv          1                                                                                                 |
            D2h            4                                                                                                 |
            D3h            6                                                                                                 |
            D5h            10                                                                                                |
            Dinfh          2                                                                                                 |
            D3d            6                                                                                                 |
            Td             12                                                                                                |
            Oh             24                                                                                                |

        mass: (float) Mass of adsorbate in amu. If unspecified,
            uses the atoms object
        inertia: (float) Reduced moment of inertia in amu*A^-2.
            If unspecified, uses atoms object

    """
    def __init__(self, vib_energies, trans_barrier_energy, rot_barrier_energy,
                 sitedensity, rotationalminima, potentialenergy=0.0, mass=None,
                 inertia=None, atoms=None, symmetrynumber=1):
        self.model = thermochemistry.HinderedThermo(
            vib_energies=vib_energies,
            trans_barrier_energy=trans_barrier_energy,
            rot_barrier_energy=rot_barrier_energy,
            sitedensity=sitedensity,
            rotationalminima=rotationalminima,
            potentialenergy=potentialenergy,
            mass=mass,
            inertia=inertia,
            atoms=atoms,
            symmetrynumber=symmetrynumber)

    def get_CpoR(self, Ts):
        """Calculate the dimensionless heat capacity (Cp/R).

        Parameters:
            Ts : float or (N,) `numpy.ndarray`_
                Temperature(s) in K
        Returns:
            CpoR : float or (N,) `numpy.ndarray`_
                Dimensionless heat capacity (Cp/R)
        """
        raise NotImplementedError

    def get_HoRT(self, Ts, verbose=False):
        """Calculate the dimensionless enthalpy.

        Parameters:
            Ts : float or (N,) `numpy.ndarray`_
                Temperature(s) in K
            verbose : bool
                Whether a table breaking down each contribution should be printed
        Returns:
            HoRT : float or (N,) `numpy.ndarray`_
                Dimensionless heat capacity (H/RT)
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


    def get_SoR(self, Ts, P=1., verbose=False):
        """Calculate the dimensionless entropy.

        Parameters:
            Ts : float or (N,) `numpy.ndarray`_
                Temperature(s) in K
            P : float
                Pressure in atm. Default is 1 atm
            verbose : bool
                Whether a table breaking down each contribution should be printed
        Returns:
            SoR : float or (N,) `numpy.ndarray`_
                Dimensionless entropy (S/R)
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

    def get_GoRT(self, Ts, P=1., verbose=False):
        """Calculate the dimensionless Gibbs energy.

        Parameters:
            Ts : float or list
                Temperature(s) in K
            P : float
                Pressure in atm. Default is 1 atm
            verbose : bool
                Whether a table breaking down each contribution should be printed
        Returns:
            GoRT: float or (N,) `numpy.ndarray`_
                Dimensionless heat capacity (G/RT) at the specified temperature
        """
        press = P * c.convert_unit(from_='atm', to='Pa')
        try:
            iter(Ts)
        except TypeError:
            GoRT = self.model.get_helmholtz_energy(Ts, verbose=verbose) / \
                   (c.kb('eV/K') * Ts)
        else:
            GoRT = np.zeros_like(Ts)
            for i, T in enumerate(Ts):
                GoRT[i] = self.model.get_helmholtz_energy(T, verbose=verbose) / \
                          (c.kb('eV/K') * T)
        return GoRT