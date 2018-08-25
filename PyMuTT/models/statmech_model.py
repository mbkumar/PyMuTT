# -*- coding: utf-8 -*-
"""Obtain the thermodynamic properties for given species.

ASE thermochemistry models are used to evalaute the standard properties
such as enthalpy and entropy
"""

import numpy as np
from ase import thermochemistry
from PyMuTT import constants as c


def get_CvoR_trans(degrees=3):
    """Calculate the dimensionless translational heat capacity

    :math:`\\frac {Cv_{trans}} {R} = \\frac {N} {2}` where N is
    the degrees of freedom

    Parameters:
        degrees (int): Degrees of freedom the specie has.
            Gas-phase species will be 3 degrees whereas surface species
            may have less depending on the thermodynamic model

    Returns:
        Dimensionless translational heat capacity
    """
    if degrees > 3:
        raise ValueError(
            'Unphysical value. More than 3 degrees of freedom not allowed.')
    return 0.5 * degrees


def get_CvoR_rot(geometry):
    """Calculate the dimensionless rotational heat capacity

    :math:`\\frac {Cv_{rot}} {R} = 0` if monatomic
    :math:`\\frac {Cv_{rot}} {R} = 1` if linear
    :math:`\\frac {Cv_{rot}} {R} = 1.5` if nonlinear

    Parameters:
        geometry (str): Geometry of the specie. Accepted values are
            "monatomic", "linear", and "nonlinear".

    Returns:
        Dimensionless rotational heat capacity
    """
    if geometry == 'monatomic':
        return 0.
    elif geometry == 'linear':
        return 1.
    elif geometry == 'nonlinear':
        return 1.5
    else:
        raise ValueError('Geometry {} not supported.'.format(geometry))


def get_CvoR_vib(vib_energies, temperature):
    """Calculate the dimensionless vibrational heat capacity

    :math:`\\frac {Cp_{vib}}{R} = \\sum_{i=1}^{n} \\bigg(\\frac {\\Theta_{V,i}}{2T}\\bigg)^2 \\frac {1}{\\big(sinh(\\frac {\\Theta_{V,i}}{2T})\\big)^2}`

    Parameters:
        vib_energies (array of floats): Vibrational energies in eV
        temperature (float): Temperature in K

    Returns:
        Dimensionless vibrational heat capacity
    """
    dimensionless_vibs = vib_energies/c.kb('eV/K')/temperature
    CvoR_vib = np.sum((0.5 * dimensionless_vibs)**2 * (
            1./np.sinh(0.5 * dimensionless_vibs))**2)
    return CvoR_vib


class Thermo:
    """PyMuTT wrapper to thermochemistry models implemented in ASE.

    Seamlessly calculate enthalpy and entropy based on the supplied ASE model.
    Parameters:
        model (str): Choice of model. Options are 'harmonic', 'hindered', 'ideal gas'
        vib_energies (array): Vibrational energies in eV
        *args: Additional arguments needed for the ASE model
        **kwargs: Additional keyword arguments associated with the ASE model.
        Refer to ASE documentation to identify the required inputs.
    """
    def __init__(self, specie_type, vib_energies, *args, **kwargs):
        if specie_type == 'ideal gas':
            model_cls = thermochemistry.IdealGasThermo
        elif specie_type == 'harmonic':
            model_cls = thermochemistry.HarmonicThermo
        elif specie_type == 'hindered':
            model_cls = thermochemistry.HinderedThermo
        else:
            raise NotImplementedError("Given model not implemented")

        self.specie_type = specie_type
        self.model = model_cls(vib_energies, *args, **kwargs)

    def get_CpoR(self, temperature):
        """Calculate dimensionless heat capacity (Cp/R).

        Parameters:
            temperature (float): Temperature in K
        """
        if self.specie_type == 'ideal gas':
            CvoR_trans = get_CvoR_trans(degrees=3)
            CvoR_vib = get_CvoR_vib(self.model.vib_energies, temperature)
            CvoR_rot = get_CvoR_rot(self.model.geometry)
            CvoR_to_CpoR = 1.
            CpoR = CvoR_trans + CvoR_vib + CvoR_rot + CvoR_to_CpoR
        elif self.specie_type == 'harmonic':
            CpoR = get_CvoR_vib(self.model.vib_energies, temperature)
        else:
            CpoR = None
        return CpoR

    def get_HoRT(self, temperature, verbose=False):
        """Calculate the dimensionless enthalpy.

        Parameters:
            temperature (float): Temperature in K
            verbose (bool): Print a table breaking down each contribution
        """
        if self.specie_type == 'ideal gas':
            enthalpy = self.model.get_enthalpy(temperature, verbose=verbose)
        else:
            enthalpy = self.model.get_internal_energy(
                temperature, verbose=verbose)

        return enthalpy / (c.kb('eV/K') * temperature)

    def get_SoR(self, temperature, pressure=1.0, verbose=False):
        """Calculate the dimensionless entropy.

        Parameters:
            temperature (float): Temperature(s) in K
            pressure (float, optional): Pressure in atm. Default is 1 atm.
            verbose (bool): Print a table breaking down each contribution
        """
        if self.specie_type == 'ideal gas':
            entropy = self.model.get_entropy(
                temperature,
                pressure=pressure*c.convert_unit(from_='atm', to='Pa'),
                verbose=verbose)
        else:
            entropy = self.model.get_entropy(temperature, verbose=verbose)

        return entropy / c.R('eV/K')

    def get_GoRT(self, temperature, pressure=1.0, verbose=False):
        """Calculate the dimensionless Gibbs energy.

        Parameters:
            temperature (float): Temperature(s) in K
            pressure (float, optional): Pressure in atm. Default is 1 atm.
            verbose (bool): Print a table breaking down each contribution
        """
        if self.specie_type == 'ideal gas':
            G = self.model.get_gibbs_energy(
                temperature,
                pressure=pressure*c.convert_unit(from_='bar', to='Pa'),
                verbose=verbose)
        else:
            G = self.model.get_helmholtz_energy(temperature, verbose=verbose)

        return G / (c.kb('eV/K') * temperature)
