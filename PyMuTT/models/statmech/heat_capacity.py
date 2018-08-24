# -*- coding: utf-8 -*-
"""Common heat capacity calculations """

import numpy as np
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
        CvoR_trans (float): Dimensionless translational heat capacity
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
        CvoR_rot (float)
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

def get_CvoR_vib(vib_energies, Ts):
    """Calculate the dimensionless vibrational heat capacity

    :math:`\\frac {Cp_{vib}}{R} = \\sum_{i=1}^{n} \\bigg(\\frac {\\Theta_{V,i}}{2T}\\bigg)^2 \\frac {1}{\\big(sinh(\\frac {\\Theta_{V,i}}{2T})\\big)^2}`

    Parameters:
        vib_energies (array of floats): Vibrational energies in eV
        Ts (float or array of floats): Temperatures in K
    Returns:
        CvoR_vib (float or array): Dimensionless vibrational heat capacity
    """
    try:
        iter(Ts)
    except TypeError:
        CvoR_vib = _get_single_CvoR_vib(vib_energies, Ts)
    else:
        CvoR_vib = np.zeros_like(Ts)
        for i, T in enumerate(Ts):
            CvoR_vib[i] = _get_single_CvoR_vib(vib_energies, T)
    return CvoR_vib

def _get_single_CvoR_vib(vib_energies, T):
    """Calculate the dimensionless vibrational heat capacity

    Parameters:
        vib_energies (array of floats): Vibrational energies in eV
        T (float): Temperatures in K
    Returns:
        CvoR_vib (float): Dimensionless vibrational heat capacity
    """
    dimensionless_vibs = vib_energies/c.kb('eV/K')/T
    CvoR_vib = np.sum((0.5 * dimensionless_vibs)**2 * (
            1./np.sinh(0.5 * dimensionless_vibs))**2)
    return CvoR_vib