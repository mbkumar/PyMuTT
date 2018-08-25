# -*- coding: utf-8 -*-
"""
Empirical models.
"""

import inspect
from matplotlib import pyplot as plt
import numpy as np
from PyMuTT import _pass_expected_arguments
from PyMuTT import constants as c
from pprint import pprint

class BaseThermo:
    """The Thermodynamic Parent class.
    Holds properties of a specie, the statistical-mechanical thermodynamic model.

    Attributes
    ----------
        name : str
            Name of the specie.
        phase : str
            Phase of the specie.
            G - gas.
            S - surface.
        elements : dict
            Composition of the species.
            Keys of dictionary are elements, values are stoichiometric values in a formula unit.
            e.g. CH3OH can be represented as:
            {'C': 1, 'H': 4, 'O': 1,}.
        thermo_model : `PyMuTT.models.statmech` object
            Statistical thermodynamic model.
            Object should have the following methods: `get_CpoR`, `get_HoRT`, `get_SoR`, `get_GoRT`.
        T_ref : float
            Temperature (in K) at which `HoRT_dft` was calculated. Only used for fitting empirical coefficients.
        HoRT_dft : float
            Dimensionless enthalpy calculated using DFT that corresponds to `T_ref`. Only used for fitting empirical coefficients.
        HoRT_ref : float
            Reference dimensionless enthalpy corresponding to `T_ref`.
        references : `PyMuTT.models.empirical.References.references` object
            Contains references to calculate `HoRT_ref`. If not specified then HoRT_dft will be used without adjustment.
        notes : str
            Any additional details you would like to include such as computational set up.

    """

    def __init__(self, name, phase=None, elements=None, thermo_model=None,
                 T_ref=c.T0('K'), HoRT_dft=None, HoRT_ref=None,
                 references=None, notes=None, **kwargs):
        self.name = name
        self.phase = phase
        self.elements = elements
        self.T_ref = T_ref
        self.references = references
        self.notes = notes

        #Assign self.thermo_model
        if inspect.isclass(thermo_model):
            #If class is passed, the required arguments will be guessed.
            self.thermo_model = _pass_expected_arguments(thermo_model, **kwargs)
        else:
            self.thermo_model = thermo_model

        #Calculate dimensionless DFT energy using thermo model
        if (HoRT_dft is None) and (self.thermo_model is not None):
            self.HoRT_dft = self.thermo_model.get_HoRT(Ts=self.T_ref)
        else:
            self.HoRT_dft = HoRT_dft

        if HoRT_ref is None: #Assign self.HoRT_ref
            if (references is None) or (self.HoRT_dft is None):
                self.HoRT_ref = self.HoRT_dft
            else:
                self.HoRT_ref = self.HoRT_dft + references.get_HoRT_offset(
                    elements=elements, Ts=self.T_ref)
        else:
            self.HoRT_ref = HoRT_ref

    def __repr__(self):
        out = ['{} object for Name: {}'.format(
            self.__class__.__name__, self.name)]
        for key, val in self.__dict__.items():
            if key != 'name':
                out.append('\t{}: {}'.format(key, val))
        return '\n'.join(out)


    def plot_empirical(self, T_low=None, T_high=None, Cp_units=None,
                       H_units=None, S_units=None, G_units=None):
        """Plot the thermodynamic profiles between ``T_low`` and ``T_high``
        using empirical relationship

        Parameters:
            T_low : float
                Lower temperature in K. If not specified, ``T_low`` attribute
                used.
            T_high : float
                Upper temperature in K. If not specified, ``T_high`` attribute
                used.
            Cp_units : str
                Units to plot heat capacity. See ``PyMuTT.constants.R`` for
                accepted units. If not specified, dimensionless units used.
            H_units : str
                Units to plot enthalpy. See ``PyMuTT.constants.R`` for accepted
                units but omit the '/K' (e.g. J/mol). If not specified,
                dimensionless units used.
            S_units : str
                Units to plot entropy. See ``PyMuTT.constants.R`` for accepted
                units. If not specified, dimensionless units used.
            G_units : str
                Units to plot Gibbs free energy. See ``PyMuTT.constants.R`` for
                accepted units but omit the '/K' (e.g. J/mol). If not specified,
                dimensionless units used.

        Returns:
            figure : `matplotlib.figure.Figure`_
                Figure
            axes : tuple of `matplotlib.axes.Axes.axis`_
                Axes of the plots.
                0. Cp
                1. H
                2. S
                3. G
        """
        if T_low is None:
            T_low = self.T_low
        if T_high  is None:
            T_high = self.T_high
        Ts = np.linspace(T_low, T_high)

        f, ax = plt.subplots(4, sharex=True)
        '''
        Heat Capacity
        '''
        ax[0].set_title('Specie: {}'.format(self.name))
        Cp_plot = np.array(map(self.get_CpoR, Ts))
        if Cp_units is None:
            ax[0].set_ylabel('Cp/R')
        else:
            ax[0].set_ylabel('Cp ({})'.format(Cp_units))
            Cp_plot *= c.R(Cp_units)
        ax[0].plot(Ts, Cp_plot, 'r-')

        '''
        Enthalpy
        '''
        H_plot = np.array(map(self.get_HoRT, Ts))
        if H_units is None:
            ax[1].set_ylabel('H/RT')
        else:
            ax[1].set_ylabel('H ({})'.format(H_units))
            H_plot *= c.R('{}/K'.format(H_units)) * Ts
        ax[1].plot(Ts, H_plot, 'g-')

        '''
        Entropy
        '''
        S_plot = np.array(map(self.get_SoR, Ts))
        if S_units is None:
            ax[2].set_ylabel('S/R')
        else:
            ax[2].set_ylabel('S ({})'.format(S_units))
            S_plot *= c.R(S_units)
        ax[2].plot(Ts, S_plot, 'b-')

        '''
        Gibbs energy
        '''
        ax[3].set_xlabel('Temperature (K)')
        G_plot = np.array(map(self.get_GoRT, Ts))
        if G_units is None:
            ax[3].set_ylabel('G/RT')
        else:
            ax[3].set_ylabel('G ({})'.format(G_units))
            G_plot *= c.R('{}/K'.format(G_units)) * Ts
        ax[3].plot(Ts, G_plot, 'k-')

        return f, ax

    def plot_thermo_model(self, T_low=None, T_high=None, Cp_units=None,
                          H_units=None, S_units=None, G_units=None):
        """Plots the thermodynamic profiles between ``T_low`` and ``T_high``
        using empirical relationship

        Parameters:
            T_low : float
                Lower temperature in K.
            T_high : float
                Upper temperature in K.
            Cp_units : str
                Units to plot heat capacity. See ``PyMuTT.constants.R`` for
                accepted units. If not specified, dimensionless units used.
            H_units : str
                Units to plot enthalpy. See ``PyMuTT.constants.R`` for
                accepted units but omit the '/K' (e.g. J/mol). If not
                specified, dimensionless units used.
            S_units : str
                Units to plot entropy. See ``PyMuTT.constants.R`` for accepted
                units. If not specified, dimensionless units used.
            G_units : str
                Units to plot Gibbs free energy. See ``PyMuTT.constants.R`` for
                accepted units but omit the '/K' (e.g. J/mol). If not specified,
                dimensionless units used.

        Returns:
            figure : `matplotlib.figure.Figure`_
                Figure
            axes : tuple of `matplotlib.axes.Axes.axis`_
                Axes of the plots.
                0. Cp
                1. H
                2. S
                3. G
        """
        if T_low is None:
            T_low = self.T_low
        if T_high  is None:
            T_high = self.T_high
        Ts = np.linspace(T_low, T_high)

        f, ax = plt.subplots(4, sharex=True)
        '''
        Heat Capacity
        '''
        ax[0].set_title('Specie: {}'.format(self.name))
        Cp_plot = np.array(map(self.thermo_model.get_CpoR, Ts))
        if Cp_units is None:
            ax[0].set_ylabel('Cp/R')
        else:
            ax[0].set_ylabel('Cp ({})'.format(Cp_units))
            Cp_plot *= c.R(Cp_units)
        ax[0].plot(Ts, Cp_plot, 'r-')

        '''
        Enthalpy
        '''
        H_plot = np.array(map(self.thermo_model.get_HoRT, Ts))
        # The below function also need to take single temp
        if self.references is not None:
            offsets = np.array(
                self.references.get_HoRT_offset(elements=self.elements, Ts=Ts))
            H_plot += offsets

        if H_units is None:
            ax[1].set_ylabel('H/RT')
        else:
            ax[1].set_ylabel('H ({})'.format(H_units))
            H_plot *= c.R('{}/K'.format(H_units)) * Ts
        ax[1].plot(Ts, H_plot, 'g-')

        '''
        Entropy
        '''
        S_plot = np.array(map(self.thermo_model.get_SoR, Ts))
        if S_units is None:
            ax[2].set_ylabel('S/R')
        else:
            ax[2].set_ylabel('S ({})'.format(S_units))
            S_plot *= c.R(S_units)
        ax[2].plot(Ts, S_plot, 'b-')

        '''
        Gibbs energy
        '''
        ax[3].set_xlabel('Temperature (K)')
        G_plot = np.array(map(self.thermo_model.get_GoRT, Ts))
        if self.references is not None:
            offsets = np.array(
                self.references.get_HoRT_offset(elements=self.elements, Ts=Ts))
            G_plot += offsets

        if G_units is None:
            ax[3].set_ylabel('G/RT')
        else:
            ax[3].set_ylabel('G ({})'.format(G_units))
            G_plot *= c.R('{}/K'.format(G_units)) * Ts
        ax[3].plot(Ts, G_plot, 'k-')

        return f, ax

    def plot_thermo_model_and_empirical(self, T_low=None, T_high=None,
                                        Cp_units=None, H_units=None,
                                        S_units=None, G_units=None):
        """Plots the thermodynamic profiles between ``T_low`` and ``T_high``
        using empirical relationship

        Parameters:
            T_low : float
                Lower temperature in K.
            T_high : float
                Upper temperature in K.
            Cp_units : str
                Units to plot heat capacity. See ``PyMuTT.constants.R``
                for accepted units. If not specified, dimensionless units used.
            H_units : str
                Units to plot enthalpy. See ``PyMuTT.constants.R`` for accepted
                units but omit the '/K' (e.g. J/mol). If not specified,
                dimensionless units used.
            S_units : str
                Units to plot entropy. See ``PyMuTT.constants.R`` for accepted
                units. If not specified, dimensionless units used.
            G_units : str
                Units to plot Gibbs free energy. See ``PyMuTT.constants.R`` for
                accepted units but omit the '/K' (e.g. J/mol). If not specified,
                dimensionless units used.

        Returns:
            figure : `matplotlib.figure.Figure`_
                Figure
            axes : tuple of `matplotlib.axes.Axes.axis`_
                Axes of the plots.
                0. Cp
                1. H
                2. S
                3. G
        """
        if T_low is None:
            T_low = self.T_low
        if T_high  is None:
            T_high = self.T_high
        Ts = np.linspace(T_low, T_high)

        f, ax = plt.subplots(4, sharex=True)
        '''
        Heat Capacity
        '''
        ax[0].set_title('Specie: {}'.format(self.name))
        Ts, Cp_plot_thermo_model, Cp_plot_empirical = self.compare_CpoR(Ts=Ts)
        if Cp_units is None:
            ax[0].set_ylabel('Cp/R')
        else:
            ax[0].set_ylabel('Cp ({})'.format(Cp_units))
            Cp_plot_thermo_model *= c.R(Cp_units)
            Cp_plot_empirical *= c.R(Cp_units)

        ax[0].plot(Ts, Cp_plot_thermo_model, 'r-', label = 'Stat Mech Model')
        ax[0].plot(Ts, Cp_plot_empirical, 'b-', label = 'Empirical Model')
        ax[0].legend()

        '''
        Enthalpy
        '''
        Ts, H_plot_thermo_model, H_plot_empirical = self.compare_HoRT(Ts=Ts)

        if H_units is None:
            ax[1].set_ylabel('H/RT')
        else:
            ax[1].set_ylabel('H ({})'.format(H_units))
            H_plot_thermo_model *= c.R('{}/K'.format(H_units)) * Ts
            H_plot_empirical *= c.R('{}/K'.format(H_units)) * Ts
        ax[1].plot(Ts, H_plot_thermo_model, 'r-')
        ax[1].plot(Ts, H_plot_empirical, 'b-')

        '''
        Entropy
        '''
        Ts, S_plot_thermo_model, S_plot_empirical = self.compare_SoR(Ts=Ts)
        if S_units is None:
            ax[2].set_ylabel('S/R')
        else:
            ax[2].set_ylabel('S ({})'.format(S_units))
            S_plot_thermo_model *= c.R(S_units)
            S_plot_empirical *= c.R(S_units)
        ax[2].plot(Ts, S_plot_thermo_model, 'r-')
        ax[2].plot(Ts, S_plot_empirical, 'b-')

        '''
        Gibbs energy
        '''
        ax[3].set_xlabel('Temperature (K)')
        Ts, G_plot_thermo_model, G_plot_empirical = self.compare_GoRT(Ts=Ts)
        if G_units is None:
            ax[3].set_ylabel('G/RT')
        else:
            ax[3].set_ylabel('G ({})'.format(G_units))
            G_plot_thermo_model *= c.R('{}/K'.format(G_units)) * Ts
            G_plot_empirical *= c.R('{}/K'.format(G_units)) * Ts
        ax[3].plot(Ts, G_plot_thermo_model, 'r-')
        ax[3].plot(Ts, G_plot_empirical, 'b-')

        return f, ax


    def compare_CpoR(self, Ts=None):
        """Compares the dimensionless heat capacity of the statistical model
        and the empirical model

        Parameters
        ----------
            Ts : (N,) `numpy.ndarray`_ or float, optional
                Temperatures (in K) to calculate CpoR. If None, generates
                a list of temperatures between self.T_low and self.T_high
        Returns
        -------
            Ts : (N,) `numpy.ndarray`_
                Temperatures in K
            CpoR_statmech : (N,) `numpy.ndarray`_ or float
                Dimensionless heat capacity of statistical thermodynamic model
            CpoR_empirical :((N,) `numpy.ndarray`_ or float
                Dimensionless heat capacity of empirical model
        """
        if Ts is None:
            Ts = np.linspace(self.T_low, self.T_high)

        CpoR_statmech = np.array(map(self.thermo_model.get_CpoR, Ts))
        CpoR_empirical = np.array(map(self.get_CpoR, Ts))

        return (Ts, CpoR_statmech, CpoR_empirical)

    def compare_HoRT(self, Ts=None):
        """Compares the dimensionless enthalpy of the statistical model and
        the empirical model

        Parameters
        ----------
            Ts : (N,) `numpy.ndarray`_ or float, optional
                Temperatures (in K) to calculate CpoR. If None, generates a
                list of temperatures between self.T_low and self.T_high
        Returns
        -------
            Ts : (N,) `numpy.ndarray`_ or float
                Temperatures in K
            CpoR_statmech : (N,) `numpy.ndarray`_ or float
                Dimensionless heat capacity of statistical thermodynamic model
            CpoR_empirical :((N,) `numpy.ndarray`_ or float
                Dimensionless heat capacity of empirical model
        """
        if Ts is None:
            Ts = np.linspace(self.T_low, self.T_high)

        if self.references is not None:
            H_offset = np.array(
                self.references.get_HoRT_offset(elements=self.elements, Ts=Ts))
        else:
            H_offset = np.zeros_like(Ts)

        HoRT_statmech = np.array(map(self.thermo_model.get_HoRT, Ts)) + H_offset
        HoRT_empirical = np.array(map(self.get_HoRT, Ts))

        return (Ts, HoRT_statmech, HoRT_empirical)

    def compare_SoR(self, Ts = None):
        """Compares the dimensionless entropy of the statistical model and
        the empirical model

        Parameters
        ----------
            Ts : (N,) `numpy.ndarray`_ or float, optional
                Temperatures (in K) to calculate CpoR. If None, generates a
                list of temperatures between self.T_low and self.T_high
        Returns
        -------
            Ts : (N,) `numpy.ndarray`_ or float
                Temperatures in K
            CpoR_statmech : (N,) `numpy.ndarray`_ or float
                Dimensionless heat capacity of statistical thermodynamic model
            CpoR_empirical :((N,) `numpy.ndarray`_ or float
                Dimensionless heat capacity of empirical model
        """
        if Ts is None:
            Ts = np.linspace(self.T_low, self.T_high)

        SoR_statmech = np.array(map(self.thermo_model.get_SoR, Ts))
        SoR_empirical = np.array(map(self.get_SoR, Ts))

        return (Ts, SoR_statmech, SoR_empirical)

    def compare_GoRT(self, Ts = None):
        """Compares the dimensionless Gibbs energy of the statistical model
        and the empirical model

        Parameters
        ----------
            Ts : (N,) `numpy.ndarray`_ or float, optional
                Temperatures (in K) to calculate CpoR. If None, generates a
                list of temperatures between self.T_low and self.T_high
        Returns
        -------
            Ts : (N,) `numpy.ndarray`_ or float
                Temperatures in K
            CpoR_statmech : (N,) `numpy.ndarray`_ or float
                Dimensionless heat capacity of statistical thermodynamic model
            CpoR_empirical : (N,) `numpy.ndarray`_ or float
                Dimensionless heat capacity of empirical model
        """
        if Ts is None:
            Ts = np.linspace(self.T_low, self.T_high)

        if self.references is not None:
            offsets = self.references.get_HoRT_offset(elements=self.elements,
                                                       Ts=Ts)
        else:
            offsets = np.zeros_like(Ts)

        GoRT_statmech = np.array(map(self.thermo_model.get_GoRT, Ts)) + offsets
        GoRT_empirical = np.array(map(self.get_GoRT, Ts))

        return (Ts, GoRT_statmech, GoRT_empirical)
