# -*- coding: utf-8 -*-
"""
Empirical models.
"""

import inspect
import numpy as np
from matplotlib import pyplot as plt
from warnings import warn
from scipy.stats import variation
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


class Nasa(BaseThermo):
    """Stores the information for an individual nasa specie
    Inherits from PyMuTT.models.empirical.BaseThermo

    The thermodynamic properties are calculated using the following form:

    :math:`\\frac {Cp} {R} = a_{1} + a_{2} T + a_{3} T^{2} + a_{4} T^{3} + a_{5} T^{4}`

    :math:`\\frac {H} {RT} = a_{1} + a_{2} \\frac {T} {2} + a_{3} \\frac {T^{2}} {3} + a_{4} \\frac {T^{3}} {4} + a_{5} \\frac {T^{4}} {5} + a_{6} \\frac {1} {T}`

    :math:`\\frac {S} {R} = a_{1} \\ln {T} + a_{2} T + a_{3} \\frac {T^{2}} {2} + a_{4} \\frac {T^{3}} {3} + a_{5}  \\frac {T^{4}} {4} + a_{7}`

    Attributes
    ----------
        T_low : float
            Lower temperature bound (in K)
        T_mid : float
            Middle temperature bound (in K)
        T_high : float
            High temperature bound (in K)
        a_low : (7,) `numpy.ndarray_`
            NASA polynomial to use between T_low and T_mid
        a_high : (7,) `numpy.ndarray_`
            NASA polynomial to use between T_mid and T_high
    """
    def __init__(self, T_low, T_mid, T_high, T_ref=c.T0('K'),
                 HoRT_ref=None, SoR_ref=None, **kwargs):
        super().__init__(T_ref=T_ref, HoRT_ref=HoRT_ref, **kwargs)

        self.T_low = T_low
        self.T_high = T_high
        self.T_mid = T_mid

        self.fit(HoRT_dft=HoRT_ref, SoR_ref=SoR_ref)

    def get_a(self, temperature):
        """Returns the correct polynomial range based on T_low, T_mid and T_high

        Parameters
        ----------
            temperature : float
                Temperature in K
        Returns
        -------
            a : (7,) `numpy.ndarray`_
                NASA polynomial coefficients
        """
        if temperature < self.T_mid:
            if temperature < self.T_low:
                warn('Temperature below T_low for {}'.format(self.name), RuntimeWarning)
            return self.a_low
        else:
            if temperature > self.T_high:
                warn('Temperature above T_high for {}'.format(self.name), RuntimeWarning)
            return self.a_high

    def get_CpoR(self, temperature):
        """Calculate the dimensionless heat capacity

        Parameters
        ----------
            temperature : float
                Temperature in K
        Returns
        -------
            CpoR : float
                Dimensionless heat capacity
        """
        T = temperature
        a = self.get_a(T)
        T_arr = np.array([1., T, T**2, T**3, T**4, 0., 0.])
        return np.dot(a, T_arr)

    def get_HoRT(self, temperature):
        """Calculate the dimensionless enthalpy

        Parameters
        ----------
            temperature : float
                Temperature in K
        Returns
        -------
            HoRT : float
                Dimensionless enthalpy
        """
        T = temperature
        a = self.get_a(T)
        T_arr = np.array([1., T/2., (T**2)/3., (T**3)/4., (T**4)/5., 1./T, 0.])
        return np.dot(a, T_arr)

    def get_SoR(self, temperature):
        """Calculate the dimensionless entropy

        Parameters
        ----------
            temperature : float
                Temperature in K
        Returns
        -------
            SoR : float
                Dimensionless entropy
        """
        T = temperature
        a = self.get_a(T)
        T_arr = np.array([np.log(T), T, (T**2)/2., (T**3)/3., (T**4)/4., 0., 1.])
        return np.dot(a, T_arr)

        return SoR

    def get_GoRT(self, temperature):
        """Calculate the dimensionless Gibbs free energy

        Parameters
        ----------
            temperature : float or (N,) `numpy.ndarray`_
                Temperature(s) in K
        Returns
        -------
            GoRT : float or (N,) `numpy.ndarray`_
                Dimensionless Gibbs free energy
        """
        return self.get_HoRT(temperature) - self.get_SoR(temperature)

    def fit(self, T_low, T_high, T_ref=None,
            HoRT_dft=None, HoRT_ref=None, SoR_ref=None, references=None):
        """Calculates the NASA polynomials using internal attributes

        Parameters
        ----------
            T_low : float
                Lower temperature to fit. If not specified, uses T_low attribute
            T_high : float
                High temperature to fit. If not specified, uses T_high attribute
            T_ref : float
                Reference temperature in K used fitting empirical coefficients.
                If not specified, uses T_ref attribute
            HoRT_dft : float
                Dimensionless enthalpy calculated using DFT that corresponds
                to T_ref. If not specified, uses HoRT_dft attribute. If the
                HoRT_dft attribute is not specified, uses
                self.thermo_model.get_HoRT
            HoRT_ref : float
                Dimensionless reference enthalpy that corresponds to T_ref.
                If this is specified, uses this value when fitting a_low[5]
                and a_high[5] instead of HoRT_dft and references
            SoR_ref : float
                Dimensionless entropy that corresponds to T_ref. If not
                specified, uses self.thermo_model.get_SoR
            references : ``PyMuTT.models.empirical.References``
                Contains references to calculate HoRT_ref. If not specified
                then HoRT_dft will be used without adjustment.
        """

        '''
        Processing inputs
        '''

        #Get temperatures and heat capacity data
        Ts = np.linspace(self.T_low, self.T_high)
        CpoR = np.array(map(self.thermo_model.get_CpoR, Ts))

        #Get reference temperature
        if T_ref is None:
            T_ref = self.T_ref

        #Get reference enthalpy
        if HoRT_dft is None:
            if self.HoRT_dft is None:
                self.HoRT_dft = self.thermo_model.get_HoRT(T_ref)
            HoRT_dft = self.HoRT_dft

        #Get reference entropy
        if SoR_ref is None:
            SoR_ref = self.thermo_model.get_SoR(T_ref)

        #Get references
        if references is not None:
            self.references = references

        #Set HoRT_ref
        #If references specified
        if HoRT_ref is not None:
            self.HoRT_ref = HoRT_ref
        else:
            if self.references is not None:
                self.HoRT_ref = HoRT_dft + self.references.get_HoRT_offset(
                    self.elements, Ts=self.T_ref)
            #If dimensionless DFT enthalpy specified
            elif HoRT_dft is not None:
                self.HoRT_ref = HoRT_dft
            HoRT_ref = self.HoRT_ref

        #Reinitialize coefficients
        self.a_low = np.zeros(7)
        self.a_high = np.zeros(7)

        '''
        Processing data
        '''
        self.fit_CpoR(Ts, CpoR)
        self.fit_HoRT(T_ref, HoRT_ref)
        self.fit_SoR(T_ref, SoR_ref)

    def fit_CpoR(self, Ts, CpoR):
        """Fit a[0]-a[4] coefficients in a_low and a_high attributes
        given the dimensionless heat capacity data

        Parameters
        ----------
            Ts : (N,) `numpy.ndarray_`
                Temperatures in K
            CpoR : (N,) `numpy.ndarray_`
                Dimensionless heat capacity
        """
        #If the Cp/R does not vary with temperature (occurs when no
        # vibrational frequencies are listed)
        if (np.mean(CpoR) < 1e-6 and np.isnan(variation(CpoR))) or \
                variation(CpoR) < 1e-3 or all(np.isnan(CpoR)):
            self.T_mid = Ts[int(len(Ts)/2)]
            self.a_low = np.zeros(7)
            self.a_high = np.zeros(7)
        else:
            max_R2 = -1
            R2 = np.zeros_like(Ts)
            for i, T_mid in enumerate(Ts):
                #Need at least 5 points to fit the polynomial
                if i > 5 and i < (len(Ts)-6):
                    #Separate the temperature and heat capacities into low and high range
                    (R2[i], a_low, a_high) = self._get_CpoR_R2(
                        Ts=Ts, CpoR=CpoR, i_mid=i)
            max_R2 = max(R2)
            max_i = np.where(max_R2 == R2)[0][0]
            (max_R2, a_low_rev, a_high_rev) = self._get_CpoR_R2(
                Ts=Ts, CpoR=CpoR, i_mid=max_i)
            empty_arr = np.zeros(2)
            self.T_mid = Ts[max_i]
            self.a_low = np.concatenate((a_low_rev[::-1], empty_arr))
            self.a_high = np.concatenate((a_high_rev[::-1], empty_arr))

    def _get_CpoR_R2(self, Ts, CpoR, i_mid):
        """Calculate the R2 polynomial regression value.

        Parameters
        ----------
            Ts : (N,) `numpy.ndarray_`
                Temperatures (K) to fit the polynomial
            CpoR : (N,) `numpy.ndarray_`
                Dimensionless heat capacities that correspond to T array
            i_mid : int
                Index that splits T and CpoR arrays into a lower and higher range
        Returns
        -------
            R2 : float)
                R2 value resulting from NASA polynomial fit to T and CpoR
            p_low : (5,) `numpy.ndarray_`
                Polynomial corresponding to lower range of data
            p_high : (5,) `numpy.ndarray_`
                Polynomial corresponding to high range of data
        """
        T_low = Ts[:i_mid]
        CpoR_low = CpoR[:i_mid]
        T_high = Ts[i_mid:]
        CpoR_high = CpoR[i_mid:]
        #Fit the polynomial
        p_low = np.polyfit(x = T_low, y = CpoR_low, deg = 4)
        p_high = np.polyfit(x = T_high, y = CpoR_high, deg = 4)

        #Find the R2
        CpoR_low_fit = np.polyval(p_low, T_low)
        CpoR_high_fit = np.polyval(p_high, T_high)
        CpoR_fit = np.concatenate((CpoR_low_fit, CpoR_high_fit))
        CpoR_mean = np.mean(CpoR)
        ss_reg = np.sum((CpoR_fit - CpoR_mean)**2)
        ss_tot = np.sum((CpoR - CpoR_mean)**2)
        R2 = ss_reg / ss_tot

        return (R2, p_low, p_high)

    def fit_HoRT(self, T_ref, HoRT_ref):
        """Fit a[5] coefficient in a_low and a_high attributes given the dimensionless enthalpy

        Parameters
        ----------
            T_ref : float
                Reference temperature in K
            HoRT_ref : float
                Reference dimensionless enthalpy
        """
        T_mid = self.T_mid
        a6_low = (HoRT_ref - get_nasa_HoRT(a=self.a_low, T=T_ref))*T_ref
        a6_high = (HoRT_ref - get_nasa_HoRT(a=self.a_high, T=T_ref))*T_ref

        #Correcting for offset
        H_low_last_T = get_nasa_HoRT(a=self.a_low, T=T_mid) + a6_low/T_mid
        H_high_first_T = get_nasa_HoRT(a=self.a_high, T=T_mid) + a6_high/T_mid
        H_offset = H_low_last_T - H_high_first_T

        self.a_low[5] = a6_low
        self.a_high[5] = T_mid * (a6_high/T_mid + H_offset)

    def fit_SoR(self, T_ref, SoR_ref):
        """Fit a[6] coefficient in a_low and a_high attributes given the
        dimensionless entropy

        Parameters
        ----------
            T_ref : float
                Reference temperature in K
            SoR_ref : float
                Reference dimensionless entropy
        """
        T_mid = self.T_mid
        a7_low = SoR_ref - get_nasa_SoR(a=self.a_low, T=T_ref)
        a7_high = SoR_ref - get_nasa_SoR(a=self.a_high, T=T_ref)

        #Correcting for offset
        S_low_last_T = get_nasa_SoR(a=self.a_low, T=T_mid) + a7_low
        S_high_first_T = get_nasa_SoR(a=self.a_high, T=T_mid) + a7_high
        S_offset = S_low_last_T - S_high_first_T

        self.a_low[6] = a7_low
        self.a_high[6] = a7_high + S_offset


def get_nasa_CpoR(a, T):
    """Calculate the dimensionless heat capacity using NASA polynomial form

    Parameters
    ----------
        a : (7,) `numpy.ndarray_`
            Coefficients of NASA polynomial
        T : float
            Temperature in K
    Returns
    -------
        CpoR: float
            Dimensionless heat capacity
    """
    T_arr = np.array([1., T, T**2, T**3, T**4, 0., 0.])
    return np.dot(a, T_arr)


def get_nasa_HoRT(a, T):
    """Calculate the dimensionless enthalpy using NASA polynomial form

    Parameters
    ----------
        a : (7,) `numpy.ndarray_`
            Coefficients of NASA polynomial
        T : float
            Temperature in K
    Returns
    -------
        HoRT : float
            Dimensionless enthalpy
    """
    T_arr = np.array([1., T/2., (T**2)/3., (T**3)/4., (T**4)/5., 1./T, 0.])
    return np.dot(a, T_arr)


def get_nasa_SoR(a, T):
    """Calculate the dimensionless entropy using NASA polynomial form

    Parameters
    ----------
        a : (7,) `numpy.ndarray_`
            Coefficients of NASA polynomial
        T : float
            Temperature in K
    Returns
    -------
        SoR : float
            Dimensionless entropy
    """
    T_arr = np.array([np.log(T), T, (T**2)/2., (T**3)/3., (T**4)/4., 0., 1.])
    return np.dot(a, T_arr)


def get_nasa_GoRT(a, T):
    """Calculate the dimensionless Gibbs free energy using NASA polynomial form

    Parameters
    ----------
        a : (7,) `numpy.ndarray_`
            Coefficients of NASA polynomial
        T : float
            Temperature in K
    Returns
    -------
        GoRT : float
            Dimensionless entropy
    """
    return get_nasa_HoRT(a=a, T=T)-get_nasa_SoR(a=a, T=T)
