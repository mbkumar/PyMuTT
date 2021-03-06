Outline to convert DFT-generated data to empirical models
=========================================================

*Sample code here can be found
in*\ `PyMuTT.examples.VASP_to_thermdat.example1`_

.. figure:: https://github.com/VlachosGroup/PyMuTT/blob/master/docs/flowchart_dft_to_empirical.png
   :alt: Flowchart of DFT data to empirical

   Flowchart of DFT data to empirical

1. **(Optional) Read the Reference data.** Reference data is needed to
   adjust DFT enthalpies to real-world enthalpies. It is typical to use
   gas-phase molecules since experimental data is readily available. The
   number of references should be greater than or equal to the number of
   elements.

For this example, our references are specified in the Excel file,
`references.xlsx`_. The file contains all the data we need for
referencing, including: \* ``name`` (identify the molecule being
described) \* ``elements~x`` (number of element x in the formula unit)
\* ``thermo_model`` (statistical thermodynamic model to use. In this
case, we use IdealGas) \* ``T_ref`` (reference temperature corresponding
to ``HoRT_ref`` in K) \* ``HoRT_ref`` (experimental dimensionless
enthalpy of formation) \* ``potentialenergy`` (electronic energy in eV),
\* ``geometry`` (linear, nonlinear, monatomic), \* ``atoms`` (the
location of the CONTCAR files), \* ``symmetrynumber`` (symmetry number
of molecule), \* ``spin`` (number of unpaired electrons) \*
``vib_wavenumber`` (vibrational frequencies in 1/cm),

``BaseThermo`` is the parent class of empirical classes. For
referencing, ``BaseThermo`` is sufficient but any empirical model could
have been used.

.. code:: python

   from pprint import pprint
   from PyMuTT.io_.excel import read_excel
   from PyMuTT.models.empirical import BaseThermo
   from PyMuTT.models.empirical.references import References

   refs_path = './references.xlsx'
   refs_input = read_excel(io=refs_path)
   refs = References([BaseThermo(**ref_input) for ref_input in refs_input])

   #Printing out refs_input to ensure data was read correctly.
   pprint(refs_input)

The printout shown by ``pprint(refs_input)`` is shown here: \``\`
[{‘HoRT_ref’: 0.0, ‘T_ref’: 298, ‘atoms’: Atoms(symbols=‘H2’, pbc=True,
cell=[20.0, 21.0, 22.0]), ‘elements’: {‘H’: 2, ‘O’: 0}, ‘geometry’:
‘linear’, ‘name’: ‘H2’, ‘phase’: ‘G’, ‘potentialenergy’: -6.7598,
‘spin’: 0, ‘symmetrynumber’: 2, ‘thermo_model’: <class
‘PyMuTT.models.statmech.idealgasthermo.IdealGasThermo’>, ‘vib_energies’:
[0.5338981843116086]}, {‘HoRT_ref’: -97.60604333597571, ‘T_ref’: 298,
‘atoms’: Atoms(symbols=‘OH2’, pbc=True, cell=[20.0, 21.0, 22.0]),
‘elements’: {‘H’: 2, ‘O’: 1}, ‘geometry’: ‘nonlinear’, ‘name’: ‘H2O’,
‘phase’: ‘G’, ‘potentialenergy’: -14.2209, ‘spin’: 0, ‘symmetrynumber’:
2, ‘thermo_model’: <class
‘PyMuTT.models.statmech.idealgasthermo.IdealGasThermo’>, ‘vib_energies’:
[0.47429336414391626, 0.460014128927786,

.. _PyMuTT.examples.VASP_to_thermdat.example1: https://github.com/VlachosGroup/PyMuTT/tree/master/examples/VASP_to_thermdat/example1
.. _`references.xlsx`: https://github.com/VlachosGroup/PyMuTT/blob/master/examples/VASP_to_thermdat/example1/references.xlsx