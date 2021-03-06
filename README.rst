Python Multiscale Thermochemistry Toolbox (PyMuTT)
==================================================

The **Py**\ thon **Mu**\ ltiscale **T**\ hermochemistry **T**\ oolbox
(PyMuTT) is a Python library for Thermochemistry developed by the
Vlachos Research Group at the University of Delaware. This code was
originally developed to convert *ab-initio* data from DFT to observable
thermodynamic properties such as heat capacity, enthalpy, entropy, and
Gibbs energy. These properties can be fit to empirical equations and
written to different formats. 

Documentation available at: https://vlachosgroup.github.io/PyMuTT/

Useful Topics
-------------

-  `Outline to convert DFT data to empirical forms`_
-  `Explanation of enthalpy referencing`_
-  `Supported IO operations`_
-  `Examples`_
-  `How to contribute`_

Developers
----------

-  Gerhard Wittreich, P.E. (wittregr@udel.edu)
-  Jonathan Lym (jlym@udel.edu)

Dependencies
------------

-  Python3
-  `Atomic Simulation Environment`_: Used for I/O operations and to
   calculate thermodynamic properties
-  `Numpy`_: Used for vector and matrix operations
-  `Pandas`_: Used to import data from Excel files
-  `SciPy`_: Used for fitting heat capacities.
-  `Matplotlib`_: Used for plotting thermodynamic data

Getting Started
---------------

1. Install the dependencies
2. Download the repository to your local machine
3. Add to parent folder to PYTHONPATH
4. Run the tests by navigating to the `tests directory`_ in a
   command-line interface and inputting the following command:

::

   python -m unittest

The expected output is shown below. The number of tests will not
necessarily be the same.

::

   .........................
   ----------------------------------------------------------------------
   Ran 25 tests in 0.020s

   OK

5. Look at `examples using the code`_

License
-------

This project is licensed under the MIT License - see the `LICENSE.md`_
file for details.

.. _Outline to convert DFT data to empirical forms: https://vlachosgroup.github.io/PyMuTT/DFT_to_Empirical_Outline.html
.. _Explanation of enthalpy referencing: https://vlachosgroup.github.io/PyMuTT/references.html
.. _Supported IO operations: https://vlachosgroup.github.io/PyMuTT/io.html
.. _Examples: https://github.com/VlachosGroup/PyMuTT/tree/master/examples
.. _How to contribute: https://vlachosgroup.github.io/PyMuTT/contributing.html
.. _Atomic Simulation Environment: https://wiki.fysik.dtu.dk/ase/
.. _Numpy: http://www.numpy.org/
.. _Pandas: https://pandas.pydata.org/
.. _SciPy: https://www.scipy.org/
.. _Matplotlib: https://matplotlib.org/
.. _tests directory: https://github.com/VlachosGroup/PyMuTT/tree/master/tests
.. _examples using the code: https://github.com/VlachosGroup/PyMuTT/tree/master/examples
.. _LICENSE.md: https://github.com/VlachosGroup/PyMuTT/blob/master/LICENSE.md