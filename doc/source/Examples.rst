Basic Examples
**************

The following examples assume the packages below have been imported:

>>> from modERec import AERA, Antenna, EnergyRec, Shower
>>> import numpy as np

General Examples
================

The first step is to declare an instance of :class:`modERec.EnergyRec`:

>>> simulation = "sim100001.hdf5"
>>> rec = EnergyRec(simulation)

Where ``simulation`` is a ``directory``, an ``.hdf5`` file or an ``ASCII`` file.

* If ``simulation`` is a ``directory`` or an ``hdf5`` file, several quantities are evaluated such as the shower plane, early-late correction weights and the fluences of the antennas in both site and shower plane.
* If ``simulation`` is an ASCII file ( Xposition in shower plane, Y position in shower plane, fluence) the AERA fit is performed (it is used for training the `C` parameters of the double gaussian model.)

Antenna positions
-----------------

The antenna possitions can be obtained in either the site plane or the shower plane.

>>> n_ant = len(rec.GRANDshower.fields.items())
>>> r_site = np.zeros((n_ant,3))
>>> for key, value in rec.GRANDshower.fields.items():
>>>     r_site[key]=value.electric.r.xyz.value
>>>
>>> r_shower = np.array([ant.r_proj for ant in rec.antenna])

Antenna fluences
----------------

In order to access the antenna fluences:

>>> fluence_arr = np.array([ant.fluence for ant in rec.antenna])

This method can bu used for the other fluences as well: :meth:`~modERec.Antenna.fluence_geo`, :meth:`~modERec.Antenna.fluence_ce`, :meth:`~modERec.Antenna.fluence_evB`, :meth:`~modERec.Antenna.fluence_evvB`.

Early-late correction
---------------------

The early-late correction is applied via the :meth:`~modERec.Antenna.wEarlyLate`.

>>> weight_arr = np.array([ant.wEarlyLate for ant in rec.antenna])

And the correction quantities are:

>>> fluence_corr = fluence_arr/(weight_arr**2)
>>>
>>> core_distance = np.linalg.norm((r_shower - rec.shower.r_Core_proj)[:,0:2],axis=1)
>>> core_distance_corr = core_distance*weight_arr

Signal inspection
=================

>>> rec.simulation_inspect()

Prints a brief summary of the simulation inputs: direction, energy, core position and geomagnetic field.

>>> id = 0
>>> rec.bool_plot=1 # Toggle plots on
>>> rec.process_antenna(id)
>>> rec.bool_plot=0

This block toggles the plots on, processes the antenna ``id`` and plots the Fourier transform of traces, the reconstructed traces in the shower plane, the hilbert envelopes and the total hilbert envelope used to evaluate the total fluence. See :func:`modERec.EnergyRec.process_antenna`

>>> rec.plot_antpos()

Plots the antenna positions in the site plane and the corresponding fluences (colorbar).

>>> rec.bool_EarlyLate
>>> rec.model_fit()

Performs the fit using the AERA double gaussian model.

>>> AERA.aeraPlot(rec)

Plots the 2D l.d.f. and the fluences in the shower plane.

hdf5 Read and Write
-------------------

For the CORSIKA/CoREAS shower one has to `localize` it before writing it to the hdf5 file. 

>>> import astropy.units as u                                                          
>>> from grand import ECEF                                                             
>>> from grand.simulation import CoreasShower, ShowerEvent                             
>>> shower = CoreasShower.load('tests/simulation/data/coreas')                         
>>> shower.localize(latitude=45 * u.deg, longitude=3 * u.deg, height=500 * u.m)        
>>> shower.dump('test-shower.hdf5')                                                    
>>> shower = ShowerEvent.load('test-shower.hdf5')                                      


Straightforward Double Gaussian fit
-----------------------------------
>>> simulation = "sim100001.hdf5"
>>> rec = EnergyRec(simulation)
>>> rec.model_fit()

Reads the simulation, performs the fit and:

* Prints the bestfit parameters, energy and radio signal to ``bestfit.out`` if ``simulation`` is a ``direcotiry`` or a ``hdf5`` file.
* Prints the bestfit parameters to ``bestfit_All.out`` if ``simulation`` is an ASCII file. Used for the trainning stage of the double gaussian method.