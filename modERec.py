##
#@mainpage
# PROTOTYPE Code for energy reconstruction \n
# Ready to read simulations from CORSIKA/Coreas \n
# This implementation is object oriented \n
# Date: September 16, 2019 \n \n
#
# A module to reconstruc the energy of events simulated using the Corsika/Coreas. \n\n
#
# The class EnergyRec is the main energy reconstruction class. \n \n
# It has the following inner classes: \n
# --> EnergyRec.Antenna which holds the antenna specific methods and variables.\n
# --> EnergyRec.AERA with AERA specific methods.\n
# --> EnergyRec.Shower that stores shower specific variables.\n
# --> EnergyRec.SymFit implements the symmetric signal fit.\n \n
# *** Updates on November 13, 2019:
#     Eval_geo_ce_fluences implemented; \n
#     EnergyRec.SymFit implemented.\n \n \n
# *** Updates on April 05, 2020:
#     grand software fully integrated; \n
# Written by Bruno L. Lago

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import ctypes
from copy import deepcopy

from scipy.fftpack import fft, ifft
from scipy.signal import hilbert
from scipy import stats
from pathlib import Path
import math

from grand.simulation import ShowerEvent, ZhairesShower
import astropy.units as u
from astropy.coordinates import CartesianRepresentation
from grand import Rotation

# for custom_from_datafile
import h5py
from grand import ECEF, LTP
from grand.simulation.shower.generic import FieldsCollection, CollectionEntry
import re
from grand.simulation import ElectricField
from grand.simulation.pdg import ParticleCode
from astropy.coordinates import PhysicsSphericalRepresentation
from datetime import datetime

import sys

from astropy.table import Table

class Antenna:
    """
    A class for the antenna signal processing.

    It has tools for the FFT, trace_recover and fluence evaluation.

    Attributes
    ----------
    ID:
        An id for the antenna
    fluence:
        he total fluence.
    sigma_f:
        An estimate of the uncertainty of the fluence.
    fluence_geo:
        The geomagnetic component of the fluence
    fluence_ce:
        The charge ecess component of the fluence
    fluence_evB:
        The fluence on the evB direction
    fluence_evvB:
        The fluence on the evvB direction
    wEarlyLate:
        A weight to be used for the early-late correction
    r_proj:
        The position of the antenna in the shower plane

    """
    ## An id for the antenna
    ID = None
    ## The total fluence.
    fluence = None
    ## An estimate of the uncertainty of the fluence.
    sigma_f = None
    ## The geomagnetic component of the fluence
    fluence_geo = None
    ## The charge ecess component of the fluence
    fluence_ce = None
    ## The fluence on the evB direction
    fluence_evB = None
    ## The fluence on the evvB direction
    fluence_evvB = None
    ## A wight to be used for the early-late correction
    wEarlyLate = None
    ## The position of the antenna in the shower plane
    r_proj = None

    def __init__(self,ID):
        """
        The default init function for the class Antenna.

        """
        self.ID = ID

    @staticmethod
    def fft_filter(traces, nu_low = 50, nu_high = 200, bool_plot = False):
        """
        Evaluates the FFT of the signal.

        A filter is applied with width given by instance.antenna.nu_high and instance.antenna.nu_low.

        Parameters
        ----------
        traces:
            The traces to be filtered
        nu_low:
            lower bound of the frequency band;
        nu_high:
            upper bound of the frequency band;
        bool_plot: bool
            toggles plots on and off;

        Returns
        -------
        traces_fft: list
            The Fourier transform of the traces.        

        """
        time_arr = traces[:,0]
        trace1 = traces[:,1]
        trace2 = traces[:,2]
        trace3 = traces[:,3]
        # Number of sample points
        N = time_arr.size
        
        # sample spacing
        sampling_rate = 1/((time_arr[1]-time_arr[0])*1.e-9) # uniform sampling rate (convert time from ns to seconds)
        T = 1/sampling_rate
        yf1 = fft(trace1)
        yf2 = fft(trace2)
        yf3 = fft(trace3)
        xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
        
        nu_low = nu_low * 10**6 # from MHz to Hz
        nu_high = nu_high * 10**6 # from MHz to Hz
        for i in range(xf.size):
            if(xf[i]<nu_low or xf[i]>nu_high):
                yf1[i]=0
                yf1[-i]=0 # negative frequencies are backordered (0 1 2 3 4 -4 -3 -2 -1)
                yf2[i]=0
                yf2[-i]=0
                yf3[i]=0
                yf3[-i]=0
        
        if(bool_plot):
            fig = plt.figure(figsize=(15,3))
            fig.suptitle('Fourier transform of the traces', fontsize=16,y=1)
            plt.subplot(131)
            plt.plot(xf, 2/N * np.abs(yf1[0:N//2]),'r')
            #plt.xlabel("frequency in Hz")
            plt.ylabel("signal in a.u.")
            plt.xlabel("frequency in Hz")
            ax = plt.gca()
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
            
            plt.subplot(132)
            plt.plot(xf, 2/N * np.abs(yf2[0:N//2]),'b')
            plt.xlabel("frequency in Hz")
            #plt.ylabel("signal in a.u.")
            ax = plt.gca()
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
            plt.subplot(133)
            plt.plot(xf, 2/N * np.abs(yf3[0:N//2]),'k')
            #plt.xlabel("frequency in Hz")
            #plt.ylabel("signal in a.u.")
            ax = plt.gca()
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            
            plt.show()
                
        return np.c_[yf1, yf2, yf3]

    @staticmethod    
    def trace_recover(t,traces_fft,bool_plot = False):
        """
        Reconstructs the trace after the FFT and filter.

        Parameters
        ----------
        t:
            The time array for the traces;
        traces_fft:
            The Fourier transform of the traces.
        bool_plot: bool
            toggles plots on and off;

        Returns
        -------
        traces_rc: list
            The reconstructed traces.
                    
        """
        yy1 = ifft(traces_fft[:,0]).real
        yy2 = ifft(traces_fft[:,1]).real
        yy3 = ifft(traces_fft[:,2]).real
        
        xx = t-np.min(t)
        
        if(bool_plot):
            fig = plt.figure(figsize=(15,3))
            fig.suptitle('Reconstructed traces in shower plane', fontsize=16,y=1)
            plt.subplot(131)
            plt.plot(xx, yy1.real,'r')
            #plt.xlabel("time in ns")
            plt.ylabel("signal in V/m")
            ax = plt.gca()
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
            
            plt.subplot(132)
            plt.plot(xx, yy2.real,'b')
            plt.xlabel("time in ns")
            #plt.ylabel("signal in V/m")
            ax = plt.gca()
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
            plt.subplot(133)
            plt.plot(xx, yy3.real,'k')
            #plt.xlabel("time in ns")
            #plt.ylabel("signal in V/m")
            ax = plt.gca()
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            
            plt.show()
        
        return np.c_[t,yy1, yy2, yy3]

    @staticmethod    
    def hilbert_envelope(traces_rec, bool_plot = False):
        r"""
        Evaluates the hilbert envelope of the recunstructed traces.

        .. math::

            \mathcal{H}\{f(x)\}:=H(x)=\frac{1}{\pi}{\rm p.v.}\int_{-\infty}^\infty \frac{f(u)}{u-x}{\rm d}u
        
        Parameters
        ----------
        traces_rec:
            The reconstructed traces.
        bool_plot:
            toggles plots on and off;

        Returns
        -------
        hilbert: list
            The hilbert envelopes [total, in evB direction, in evvB direction, in ev direction].
        """
        tt = traces_rec[:,0]-np.min(traces_rec[:,0])
        envelope1 = hilbert(traces_rec[:,1].real)
        envelope2 = hilbert(traces_rec[:,2].real)
        envelope3 = hilbert(traces_rec[:,3].real)
    
        if(bool_plot):    
            fig = plt.figure(figsize=(15,3))
            fig.suptitle('Hilbert Envelopes in the shower plane', fontsize=16,y=1)
            plt.subplot(131)
            plt.plot(tt,traces_rec[:,1].real, 'r', label='signal')
            plt.plot(tt, np.abs(envelope1), label='envelope')
            #plt.xlabel("time in ns")
            plt.ylabel("signal in V/m")
            ax = plt.gca()
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
            
            plt.subplot(132)
            plt.plot(tt,traces_rec[:,2].real, 'b', label='signal')
            plt.plot(tt, np.abs(envelope2), label='envelope')
            plt.xlabel("time in ns")
            #plt.ylabel("signal in V/m")
            ax = plt.gca()
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
            plt.subplot(133)
            plt.plot(tt,traces_rec[:,3].real, 'k', label='signal')
            plt.plot(tt, np.abs(envelope3), label='envelope')
            #plt.xlabel("time in ns")
            #plt.ylabel("signal in V/m")
            ax = plt.gca()
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            
            plt.show()
    
        hilbert_env = np.sqrt(envelope1**2 + envelope2**2 + envelope3**2)
        return np.c_[hilbert_env,envelope1,envelope2,envelope3]
        
    def compute_fluence(self,t,hilbert_env, SNR_thres = 10, bool_plot = False):
        r"""
        Computes the fluence for a given antenna.
        
        .. math::
            f = \epsilon_0 c\left(\Delta t \sum_{t_1}^{t_2} \left| \vec{E}(t_i)\right|^2 - \Delta t \frac{t_2-t_1}{t_4-t_3} \sum_{t_3}^{t_4} \left| \vec{E}(t_i)\right|^2 \right)
            
        It has a threshold for the SNR set by instance.SNR_thres.

        Parameters
        ----------
        self: modERec.EnergyRec.Antenna
            A class instance.
        t:
            The time array for the traces;
        hilbert_env:
            The hilbert envelopes;
        SNR_thres:
            The signal to noise ratio threshold;
        bool_plot: bool
            toggles plots on and off;


        Notes
        -----
        Fills self.fluence, self.fluence_geo, self.fluence_ce, self.fluence_evB and self.luence_evvB.
        
        """
        tt = t-np.min(t)
        delta_tt = (tt[1]-tt[0])*1e-9 # convert from ns to seconds
        
        envelope2 = hilbert_env[:,0]**2.
    
        tmean_index = np.where(envelope2==np.max(envelope2))[0][0] # Index of the hilbert envelope maximum
        
        if(bool_plot):
            plt.plot(tt,np.abs(envelope2))
            plt.xlabel("time in nanoseconds")
            plt.ylabel(r"E$^2$ in V$^2$/m$^2$")
            plt.show()

        if(tt[-1]-tt[tmean_index] < 150):
            #print("Peak too close to the end of the signal. Skipping")
            self.fluence = -2e-5
            self.fluence_geo = -2e-5
            self.fluence_ce = -2e-5
            self.fluence_evB = -2e-5
            self.fluence_evvB = -2e-5
            return
        
        time_100ns_index = np.where(tt<=100)[0][-1] # Index of 100 ns bin
    
        if(tmean_index<time_100ns_index):
            #tmean_index = time_100ns_index
            #print("Peak too close to the beginning of the signal. Skipping")
            self.fluence = -1e-5
            self.fluence_geo = -1e-5
            self.fluence_ce = -1e-5
            self.fluence_evB = -1e-5
            self.fluence_evvB = -1e-5
            return
        
        
        t1 = tt[tmean_index-time_100ns_index]
        t2 = tt[tmean_index+time_100ns_index]
        
        if(tmean_index < tt.size//2):
            # background from the end of the trace
            t3 = tt[-1-2*time_100ns_index]
            t4 = tt[-1]
        else:
            # background from the beginning of the trace
            t3 = tt[0]
            t4 = tt[2*time_100ns_index]
    
        signal = np.sum(np.abs(envelope2)[(tt>=t1) & (tt<=t2)])*delta_tt # N^2 * Coulomb^-2 * s
        bkg = np.sum(np.abs(envelope2)[(tt>=t3) & (tt<t4)])*delta_tt*(t2-t1)/(t4-t3)
        
        epsilon0 = 8.8541878128e-12 # Coulomb^2 * N^-1 * m^-2
        c = 299792458 # m * s^-1
        
        Joule_to_eV = 1/1.602176565e-19
        
        SNR = np.sqrt(signal/bkg)
        
        if(SNR>SNR_thres):
            my_fluence = epsilon0*c* (signal - bkg)*Joule_to_eV # eV * m^-2
            self.fluence = my_fluence

            signal_evB = np.sum(np.abs(hilbert_env[:,1]**2.)[(tt>=t1) & (tt<=t2)])*delta_tt
            bkg_evB = np.sum(np.abs(hilbert_env[:,1]**2.)[(tt>=t3) & (tt<t4)])*delta_tt*(t2-t1)/(t4-t3)
            self.fluence_evB = (signal_evB - bkg_evB)*epsilon0*c*Joule_to_eV

            signal_evvB = np.sum(np.abs(hilbert_env[:,2]**2.)[(tt>=t1) & (tt<=t2)])*delta_tt
            bkg_evvB = np.sum(np.abs(hilbert_env[:,2]**2.)[(tt>=t3) & (tt<t4)])*delta_tt*(t2-t1)/(t4-t3)
            self.fluence_evvB = (signal_evvB-bkg_evvB)*epsilon0*c*Joule_to_eV
    
        else:
            self.fluence = -1
            self.fluence_evB = -1
            self.fluence_evB = -1


    @staticmethod    
    def offset_and_cut(traces, bool_cut = True):
        """
        Performs cuts and offsets the traces.
        
        The offset prevents problems due to traces too close to the time origin.
        The cut, reduces the time window to speed up the code.

        Parameters
        ----------
        traces:
            The traces to be cut;
        bool_cut:
            Toggles the bandwith cut;

        Returns
        -------
        traces_cut: list
            The cut traces.
        
        """
        traces0 = traces[:,0]
        deltat = traces0[1]-traces0[0]
        min_time = traces0[0]
    
        offset = int(100/deltat)
    
        extra_time = np.linspace(min_time-offset*deltat,min_time-1,offset)
        traces0 = np.insert(traces[:,0],0,extra_time)
        traces1 = np.insert(traces[:,1],0,np.zeros(offset))
        traces2 = np.insert(traces[:,2],0,np.zeros(offset))
        traces3 = np.insert(traces[:,3],0,np.zeros(offset))

        my_traces = np.c_[traces0,traces1,traces2,traces3]
        
        if(bool_cut):
            global_peak = np.max(np.abs(my_traces[:,1:4]))
            peak_index = np.where(np.abs(my_traces[:,1:4])==global_peak)[0][0]
            peak_time = my_traces[:,0][peak_index]
            sel = ((my_traces[:,0]>peak_time-1000) & (my_traces[:,0]<peak_time+1000))
        
        return my_traces[sel,0:4]

class Shower:
        r"""
        A class for the shower parameters.

        Attributes
        ----------

        ev:
            Unitary along :math:`\vec{v}`.
        eB:
            Unitary along :math:`\vec{B}`.
        evB:
            Unitary along :math:`\vec{v}\times\vec{B}`.
        evvB:
            Unitary along :math:`\vec{v}\times\vec{v}\times\vec{B}`.
        thetaCR:
            The shower zenith in deg.
        phitCR:
            The shower azimuth in deg.
        ECR:
            The cosmic ray energy
        r_Core_proj:
            The position of the core projected into the shower plane.
        bool_plot: bool
            Toggles the plots on and off.
        d_Xmax:
            Distance to Xmax from the simultaion.
    
        """
        ## Unitary along \f$\vec{v}\f$.
        ev = None
        ## Unitary along \f$\vec{B}\f$.
        eB = None
        ## Unitary along \f$\vec{v}\times\vec{B}\f$.
        evB = None
        ## Unitary along \f$\vec{v}\times\vec{v}\times\vec{B}\f$.
        evvB = None
        ## The shower inclination.
        thetaCR = None
        ## The shower azimuthal angle.
        phiCR = None
        ## The energy of the shower.
        ECR = None
        ## The position of the core projected into the shower plane.
        r_Core_proj = None
        ## Toggles the plots on and off.
        bool_plot = False
        ## Distance to Xmax.
        d_Xmax = None

        def __init__(self):
            """
            The default init function for the class Shower.
    
            Parameters
            ----------
            self: modERec.EnergyRec.Shower
                    A class instance.
            """

class EnergyRec:
    """
    A class for the energy reconstruction.
    
    It has the inner classes: AERA, Antenna and Shower.

    Attributes
    ----------

    bool_plot: bool
        Toggles the plots on and off (default: false).
    
    bool_EarlyLate: bool
        Toggles the early-late correction on and off (default: true).
    
    nu_low:
        The lower frequency of the signal filter in MHz (default: 50).
    nu_high:
        The upper frequency of the signal filter in MHz (default: 200).
    SNR_thres:
        The signal to noise ratio threshold (default: 10).
    thres_low:
        A initial lower threshold for selecting antennas in V/m (default: 0.1e-6).
    thres_high:
        A initial upper threshold for selecting antennas in V/m (default: 1).
    f_thres:
        A final lower threshold for selecting antennas in eV/m^2 (default: 0.01).
    
    simulation: path, file
        The path to the simulation directory or file (default: None).
    
    antenna: modERec.EnergyRec.Antenna
        An array of type modERec.EnergyRec.Antenna.

    shower: modERec.EnergyRec.Shower
        An instance of the class modERec.EnergyRec.Shower.

    bestfit:
        The bestfit values of the parameters (default: None).
    
    GRANDshower:
        The shower imported using the standard grand package(default: None).
    printLevel: int
        A print level variable (default: 0).
    """
    ## Toggles the plots on and off.
    bool_plot = False
    ## Toggles the early-late correction on and off.
    bool_EarlyLate = True
    ## The lower frequency of the signal filter in MHz.
    nu_low = 50
    ## The upper frequency of the signal filter in MHz.
    nu_high = 200
    ## The signal to noise ratio threshold.
    SNR_thres = 10
    ## A initial lower threshold for selecting antennas in V/m.
    thres_low = 0.1e-6
    ## A initial upper threshold for selecting antennas in V/m.
    thres_high = 1
    ## A final lower threshold for selecting antennas in eV/m^2.
    f_thres = 0.01
    ## The path to the simulation.
    simulation = None
    ## The simulation type (coreas, zhaires, custom).
    simulation_type = "coreas"
    ## An instance of the class Antenna
    antenna = None
    ## An instance of the class Shower
    shower = None
    ## The bestfit values of the parameters
    bestfit = None
    ## The shower imported using the standard grand package
    GRANDshower = None
    ## The height of the site
    site_height = 0
    ## A print level variable
    printLevel = 0

    def __init__(self,simulation):
        """
        The default init function for the class EnergyRec.
    
        Parameters
        ----------
        self: modERec.EnergyRec
            A class instance.
        simulation:
            The path to the simulation directory or file.
        """
        self.simulation = simulation
        self.shower = Shower()

        if Path(self.simulation).is_dir() or self.simulation.endswith('hdf5'):

            if self.simulation_type == "coreas":
                self.GRANDshower = ShowerEvent.load(self.simulation)
            elif self.simulation_type == "zhaires":
                self.GRANDshower = ZhairesShower._from_datafile(self.simulation)

                # Fixing Aires to GRAND conventions
                for ant in range(len(self.GRANDshower.fields)):
                    self.GRANDshower.fields[ant].electric.E=self.GRANDshower.fields[ant].electric.E[0]

                    from astropy.coordinates.matrix_utilities import rotation_matrix
                    rotation = rotation_matrix(-90 * u.deg, axis='z')

                    self.GRANDshower.fields[ant].electric.E = self.GRANDshower.fields[ant].electric.E.transform(rotation)

                    self.GRANDshower.fields[ant].electric.r = self.GRANDshower.fields[ant].electric.r.transform(rotation)

                self.GRANDshower.localize(latitude=45.5 * u.deg, longitude=90.5 * u.deg)

            elif self.simulation_type == "custom" or self.simulation_type == "starshape":
                self.GRANDshower, bool_traces = self.custom_from_datafile(self.simulation, self.site_height)
                
                if(bool_traces):
                    # Fixing Aires to GRAND conventions
                    for ant in range(len(self.GRANDshower.fields)):
                        self.GRANDshower.fields[ant].electric.E=self.GRANDshower.fields[ant].electric.E[0]
                        from astropy.coordinates.matrix_utilities import rotation_matrix
                        rotation = rotation_matrix(-0 * u.deg, axis='z')

                        self.GRANDshower.fields[ant].electric.E = self.GRANDshower.fields[ant].electric.E.transform(rotation)

                        self.GRANDshower.fields[ant].electric.r = self.GRANDshower.fields[ant].electric.r.transform(rotation)

                    if self.simulation_type == "custom":
                        self.simulation_type == "zhaires"

            if Path(self.simulation).is_dir():
                self.GRANDshower.localize(latitude=45.5 * u.deg, longitude=90.5 * u.deg)   
            
            self.antenna = {ant : Antenna(ant) for ant in self.GRANDshower.fields.keys()}
            self.shower_projection()

            self.Eval_fluences()
            #self.plot_antpos()

            ev = self.GRANDshower.core - self.GRANDshower.maximum
            ev /= ev.norm()
            self.shower.ev = ev.xyz.value
            evB = ev.cross(self.GRANDshower.geomagnet)
            evB /= evB.norm()
            self.shower.evB = evB.xyz.value
            evvB = ev.cross(evB)
            self.shower.evvB = evvB.xyz.value
            eB = self.GRANDshower.geomagnet.xyz.value
            eB /= np.linalg.norm(eB)
            self.shower.eB = eB

            self.early_late()

            if(self.printLevel>0):
                print("* EnergyRec instance starting values summary:")
                print("--> bool_plot = ",self.bool_plot)
                print("--> bool_EarlyLate = ",self.bool_EarlyLate)
                print("--> nu_low = ",self.nu_low)
                print("--> nu_high = ",self.nu_high)
                print("--> SNR_thres = ",self.SNR_thres)
                print("--> thres_low = ",self.thres_low)
                print("--> thres_high = ",self.thres_high)
                print("--> f_thres = ",self.f_thres)
                print("\n")

        elif Path(self.simulation).is_file():
            self.model_fit(self.simulation)
        
        else:
            message = "ERROR: " + self.simulation + " not found!"
            raise ValueError(message)

    @staticmethod
    def custom_from_datafile(path: Path, site_height = 0) -> ZhairesShower:
        with h5py.File(path, 'r') as fd:
            if not 'RunInfo.__table_column_meta__' in fd['/']:
                return super()._from_datafile(path)

            for name in fd['/'].keys():
                if not name.startswith('RunInfo'):
                    break

            bool_traces = True
            event = fd[f'{name}/EventInfo']
            antennas = fd[f'{name}/AntennaInfo']
            try:
                traces = fd[f'{name}/AntennaTraces']
            except:
                bool_traces = False

            fields = FieldsCollection()

            pattern = re.compile('([0-9]+)$')
            for antenna, x, y, z, *_ in antennas:
                r = CartesianRepresentation(
                    float(x), float(y), float(z), unit=u.m)
                if(bool_traces):
                    tag = antenna.decode()
                    antenna = int(pattern.search(tag)[1])
                    tmp = traces[f'{tag}/efield'][:]
                    efield = tmp.view('f4').reshape(tmp.shape + (-1,))
                    t = np.asarray(efield[:,0], 'f8') << u.ns
                    Ex = np.asarray(efield[:,1], 'f8') << u.uV / u.m
                    Ey = np.asarray(efield[:,2], 'f8') << u.uV / u.m
                    Ez = np.asarray(efield[:,3], 'f8') << u.uV / u.m
                    E = CartesianRepresentation(Ex, Ey, Ez, copy=False),
                else:
                    t = None
                    E = None

                fields[antenna] = CollectionEntry(
                    electric=ElectricField(t = t, E = E, r = r))

            primary = {
                'Fe^56'  : ParticleCode.IRON,
                'Gamma'  : ParticleCode.GAMMA,
                'Proton' : ParticleCode.PROTON
            }[event[0, 'Primary'].decode()]

            geomagnet = PhysicsSphericalRepresentation(
                theta = float(90 + event[0, 'BFieldIncl']) << u.deg,
                phi = 0 << u.deg,
                r = float(event[0, 'BField']) << u.uT)

            try:
                latitude = event[0, 'Latitude'].decode("utf-8") << u.deg
                longitude = event[0, 'Longitude'].decode("utf-8") << u.deg
                declination = event[0, 'BFieldDecl'] << u.deg
                obstime = datetime.strptime(event[0, 'Date'].decode("utf-8").strip(),
                                            '%d/%b/%Y')
            except ValueError:
                frame = None
            else:
                origin = ECEF(latitude, longitude, 0 * u.m, # Site height = 0
                              representation_type='geodetic')
                frame = LTP(location=origin, orientation='NWU',
                            declination=declination, obstime=obstime)

            #my_core = event[0, 'CorePosition'] + np.array([0, 0, site_height])
            my_core = np.array([0, 0, site_height])

            return ZhairesShower(
                energy = float(event[0, 'Energy']) << u.EeV,
                zenith = (180 - float(event[0, 'Zenith'])) << u.deg,
                azimuth =(180 - float(event[0, 'Azimuth'])) << u.deg,
                primary = primary,

                frame = frame,
                core = CartesianRepresentation(*my_core, unit='m'),
                geomagnet = geomagnet.represent_as(CartesianRepresentation),
                maximum = CartesianRepresentation(*event[0, 'XmaxPosition'],
                                                  unit='m'),

                fields = fields
            ), bool_traces

    def simulation_inspect(self):
        """
        Outputs theta, phi, Energy and the core position.

        Parameters
        ----------
        self: modERec.EnergyRec
            A class instance.
            
        """
        

        thetaCR = self.GRANDshower.zenith
        phiCR = self.GRANDshower.azimuth
        E = self.GRANDshower.energy
        Core = self.GRANDshower.core
        B = self.GRANDshower.geomagnet

        print("* Simulation summary:")
        print("--> thetaCR = ",thetaCR)
        print("--> phiCR = ",phiCR)
        print("--> E = ",E)
        
        print("--> Core position = ", Core)

        print("--> Geomagnetic field = ", B)

    def inspect_antenna(self,id):
        """
        Plots the traces for a given antenna.   

        Parameters
        ----------
        self: modERec.EnergyRec
            A class instance.
        id: int
            The antenna id.  
    
        """
                
        if(id<len(self.GRANDshower.fields)):
            Ex = self.GRANDshower.fields[id].electric.E.x.to("V/m")
            Ey = self.GRANDshower.fields[id].electric.E.y.to("V/m")
            Ez = self.GRANDshower.fields[id].electric.E.z.to("V/m")
            EvB = self.shower.traces_proj[id].x.to("V/m").value
            EvvB = self.shower.traces_proj[id].y.to("V/m").value
            Ev = self.shower.traces_proj[id].z.to("V/m").value
            time = self.GRANDshower.fields[id].electric.t.to("ns")

        global_peak = np.max(np.abs([Ex,Ey,Ez]))
        peak_index = np.where(np.abs([Ex,Ey,Ez])==global_peak)[0][0]
        peak_time = time[peak_index]
            
        if(self.bool_plot):
            fig = plt.figure(figsize=(15,3))
            fig.suptitle('Traces', fontsize=16,y=1)
            plt.subplot(131)
            plt.plot(time,Ex,'r')
            plt.ylabel("signal in V/m")
            ax = plt.gca()
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
        
            plt.subplot(132)
            plt.plot(time,Ey,'b')
            plt.xlabel("time in ns")
            ax = plt.gca()
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
            plt.subplot(133)
            plt.plot(time,Ez,'k')
            ax = plt.gca()
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))

            fig = plt.figure(figsize=(15,3))
            fig.suptitle('Traces in shower plane', fontsize=16,y=1)
            plt.subplot(131)
            plt.plot(time,EvB,'r')
            plt.ylabel("signal in V/m")
            ax = plt.gca()
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
        
            plt.subplot(132)
            plt.plot(time,EvB,'b')
            plt.xlabel("time in ns")
            ax = plt.gca()
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    
            plt.subplot(133)
            plt.plot(time,Ev,'k')
            ax = plt.gca()
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
            
            plt.show()
            
    def process_antenna(self, id):
        """
        Process a given antenna for inspection.
        
        For a given initialized antenna, performs offset and cut, fft, trace recover, hilbert envelope and computes the fluence.
        Calls EnergyRec.Antenna.compute_fluence;

        Parameters
        ----------
        self: modERec.EnergyRec
            A class instance.
        
        """

        if(id<len(self.GRANDshower.fields)):
            time = self.GRANDshower.fields[id].electric.t.to("ns").value
            
            # Ex = self.GRANDshower.fields[id].electric.E.x.to("V/m").value
            # Ey = self.GRANDshower.fields[id].electric.E.y.to("V/m").value
            # Ez = self.GRANDshower.fields[id].electric.E.z.to("V/m").value
            # traces  = np.c_[time,Ex,Ey,Ez]
        
            EvB = self.shower.traces_proj[id].x.to("V/m").value
            EvvB = self.shower.traces_proj[id].y.to("V/m").value
            Ev = self.shower.traces_proj[id].z.to("V/m").value
            traces  = np.c_[time,EvB,EvvB,Ev]
        else:
            message = "ERROR: id = " + str(id) + " is out of the antenna array bounds!"
            raise ValueError(message)

        # Check if peak is within the threshold range
        peak = np.max(np.abs(traces[:,1:4]))
        if(peak < self.thres_low or peak > self.thres_high):
            self.antenna[id].fluence = -1
            return
            
        traces_cut = Antenna.offset_and_cut(traces)
        traces_fft = Antenna.fft_filter(traces_cut, self.nu_low, self.nu_high, self.bool_plot)
        traces_rec = Antenna.trace_recover(traces_cut[:,0],traces_fft,self.bool_plot)

        # Check if peak is within the threshold range after offset, cut and trace recover.
        if(np.max(np.abs(traces_rec[:,1:4]))<self.thres_low):
            self.antenna[id].fluence = -1
            self.antenna[id].fluence_geo = -1
            self.antenna[id].fluence_ce = -1
            self.antenna[id].fluence_evB = -1
            self.antenna[id].fluence_evvB = -1
            return
        else:
            hilbert_env = Antenna.hilbert_envelope(traces_rec, self.bool_plot)
            self.antenna[id].compute_fluence(traces_rec[:,0],hilbert_env,self.SNR_thres,self.bool_plot)
                
    
    def Eval_fluences(self):
        """
        Evaluates the geomagnetic and charge excess fluences for a set os antennas.
            
        It has a lower threshold for the fluence f_thres.

        Parameters
        ----------
        self: modERec.EnergyRec
            A class instance.

        Notes
        -----
        Fills self.antenna.fluence, self.antenna.fluence_evB, self.antenna.fluence_evvB, self.antenna.fluence geo and self.antenna.fluence_ce for all antennas.
        
        """
        n_ant = len(self.GRANDshower.fields)
    
        step = round(n_ant/10)

        try:
            antenna_list = self.antenna.values()
        except:
            antenna_list = self.antenna
            
        if self.simulation_type == "custom":
            shower_frame = self.GRANDshower.shower_frame()
            RunInfo = Table.read(self.simulation, path="RunInfo")
            EventName = RunInfo["EventName"][0]
            AntennaFluenceInfo = Table.read(self.simulation, EventName + "/AntennaFluenceInfo")
            for ant in AntennaFluenceInfo:
                idx = ant['ID']
                self.antenna[idx].fluence = ant["Fluence_efield"]

                fluence_site = CartesianRepresentation(
                    ant["Fluencex_efield"], ant["Fluencey_efield"], ant["Fluencez_efield"], unit=u.eV/u.m**2)
                fluence_shower = self.GRANDshower.transform(fluence_site,shower_frame).cartesian.xyz.value

                self.antenna[idx].fluence_geo = -1
                self.antenna[idx].fluence_ce = -1
                self.antenna[idx].fluence_evB = np.abs(fluence_shower[0])
                self.antenna[idx].fluence_evvB = np.abs(fluence_shower[1])
                self.antenna[idx].fluence = np.linalg.norm(fluence_shower)

        if(self.printLevel>0):
            print("* Evaluating the fluences:")
            print("--> 0 % complete;")
        for counter, ant in enumerate(antenna_list):
            #Read traces or voltages
            if (step>0 and (counter+1)%step == 0 and self.printLevel > 0):
                print("-->",int((counter+1)/(10*step)*100),"% complete;")

            if self.simulation_type != "custom":
                self.process_antenna(ant.ID)
            if ant.fluence > 0:
                ant.sigma_f = np.sqrt(ant.fluence)

            if(ant.fluence > self.f_thres):
                r_plane =ant.r_proj[0:2]
                cosPhi = np.dot(r_plane,np.array([1,0]))/np.linalg.norm(r_plane)
                sinPhi = np.sqrt(1-cosPhi*cosPhi)
                
                if sinPhi >=0.2: # Exclude stations too close to the v\times B direction 
                    my_fluence_geo = np.sqrt(ant.fluence_evB)-(cosPhi/sinPhi)*np.sqrt(ant.fluence_evvB)
                    ant.fluence_geo = my_fluence_geo*my_fluence_geo
                    ant.fluence_ce = ant.fluence_evvB/(sinPhi*sinPhi)
                else:
                    ant.fluence_geo = -1
                    ant.fluence_ce = -1    

            else:
                ant.fluence_geo = -1
                ant.fluence_ce = -1

        if(self.printLevel>0):      
            print("\n")

    def Eval_par_fluences(self,par):
        """
        Evaluates the fluence par for a give set of parameters.
        Uses bool_EarlyLate to toggle early-late correction.
        
        Parameters
        ----------
        self: modERec.EnergyRec
            A class instance.
        par: array
            The parameters of the :math:`a_{ratio}` parametrization.

        Returns
        -------
        fluence_par: array
            Parametrized fluence array.
        """

        try:
            antenna_list = self.antenna.values()
        except:
            antenna_list = self.antenna
        
        fluence_arr = np.array([ant.fluence for ant in antenna_list])

        fluence_par = {}
        eB = self.shower.eB
        alpha = np.arccos(np.dot(self.shower.ev,eB))
        d_Xmax = np.linalg.norm((self.GRANDshower.core - self.GRANDshower.maximum).xyz.value)
        rho_Xmax = SymFit.rho(d_Xmax,-self.shower.ev)

        for ant in antenna_list:
            if(self.bool_EarlyLate and ant.wEarlyLate is not None):
                weight = ant.wEarlyLate
            else:
                weight = 1
            
            r_plane = ant.r_proj[0:2]*weight
            fluence = ant.fluence_evB/(weight**2)
            phi = np.arccos(np.dot(r_plane,np.array([1,0]))/np.linalg.norm(r_plane))
            dist = np.linalg.norm((ant.r_proj - self.shower.r_Core_proj)[0:2])*weight
            fluence_par[ant.ID] = SymFit.f_par_geo(fluence,phi,alpha,dist,d_Xmax,par,rho_Xmax)

        return fluence_par

    @staticmethod
    def eval_mean_fluences(antpos_fluences):
        r"""
        Evaluates the fluence mean and sigma (stdev) for a given set of simulated fluences
        
        Parameters
        ----------
        antpos_fluences: list
            An input with columns: id, x_proj, y_proj, fluence

        Returns
        -------
        x: array
            Antenna x positions in the shower plane.
        y: array
            Antenna y positions in the shower plane.
        mean fluence: array
            The mean fluence fluence in a given (x,y) position.
        sigma_f: array
            The unncertainty in the fluence (standard deviation).
        """

        ID = antpos_fluences[:,0]
        x_proj = antpos_fluences[:,1]
        y_proj = antpos_fluences[:,2]
        fluence_arr = antpos_fluences[:,3]

        fluence_mean = {}
        fluence_mean2 = {}
        counter = {}
        pos = {}
        for i in range(len(fluence_arr)):
            #label = str(x_proj[i]) + str(y_proj[i])
            label = ID[i]

            if label in counter:
                counter[label] = counter[label] + 1
            else:
                counter[label] = 1

            pos[label] = [x_proj[i],y_proj[i]]

            if label in fluence_mean:
                fluence_mean[label] += fluence_arr[i]
                fluence_mean2[label] += fluence_arr[i]**2
            else:
                fluence_mean[label] = fluence_arr[i]
                fluence_mean2[label] = fluence_arr[i]**2

        f_mean = np.zeros(len(pos))
        f_mean2 = np.zeros(len(pos))
        r = np.zeros((len(pos),2))

        trim_counter = {}
        for key, value in counter.items():
            if value < 2:
                pos.pop(key)

        index = 0
        for key, value in pos.items():
            label = key
            f_mean[index] = fluence_mean[label]/counter[label]
            f_mean2[index] = fluence_mean2[label]/counter[label]
            r[index] = value
            index += 1

        return r[:,0], r[:,1], f_mean, np.sqrt(f_mean2-f_mean**2)

    def plot_antpos(self):
        """
        Plots the fluence and antenna positions in the site plane.

        Parameters
        ----------
        self: modERec.EnergyRec
            A class instance.

        """
 
        n_ant = len(self.GRANDshower.fields)

        try:
            antenna_list = self.antenna.values()
        except:
            antenna_list = self.antenna

        r_ant = np.zeros((n_ant,3))
        for idx, (key, value) in enumerate(self.GRANDshower.fields.items()):
            r_ant[idx]=value.electric.r.xyz.value
    
        fluence_arr = np.array([ant.fluence for ant in antenna_list])
        sel = np.where(fluence_arr>0)
    
        fig= plt.figure(figsize=(10,7))
        ax = plt.gca()

        plt.scatter(r_ant[:,0][sel], r_ant[:,1][sel], c=fluence_arr[sel], cmap='viridis')

        plt.xlabel("x (in m)")
        plt.ylabel("y (in m)")
        plt.colorbar().ax.set_ylabel(r"Energy fluence (eV/m$^2$)")
        plt.show()
    
    def signal_output(self):
        """
        Prints the antena positions (in the shower plane) and fluences to a file.
        
        The filename has the structure ``fluence_ShowerPlane_THETACR.out`` and is open with append option.

        Parameters
        ----------
        self: modERec.EnergyRec
            A class instance.
        """

        try:
            antenna_list = self.antenna.values()
        except:
            antenna_list = self.antenna

        fluence_arr = np.array([ant.fluence for ant in antenna_list])
        sel = np.where(fluence_arr>0)

        signal = np.c_[self.shower.r_proj[sel],fluence_arr[sel]]
    
        fluence_file=open('fluence_ShowerPlane_'+str(round(self.shower.thetaCR))+'.out', 'a')
        for entry in signal:
            print(str(entry)[1:-1],file=fluence_file)
        fluence_file.close()

    def shower_projection(self):
        """
        Projects the antenna positions and traces into the shower plane.

        Parameters
        ----------
        self: modERec.EnergyRec
            A class instance.

        Notes
        -----
        Fills self.shower.core_proj, self.antenna.r_proj for all antennas and self.shower.traces_proj.
        
        """
        #The antenna projection
        n_ant = len(self.GRANDshower.fields)

        shower_frame = self.GRANDshower.shower_frame()
        traces_proj = {}

        for key, value in self.GRANDshower.fields.items():
            r_ant = value.electric.r - self.GRANDshower.core
            #if self.simulation_type == "starshape":
            #    self.antenna[key].r_proj = r_ant.xyz.value
            #else:    
            self.antenna[key].r_proj = self.GRANDshower.transform(r_ant,shower_frame).cartesian.xyz.value

            if self.simulation_type != "custom":
                E = self.GRANDshower.fields[key].electric.E
                traces_proj[key] = self.GRANDshower.transform(E,shower_frame)
            else:
                traces_proj[key] = None

        core = self.GRANDshower.core - self.GRANDshower.core
        r_Core_proj = self.GRANDshower.transform(core,shower_frame).cartesian.xyz

        self.shower.r_Core_proj = r_Core_proj.value
        self.shower.traces_proj = traces_proj

    def model_fit(self,filename = "",Cs = None):
        """
        Performs the fit using a given model (set in the EnergyRec instance).
        
        **If** ``filename = ""`` (default) fits a given simulation.
        **else** it reads the file with antenna positions (in shower plane) and fluences,
        and performs the fit.

        Parameters
        ----------
        self: modERec.EnergyRec
            A class instance.
        filename:
            File with antenna positions and fluences for a given shower inclination (default = "");
        Cs: array
            The LDF parameters to be used in the fit if ``filename`` != ""

        Notes
        -----
        Fills self.bestfit.
        
        """ 

        try:
            antenna_list = self.antenna.values()
        except:
            antenna_list = self.antenna

        if(self.printLevel or filename!=""):
            print("* Model fit:")
        if(filename==""):
            fluence_arr = np.array([ant.fluence for ant in antenna_list])
            if all(f is None for f in fluence_arr):
                raise ValueError("--> fluence_arr == None. instance.Eval_fluences() has to be run!")
            
            if(self.bool_EarlyLate and self.printLevel):
                print("--> Early-late correction will be applied!")
        
        else:
            if not Path(filename).is_file():
                message = "ERROR: file " + filename + " not found!"
                raise ValueError(message)

            datafile = open(filename,'r')
            antpos_fluences = np.loadtxt(datafile)
            datafile.close()
            #x_proj, y_proj, fluence_arr, sigma_arr = self.eval_mean_fluences(antpos_fluences)

            #self.antenna = [self.Antenna() for ant in range(len(fluence_arr))]
            self.antenna = [Antenna(ant) for ant in range(len(antpos_fluences))]

            for ant in range(len(antpos_fluences)):
                # self.antenna[ant].fluence = fluence_arr[ant]
                # self.antenna[ant].r_proj = [x_proj[ant],y_proj[ant],0]
                # self.antenna[ant].sigma_f = sigma_arr[ant]
                self.antenna[ant].ID = antpos_fluences[ant,0]
                self.antenna[ant].r_proj = [antpos_fluences[ant,1],antpos_fluences[ant,2],0]
                self.antenna[ant].fluence = antpos_fluences[ant,3]
                self.antenna[ant].sigma_f = antpos_fluences[ant,4]
            
            self.shower.r_Core_proj = np.array([0,0,0])


        AERA.aeraFit(self,filename,Cs)
        if(self.printLevel or filename!=""):
            print("--> Done!")
            print("\n")


    def early_late(self):
        """
        Evaluates the early-late correction factor.

        Parameters
        ----------
        self: modERec.EnergyRec
            A class instance.

        Notes
        -----
        Fills self.antenna.wEarlyLate for all the antennas and self.shower.d_Xmax.

        """
        rCore = self.GRANDshower.core.xyz.value
        rXmax = self.GRANDshower.maximum.xyz.value - rCore

        self.shower.d_Xmax = np.linalg.norm(rXmax)
        R_0 = np.linalg.norm(rXmax)

        try:
            antenna_list = self.antenna.values()
        except:
            antenna_list = self.antenna

        for ant in antenna_list:
            r_ant = ant.r_proj - self.shower.r_Core_proj
            R = R_0 + r_ant[2] ## R_ant[2] is the distance from the core projected into ev
            ant.wEarlyLate = R_0/R        




class AERA:
    """
    A class with aera specific methods.
    """
    @staticmethod
    def aeraLDF(par_fit, Cs, evB, x,y):
        r"""
        The AERA 2d LDF.

        .. math::
            f(\vec{r})=A\left[\exp\left(\frac{-(\vec{r}+C_{1}\vec{e}_{\vec{v}\times\vec{B}}-\vec{r}_{{\rm core}})^{2}}{\sigma^{2}}\right) -C_{0}\exp\left(\frac{-(\vec{r}+C_{2}\vec{e}_{\vec{v}\times\vec{B}}-\vec{r}_{{\rm core}})^{2}}{\left(C_{3}e^{C_{4}\sigma}\right)^{2}}\right)\right].

        Parameters
        ----------
        par_fit: array
            The parameters to be obtimized.
        Cs: array
            The parameters to be fixed.
        evB:
            The versor in the direction of :math:`\vec{v} \times \vec{B}`
        x: array
            The position of the antenna along the x axis :math:`(\vec{v} \times \vec{B})`.
        y: array
            The position of the antenna along the y axis :math:`(\vec{v}\times\vec{v} \times \vec{B}\)`.

        Returns
        -------
        :math:`f`: double
            The function value.

        """

        A = par_fit[0]
        sigma = par_fit[1]
        rcore = Shower.r_Core_proj[0:2]

        if Cs is None:
            C0 = par_fit[2]
            C1 = par_fit[3]
            C2 = par_fit[4]
            C3 = par_fit[5]
            C4 = par_fit[6]
        elif len(par_fit)==6:
            C0 = Cs[0]
            C1 = par_fit[2]
            C2 = par_fit[3]
            C3 = par_fit[4]
            C4 = par_fit[5]
        else:
            C0 = Cs[0]
            C1 = Cs[1]
            C2 = Cs[2]
            C3 = Cs[3]
            C4 = Cs[4]
        
        r = np.array([x,y])
        numA = r-C1*evB-rcore
        partA = np.exp(-(np.linalg.norm(numA)**2)/sigma**2)
        
        numB = r-C2*evB-rcore
        partB = np.exp(-(np.linalg.norm(numB)**2)/(C3*np.exp(C4*sigma))**2)
        
        result = A*(partA - C0*partB)
        if(A<0 or sigma<0 or C0<0 or C3<0 or C4<0):
            result = np.inf
        if result < 0:
            result = 0.
        return result
    
    @staticmethod
    def aeraChi2(par_fit,Cs,self):
        """
        The chi2 for the AERA fit.

        The model for the uncertainty in the fluence is :math:`\sqrt{fluence}`.
        self.bool_EarlyLate toggles the early-late correction;

        Parameters
        ----------
        par: array
            The parameters to be obtimized.
        Cs: array
            The parameters to be fixed.


        Returns
        -------
        Chi2: double
            The :math:`\chi^2` value.

        
        """
        Chi2 = 0.
        i=0
        
        # if len(par_fit)==7:
        #     sigma = par_fit[1]
        #     C0 = par_fit[2]
        #     C3 = par_fit[5]
        #     C4 = par_fit[6]

        #     if(sigma**2 - C0*(C3**2)*np.exp(2*C4*sigma) < 0):
        #         return np.inf
        # elif len(par_fit)==6:
        #     sigma = par_fit[1]
        #     C0 = Cs[0]
        #     C3 = par_fit[4]
        #     C4 = par_fit[5]

        #     if(sigma**2 - C0*(C3**2)*np.exp(2*C4*sigma) < 0):
        #         return np.inf

        try:
            antenna_list = self.antenna.values()
        except:
            antenna_list = self.antenna

        Shower.r_Core_proj = self.shower.r_Core_proj ## Using class definition as a global variable!!

        # check_dist_ldf = np.linspace(0,2*sigma,100)
        # check_ldf = []
        # for check_dist in check_dist_ldf:
        #     check_ldf.append(AERA.aeraLDF(par_fit,Cs, np.array([1,0]), check_dist, 0))

        # if np.sum(check_ldf[0:30]) > np.sum(check_ldf[30:None]): # try to prevent maximum at (0,0)
        #     return np.inf

        max_ldf = -np.inf
        for ant in antenna_list:
            if ant.fluence <= self.f_thres:
                continue
            elif ant.sigma_f == 0:
                continue

            if(self.bool_EarlyLate and ant.wEarlyLate is not None):
                weight = ant.wEarlyLate
            else:
                weight = 1

            x = ant.r_proj[0]*weight
            y = ant.r_proj[1]*weight
            f = ant.fluence/(weight**2)
            sigma = ant.sigma_f/(weight**2)
            ldf_val = AERA.aeraLDF(par_fit,Cs, np.array([1,0]), x, y)
            if ldf_val > max_ldf:
                max_ldf = ldf_val
                max_r_proj = ant.r_proj
            Chi2 = Chi2 + ((ldf_val-f)/sigma)**2
            i = i + 1

        return Chi2

    @staticmethod
    def aeraFit(self,filename,Cs):
        """
        Performs the fit using the AERA 2d LDF.
        
        **If** ``filename = ""`` (default) fits a given simulation.
        **else** it reads the file (antenna position (in shower plane) and fluences)
        and performs the fit. It is used for the training stage.
        
        Parameters
        ----------
        self: modERec.EnergyRec
            A class instance.
        filename: str
            File with antenna positions and fluences for a given shower inclination.
        Cs: array
            The parameters to be fixed.

        Notes
        -----
        Fills self.bestfit.

        """ 
        if(filename==""):
            bestfit_out = "bestfit.out"
        
        else:
            bestfit_out = "bestfit_All.out"

        my_evB = np.array([1,0])
    
        try:
            antenna_list = self.antenna.values()
        except:
            antenna_list = self.antenna

        # amplitude guess
        fluence_arr = np.array([ant.fluence for ant in antenna_list])
        sel = np.where(fluence_arr > np.mean(fluence_arr) + np.std(fluence_arr))
        init_A = 2*np.mean(fluence_arr[sel])
    
        # core position guess
        #core_index = np.where(fluence_arr==np.max(fluence_arr))[0][0]
        #init_xCore =  0 #antpos_proj[core_index,0]
        #init_yCore =  0 #antpos_proj[core_index,1]
    
        # sigma guess
        distances = np.array([np.linalg.norm(ant.r_proj[0:2]) for ant in antenna_list])
        
        init_sigma = np.mean(distances[sel])        
        
        #Cs_aera = [0.5,-10,20,16,0.01]
        Cs_aera = [0.9, 0, 0, 6, 0.003]
        if Cs is None:
            par_fit = [init_A,init_sigma,Cs_aera[0],Cs_aera[1],Cs_aera[2],Cs_aera[3],Cs_aera[4]]
            res = sp.optimize.minimize(AERA.aeraChi2,par_fit,args=(Cs,self),method='Nelder-Mead')
            resx = res.x
        elif len(Cs) == 1:
            par_fit = [init_A,init_sigma,Cs_aera[1],Cs_aera[2],Cs_aera[3],Cs_aera[4]]
            res = sp.optimize.minimize(AERA.aeraChi2,par_fit,args=(Cs,self),method='Nelder-Mead')
            resx = np.insert(res.x,2,Cs[0])
            Cs = None
        else:
            par_fit = [init_A,init_sigma]
            res = sp.optimize.minimize(AERA.aeraChi2,par_fit,args=(Cs,self),method='Nelder-Mead')
            resx = np.append(res.x,Cs)
            Cs_aera = Cs

        chi2min = AERA.aeraChi2(resx,Cs,self)
        ndof = fluence_arr[fluence_arr>self.f_thres].size - res.x.size

        if(self.printLevel > 0):
            print("** AERA fit:")
            print("---> ","{:6} {:>10} {:>10}".format("Par","Initial","Bestfit"))
            print("---> ","----------------------------")
            print("---> ","{:6} {:10} {:10}".format("A",round(init_A,3),round(resx[0],4)))
            print("---> ","{:6} {:10} {:10}".format("sigma",round(init_sigma,2),round(resx[1],4)))
            print("---> ","{:6} {:10} {:10}".format("C0",round(Cs_aera[0],2),round(resx[2],4)))
            print("---> ","{:6} {:10} {:10}".format("C1",round(Cs_aera[1],2),round(resx[3],4)))
            print("---> ","{:6} {:10} {:10}".format("C2",round(Cs_aera[2],2),round(resx[4],4)))
            print("---> ","{:6} {:10} {:10}".format("C3",round(Cs_aera[3],2),round(resx[5],4)))
            print("---> ","{:6} {:10} {:10}".format("C4",round(Cs_aera[4],2),round(resx[6],4)))
            print("---> ","----------------------------")
            print("---> ","Chi2min/n.d.o.f = ",str(round(chi2min,2))," / ",int(ndof))
    
        bestfit=open(bestfit_out, 'a')

        if(filename==""):
            A=resx[0]
            sigma=resx[1]
            sin2Alpha = 1-np.dot(self.shower.ev,self.shower.eB)**2.
            Cs = np.array([resx[2],resx[3],resx[4],resx[5],resx[6]])
            Sradio = (A*np.pi/sin2Alpha)*(sigma**2. - Cs[0]*(Cs[3]**2.)*np.exp(2*Cs[4]*sigma))
            print(*resx,self.GRANDshower.energy.value,Sradio,file=bestfit)
            
        else:
            print(*resx,file=bestfit)
        
        bestfit.close()
        
        ## \cond
        self.bestfit = resx
        ## \endcond

    def aeraPlot(self):
        """
        Plots the result of the aera Fit.

        It plots the 2D LDF with the stations used in the fit.
        It also plots a 1D LDF and a residual plot.

        """

        my_evB = np.array([1,0])
        
        try:
            antenna_list = self.antenna.values()
        except:
            antenna_list = self.antenna

        A = self.bestfit[0]
        sigma = self.bestfit[1]
        rcore = self.shower.r_Core_proj[0:2]
        #sin2Alpha = (1-np.dot(self.shower.ev,self.shower.eB)**2.)
    
        Cs = np.array([self.bestfit[2],self.bestfit[3],self.bestfit[4],self.bestfit[5],self.bestfit[6]])
        #Sradio = (A*np.pi/sin2Alpha)*(sigma**2. - Cs[0]*(Cs[3]**2.)*np.exp(2*Cs[4]*sigma))
        #print('S_radio=',round(Sradio,2))

        par = [A,sigma,Cs[0],Cs[1],Cs[2],Cs[3],Cs[4]]

        fluence_arr = np.array([ant.fluence for ant in antenna_list])
        if(self.bool_EarlyLate):
            weight =  np.array([ant.wEarlyLate for ant in antenna_list])
            if all(w is None for w in weight):
                weight = np.full(len(fluence_arr),1)
        else:
            weight = np.full(len(fluence_arr),1)
        sel = np.where(fluence_arr>0)
        weight = weight[sel]
        fluence_arr=fluence_arr[sel]/(weight**2)

        r_proj = np.array([ant.r_proj for ant in antenna_list])
        x_proj = r_proj[:,0][sel]*weight - rcore[0]
        y_proj = r_proj[:,1][sel]*weight - rcore[1]
        
        delta_X = np.max(x_proj) - np.min(x_proj)
        delta_Y = np.max(y_proj) - np.min(y_proj)
        mean_X = np.min(x_proj) + delta_X/2
        mean_Y = np.min(y_proj) + delta_Y/2

        delta_XY = np.max([delta_X,delta_Y])*1.10

        minXAxis = mean_X-delta_XY/2
        maxXAxis = mean_X+delta_XY/2
        minYAxis = mean_Y-delta_XY/2
        maxYAxis = mean_Y+delta_XY/2

        xx = np.arange(minXAxis,maxXAxis,delta_XY/500)
        yy = np.arange(minYAxis,maxYAxis,delta_XY/500)
        X,Y = np.meshgrid(xx, yy) # grid of point


        Z=np.zeros((yy.size,xx.size))

        Shower.r_Core_proj = np.array([0, 0, 0]) # warning: global variable

        for i in range(yy.size):
            for j in range(xx.size):
                Z[i,j] = AERA.aeraLDF(par, None, my_evB, X[i,j], Y[i,j]) # evaluation of the function on the grid
            
        fig = plt.figure(figsize=[14,5])
        plt.subplot(121)
        im = plt.imshow(Z,cmap='viridis',origin = 'lower', extent=[minXAxis,maxXAxis,minYAxis,maxYAxis]) # drawing the function

        plt.scatter(x_proj, y_proj, c=fluence_arr, cmap='viridis', s = 100, edgecolors=(1,1,1,0.2))
        plt.clim(np.min([np.min(Z),np.min(fluence_arr)]), np.max([np.max(Z),np.max(fluence_arr)]))
        plt.colorbar().ax.set_ylabel(r"energy fluence in eV/m$^2$")

        plt.xlabel(r'distante along $\vec{v}\times\vec{B}$ (in m)')
        plt.ylabel(r'distante along $\vec{v}\times\vec{v}\times\vec{B}$ (in m)')

        plt.plot(0, 0,'w*')

        plt.xlim(minXAxis,maxXAxis)
        plt.ylim(minYAxis,maxYAxis)

        plt.subplot(122)
        plt.scatter(x_proj, y_proj, c=fluence_arr, cmap='viridis', s = 100, edgecolors=(1,1,1,0.2))
        plt.xlabel(r'distante along $\vec{v}\times\vec{B}$ (in m)')
        plt.ylabel(r'distante along $\vec{v}\times\vec{v}\times\vec{B}$ (in m)')
        plt.colorbar().ax.set_ylabel(r"energy fluence in eV/m$^2$")
        plt.xlim(minXAxis,maxXAxis)
        plt.ylim(minYAxis,maxYAxis)
        
        # 1D LDF
        temp_x = x_proj
        temp_y = y_proj

        temp_dist = np.sqrt(temp_x*temp_x+temp_y*temp_y)
        
        fig_ldf = plt.figure(figsize=[14,5])
        plt.subplot(121)
        yerr = np.array([ant.sigma_f for ant in antenna_list])[sel]
        plt.errorbar(temp_dist,fluence_arr,yerr=yerr,fmt='.')
        plt.xlabel("Distance from core in m")
        plt.ylabel(r"Fluence in eV/m$^2$")
        plt.gca().set_yscale('log')
            
        residual = np.zeros(fluence_arr.size)
        
        for i in range(fluence_arr.size):
            residual[i] = (fluence_arr[i] - AERA.aeraLDF(par, None,my_evB, x_proj[i], y_proj[i]))/np.sqrt(fluence_arr[i])


        plt.subplot(122)
        plt.errorbar(temp_dist,residual,yerr=1,fmt='.')
        plt.xlabel("Distance from core in m")
        plt.ylabel(r"($f$ - fit)/$\sigma_f$")    
        #plt.xlim(0,500)
        plt.ylim(-2,2)
        plt.grid()

class SymFit:
    """
    A class with the symmetric signal distribution specific methods.

    """
    ## A parameter for the a_ratio.
    a_par0 = 0.373
    ## A parameter for the a_ratio.
    a_par1 = 762.6
    ## A parameter for the a_ratio.
    a_par2 = 0.149
    ## A parameter for the a_ratio.
    a_par3 = 0.189

    @staticmethod
    def a_ratio(r, d_Xmax, par, rho_max):
        """
        Evaluates the charge-excess to geomagnetic ratio.
        
        Parameters
        ----------
        r:
            Antenna position in the shower plane;
        d_Xmax:
            Distance from core to shower maximum in meters;
        par:
            The parameters;
        rho_max:
            Air density at shower maximum.

        Returns
        -------
        a_ratio: double
            The charge-excess to geomagnetic ratio.
        """ 
        rho_mean = 0.327
        return par[0]*(r/d_Xmax)*np.exp(r/par[1])*(np.exp((rho_max-rho_mean)/par[2])-par[3])

    @staticmethod
    def f_par_geo(f_vB, phi, alpha, r, d_Xmax, par, rho_max):
        """
        Evaluates the parametrized geomagnetic fluence.
        
        Parameters
        ----------
        f_vB:
            Fluence in the v times B direction;
        phi:
            The angle between the antenna position and the v times B direciton;
        alpha:
            The geomagnetic angle.
        r:
            Antenna position in the shower plane;
        d_Xmax:
            Distance from core to shower maximum in meters;
        par:
            The parameters;
        rho_max:
            Air density at shower maximum.

        Returns
        -------
        f_par_geo: double
            The parametrized geomagnetic fluence.
        """
        sqrta = np.sqrt(SymFit.a_ratio(r, d_Xmax, par, rho_max))
        cos_sin_ratio = np.cos(phi)/np.abs(np.sin(alpha))
        return f_vB/((1+cos_sin_ratio*sqrta)**2)
    
    @staticmethod
    def rho(r, e_vec):
        """
        Evaluates the air density at a given position.
        
        Parameters
        ----------
        r:
            The distance to the position in meters;
        e_vec:
            The direction of the position (unitary vector);

        Returns
        -------
        rho: double
            The air density.
        """

        height = np.dot(r*e_vec,np.array([0,0,1]))/1000 # from m to km

        H = 10.4 # in km
        rho_0 = 1.225 # in kg/m^3
        site_height = EnergyRec.site_height/1000 # in km (Hard Coded Using Class Definition!!!)

        return rho_0*np.exp(-(site_height+height)/H)

    
    @staticmethod
    def a_ratio_chi2(par,fluence_geo, fluence_ce,alpha, r, d_Xmax, rho_Xmax):
        """
        Chi2 for the a_ratio fit.
        
        Parameters
        ----------
        par:
            The parameters;
        fluence_geo:
            An array with the geomagnetic fluences;
        fluence_ce:
            An array with the charge excess fluences;
        alpha:
            An array with the geomagnetic angles;
        r:
            An array with the antenna distances to the core in the shower plane;
        d_Xmax:
            The distance from the core to the Xmax;
        rho_Xmax:
            The atmospheric density in the Xmax.

        Returns
        -------
        Chi2: double
            The :math:`\chi^2` value.
        """
        sel = np.where(fluence_geo > 0)

        a_arr = (np.sin(alpha[sel])**2)*fluence_ce[sel]/fluence_geo[sel]

        Chi2 = 0
        for i in range(fluence_geo[sel].size):
            a_theo = SymFit.a_ratio(r[sel][i], d_Xmax[sel][i], par, rho_Xmax[sel][i])
            #if a_arr[i] < 1:
            Chi2 = Chi2 + (a_arr[i] -a_theo)**2

        return Chi2

    @staticmethod
    def a_ratio_fit(fluence_geo, fluence_ce,alpha, r, d_Xmax, rho_max):
        """
        Fits the a_ratio.
        
        Parameters
        ----------
        fluence_geo:
            An array with the geomagnetic fluences;
        fluence_ce:
            An array with the charge excess fluences;
        alpha:
            An array with the geomagnetic angles;
        r:
            An array with the antenna distances to the core in the shower plane;
        d_Xmax:
            An array with the distances to shower maximum;
        rho_max:
            An array with the densities at shower maximum.

        Returns
        -------
        bestfit: array
            The bestfit parameters array.
        """
        par = [0.373, 762.6, 0.1490, 0.189]
        res = sp.optimize.minimize(SymFit.a_ratio_chi2,par,args=(fluence_geo, fluence_ce,alpha, r, d_Xmax, rho_max),method='Nelder-Mead')
        return res.x
    
    @staticmethod
    def SymLDF(par,r):
        r"""
        The symmetric ldf to be fit to the fluence_par data.

        .. math::
            f_{ABCD}(r) = A.exp\left[-B.r-C.r^2-D.r^3\right]
        
        Parameters
        ----------
        par:
            The parameter array;
        r:
            The distance to the axis.

        Returns
        -------
        LDF: double
            The ldf value at distance r.
        """
        A = par[0]
        B = par[1]
        C = par[2]
        D = par[3]

        LDF = A*np.exp(-B*r-C*r**2-D*r**3)
        return LDF
    
    @staticmethod
    def LDF_chi2(par,r,fluence_par):
        """
        The LDF chi2.
        
        Parameters
        ----------
        par:
            The parameter array;
        r:
            The distance to the axis;
        fluence_par:
            The array with the symmetrized signal.

        Returns
        -------
        Chi2: double
            The :math:`\chi^2` value.
        """
        sel = np.where(fluence_par > 0)
        f_par = fluence_par[sel]
        Chi2 = 0
        for i in range(f_par.size):
            LDF = SymFit.SymLDF(par,r[sel][i])
            #if a_arr[i] < 1:
            Chi2 = Chi2 + ((f_par[i] -LDF)/np.sqrt(f_par[i]))**2

        return Chi2


    @staticmethod
    def SymLDF_fit(r,fluence_par):
        """
        Fits the symmetric LDF to the fluence_par data.
        
        Parameters
        ----------
        r:
            The distance to the axis;
        fluence_par:
            The array with the symmetrized signal.

        Returns
        -------
        bestfit: array
            The bestfit parameters array.
        """

        # Estimating the parameters
        idx   = np.argsort(r)
        sort_dist = np.array(r)[idx]
        sort_f = np.array(fluence_par)[idx]

        n_ant = len(sort_dist)
        r0 = sort_dist[0]
        f0 = sort_f[0]

        idx_max = np.where(sort_f == np.max(sort_f))[0][0]
        r1 = sort_dist[idx_max]
        f1 = sort_f[idx_max]
        #r2 = sort_dist[(idx_max+ n_ant)//2] // Did not work for ldfs 'clustered' (ex. three well sampled distances)
        #f2 = sort_f[(idx_max+ n_ant)//2]
        r3 = sort_dist[-1]
        f3 = sort_f[-1]
        r2 = (r1+r3)/2
        f2 = (f1+f3)/2

        a = np.array([
            [1,-r0,-r0**2,-r0**3],
            [1,-r1,-r1**2,-r1**3],
            [1,-r2,-r2**2,-r2**3],
            [1,-r3,-r3**2,-r3**3]])
        b = np.array([np.log(f0),np.log(f1),np.log(f2),np.log(f3)])

        par = np.linalg.solve(a, b)

        par[0] = np.exp(par[0])
        res = sp.optimize.minimize(SymFit.LDF_chi2,par,args=(r, fluence_par),method='Nelder-Mead')
        return res.x

    @staticmethod
    def Sradio_geo(par,ldf_par,alpha,rho_Xmax):
        """
        The radiation energy corrected for the scaling of the emission strength with the geomagnetic angle and the atmospheric density.
        
        Parameters
        ----------
        par:
            The free parameters of the correction;
        ldf_par:
            The parameters to be used in the symmetric LDF;
        alpha:
            The geomagnetic angle;
        rho_Xmax:
            The density in the X_max.

        Returns
        -------
        S_radio_geo: double
            The corrected radiation energy
        """
        E_rad = 2*np.pi*sp.integrate.quad(lambda r: r*SymFit.SymLDF(ldf_par,r), 0, 2000)[0]
        sin2alpha = np.sin(alpha)**2.

        p0 = par[0]
        p1 = par[1]
        #rho_mean = 0.648
        rho_mean = 0.327
        den = sin2alpha*(1 - p0 + p0*np.exp(p1*(rho_Xmax-rho_mean)))

        return E_rad*1.e-9/den # in GeV

    @staticmethod
    def Sradio_mod(par,E):
        """
        The model for the relation between S_radio and the energy.
        
        Parameters
        ----------
        par:    
            The free parameters of the model;
        E:
            The energy of the event in EeV.

        Returns
        ------
        S_radio_mod: double
            The model :math:`S_{radio}`.
        """

        S_19 = par[0]
        gamma = par[1]

        return S_19*(E/10)**gamma

    @staticmethod
    def Chi2_joint_S(par,ldf_par_arr,alpha_arr,rho_Xmax_arr,E_arr):
        """
        The chi2 for the joint fit os Sradio_geo and Sradio_mod.
        
        Parameters
        ----------
        par:
            The full parameter array;
        ldf_par_arr:
            The array with the ldf_par for each simulation;
        alpha_arr:
            The array with the geomagnetic angles of each simulation;
        rho_Xmax:
            The array with the density at Xmax of each simulation;
        E_arr:
            The array with the energies of each simulation in GeV;

        Returns
        -------
        Chi2: double
            The :math:`\chi^2` value.
        """
        
        Chi2 = 0 

        for i in range(len(ldf_par_arr)):
            S_geo = SymFit.Sradio_geo(par[0:2],ldf_par_arr[i],alpha_arr[i],rho_Xmax_arr[i])
            S_mod = SymFit.Sradio_mod(par[2:4],E_arr[i])

            Chi2 = Chi2 + ((S_geo - S_mod)/np.sqrt(S_geo))**2
        
        return Chi2

    @staticmethod
    def joint_S_fit(ldf_par_arr,alpha_arr,rho_Xmax_arr,E_arr):
        """
        Performs the joint fit of the S_radio.
        
        Parameters
        ----------
        ldf_par_arr:
            The array with the ldf_par for each simulation;
        alpha_arr:
            The array with the geomagnetic angles of each simulation;
        rho_Xmax:
            The array with the density at Xmax of each simulation;
        E_arr:
            The array with the energies of each simulation in GeV;

        Returns
        ------
        bestfit: array
            The bestfit array.
        """

        p0 = 0.394
        p1 = -2.370 #m^3/kg
        S_19 = 1.408 #in GeV
        gamma = 1.995
        par = [p0,p1,S_19,gamma]

        res = sp.optimize.minimize(SymFit.Chi2_joint_S,par,args=(ldf_par_arr,alpha_arr,rho_Xmax_arr,E_arr),method='Nelder-Mead')
        return res.x



print("* EnergyRec default values summary:")
print("--> bool_plot = ",EnergyRec.bool_plot)
print("--> bool_EarlyLate = ",EnergyRec.bool_EarlyLate)
print("--> nu_low = ",EnergyRec.nu_low)
print("--> nu_high = ",EnergyRec.nu_high)
print("--> SNR_thres = ",EnergyRec.SNR_thres)
print("--> thres_low = ",EnergyRec.thres_low)
print("--> thres_high = ",EnergyRec.thres_high)
print("--> f_thres = ",EnergyRec.f_thres)
print("--> printLevel = ",EnergyRec.printLevel)
print("\n")