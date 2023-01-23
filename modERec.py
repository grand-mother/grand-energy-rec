##
# @mainpage
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

import ctypes
import glob
import math
import re
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import astropy.units as u
# for custom_from_datafile
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from grand.tools.coordinates import (CartesianRepresentation,
                                 SphericalRepresentation)
from astropy.table import Table
from grand import ECEF, LTP, Geodetic, Rotation
from grand.simulation import ElectricField, ShowerEvent, ZhairesShower
from grand.simulation.pdg import ParticleCode
from grand.simulation.shower.generic import CollectionEntry, FieldsCollection
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from scipy.fftpack import fft, ifft
from scipy.signal import hilbert

import ROOT
event_lib = "../../MachineLearning/EnergyRec/Event/libSimuEvent.so"

try:
    ROOT.gSystem.Load(event_lib)
except:
    pass

class Input:
    """
    A class for the input attributes.

    Attributes
    ----------
    r_ant_arr:
        The antenna positions
    traces_x:
        The traces in x direction
    traces_y:
        The traces in y direction
    traces_z:
        The traces in z direction
    ev:
        Unitary vector in the velocity direction
    eB:
        Unitary vector on the magnetic field direction
    r_core:
        Shower core coordinates
    r_x_max:
        X max position w.r.t. the shower core
    energy:
        Monte Carlo energy
    """

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
    r_ground:
        The position of the antenna in the ground
    r_proj:
        The position of the antenna in the shower plane
    trace_x:
        Electric field trace on the x direction
    trace_y:
        Electric field trace on the y direction
    trace_z:
        Electric field trace on the z direction
    trace_evB:
        Electric field trace on the evB direction
    trace_evvB:
        Electric field trace on the evvB direction
    trace_ev:
        Electric field trace on the ev direction
    time:
        Time array for the traces    

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
    ## Electric field trace on the x direction
    trace_x = None
    ## Electric field trace on the y direction    
    trace_y = None
    ## Electric field trace on the z direction
    trace_z = None
    ## Electric field trace on the evB direction
    trace_evB = None
    ## Electric field trace on the evvB direction    
    trace_evvB = None
    ## Electric field trace on the ev direction
    trace_ev = None
    ## Time array for the traces
    time = None

    def __init__(self, ID):
        """
        The default init function for the class Antenna.

        """
        self.ID = ID

    @staticmethod
    def fft_filter(time_arr, trace, nu_low=50, nu_high=200):
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
        # Number of sample points
        N = time_arr.size

        # sample spacing
        sampling_rate = 1 / (
            (time_arr[1] - time_arr[0]) * 1.0e-9
        )  # uniform sampling rate (convert time from ns to seconds)
        T = 1 / sampling_rate
        yf = fft(trace)
        xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

        nu_low = nu_low * 10 ** 6  # from MHz to Hz
        nu_high = nu_high * 10 ** 6  # from MHz to Hz
        for i in range(xf.size):
            if xf[i] < nu_low or xf[i] > nu_high:
                yf[i] = 0
                yf[
                    -i
                ] = 0  # negative frequencies are backordered (0 1 2 3 4 -4 -3 -2 -1)

        return yf

    @staticmethod
    def trace_recover(time_arr, trace_fft):
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
        yy = ifft(trace_fft).real

        xx = time_arr - np.min(time_arr)

        return yy

    @staticmethod
    def hilbert_envelope(time_arr, trace_rec):
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
        hilbert_env = hilbert(trace_rec)

        return hilbert_env

    @staticmethod
    def compute_fluence(time_arr, trace, SNR_thres=10):
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
        Fills self.fluence, self.fluence_geo, self.fluence_ce, self.fluence_evB and self.fluence_evvB
        """

        trace_fft = Antenna.fft_filter(time_arr, trace)
        trace_rec = Antenna.trace_recover(time_arr, trace_fft)

        # Check if peak is within the threshold range after offset, cut and trace recover.
        hilbert_env = Antenna.hilbert_envelope(time_arr, trace_rec)

        tt = time_arr - np.min(time_arr)
        delta_tt = (tt[1] - tt[0]) * 1e-9  # convert from ns to seconds

        envelope2 = hilbert_env ** 2.0

        tmean_index = np.where(envelope2 == np.max(envelope2))[0][
            0
        ]  # Index of the hilbert envelope maximum

        if tt[-1] - tt[tmean_index] < 150:
            # print("Peak too close to the end of the signal. Skipping")
            fluence = 0
            return fluence, 0

        time_100ns_index = np.where(tt <= 100)[0][-1]  # Index of 100 ns bin

        if tmean_index < time_100ns_index:
            # tmean_index = time_100ns_index
            # print("Peak too close to the beginning of the signal. Skipping")
            fluence = 0
            return fluence, 0

        t1 = tt[tmean_index - time_100ns_index]
        t2 = tt[tmean_index + time_100ns_index]

        if tmean_index < tt.size // 2:
            # background from the end of the trace
            t3 = tt[-1 - 2 * time_100ns_index]
            t4 = tt[-1]
        else:
            # background from the beginning of the trace
            t3 = tt[0]
            t4 = tt[2 * time_100ns_index]

        signal = (
            np.sum(np.abs(envelope2)[(tt >= t1) & (tt <= t2)]) * delta_tt
        )  # N^2 * Coulomb^-2 * s
        bkg = (
            np.sum(np.abs(envelope2)[(tt >= t3) & (tt < t4)])
            * delta_tt
            * (t2 - t1)
            / (t4 - t3)
        )

        epsilon0 = 8.8541878128e-12  # Coulomb^2 * N^-1 * m^-2
        c = 299792458  # m * s^-1

        Joule_to_eV = 1 / 1.602176565e-19

        SNR = np.sqrt(signal / bkg)

        fluence = epsilon0 * c * (signal - bkg) * Joule_to_eV # eV * m^-2

        return fluence, SNR

    @staticmethod
    def compute_f_geo_ce(fluence_evB, fluence_evvB , r_plane):
        cosPhi = np.dot(r_plane, np.array([1, 0])) / np.linalg.norm(r_plane)
        sinPhi = np.sqrt(1 - cosPhi * cosPhi)

        if (
            sinPhi >= 0.2
        ):  # Exclude stations too close to the v\times B direction
            fluence_geo = np.sqrt(fluence_evB) - (
                cosPhi / sinPhi
            ) * np.sqrt(fluence_evvB)
            fluence_geo = fluence_geo * fluence_geo
            fluence_ce = fluence_evvB / (sinPhi * sinPhi)
        else:
            fluence_geo = 0
            fluence_ce = 0

        return fluence_geo, fluence_ce

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
    energy:
        The cosmic ray energy
    r_Core_proj:
        The position of the core projected into the shower plane.
    bool_plot: bool
        Toggles the plots on and off.
    d_Xmax:
        Distance to Xmax from the simulation.

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
    energy = None
    ## The position of the core projected into the shower plane.
    r_Core_proj = None
    ## Toggles the plots on and off.
    bool_plot = False
    ## Distance to Xmax.
    d_Xmax = None
    ## Projection matrix
    projection_mat = None

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
    ## An instance of the class Input
    shower = None
    ## The bestfit values of the parameters
    bestfit = None
    ## The shower imported using the standard grand package
    GRANDshower = None
    ## The height of the site
    site_height = 0
    ## For PengXiong files with more than one event
    evt_num = 0
    ## A print level variable
    printLevel = 0

    def __init__(self, simulation):
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

        if Path(self.simulation).is_dir() or self.simulation.endswith("hdf5") or self.simulation.endswith("root"):

            if self.simulation_type == "coreas":
                self.init_coreas()
            elif self.simulation_type == "zhaires":
                self.init_zhaires()
                

            elif (
                self.simulation_type == "custom" or self.simulation_type == "starshape"
            ):
                self.init_custom()

            elif (self.simulation_type == "pengxiong"):
                self.init_pengxiong()
            
            elif (self.simulation_type == "felix"):
                self.init_felix()
                return


            if Path(self.simulation).is_dir():
                self.GRANDshower.localize(latitude=45.5, longitude=90.5)

            
            ev = self.shower.r_core - self.shower.r_x_max
            ev /= ev.norm()
            ev = ev.T[0]
            self.shower.ev = ev

            evB = np.cross(ev, self.shower.eB)
            evB /= np.linalg.norm(evB)
            self.shower.evB = evB
            evvB = np.cross(ev, evB)
            self.shower.evvB = evvB
            eB = self.shower.eB
            eB /= np.linalg.norm(eB)
            self.shower.eB = eB

            self.shower.projection_mat = np.linalg.inv(np.array([
                evB, evvB, ev
                ]).T)

            self.shower_projection()

            self.Eval_fluences()
            # self.plot_antpos()

            d_Xmax = np.linalg.norm(
               (self.shower.r_core - self.shower.r_x_max)
            )
            try:
                antenna_list = self.antenna.values()
            except:
                antenna_list = self.antenna
            for ant in antenna_list:
                ant.wEarlyLate = self.early_late(ant.r_proj, self.shower.r_Core_proj, d_Xmax)

            if self.printLevel > 0:
                print("* EnergyRec instance starting values summary:")
                print("--> bool_plot = ", self.bool_plot)
                print("--> bool_EarlyLate = ", self.bool_EarlyLate)
                print("--> nu_low = ", self.nu_low)
                print("--> nu_high = ", self.nu_high)
                print("--> SNR_thres = ", self.SNR_thres)
                print("--> thres_low = ", self.thres_low)
                print("--> thres_high = ", self.thres_high)
                print("--> f_thres = ", self.f_thres)
                print("\n")

        elif Path(self.simulation).is_file():
            self.model_fit(self.simulation)

        else:
            message = "ERROR: " + self.simulation + " not found!"
            raise ValueError(message)

    def init_coreas(self):
        self.GRANDshower = ShowerEvent.load(self.simulation)
        self.antenna = {ant: Antenna(ant) for ant in self.GRANDshower.fields.keys()}
        self.fill_from_GRANDshower()
    
    def init_zhaires(self):
        self.GRANDshower = ZhairesShower._from_datafile(self.simulation)

        # Fixing Aires to GRAND conventions
        for ant in range(len(self.GRANDshower.fields)):
            self.GRANDshower.fields[ant].electric.E = self.GRANDshower.fields[
                ant
            ].electric.E[0]

            from astropy.coordinates.matrix_utilities import \
                rotation_matrix

            rotation = rotation_matrix(-90 * u.deg, axis="z")

            self.GRANDshower.fields[ant].electric.E = self.GRANDshower.fields[
                ant
            ].electric.E.transform(rotation)

            self.GRANDshower.fields[ant].electric.r = self.GRANDshower.fields[
                ant
            ].electric.r.transform(rotation)

        self.GRANDshower.localize(latitude=45.5, longitude=90.5)
        self.antenna = {ant: Antenna(ant) for ant in self.GRANDshower.fields.keys()}
        self.fill_from_GRANDshower()

    def init_custom(self):
        self.GRANDshower, bool_traces = self.custom_from_datafile(
                    self.simulation, self.site_height
                )

        if bool_traces:
            # Fixing Aires to GRAND conventions
            for ant in range(len(self.GRANDshower.fields)):
                self.GRANDshower.fields[
                    ant
                ].electric.E = self.GRANDshower.fields[ant].electric.E[0]
                from astropy.coordinates.matrix_utilities import \
                    rotation_matrix

                rotation = rotation_matrix(-0 * u.deg, axis="z")

                self.GRANDshower.fields[
                    ant
                ].electric.E = self.GRANDshower.fields[
                    ant
                ].electric.E.transform(
                    rotation
                )

                self.GRANDshower.fields[
                    ant
                ].electric.r = self.GRANDshower.fields[
                    ant
                ].electric.r.transform(
                    rotation
                )

            if self.simulation_type == "custom":
                self.simulation_type == "zhaires"
        self.antenna = {ant: Antenna(ant) for ant in self.GRANDshower.fields.keys()}
        self.fill_from_GRANDshower()

    def init_pengxiong(self):
        self.GRANDshower, bool_traces = self.custom_from_pengxiong(
                    self.simulation, self.site_height, self.evt_num
                )
        self.antenna = {ant: Antenna(ant) for ant in self.GRANDshower.fields.keys()}
        self.fill_from_GRANDshower()
    
    def init_felix(self):
        self.GRANDshower, bool_traces = self.custom_from_felix(
                    self.simulation
                )
        self.antenna = {ant: Antenna(ant) for ant in self.GRANDshower.fields.keys()}
        with h5py.File(self.simulation, "r") as file:
            antenna_list = self.antenna.values()
            dt = file['highlevel']['obsplane_2900_gp_vB_vvB']
            for ant in antenna_list:
                idx = ant.ID
                self.antenna[idx].fluence = dt['energy_fluence'][idx]

                f_vec = dt['energy_fluence_vector'][idx]
                self.antenna[idx].fluence_geo = -1
                self.antenna[idx].fluence_ce = -1
                self.antenna[idx].fluence_evB = np.abs(f_vec[0])
                self.antenna[idx].fluence_evvB = np.abs(f_vec[1])
                self.antenna[idx].r_proj = dt['antenna_position_vBvvB'][idx][:]

    def fill_from_GRANDshower(self):
        self.shower.r_core = self.GRANDshower.core
        self.shower.r_x_max = self.GRANDshower.maximum
        self.shower.eB = self.GRANDshower.geomagnet.T[0]
        self.shower.energy = self.GRANDshower.energy

        try:
            antenna_list = self.antenna.values()
        except:
            antenna_list = self.antenna

        for ant in antenna_list:
            idx = ant.ID
            self.antenna[idx].trace_x  = self.GRANDshower.fields[idx].electric.E.x
            self.antenna[idx].trace_y  = self.GRANDshower.fields[idx].electric.E.y
            self.antenna[idx].trace_z  = self.GRANDshower.fields[idx].electric.E.z
            self.antenna[idx].time = self.GRANDshower.fields[idx].electric.t
            self.antenna[idx].r_ground = self.GRANDshower.fields[idx].electric.r.T[0]


    

    @staticmethod
    def custom_from_datafile(path: Path, site_height=0) -> ZhairesShower:
        with h5py.File(path, "r") as fd:
            if not "RunInfo.__table_column_meta__" in fd["/"]:
                return super()._from_datafile(path)

            for name in fd["/"].keys():
                if not name.startswith("RunInfo"):
                    break

            bool_traces = True
            event = fd[f"{name}/EventInfo"]
            antennas = fd[f"{name}/AntennaInfo"]
            try:
                traces = fd[f"{name}/AntennaTraces"]
            except:
                bool_traces = False

            fields = FieldsCollection()

            pattern = re.compile("([0-9]+)$")
            for antenna, x, y, z, *_ in antennas:
                r = CartesianRepresentation(float(x), float(y), float(z))
                if bool_traces:
                    tag = antenna.decode()
                    antenna = int(pattern.search(tag)[1])
                    tmp = traces[f"{tag}/efield"][:]
                    efield = tmp.view("f4").reshape(tmp.shape + (-1,))
                    t = np.asarray(efield[:, 0], "f8")
                    Ex = np.asarray(efield[:, 1], "f8")
                    Ey = np.asarray(efield[:, 2], "f8")
                    Ez = np.asarray(efield[:, 3], "f8")
                else:
                    t = None
                    E = None
                electric = ElectricField(
                    t, CartesianRepresentation(x=Ex, y=Ey, z=Ez), r
                    )
                fields[antenna] = CollectionEntry(electric)

            primary = {
                "Fe^56": ParticleCode.IRON,
                "Gamma": ParticleCode.GAMMA,
                "Proton": ParticleCode.PROTON,
            }[event[0, "Primary"].decode()]

            geomagnet = SphericalRepresentation(
                theta=float(90 + event[0, "BFieldIncl"]),
                phi=0,
                r=float(event[0, "BField"])
            )

            try:
                latitude = event[0, "Latitude"].decode("utf-8")
                longitude = event[0, "Longitude"].decode("utf-8")
                declination = event[0, "BFieldDecl"]
                obstime = datetime.strptime(
                    event[0, "Date"].decode("utf-8").strip(), "%d/%b/%Y"
                )
            except ValueError:
                frame = None
            else:
                geodetic = Geodetic(latitude=latitude, longitude=longitude, height=0.0)
                origin = ECEF(geodetic)
                frame = LTP(
                    location=origin,
                    orientation="NWU",
                    declination=declination,
                    obstime=obstime,
                )

            # my_core = event[0, 'CorePosition'] + np.array([0, 0, site_height])
            my_core = np.array([0, 0, site_height])

            return (
                ZhairesShower(
                    energy=float(event[0, "Energy"]),
                    zenith=(180 - float(event[0, "Zenith"])),
                    azimuth=(180 - float(event[0, "Azimuth"])),
                    primary=primary,
                    frame=frame,
                    core=CartesianRepresentation(*my_core),
                    geomagnet=CartesianRepresentation(geomagnet),
                    maximum=CartesianRepresentation(
                        *event[0, "XmaxPosition"]
                    ),
                    fields=fields,
                ),
                bool_traces,
            )


    @staticmethod
    def custom_from_pengxiong(path: Path, site_height=0, evt=0) -> ZhairesShower:
        rootFile = ROOT.TFile(path)

        # getting the trees
        treeSimu = rootFile.Get("SimuCollection");
        treeEfield = rootFile.Get("EfieldCollection");
        treeSignal = rootFile.Get("SignalCollection");

        #setting the object of the classes where Im going to instantiate the trees
        branch_SimShower = ROOT.SimShower()
        branch_SimEfield = ROOT.SimEfield()
        branch_SimSignal = ROOT.SimSignal()

        #setting the adresses of the branches on the created objects of the specific classes
        treeSimu.SetBranchAddress("SimShowerBranch", branch_SimShower)
        treeEfield.SetBranchAddress("SimEfieldBranch", branch_SimEfield)
        treeSignal.SetBranchAddress("SimSignalBranch", branch_SimSignal)

        # read the event
        treeSimu.GetEntry(evt)
        treeEfield.GetEntry(evt)
        treeSignal.GetEntry(evt)

        # TREE SimShower------------------------------------------------------
        shower_energy = branch_SimShower.Get_primary_Energy()
        shower_azimuth = branch_SimShower.shower_azimuth
        shower_zenith = branch_SimShower.shower_zenith
        shower_Bfield = branch_SimShower.magnetic_field
            
        # TREE SimEfield------------------------------------------------------
        positions = branch_SimEfield.Detectors_det_pos_shc
        t0 = branch_SimEfield.Detectors_t_0
        t_bin_size = branch_SimEfield.t_bin_size
        id = branch_SimEfield.Detectors_det_id
        Detectors_trace_Ex = branch_SimEfield.Detectors_trace_x
        Detectors_trace_Ey = branch_SimEfield.Detectors_trace_y
        Detectors_trace_Ez = branch_SimEfield.Detectors_trace_z

        fields = FieldsCollection()
        for ant in range(len(Detectors_trace_Ex)):
            r = CartesianRepresentation(x=float(positions[ant][0]),
                y=float(positions[ant][1]), z=float(positions[ant][2])
                )
            antenna = int(id[ant])
            t = np.asarray(
                    np.linspace(t0[ant], t0[ant] + 
                    t_bin_size * len(Detectors_trace_Ex[ant]),
                    len(Detectors_trace_Ex[ant])),"f8"
            )
            Ex = np.asarray(Detectors_trace_Ex[ant], "f8") * 1.e-6 # muV/m to V/m
            Ey = np.asarray(Detectors_trace_Ey[ant], "f8") * 1.e-6 # muV/m to V/m
            Ez = np.asarray(Detectors_trace_Ez[ant], "f8") * 1.e-6 # muV/m to V/m
            electric = ElectricField(
                t, CartesianRepresentation(x=Ex, y=Ey, z=Ez), r
                )
            fields[antenna] = CollectionEntry(electric)

        primary = {
            "Fe^56": ParticleCode.IRON,
            "22": ParticleCode.GAMMA,
            "2212": ParticleCode.PROTON,
        }[str(*branch_SimShower.shower_type)]

        geomagnet = SphericalRepresentation(
            theta=float(90 + np.arctan2(shower_Bfield[2],
                  shower_Bfield[0])*180/np.pi),
            phi=0,
            r=float(np.linalg.norm(shower_Bfield))
        )

        try:
            latitude = branch_SimShower.site_lat_long[0]
            longitude = branch_SimShower.site_lat_long[1]
            declination = 0
            obstime = datetime.strptime(
                str(*branch_SimShower.date).strip(), "%d/%b/%Y"
            )
        except ValueError:
            frame = None
        else:
            geodetic = Geodetic(latitude=latitude, longitude=longitude, height=0.0)
            origin = ECEF(geodetic)
            frame = LTP(
                location=origin,
                orientation="NWU",
                declination=declination,
                obstime=obstime,
            )

        # my_core = event[0, 'CorePosition'] + np.array([0, 0, site_height])
        my_core = np.array([0, 0, site_height])

        return (
            ZhairesShower(
                energy=float(shower_energy),
                zenith=(180 - float(shower_zenith)),
                azimuth=(180 - float(shower_azimuth)),
                primary=primary,
                frame=frame,
                core=CartesianRepresentation(x=my_core[0], y=my_core[1], z=my_core[2]),
                geomagnet=CartesianRepresentation(geomagnet),
                maximum=CartesianRepresentation(
                    x=branch_SimShower.xmax_pos_shc[0],
                    y=branch_SimShower.xmax_pos_shc[1],
                    z=branch_SimShower.xmax_pos_shc[2]
                ),
                fields=fields,
            ),
            True,
        )

    @staticmethod
    def custom_from_felix(path: Path) -> ZhairesShower:
        bool_traces = False
        with h5py.File(path, "r") as file:
            fields = FieldsCollection()

            n_ant = file['highlevel']['obsplane_2900_gp_vB_vvB']['antenna_names'].shape[0]
            for ant in range(n_ant):
                x = file['highlevel']['obsplane_2900_gp_vB_vvB']["antenna_position"][ant, 0]
                y = file['highlevel']['obsplane_2900_gp_vB_vvB']["antenna_position"][ant, 1]
                z = file['highlevel']['obsplane_2900_gp_vB_vvB']["antenna_position"][ant, 2]
                r = CartesianRepresentation(x=float(x), y=float(y), z=float(z))  # RK
                
                t = None
                E = None
                fields[ant] = CollectionEntry(electric=ElectricField(t=t, E=E, r=r))

            return ZhairesShower(
                energy=1.0e9,  # EeV --> GeV HARD CODED!!! For testing
                fields=fields,
            ), bool_traces

    def simulation_inspect(self):
        """
        Outputs theta, phi, Energy and the core position.

        Parameters
        ----------
        self: modERec.EnergyRec
            A class instance.

        """

        thetaCR = self.shower.thetaCR
        phiCR = self.shower.phiCR
        E = self.shower.energy
        Core = self.shower.r_core
        B = self.GRANDshower.geomagnet

        print("* Simulation summary:")
        print("--> thetaCR = ", thetaCR)
        print("--> phiCR = ", phiCR)
        print("--> E = ", E)

        print("--> Core position = ", Core)

        print("--> Geomagnetic field = ", B)

    def inspect_antenna(self, id):
        """
        Plots the traces for a given antenna.

        Parameters
        ----------
        self: modERec.EnergyRec
            A class instance.
        id: int
            The antenna id.

        """

        if id < len(self.antenna):
            Ex = self.antenna[id].trace_x
            Ey = self.antenna[id].trace_y
            Ez = self.antenna[id].trace_z
            EvB = self.antenna[id].trace_evB
            EvvB = self.antenna[id].trace_evvB
            Ev = self.antenna[id].trace_ev
            time = self.antenna[id].t

        global_peak = np.max(np.abs([Ex, Ey, Ez]))
        peak_index = np.where(np.abs([Ex, Ey, Ez]) == global_peak)[0][0]
        peak_time = time[peak_index]

        if self.bool_plot:
            fig = plt.figure(figsize=(15, 3))
            fig.suptitle("Traces", fontsize=16, y=1)
            plt.subplot(131)
            plt.plot(time, Ex, "r")
            plt.ylabel("signal in V/m")
            ax = plt.gca()
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

            plt.subplot(132)
            plt.plot(time, Ey, "b")
            plt.xlabel("time in ns")
            ax = plt.gca()
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

            plt.subplot(133)
            plt.plot(time, Ez, "k")
            ax = plt.gca()
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

            fig = plt.figure(figsize=(15, 3))
            fig.suptitle("Traces in shower plane", fontsize=16, y=1)
            plt.subplot(131)
            plt.plot(time, EvB, "r")
            plt.ylabel("signal in V/m")
            ax = plt.gca()
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

            plt.subplot(132)
            plt.plot(time, EvB, "b")
            plt.xlabel("time in ns")
            ax = plt.gca()
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

            plt.subplot(133)
            plt.plot(time, Ev, "k")
            ax = plt.gca()
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

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

        if id < len(self.antenna):
            time = self.antenna[id].time

            EvB = self.antenna[id].trace_evB
            EvvB = self.antenna[id].trace_evvB
            Ev = self.antenna[id].trace_ev
            traces = np.c_[time, EvB, EvvB, Ev]
        else:
            message = "ERROR: id = " + str(id) + " is out of the antenna array bounds!"
            raise ValueError(message)

        fluence_evB, SNR = Antenna.compute_fluence(time, EvB)
        fluence_evvB, SNR = Antenna.compute_fluence(time, EvvB)
        fluence_ev, SNR = Antenna.compute_fluence(time, Ev)
        fluence = np.sqrt(fluence_evB * fluence_evB
                          + fluence_evvB * fluence_evvB
                          + fluence_ev * fluence_ev)

        r_plane = self.antenna[id].r_proj[0:2]
        fluence_geo, fluence_ce = Antenna.compute_f_geo_ce(fluence_evB, fluence_evvB , r_plane)


        self.antenna[id].fluence = fluence
        self.antenna[id].fluence_evB = fluence_evB
        self.antenna[id].fluence_evvB = fluence_evvB
        self.antenna[id].fluence_geo = fluence_geo
        self.antenna[id].fluence_ce = fluence_ce

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
        n_ant = len(self.antenna)

        step = round(n_ant / 10)

        try:
            antenna_list = self.antenna.values()
        except:
            antenna_list = self.antenna

        if self.simulation_type == "custom":
            RunInfo = Table.read(self.simulation, path="RunInfo")
            EventName = RunInfo["EventName"][0]
            AntennaFluenceInfo = Table.read(
                self.simulation, EventName + "/AntennaFluenceInfo"
            )
            for ant in AntennaFluenceInfo:
                idx = ant["ID"]
                self.antenna[idx].fluence = ant["Fluence_efield"]

                fluence_site = CartesianRepresentation(
                    ant["Fluencex_efield"],
                    ant["Fluencey_efield"],
                    ant["Fluencez_efield"]
                )
                fluence_shower = np.array([
                    np.dot(self.shower.projection_mat, f_site) for f_site in fluence_site
                    ])

                self.antenna[idx].fluence_geo = -1
                self.antenna[idx].fluence_ce = -1
                self.antenna[idx].fluence_evB = np.abs(fluence_shower[0])
                self.antenna[idx].fluence_evvB = np.abs(fluence_shower[1])
                self.antenna[idx].fluence = np.linalg.norm(fluence_shower)

        if self.printLevel > 0:
            print("* Evaluating the fluences:")
            print("--> 0 % complete;")
        for counter, ant in enumerate(antenna_list):
            # Read traces or voltages
            if step > 0 and (counter + 1) % step == 0 and self.printLevel > 0:
                print("-->", int((counter + 1) / (10 * step) * 100), "% complete;")

            if self.simulation_type != "custom":
                self.process_antenna(ant.ID)
            if ant.fluence > 0:
                ant.sigma_f = np.sqrt(ant.fluence)

        if self.printLevel > 0:
            print("\n")

    def Eval_par_fluences(self, par):
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
        alpha = np.arccos(np.dot(self.shower.ev, eB))
        d_Xmax = np.linalg.norm(
            (self.shower.r_core - self.shower.r_x_max)
        )
        rho_Xmax = SymFit.rho(d_Xmax, -self.shower.ev)

        for ant in antenna_list:
            if self.bool_EarlyLate and ant.wEarlyLate is not None:
                weight = ant.wEarlyLate
            else:
                weight = 1

            r_plane = ant.r_proj[0:2] * weight
            fluence = ant.fluence_evB / (weight ** 2)
            phi = np.arccos(np.dot(r_plane, np.array([1, 0])) / np.linalg.norm(r_plane))
            dist = np.linalg.norm((ant.r_proj - self.shower.r_Core_proj)[0:2]) * weight
            fluence_par[ant.ID] = SymFit.f_par_geo(
                fluence, phi, alpha, dist, d_Xmax, par, rho_Xmax
            )

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

        ID = antpos_fluences[:, 0]
        x_proj = antpos_fluences[:, 1]
        y_proj = antpos_fluences[:, 2]
        fluence_arr = antpos_fluences[:, 3]

        fluence_mean = {}
        fluence_mean2 = {}
        counter = {}
        pos = {}
        for i in range(len(fluence_arr)):
            # label = str(x_proj[i]) + str(y_proj[i])
            label = ID[i]

            if label in counter:
                counter[label] = counter[label] + 1
            else:
                counter[label] = 1

            pos[label] = [x_proj[i], y_proj[i]]

            if label in fluence_mean:
                fluence_mean[label] += fluence_arr[i]
                fluence_mean2[label] += fluence_arr[i] ** 2
            else:
                fluence_mean[label] = fluence_arr[i]
                fluence_mean2[label] = fluence_arr[i] ** 2

        f_mean = np.zeros(len(pos))
        f_mean2 = np.zeros(len(pos))
        r = np.zeros((len(pos), 2))

        trim_counter = {}
        for key, value in counter.items():
            if value < 2:
                pos.pop(key)

        index = 0
        for key, value in pos.items():
            label = key
            f_mean[index] = fluence_mean[label] / counter[label]
            f_mean2[index] = fluence_mean2[label] / counter[label]
            r[index] = value
            index += 1

        return r[:, 0], r[:, 1], f_mean, np.sqrt(f_mean2 - f_mean ** 2)

    def plot_antpos(self):
        """
        Plots the fluence and antenna positions in the site plane.

        Parameters
        ----------
        self: modERec.EnergyRec
            A class instance.

        """

        n_ant = len(self.antenna)

        try:
            antenna_list = self.antenna.values()
        except:
            antenna_list = self.antenna

        r_ant = np.zeros((n_ant, 3))
        for idx, ant in enumerate(antenna_list):
            r_ant[idx] = ant.r_ground

        fluence_arr = np.array([ant.fluence for ant in antenna_list])
        sel = np.where(fluence_arr > 0)[0]

        fig = plt.figure(figsize=(10, 7))
        ax = plt.gca()

        plt.scatter(
            r_ant[:, 0][sel], r_ant[:, 1][sel], c=fluence_arr[sel], cmap="viridis"
        )

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
        sel = np.where(fluence_arr > 0)[0]

        signal = np.c_[self.shower.r_proj[sel], fluence_arr[sel]]

        fluence_file = open(
            "fluence_ShowerPlane_" + str(round(self.shower.thetaCR)) + ".out", "a"
        )
        for entry in signal:
            print(str(entry)[1:-1], file=fluence_file)
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
        try:
            antenna_list = self.antenna.values()
        except:
            antenna_list = self.antenna

        for ant in antenna_list:
            r_ant = ant.r_ground - self.shower.r_core.T[0]
            ant.r_proj = np.dot(self.shower.projection_mat, r_ant)

            if self.simulation_type != "custom":
                E = zip(ant.trace_x, ant.trace_y, ant.trace_z)
                traces_proj = np.array([np.dot(self.shower.projection_mat, E_entry) for E_entry in E])
                ant.trace_evB = traces_proj[:,0]
                ant.trace_evvB = traces_proj[:,1]
                ant.trace_ev = traces_proj[:,2]
            else:
                ant.trace_evB = None
                ant.trace_evvB = None
                ant.trace_ev = None

        core = self.shower.r_core.T[0] - self.shower.r_core.T[0]
        r_Core_proj = np.dot(self.shower.projection_mat, core)

        self.shower.r_Core_proj = r_Core_proj

    def model_fit(self, filename="", Cs=None):
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

        if self.printLevel or filename != "":
            print("* Model fit:")
        if filename == "":
            fluence_arr = np.array([ant.fluence for ant in antenna_list])
            if all(f is None for f in fluence_arr):
                raise ValueError(
                    "--> fluence_arr == None. instance.Eval_fluences() has to be run!"
                )

            if self.bool_EarlyLate and self.printLevel:
                print("--> Early-late correction will be applied!")

        else:
            if not Path(filename).is_file():
                message = "ERROR: file " + filename + " not found!"
                raise ValueError(message)

            datafile = open(filename, "r")
            antpos_fluences = np.loadtxt(datafile)
            datafile.close()
            # x_proj, y_proj, fluence_arr, sigma_arr = self.eval_mean_fluences(antpos_fluences)

            # self.antenna = [self.Antenna() for ant in range(len(fluence_arr))]
            self.antenna = [Antenna(ant) for ant in range(len(antpos_fluences))]

            for ant in range(len(antpos_fluences)):
                # self.antenna[ant].fluence = fluence_arr[ant]
                # self.antenna[ant].r_proj = [x_proj[ant],y_proj[ant],0]
                # self.antenna[ant].sigma_f = sigma_arr[ant]
                self.antenna[ant].ID = antpos_fluences[ant, 0]
                self.antenna[ant].r_proj = [
                    antpos_fluences[ant, 1],
                    antpos_fluences[ant, 2],
                    0,
                ]
                self.antenna[ant].fluence = antpos_fluences[ant, 3]
                self.antenna[ant].sigma_f = antpos_fluences[ant, 4]

            self.shower.r_Core_proj = np.array([0, 0, 0])

        AERA.aeraFit(self, filename, Cs)
        if self.printLevel or filename != "":
            print("--> Done!")
            print("\n")

    @staticmethod
    def early_late(r_proj, r_core_proj, d_xmax):
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

        R_0 = d_xmax
        
        r_ant = r_proj - r_core_proj
        R = (
            R_0 + r_ant[2]
        )  ## R_ant[2] is the distance from the core projected into ev
        wEarlyLate = R_0 / R

        return wEarlyLate


class AERA:
    """
    A class with aera specific methods.
    """

    @staticmethod
    def aeraLDF(par_fit, Cs, evB, x, y):
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
        elif len(par_fit) == 6:
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

        r = np.array([x, y])
        numA = r - C1 * evB - rcore
        partA = np.exp(-(np.linalg.norm(numA) ** 2) / sigma ** 2)

        numB = r - C2 * evB - rcore
        partB = np.exp(-(np.linalg.norm(numB) ** 2) / (C3 * np.exp(C4 * sigma)) ** 2)

        result = A * (partA - C0 * partB)
        if A < 0 or sigma < 0 or C0 < 0 or C3 < 0 or C4 < 0:
            result = np.inf
        if result < 0:
            result = 0.0
        return result

    @staticmethod
    def aeraChi2(par_fit, Cs, self):
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
        Chi2 = 0.0
        i = 0

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

        Shower.r_Core_proj = (
            self.shower.r_Core_proj
        )  ## Using class definition as a global variable!!

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

            if self.bool_EarlyLate and ant.wEarlyLate is not None:
                weight = ant.wEarlyLate
            else:
                weight = 1

            x = ant.r_proj[0] * weight
            y = ant.r_proj[1] * weight
            f = ant.fluence / (weight ** 2)
            sigma = ant.sigma_f / (weight ** 2)
            ldf_val = AERA.aeraLDF(par_fit, Cs, np.array([1, 0]), x, y)
            if ldf_val > max_ldf:
                max_ldf = ldf_val
                max_r_proj = ant.r_proj
            Chi2 = Chi2 + ((ldf_val - f) / sigma) ** 2
            i = i + 1

        return Chi2

    @staticmethod
    def aeraFit(self, filename, Cs):
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
        if filename == "":
            bestfit_out = "bestfit.out"

        else:
            bestfit_out = "bestfit_All.out"

        my_evB = np.array([1, 0])

        try:
            antenna_list = self.antenna.values()
        except:
            antenna_list = self.antenna

        # amplitude guess
        fluence_arr = np.array([ant.fluence for ant in antenna_list])
        sel = np.where(fluence_arr > np.mean(fluence_arr) + np.std(fluence_arr))[0]
        init_A = 2 * np.mean(fluence_arr[sel])

        # core position guess
        # core_index = np.where(fluence_arr==np.max(fluence_arr))[0][0]
        # init_xCore =  0 #antpos_proj[core_index,0]
        # init_yCore =  0 #antpos_proj[core_index,1]

        # sigma guess
        distances = np.array([np.linalg.norm(ant.r_proj[0:2]) for ant in antenna_list])

        init_sigma = np.mean(distances[sel])

        # Cs_aera = [0.5,-10,20,16,0.01]
        Cs_aera = [0.9, 0, 0, 6, 0.003]
        if Cs is None:
            par_fit = [
                init_A,
                init_sigma,
                Cs_aera[0],
                Cs_aera[1],
                Cs_aera[2],
                Cs_aera[3],
                Cs_aera[4],
            ]
            res = sp.optimize.minimize(
                AERA.aeraChi2, par_fit, args=(Cs, self), method="Nelder-Mead"
            )
            resx = res.x
        elif len(Cs) == 1:
            par_fit = [
                init_A,
                init_sigma,
                Cs_aera[1],
                Cs_aera[2],
                Cs_aera[3],
                Cs_aera[4],
            ]
            res = sp.optimize.minimize(
                AERA.aeraChi2, par_fit, args=(Cs, self), method="Nelder-Mead"
            )
            resx = np.insert(res.x, 2, Cs[0])
            Cs = None
        else:
            par_fit = [init_A, init_sigma]
            res = sp.optimize.minimize(
                AERA.aeraChi2, par_fit, args=(Cs, self), method="Nelder-Mead"
            )
            resx = np.append(res.x, Cs)
            Cs_aera = Cs

        chi2min = AERA.aeraChi2(resx, Cs, self)
        ndof = fluence_arr[fluence_arr > self.f_thres].size - res.x.size

        if self.printLevel > 0:
            print("** AERA fit:")
            print("---> ", "{:6} {:>10} {:>10}".format("Par", "Initial", "Bestfit"))
            print("---> ", "----------------------------")
            print(
                "---> ",
                "{:6} {:10} {:10}".format("A", round(init_A, 3), round(resx[0], 4)),
            )
            print(
                "---> ",
                "{:6} {:10} {:10}".format(
                    "sigma", round(init_sigma, 2), round(resx[1], 4)
                ),
            )
            print(
                "---> ",
                "{:6} {:10} {:10}".format(
                    "C0", round(Cs_aera[0], 2), round(resx[2], 4)
                ),
            )
            print(
                "---> ",
                "{:6} {:10} {:10}".format(
                    "C1", round(Cs_aera[1], 2), round(resx[3], 4)
                ),
            )
            print(
                "---> ",
                "{:6} {:10} {:10}".format(
                    "C2", round(Cs_aera[2], 2), round(resx[4], 4)
                ),
            )
            print(
                "---> ",
                "{:6} {:10} {:10}".format(
                    "C3", round(Cs_aera[3], 2), round(resx[5], 4)
                ),
            )
            print(
                "---> ",
                "{:6} {:10} {:10}".format(
                    "C4", round(Cs_aera[4], 2), round(resx[6], 4)
                ),
            )
            print("---> ", "----------------------------")
            print(
                "---> ", "Chi2min/n.d.o.f = ", str(round(chi2min, 2)), " / ", int(ndof)
            )

        bestfit = open(bestfit_out, "a")

        if filename == "":
            A = resx[0]
            sigma = resx[1]
            sin2Alpha = 1 - np.dot(self.shower.ev, self.shower.eB) ** 2.0
            Cs = np.array([resx[2], resx[3], resx[4], resx[5], resx[6]])
            Sradio = (A * np.pi / sin2Alpha) * (
                sigma ** 2.0 - Cs[0] * (Cs[3] ** 2.0) * np.exp(2 * Cs[4] * sigma)
            )
            print(*resx, self.shower.energy, Sradio, file=bestfit)

        else:
            print(*resx, file=bestfit)

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

        my_evB = np.array([1, 0])

        try:
            antenna_list = self.antenna.values()
        except:
            antenna_list = self.antenna

        A = self.bestfit[0]
        sigma = self.bestfit[1]
        rcore = self.shower.r_Core_proj[0:2]
        # sin2Alpha = (1-np.dot(self.shower.ev,self.shower.eB)**2.)

        Cs = np.array(
            [
                self.bestfit[2],
                self.bestfit[3],
                self.bestfit[4],
                self.bestfit[5],
                self.bestfit[6],
            ]
        )
        # Sradio = (A*np.pi/sin2Alpha)*(sigma**2. - Cs[0]*(Cs[3]**2.)*np.exp(2*Cs[4]*sigma))
        # print('S_radio=',round(Sradio,2))

        par = [A, sigma, Cs[0], Cs[1], Cs[2], Cs[3], Cs[4]]

        fluence_arr = np.array([ant.fluence for ant in antenna_list])
        if self.bool_EarlyLate:
            weight = np.array([ant.wEarlyLate for ant in antenna_list])
            if all(w is None for w in weight):
                weight = np.full(len(fluence_arr), 1)
        else:
            weight = np.full(len(fluence_arr), 1)
        sel = np.where(fluence_arr > 0)[0]
        weight = weight[sel]
        fluence_arr = fluence_arr[sel] / (weight ** 2)

        r_proj = np.array([ant.r_proj for ant in antenna_list])
        x_proj = r_proj[:, 0][sel] * weight - rcore[0]
        y_proj = r_proj[:, 1][sel] * weight - rcore[1]

        delta_X = np.max(x_proj) - np.min(x_proj)
        delta_Y = np.max(y_proj) - np.min(y_proj)
        mean_X = np.min(x_proj) + delta_X / 2
        mean_Y = np.min(y_proj) + delta_Y / 2

        delta_XY = np.max([delta_X, delta_Y]) * 1.10

        minXAxis = mean_X - delta_XY / 2
        maxXAxis = mean_X + delta_XY / 2
        minYAxis = mean_Y - delta_XY / 2
        maxYAxis = mean_Y + delta_XY / 2

        xx = np.arange(minXAxis, maxXAxis, delta_XY / 500)
        yy = np.arange(minYAxis, maxYAxis, delta_XY / 500)
        X, Y = np.meshgrid(xx, yy)  # grid of point

        Z = np.zeros((yy.size, xx.size))

        Shower.r_Core_proj = np.array([0, 0, 0])  # warning: global variable

        for i in range(yy.size):
            for j in range(xx.size):
                Z[i, j] = AERA.aeraLDF(
                    par, None, my_evB, X[i, j], Y[i, j]
                )  # evaluation of the function on the grid

        fig = plt.figure(figsize=[14, 5])
        plt.subplot(121)
        im = plt.imshow(
            Z,
            cmap="viridis",
            origin="lower",
            extent=[minXAxis, maxXAxis, minYAxis, maxYAxis],
        )  # drawing the function

        plt.scatter(
            x_proj,
            y_proj,
            c=fluence_arr,
            cmap="viridis",
            s=100,
            edgecolors=(1, 1, 1, 0.2),
        )
        plt.clim(
            np.min([np.min(Z), np.min(fluence_arr)]),
            np.max([np.max(Z), np.max(fluence_arr)]),
        )
        plt.colorbar().ax.set_ylabel(r"energy fluence in eV/m$^2$")

        plt.xlabel(r"distante along $\vec{v}\times\vec{B}$ (in m)")
        plt.ylabel(r"distante along $\vec{v}\times\vec{v}\times\vec{B}$ (in m)")

        plt.plot(0, 0, "w*")

        plt.xlim(minXAxis, maxXAxis)
        plt.ylim(minYAxis, maxYAxis)

        plt.subplot(122)
        plt.scatter(
            x_proj,
            y_proj,
            c=fluence_arr,
            cmap="viridis",
            s=100,
            edgecolors=(1, 1, 1, 0.2),
        )
        plt.xlabel(r"distante along $\vec{v}\times\vec{B}$ (in m)")
        plt.ylabel(r"distante along $\vec{v}\times\vec{v}\times\vec{B}$ (in m)")
        plt.colorbar().ax.set_ylabel(r"energy fluence in eV/m$^2$")
        plt.xlim(minXAxis, maxXAxis)
        plt.ylim(minYAxis, maxYAxis)

        # 1D LDF
        temp_x = x_proj
        temp_y = y_proj

        temp_dist = np.sqrt(temp_x * temp_x + temp_y * temp_y)

        fig_ldf = plt.figure(figsize=[14, 5])
        plt.subplot(121)
        yerr = np.array([ant.sigma_f for ant in antenna_list])[sel]
        plt.errorbar(temp_dist, fluence_arr, yerr=yerr, fmt=".")
        plt.xlabel("Distance from core in m")
        plt.ylabel(r"Fluence in eV/m$^2$")
        plt.gca().set_yscale("log")

        residual = np.zeros(fluence_arr.size)

        for i in range(fluence_arr.size):
            residual[i] = (
                fluence_arr[i] - AERA.aeraLDF(par, None, my_evB, x_proj[i], y_proj[i])
            ) / np.sqrt(fluence_arr[i])

        plt.subplot(122)
        plt.errorbar(temp_dist, residual, yerr=1, fmt=".")
        plt.xlabel("Distance from core in m")
        plt.ylabel(r"($f$ - fit)/$\sigma_f$")
        # plt.xlim(0,500)
        plt.ylim(-2, 2)
        plt.grid()


class SymFit:
    """
    A class with the symmetric signal distribution specific methods.

    """

    ## The initial guess for the a_ratio fit.
    g_a_par = [0.373, 762.6, 0.149, 0.189]

    ## The mean air density at shower maximum
    g_rho_mean = 0.327

    ## The initial guess for the ldf par fit.
    g_ldf_par = None

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
        rho_mean = SymFit.g_rho_mean
        return (
            par[0]
            * (r / d_Xmax)
            * np.exp(r / par[1])
            * (np.exp((rho_max - rho_mean) / par[2]) - par[3])
        )

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
        cos_sin_ratio = np.cos(phi) / np.abs(np.sin(alpha))
        return f_vB / ((1 + cos_sin_ratio * sqrta) ** 2)

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

        height = np.dot(r * e_vec, np.array([0, 0, 1])) / 1000  # from m to km

        H = 10.4  # in km
        rho_0 = 1.225  # in kg/m^3
        site_height = (
            EnergyRec.site_height / 1000
        )  # in km (Hard Coded Using Class Definition!!!)

        return rho_0 * np.exp(-(site_height + height) / H)

    @staticmethod
    def a_ratio_chi2(par, fluence_geo, fluence_ce, alpha, r, d_Xmax, rho_Xmax):
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
        sel = np.where(fluence_geo > 0)[0]

        a_arr = (np.sin(alpha[sel]) ** 2) * fluence_ce[sel] / fluence_geo[sel]

        Chi2 = 0
        for i in range(fluence_geo[sel].size):
            a_theo = SymFit.a_ratio(r[sel][i], d_Xmax[sel][i], par, rho_Xmax[sel][i])
            # if a_arr[i] < 1:
            Chi2 = Chi2 + (a_arr[i] - a_theo) ** 2

        return Chi2

    @staticmethod
    def a_ratio_fit(fluence_geo, fluence_ce, alpha, r, d_Xmax, rho_max):
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
        par = SymFit.g_a_par
        res = sp.optimize.minimize(
            SymFit.a_ratio_chi2,
            par,
            args=(fluence_geo, fluence_ce, alpha, r, d_Xmax, rho_max),
            method="Nelder-Mead",
        )
        return res.x

    @staticmethod
    def SymLDF(par, r):
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

        LDF = A * np.exp(-B * r - C * r ** 2 - D * r ** 3)
        return LDF
    
    @staticmethod
    def SymLDF_2022(par, r):
        r"""
        The symmetric ldf to be fit to the fluence_par data.

        .. math::
            f_{GS}(r) = f_0\left[\exp\left(-\left(\frac{r-r_0^{fit}}{\sigma}\right)^{p(r)}\right)+\frac{a_{rel}}{1+\exp(s.[r/r_0^{fit}-r_{02}]))}\right]

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
        f_0 = par[0]
        r_0 = par[1]
        sigma = par[2]
        p_r = 2
        a_rel = par[3]
        s = par[4]
        r_02 = par[5]

        A = np.exp(-pow((r - r_0) / sigma, p_r))
        B = a_rel / (1 + np.exp(s * (r / r_0 - r_02)))
        LDF = f_0 * (A + B)
        return LDF

    @staticmethod
    def LDF_chi2(par, r, fluence_par):
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

        # Constraints from parameter estimation
        #if(par[0]>700):
        #    return np.inf
        #if(par[3]<0):
        #    return np.inf

        sel = np.where(fluence_par > 0)[0]
        f_par = fluence_par[sel]

        if(par[0] > np.max(f_par)): # To prevent LDF too high close to the core
            return np.inf

        Chi2 = 0
        sigma = 0.03 * f_par + 1.e-4 * np.max(f_par)
        for i in range(f_par.size):
            LDF = SymFit.SymLDF_2022(par, r[sel][i])
            # if a_arr[i] < 1:
            Chi2 = Chi2 + ((f_par[i] - LDF) / sigma[i]) ** 2
            #Chi2 = Chi2 + ((f_par[i] - LDF)) ** 2

        #Chi2 = Chi2 + (SymFit.SymLDF_2022(par, 10000))** 2 # Asymptotical limit

        return Chi2

    @staticmethod
    def SymLDF_fit(r, fluence_par):
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

        if SymFit.g_ldf_par is None:
            # Estimating the parameters
            idx = np.argsort(r)
            sort_dist = np.array(r)[idx]
            sort_f = np.array(fluence_par)[idx]

            n_ant = len(sort_dist)
            r0 = sort_dist[0]
            f0 = sort_f[0]

            idx_max = np.where(sort_f == np.max(sort_f))[0][0]
            r1 = sort_dist[idx_max]
            f1 = sort_f[idx_max]
            r2 = sort_dist[
                (idx_max + n_ant) // 2
            ]  # Did not work for ldfs 'clustered' (ex. three well sampled distances)
            f2 = sort_f[(idx_max + n_ant) // 2]
            r3 = sort_dist[-1]
            f3 = sort_f[-1]
            # r2 = (r1+r3)/2 # Possible alternative to 'clustered' ldfs
            # f2 = (f1+f3)/2

            a = np.array(
                [
                    [1, -r0, -(r0 ** 2), -(r0 ** 3)],
                    [1, -r1, -(r1 ** 2), -(r1 ** 3)],
                    [1, -r2, -(r2 ** 2), -(r2 ** 3)],
                    [1, -r3, -(r3 ** 2), -(r3 ** 3)],
                ]
            )
            b = np.array([np.log(f0), np.log(f1), np.log(f2), np.log(f3)])

            par = np.linalg.solve(a, b)

            par[0] = np.exp(par[0])

        else:
            par = SymFit.g_ldf_par

        res = sp.optimize.minimize(
            SymFit.LDF_chi2, par, args=(r, fluence_par), method="Nelder-Mead"
        )
        return res.x

    @staticmethod
    def Sradio_geo(par, ldf_par, alpha, rho_Xmax):
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
        E_rad = (
            2
            * np.pi
            * sp.integrate.quad(lambda r: r * SymFit.SymLDF_2022(ldf_par, r), 0, 2000)[0]
        )
        sin2alpha = np.sin(alpha) ** 2.0

        p0 = par[0]
        p1 = par[1]
        # rho_mean = 0.648
        # rho_mean = 0.327
        rho_mean = SymFit.g_rho_mean
        den = sin2alpha * (1 - p0 + p0 * np.exp(p1 * (rho_Xmax - rho_mean)))

        return E_rad * 1.0e-9 / den  # in GeV

    @staticmethod
    def Sradio_mod(par, E):
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

        return S_19 * (E / 10) ** gamma

    @staticmethod
    def Chi2_joint_S(par, ldf_par_arr, alpha_arr, rho_Xmax_arr, E_arr):
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
            S_geo = SymFit.Sradio_geo(
                par[0:2], ldf_par_arr[i], alpha_arr[i], rho_Xmax_arr[i]
            )
            S_mod = SymFit.Sradio_mod(par[2:4], E_arr[i])
            #sigma = np.sqrt(S_geo)
            sigma = 1
            Chi2 = Chi2 + ((S_geo - S_mod) / sigma) ** 2

        return Chi2

    @staticmethod
    def joint_S_fit(ldf_par_arr, alpha_arr, rho_Xmax_arr, E_arr):
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
        p1 = -2.370  # m^3/kg
        S_19 = 1.408  # in GeV
        gamma = 1.995
        par = [p0, p1, S_19, gamma]

        res = sp.optimize.minimize(
            SymFit.Chi2_joint_S,
            par,
            args=(ldf_par_arr, alpha_arr, rho_Xmax_arr, E_arr),
            method="Nelder-Mead",
        )
        return res.x


print("* EnergyRec default values summary:")
print("--> bool_plot = ", EnergyRec.bool_plot)
print("--> bool_EarlyLate = ", EnergyRec.bool_EarlyLate)
print("--> nu_low = ", EnergyRec.nu_low)
print("--> nu_high = ", EnergyRec.nu_high)
print("--> SNR_thres = ", EnergyRec.SNR_thres)
print("--> thres_low = ", EnergyRec.thres_low)
print("--> thres_high = ", EnergyRec.thres_high)
print("--> f_thres = ", EnergyRec.f_thres)
print("--> printLevel = ", EnergyRec.printLevel)
print("\n")
