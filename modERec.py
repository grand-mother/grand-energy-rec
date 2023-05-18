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


import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.fftpack import fft, ifft
from scipy.signal import hilbert


class RawData:
    """
    A class for the raw input attributes.

    Attributes
    ----------
    ant_ID:
        The antenna ID
    r_ground:
        The antenna positions
    r_core:
        Shower core coordinates
    traces_time:
        The trace times
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
    r_x_max:
        X max position w.r.t. the shower core
    energy:
        Monte Carlo energy
    """

    ant_ID = None
    r_ground = None
    r_core = None
    traces_time = None
    traces_x = None
    traces_y = None
    traces_z = None
    ev = None
    eB = None
    r_x_max = None
    energy = None

    def __init__(self, dict):
        self.ant_ID = dict["ant_ID"]
        self.r_ground = dict["r_ground"]
        self.r_core = dict["r_core"]
        self.traces_time = dict["traces_time"]
        self.traces_x = dict["traces_x"]
        self.traces_y = dict["traces_y"]
        self.traces_z = dict["traces_z"]
        self.ev = dict["ev"]
        self.eB = dict["eB"]
        self.r_x_max = dict["r_x_max"]
        self.energy = dict["energy"]

class FluenceData:
    """
    A class for the reconstruction input attributes.

    Attributes
    ----------
    r_shower:
        The antenna positions in the shower plane
    r_core_shower:
        The shower core position in the shower plane
    fluence_evB:
        Fluence in the v x B direction
    fluence_evvB:
        Fluence in the v x v x B direction
    fluence:
        Total fluence
    energy:
        Monte Carlo energy
    """

    r_shower = None
    r_core_shower = None
    fluence_evB = None
    fluence_evvB = None
    fluence = None
    energy = None

    def __init__(self, dict):
        self.r_shower = dict["r_shower"]
        self.r_core_shower = dict["r_core_shower"]
        self.fluence_evB = dict["fluence_evB"]
        self.fluence_evvB = dict["fluence_evvB"]
        self.fluence = dict["fluence"]
        self.energy = dict["energy"]

class Antenna:
    """
    A class for the antenna signal processing.

    It has tools for the FFT, trace_recover and fluence evaluation.  

    """

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
    def hilbert_envelope(trace_rec):
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
    def compute_fluence(time_arr, trace, nu_low = 50, nu_high = 200, SNR_thres=10):
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

        trace_fft = Antenna.fft_filter(time_arr, trace, nu_low, nu_high)
        trace_rec = Antenna.trace_recover(time_arr, trace_fft)

        # Check if peak is within the threshold range after offset, cut and trace recover.
        hilbert_env = Antenna.hilbert_envelope(trace_rec)

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

class EnergyRec:
    """
    A class for the energy reconstruction.

    It has the inner classes: AERA, Antenna and Shower.

    Attributes
    ----------

    bool_plot: bool
        Toggles the plots on and off (default: false).
    bool_el: bool
        Toggles the early late correction on and off (default: true).
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
    raw_data:
        An instance of the RawData class
    fluence_data:
        An instance of the FluenceData class
    bestfit:
        The bestfit values of the parameters (default: None).
    printLevel: int
        A print level variable (default: 0).
    """

    ## Toggles the plots on and off.
    bool_plot = False
    ## Toggles early late correction on and off.
    bool_el = True
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
    ## An instance of the class RawData
    raw_data = None
    ## An instance of the class FluenceData
    fluence_data = None
    ## The bestfit values of the parameters
    bestfit = None
    ## A print level variable
    printLevel = 0

    def __init__(self, my_input):
        """
        The default init function for the class EnergyRec.

        Parameters
        ----------
        self: modERec.EnergyRec
            A class instance.
        simulation:
            The path to the simulation directory or file.
        """

        if(isinstance(my_input, FluenceData)):
            self.fluence_data = my_input

        elif(isinstance(my_input, RawData)):
            self.raw_data = my_input
            ev = my_input.ev

            evB = np.cross(ev, my_input.eB)
            evB /= np.linalg.norm(evB)
            evvB = np.cross(ev, evB)
            projection_mat = np.linalg.inv(np.array([
                evB, evvB, ev
                ]).T)
            
            r_shower = np.array([
                    np.dot(projection_mat, r_ground) for r_ground in my_input.r_ground
                    ])
            
            r_core_shower = np.dot(projection_mat, my_input.r_core)
            
            if self.bool_el:
                w_early_late = [self.early_late(rr, r_core_shower, np.linalg.norm(my_input.r_x_max - my_input.r_core))
                            for rr in r_shower]
            else:
                w_early_late = np.ones(len(r_shower))
            
            r_shower = np.array([(rr - r_core_shower) * ww for rr, ww in zip(r_shower, w_early_late)]) + r_core_shower
            
            fluence_x = [
                Antenna.compute_fluence(xx, yy, self.nu_low, self.nu_high)[0]
                for xx, yy in zip(my_input.traces_time, my_input.traces_x)
                ] / (np.array(w_early_late)**2)
            fluence_y = [
                Antenna.compute_fluence(xx, yy, self.nu_low, self.nu_high)[0]
                for xx, yy in zip(my_input.traces_time, my_input.traces_y)
                ] / (np.array(w_early_late)**2)
            fluence_z = [
                Antenna.compute_fluence(xx, yy, self.nu_low, self.nu_high)[0]
                for xx, yy in zip(my_input.traces_time, my_input.traces_z)
                ] / (np.array(w_early_late)**2)
            
            fluence_site = np.c_[fluence_x, fluence_y, fluence_z]
            fluence_shower = np.array([
                    np.dot(projection_mat, f_site) for f_site in fluence_site
                    ])

            fluence_evB = np.abs(fluence_shower[:,1])
            fluence_evvB = np.abs(fluence_shower[:,2])
            fluence = [
                    np.linalg.norm(ff) for ff in fluence_shower
                    ]
            energy = my_input.energy

            self.fluence_data = FluenceData({"r_shower" : r_shower,
                                  "r_core_shower" : r_core_shower,
                                  "fluence_evB" : fluence_evB,
                                  "fluence_evvB" : fluence_evvB,
                                  "fluence" : fluence,
                                  "energy" : energy
                                 })

        if self.printLevel > 0:
            print("* EnergyRec instance starting values summary:")
            print("--> bool_plot = ", self.bool_plot)
            print("--> nu_low = ", self.nu_low)
            print("--> nu_high = ", self.nu_high)
            print("--> SNR_thres = ", self.SNR_thres)
            print("--> thres_low = ", self.thres_low)
            print("--> thres_high = ", self.thres_high)
            print("--> f_thres = ", self.f_thres)
            print("\n")

    def simulation_inspect(self):
        """
        Outputs theta, phi, Energy and the core position.

        Parameters
        ----------
        self: modERec.EnergyRec
            A class instance.

        """
        if self.raw_data is not None:
            print("This is the raw input:")
            print(self.raw_data)
            print()

        if self.fluence_data is not None:
            print("This is the rec input:")
            print(self.fluence_data)


    def Eval_par_fluences(self, par):
        """
        Evaluates the fluence par for a give set of parameters.

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

        fluence_arr = np.array([ant.fluence for ant in antenna_list])

        fluence_par = {}
        alpha = np.arccos(np.dot(self.raw_data.ev, self.raw_data.eB))
        d_Xmax = np.linalg.norm(
            (self.raw_data.er_core - self.raw_data.r_x_max)
        )
        rho_Xmax = SymFit.rho(d_Xmax, - self.raw_data.ev)

        rr_core = self.fluence_data.r_core_shower
        for i in range(len(self.fluence_data.r_shower)):
            rr = self.fluence_data.r_shower[i]
            r_plane = rr[0:2]
            phi = np.arccos(
                  np.dot(r_plane, np.array([1, 0])) / np.linalg.norm(r_plane)
                )
            dist = np.linalg.norm(
                (rr - rr_core)[0:2]
                )
            fluence_par[i] = SymFit.f_par_geo(
                self.fluence_data.fluence[i], phi, alpha, dist, d_Xmax, par, rho_Xmax
            )

        return fluence_par

    
    def plot_antpos(self):
        """
        Plots the fluence and antenna positions in the site plane.

        Parameters
        ----------
        self: modERec.EnergyRec
            A class instance.

        """

        sel = np.where(self.fluence_data["fluence"] > 0)[0]

        fig = plt.figure(figsize=(10, 7))
        ax = plt.gca()

        plt.scatter(
            self.fluence_data["r_shower"][:, 0][sel],
            self.fluence_data["r_shower"][:, 1][sel],
            c=fluence_arr[sel],
            cmap="viridis"
        )

        plt.xlabel("x (in m)")
        plt.ylabel("y (in m)")
        plt.colorbar().ax.set_ylabel(r"Energy fluence (eV/m$^2$)")
        plt.show()

    @staticmethod
    def early_late(r_shower, r_core_shower, d_xmax):
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
        
        r_ant = r_shower - r_core_shower
        R = (
            R_0 + r_ant[2]
        )  ## R_ant[2] is the distance from the core projected into ev
        wEarlyLate = R_0 / R

        return wEarlyLate


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

    ## The initial guess for the joint fit.
    g_joint_par = None

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
    def rho(r, e_vec, site_height=0):
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
    def ressonance(pars, x):
        A = pars[0]
        omega_0 = pars[1]
        gamma = pars[2]
        omega_star = np.sqrt(omega_0**2 - 2*gamma**2)
        n = 2
        a = 1
        sigma = pars[3]
        d0 = pars[4]
        
        temp = ((a*x) ** n - omega_0 * omega_0)
        den = (4 * gamma * gamma * (a*x) ** n + temp ** 2)
        return A*np.exp(-((x-d0)/sigma)**2)/den

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
            #if(r[sel][i] > 5 * par[1]): # Not too far from Cerennkov radius
            #    continue
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
            #* sp.integrate.quad(lambda r: r * SymFit.ressonance(ldf_par, r), 0, 2000)[0]
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
            sigma = S_geo
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
        if SymFit.g_ldf_par is None:
            p0 = 0.394
            p1 = -2.370  # m^3/kg
            S_19 = 1.408  # in GeV
            gamma = 1.995
            par = [p0, p1, S_19, gamma]
        else:
            par = SymFit.g_ldf_par

        res = sp.optimize.minimize(
            SymFit.Chi2_joint_S,
            par,
            args=(ldf_par_arr, alpha_arr, rho_Xmax_arr, E_arr),
            method="Nelder-Mead",
        )
        return res.x


print("* EnergyRec default values summary:")
print("--> bool_plot = ", EnergyRec.bool_plot)
print("--> nu_low = ", EnergyRec.nu_low)
print("--> nu_high = ", EnergyRec.nu_high)
print("--> SNR_thres = ", EnergyRec.SNR_thres)
print("--> thres_low = ", EnergyRec.thres_low)
print("--> thres_high = ", EnergyRec.thres_high)
print("--> f_thres = ", EnergyRec.f_thres)
print("--> printLevel = ", EnergyRec.printLevel)
print("\n")
