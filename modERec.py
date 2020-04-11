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
# --> EnergyRec.MCMC that contains MCMC methods.\n
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

from grand.simulation import CoreasShower
import astropy.units as u
from astropy.coordinates import CartesianRepresentation
from grand import Rotation

class EnergyRec:
    """
    A class for the energy reconstruction.
    
    It has the inner classes: AERA, Antenna, MCMC and Shower.
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
    ## The path to the simulation directory.
    sim_dir = None
    ## An instance of the class Antenna
    antenna = None
    ## An instance of the class Shower
    shower = None
    ## An instance of the class MCMC
    mcmc = None
    ## The bestfit values of the parameters
    bestfit = None
    ## The shower imported using the standard grand package
    GRANDshower = None

    def __init__(self,sim_dir):
        """
        The default init function for the class EnergyRec.
    
        Args:
            self: An instance of EnergyRec;
            sim_dir: The path to the simulation directory;
        """
        self.sim_dir = sim_dir
        self.shower = self.Shower()
        self.mcmc = self.MCMC()

        if not Path(self.sim_dir).is_dir():
            print("ERROR: directory ",self.sim_dir," not found!")
            raise SystemExit("Stop right there!")

        self.GRANDshower = CoreasShower.load(self.sim_dir)
        
        n_ant = len(self.GRANDshower.fields)
        self.antenna = [self.Antenna() for ant in range(n_ant)]
        self.shower_projection()

        self.Eval_fluences()
        self.plot_antpos()

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

        print("* EnergyRec instance starting values summary:")
        print("--> bool_plot = ",self.bool_plot)
        print("--> bool_EarlyLate = ",self.bool_EarlyLate)
        print("--> nu_low = ",self.nu_low)
        print("--> nu_high = ",self.nu_high)
        print("--> SNR_thres = ",self.SNR_thres)
        print("--> thres_low = ",self.thres_low)
        print("--> thres_high = ",self.thres_high)
        print("--> f_thres = ",self.f_thres)
      
    def simulation_inspect(self):
        """
        Checks whether the files needed for the reconstruction are available.
            
        Outputs theta, phi, Energy and the core position.

        Args:
            self: An instance of EnergyRec.
            
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

        Args:
            self: An instance of EnergyRec.Antenna.
            id: The antenna id.  

        Fills:
            EnergyRec.Antenna.traces.
    
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

        Args:
            self: An instance of EnergyRec.Antenna.
        
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
            print("ERROR: id = ",id," is out of the antenna array bounds!")
            exit()

        # Check if peak is within the threshold range
        peak = np.max(np.abs(traces[:,1:4]))
        if(peak < self.thres_low or peak > self.thres_high):
            self.antenna[id].fluence = -1
            return
            
        traces_cut = self.Antenna.offset_and_cut(traces)
        traces_fft = self.Antenna.fft_filter(traces_cut, self.nu_low, self.nu_high, self.bool_plot)
        traces_rec = self.Antenna.trace_recover(traces_cut[:,0],traces_fft,self.bool_plot)

        # Check if peak is within the threshold range after offset, cut and trace recover.
        if(np.max(np.abs(traces_rec[:,1:4]))<self.thres_low):
            self.antenna[id].fluence = -1
            self.antenna[id].fluence_geo = -1
            self.antenna[id].fluence_ce = -1
            self.antenna[id].fluence_evB = -1
            self.antenna[id].fluence_evvB = -1
            return
        else:
            hilbert_env = self.Antenna.hilbert_envelope(traces_rec, self.bool_plot)
            self.antenna[id].compute_fluence(traces_rec[:,0],hilbert_env,self.SNR_thres,self.bool_plot)
                
    
    def Eval_fluences(self):
        """
        Evaluates the geomagnetic and charge excess fluences for a set os antennas.
            
        It has two thresholds for the fluence: thres_low and thres_high.

        The early-late correction is performed automaticaly in this function.

        Args:
            self: An instance of EnergyRec.

        Fills:
            EnergyRec.Antenna.fluence for all antennas
            EnergyRec.Antenna.fluence_evB for all antennas
            EnergyRec.Antenna.fluence_evvB for all antennas
            EnergyRec.Antenna.fluence geo for all antennas
            EnergyRec.Antenna.fluence_ce for all antennas
        
        """
        n_ant = len(self.GRANDshower.fields)
    
        step = int(n_ant/10)
        counter = 0

        print("* Evaluating the fluences:")
        print("--> 0 % complete;")
        for ant in range(n_ant):
            #Read traces or voltages
            if ((ant+1)%step == 0):
                print("-->",int((ant+1)/(10*step)*100),"% complete;")

            self.process_antenna(ant)

            if(self.antenna[ant].fluence > self.f_thres):
                r_plane =self.antenna[ant].r_proj[0:2]
                cosPhi = np.dot(r_plane,np.array([1,0]))/np.linalg.norm(r_plane)
                sinPhi = np.sqrt(1-cosPhi*cosPhi)
                my_fluence_geo = np.sqrt(self.antenna[ant].fluence_evB)-(cosPhi/sinPhi)*np.sqrt(self.antenna[ant].fluence_evvB)
                self.antenna[ant].fluence_geo = my_fluence_geo*my_fluence_geo
                self.antenna[ant].fluence_ce = self.antenna[ant].fluence_evvB/(sinPhi*sinPhi)

            else:
                self.antenna[ant].fluence_geo = -1
                self.antenna[ant].fluence_ce = -1
              
        print("\n")

    def Eval_par_fluences(self,par):
        """
        Evaluates the fluence par for a give set of parameters
        
        Args:
            self: An instance of EnergyRec
            par: The paramters of the a_ratio parametrization.

        Retuns:
            fluence_par: Parametrized fluence array.
        """

        
        fluence_arr = np.array([ant.fluence for ant in self.antenna])
        if all(f is None for f in f_list):
            self.Eval_fluences()

        n_ant = len(self.GRANDshower.fields)
        fluence_par = np.zeros(n_ant)
        eB = self.shower.eB
        alpha = np.arccos(np.dot(self.shower.ev,eB))
        d_Xmax = np.linalg.norm((self.GRANDshower.core - self.GRANDshower.maximum).xyz.value)
        rho_Xmax = self.SymFit.rho(d_Xmax,-self.shower.ev)

        for ant in self.antenna:
            r_plane = ant.r_proj[0:2]
            phi = np.arccos(np.dot(r_plane,np.array([1,0]))/np.linalg.norm(r_plane))
            dist = np.linalg.norm((ant.r_proj - self.shower.r_Core_proj)[0:2])
            fluence_par[ant] = EnergyRec.SymFit.f_par_geo(ant.fluence_evB,phi,alpha,dist,d_Xmax,par,rho_Xmax)

        return fluence_par

    def plot_antpos(self):
        """
        Reads the position of the antennas.

        Args:
            self: An instance of EnergyRec.

        """
 
        n_ant = len(self.GRANDshower.fields)

        r_ant = np.zeros((n_ant,3))
        for key, value in self.GRANDshower.fields.items():
            r_ant[key]=value.electric.r.xyz.value
    
        fluence_arr = np.array([ant.fluence for ant in self.antenna])
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
        
        The filename has the structure fluence_ShowerPlane_THETACR.out' and is open with append option.

        Args:
            self: An instance of EnergyRec.
        """

        fluence_arr = np.array([ant.fluence for ant in self.antenna])
        sel = np.where(fluence_arr>0)

        signal = np.c_[self.shower.r_proj[sel],fluence_arr[sel]]
    
        fluence_file=open('fluence_ShowerPlane_'+str(round(self.shower.thetaCR))+'.out', 'a')
        for entry in signal:
            print(str(entry)[1:-1],file=fluence_file)
        fluence_file.close()

    def shower_projection(self):
        """
        Projects the antenna positions and traces into the shower plane.

        Args:
            self: An instance of EnergyRec.

        Fills:
            EnergyRec.Shower.core_proj;
            EnergyRec.Antenna.r_proj;
            EnergyRec.Shower.traces_proj.
        
        """
        #The antenna projection
        n_ant = len(self.GRANDshower.fields)
        antpos_proj = np.zeros((n_ant,3))

        self.GRANDshower.localize(latitude=45.5 * u.deg, longitude=90.5 * u.deg)

        shower_frame = self.GRANDshower.shower_frame()
        traces_proj = {}

        for key, value in self.GRANDshower.fields.items():
            r_ant = value.electric.r #- self.GRANDshower.core
            self.antenna[key].r_proj = self.GRANDshower.transform(r_ant,shower_frame).cartesian.xyz

            E = self.GRANDshower.fields[key].electric.E
            traces_proj[key] = self.GRANDshower.transform(E,shower_frame)

        core = self.GRANDshower.core
        r_Core_proj = self.GRANDshower.transform(core,shower_frame).cartesian.xyz

        self.shower.r_Core_proj = r_Core_proj.value
        self.shower.traces_proj = traces_proj

    def model_fit(self,filename = "",Cs = None):
        """
        Performs the fit using a given model (set in the EnergyRec instance).
        
        If filename = "" (default) fits a given simulation.
            else it reads the file (antenna position (in shower plane) and fluences)
            and performs the fit.

        Args:
            self: An instance of EnergyRec.
            filename:   File with antenna positions and fluences for a given shower inclination;
            Cs: The LDF parameters to be used in the fit if filename != ""

        Fills:
            EnergyRec.bestfit.
        
        """ 

        print("* Model fit:")
        if(filename==""):
            fluence_arr = np.array([ant.fluence for ant in self.antenna])
            if all(f is None for f in fluence_arr):
                print("--> fluence_arr == None. instance.Eval_fluences() has to be run!")
                exit()            
        
        else:
            if not Path(filename).is_file():
                print("ERROR: file ",filename," not found!")
                raise SystemExit("Stop right there!")
            datafile = open(filename,'r')
            pos_fluence = np.loadtxt(datafile)
            datafile.close()
            for ant in self.antenna:
                ant.fluence = pos_fluence[:,2][ant]
                ant.r_proj = [pos_fluence[:,0],pos_fluence[:,1],0]

        if(self.bool_EarlyLate):
            print("--> Early-late correction applied!")

        EnergyRec.AERA.aeraFit(self,filename,Cs)
        print("\n")


    def early_late(self):
        """
        Evaluates the early-late correction factor.

        Args:
            self: An instance of EnergyRec.

        Fills:
            EnergyRec.Antenna.wEarlyLate;
            EnergyRec.Shower.d_Xmax.

        """
        rCore = self.GRANDshower.core.xyz.value
        rXmax = self.GRANDshower.maximum.xyz.value - rCore

        self.shower.d_Xmax = np.linalg.norm(rXmax)
        R_0 = np.linalg.norm(rXmax)

        for ant in self.antenna:
            r_ant = ant.r_proj.value - self.shower.r_Core_proj
            R = R_0 + r_ant[2] ## R_ant[2] is the distance from the core projected into ev
            ant.wEarlyLate = R_0/R        


    class Shower:
        """
        A class for the shower parameters.
    
        It stores the shower plane, theta.
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
        ## The position of the antennas projected into the shower plane.
        r_proj = None
        ## The position of the core projected into the shower plane.
        r_Core_proj = None
        ## Toggles the plots on and off.
        bool_plot = False
        ## X position of the shower core.
        CoreX = None
        ## Y position of the shower core.
        CoreY = None
        ## Distance to Xmax.
        d_Xmax = None

        def __init__(self):
            """
            The default init function for the class Shower.
    
            Args:
                self: An instance of EnergyRec.Shower.
            """


    class Antenna:
        """
        A class for the antenna signal processing.
    
        It has tools for the FFT, trace_recover and fluence evaluation.
        """
        ## The total fluence.
        fluence = None
        ## The geomagnetic component of the fluence
        fluence_geo = None
        ## The charge ecess component of the fluence
        fluence_ce = None
        ## The fluence on the evB direction
        fluence_evB = None
        ## The fluence on the evvB direction
        fluence_evvB = None
        ## The position of the antenna in the shower plane
        r_proj = None

        def __init__(self):
            """
            The default init function for the class Antenna.
    
            """
  
        @staticmethod
        def fft_filter(traces, nu_low = 50, nu_high = 200, bool_plot = False):
            """
            Evaluates the FFT of the signal.

            A filter is applied with width given by instance.antenna.nu_high and instance.antenna.nu_low.

            Args:
                traces: The traces to be filtered

            Returns:
                traces_fft: The Fourier transform of the traces.        

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

            Args:
                traces_fft: The Fourier transform of the traces'.

            Returns:
                traces_rc: The reconstructed traces.
                        
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
            """
            Evaluates the hilbert envelope of the recunstructed traces.

            \f$\mathcal{H}\{f(x)\}:=H(x)=\frac{1}{\pi}{\rm p.v.}\int_{-\infty}^\infty \frac{f(u)}{u-x}{\rm d}u\f$
            
            Args:
                traces_rec: The reconstructed traces.

            Returns:
                hilbert: The hilbert envelopes [total, in evB direction, in evvB direction, in ev direction].
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
            """
            Computes the fluence for a given antenna.
            
            \f$ f = \epsilon_0 c\left(\Delta t \sum_{t_1}^{t_2} \left| \vec{E}(t_i)\right|^2 - \Delta t \frac{t_2-t_1}{t_4-t_3} \sum_{t_3}^{t_4} \left| \vec{E}(t_i)\right|^2 \right) \f$
            \n \n It has a threshold for the SNR set by EnergyRec.SNR_thres.

            Args:
                self: an instance of EnergyRec.Antenna.
                hilbert_env: The hilbert envelopes.

            Fills:
                EnergyRec.Antenna.fluence;
                EnergyRec.Antenna.fluence_geo;
                EnergyRec.Antenna.fluence_ce;
                EnergyRec.Antenna.fluence_evB;
                EnergyRec.Antenna.fluence_evvB.
            
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
            
            The offset prevents problems due to traces to close to the time origin.
            The cut, reduces the time window to speed up the code.

            Args:
                traces: The traces to be cut.
            
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

    class MCMC:
        """
        A class to handle MCMC for parameter p.d.f.s estimation.
        """
        ## The number of sampling links
        num_samples = 100000
        ## The number of burn in links
        burn_in = 1000
        ## The chain itself
        chain = None
        
        def __init__(self):
            """
            The standard initialization class.
            """
            pass

        def metropolis_hastings(self):
            """
            Performs the MCMC sampling using the Metropolis-Hastings algorithm.

            Args:
                self: An instance of EnergyRec.MCMC.

            Fills:
                EnergyRec.MCMC.chain.
            """
            print("* Performing the MCMC sampling:")
            my_evB = np.array([1,0])
            Cs=0
            n_par = self.bestfit.size

            # sample normal values as stepsize for the updates
            # important: g is symmetric, so we don't have to use it in the calculation of alpha below
            steps = np.ndarray(shape=(EnergyRec.MCMC.num_samples,n_par))
            for i in range(n_par):
                steps[:,i] = np.random.normal(0, np.abs(self.bestfit[i])/100., EnergyRec.MCMC.num_samples)

            # with some bookkeeping, I only have to call the pdf of f once per loop iteration
            # that is initialized here
            x = deepcopy(self.bestfit)

            first_step = np.zeros(n_par)
            for i in range(n_par):
                first_step[i] = np.random.normal(0, np.abs(self.bestfit[i])/100.)
            x_next = x + first_step

            chi2min = EnergyRec.AERA.aeraChi2(x,self)

            myChi2 = EnergyRec.AERA.aeraChi2(x,self)
            current_prob = np.exp(-(myChi2-chi2min)/2)

            myChi2 = EnergyRec.AERA.aeraChi2(x_next,self)
            next_prob = np.exp(-(myChi2-chi2min)/2)

            x_chosen = np.zeros((EnergyRec.MCMC.num_samples,n_par))

            probs =  np.zeros(EnergyRec.MCMC.num_samples)

            step = int(EnergyRec.MCMC.num_samples/10)

            print("--> 0% complete;")
            for i in range(EnergyRec.MCMC.num_samples):

                if ((i+1)%step == 0):
                    print("-->",int((i+1)/(10*step)*100),"% complete;")

                # to account for cases where the pdf is 0
                # it would be good to avoid them by having a sensible starting point for x
                # they can also occur if the stepsize is so huge that our samples run out of domain
                # so this is a security measure
                if current_prob == 0:
                    # we always accept the next sample
                    alpha = 1
                elif next_prob == 0:
                    # we never accept the next sample accept the next sample
                    alpha = 0
                else:
                    # this is the normal MH alpha calculation
                    alpha = next_prob / current_prob

                if np.random.rand() < alpha:
                    x = x_next
                    current_prob = next_prob

                probs[i] = current_prob

                x_next = x + steps[i]

                myChi2 = EnergyRec.AERA.aeraChi2(x_next,self)
                next_prob = np.exp(-(myChi2-chi2min)/2)

                x_chosen[i] = x

            #plt.plot(range(num_samples),probs)
            x_final = x_chosen[EnergyRec.MCMC.burn_in:]

            self.chain = x_final

        def contour1DMCMC(self,par,xlabel="par1"):
            """
            Plots the p.d.f. for a given parameter from the chain.

            Args:
                self: An instance EnergyRec.MCMC;
                par: The corresponding parameter;
                xlabel: A label for the parameter.
            """

            my_MCMC = deepcopy(self.chain)
            weights = np.zeros(my_MCMC[:,0].size)+1

            for i in range(my_MCMC[0,:].size):
                if(i!=par):
                    stdev = np.std(self.chain[:,i]) # Narrow marginalization
                    sel = np.abs(my_MCMC[:,i]-self.bestfit[i]) < 1*stdev
                    weights = weights*stats.norm.pdf(my_MCMC[:,i],self.bestfit[i],stdev)

            n_sigma = 5
            par_arr = my_MCMC[:,par]
            xmean = np.sum(par_arr*weights)/np.sum(weights)
            xstd =  np.sqrt(np.sum((weights*(par_arr-xmean)**2))/np.sum(weights))
            xmin = xmean-n_sigma*xstd
            xmax = xmean+n_sigma*xstd

            #Perform a kernel density estimate on the data:

            xx = np.linspace(xmin,xmax,200)
            kernel = stats.gaussian_kde(par_arr,weights=weights)
            yy = kernel(xx)

            dx = xx[1]-xx[0]

            integral = np.sum(yy)*(dx)

            yy = yy/integral

            ymax = np.max(yy)

            hh1 = deepcopy(ymax)
            h_step = hh1/100
            cl1 = 0
            cl2 = 0

            while(cl1<0.68):
                hh1 = hh1-h_step
                cl1 = np.sum(yy[yy>hh1])*(dx)

            hh2 = deepcopy(hh1)
            while(cl2<0.95):
                hh2 = hh2-h_step
                cl2 = np.sum(yy[yy>hh2])*(dx)

            print("---> p.d.f for par: ","{:8} {} {:8} {} {:8}".format(xlabel," --> Integral: cl1 =", round(cl1,4), " ; cl2 =", round(cl2,4)))

            plt.plot(xx,yy,label="p.d.f")
            plt.axhline(y=hh1,color='orange', label=r"1$\sigma$ C.L.", alpha = 0.5, linestyle = "--")
            plt.axhline(y=hh2,color='blue', label=r"2$\sigma$ C.L.", alpha = 0.5, linestyle = "--")
            plt.xlabel(xlabel)
            plt.ylabel("p.d.f")

            xmean = np.sum(par_arr*weights)/np.sum(weights)
            xstd =  np.sqrt(np.sum((weights*(par_arr-xmean)**2))/np.sum(weights))
            xmin = xmean-3*xstd
            xmax = xmean+3*xstd
            plt.xlim([xmin,xmax])
            plt.legend()

        def contour2DMCMC(self,par1,par2,xlabel="par1",ylabel="par2"):
            """
            Plots the 2D contours for a given parameter pair from the chain.

            Args:
                self: An instance of EnergyRec.MCMC;
                par1: The 'x' corresponding parameter;
                par2: The 'y' corresponding parameter;
                xlabel: A label for par1;
                ylabel: A label for par2;
            """
            my_MCMC = deepcopy(self.chain)
            weights = np.zeros(my_MCMC[:,0].size)+1

            for i in range(my_MCMC[0,:].size):
                if(i!=par1 and i!=par2):
                    stdev = np.std(self.chain[:,i]) # Narrow marginalization
                    sel = np.abs(my_MCMC[:,i]-self.bestfit[i]) < 1*stdev
                    weights = weights*stats.norm.pdf(my_MCMC[:,i],self.bestfit[i],stdev)

            par1_arr = my_MCMC[:,par1]
            par2_arr = my_MCMC[:,par2]

            #Perform a kernel density estimate on the data:
            n_sigma = 5
            xmean = np.sum(par1_arr*weights)/np.sum(weights)
            xstd =  np.sqrt(np.sum((weights*(par1_arr-xmean)**2))/np.sum(weights))
            xmin = xmean-n_sigma*xstd
            xmax = xmean+n_sigma*xstd

            ymean = np.sum(par2_arr*weights)/np.sum(weights)
            ystd =  np.sqrt(np.sum((weights*(par2_arr-ymean)**2))/np.sum(weights))
            ymin = ymean-n_sigma*ystd
            ymax = ymean+n_sigma*ystd

            X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            values = np.vstack([par1_arr, par2_arr])
            kernel = stats.gaussian_kde(values,weights=weights)
            Z = np.reshape(kernel(positions).T, X.shape)

            dx = X[0,0]-X[1,0]
            dy = Y[0,0]-Y[0,1]

            integral = np.sum(Z)*(dx)*(dy)

            Z = Z/integral

            Zmax = np.max(Z)

            hh1 = deepcopy(Zmax)
            h_step = hh1/100
            cl1 = 0
            cl2 = 0

            while(cl1<0.68):
                hh1 = hh1-h_step
                cl1 = np.sum(Z[Z>hh1])*(dx)*(dy)

            hh2 = deepcopy(hh1)
            while(cl2<0.95):
                hh2 = hh2-h_step
                cl2 = np.sum(Z[Z>hh2])*(dx)*(dy)

            
            print("---> 2D contour for par: (","{} {} {} {} {:8} {} {:8}".format(xlabel,",",ylabel,") --> Integral: cl1 =", round(cl1,4), " ; cl2 =", round(cl2,4)))

            ax = plt.gca()

            ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,extent=[xmin, xmax, ymin, ymax], aspect='auto')

            CS = ax.contour(X, Y, Z, levels=[hh2,hh1])

            xmean = np.sum(par1_arr*weights)/np.sum(weights)
            xstd =  np.sqrt(np.sum((weights*(par1_arr-xmean)**2))/np.sum(weights))
            xmin = xmean-3*xstd
            xmax = xmean+3*xstd

            ymean = np.sum(par2_arr*weights)/np.sum(weights)
            ystd =  np.sqrt(np.sum((weights*(par2_arr-ymean)**2))/np.sum(weights))
            ymin = ymean-3*ystd
            ymax = ymean+3*ystd

            plt.xlim([xmin,xmax])
            plt.ylim([ymin,ymax])
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

        def plotContour_MCMC(self):
            """
            Produces the 1D and 2D plots from the MCMC.

            Args:
                self: An instance of EnergyRec.
            """
            print("* Performing MCMC statistics:")
            # 1D p.d.f.
            print("** 1D analysis:")
            my_labels = ["A",r"$\sigma$",r"$C_0$",r"$C_1$",r"$C_2$",r"$C_3$",r"$C_4$"]
            my_2D = [1,2,3,4,6,7,8,11,12,16]
            fig = plt.figure(figsize=(18,12))
            for i in range(7):
                plt.subplot(331+i)
                EnergyRec.MCMC.contour1DMCMC(self,i,my_labels[i])

            # 2D contours
            print("** 2D analysis:")
            fig = plt.figure(figsize=(23,15))
            counter = 0
            for i in range(2,7):
                for j in range(i+1,7):
                    plt.subplot(4,4,my_2D[counter])
                    EnergyRec.MCMC.contour2DMCMC(self,i,j,my_labels[i],my_labels[j])
                    counter = counter + 1
            
            #ax = plt.gca()
            #ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))

    class AERA:
        """
        A class with aera specific methods.
        """
        @staticmethod
        def aeraLDF(par_fit, Cs, evB, x,y):
            """
            The AERA 2d LDF.

            \f$ f(\vec{r})=A\left[\exp\left(\frac{-(\vec{r}+C_{1}\vec{e}_{\vec{v}\times\vec{B}}-\vec{r}_{{\rm core}})^{2}}{\sigma^{2}}\right) -C_{0}\exp\left(\frac{-(\vec{r}+C_{2}\vec{e}_{\vec{v}\times\vec{B}}-\vec{r}_{{\rm core}})^{2}}{\left(C_{3}e^{C_{4}\sigma}\right)^{2}}\right)\right].\f$

            Args:
                par_fit:   The parameters to be obtimized.
                Cs:   The parameters to be fixed.
                evB:   The versor in the direction of \f$\vec{v} \times \vec{B}\f$
                x:   The position of the antenna along the x axis (\f$\vec{v} \times \vec{B}\f$).
                y:   The position of the antenna along the y axis (\f$\vec{v}\times\vec{v} \times \vec{B}\f$).

            Returns:
                \f$ f \f$
            """
            A = par_fit[0]
            sigma = par_fit[1]
            rcore = EnergyRec.Shower.r_Core_proj[0:2]

            if Cs is None:
                C0 = par_fit[2]
                C1 = par_fit[3]
                C2 = par_fit[4]
                C3 = par_fit[5]
                C4 = par_fit[6]
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
                result = 1e50
            return result
        
        def aeraChi2(par_fit,Cs,self):
            """
            The chi2 for the AERA fit.

            The model for the uncertainty in the fluence is \f$ \sqrt{fluence} \f$

            Args:
                par:   The parameters to be obtimized.
                Cs:   The parameters to be fixed.


            Returns:
                \f$ \chi^2 \f$
            
            """
            Chi2 = 0.
            i=0
            
            EnergyRec.Shower.r_Core_proj = self.shower.r_Core_proj ## Using class definition as a global variable!!
            for ant in self.antenna:
                if ant.fluence <= self.f_thres:
                    continue

                if(self.bool_EarlyLate):
                    weight = ant.wEarlyLate
                else:
                    weight = 1

                x = ant.r_proj[0].value*weight
                y = ant.r_proj[1].value*weight
                f = ant.fluence/(weight**2)
                sigma = np.sqrt(ant.fluence)
                Chi2 = Chi2 + ((EnergyRec.AERA.aeraLDF(par_fit,Cs, np.array([1,0]), x, y)-f)/sigma)**2
                i = i + 1 
            return Chi2

        def aeraFit(self,filename,Cs):
            """
            Performs the fit using the AERA 2d LDF.
            
            If filename = "" (default) fits a given simulation.
                else it reads the file (antenna position (in shower plane) and fluences)
                and performs the fit.
            
            Args:
                filename:   File with antenna positions and fluences for a given shower inclination.
                Cs:   The parameters to be fixed.

            Fills:
                EnergyRec.bestfit.
            """ 
            if(filename==""):
                bestfit_out = "bestfit.out"
            
            else:
                bestfit_out = "bestfit_All.out"

            my_evB = np.array([1,0])
        
            # amplitude guess
            fluence_arr = np.array([ant.fluence for ant in self.antenna])
            init_A = np.max(fluence_arr)
        
            # core position guess
            #core_index = np.where(fluence_arr==np.max(fluence_arr))[0][0]
            #init_xCore =  0 #antpos_proj[core_index,0]
            #init_yCore =  0 #antpos_proj[core_index,1]
        
            # sigma guess
            init_sigma = 300
            
            Cs_aera = [0.5,-10,20,16,0.01]
            if Cs is None:
                par_fit = [init_A,init_sigma,Cs_aera[0],Cs_aera[1],Cs_aera[2],Cs_aera[3],Cs_aera[4]]
                res = sp.optimize.minimize(EnergyRec.AERA.aeraChi2,par_fit,args=(Cs,self),method='Nelder-Mead')
                resx = res.x

            else:
                par_fit = [init_A,init_sigma]
                res = sp.optimize.minimize(EnergyRec.AERA.aeraChi2,par_fit,args=(Cs,self),method='Nelder-Mead')
                resx = np.append(res.x,Cs)

            chi2min = EnergyRec.AERA.aeraChi2(resx,Cs,self)
            ndof = fluence_arr[fluence_arr>self.f_thres].size - res.x.size

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
            CR_input_Sradio=open("CR_input_Sradio.out", 'a')

            
            A=resx[0]
            sigma=resx[1]
            sin2Alpha = 1-np.dot(self.shower.ev,self.shower.eB)**2.
            Cs = np.array([resx[2],resx[3],resx[4],resx[5],resx[6]])
            Sradio = (A*np.pi/sin2Alpha)*(sigma**2. - Cs[0]*(Cs[3]**2.)*np.exp(2*Cs[4]*sigma))

            if(filename==""):
                print(str(resx)[1:-1],Sradio,file=bestfit)
                
            else:
                print(str(resx)[1:-1],file=bestfit)
            
            #print(self.shower.thetaCR,self.shower.phiCR,self.shower.ECR,Sradio,file=CR_input_Sradio)
            print(Sradio,file=CR_input_Sradio)

            bestfit.close()
            CR_input_Sradio.close()
            
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
            
            A = self.bestfit[0]
            sigma = self.bestfit[1]
            rcore = self.shower.r_Core_proj[0:2]
            sin2Alpha = (1-np.dot(self.shower.ev,self.shower.eB)**2.)
        
            Cs = np.array([self.bestfit[2],self.bestfit[3],self.bestfit[4],self.bestfit[5],self.bestfit[6]])
            Sradio = (A*np.pi/sin2Alpha)*(sigma**2. - Cs[0]*(Cs[3]**2.)*np.exp(2*Cs[4]*sigma))
            print('S_radio=',round(Sradio,2))

            par = [A,sigma,Cs[0],Cs[1],Cs[2],Cs[3],Cs[4]]

            fluence_arr = np.array([ant.fluence for ant in self.antenna])
            if(self.bool_EarlyLate):
                weight =  np.array([ant.wEarlyLate for ant in self.antenna])
            else:
                weight = np.full(len(fluence_arr),1)
            sel = np.where(fluence_arr>0)
            weight = weight[sel]
            fluence_arr=fluence_arr[sel]/(weight**2)

            r_proj = np.array([ant.r_proj for ant in self.antenna])
            x_proj = r_proj[:,0][sel]*weight
            y_proj = r_proj[:,1][sel]*weight
            
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

            for i in range(yy.size):
                for j in range(xx.size):
                    Z[i,j] = EnergyRec.AERA.aeraLDF(par, None, my_evB, X[i,j], Y[i,j]) # evaluation of the function on the grid
                
            fig = plt.figure(figsize=[14,5])
            plt.subplot(121)
            im = plt.imshow(Z,cmap='viridis',origin = 'lower', extent=[minXAxis,maxXAxis,minYAxis,maxYAxis]) # drawing the function

            plt.scatter(x_proj, y_proj, c=fluence_arr, cmap='viridis', s = 100, edgecolors=(1,1,1,0.2))
            plt.clim(np.min([np.min(Z),np.min(fluence_arr)]), np.max([np.max(Z),np.max(fluence_arr)]))
            plt.colorbar().ax.set_ylabel(r"energy fluence in eV/m$^2$")

            plt.xlabel(r'distante along $\vec{v}\times\vec{B}$ (in m)')
            plt.ylabel(r'distante along $\vec{v}\times\vec{v}\times\vec{B}$ (in m)')

            plt.plot(rcore[0], rcore[1],'w*')

            plt.subplot(122)
            plt.scatter(x_proj, y_proj, c=fluence_arr, cmap='viridis', s = 100, edgecolors=(1,1,1,0.2))
            plt.xlabel(r'distante along $\vec{v}\times\vec{B}$ (in m)')
            plt.ylabel(r'distante along $\vec{v}\times\vec{v}\times\vec{B}$ (in m)')
            plt.colorbar().ax.set_ylabel(r"energy fluence in eV/m$^2$")
            plt.xlim(minXAxis,maxXAxis)
            plt.ylim(minYAxis,maxYAxis)
            
            # 1D LDF
            temp_x = x_proj - rcore[0]
            temp_y = y_proj - rcore[1]

            temp_dist = np.sqrt(temp_x*temp_x+temp_y*temp_y)
            
            fig_ldf = plt.figure(figsize=[14,5])
            plt.subplot(121)
            plt.errorbar(temp_dist,fluence_arr,yerr=np.sqrt(fluence_arr),fmt='.')
            plt.xlabel("Distance from core in m")
            plt.ylabel(r"Fluence in eV/m$^2$")
            plt.gca().set_yscale('log')
                
            residual = np.zeros(fluence_arr.size)
            
            for i in range(fluence_arr.size):
                residual[i] = (fluence_arr[i] - EnergyRec.AERA.aeraLDF(par, None,my_evB, x_proj[i], y_proj[i]))/np.sqrt(fluence_arr[i])


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
        def a_ratio(r, d_Xmax, par, rho_max = 0.4):
            """
            Evaluates the charge-excess to geomagnetic ratio.
            
            Args:
                r:   Antenna position in the shower plane;
                d_Xmax:  Distance from core to shower maximum in meters;
                rho_max:    Air density at shower maximum.

            Retuns:
                a_ratio: The charge-excess to geomagnetic ratio.
            """ 
            rho_mean = 0.4
            return par[0]*(r/d_Xmax)*np.exp(r/par[1])*(np.exp((rho_max-rho_mean)/par[2])-par[3])

        @staticmethod
        def f_par_geo(f_vB, phi, alpha, r, d_Xmax, par, rho_max = 0.4):
            """
            Evaluates the parametrized geomagnetic fluence.
            
            Args:
                f_vB:   Fluence in the v times B direction;
                phi:  The angle between the antenna position and the v times B direciton;
                alpha:    The geomagnetic angle.
                r:   Antenna position in the shower plane;
                d_Xmax:  Distance from core to shower maximum in meters;
                rho_max:    Air density at shower maximum.

            Retuns:
                f_par_geo: The parametrized geomagnetic fluence.
            """
            sqrta = np.sqrt(EnergyRec.SymFit.a_ratio(r, d_Xmax, par, rho_max))
            cos_sin_ratio = np.cos(phi)/np.abs(np.sin(alpha))
            return f_vB/((1+cos_sin_ratio*sqrta)**2)
        
        @staticmethod
        def rho(r, e_vec):
            """
            Evaluates the air density at a given position.
            
            Args:
                r:   The distance to the position in meters;
                e_vec:  The direction of the position (unitary vector);

            Retuns:
                rho: The air density.
            """

            height = np.dot(r*e_vec,np.array([0,0,1]))/1000 # from m to km

            H = 10.4 # in km
            rho_0 = 1.225 # in kg/m^3
            site_height = 2.8 # in km

            return rho_0*np.exp(-(site_height+height)/H)

        
        @staticmethod
        def a_ratio_chi2(par,fluence_geo, fluence_ce,alpha, r, d_Xmax, rho_Xmax):
            """
            Chi2 for the a_ratio fit.
            
            Args:
                fluence_geo:   An array with the geomagnetic fluences;
                fluence_ce:  An array with the charge excess fluences;
                alpha: An array with the geomagnetic angles;
                r: An array with the antenna distances to the core in the shower plane;
                d_Xmax: The distance from the core to the Xmax;
                rho_Xmax: The atmospheric density in the Xmax.

            Retuns:
                Chi2: The Chi2 value.
            """
            sel = np.where(fluence_geo > 0)

            a_arr = (np.sin(alpha[sel])**2)*fluence_ce[sel]/fluence_geo[sel]

            Chi2 = 0
            for i in range(fluence_geo[sel].size):
                a_theo = EnergyRec.SymFit.a_ratio(r[sel][i], d_Xmax[sel][i], par, rho_Xmax[sel][i])
                #if a_arr[i] < 1:
                Chi2 = Chi2 + (a_arr[i] -a_theo)**2

            return Chi2

        @staticmethod
        def a_ratio_fit(fluence_geo, fluence_ce,alpha, r, d_Xmax, rho_max):
            """
            Fits the a_ratio.
            
            Args:
                fluence_geo:   An array with the geomagnetic fluences;
                fluence_ce:  An array with the charge excess fluences;
                alpha: An array with the geomagnetic angles;
                r: An array with the antenna distances to the core in the shower plane;
                d_Xmax: An array with the distances to shower maximum;
                rho_max: An array with the densities at shower maximum.

            Retuns:
                bestfit: The bestfit parameters array.
            """
            par = [0.373, 762.6, 0.1490, 0.189]
            res = sp.optimize.minimize(EnergyRec.SymFit.a_ratio_chi2,par,args=(fluence_geo, fluence_ce,alpha, r, d_Xmax, rho_max),method='Nelder-Mead')
            return res.x
        
        @staticmethod
        def SymLDF(par,r):
            """
            The symmetric ldf to be fit to the fluence_par data.
            \f$ f_{ABCD}(r) = A.exp\left[-B.r-C.r^2-D.r^3\right] \f$
            
            Args:
                par: The parameter array;
                r:   The distance to the axis.

            Retuns:
                LDF: The ldf value at distance r.
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
            
            Args:
                par: The parameter array;
                r:   The distance to the axis;
                fluence_par: The array with the symmetrized signal.

            Retuns:
                Chi2: The Chi2 value.
            """
            sel = np.where(fluence_par > 0)
            f_par = fluence_par[sel]
            Chi2 = 0
            for i in range(f_par.size):
                LDF = EnergyRec.SymFit.SymLDF(par,r[sel][i])
                #if a_arr[i] < 1:
                Chi2 = Chi2 + (f_par[i] -LDF)**2

            return Chi2


        @staticmethod
        def SymLDF_fit(r,fluence_par):
            """
            Fits the symmetric LDF to the fluence_par data.
            
            Args:
                r:   The distance to the axis;
                fluence_par: The array with the symmetrized signal.

            Retuns:
                bestfit: The bestfit parameters array.
            """

            # Estimating the parameters
            r0 = np.min(r)
            i_d0 = np.where(r==r0)[0][0]
            f0 = fluence_par[i_d0]

            r1 =np.max(r[r<300])
            i_d1 = np.where(r==r1)[0][0]
            f1 = fluence_par[i_d1]

            r2 =np.max(r[r<600])
            i_d2 = np.where(r==r2)[0][0]
            f2 = fluence_par[i_d2]

            r3 =np.max(r[r<900])
            i_d3 = np.where(r==r3)[0][0]
            f3 = fluence_par[i_d3]

            a = np.array([
                [1,-r0,-r0**2,-r0**3],
                [1,-r1,-r1**2,-r1**3],
                [1,-r2,-r2**2,-r2**3],
                [1,-r3,-r3**2,-r3**3]])
            b = np.array([np.log(f0),np.log(f1),np.log(f2),np.log(f3)])

            par = np.linalg.solve(a, b)

            par[0] = np.exp(par[0])
            res = sp.optimize.minimize(EnergyRec.SymFit.LDF_chi2,par,args=(r, fluence_par),method='Nelder-Mead')
            return res.x

        @staticmethod
        def Sradio_geo(par,ldf_par,alpha,rho_Xmax):
            """
            The radiation energy corrected for the scaling of the emission strength with the geomagnetic angle and the atmospheric density.
            
            Args:
                par: The free parameters of the correction;
                ldf_par: The parameters to be used in the symmetric LDF;
                alpha: The geomagnetic angle;
                rho_Xmax: The density in the X_max.

            Retuns:
                S_radio_geo: The corrected radiation energy
            """
            E_rad = 2*np.pi*sp.integrate.quad(lambda r: r*EnergyRec.SymFit.SymLDF(ldf_par,r), 0, 2000)[0]
            sin2alpha = np.sin(alpha)**2.

            p0 = par[0]
            p1 = par[1]
            rho_mean = 0.648
            den = sin2alpha*(1 - p0 + p0*np.exp(p1*(rho_Xmax-rho_mean)))

            return E_rad/den

        @staticmethod
        def Sradio_mod(par,E):
            """
            The model for the relation between S_radio and the energy.
            
            Args:
                par: The free parameters of the model;
                E: The energy of the event in EeV.

            Retuns:
                S_radio_mod: The model energy.
            """

            S_19 = par[0]
            gamma = par[1]

            return S_19*(E/10)**gamma

        @staticmethod
        def Chi2_joint_S(par,ldf_par_arr,alpha_arr,rho_Xmax_arr,E_arr):
            """
            The chi2 for the joint fit os Sradio_geo and Sradio_mod.
            
            Args:
                par: The full parameter array;
                ldf_par_arr: The array with the ldf_par for each simulation;
                alpha_arr: The array with the geomagnetic angles of each simulation;
                rho_Xmax: The array with the density at Xmax of each simulation;
                E_arr: The array with the energies of each simulation in GeV;

            Retuns:
                Chi2: The Chi2 value.
            """
            
            Chi2 = 0 
            for i in range(len(ldf_par_arr)):
                S_geo = EnergyRec.SymFit.Sradio_geo(par[0:2],ldf_par_arr[i],alpha_arr[i],rho_Xmax_arr[i])
                S_mod = EnergyRec.SymFit.Sradio_mod(par[2:4],E_arr[i])

                Chi2 = Chi2 + (S_geo - S_mod)**2
            
            return Chi2

        @staticmethod
        def joint_S_fit(ldf_par_arr,alpha_arr,rho_Xmax_arr,E_arr):
            """
            Performs the joint fit of the S_radio.
            
            Args:
                ldf_par_arr: The array with the ldf_par for each simulation;
                alpha_arr: The array with the geomagnetic angles of each simulation;
                rho_Xmax: The array with the density at Xmax of each simulation;
                E_arr: The array with the energies of each simulation in GeV;

            Retuns:
                bestfit: The bestfit array.
            """

            p0 = 0.394
            p1 = -2.370 #m^3/kg
            S_19 = 1.408 #GeV
            gamma = 1.995
            par = [p0,p1,S_19,gamma]

            res = sp.optimize.minimize(EnergyRec.SymFit.Chi2_joint_S,par,args=(ldf_par_arr,alpha_arr,rho_Xmax_arr,E_arr),method='Nelder-Mead')
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
print("\n")