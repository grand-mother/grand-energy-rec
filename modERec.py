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
    bool_EarlyLate = False
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
    ## The number of the simulation to be reconstructed.
    sim_number = None
    ## An instance of the class Antenna
    antenna = None
    ## An instance of the class Shower
    shower = None
    ## An instance of the class MCMC
    mcmc = None
    ## The array with the antenna fluences
    fluence_arr = None
    ## The array with the geomagnetic antenna fluences
    fluence_geo = None
    ## The array with the charge excess antenna fluences
    fluence_ce = None
    ## The position of the antennas
    r_ant = None
    ## The bestfit values of the parameters
    bestfit = None
    ## The shower imported using the standard grand package
    GRANDshower = None

    def __init__(self,sim_dir,sim_number):
        """
        The default init function for the class EnergyRec.
    
        Args:
            self: An instance of EnergyRec;
            sim_dir: The path to the simulation directory;
            sim_number: The number of the simulation to be reconstructed.
        """
        self.sim_dir = sim_dir
        self.sim_number = sim_number
        self.antenna = self.Antenna(self.nu_low,self.nu_high,self.SNR_thres)
        self.shower = self.Shower(self.sim_dir,self.sim_number)
        self.mcmc = self.MCMC()

        if not Path(self.sim_dir).is_dir():
            print("ERROR: directory ",self.sim_dir," not found!")
            raise SystemExit("Stop right there!")

        self.GRANDshower = CoreasShower.load(self.sim_dir)

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

    def shower_plane(self):
            """
            Evaluates the shower plane and reads the theta from the simulation.

            Args:
                self: An instance of EnergyRec.Shower.

            Fills:
                EnergyRec.Shower.ev, EnergyRec.Shower.eB, EnergyRec.Shower.evB, EnergyRec.Shower.evvB and EnergyRec.Shower.thetaCR (as integer in deg).
            
            """
            
            eB = (self.GRANDshower.geomagnet/self.GRANDshower.geomagnet.norm())
            ev = self.GRANDshower.maximum - self.GRANDshower.core
            ev /= -ev.norm()
        
            evB = ev.cross(eB)
            evB /= evB.norm()
            evvB = ev.cross(evB)
            evvB /= evvB.norm()
            
            if(self.bool_plot):
                fig = plt.figure(figsize=(10,7))
                ax = fig.add_subplot(111, projection='3d')
                ax.quiver(0,0,0, ev.x, ev.y, ev.z,color='r', label=r'$\vec{v}$')
                ax.quiver(0,0,0, eB.x, eB.y, eB.z,color='b', label=r'$\vec{B}$')
                ax.quiver(0,0,0, evB.x, evB.y, evB.z,color='k', label=r'$\vec{v}\times\vec{B}$')
                ax.quiver(0,0,0, evvB.x, evvB.y, evvB.z,color='gray', label=r'$\vec{v}\times\vec{v}\times\vec{B}$')
                ax.set_xlim([-0.9,0.9])
                ax.set_ylim([-0.9,0.9])
                ax.set_zlim([-0.9,0.9])
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                ax.legend()

            self.shower.ev = ev.xyz
            self.shower.eB = eB.xyz
            self.shower.evB = evB.xyz
            self.shower.evvB = evvB.xyz
            self.shower.thetaCR = self.GRANDshower.zenith
            self.shower.phiCR = self.GRANDshower.azimuth

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
            
            plt.show()
        
    #self.traces = traces
    
    def process_antenna(self, id):
        """
        Process a given antenna for inspection.
        
        For a given initialized antenna, performs offset and cut, fft, trace recover, hilbert envelope and computes the fluence.

        Useful if bool_plot = True.

        Args:
            self: An instance of EnergyRec.Antenna.
        
        """

        if(id<len(self.GRANDshower.fields)):
            time = self.GRANDshower.fields[id].electric.t.to("ns").value
            Ex = self.GRANDshower.fields[id].electric.E.x.to("V/m").value
            Ey = self.GRANDshower.fields[id].electric.E.y.to("V/m").value
            Ez = self.GRANDshower.fields[id].electric.E.z.to("V/m").value
        
        self.antenna.traces  = np.c_[time,Ex,Ey,Ez]

        # Check if peak is within the threshold range
        peak = np.max(np.abs(self.antenna.traces[:,1:4]))
        if(peak < self.thres_low or peak > self.thres_high):
            self.antenna.fluence = -1
            return
            
        self.antenna.offset_and_cut()
        self.antenna.fft_filter()
        self.antenna.trace_recover()

        # Check if peak is within the threshold range after offset, cut and trace recover.
        if(np.max(np.abs(self.antenna.traces[:,1:4]))<self.thres_low):
            self.antenna.fluence = -1
            return
        else:
            self.antenna.hilbert_envelope()
            self.antenna.compute_fluence(None,None)
                
    
    def Eval_fluences(self):
        """
        Evaluates the fluence for a set os antennas.
            
        It has two thresholds for the fluence: thres_low and thres_high.

        Args:
            self: An instance of EnergyRec.

        Fills:
            EnergyRec.fluence_arr.
        
        """

        n_ant = len(self.GRANDshower.fields)
        fluence_arr = np.zeros(n_ant)
    
        step = int(n_ant/10)
        counter = 0
    
        print("* Evaluating the fluences:")
        print("--> 0 % complete;")
        for ant in range(n_ant):
            #Read traces or voltages
            if ((ant+1)%step == 0):
                print("-->",int((ant+1)/(10*step)*100),"% complete;")

            self.process_antenna(ant)
            fluence_arr[ant] = self.antenna.fluence
       
        print("\n")
        self.fluence_arr = fluence_arr

    def Eval_geo_ce_fluences(self):
        """
        Evaluates the geomagnetic and charge excess fluences for a set os antennas.
            
        It has two thresholds for the fluence: thres_low and thres_high.

        The early-late correction is performed automaticaly in this function.

        Args:
            self: An instance of EnergyRec.

        Fills:
            EnergyRec.fluence_geo;
            EnergyRec.fluence_ce.
        
        """
        my_dir = self.sim_dir+'/SIM'+str(self.sim_number).zfill(6)+'_coreas/'
        if not Path(my_dir).is_dir():
            print("ERROR: directory ",my_dir," not found!")
            raise SystemExit("Stop right there!")
            
        output_list = glob.glob(my_dir+'raw_ant*.dat')
    
        nbe = np.zeros(len(output_list))
        fluence_arr = np.zeros(len(output_list))
        fluence_geo = np.zeros(len(output_list))
        fluence_ce = np.zeros(len(output_list))

    
        step = int(len(output_list)/10)
        counter = 0

        self.read_antpos()
        self.antenna_projection()

        if self.shower.R0_R is None:
            self.early_late()

        print("* Evaluating the fluences:")
        print("--> 0 % complete;")
        for output in output_list: # loop over all antennas
            #Read traces or voltages
            index = int(output.split('/raw_ant')[-1].split('.dat')[0])
            if ((counter+1)%step == 0):
                print("-->",int((counter+1)/(10*step)*100),"% complete;")

            counter = counter+1
            self.init_antenna(index)
            self.antenna.offset_and_cut()
            peak = np.max(np.abs(self.antenna.traces[:,1:4]))
            if(peak < self.thres_low or peak > self.thres_high):
                fluence_arr[index] = -1
                continue
    
            self.antenna.fft_filter()
            self.antenna.trace_recover()
    
            if(np.max(np.abs(self.antenna.traces[:,1:4]))<self.thres_low):
                fluence_arr[index] = -1
                fluence_geo[index] = -1
                fluence_ce[index] = -1
                continue
            else:
                self.antenna.hilbert_envelope()
                self.shower.shower_plane()
                self.antenna.compute_fluence(self.shower.evB,self.shower.evvB)
                if(self.antenna.fluence < 0):
                    fluence_arr[index] = -1
                    fluence_geo[index] = -1
                    fluence_ce[index] = -1
                    continue

                cosPhi = self.shower.r_proj[index][0]/np.linalg.norm(self.shower.r_proj[index])
                sinPhi = np.sqrt(1-cosPhi*cosPhi)
                my_fluence_geo = np.sqrt(self.antenna.fluence_evB)-(cosPhi/sinPhi)*np.sqrt(self.antenna.fluence_evvB)
                fluence_geo[index] = my_fluence_geo*my_fluence_geo/(self.shower.R0_R[index]*self.shower.R0_R[index])
                fluence_ce[index] = self.antenna.fluence_evvB/(sinPhi*sinPhi)/(self.shower.R0_R[index]*self.shower.R0_R[index])
                fluence_arr[index] = self.antenna.fluence/(self.shower.R0_R[index]*self.shower.R0_R[index])
                self.shower.r_proj[index] = self.shower.r_proj[index] * self.shower.R0_R[index]
       
        print("\n")
        self.fluence_arr = fluence_arr 
        self.fluence_geo = fluence_geo
        self.fluence_ce = fluence_ce

    def read_antpos(self):
        """
        Reads the position of the antennas.

        Args:
            self: An instance of EnergyRec.

        Fills:
            EnergyRec.r_ant.
        
        """
 
        n_ant = len(self.GRANDshower.fields)

        r_ant = np.zeros((n_ant,3))
        for key, value in self.GRANDshower.fields.items():
            r_ant[key]=value.electric.r.xyz.value
    
    
        if(self.bool_plot):
            fig= plt.figure(figsize=(10,7))
            ax = plt.gca()
    
            plt.scatter(r_ant[:,0][self.fluence_arr>=0], r_ant[:,1][self.fluence_arr>=0], c=self.fluence_arr[self.fluence_arr>=0], cmap='viridis')
    
            plt.xlabel("x (in m)")
            plt.ylabel("y (in m)")
            plt.colorbar().ax.set_ylabel(r"Energy fluence (eV/m$^2$)")
            plt.show()
            
        self.r_ant = r_ant

    
    def signal_output(self):
        """
        Prints the antena positions (in the shower plane) and fluences to a file.
        
        The filename has the structure fluence_ShowerPlane_THETACR.out' and is open with append option.

        Args:
            self: An instance of EnergyRec.
        """
        signal = np.c_[self.shower.r_proj[self.fluence_arr>0],self.fluence_arr[self.fluence_arr>0]]
    
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
            EnergyRec.Shower.r_proj;
            EnergyRec.Shower.traces_proj.
        
        """
        #The antenna projection
        n_ant = len(self.GRANDshower.fields)
        antpos_proj = np.zeros((n_ant,3))

        self.GRANDshower.localize(latitude=45.5 * u.deg, longitude=90.5 * u.deg)

        if self.shower.ev is None:
            self.shower_plane()

        r = Rotation.from_matrix(np.matrix([self.shower.evB,self.shower.evvB,self.shower.ev]).T)
        shower_frame = self.GRANDshower.frame.rotated(r)
        traces_proj = {}

        for key, value in self.GRANDshower.fields.items():
            r_ant = value.electric.r - self.GRANDshower.core
            position = self.GRANDshower.frame._replicate(r_ant, copy=False)
            antpos_proj[key] = position.transform_to(shower_frame).cartesian.xyz

            E = self.GRANDshower.fields[key].electric.E
            initial_trace = self.GRANDshower.frame._replicate(E, copy=False)
            traces_proj[key] = initial_trace.transform_to(shower_frame)

        core = self.GRANDshower.core
        position = self.GRANDshower.frame._replicate(core, copy=False)
        r_Core_proj = position.transform_to(shower_frame).cartesian.xyz

        self.shower.r_Core_proj = r_Core_proj.value
        self.shower.r_proj = antpos_proj
        self.shower.traces_proj = traces_proj

    def model_fit(self,filename=""):
        """
        Performs the fit using a given model (set in the EnergyRec instance).
        
        If filename = "" (default) fits a given simulation.
            else it reads the file (antenna position (in shower plane) and fluences)
            and performs the fit.

        Args:
            self: An instance of EnergyRec.
            filename:   File with antenna positions and fluences for a given shower inclination.

        Fills:
            EnergyRec.bestfit.
        
        """ 

        print("* Model fit:")
        if(filename==""):
            self.read_antpos()
            self.shower_plane();
            self.shower_projection();            

            # Early-late effect
            if self.bool_EarlyLate:
                if self.shower.R0_R is None:
                    self.early_late()
                    i = 0
                    for weight in self.shower.R0_R:
                        self.fluence_arr[i] = self.fluence_arr[i]/(weight**2)
                        self.shower.r_proj[i] = self.shower.r_proj[i]*weight
                        i = i+1

                    print("--> Early-late correction applied")
                else:
                    print("--> Early-late already applied")
            elif self.shower.R0_R is not None:
                print("--> Early-late correction already applied but instance.bool_EarlyLate != True")
                exit

        elif self.fluence_arr is not None:
            print("--> fluence_arr == None. instance.Eval_fluences() has to be run!")
            exit
        
        else:
            if not Path(filename).is_file():
                print("ERROR: file ",filename," not found!")
                raise SystemExit("Stop right there!")
            datafile = open(filename,'r')
            pos_fluence = np.loadtxt(datafile)
            datafile.close()
            self.shower.r_proj = np.c_[pos_fluence[:,0],pos_fluence[:,1]]
            self.fluence_arr = pos_fluence[:,2]

        EnergyRec.AERA.aeraFit(self,filename)
        print("\n")


    def early_late(self):
        """
        Evaluates the early-late correction factor.

        Args:
            self: An instance of EnergyRec.

        Fills:
            EnergyRec.Shower.R0_R.

        """
        rCore = self.GRANDshower.core.xyz.value
        rXmax = self.GRANDshower.maximum.xyz.value - rCore

        n_ant = len(self.GRANDshower.fields)
        r_ant = np.zeros((n_ant,3))

        for key, value in self.GRANDshower.fields.items():
            r_ant[key] = value.electric.r.xyz.value - rCore
        
        R = np.linalg.norm(r_ant - rXmax,axis=1)
        R_0 = np.linalg.norm(rXmax)
        self.shower.R0_R = R_0/R
        self.shower.d_Xmax = np.linalg.norm(rXmax)


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
        ## The path to the simulation directory.
        sim_dir = None
        ## The number of the simulation to be reconstructed.
        sim_number = None
        ## Toggles the plots on and off.
        bool_plot = False
        ## The ratio of distances used in the early-late correction.
        R0_R = None
        ## X position of the shower core.
        CoreX = None
        ## Y position of the shower core.
        CoreY = None
        ## Distance to Xmax.
        d_Xmax = None

        def __init__(self,sim_dir,sim_number):
            """
            The default init function for the class Shower.
    
            Args:
                sim_dir: The path to the simulation directory.
                sim_number: The number of the simulation to be reconstructed.
            """
            self.sim_dir = sim_dir
            self.sim_number = sim_number


    class Antenna:
        """
        A class for the antenna signal processing.
    
        It has tools for the FFT, trace_recover and fluence evaluation.
        """
        ## The lower frequency of the signal filter.
        nu_low = None
        ## The upper frequency of the signal filter.
        nu_high = None
        ## The signal to noise ratio threshold.
        SNR_thres = None
        ## Toggles the plots on and off.
        bool_plot = False
        ## Toggles the trace offset on and off.
        bool_offset = None
        ## Toggles the trace cut in time on and off.
        bool_cut = None
        ## Stores the fluence in the v times B direction.
        fluence_evB = None
        ## Stores the fluence in the v times v times B direction.
        fluence_evvB = None
        ## Stores the fluence for a given antenna.
        fluence = None
        ## Stores the FFT of the traces.
        FFTtraces = None
        ## Stores the reconstructed traces.
        tracesRec = None
        ## Stores the Hilbert envelope of the polarizations.
        hilbert_pol = None
        ## Stores the Hilbert envelope of the total trace.
        hilbert = None
        ## Stoeres the raw traces.
        traces = None

        def __init__(self,nu_low, nu_high, SNR_thres):
            """
            The default init function for the class Antenna.
    
            Args:
                nu_low:   The lower frequency of the signal filter.
                nu_high:   The upper frequency of the signal filter.
                SNR_thres:   The signal to noise ratio threshold.
            """
            self.nu_low = nu_low
            self.nu_high = nu_high
            self.SNR_thres = SNR_thres
            self.bool_offset = 1
            self.bool_cut = 1
            self.fluence = 0

        

        def fft_filter(self, col='r'):
            """
            Evaluates the FFT of the signal.

            A filter is applied with width given by instance.antenna.nu_high and instance.antenna.nu_low.

            Args:
                self: An instance of EnergyRec.Antenna.
                col: A string to change the color of the plot.

            Fills:
                EnergyRec.Antenna.FFTtraces.         

            """
            time_arr = self.traces[:,0]
            trace1 = self.traces[:,1]
            trace2 = self.traces[:,2]
            trace3 = self.traces[:,3]
            # Number of sample points
            N = time_arr.size
            
            # sample spacing
            sampling_rate = 1/((time_arr[1]-time_arr[0])*1.e-9) # uniform sampling rate (convert time from ns to seconds)
            T = 1/sampling_rate
            yf1 = fft(trace1)
            yf2 = fft(trace2)
            yf3 = fft(trace3)
            xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
            
            nu_low = self.nu_low * 10**6 # from MHz to Hz
            nu_high = self.nu_high * 10**6 # from MHz to Hz
            for i in range(xf.size):
                if(xf[i]<nu_low or xf[i]>nu_high):
                    yf1[i]=0
                    yf1[-i]=0 # negative frequencies are backordered (0 1 2 3 4 -4 -3 -2 -1)
                    yf2[i]=0
                    yf2[-i]=0
                    yf3[i]=0
                    yf3[-i]=0
            
            if(self.bool_plot):
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
                    
            self.FFTtraces = np.c_[yf1, yf2, yf3]
            
        def trace_recover(self, col='r'):
            """
            Reconstructs the trace after the FFT and filter.

            Args:
                self: An instance of EnergyRec.Antenna.
                col:   A string to change the color of the plot.

            Fills:
                EnergyRec.Antenna.tracesRec.
                        
            """
            yy1 = ifft(self.FFTtraces[:,0])
            yy2 = ifft(self.FFTtraces[:,1])
            yy3 = ifft(self.FFTtraces[:,2])
           
            xx = self.traces[:,0]-np.min(self.traces[:,0])
            
            if(self.bool_plot):
                fig = plt.figure(figsize=(15,3))
                fig.suptitle('Reconstructed traces', fontsize=16,y=1)
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
            
            self.tracesRec = np.c_[yy1, yy2, yy3]
            
        def hilbert_envelope(self, col='r'):
            """
            Evaluates the hilbert envelope of the recunstructed traces.

            \f$\mathcal{H}\{f(x)\}:=H(x)=\frac{1}{\pi}{\rm p.v.}\int_{-\infty}^\infty \frac{f(u)}{u-x}{\rm d}u\f$
            
            Args:
                self: An instance of EnergyRec.Antenna.
                col:   A string to change the color of the plot.

            Fills:
                EnergyRec.Antenna.hilbert.
            """
            tt = self.traces[:,0]-np.min(self.traces[:,0])
            envelope1 = hilbert(self.tracesRec[:,0].real)
            envelope2 = hilbert(self.tracesRec[:,1].real)
            envelope3 = hilbert(self.tracesRec[:,2].real)
        
            if(self.bool_plot):    
                fig = plt.figure(figsize=(15,3))
                fig.suptitle('Hilbert Envelopes', fontsize=16,y=1)
                plt.subplot(131)
                plt.plot(tt,self.tracesRec[:,0].real, 'r', label='signal')
                plt.plot(tt, np.abs(envelope1), label='envelope')
                #plt.xlabel("time in ns")
                plt.ylabel("signal in V/m")
                ax = plt.gca()
                ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        
               
                plt.subplot(132)
                plt.plot(tt,self.tracesRec[:,1].real, 'b', label='signal')
                plt.plot(tt, np.abs(envelope2), label='envelope')
                plt.xlabel("time in ns")
                #plt.ylabel("signal in V/m")
                ax = plt.gca()
                ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
        
                plt.subplot(133)
                plt.plot(tt,self.tracesRec[:,2].real, 'k', label='signal')
                plt.plot(tt, np.abs(envelope3), label='envelope')
                #plt.xlabel("time in ns")
                #plt.ylabel("signal in V/m")
                ax = plt.gca()
                ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
                
                plt.show()
        
            self.hilbert = np.sqrt(envelope1**2 + envelope2**2 + envelope3**2)
            self.hilbert_pol = np.c_[envelope1,envelope2,envelope3]
            
        def compute_fluence(self,evB,evvB):
            """
            Computes the fluence for a given antenna.
            
            \f$ f = \epsilon_0 c\left(\Delta t \sum_{t_1}^{t_2} \left| \vec{E}(t_i)\right|^2 - \Delta t \frac{t_2-t_1}{t_4-t_3} \sum_{t_3}^{t_4} \left| \vec{E}(t_i)\right|^2 \right) \f$
            \n \n It has a threshold for the SNR set by EnergyRec.SNR_thres.

            Args:
                self: An instance of EnergyRec.Antenna.
                evB: versor along \f$ \vec{v} \times \vec{B} \f$
                evvB: versor along \f$ \vec{v} \times \vec{v} \times \vec{B} \f$

            Fills:
                EnergyRec.Antenna.fluence;
                EnergyRec.Antenna.fluence_evB;
                EnergyRec.Antenna.fluence_evvB.
            
            """
            tt = self.traces[:,0]-np.min(self.traces[:,0])
            delta_tt = (tt[1]-tt[0])*1e-9 # convert from ns to seconds
            
            envelope2 = self.hilbert**2.
        
            tmean_index = np.where(envelope2==np.max(envelope2))[0][0] # Index of the hilbert envelope maximum
            
            if(tt[-1]-tt[tmean_index] < 150):
                #print("Peak too close to the end of the signal. Skipping")
                self.fluence = -2e-5
                return
            
            time_100ns_index = np.where(tt<=100)[0][-1] # Index of 100 ns bin
        
            if(tmean_index<time_100ns_index):
                #tmean_index = time_100ns_index
                #print("Peak too close to the beginning of the signal. Skipping")
                self.fluence = -1e-5
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
        
            if(self.bool_plot):
                plt.plot(tt,np.abs(envelope2))
                plt.xlabel("time in nanoseconds")
                plt.ylabel(r"E$^2$ in V$^2$/m$^2$")
            
            epsilon0 = 8.8541878128e-12 # Coulomb^2 * N^-1 * m^-2
            c = 299792458 # m * s^-1
            
            Joule_to_eV = 1/1.602176565e-19
            
            SNR = np.sqrt(signal/bkg)
            
            if(SNR>self.SNR_thres):
                my_fluence = epsilon0*c* (signal - bkg)*Joule_to_eV # eV * m^-2
        
            else:
                self.fluence = -1
                self.fluence_evB = -1
                self.fluence_evB = -1
                return

            self.fluence = my_fluence

            if not evB is None:
                e_pol = np.c_[np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]

                E_vB2= (self.hilbert_pol[:,0]*np.dot(e_pol[0],evB))**2
                E_vvB2 = (self.hilbert_pol[:,0]*np.dot(e_pol[0],evvB))**2
                for pol in range(2):
                    E_vB2 = E_vB2 + (self.hilbert_pol[:,pol+1]*np.dot(e_pol[pol+1],evB))**2
                    E_vvB2 = E_vvB2 + (self.hilbert_pol[:,pol+1]*np.dot(e_pol[pol+1],evvB))**2

                signal_evB = np.sum(np.abs(E_vB2)[(tt>=t1) & (tt<=t2)])*delta_tt
                self.fluence_evB = (signal_evB-bkg)*epsilon0*c*Joule_to_eV
                signal_evvB = np.sum(np.abs(E_vvB2)[(tt>=t1) & (tt<=t2)])*delta_tt
                self.fluence_evvB = (signal_evvB-bkg)*epsilon0*c*Joule_to_eV
            
        def offset_and_cut(self):
            """
            Performs cuts and offsets the traces.
            
            The offset prevents problems due to traces to close to the time origin.
            The cut, reduces the time window to speed up the code.

            Args:
                self: An instance of EnergyRec.Antenna.
            
            """
            if(self.bool_offset):
                traces0 = self.traces[:,0]
                deltat = traces0[1]-traces0[0]
                min_time = traces0[0]
            
                offset = int(100/deltat)
            
                extra_time = np.linspace(min_time-offset*deltat,min_time-1,offset)
                traces0 = np.insert(self.traces[:,0],0,extra_time)
                traces1 = np.insert(self.traces[:,1],0,np.zeros(offset))
                traces2 = np.insert(self.traces[:,2],0,np.zeros(offset))
                traces3 = np.insert(self.traces[:,3],0,np.zeros(offset))
        
                self.traces = np.c_[traces0,traces1,traces2,traces3]
            
            if(self.bool_cut):
                global_peak = np.max(np.abs(self.traces[:,1:4]))
                peak_index = np.where(np.abs(self.traces[:,1:4])==global_peak)[0][0]
                peak_time = self.traces[:,0][peak_index]
                sel = ((self.traces[:,0]>peak_time-1000) & (self.traces[:,0]<peak_time+1000))
            
                self.traces = self.traces[sel,0:4]

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
        def aeraLDF(par, evB, x,y):
            """
            The AERA 2d LDF.

            \f$ f(\vec{r})=A\left[\exp\left(\frac{-(\vec{r}+C_{1}\vec{e}_{\vec{v}\times\vec{B}}-\vec{r}_{{\rm core}})^{2}}{\sigma^{2}}\right) -C_{0}\exp\left(\frac{-(\vec{r}+C_{2}\vec{e}_{\vec{v}\times\vec{B}}-\vec{r}_{{\rm core}})^{2}}{\left(C_{3}e^{C_{4}\sigma}\right)^{2}}\right)\right].\f$

            Args:
                par:   The parameters to be obtimized.
                evB:   The versor in the direction of \f$\vec{v} \times \vec{B}\f$
                x:   The position of the antenna along the x axis (\f$\vec{v} \times \vec{B}\f$).
                y:   The position of the antenna along the y axis (\f$\vec{v}\times\vec{v} \times \vec{B}\f$).

            Returns:
                \f$ f \f$
            """
            A = par[0]
            sigma = par[1]
            rcore = EnergyRec.Shower.r_Core_proj[0:2]
        
            C0 = par[2]
            C1 = par[3]
            C2 = par[4]
            C3 = par[5]
            C4 = par[6]
            
            r = np.array([x,y])
            numA = r-C1*evB-rcore
            partA = np.exp(-(np.linalg.norm(numA)**2)/sigma**2)
            
            numB = r-C2*evB-rcore
            partB = np.exp(-(np.linalg.norm(numB)**2)/(C3*np.exp(C4*sigma))**2)
            
            result = A*(partA - C0*partB)
            if(A<0 or sigma<0 or C0<0 or C3<0 or C4<0):
                result = 1e50
            return result
        
        def aeraChi2(par,self):
            """
            The chi2 for the AERA fit.

            The model for the uncertainty in the fluence is \f$ \sqrt{fluence} \f$

            Args:
                par:   The parameters to be obtimized.

            Returns:
                \f$ \chi^2 \f$
            
            """
            Chi2 = 0.
            i=0
            
            EnergyRec.Shower.r_Core_proj = self.shower.r_Core_proj
            for ant in self.shower.r_proj:
                if self.fluence_arr[i] <= self.f_thres:
                    i = i + 1
                    continue
                sigma = np.sqrt(self.fluence_arr[i])
                #sigma=np.sqrt(np.max(self.fluence_arr))
                Chi2 = Chi2 + ((EnergyRec.AERA.aeraLDF(par, np.array([1,0]), ant[0], ant[1])-self.fluence_arr[i])/sigma)**2
                i = i + 1 
            return Chi2

        def aeraFit(self,filename=""):
            """
            Performs the fit using the AERA 2d LDF.
            
            If filename = "" (default) fits a given simulation.
                else it reads the file (antenna position (in shower plane) and fluences)
                and performs the fit.
            
            Args:
                filename:   File with antenna positions and fluences for a given shower inclination.

            Fills:
                EnergyRec.bestfit.
            """ 
            if(filename==""):
                bestfit_out = "bestfit.out"
            
            else:
                bestfit_out = "bestfit_All.out"

            my_evB = np.array([1,0])
        
            # amplitude guess
            init_A = np.max(self.fluence_arr)
        
            # core position guess
            #core_index = np.where(fluence_arr==np.max(fluence_arr))[0][0]
            #init_xCore =  0 #antpos_proj[core_index,0]
            #init_yCore =  0 #antpos_proj[core_index,1]
        
            # sigma guess
            init_sigma = 300
            
            Cs = [0.5,-10,20,16,0.01]
        
        
            res = sp.optimize.minimize(EnergyRec.AERA.aeraChi2,(init_A,init_sigma,Cs[0],Cs[1],Cs[2],Cs[3],Cs[4]),args=(self),method='Nelder-Mead')

            chi2min = EnergyRec.AERA.aeraChi2(res.x,self)
            ndof = self.fluence_arr[self.fluence_arr>self.f_thres].size - res.x.size

            print("** AERA fit:")
            print("---> ","{:6} {:>10} {:>10}".format("Par","Initial","Bestfit"))
            print("---> ","----------------------------")
            print("---> ","{:6} {:10} {:10}".format("A",round(init_A,3),round(res.x[0],4)))
            print("---> ","{:6} {:10} {:10}".format("sigma",round(init_sigma,2),round(res.x[1],4)))
            print("---> ","{:6} {:10} {:10}".format("C0",round(Cs[0],2),round(res.x[2],4)))
            print("---> ","{:6} {:10} {:10}".format("C1",round(Cs[1],2),round(res.x[3],4)))
            print("---> ","{:6} {:10} {:10}".format("C2",round(Cs[2],2),round(res.x[4],4)))
            print("---> ","{:6} {:10} {:10}".format("C3",round(Cs[3],2),round(res.x[5],4)))
            print("---> ","{:6} {:10} {:10}".format("C4",round(Cs[4],2),round(res.x[6],4)))
            print("---> ","----------------------------")
            print("---> ","Chi2min/n.d.o.f = ",str(round(chi2min,2))," / ",int(ndof))
        
            bestfit=open(bestfit_out, 'a')
            CR_input_Sradio=open("CR_input_Sradio.out", 'a')

            
            A=res.x[0]
            sigma=res.x[1]
            sin2Alpha = 1-np.dot(self.shower.ev,self.shower.eB)**2.
            Cs = np.array([res.x[2],res.x[3],res.x[4],res.x[5],res.x[6]])
            Sradio = (A*np.pi/sin2Alpha)*(sigma**2. - Cs[0]*(Cs[3]**2.)*np.exp(2*Cs[4]*sigma))

            if(filename==""):
                print(str(res.x)[1:-1],Sradio,file=bestfit)
                
            else:
                print(str(res.x)[1:-1],file=bestfit)
            
            print(self.shower.thetaCR,self.shower.phiCR,self.shower.ECR,Sradio,file=CR_input_Sradio)

            bestfit.close()
            
            ## \cond
            self.bestfit = res.x
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
            sin2Alpha = (1-np.dot(self.shower.ev,self.shower.eB)**2.).value
        
            Cs = np.array([self.bestfit[2],self.bestfit[3],self.bestfit[4],self.bestfit[5],self.bestfit[6]])
            Sradio = (A*np.pi/sin2Alpha)*(sigma**2. - Cs[0]*(Cs[3]**2.)*np.exp(2*Cs[4]*sigma))
            print('S_radio=',round(Sradio,2))

            par = [A,sigma,Cs[0],Cs[1],Cs[2],Cs[3],Cs[4]]
            
            sel_signal = np.where(self.fluence_arr>0)
            
            delta_X = np.max(self.shower.r_proj[:,0][sel_signal]) - np.min(self.shower.r_proj[:,0][sel_signal])
            delta_Y = np.max(self.shower.r_proj[:,1][sel_signal]) - np.min(self.shower.r_proj[:,1][sel_signal])
            mean_X = np.min(self.shower.r_proj[:,0][sel_signal]) + delta_X/2
            mean_Y = np.min(self.shower.r_proj[:,1][sel_signal]) + delta_Y/2

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
                    Z[i,j] = EnergyRec.AERA.aeraLDF(par, my_evB, X[i,j], Y[i,j]) # evaluation of the function on the grid
                
            fig = plt.figure(figsize=[14,5])
            plt.subplot(121)
            im = plt.imshow(Z,cmap='viridis',origin = 'lower', extent=[minXAxis,maxXAxis,minYAxis,maxYAxis]) # drawing the function
            plt.colorbar().ax.set_ylabel(r"energy fluence in eV/m$^2$")

            x_proj = self.shower.r_proj[:,0]
            y_proj = self.shower.r_proj[:,1]
            plt.scatter(x_proj[sel_signal], y_proj[sel_signal], c=self.fluence_arr[sel_signal], cmap='viridis', s = 100, edgecolors=(1,1,1,0.2))
            plt.clim(np.min([np.min(Z),np.min(self.fluence_arr[sel_signal])]), np.max([np.max(Z),np.max(self.fluence_arr[sel_signal])]))

            plt.xlabel(r'distante along $\vec{v}\times\vec{B}$ (in m)')
            plt.ylabel(r'distante along $\vec{v}\times\vec{v}\times\vec{B}$ (in m)')

            plt.plot(rcore[0], rcore[1],'w*')

            plt.subplot(122)
            plt.scatter(x_proj[sel_signal], y_proj[sel_signal], c=self.fluence_arr[sel_signal], cmap='viridis', s = 100, edgecolors=(1,1,1,0.2))
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
            plt.errorbar(temp_dist[sel_signal],self.fluence_arr[sel_signal],yerr=np.sqrt(self.fluence_arr[sel_signal]),fmt='.')
            plt.xlabel("Distance from core in m")
            plt.ylabel(r"Fluence in eV/m$^2$")
            plt.gca().set_yscale('log')
                
            residual = np.zeros(self.fluence_arr[sel_signal].size)
            
            for i in range(self.fluence_arr[sel_signal].size):
                residual[i] = (self.fluence_arr[sel_signal][i] - EnergyRec.AERA.aeraLDF(par, my_evB, x_proj[sel_signal][i], y_proj[sel_signal][i]))/np.sqrt(self.fluence_arr[sel_signal][i])


            plt.subplot(122)
            plt.errorbar(temp_dist[sel_signal],residual,yerr=1,fmt='.')
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
        def a_ratio(r, d_Xmax, rho_max = 0.4, p0 = 0.373, p1 = 762.6, p2 = 0.149, p3 = 0.189):
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
            return p0*(r/d_Xmax)*np.exp(r/p1)*(np.exp((rho_max-rho_mean)/p2)-p3)

        @staticmethod
        def f_par_geo(f_vB, phi, alpha, r, d_Xmax, rho_max = 0.4):
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
            return f_vB/(1+(np.cos(phi)/np.abs(np.sin(alpha))*np.sqrt(EnergyRec.SymFit.a_ratio(r, d_Xmax, rho_max)))**2)
        
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

            height = np.dot(-r*e_vec,np.array([0,0,1]))/1000 # from m to km

            H = 8 # in km
            rho_0 = 1.225 # in kg/m^3
            site_height = 2.8 # in km

            return rho_0*np.exp(-(site_height+height)/H)

        
        @staticmethod
        def a_ratio_chi2(par,fluence_geo, fluence_ce,alpha, r_proj, d_Xmax, rho_max):
            """
            Chi2 for the a_ratio fit.
            
            Args:
                fluence_geo:   An array with the geomagnetic fluences;
                fluence_ce:  An array with the charge excess fluences;
                alpha: An array with the geomagnetic angles.

            Retuns:
                Chi2: The Chi2 value.
            """
            p0 = par[0]
            p1 = par[1]
            p2 = par[2]
            p3 = par[3]

            sel = np.where(fluence_geo > 0)

            a_arr = (np.sin(alpha[sel])**2)*fluence_ce[sel]/fluence_geo[sel]

            Chi2 = 0
            for i in range(fluence_geo[sel].size):
                a_theo = EnergyRec.SymFit.a_ratio(np.linalg.norm(r_proj[sel][i]), d_Xmax[sel][i], rho_max[sel][i], p0, p1, p2,p3)
                if a_arr[i] < 1:
                    Chi2 = Chi2 + (a_arr[i] -a_theo)**2

            return Chi2

        @staticmethod
        def a_ratio_fit(fluence_geo, fluence_ce,alpha, r_proj, d_Xmax, rho_max):
            """
            Fits the a_ratio.
            
            Args:
                fluence_geo:   An array with the geomagnetic fluences;
                fluence_ce:  An array with the charge excess fluences;
                alpha: An array with the geomagnetic angles;
                r_proj: An array with the antenna positions in the shower plane;
                d_Xmax: An array with the distances to shower maximum;
                rho_max: An array with the densities at shower maximum.

            Retuns:
                bestfit: The bestfit parameters array.
            """
            par = [0.373, 762.6, 0.1490, 0.189]
            res = sp.optimize.minimize(EnergyRec.SymFit.a_ratio_chi2,par,args=(fluence_geo, fluence_ce,alpha, r_proj, d_Xmax, rho_max))
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