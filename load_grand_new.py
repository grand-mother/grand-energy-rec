#!/usr/bin/python
# An example of using DataFile for ROOT file reading
import numpy as np
import sys
import os
sys.path.append("/mnt/d/Dropbox/Pesquisa/GitHub/grand-mother/grand")

from grand.dataio.root_trees import *
from types import SimpleNamespace
import matplotlib.pyplot as plt

def load(simulation, run_number = 0, site_height = 0):

    directory_path, fname = os.path.split(simulation)
    directory_path = directory_path + "/"

    f_input_TRun=directory_path+"trun_"+ fname[8:]              
    f_input_TShower=directory_path+"tshower_"+ fname[8:]
    f_input_TEfield=directory_path+"tefield_"+ fname[8:] 

    gt = SimpleNamespace()
    gt.trun = TRun(f_input_TRun)
    gt.trun.get_entry(run_number)
    gt.tshower = TShower(f_input_TShower)
    gt.tshower.get_entry(run_number)
    gt.tefield = TEfield(f_input_TEfield)
    gt.tefield.get_entry(run_number)

    timeseconds=gt.tefield.time_seconds
    timenanoseconds=gt.tefield.time_nanoseconds

    #change all times relative to the first time TODO:(could be done to the event time, but this first events have time set to 0)
    du_seconds=np.array(gt.tefield.du_seconds)
    du_seconds=du_seconds-du_seconds[0]

    etbinsize=float(gt.trun.t_bin_size[0]) #TODO this should be in trunefieldsim
    etracelength=len(gt.tefield.trace[0][0])

    id = gt.trun.du_id
    r_ground = gt.trun.du_xyz
    r_core = gt.tshower.shower_core_pos
    traces_time = [du_second*1e9 + 20 * np.linspace(0, etracelength-1, etracelength) for du_second in du_seconds]

    traces = np.array(gt.tefield.trace)
    traces_x = [trace[0] for trace in traces]
    traces_y = [trace[1] for trace in traces]
    traces_z = [trace[2] for trace in traces]
    r_x_max = gt.tshower.xmax_pos
    ev = r_core - r_x_max
    ev /= np.linalg.norm(ev)
    #ev = ev.T[0]

    eB = gt.tshower.magnetic_field
    # eB[2] = -eB[2] #Check if needed!!!!
    eB /= np.linalg.norm(eB)

    energy = gt.tshower.energy_primary

    return{"ant_ID" : id,
            "r_ground" : np.array(list(r_ground)),
            "r_core" : r_core,
            "traces_time" : list(traces_time),
            "traces_x" : list(traces_x),
            "traces_y" : list(traces_y),
            "traces_z" : list(traces_z),
            "ev" : ev,
            "eB" : eB,
            "r_x_max" : r_x_max,
            "energy" : energy
            }