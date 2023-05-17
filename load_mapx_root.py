import numpy as np
import ROOT
event_lib = "/mnt/d/Dropbox/Pesquisa/GRAND/MachineLearning/EnergyRec/Event/libSimuEvent.so"
ROOT.gSystem.Load(event_lib)

def load(path, site_height=0, evt=0):
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

    r_core = np.asarray([0, 0, site_height])
    r_x_max = np.asarray([branch_SimShower.xmax_pos_shc[0],
                          branch_SimShower.xmax_pos_shc[1],
                          branch_SimShower.xmax_pos_shc[2]
                        ])
    shower_B = np.asarray(shower_Bfield)

    trace_x = {}
    trace_y = {}
    trace_z = {}
    time = {}
    r_ground = {}
    for ant in range(len(Detectors_trace_Ex)):
        trace_x[ant]  = np.asarray(Detectors_trace_Ex[ant], "f8") * 1.e-6 # muV/m to V/m
        trace_y[ant]  = np.asarray(Detectors_trace_Ey[ant], "f8") * 1.e-6 # muV/m to V/m
        trace_z[ant]  = np.asarray(Detectors_trace_Ez[ant], "f8") * 1.e-6 # muV/m to V/m
        time[ant] = np.asarray(
                        np.linspace(t0[ant], t0[ant] + 
                        t_bin_size * len(Detectors_trace_Ex[ant]),
                        len(Detectors_trace_Ex[ant])),"f8"
                    )
        r_ground[ant] = np.asarray([float(positions[ant][0]),
                                    float(positions[ant][1]),
                                    float(positions[ant][2])
                                  ])
    
    fluence_dict = None

    ev = r_core - r_x_max
    ev /= np.linalg.norm(ev)
    #ev = ev.T[0]

    eB = shower_B
    eB[2] = -eB[2]
    eB /= np.linalg.norm(eB)

    return{"ant_ID" : id,
           "r_ground" : np.array(list(r_ground.values())),
           "r_core" : r_core,
           "traces_time" : list(time.values()),
           "traces_x" : list(trace_x.values()),
           "traces_y" : list(trace_y.values()),
           "traces_z" : list(trace_z.values()),
           "ev" : ev,
           "eB" : eB,
           "r_x_max" : r_x_max,
           "energy" : shower_energy
           }