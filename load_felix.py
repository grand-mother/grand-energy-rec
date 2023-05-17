import numpy as np
import h5py

def load(path):
    r_proj = {}
    r_ground = {}
    fluence_evB = {}
    fluence_evvB = {}
    fluence_ev = {}

    with h5py.File(path, "r") as file:
        n_ant = file['highlevel']['obsplane_2900_gp_vB_vvB']['antenna_names'].shape[0]
        dt = file['highlevel']['obsplane_2900_gp_vB_vvB']
        for ant in range(n_ant):
            x = dt["antenna_position"][ant, 0]
            y = dt["antenna_position"][ant, 1]
            z = dt["antenna_position"][ant, 2]

            ff = dt['energy_fluence'][ant]
            ff_evB = np.abs(dt['energy_fluence_vector'][ant][0])
            ff_evvB = np.abs(dt['energy_fluence_vector'][ant][0])
            
            rr_proj = dt['antenna_position_vBvvB'][ant][:]

            r_ground[ant] = np.array([x, y, z])
            r_proj[ant] = rr_proj
            fluence_evB[ant] = ff_evB
            fluence_evvB[ant] = ff_evvB
            fluence_ev[ant] = np.sqrt(ff * ff - ff_evB * ff_evB - ff_evvB * ff_evvB)
            
    
    fluence_dict = {
        "fluence_x" : None,
        "fluence_y" : None,
        "fluence_z" : None,
        "fluence_evB" : fluence_evB,
        "fluence_evvB" : fluence_evvB,
        "fluence_ev" : fluence_ev,
    }

    return {"shower_azimuth" : 0,
            "shower_zenith" : 0,
            "r_core" : np.asarray([0, 0, 0]),
            "r_x_max" : np.asarray([0, 0, 0]),
            "shower_B" : np.asarray([0, 0, 0]),
            "shower_energy" : 1e18,
            "trace_x" : None,
            "trace_y" : None,
            "trace_z" : None,
            "time" : None,
            "r_ground" : None,
            "r_proj" : r_proj,
            "fluence_dict" : fluence_dict
            }