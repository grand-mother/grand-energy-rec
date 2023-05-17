import numpy as np
from pathlib import Path
import re

def load(simulation_dir):

    path = Path(simulation_dir)
    inp = {}
    try:
        sry_path = path.glob("*.sry").__next__()
    except:
        pass
    else:
        def parse_quantity(string: str):
            words = string.split()
            return float(words[0])
        
        def parse_geomagnet_intensity(string: str):  # -> u.Quantity:
            return float(string.split()[0]) * 1e-3  # uT --> nT

        def parse_geomagnet_angles(string: str): #-> CartesianRepresentation:
            intensity = inp["geomagnet"]
            inclination, _, _, declination, _ = string.split()
            theta = 90 + float(inclination)  # deg
            inp["_declination"] = float(declination)  # deg
            # phi=0 because x-axis is parallel to the magnetic north.
            spherical = [theta, 0, intensity]
            return spherical
        
        def parse_maximum(string: str): # -> CartesianRepresentation:
            _, _, *xyz = string.split()
            x, y, z = map(float, xyz)

            ## Xmax is given as CartesianRepresentation defined in the shower frame.
            # Later (below) Xmax is saved wrt LTP frame making it independent of shower info.
            ## "Previously: Dirty hack by OMH for now" -> not necessary now. RK.
            try:
                inp_file = path.glob("*.inp").__next__()
                #logger.info("### zhaires.py: reading groundaltitude from. inp file.")
                with open(inp_file, encoding="UTF-8") as f:
                    for line in f:
                        if "GroundAltitude" in line:
                            ground_alt = float(line.split()[1])  # m
                            inp["ground_alt"] = ground_alt
                            inp["core"] = [0, 0, ground_alt]
            except:
                pass
            return [1000 * x, 1000 * y, 1000 * z]  # RK. km --> m
        
        converters = (
                ("Primary energy", "energy", parse_quantity),
                ("Primary zenith angle", "zenith", parse_quantity),
                ("Primary azimuth angle", "azimuth", parse_quantity),
                (
                    "Geomagnetic field: Intensity:",
                    "geomagnet",
                    parse_geomagnet_intensity,
                ),
                ("I:", "geomagnet", parse_geomagnet_angles),
                ("Pos. Max.:", "maximum", parse_maximum),
            )

        i = 0
        tag, k, convert = converters[i]
        with sry_path.open() as f:
            for line in f:
                start = line.find(tag)
                if start < 0:
                    continue
                inp[k] = convert(line[start + len(tag) + 1 :])
                #inp[k] = line[start + len(tag) + 1 :]
                i = i + 1
                try:
                    tag, k, convert = converters[i]
                except IndexError:
                    break

    positions = {}
    ant_file = path / "antpos.dat"  # Ex: 1 A0  0.00000E+00  2.70450E+02  2.90000E+03
    if ant_file.exists():
        pattern = re.compile("A([0-9]+)$")
        with ant_file.open() as f:
            for line in f:
                if not line:
                    continue
                words = line.split()
                match = pattern.search(words[1])

                if match is None:
                    raise InvalidAntennaName(words[1])
                antenna = int(match.group(1))
                positions[antenna] = [
                    float(words[2]),  # m, # x-coordinate from shower core.
                    float(words[3]),  # m, # y-coordinate from shower core.
                    float(words[4]),  # m, # z-coordinate from shower core.
                ]

    path2 = Path(simulation_dir + "/" + path.name)
    t = {}
    Ex = {}
    Ey = {}
    Ez = {}
    for field_path in path2.glob("a*.trace"):
        # Example field_path => ..../grand/tests/simulation/data/zhaires/a1.trace
        #                    =>    time [ns]      Ex [uVm]    Ey [uVm]   Ez [uVm]
        #                    => -1.1463000E+04  -5.723E-05  -1.946E-04  4.324E-04
        antenna = int(field_path.name[1:].split(".", 1)[0])
        # logger.debug(f"Loading trace for antenna {antenna}")
        data = np.loadtxt(field_path)
        #t[antenna] = data[:, 0] * 1.0e-9  # ns --> s
        t[antenna] = data[:, 0]  # ns
        Ex[antenna] = data[:, 1] * 1.0e-6 # uVm --> V/m
        Ey[antenna] = data[:, 2] * 1.0e-6 # uVm --> V/m
        Ez[antenna] = data[:, 3] * 1.0e-6 # uVm --> V/m


    ev =  -(np.array(inp['maximum']) - np.array(inp['core']))
    ev = ev/np.linalg.norm(ev)
    sin_theta = np.sin(inp['geomagnet'][0]*np.pi/180)
    cos_theta = np.cos(inp['geomagnet'][0]*np.pi/180)
    eB = np.array([sin_theta, 0, cos_theta])

    return{"ant_ID" : range(len(t)),
            "r_ground" : np.array(list(positions.values())),
            "r_core" : inp['core'],
            "traces_time" : list(t.values()),
            "traces_x" : list(Ex.values()),
            "traces_y" : list(Ey.values()),
            "traces_z" : list(Ez.values()),
            "ev" : ev,
            "eB" : eB,
            "r_x_max" : np.array(inp['maximum']),
            "energy" : inp['energy']
            }