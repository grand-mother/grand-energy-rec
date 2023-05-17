import re
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
from grand import ECEF, LTP, Geodetic
from grand.simulation import ElectricField, ShowerEvent, ZhairesShower
from grand.simulation.pdg import ParticleCode
from grand.simulation.shower.generic import CollectionEntry, FieldsCollection

import os

def load(simulation, simulation_type, site_height = 0):
    if Path(simulation).is_dir() or simulation.endswith("hdf5") or simulation.endswith("root"):

        if simulation_type == "coreas":
            GRANDshower = init_coreas(simulation)
            return fill_from_GRANDshower(GRANDshower)
        
        elif simulation_type == "zhaires":
            GRANDshower = init_zhaires(simulation)
            return fill_from_GRANDshower(GRANDshower)
            
        elif (
            simulation_type == "custom" or simulation_type == "starshape"
        ):
            GRANDshower, fluence_dict = init_custom(simulation, site_height)
            return fill_from_GRANDshower(GRANDshower, fluence_dict)

def init_coreas(simulation):
        GRANDshower = ShowerEvent.load(simulation)
        return GRANDshower
    
def init_zhaires(simulation):
    if(os.path.isfile(simulation)):
        GRANDshower = ZhairesShower._from_datafile(simulation)
    else:
        GRANDshower = ZhairesShower._from_dir(simulation)

    # Fixing Aires to GRAND conventions
    for ant in range(len(GRANDshower.fields)):
        GRANDshower.fields[ant].electric.E = GRANDshower.fields[
            ant
        ].electric.E[0]

        #from astropy.coordinates.matrix_utilities import \
        #    rotation_matrix

        #rotation = rotation_matrix(-90 * u.deg, axis="z")

        #GRANDshower.fields[ant].electric.E = GRANDshower.fields[
        #    ant
        #].electric.E.transform(rotation)

        #GRANDshower.fields[ant].electric.r = GRANDshower.fields[
        #    ant
        #].electric.r.transform(rotation)

    GRANDshower.localize(latitude=45.5, longitude=90.5)
    return GRANDshower

def init_custom(simulation, site_height):
    GRANDshower, bool_traces = custom_from_datafile(
                simulation, site_height
            )

    fluence_x = {}
    fluence_y = {}
    fluence_z = {}

    if not bool_traces:
        RunInfo = Table.read(self.simulation, path="RunInfo")
        EventName = RunInfo["EventName"][0]
        AntennaFluenceInfo = Table.read(
            simulation, EventName + "/AntennaFluenceInfo"
        )
        for ant in AntennaFluenceInfo:
            idx = ant["ID"]
            fluence_x[idx] = ant["Fluencex_efield"]
            fluence_y[idx] = ant["Fluencey_efield"]
            fluence_z[idx] = ant["Fluencez_efield"]
    
    else:
        fluence_x = None
        fluence_y = None
        fluence_z = None
    
    fluence_dict = {
        "fluence_x" : fluence_x,
        "fluence_y" : fluence_y,
        "fluence_z" : fluence_z,
        "fluence_evB" : None,
        "fluence_evvB" : None,
        "fluence_ev" : None,
    }

    if fluence_x is None:
        fluence_dict = None

    return GRANDshower, fluence_dict

def custom_from_datafile(path: Path, site_height=0.) -> ZhairesShower:
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
            r = CartesianRepresentation(0, float(x), float(y), float(z))
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
                t = np.asarray([0], "f8")
                Ex = np.asarray([0], "f8")
                Ey = np.asarray([0], "f8")
                Ez = np.asarray([0], "f8")
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
            geodetic = Geodetic(latitude=float(latitude), longitude=float(longitude), height=0.0)
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
                core=CartesianRepresentation(0, *my_core),
                geomagnet=CartesianRepresentation(geomagnet),
                maximum=CartesianRepresentation(
                    0, *event[0, "XmaxPosition"]
                ),
                fields=fields,
            ),
            bool_traces,
        )

@classmethod
def _from_dir(cls, path: Path) -> ZhairesShower:
    if not path.exists():
        raise FileNotFoundError(path)

    # Note: Zhaires has a fixed coordinate frame at a location with 'NWU' orientation at the sea level.
    # 'N' is the magnetic north, 'W' is 90 deg west from 'N', and 'U' is upward towards zenith.
    inp: Dict[str, Any] = {}
    try:
        sry_path = path.glob("*.sry").__next__()
    except StopIteration as from_dir_exit:
        raise FileNotFoundError(path / "*.sry") from from_dir_exit
    else:

        def parse_quantity(string: str):
            words = string.split()
            return float(words[0])

        def parse_geomagnet_intensity(string: str):  # -> u.Quantity:
            return float(string.split()[0]) * 1e-3  # uT --> nT

        def parse_geomagnet_angles(string: str) #-> CartesianRepresentation:
            intensity = inp["geomagnet"]
            inclination, _, _, declination, _ = string.split()
            theta = 90 + float(inclination)  # deg
            inp["_declination"] = float(declination)  # deg
            # phi=0 because x-axis is parallel to the magnetic north.
            spherical = [theta, 0, intensity]
            return spherical

        def parse_maximum(string: str) #-> CartesianRepresentation:
            _, _, *xyz = string.split()
            x, y, z = map(float, xyz)

            ## Xmax is given as CartesianRepresentation defined in the shower frame.
            # Later (below) Xmax is saved wrt LTP frame making it independent of shower info.
            ## "Previously: Dirty hack by OMH for now" -> not necessary now. RK.
            try:
                inp_file = path.glob("*.inp").__next__()
                logger.info("### zhaires.py: reading groundaltitude from. inp file.")
                with open(inp_file, encoding="UTF-8") as f:
                    for line in f:
                        if "GroundAltitude" in line:
                            ground_alt = float(line.split()[1])  # m
                            inp["ground_alt"] = ground_alt
            except StopIteration as parse_maximum_exit:
                raise FileNotFoundError(path / "*.inp") from parse_maximum_exit
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
            ("Location of max.(Km)", "maximum", parse_maximum),
        )

        i = 0
        tag, k, convert = converters[i]
        with sry_path.open() as f:
            for line in f:
                start = line.find(tag)
                if start < 0:
                    continue
                inp[k] = convert(line[start + len(tag) + 1 :])
                i = i + 1
                try:
                    tag, k, convert = converters[i]
                except IndexError:
                    break

    ground_alt = inp["ground_alt"]

    inp["core"] = [0, 0, ground_alt]
    # RK. Save Xmax in LTP frame. It will be easier to convert to antenna frame.
    #    But it takes more space (about 8 times/antenna).
    Xmax = inp["maximum"]
    inp["maximum"] = [Xmax.x, Xmax.y, Xmax.z]  # RK

    # Positions are in LTP frame with origin at shower core. Usually shower frame has 'NWU' orientation,
    # where N=magnetic north. Defined in ..../grand/tests/simulation/data/zhaires/*.sry file.
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
                    x=float(words[2]),  # m, # x-coordinate from shower core.
                    y=float(words[3]),  # m, # y-coordinate from shower core.
                    z=float(words[4]),  # m, # z-coordinate from shower core.
                ]
                # print("### Warning: Forcing antenna height = 0m")
                # RK. Note: this is time consuming but useful.
                #     CartesianRepresentation~200Bytes/antenna, LTP~900Bytes/antenna.
                # positions[antenna] = LTP(
                #    x = float(words[2]), #* u.m, # x-coordinate in LTP frame.
                #    y = float(words[3]), #* u.m, # y-coordinate in LTP frame.
                #    z = float(words[4]), #* u.m, # z-coordinate in LTP frame.
                #    frame = inp['frame']
                # )

    raw_fields = {}
    for field_path in path.glob("a*.trace"):
        # Example field_path => ..../grand/tests/simulation/data/zhaires/a1.trace
        #                    =>    time [ns]      Ex [uVm]    Ey [uVm]   Ez [uVm]
        #                    => -1.1463000E+04  -5.723E-05  -1.946E-04  4.324E-04
        antenna = int(field_path.name[1:].split(".", 1)[0])
        # logger.debug(f"Loading trace for antenna {antenna}")
        data = numpy.loadtxt(field_path)
        t = data[:, 0] * 1.0e-9  # ns --> s
        Ex = data[:, 1]  # uVm
        Ey = data[:, 2]  # uVm
        Ez = data[:, 3]  # uVm
        raw_fields[antenna] = [t, Ex, Ey, Ez, positions[antenna]]

    return cls(fields=fields, **inp)
    
def fill_from_GRANDshower(GRANDshower, fluence_dict = None):
    r_core = np.asarray(GRANDshower.core)
    r_x_max = np.asarray(GRANDshower.maximum)
    shower_B = np.asarray(GRANDshower.geomagnet.T[0])
    shower_energy = GRANDshower.energy
    shower_zenith = GRANDshower.zenith
    shower_azimuth = GRANDshower.azimuth

    trace_x = {}
    trace_y = {}
    trace_z = {}
    time = {}
    r_ground = {}
    for ant in GRANDshower.fields.keys():
        trace_x[ant]  = np.asarray(GRANDshower.fields[ant].electric.E.x)
        trace_y[ant]  = np.asarray(GRANDshower.fields[ant].electric.E.y)
        trace_z[ant]  = np.asarray(GRANDshower.fields[ant].electric.E.z)
        time[ant] = np.asarray(GRANDshower.fields[ant].electric.t)
        r_ground[ant] = np.asarray(GRANDshower.fields[ant].electric.r.T[0])
    
    return {"shower_azimuth" : shower_azimuth,
            "shower_zenith" : shower_zenith,
            "r_core" : r_core,
            "r_x_max" : r_x_max,
            "shower_B" : shower_B,
            "shower_energy" : shower_energy,
            "trace_x" : trace_x,
            "trace_y" : trace_y,
            "trace_z" : trace_z,
            "time" : time,
            "r_ground" : r_ground,
            "r_proj" : None,
            "fluence_dict" : fluence_dict
            }