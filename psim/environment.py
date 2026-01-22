import numpy as np
from numba import jit
from psim.constants import *
from psim.helpers import logger
import ISA_module as ISA # Assumes ISA_module.py is in the root directory
# digital elevation model
import srtm

# Initialize the DEM data handler
elevation_data = srtm.get_data()

# This function does not accept JIT/Numba
#@jit(nopython=True)
def get_rho(altitude:float)->float:
    '''
    calculate the air density given an altitude in meters
    '''
    return ISA.rho_SL * ISA.sigma(altitude * M2FT) # ISA expects alt in ft


# Velocity, FPA and Geodesy functions
@jit(nopython=True)
def VA(uvw:np.ndarray) -> float:
    '''
    Calculate true airspeed
    input:
        uvw: vector of 3 speeds u, v, w
    returns:
        true airspeed
    '''
    return np.sqrt(np.dot(uvw.T, uvw))


@jit(nopython=True)
def fpa(V_NED:np.ndarray)->float:
    '''
    returns flight path angle
    input is a vector with North, East and Down velocities
    '''
    return np.arctan2(-V_NED[2], np.sqrt(V_NED[0]**2 + V_NED[1]**2))

def get_AGL(current_latlon_deg: np.ndarray, current_alt_m: float, sim_visual_offset: float)->float:
    '''
    this function fetches the current AGL n meters from the SRTM database
    needs lat/lon in degrees
    '''
    ground_alt = elevation_data.get_elevation(current_latlon_deg[0], current_latlon_deg[1])
    if ground_alt is None:
        ground_alt = 0.0 # Default to Sea Level over oceans
        logger.debug('NO DEM DATA')
        print('DEBUG - get_AGL: NO DEM DATA')
    elif ground_alt < 0:
        ground_alt = 0.0
        print('Below ground!')
        logger.warning('get_AGL: below ground!')


    return current_alt_m - ground_alt - sim_visual_offset


def course(V_NED:np.ndarray)->float:
    '''
    returns the course, given NED velocities
    '''
    return np.pi/2 - np.arctan2(V_NED[0], V_NED[1])


@jit(nopython=True)
def add_wind(NED:np.ndarray, std_dev:np.ndarray)->np.ndarray:
    '''
    returns wind at altitude Hp.
    inputs:
        NED: vector with wind speed
        std_dev: vector with standard deviations for wind (one value for each N, E, D)
    output:
        wind speed vector
    '''
    return NED + np.multiply(np.random.rand(3), std_dev)


# geodsy
# https://www.youtube.com/watch?v=4BJ-GpYbZlU
@jit(nopython=True)
def WGS84_MN(lat:float)->float:
    '''
    Meridian Radius of Curvature
    Prime Vertical Radius of Curvature
    for WGS-84
    
    Input is latitude in degress (decimal)
    '''
    a = 6378137.0 #meters
    e_sqrd = 6.69437999014E-3
    M = (a * (1 - e_sqrd)) / ((1 - e_sqrd * np.sin(lat)**2)**(1.5))
    N = a / ((1 - e_sqrd * np.sin(lat)**2)**(0.5))
    return M, N


@jit(nopython=True)
def latlonh_dot(V_NED:np.ndarray, lat:float, h:float)->np.ndarray:
    '''
    V_NED: m/s
    lat: latitude in degrees (decimal)
    h: altitude in meters
    '''
    M, N = WGS84_MN(lat)
    return np.array([(V_NED[0]) / (M + h), 
                     (V_NED[1]) / ((N + h) * np.cos(lat)),
                     -V_NED[2]])


 ############################################################################
# Navigation Equations
# ############################################################################

# source:
# Christopher Lum - "The Naviation Equations: Computing Position North, East and Down"
# https://www.youtube.com/watch?v=XQZV-YZ7asE


@jit(nopython=True)
def NED(uvw:np.ndarray, phithetapsi:np.ndarray)->np.ndarray:
    '''
    compute the NED velocities from:
    inputs
    uvw: array with u, v, w
    phithetapsi: array with phi, theta, psi
    
    returns
    velocities in NED
    
    remember that h_dot = -Vd
    '''
    
    u = uvw[0]
    v = uvw[1]
    w = uvw[2]
    phi = phithetapsi[0]
    the = phithetapsi[1]
    psi = phithetapsi[2]
    c1v = np.array([[np.cos(psi), np.sin(psi), 0.0],
                    [-np.sin(psi), np.cos(psi), 0.0],
                    [0.0, 0.0, 1.0]])
    
    c21 = np.array([[np.cos(the), 0.0, -np.sin(the)],
                    [0.0, 1.0, 0.0],
                    [np.sin(the), 0.0, np.cos(the)]])
    
    cb2 = np.array([[1.0, 0.0, 0.0],
                    [0.0, np.cos(phi), np.sin(phi)],
                    [0.0, -np.sin(phi), np.cos(phi)]])
    
    cbv = np.dot(cb2, np.dot(c21,c1v)) #numba does not support np.matmul
    return np.dot(cbv.T, uvw)
    