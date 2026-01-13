# FGROOT = /usr/share/games/flightgear

# DRI_PRIME=1 fgfs --airport=SBGP  --aircraft=Embraer170 --aircraft-dir=./FlightGear/Aircraft/E-jet-family/ --native-fdm=socket,in,60,,5500,udp --fdm=null --enable-hud --in-air --fog-disable --shading-smooth --texture-filtering=4 --timeofday=morning --altitude=2500 --prop:/sim/hud/path[1]=Huds/NTPS.xml
# DRI_PRIME=1 fgfs --airport=LOWI  --aircraft=Embraer170 --aircraft-dir=./FlightGear/Aircraft/E-jet-family/ --native-fdm=socket,in,60,,5500,udp --fdm=null --enable-hud --in-air --fog-disable --shading-smooth --texture-filtering=4 --timeofday=morning --altitude=2500 --prop:/sim/hud/path[1]=Huds/fte.xml 2>/dev/null


# FG with JSBSim:
# DRI_PRIME=1 fgfs --airport=SBGP  --aircraft=Embraer170 --aircraft-dir=./FlightGear/Aircraft/E-jet-family/  --enable-hud  --fog-disable --shading-smooth --texture-filtering=4 --timeofday=morning
# DRI_PRIME=1 fgfs --airport=KSFO --runway=28R  --aircraft=757-200-RB211 --aircraft-dir=~/.fgfs/Aircraft/org.flightgear.fgaddon.stable_2020/Aircraft/757-200  --enable-hud  --fog-disable --shading-smooth --texture-filtering=4 --timeofday=morning

# "v" muda o visual
# https://wiki.flightgear.org/Command_line_options


'''
Partial Python implementation of the non-linear flight dynamics model proposed by:
Group for Aeronautical Research and Technology Europe (GARTEUR) - Research Civil Aircraft Model (RCAM)
http://garteur.org/wp-content/reports/FM/FM_AG-08_TP-088-3.pdf
HOWEVER!!!
    # many equations and values are only available in the newer RCAM document (rev Feb 1997)
    # which is not availble to the public
    # the values from this reference were obtained from the youtube videos below

The excellent tutorials by Christopher Lum (for Matlab/Simulink) were used as a guide:
1 - Equations/Modeling
https://www.youtube.com/watch?v=bFFAL9lI2IQ
2 - Matlab implementation
https://www.youtube.com/watch?v=m5sEln5bWuM

The program runs the integration loop at a target pf 400Hz, adjusting the integration steps to the available computing cycles
It uses Numba to speed up the main functions involved in the integration loop

Output is sent to FlightGear (FG), over UDP, at a reduced frame rate (60)
The FG interface uses the class implemented by Andrew Tridgel (fgFDM):
https://github.com/ArduPilot/pymavlink/blob/master/fgFDM.py

currently, the UDP address is set to the local machine.
A second UDP address is available for an extra screen/instance of FG

Run this Python program and from a separate terminal, start FG with one of the following commands (depending on the aircraft addons installed):
fgfs --airport=KSFO --runway=28R --aircraft=ufo --native-fdm=socket,in,60,,5500,udp --fdm=null
fgfs --airport=KSFO --runway=28R --aircraft=Embraer170 --aircraft-dir=./FlightGear/Aircraft/E-jet-family/ --native-fdm=socket,in,60,,5500,udp --fdm=null
fgfs --airport=KSFO --runway=28R --aircraft=757-200-RB211 --aircraft-dir=~/.fgfs/Aircraft/org.flightgear.fgaddon.stable_2020/Aircraft/757-200 --native-fdm=socket,in,60,,5500,udp --fdm=null
fgfs --airport=KSFO --runway=28R --aircraft=757-200-RB211 --aircraft-dir=~/.fgfs/Aircraft/org.flightgear.fgaddon.stable_2020/Aircraft/757-200 --native-fdm=socket,in,60,,5500,udp --fdm=null --enable-hud --turbulence=0.5 --in-air  --enable-rembrandt
DRI_PRIME=1 fgfs --airport=LOWI  --aircraft=Embraer170 --aircraft-dir=./FlightGear/Aircraft/E-jet-family/ --native-fdm=socket,in,60,,5500,udp --fdm=null --enable-hud --in-air --fog-disable --shading-smooth --texture-filtering=4 --timeofday=morning --altitude=2500 --prop:/sim/hud/path[1]=Huds/fte.xml 2>/dev/null

If a joystick is detected, then inputs come from it
Otherwise, offline simulation is run


TODO:
    1) add engine dynamics (spool up/down) [DONE]
    2) add atmospheric disturbances/turbulence [DONE - wind]
    3) add other actuator dynamics [DONE]
    4) save/read trim point
    5) fuel detot / inertia update
    6) add engine cut logic with dynamics per RCAM and controls from joystick
    7) update wind/turbulence per RCAM
    8) add flaps (delta CL, CM, CD) with controls from joystick
    9) add landing ger (delta CM, CD) with controls from joystick
    10) add ground effect (delta CL) with radalt/height


'''

# imports
import numpy as np
from functools import partial
from scipy import integrate
from scipy.optimize import minimize # for trimming routine
from numba import jit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import ISA_module as ISA # International Standard Atmosphere library

import time
import sys

sys.path.insert(1, '../')

from psim.io.fgDFM import * # FlightGear comm class
import socket

# threading for FG comms
import threading
import queue

# multiprocessing for engine deck
import multiprocessing as mp

import pygame #joystick interface




from psim.constants import *
from psim.config import load_aircraft_parameters
import psim.environment as env
import psim.propulsion as prop
import psim.helpers as helpers
import psim.io.joystick as joy
import psim.io.network as net

  



# ############################################################################
# Load Aircraft Parameters into Global Scope
#
# Numba's JIT compiler captures global variables when a function is first
# compiled. By loading our parameters into the global scope, we make them
# available to the performance-critical `RCAM_model` and `control_sat`
# functions without needing to pass them as arguments on every call.
# ############################################################################

############################################################################
# we first need the joystick name, to load the correct parameters...
# JOYSTICK INIT AND CHECK
pygame.init() # automatically initializes joystick also

# check if joystick is connected
joystick_count = pygame.joystick.get_count()
if joystick_count == 0:
    joy_name = None
else:
    this_joy = pygame.joystick.Joystick(0)
    joy_name = this_joy.get_name()


try:
    # Unpack the dictionary into global variables
    consts = load_aircraft_parameters('rcam_parameters.json', joy_name)
    globals().update(consts)
    joy.initialize_constants(consts)
except FileNotFoundError:
    print("ERROR: `rcam_parameters.json` not found. Please create it.")
    sys.exit(1)
except (KeyError, json.JSONDecodeError) as e:
    print(f"ERROR: Invalid format in `rcam_parameters.json`: {e}")
    sys.exit(1)


if OFFLINE:
    if joy_name == None:
        print()
        print('Will run OFFLINE simulation, no joystick detected!')
    else:
        print()
        print(f'Will run OFFLINE simulation, joystick model {joy_name} not in JSON config file!')
else:
    print()
    print(f'found {joystick_count} joysticks connected: {joy_name}, axes={this_joy.get_numaxes()}')


# ############################################################################
# Landing Gear Model Initialization
# ############################################################################
# Define contact points relative to the Center of Gravity (CG)
# Frame: Body Axis (X=Forward, Y=Right, Z=Down)
# Units: Meters

print(f"Landing Gear Model Loaded:")
print(f"  Nose Rel Pos: {LG_NOSE_POS}")
print(f"  Main Rel Pos: {LG_MAIN_L_POS}")

# ############################################################################
# High Lift Devices Interpolator
# ############################################################################
# note: numba does like np.clip if we pass in a single number float...it needs a numpy array

@jit(nopython=True)
def high_lift_interp(x:float) -> np.array:
    x = max(0.0, min(float(MAX_FLAP), x))
    idx = int(x)
    frac = x - idx
    if idx >= MAX_FLAP: return HIGH_LIFT_COEFFS[MAX_FLAP]
    return HIGH_LIFT_COEFFS[idx] + (HIGH_LIFT_COEFFS[idx+1] - HIGH_LIFT_COEFFS[idx]) * frac


# ############################################################################
# Controls and Actuators
# ############################################################################

@jit(nopython=True)
def control_norm(U:np.array) -> np.array:
    '''
    normalizes controls to be sent to FG
    inputs:
        U controls: positions (in radians)
        [U_lim: control limits (in radians) moved to global variable for speed]
    returns:
        vector with control positions normalized between 1 and -1
    '''
    # Extract limits for first 3 channels
    mins = U_LIMITS_MIN[:3]
    maxs = U_LIMITS_MAX[:3]
    
    # Avoid divide by zero
    mins = np.where(mins == 0, 1.0, mins)
    maxs = np.where(maxs == 0, 1.0, maxs)

    # Slice input
    u_subset = U[:3]
    
    # Vectorized normalization
    # If U < 0: U / abs(min)
    # If U >= 0: U / max
    U_norm = np.where(u_subset < 0, 
                      u_subset / np.abs(mins), 
                      u_subset / maxs)
    
    return U_norm


@jit(nopython=True)
def control_sat(U:np.ndarray) -> np.ndarray:
    '''
    saturates the control inputs to maximum allowable in RCAM model
    '''
    return np.clip(U, U_LIMITS_MIN, U_LIMITS_MAX)


@jit(nopython=True)
def update_actuators(U_cmd:np.ndarray, U_actual:np.ndarray, dt:float, tau:np.ndarray) -> np.ndarray:
    """
    Simulates first order lag for control surfaces.
    y_dot = (u_cmd - y_current) / tau
    y_new = y_current + y_dot * dt
    """
    U_new = np.zeros_like(U_actual)

    # 1. Aerodynamic Surfaces (First Order Lag)
    rate = (U_cmd - U_actual) / tau 
    U_new = U_actual + rate * dt
             
    # 2. Engines (Pass through - The engine deck handles spool up dynamics)
    # testing fast time constant for engine, which should have no effect
    #U_new[3:5] = U_cmd[3:5]
    
    return U_new


# ############################################################################
# Ground Reactions and Detection Logic
# ############################################################################

@jit(nopython=True)
def calculate_gear_compression(X:np.ndarray, h_cg:float) -> np.ndarray:
    """
    Calculates the vertical compression of each landing gear strut.
    Positive value = Gear is in contact with ground (compressed).
    Negative value = Gear is in the air.
    
    Inputs:
        X: State vector (needs phi [6] and theta [7])
        h_cg: Current altitude of the Center of Gravity (meters)
        rwy_alt: height of the ground/runway (meters)
        
    Returns:
        compressions: np.array([nose_comp, main_l_comp, main_r_comp])
    """
    phi = X[IDX_PHI]
    theta = X[IDX_THETA]
    
    # The bottom row of the Direction Cosine Matrix (Body to NED)
    # This vector transforms a body vector [x,y,z] into the Down component
    DCM_z_row = np.array([
        -np.sin(theta), 
        np.sin(phi) * np.cos(theta), 
        np.cos(phi) * np.cos(theta)
    ])
    
    # Stack gear positions into a 3x3 matrix for batch processing
    # Rows: Nose, MainL, MainR
    gear_positions = np.zeros((3, 3))
    gear_positions[0, :] = LG_NOSE_POS
    gear_positions[1, :] = LG_MAIN_L_POS
    gear_positions[2, :] = LG_MAIN_R_POS
    
    # Calculate vertical distance (dz) for all 3 gears at once
    # Result is a vector of 3 elements
    dz = gear_positions @ DCM_z_row 
    
    # Calculate tips
    h_tips = h_cg - dz
    
    # Compression is simply -h_tips (assuming ground is 0 relative to h_cg passed in)
    # We clip negative values (air) to 0.0
    compressions = np.maximum(-h_tips, 0.0)
    
    # Clip max compression
    max_travel = np.array([LG_NOSE_POS[2], LG_MAIN_L_POS[2], LG_MAIN_R_POS[2]])
    compressions = np.minimum(compressions, max_travel)
    
    return compressions


@jit(nopython=True)
def get_air_ground_state(compressions:np.ndarray) -> bool:
    """
    Returns True (Ground) if ANY gear is compressed, False (Air) otherwise.
    This acts as the requested 'air_ground' variable.
    """
    if compressions[0] > 0 or compressions[1] > 0 or compressions[2] > 0:
        return True
    else:
        return False


@jit(nopython=True)
def calculate_ground_forces(X:np.ndarray, h_cg:float, brake:float) -> np.ndarray:
    """
    Calculates total Forces and Moments (Body Frame) from all 3 landing gears.
    h_cg is height AGL (RADALT) in (m)
    brake is a float between 0 and 1 - represents brake percent
    Returns: 6-element array [Fx, Fy, Fz, Mx, My, Mz]

    """
    
    # Get current compressions
    compressions = calculate_gear_compression(X, h_cg)

    
    # State extractions for velocity calc
    u, v, w = X[IDX_U], X[IDX_V], X[IDX_W]
    p, q, r = X[IDX_P], X[IDX_Q], X[IDX_R]
    wbe_b = np.array([p, q, r])
    V_cg_b = np.array([u, v, w])
    
    total_F = np.zeros(3)
    total_M = np.zeros(3)
    
    # Loop through the 3 gears
    # 0=Nose, 1=MainL, 2=MainR
    gears_pos = [LG_NOSE_POS, LG_MAIN_L_POS, LG_MAIN_R_POS]
    
    for i in range(3):
        delta_z = compressions[i]
        
        if delta_z > 0:
            
            r_gear = gears_pos[i]
            V_gear = V_cg_b + np.cross(wbe_b, r_gear)
            
            # Vertical Velocity of the gear (Positive = Moving Down/Compressing)
            v_z_gear = V_gear[2] 
            
            # --- ASYMMETRIC DAMPING LOGIC ---
            if v_z_gear > 0: 
                # Compressing (moving into ground)
                # Use standard damping to absorb energy
                damping_force = LG_DAMP_COMPRESSION * v_z_gear
            else:
                # Rebounding (spring pushing plane up)
                # Use HIGH damping to prevent the spring from shooting the plane up
                damping_force = LG_DAMP_REBOUND * v_z_gear
            
            # Normal Force = Spring Force - Damping Force
            # Note: We use max(0, ...) on the spring to ensure it pushes up.
            F_normal = -max(0.0, LG_SPRING_K * delta_z) - damping_force
            
            # Sanity Check: The ground cannot pull the plane down. 
            # If the damping force is so high it overcomes the spring during rebound, 
            # clamp force to zero.
            if F_normal > 0: 
                F_normal = 0 
            
            # Friction
            # A simple "stiffness" approach to friction to stop sliding
            # Side force
            F_y = -V_gear[1] * 50000.0 
            raw_fy = -V_gear[1] * LG_FRICTION_STIFFNESS
            # for longitudinal force, we have the brakes
            F_x = F_normal * (LG_MU_BRAKE * brake + LG_ROLLING_FRICTION_MU)
            raw_fx = -V_gear[0] * LG_FRICTION_STIFFNESS

            
            # Optional: Cap friction so it doesn't exceed mu * Normal Force (Coulomb friction)
            # This prevents numerical instability if sliding sideways fast
            max_fx = abs(F_normal * (LG_MU_BRAKE + LG_ROLLING_FRICTION_MU))
            max_fy = abs(F_normal * LG_SIDE_FRICTION_MU)
            if abs(V_gear[0]) > 10.0:
                if abs(F_x) > max_fx: F_x = np.sign(F_x) * max_fx # we are fast, use normal friction
            else:
                F_x = raw_fx * (1 + 4 * brake)# we are at low speed, use stiffness and brake 

            if abs(V_gear[1]) > 10.0:
                if abs(F_y) > max_fy: F_y = np.sign(F_y) * max_fy
            else:
                F_y = raw_fy

            F_gear_b = np.array([F_x, F_y, F_normal])
            M_gear_b = np.cross(r_gear, F_gear_b)
            
            total_F += F_gear_b
            total_M += M_gear_b

    return np.concatenate((total_F, total_M))



# ############################################################################
# RCAM flight dynamics model
# ############################################################################

@jit(nopython=True)
def RCAM_model(X:np.ndarray, U:np.ndarray, rho:float, h:float) -> np.ndarray:
    """
    RCAM model implementation
    sources: RCAM docs and Christopher Lum
    Group for Aeronautical Research and Technology Europe (GARTEUR) - Research Civil Aircraft Model (RCAM)
    http://garteur.org/wp-content/reports/FM/FM_AG-08_TP-088-3.pdf

    Christopher Lum - Equations/Modeling
    https://www.youtube.com/watch?v=bFFAL9lI2IQ
    Christopher Lum - Matlab implementation
    https://www.youtube.com/watch?v=m5sEln5bWuM

    inputs:
        X: states (TP-088-3, p. 6, para 2.2, table 2.2)
            0: u (m/s)
            1: v (m/s)
            2: w (m/s)
            3: p (rad/s)
            4: q (rad/s)
            5: r (rad/s)
            6: phi (rad)
            7: theta (rad)
            8: psi (rad)
        U: controls (TP-088-3, p. 6, para 2.2, table 2.1)
            0: aileron (rad)
            1: stabilator (rad)
            2: rudder (rad)
            3: E1 THRUST (N) (original RCAM was throttle 1 in %)
            4: E2 THRUST (N)  (original RCAM was throttle 2 in %)
            5: Flaps position
            6: Landing gear position (0=up / 1=dn)
            7: spoilers (%) (not included in original RCAM)
            8: wheel brakes (%) (not included in original RCAM)
        rho: density for current altitude (kg/m3)
        h: height above ground (m)
    outputs:
        X_dot: derivatives of states (same order)
    """
   
    # ------------------------- states ----------------------------------
    u, v, w = X[IDX_U], X[IDX_V], X[IDX_W] # m/s
    p, q, r = X[IDX_P], X[IDX_Q], X[IDX_R] # rad/s
    phi, theta, psi = X[IDX_PHI], X[IDX_THETA], X[IDX_PSI] # rad

    # ----------------------- controls ----------------------------------
    da, de, dr = U[IDX_AIL], U[IDX_ELE], U[IDX_RUD]
    dt1, dt2 = U[IDX_THR1], U[IDX_THR2]
    flap_pos, gear_pos = U[IDX_FLAP], U[IDX_GEAR]
    dgsp, brake = U[IDX_GNDSP], U[IDX_BRAKE]

    #----------------- intermediate variables ---------------------------
    # airspeed
    Va = np.sqrt(u**2 + v**2 + w**2) # m/s
    
    # alpha and beta
    # Protect against divide by zero if Va is very small (on ground)
    if Va < 0.1:
        alpha = 0.0
        beta = 0.0
    else:
        alpha = np.arctan2(w, u)
        beta = np.arcsin(v / Va)
    
    # dynamic pressure
    Q = 0.5 * rho * Va**2
    
    # define vectors wbe_b and V_b
    wbe_b = np.array([p, q, r])
    V_b = np.array([u, v, w])
    
    #----------------- aerodynamic force coefficients ---------------------
        # this is only available in the newer RCAM document (rev Feb 1997)
    # which is not availble to the public
    # CL - wing + body
    # adding spoiler to kill lift
    CL_wb = N * (alpha - ALPHA_L0) * (1 - dgsp) if alpha <= ALPHA_SWITCH else (A3 * alpha**3 + A2 * alpha**2 + A1 * alpha + A0) * (1 - dgsp)

    # CL thrust
    epsilon = DEPSDA * (alpha - ALPHA_L0)
    # Prevent divide by zero in epsilon_dot term
    q_term = (EPSILON_DOT * q * LT / Va) if Va > 0.1 else 0.0
    alpha_t = alpha - epsilon + de + q_term
    CL_t = NT * (ST / S) * alpha_t

    # Total CL
    CL = CL_wb + CL_t

    # Total CD (in stability frame)
    CD = CDMIN + D1 * (N * alpha + D0)**2

    # Total side force CY (stability frame)
    CY = CY_BETA * beta + CY_DR * dr

    #------------------- dimensional aerodynamic forces --------------------
    # forces in F_s
    FA_s = np.array([-CD * Q * S, CY * Q * S, -CL * Q * S])

    # rotate forces to body axis (F_b)
    C_bs = np.array([[np.cos(alpha), 0.0, -np.sin(alpha)],
                     [0.0, 1.0, 0.0],
                     [np.sin(alpha), 0.0, np.cos(alpha)]], dtype=np.dtype('f8'))

    FA_b = np.dot(C_bs, FA_s)   
    
    #------------------ aerodynamic moment coefficients about AC -----------
    # moments in F_b
    eta11 = C_l_BETA * beta
    eta21 = C_m_ALPHA - (NT * (ST * LT) / (S * CBAR)) * (alpha - epsilon)
    eta31 = (1 - alpha * C_n_BETA) * beta

    eta = np.array([eta11, eta21, eta31])
    
    # Prevent divide by zero in damping terms
    inv_Va = (CBAR / Va) if Va > 0.1 else 0.0

    dCMdx = inv_Va * np.array([[C_l_P, C_l_Q, C_l_R], 
                                    [C_m_P, (C_m_Q * (ST * LT**2) / (S * CBAR**2)), C_m_R], 
                                    [C_n_P, C_n_Q, C_n_R]], dtype=np.dtype('f8'))
    dCMdu = np.array([[C_l_DA , C_l_DE, C_l_DR],
                      [C_m_DA, (C_m_DE * (ST * LT) / (S * CBAR)), C_m_DR],
                      [C_n_DA, C_n_DE, C_n_DR]], dtype=np.dtype('f8'))
    
    # CM about AC in Fb
    CMac_b = eta + np.dot(dCMdx, wbe_b) + np.dot(dCMdu, np.array([da, de, dr]))

    #------------------- aerodynamic moment about AC -------------------------
    # normalize to aerodynamic moment
    MAac_b = CMac_b * Q * S * CBAR

    #-------------------- aerodynamic moment about CG ------------------------
    rcg_b = np.array([XCG, YCG, ZCG])
    rac_b = np.array([XAC, YAC, ZAC])

    MAcg_b = MAac_b + np.cross(FA_b, rcg_b - rac_b)
    
    #---------------------- engine force and moment --------------------------
    # thrust
    #F1 = dt1 * M * G # orginal RCAM
    #F2 = dt2 * M * G
    F1 = dt1
    F2 = dt2

    # thrust vectors (assuming aligned with x axis)
    FE1_b = np.array([F1, 0, 0])
    FE2_b = np.array([F2, 0, 0])

    FE_b = FE1_b + FE2_b

    # engine moments
    mew1 = np.array([XCG - XAPT1, YAPT1 - YCG, ZCG - ZAPT1])
    mew2 = np.array([XCG - XAPT2, YAPT2 - YCG, ZCG - ZAPT2])

    MEcg1_b = np.cross(mew1, FE1_b)
    MEcg2_b = np.cross(mew2, FE2_b)

    MEcg_b = MEcg1_b + MEcg2_b
    
    #---------------------- gravity effects ----------------------------------
    g_b = np.array([-G * np.sin(theta), G * np.cos(theta) * np.sin(phi), G * np.cos(theta) * np.cos(phi)])

    Fg_b = M * g_b

    #---------------------- GROUND REACTION ----------------------------------
    # This is the new part for Step 3
    Gnd_Reac = calculate_ground_forces(X, h, brake)
    F_gnd_b = Gnd_Reac[:3]
    M_gnd_b = Gnd_Reac[3:]
    
    #---------------------- state derivatives --------------------------------
    
    # form F_b and calculate u, v, w dot
    # Added F_gnd_b to the sum
    F_b = Fg_b + FE_b + FA_b + F_gnd_b
    
    u_v_w_dot  = (1 / M) * F_b - np.cross(wbe_b, V_b)
    
    # form Mcg_b and calc p, q r dot
    # Added M_gnd_b to the sum
    Mcg_b = MAcg_b + MEcg_b + M_gnd_b
    
    p_q_r_dot = np.dot(INV_INERTIA_TENSOR_b, (Mcg_b - np.cross(wbe_b, np.dot(INERTIA_TENSOR_b , wbe_b))))
    
    # phi, theta, psi dot
    H_phi = np.array([[1.0, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                      [0.0, np.cos(phi), -np.sin(phi)],
                      [0.0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]], dtype=np.dtype('f8'))
    
    phi_theta_psi_dot = np.dot(H_phi, wbe_b)
    
    #--------------------- place in first order form --------------------------
    X_dot = np.concatenate((u_v_w_dot, p_q_r_dot, phi_theta_psi_dot))
    
    return X_dot


# for efficiency, we create a new function twin just to calculate the internal states and return them for logging. 
# we do not need to log at full simulation frame rate.
@jit(nopython=True)
def RCAM_observe(X:np.ndarray, U:np.ndarray, rho:float, h:float) -> np.ndarray:
    """
    Performs the same calculations as RCAM_model but returns internal variables
    for logging purposes instead of state derivatives.
    
    Returns array:
    [0:Va, 1:alpha, 2:beta, 3:CL, 4:CD, 5:CY, 6:Gnd_Fx, 7:Gnd_Fy, 8:Gnd_Fz]
    """
   
    # ------------------------- states ----------------------------------
    u, v, w = X[IDX_U], X[IDX_V], X[IDX_W] # m/s
    p, q, r = X[IDX_P], X[IDX_Q], X[IDX_R] # rad/s
    phi, theta, psi = X[IDX_PHI], X[IDX_THETA], X[IDX_PSI] # rad

       # ----------------------- controls ----------------------------------
    da, de, dr = U[IDX_AIL], U[IDX_ELE], U[IDX_RUD]
    dt1, dt2 = U[IDX_THR1], U[IDX_THR2]
    flap_pos, gear_pos = U[IDX_FLAP], U[IDX_GEAR]
    dgsp, brake = U[IDX_GNDSP], U[IDX_BRAKE]

    #----------------- intermediate variables ---------------------------
    # airspeed
    Va = np.sqrt(u**2 + v**2 + w**2) # m/s
    
    # alpha and beta
    # Protect against divide by zero if Va is very small (on ground)
    if Va < 0.1:
        alpha = 0.0
        beta = 0.0
    else:
        alpha = np.arctan2(w, u)
        beta = np.arcsin(v / Va)
       
    #----------------- aerodynamic force coefficients ---------------------
        # this is only available in the newer RCAM document (rev Feb 1997)
    # which is not availble to the public
    # CL - wing + body
    # adding spoiler to kill lift
    CL_wb = N * (alpha - ALPHA_L0) * (1 - dgsp) if alpha <= ALPHA_SWITCH else (A3 * alpha**3 + A2 * alpha**2 + A1 * alpha + A0) * (1 - dgsp)

    # CL thrust
    epsilon = DEPSDA * (alpha - ALPHA_L0)
    # Prevent divide by zero in epsilon_dot term
    q_term = (EPSILON_DOT * q * LT / Va) if Va > 0.1 else 0.0
    alpha_t = alpha - epsilon + de + q_term
    CL_t = NT * (ST / S) * alpha_t

    # Total CL
    CL = CL_wb + CL_t

    # Total CD (in stability frame)
    CD = CDMIN + D1 * (N * alpha + D0)**2

    # Total side force CY (stability frame)
    CY = CY_BETA * beta + CY_DR * dr

    
    #---------------------- engine force and moment --------------------------
    # thrust
    F1 = dt1
    F2 = dt2



    #---------------------- GROUND REACTION ----------------------------------
    # This is the new part for Step 3
    Gnd_Reac = calculate_ground_forces(X, h, brake)
    F_gnd_b = Gnd_Reac[:3]
    F_gnd_x = Gnd_Reac[0]
    F_gnd_y = Gnd_Reac[1]
    F_gnd_z = Gnd_Reac[2]



    
    return np.array([Va, alpha*RAD2DEG, beta*RAD2DEG, CL, CD, CY, F_gnd_x, F_gnd_y, F_gnd_z])





# ############################################################################
# Model Integration
# ############################################################################

# # # wrappers
    # Scipy's "integrate.ode" does not accept a numba/@jit(nopython=True) compiled function
    # therefore, we need to create dummy wrappers

def RCAM_model_wrapper(t, X, U, rho, h):
    return RCAM_model(X, U, rho, h)

def NED_wrapper(t, X, NED):
    return env.NED

def latlonh_dot_wrapper(t, X, V_NED, lat, h):
    return env.latlonh_dot(V_NED, lat, h)


# # # integrators
def ss_integrator(t_ini:float, X0:np.ndarray, U:np.ndarray, rho:float, h:float):
    """
    single step integrator for FDM
    returns scipy object, initialized
    """

    RK_integrator = integrate.ode(RCAM_model_wrapper)
    RK_integrator.set_integrator('dopri5') 
    RK_integrator.set_f_params(U, rho, h) # Pass initial h

    RK_integrator.set_initial_value(X0, t_ini)
    return RK_integrator


def latlonh_int(t_ini:float, latlonh0:np.ndarray, V_NED):
        
    '''
    single step integrator for lat/long/height
    returns scipy object, initialized
    '''
    
    RK_integrator = integrate.ode(latlonh_dot_wrapper)
    RK_integrator.set_integrator('dopri5')
    RK_integrator.set_f_params(V_NED, latlonh0[0], latlonh0[2])
    RK_integrator.set_initial_value(latlonh0, t_ini)
    return RK_integrator





# ############################################################################
# Model Trimming Function
# ############################################################################

# define partial function with fixed flaps, gear position, ground spoilers and brake
# this partial function leaves "floating" only the parameters that the optimizer can vary
# the problem is that "partial" fixes the parameters from left to right
# so there is no easy way to fix the last parameters (they'd have to be named)
# because we trim only once, no big deal defining these special functions


def trim_functional3(Z:np.ndarray, VA_trim, gamma_trim, side_speed_trim, phi_trim, psi_trim, rho_trim, h_trim, 
                     flap_pos, gear_pos, gnd_sp_pos, brakes_pos) -> np.dtype('f8'):
    """
    functional to calculate a cost for minimizer (used to find trim point)
    no constraints yet
    inputs:
        Z: lumped vector of X (states) and U (control)
        trim targets:
        VA_trim: airspeed [m/s]
        gamma_trim: climb gradient [rad]
        side_speed_trim: lateral (v) speed [m/s]
        phi_trim: roll angle [rad]
        psi_trim: course angle [rad]
        h_trim: height above the ground (m) - for ground proximity checks

    ****
    method
    Q.T*H*Q
    with H = diagonal matrix of "1"s (equal weights for all states)
    this returns the squares of the elements in Q
    
    returns:
        cost [float]
    """

    X = Z[:9]
    #U = Z[9:]
    U = np.zeros(Z.shape[0]-9+4)
    for i in range (Z.shape[0]-9):
        U[i] = Z[9+i]
    U[Z.shape[0]-9+0] = flap_pos
    U[Z.shape[0]-9+1] = gear_pos
    U[Z.shape[0]-9+2] = gnd_sp_pos
    U[Z.shape[0]-9+3] = brakes_pos

    
    # PASS h_trim to the model here:
    X_dot = RCAM_model(X, U, rho_trim, h_trim)

    
    VA_current = env.VA(X[:3])
    
    gamma_current = X[IDX_THETA] - np.arctan2(X[IDX_W], X[IDX_U]) 
     
    Q = np.concatenate((X_dot, [VA_current - VA_trim], [gamma_current - gamma_trim], [X[IDX_V] - side_speed_trim], [X[IDX_PHI] - phi_trim], [X[IDX_PSI] - psi_trim]))
    square_ones = np.ones(Q.shape[0])
    H = np.diag(square_ones)
    
    return np.dot(np.dot(Q.T, H), Q)


def trim_model(VA_trim=85.0, gamma_trim=0.0, side_speed_trim=0.0, phi_trim=0.0, psi_trim=0.0, rho_trim=1.225, 
               h_trim=100.0, flaps=0, gear=0, gnd_sp=0, brakes=0,
               X0=np.array([85.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
               U0=np.array([1.0, 1.0, 1.0, 0.08, 0.08, 0.0, 0.0, 0.0, 0.0])) -> np.ndarray:
    """
    uses scipy minimize on functional to find trim point
    X0 states:
        u, v, w, p, q, r, phi, theta, psi
    U0 controls:
        ail, ele, rud, thr1, thr2, gnd spoiler, brake, flaps position, gear position
    h_trim is passed on to check ground proximity

    """

    # add target trim values to X0 vector, as a better initial guess for the states
    X0[IDX_U] = VA_trim
    X0[IDX_PHI] = phi_trim
    X0[IDX_PSI] = psi_trim

    # REMOVE MAGIC NUMBERS LATER
    U0[5] = flaps
    U0[6] = gear
    U0[7] = gnd_sp
    U0[8] = brakes


    MAX_ITER = 10 
    iter_counter = 0
    epsilon = 1E-9
    converge = False

    # concatenate states and inputs into single vector
    # for trimming, ground spoilers and brakes are not a valid control
    Z0 = np.concatenate((X0, U0[:-4])) # removing ground spoilers and brake from trim variables

    print(f'initial cost: {trim_functional3(Z0, VA_trim, gamma_trim, side_speed_trim, phi_trim, psi_trim, rho_trim, h_trim,
                             flaps, gear, gnd_sp, brakes):.3e}')

    # TODO: ONLY TRIM IF IN AIR

    while iter_counter <= MAX_ITER and not converge:
        # AQUI EU TENHO QUE MANDAR ZO SEM OS 4 ULTIMOS
        # Updated args tuple to include h_trim
        result = minimize(trim_functional3, Z0, args=(VA_trim, gamma_trim, side_speed_trim, phi_trim, psi_trim, rho_trim, h_trim,
                          flaps, gear, gnd_sp, brakes),
                method='Nelder-Mead', options={'maxiter':50000,\
                                               'maxfev':40000})
        
        # Updated cost check with h_trim
        current_cost = trim_functional3(result.x, env.VA(result.x[:3]), result.x[IDX_THETA] - np.arctan2(result.x[IDX_W], result.x[IDX_U]), result.x[IDX_V], result.x[IDX_PHI], result.x[IDX_PSI], rho_trim, h_trim,
                                         flaps, gear, gnd_sp, brakes)
        print(f'iter: {iter_counter}, functional cost: {current_cost:.3e}')

        if current_cost < epsilon:
            converge = True
        else:
            iter_counter += 1
            Z0 = result.x.copy()


    if converge:
        print()
        print('Trim converged!')
        print(f'trimmed speed = {env.VA(Z0[:3]):.1f} m/s')
        
        # Updated X_dot check
        #X_dot = RCAM_model(result.x[:9], result.x[9:], rho_trim, h_trim)

        print(f'check gamma {result.x[IDX_THETA] - np.arctan2(result.x[IDX_W], result.x[IDX_U])} RAD')
        print(f'check side vel {result.x[IDX_V]:.1f} m/s')
        print(f'check phi {result.x[IDX_PHI] * RAD2DEG:.1f} Deg')
        print(f'check psi {result.x[IDX_PSI]* RAD2DEG:.1f} Deg')
    else:
        print('FAILED TO CONVERGE')


    return result.x, result.message #remember that the control vector is missing ground spoilers now



# ############################################################################
# Model Initialization
# ############################################################################

def initialize(VA_t=85.0, gamma_t=0.0, latlon=np.zeros(2), altitude=10000.0, psi_t=0.0, height=0.0, flaps=0, gear=0):
    """
    this initializes the integrators at a straight and level flight condition
    inputs:
        VA_t: true airspeed at trim (m/s)
        gamma_t: flight path angle at trim (rad)
        latlon: initial lat and long (rad)
        altitude: trim altitude (ft)
        psi_t: initial heading (rad)
        height: height above ground (m) for air/ground purposes
        flap: flap position
        gear: gear position (0=up, 1=dn)
    outputs:
        AC_integrator: aircraft integrator object
        X0: initial states found at trim point
        U0: initial commands found at trim point
        latlonh_integrator: navigation equation scipy object integrator
    """
    t0 = 0.0 #intial time for integrators
    alt_m = altitude * FT2M
    rho_trim = env.get_rho(alt_m)

    print(f'initializing model with altitude {altitude} ft, rho={rho_trim:.4f} kg/m3')
    

    latlonh0 = np.array([latlon[0]*DEG2RAD, latlon[1]*DEG2RAD, alt_m])

    if VA_t > 15:
        # we are flying
        # trim model
        res4, res4_status = trim_model(VA_trim=VA_t, gamma_trim=gamma_t, side_speed_trim=0, 
                                    phi_trim=0.0, psi_trim=psi_t*DEG2RAD, rho_trim=rho_trim, h_trim=height,
                                    flaps=flaps, gear=gear)
        print()
        print('Trimming',res4_status)
        print()
        X0 = res4[:9] # separate states and controls
        U0 = np.concatenate((res4[9:], np.array([flaps, gear, 0.0, 0.0]))) # add back ground spoiler and brakes to control vector
        print(f'initial states: {X0}')
        print(f'initial inputs: {U0}')
        print()
    else:
        # we are on the ground
        X0=np.array([VA_t, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, INIT_HDG_DEG * DEG2RAD])
        U0=np.array([0.0, 0.0, 0.0, 0.0, 0.0, flaps, gear, 0.0, 0.0])

    # initialize integrators
    AC_integrator = ss_integrator(t0, X0, U0, rho_trim, height)
    
    NED0 = env.NED(X0[:3], X0[6:]) #uvw and phithetapsi
    
    latlonh_integrator = latlonh_int(t0, latlonh0, NED0)
    
    return AC_integrator, X0, U0, latlonh_integrator    



# ############################################################################
# MAIN
# ############################################################################

if __name__ == "__main__":


############################################################################
    # SELECT STARTING POINT: ON GROUND OR IN AIR
    TRIM_ON_GROUND = False

    # INITIAL CONDITIONS (for trim)
    if TRIM_ON_GROUND:
        # ON GROUND  
        INIT_ALT_FT = 585.553 * M2FT #ft
        V_TRIM_MPS = 0 * KT2MS # m/s
        INIT_LATLON_DEG = np.array([.8248243303439*RAD2DEG, 0.1977872426444*RAD2DEG]) #LOWI, RWY 08
    else:
        INIT_ALT_FT = 2400 #ft
        V_TRIM_MPS = 160 * KT2MS # m/s
        INIT_LATLON_DEG = np.array([47.2548, 11.2963]) #in degrees - LOWI short final TFB

    
    GAMMA_TRIM_RAD = 0.0 * DEG2RAD # RAD
    INIT_HDG_DEG = 82.0 # DEG
    # Lat/Lon
    #INIT_LATLON_DEG = np.array([37.6213, -122.3790]) #in degrees - the func initialize transforms to radians internally
    #INIT_LATLON_DEG = np.array([-21.7632, -48.4051]) #in degrees - SBGP
    #INIT_LATLON_DEG = np.array([47.2548, 11.2963]) #in degrees - LOWI short final TFB
    # wind
    WIND_NED_MPS = np.array([0, 0, 0]) # (m/s), NED
    WIND_STDDEV_MPS = np.array([1, 1, 0]) # wind standard deviation, NED

###########################################################################
    # SIMULATION OPTIONS
    SIM_TOTAL_TIME_S = 10 * 60 # (s) total simulation time
    SIM_LOOP_HZ = 400 # (Hz) simulation loop frame rate throttling
    FG_OUTPUT_LOOP_HZ = 60 # (Hz) frames per second to be sent out to FlightGear AND for recording data
    DECK_LOOP_HZ = 10 # (Hz) fra1me rate to calculate engine deck
    SIM_VISUAL_OFFSET = 0 # Simulator Visual offset so that landing is on the runway. Difference in Sim and SRTM values for ground elevation
    USE_FG_AS_TERRAIN_DB = True # if False, use SRTM database instead
    DATA_LOGGING_HZ = 10 # frames per second to be logged
    RESULTS_FILE = 'test_data.csv' # name of log file
    ENG_LOG_PARAMETERS = ['Fn', 'Fg', 'F_ram', 'TSFC', 'Wf', 'N1','N2']

###########################################################################
    # TERRAIN SHARED DATA
    terrain_shared_data = {'ground_alt': 0.0}

##########################################################################
    signals_header = ['u', 'v', 'w', 'p', 'q', 'r', 'phi', 'theta', 'psi', 'lat', 'lon', 'h', 'V_N', 'V_E', 'V_D', 'dA', 'dE', 'dR', 'dT1', 'dT2', 'flap_pos', 'gear_pos', 'dgsp', 'brake']
    internals_header = ['Va', 'alpha_deg', 'beta_deg', 'CL', 'CD', 'CY', 'Gnd_Fx', 'Gnd_Fy', 'Gnd_Fz']
    engine_header = []
    for eng_prefix in ['E1', 'E2']:
        for param in ENG_LOG_PARAMETERS:
            engine_header.append(eng_prefix+param)
    full_header = signals_header + internals_header + engine_header

    
###########################################################################
    # FlightGear Threads and Engine Deck Process Initialization
    # we only start the network and multiprocessing if doing online sim, at least for now
    if OFFLINE == False:
    ############################################################################
        # FLIGHTGEAR SOCKS
        # OUTGOING data (from Python to FG)
        # Open network sockets to communicate with FlightGear
        UDP_IP1 = "127.0.0.1" # set to localhost
        UDP_PORT1 = 5500
        
        UDP_IP2 = "192.168.0.163" # set to a remote computer on the same network
        UDP_PORT2 = 5501

        sock1 = socket.socket(socket.AF_INET, # Internet
                            socket.SOCK_DGRAM) # UDP
        sock2 = socket.socket(socket.AF_INET, # Internet
                            socket.SOCK_DGRAM) # UDP
        socks = [sock1, sock2]
        fg_addresses = [(UDP_IP1, UDP_PORT1), (UDP_IP2, UDP_PORT2)]

        fdm_packet_queue = queue.Queue() # async queue that will send the packets

        # THREADING: Create and start the network worker thread.
        # It's a daemon thread, so it will exit automatically if the main program exits.
        network_thread = threading.Thread(
            target=net.network_worker,
            args=(socks, fdm_packet_queue, fg_addresses),
            daemon=True
        )
        try:
            network_thread.start()
            print("... started!")
        except Exception as e:
            print(f"Error in network thread: {e}")
            exit()

        # INCOMING DATA (from FG to Python)
        # ... UDP RX Setup ...
        # --- TERRAIN UDP RECEIVER ---
        TERRAIN_RX_IP = "127.0.0.1" 
        TERRAIN_RX_PORT = 5502 # Port we listen ON

         # Queue used only for shutdown signal
        terrain_shutdown_queue = queue.Queue()
        
        terrain_thread = threading.Thread(
            target=net.terrain_udp_worker,
            args=(TERRAIN_RX_IP, TERRAIN_RX_PORT, terrain_shared_data, terrain_shutdown_queue),
            daemon=True
        )
        terrain_thread.start()



        # instantiate FG comms object and initialize it
        my_fgFDM = fgFDM()
        my_fgFDM.set('latitude', INIT_LATLON_DEG[0], units='degrees')
        my_fgFDM.set('longitude', INIT_LATLON_DEG[1], units='degrees')
        my_fgFDM.set('altitude', INIT_ALT_FT, units='feet')
        #my_fgFDM.set('agl', INIT_ALT_FT, units='meters')
        my_fgFDM.set('num_engines', 2)
        my_fgFDM.set('num_tanks', 1)
        my_fgFDM.set('num_wheels', 3)
        my_fgFDM.set('cur_time', int(time.perf_counter()), units='seconds')



    #######################################################################################
        # engine
        # --- Multiprocessing Setup ---
        # MULTIPROCESSING: Use Queues from the multiprocessing module.
        # These queues handle the necessary serialization (pickling) to pass
        # data between process memory spaces.
        jobs_queue = mp.Queue(maxsize=1)
        results_queue = mp.Queue(maxsize=1)

        # MULTIPROCESSING: Create and start the engine deck as a Process, not a Thread.
        engine_process = mp.Process(
            target=prop.engine_worker,
            args=(jobs_queue, results_queue),
            daemon=True  # Daemon processes are terminated when the parent exits
        )
        engine_process.start()



###########################################################################
    # SIMULATION VARIABLES INITIALIZATION
    data_collector, t_vector_collector = [], [] # data collectors
    
    prev_uvw = np.array([0,0,0])
    current_uvw = np.array([0,0,0])

    # aircraft initialization (includes trimming)
    this_AC_int, X_trim, U1, this_latlonh_int = initialize(VA_t=V_TRIM_MPS, gamma_t=GAMMA_TRIM_RAD, latlon=INIT_LATLON_DEG, altitude=INIT_ALT_FT, psi_t=INIT_HDG_DEG, height=100.0)
    # Vector U1 has the controls for the trimmed state
    U_man = U1.copy() # we set U_man (for manual controls) as a copy of the trimmed control states first.

    # Initialize Actual Surface Positions
    # We start with actual = commanded (assuming stable trim)
    U_actual = U1.copy() # U_actual will be the controls after applying the actuator dynamics

    e1_thrust = U1[3]
    e2_thrust = U1[4]


    # flaps variables
    toggle_flaps_debounce = 0

    # landing gear variables
    toggle_gear_debounce = 0

    # aircraft position variables
    current_alt_m = INIT_ALT_FT * FT2M # m
    current_latlon_rad = INIT_LATLON_DEG
    current_AGL_m = env.get_AGL(INIT_LATLON_DEG, current_alt_m, SIM_VISUAL_OFFSET)
    
    # frame variables
    frame_count = 0
    last_frame_time = 0 # holds the time from last 100 frame to calc frame rate at print statement

    fgdt = 1.0 / FG_OUTPUT_LOOP_HZ # (s) fg frame period
    simdt = 1 / SIM_LOOP_HZ # (s) desired simulation time step
    deckdt = 1 / DECK_LOOP_HZ
    datalogdt = 1 / DATA_LOGGING_HZ
    
    # semaphores
    send_frame_trigger = False
    run_sim_loop = False # this is a semaphore. it will wait for the clock to reach the next "simdt" and run the simulation
    calc_eng_trigger = True
    datalog_trigger = True
    
    # time tracking
    sim_time_adder, fg_time_adder = 0, 0 # counts the time between integration steps to trigger next simulation frame and FG dispatch
    eng_time_adder = 0 # loop to calculate engine
    datalog_time_adder = 0
    
    dt = 0 # actual integration time step
    prev_dt = dt

    exit_signal = 0 # if joystick button #1 is pressed, end simulation
    
###########################################################################
    # RUN SIMULATION
    # if no joystick detected, run offline
    if OFFLINE:
        # code for offline simulation
        # create time vector
        t_vector = np.arange(0, SIM_TOTAL_TIME_S, simdt)
        print(f'Offline sim time vector: {t_vector[0]:.2f}s to {t_vector[-1]:.2f}s')

        # create control inputs and set equal to trim
        sim_U = np.zeros((U_man.shape[0],t_vector.shape[0]))
        for i in range(sim_U.shape[0]):
            sim_U[i,:] = sim_U[i,:] + U_man[i]
        
        # all doublets have zero as starting amplitude
        pitch_doublet = helpers.get_doublet(t_vector,t=5, duration=2, amplitude=0.2)
        roll_doublet = helpers.get_doublet(t_vector,t=200, duration=2, amplitude=0.2)
        yaw_doublet = helpers.get_doublet(t_vector,t=400, duration=4, amplitude=0.2)
        # therefore we sum on top of the trim
        sim_U[0,:] += roll_doublet
        sim_U[1,:] += pitch_doublet
        sim_U[2,:] += yaw_doublet
        
        

        # single step integrate through each time step
        for idx, t in enumerate(t_vector):
            current_rho = env.get_rho(current_alt_m)

            # add actuator dynamics to control inputs:
            U_actual = update_actuators(sim_U[:,idx], U_actual, simdt, ACT_TAU)
            
            # integrate 6-DOF
            this_AC_int.set_f_params(U_actual, current_rho, current_AGL_m)
            this_AC_int.integrate(this_AC_int.t + simdt)

            # integrate navigation equations
            current_NED = env.NED(this_AC_int.y[:3], this_AC_int.y[6:])
            this_wind = env.add_wind(WIND_NED_MPS, WIND_STDDEV_MPS)
            this_latlonh_int.set_f_params(current_NED + this_wind, current_latlon_rad[0], current_alt_m)
            this_latlonh_int.integrate(this_latlonh_int.t + simdt) #in radians and alt in meters
            
            # store current state and time vector
            current_latlon_rad = this_latlonh_int.y[0:2] # store lat and long (RAD)
            
            if current_AGL_m != 0 : 
                current_alt_m = this_latlonh_int.y[2] # store altitude (m)
            else:
                this_latlonh_int.y[2] = current_alt_m

            data_collector.append(np.concatenate((this_AC_int.y, this_latlonh_int.y, current_NED + this_wind, U_actual)))
            current_alt_m = this_latlonh_int.y[2] # store altitude (m)
            
            t_vector_collector.append(this_AC_int.t)

        print(f'Enf of simulation; {len(t_vector_collector)} time steps!')
        
    # with joystick attached, run online
    else:
        ##### ONLINE #####

        # adjust engine command
        # what comes out of the trimming function is thrust directly
        # for online sim, we can't use it
        # let's run the reverse deck to get the thrust lever angle:
        print(f'running inverse deck with alt: {INIT_ALT_FT:.1f} ft, Mach: {ISA.Vc2M(V_TRIM_MPS*MS2KT, INIT_ALT_FT):.3f}, Thrust: {U1[3]*N2LBF:.0f} lbf')

        U1[IDX_THR1] = prop.E1_deck.interp_altMNFN(INIT_ALT_FT, ISA.Vc2M(V_TRIM_MPS*MS2KT, INIT_ALT_FT), e1_thrust*N2LBF)['PC']
        U1[IDX_THR2] = prop.E2_deck.interp_altMNFN(INIT_ALT_FT, ISA.Vc2M(V_TRIM_MPS*MS2KT, INIT_ALT_FT), e2_thrust*N2LBF)['PC']
        U_man[IDX_THR1] = U1[IDX_THR1]
        U_man[IDX_THR2] = U1[IDX_THR2]

        print(f'this is the inverse deck response: E1:{U1[IDX_THR1]:.4f}; E2:{U1[IDX_THR2]:.4f} % power')
        print()

        # run deck
        new_job = (current_alt_m*M2FT, ISA.Vt2M(V_TRIM_MPS*MS2KT, current_alt_m*M2FT), U_man[IDX_THR1], U_man[IDX_THR2], TRIM_ON_GROUND, time.perf_counter())
        jobs_queue.put(new_job, block=False)
        # need to give time for deck to run
        time.sleep(.5)

        ##### SIMULATION LOOP #####
        while this_AC_int.t <= SIM_TOTAL_TIME_S and exit_signal == 0:
            # get clock
            start = time.perf_counter()

            if run_sim_loop:

                #pygame.event.pump() # More efficient than event.get() if just reading axes
                joy_events = pygame.event.get()

                # -- Inputs & Actuators
                current_throttle = [U_man[IDX_THR1], U_man[IDX_THR2]] # keep track of throttle to zero-out the trim bias
                U_man, U1, exit_signal = joy.get_joy_inputs(this_joy, joy_events, U1, SIM_LOOP_HZ, JOY_TRIM_PARAMS, JOY_FACTORS)

                # U_man is the manual control inputs (as the joystick is moved)
                # U1 is the trim state, or the zero input values for each control.
                
                # for throtlle, initial trim state is always positive, so we washout if throttles move back
                # if engine trim state is negative, it means engine is OFF
                delta_throttle_1 = U_man[IDX_THR1] - current_throttle[0] #we look only at #1 engine for simplicity
                if delta_throttle_1 < 0 and U1[IDX_THR1] > 0: # if we retard throttle and have positive trim bias
                    if delta_throttle_1 > U1[IDX_THR1]: # if we move the throttle a lot, limit washout to zero
                        U1[IDX_THR1] = 0
                        U1[IDX_THR2] = 0
                    else: #washout from the trim bias, the amount we moved the throttle lever
                        U1[IDX_THR1] += delta_throttle_1
                        U1[IDX_THR2] += delta_throttle_1
                        if U1[IDX_THR1] < 0 : U1[IDX_THR1] = 0 # ensure it is never less than zero
                        if U1[IDX_THR2] < 0 : U1[IDX_THR2] = 0

                # toggle ground spoilers if button is pressed -> this is done in joystick submodule
                # ground spoiler arm/disarm state is passaed through U1[IDX_GNDSP]
                # if spoilers are armed and we are on ground, set ground spoilers to open
                if (get_air_ground_state(calculate_gear_compression(this_AC_int.y[:9], current_AGL_m)) and (U1[IDX_GNDSP] == 1)):
                    U_man[IDX_GNDSP] = 0.4 # 40% lift dump
                else:
                    U_man[IDX_GNDSP] = 0.0 # close
                

                U_man = control_sat(U_man) # saturate commands


                # Calculate the time step for this specific loop iteration
                # If this is the first step, prev_dt might be 0, so guard against it
                actuator_dt = dt if dt > 0 else simdt 
                
                # Update the physical position of the surfaces with actuator dynamics
                U_actual = update_actuators(U_man, U_actual, actuator_dt, ACT_TAU)




                # Update thrust values (Engine deck results)
                U_actual[IDX_THR1] = e1_thrust
                U_actual[IDX_THR2] = e2_thrust

                # Interpolate for high lift devices influence
                high_lift_deltas = high_lift_interp(U_actual[IDX_FLAP])

                # -- Engines - multiprocessing
                # if there are new deck values, fetch them,
                # if not, keep what we have
                try:
                    eng_vals = results_queue.get(block=False) # block=False is equivalent to get_nowait()
                    if U1[IDX_THR1] < -0.5:
                        # TODO: make time constants variable???
                        e1_thrust = update_actuators(-eng_vals[0]['F_ram'] * LBF2N, U_actual[IDX_THR1], 0.1, 1.5) # FOR NOW, FIXED TIME CONSTANT AND DT
                    else:
                        e1_thrust = eng_vals[0]['Fn'] * LBF2N # deck returns lbf, need to convert to N
                    U_actual[IDX_THR1] = e1_thrust
                    e2_thrust = eng_vals[1]['Fn'] * LBF2N
                    U_actual[IDX_THR2] = e2_thrust
                except mp.queues.Empty:
                    pass              


                # -------------------------------------------------------
                # PREPARE FOR INTEGRATOR
                # -------------------------------------------------------
                prev_uvw = current_uvw
                current_rho = env.get_rho(current_alt_m)


                # -- Integrate Physics
                this_AC_int.set_f_params(U_actual, current_rho, current_AGL_m)
                this_AC_int.integrate(this_AC_int.t + dt)
                current_uvw = this_AC_int.y[0:3]

                # -- Integrate navigation equations
                current_NED = env.NED(this_AC_int.y[:3], this_AC_int.y[6:])
                this_wind = env.add_wind(WIND_NED_MPS, WIND_STDDEV_MPS)

                this_latlonh_int.set_f_params(current_NED + this_wind, current_latlon_rad[0], current_alt_m)
                this_latlonh_int.integrate(this_latlonh_int.t + dt) #in radians and alt in meters
                
                # store current state and time vector for next iteration
                current_latlon_rad = this_latlonh_int.y[0:2]
                current_alt_m = this_latlonh_int.y[2]
                if USE_FG_AS_TERRAIN_DB:
                    # use FlightGear as a terrain database...
                    current_AGL_m = current_alt_m - terrain_shared_data['ground_alt'] # this in meters
                else:
                    # alternate: use SRTM instead...
                    current_AGL_m = env.get_AGL(current_latlon_rad*RAD2DEG, current_alt_m, SIM_VISUAL_OFFSET)
                

                
                # -- FlightGear Output
                if send_frame_trigger:
                    # for efficiency, we will use this loop to 
                    # 1. send the datagram to FlightGear
                    # 2. check/toggle ground spoilers
                    # because we are not doing flexible structures sim, we do not need to log at full sim frame rate
                    # also, understood that altitude and rho will be "ahead" one step

                    # -- Send data to FlightGear
                    # it is easier to calculate body accelerations instead of reaching into the RCAM function
                    if dt == 0:
                        body_accels = np.zeros(prev_uvw.shape)
                    else:
                        body_accels = (current_uvw - prev_uvw) / dt
                    # add gravity
                    g_b = np.array([-G * np.sin(this_AC_int.y[IDX_THETA]),
                                    G * np.cos(this_AC_int.y[IDX_THETA]) * np.sin(this_AC_int.y[IDX_PHI]),
                                    G * np.cos(this_AC_int.y[IDX_THETA]) * np.cos(this_AC_int.y[IDX_PHI])])
                    body_accels = body_accels + g_b
                    body_accels[2] = -body_accels[2] # FG expects Z-up

                    # set values and send frames
                    net.set_FDM(my_fgFDM, this_AC_int.y, 
                            control_norm(U_actual), 
                            current_latlon_rad, 
                            current_alt_m,
                            body_accels)
                    my_pack = my_fgFDM.pack()
                    try:
                        fdm_packet_queue.put_nowait(my_pack)
                    except queue.Full:
                        # This should rarely happen unless the network thread
                        # is completely stalled. We can just drop the frame.
                        pass
                    send_frame_trigger = False

                

                # -- Engine Deck trigger
                # deck calculation is CPU intensive
                # and engine dynamics are slow
                # so we only trigger engine deck calc at a much slower frame rate
                if calc_eng_trigger:
                    # Trigger Engine Deck Calculation
                    on_ground = get_air_ground_state(calculate_gear_compression(this_AC_int.y[:9], current_AGL_m))
                    if jobs_queue.empty():
                        #print(f"[Main Process] Triggering new engine calculation...{VA(current_uvw)*MS2KT:.2f}, {current_alt_m*M2FT:.1f}")
                        new_job = (current_alt_m*M2FT, ISA.Vt2M(env.VA(current_uvw)*MS2KT, current_alt_m*M2FT), U_man[IDX_THR1], U_man[IDX_THR2], on_ground, time.perf_counter())
                        try:
                            jobs_queue.put(new_job, block=False)
                            eng_time_adder = 0
                        except mp.queues.Full:
                            #print("[Main Process] Engine is busy, skipping this trigger.")
                            pass
                    else:
                        #print("[Main Process] Engine is still busy with a pending job, skipping this trigger.")
                        pass
                    calc_eng_trigger = False

                
                if datalog_trigger:
                    # -- Data Logging
                    internals = RCAM_observe(this_AC_int.y, U_actual, current_rho, current_AGL_m) # get internal FDM states
                    # engine parameters:
                    eng1_states = np.zeros(len(ENG_LOG_PARAMETERS))
                    eng2_states = np.zeros(len(ENG_LOG_PARAMETERS))
                    if eng_vals:
                        for idx, p in enumerate(ENG_LOG_PARAMETERS):
                            eng1_states[idx] = eng_vals[0][p]
                            eng2_states[idx] = eng_vals[1][p]
                    data_collector.append(np.concatenate((this_AC_int.y, this_latlonh_int.y, current_NED + this_wind, U_man, internals, eng1_states, eng2_states)))
                    t_vector_collector.append(this_AC_int.t)
                    datalog_trigger = False

                
                # -- Next frame setup
                frame_count += 1

                # DEBUG ONLY - ################################################################################################################################################
                # print out stuff every so often
                if (frame_count % 100) == 0:
                    #print(f'frame: {frame_count}, time: {this_AC_int.t:0.2f}, theta:{this_AC_int.y[7]:0.6f}, Elev:{this_joy.get_axis(1) * elev_factor}')
                    #print(f'frame: {frame_count}, time: {this_AC_int.t:0.2f}, lat:{current_latlon_rad[0]:0.6f}, lon:{current_latlon_rad[1]:0.6f}')
                    #print(f'time: {this_AC_int.t:0.2f}, N:{current_NED[0]:0.3f}, E:{current_NED[1]:0.3f}, D:{current_NED[2]:0.3f}')
                    #print(f'time: {this_AC_int.t:0.1f}s, dt: {this_AC_int.t - last_frame_time:0.2f}s Vcas_2fg:{my_fgFDM.get("vcas"):0.1f}KCAS, U_man={U_man[3]:0.3f},{U_man[4]:0.3f}, U1={U1[3]:0.3f},{U1[4]:0.3f}, E12T={U_actual[IDX_THR1]:0.0f},{U_actual[IDX_THR2]:0.0f}N, AGL={current_AGL_m*M2FT:0.0f}ft, alt={current_alt_m*M2FT:0.1f}, gnd_sp_arm:{gnd_spoilers_armed}')
                    #print(f'fr#:{frame_count}, time: {this_AC_int.t:0.1f}s, alt={current_alt_m*M2FT:0.0f}, E12T={e1_thrust:0.0f},{e2_thrust:0.0f}N, AGL={current_AGL_m*M2FT:0.0f}, {open_gnd_spoiler=}, {gnd_spoilers_armed=}, {toggle_gnd_spoiler_debounce=}, {U_man[IDX_GNDSP]=}')
                    #print(f'time: {this_AC_int.t:0.1f}s, alt={current_alt_m*M2FT:0.0f}, U_man={U_man[3]:0.3f},{U_man[4]:0.3f}, U1={U1[3]:0.3f},{U1[4]:0.3f}, E12T={U_actual[IDX_THR1]:0.0f},{U_actual[IDX_THR2]:0.0f}N, Flap_U1={U1[IDX_FLAP]}, U1GNDSP={U1[IDX_GNDSP]:0.4f}, UmanGNDSP={U_man[IDX_GNDSP]:0.4f}, UactualGNDSP={U_actual[IDX_GNDSP]}')
                    #print(f'time: {this_AC_int.t:0.1f}s, alt={current_alt_m*M2FT:0.0f}, U_man={U_man[3]:0.3f},{U_man[4]:0.3f}, U1={U1[3]:0.3f},{U1[4]:0.3f}, E12T={U_actual[IDX_THR1]:0.0f},{U_actual[IDX_THR2]:0.0f}N, Flap_U1={U1[IDX_FLAP]}, U1GEAR={U1[IDX_GEAR]:0.4f}, UmanGEAR={U_man[IDX_GEAR]:0.4f}, UactualGEAR={U_actual[IDX_GEAR]}')
                    #print(f'time: {this_AC_int.t:0.1f}s, alt={current_alt_m*M2FT:0.0f}, U_man={U_man[3]:0.3f},{U_man[4]:0.3f}, U1={U1[3]:0.3f},{U1[4]:0.3f}, E12T={U_actual[IDX_THR1]:0.0f},{U_actual[IDX_THR2]:0.0f}N, Flap_U1={U1[IDX_FLAP]}, U1THR1={U1[IDX_THR1]:0.4f}, UmanTHR1={U_man[IDX_THR1]:0.4f}, UactualTHR1={U_actual[IDX_THR1]}')
                    print(f'time: {this_AC_int.t:0.1f}s, alt={current_alt_m*M2FT:0.0f}, E12T={U_actual[IDX_THR1]:0.0f}, U1GEAR={U1[IDX_GEAR]:0.4f}, UmanGEAR={U_man[IDX_GEAR]:0.4f}, UactualGEAR={U_actual[IDX_GEAR]}, U1FLAP={U1[IDX_FLAP]:0.4f}, UmanFLAP={U_man[IDX_FLAP]:0.4f}, UactualFLAP={U_actual[IDX_FLAP]}, HLDeltas={high_lift_deltas}')
                    last_frame_time = this_AC_int.t
                #################################################################################################################################################################

                # reset integrator timestep counter
                # performance check
                # if you want to check for dropped frames, uncomment below
                #if dt > simdt:
                    #actual_fps = 1.0 / dt
                    #print(f"[WARNING] Sim Loop Lag: Target {SIM_LOOP_HZ}Hz | Actual {actual_fps:.1f}Hz | Calc Time {calc_duration*1000:.2f}ms")
                prev_dt = dt
                dt = 0
                run_sim_loop = False

            #check/set frame triggers
            if fg_time_adder >= fgdt:
                fg_time_adder = 0
                dt = sim_time_adder
                send_frame_trigger = True

            # check/set engine calc trigger
            if eng_time_adder >= deckdt:
                eng_time_adder = 0
                calc_eng_trigger = True

            # check/set datalog trigger
            if datalog_time_adder >= datalogdt:
                datalog_time_adder = 0
                datalog_trigger = True

            # parking lot
            # it will keep off the simulation loop while time does not catch up with the desired "simdt".
            # continuously adds time until that point, then releases the semaphore to run the sim
            if sim_time_adder >= simdt:
                dt = sim_time_adder
                sim_time_adder = 0
                run_sim_loop = True

            # end-of-frame 
            end = time.perf_counter()
            this_frame_dt = end - start
            fg_time_adder += this_frame_dt
            sim_time_adder += this_frame_dt
            eng_time_adder += this_frame_dt
            datalog_time_adder += this_frame_dt


    if OFFLINE == False:
        # close threads
        # -- Stop TX threads
        print()
        print("Shutting down network threads...")
        fdm_packet_queue.put(None)  # Send the shutdown signal
        network_thread.join(timeout=1.0) # Wait for the thread to finish
        for s in socks:
            s.close()
                
        # -- Stop RX thread
        terrain_shutdown_queue.put(True)
        terrain_thread.join(timeout=1.0)

        # close engine process
        jobs_queue.put(None)
        engine_process.join(timeout=2.0) # Wait for the worker process to finish
        # It's good practice to terminate if it doesn't join cleanly
        if engine_process.is_alive():
            print("[Main Process] Worker did not shut down cleanly. Terminating.")
            engine_process.terminate()
        
    # save data to disk
    helpers.save2disk(RESULTS_FILE, x_data=np.array(t_vector_collector), y_data=np.array(data_collector), header=full_header, skip=0)
    fig1 = helpers.make_plots(x_data=np.array(t_vector_collector), y_data=np.array(data_collector), header=full_header, skip=0)
    plt.show();
