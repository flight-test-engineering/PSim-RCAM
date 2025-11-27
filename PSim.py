# FGROOT = /usr/share/games/flightgear

# DRI_PRIME=1 fgfs --airport=SBGP  --aircraft=Embraer170 --aircraft-dir=./FlightGear/Aircraft/E-jet-family/ --native-fdm=socket,in,60,,5500,udp --fdm=null --enable-hud --in-air --fog-disable --shading-smooth --texture-filtering=4 --timeofday=morning --altitude=2500 --prop:/sim/hud/path[1]=Huds/NTPS.xml
# DRI_PRIME=1 fgfs --airport=LOWI  --aircraft=Embraer170 --aircraft-dir=./FlightGear/Aircraft/E-jet-family/ --native-fdm=socket,in,60,,5500,udp --fdm=null --enable-hud --in-air --fog-disable --shading-smooth --texture-filtering=4 --timeofday=morning --altitude=2500 --prop:/sim/hud/path[1]=Huds/fte.xml 2>/dev/null


# FG with JSBSim:
# DRI_PRIME=1 fgfs --airport=SBGP  --aircraft=Embraer170 --aircraft-dir=./FlightGear/Aircraft/E-jet-family/  --enable-hud  --fog-disable --shading-smooth --texture-filtering=4 --timeofday=morning
# DRI_PRIME=1 fgfs --airport=KSFO --runway=28R  --aircraft=757-200-RB211 --aircraft-dir=~/.fgfs/Aircraft/org.flightgear.fgaddon.stable_2020/Aircraft/757-200  --enable-hud  --fog-disable --shading-smooth --texture-filtering=4 --timeofday=morning

# "v" muda o visual
# https://wiki.flightgear.org/Command_line_options

# prototype for moving engine deck to a dedicated CPU core

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

The program runs the integration loop as fast as possible, adjusting the integration steps to the available computing cycles
It uses Numba to speed up the main functions involved in the integration loop

Output is sent to FlightGear (FG), over UDP, at a reduced frame rate (60)
The FG interface uses the class implemented by Andrew Tridgel (fgFDM):
https://github.com/ArduPilot/pymavlink/blob/master/fgFDM.py

currently, the UDP address is set to the local machine.

Run this program and from a separate terminal, start FG with one of the following commands (depending on the aircraft addons installed):
fgfs --airport=KSFO --runway=28R --aircraft=ufo --native-fdm=socket,in,60,,5500,udp --fdm=null
fgfs --airport=KSFO --runway=28R --aircraft=Embraer170 --aircraft-dir=./FlightGear/Aircraft/E-jet-family/ --native-fdm=socket,in,60,,5500,udp --fdm=null
fgfs --airport=KSFO --runway=28R --aircraft=757-200-RB211 --aircraft-dir=~/.fgfs/Aircraft/org.flightgear.fgaddon.stable_2020/Aircraft/757-200 --native-fdm=socket,in,60,,5500,udp --fdm=null
fgfs --airport=KSFO --runway=28R --aircraft=757-200-RB211 --aircraft-dir=~/.fgfs/Aircraft/org.flightgear.fgaddon.stable_2020/Aircraft/757-200 --native-fdm=socket,in,60,,5500,udp --fdm=null --enable-hud --turbulence=0.5 --in-air  --enable-rembrandt
DRI_PRIME=1 fgfs --airport=LOWI  --aircraft=Embraer170 --aircraft-dir=./FlightGear/Aircraft/E-jet-family/ --native-fdm=socket,in,60,,5500,udp --fdm=null --enable-hud --in-air --fog-disable --shading-smooth --texture-filtering=4 --timeofday=morning --altitude=2500 --prop:/sim/hud/path[1]=Huds/fte.xml 2>/dev/null

If a joystick is detected, then inputs come from it
Otherwise, offline simulation is run

this version separates FlightGear packet sending to its own thread to improve speed and eliminate frame rate stutter


TODO:
    1) add engine dynamics (spool up/down)
    2) add atmospheric disturbances/turbulence
    3) add other actuator dynamics
    4) save/read trim point
    5) fuel detot / inertia update


'''

# imports
import numpy as np
from scipy import integrate
from scipy.optimize import minimize # for trimming routine
from numba import jit
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

import ISA_module as ISA # International Standard Atmosphere library
from engine_deck import Turbofan_Deck

import time
import csv
import sys
import json

sys.path.insert(1, '../')

from fgDFM import * # FlightGear comm class
import socket

# threading for FG comms
import threading
import queue

# multiprocessing for engine deck
import multiprocessing as mp

import pygame #joystick interface

# ############################################################################
# Consolidated Constants
# ############################################################################
# .. Physical and Mathematical Constants ..
G = 9.81  # Gravity, m/s^2
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi
FT2M = 0.3048
M2FT = 1 / FT2M
KT2MS = 0.51444444 #knots to meters per second
MS2KT = 1 / KT2MS
LBF2N = 4.44822 # from pound force to N


# ############################################################################
# Aircraft Parameter Loader
# ############################################################################

def load_aircraft_parameters(filepath: str) -> dict:
    """
    Loads aircraft parameters from a JSON file, processes them, and returns
    them as a dictionary of constants ready for the simulation.
    """
    with open(filepath, 'r') as f:
        params = json.load(f)

    print(f"Loading aircraft model: {params['aircraft_name']}")

    consts = {}

    # .. Nominal vehicle constants ..
    # (TP-088-3, p. 9, para 2.2, table 2.4)
    mg = params['mass_and_geometry']

    # (TP-088-3, p. 9, para 2.2, table 2.5)
    consts['M'] = mg['mass'] # kg - total mass

    # (TP-088-3, p. 9, para 2.2, table 2.4)
    consts['CBAR'] = mg['wing_mean_aerod_chord'] # m - mean aerodynamic chord
    consts['S'] = mg['wing_area'] # m^2 - wing area
    consts['ST'] = mg['tail_area'] # m^2 - tail area
    consts['LT'] = mg['tail_arm'] # m - tail aerodynamic center distance to CG

    # Derived Geometry (CG, AC)
    cgap = params['cg_and_ac_positions']
    # .. centre of gravity position ..
    consts['XCG'] = cgap['xcg'] * consts['CBAR'] # m - x pos of CG
    consts['YCG'] = cgap['ycg'] # m - y pos of CG
    consts['ZCG'] = cgap['zcg'] * consts['CBAR'] # m - z pos of CG
    # .. aerodynamic centre position .. (TP-088-3, p. 9, para 2.2, table 2.4)
    consts['XAC'] = cgap['xac'] * consts['CBAR']
    consts['YAC'] = cgap['yac']
    consts['ZAC'] = cgap['zac']

    # .. engines point of thrust application .. (TP-088-3, p. 9, para 2.2, table 2.4)
    ep = params['engine_positions']
    consts['XAPT1'], consts['YAPT1'], consts['ZAPT1'] = ep[0]['x'], ep[0]['y'], ep[0]['z']
    consts['XAPT2'], consts['YAPT2'], consts['ZAPT2'] = ep[1]['x'], ep[1]['y'], ep[1]['z']

    # .. aerodynamic properties - lift ..
    ac = params['aerodynamic_coeffs']
    consts['DEPSDA'] = ac['depsda'] # rad/rad - change in downwash wrt alpha # (TP-088-3, p. 14, para 2.3.4, eq 2.30)
    consts['ALPHA_L0'] = ac['alpha_l0_deg'] * DEG2RAD # rad - zero lift AOA
    consts['ALPHA_SWITCH'] = ac['alpha_switch_deg'] * DEG2RAD # rad - kink point of lift slope
    # these values are from the 1997 RCAM revision
    consts['N'] = ac['lift_slope_n'] # adm - slope of linear region of lift slope # (TP-088-3, p. 14, para 2.3.4, eq 2.25)
    consts['A3'] = ac['lift_poly_coeffs']['a3'] # adm - coeff of alpha^3
    consts['A2'] = ac['lift_poly_coeffs']['a2'] # adm - coeff of alpha^2
    consts['A1'] = ac['lift_poly_coeffs']['a1'] # adm - coeff of alpha^1
    consts['A0'] = ac['lift_poly_coeffs']['a0'] # adm - coeff of alpha^0
    # ... tail ...
    # adm - slope of linear region of TAIL lift slope
    # (TP-088-3, p. 15, eq 2.27)
    consts['NT'] = ac['htail_coeffs']['nt'] 
    # adm multiplier for tail dynamic downwash response wrt pitch rate
    # (TP-088-3, p. 15, eq 2.28)
    consts['EPSILON_DOT'] = ac['htail_coeffs']['epsilon_dot'] 
    

    # .. aerodynamic properties - drag ..
    drag_coeffs = params['drag_coeffs']
    # (TP-088-3, p. 14, para 2.3.4, eq 2.31)
    consts['CDMIN'] = drag_coeffs['cdmin'] # adm - CD min - bottom of CDxALpha curve
    consts['D1'] = drag_coeffs['d1'] # adm - coeff of alpha^2
    consts['D0'] = drag_coeffs['d0'] # adm - coeff of alpha^0

    # .. aerodynamic properties - side force ..
    # (TP-088-3, p. 14, para 2.3.4, eq 2.32)
    side_force_coeffs = params['side_force_coeffs']
    consts['CY_BETA'] = side_force_coeffs['cy_beta'] # adm - side force coeff with sideslip
    consts['CY_DR'] = side_force_coeffs['cy_dr'] # adm - side force coeff with rudder deflection

    # .. aerodynamic properties - moment coefficients ..
    # (TP-088-3, p. 14, para 2.3.4, eq 2.33)
    moment_coeffs = params['moment_coeffs']
    consts['C_l_BETA'] = moment_coeffs['c_l_beta'] # adm - roll moment due to beta
    consts['C_m_ALPHA'] = moment_coeffs['c_m_alpha'] # adm - pitch moment due to alpha
    consts['C_n_BETA'] = moment_coeffs['c_n_beta'] * RAD2DEG # per RCAM doc, need to mult by 180/pi


    # ... roll, pitch, yaw moments with rates ..,
    # (TP-088-3, p. 14, para 2.3.4, eq 2.33)
    moment_rate_coeffs = params["pqr_moment_coeffs"]
    consts['C_l_P'] = moment_rate_coeffs['c_l_p']
    consts['C_l_Q'] = moment_rate_coeffs['c_l_q']
    consts['C_l_R'] = moment_rate_coeffs['c_l_r']
    consts['C_m_P'] = moment_rate_coeffs['c_m_p']
    consts['C_m_Q'] = moment_rate_coeffs['c_m_q']
    consts['C_m_R'] = moment_rate_coeffs['c_m_r']
    consts['C_n_P'] = moment_rate_coeffs['c_n_p']
    consts['C_n_Q'] = moment_rate_coeffs['c_n_q']
    consts['C_n_R'] = moment_rate_coeffs['c_n_r']

    # ... roll, pitch, yaw moments with controls ...
    # (TP-088-3, p. 14, para 2.3.4, eq 2.33)
    moment_controls_coeffs = params["controls_moment_coeffs"]
    consts['C_l_DA'] = moment_controls_coeffs['c_l_da']
    consts['C_l_DE'] = moment_controls_coeffs['c_l_de']
    consts['C_l_DR'] = moment_controls_coeffs['c_l_dr']
    consts['C_m_DA'] = moment_controls_coeffs['c_m_da']
    consts['C_m_DE'] = moment_controls_coeffs['c_m_de']
    consts['C_m_DR'] = moment_controls_coeffs['c_m_dr']
    consts['C_n_DA'] = moment_controls_coeffs['c_n_da']
    consts['C_n_DE'] = moment_controls_coeffs['c_n_de']
    consts['C_n_DR'] = moment_controls_coeffs['c_n_dr']

    # .. inertia tensor ..
    mass = consts['M']
    tensor_per_unit_mass = np.array(params['inertia']['tensor_per_unit_mass'])
    consts['INERTIA_TENSOR_b'] = mass * tensor_per_unit_mass # each element in m2 - (TP-088-3, p. 12, para 2.3.1.2, eq 2.11)
    consts['INV_INERTIA_TENSOR_b'] = np.linalg.inv(consts['INERTIA_TENSOR_b'])

    # .. Control Surface and Throttle Limits ..
    # (TP-088-3, p. 19, para 2.5)
    # however, the 1997 RCAM has [0.5 and 10] * pi/180 for the throttles
    # this equates to a 0.35 thrust to weight ratio
    # as per CLum's video @ 7:45

    cl_deg = params['control_limits_deg']
    consts['U_LIMITS_RAD'] = {k: (v[0] * DEG2RAD, v[1] * DEG2RAD) for k, v in cl_deg.items()}
    consts['U_LIMITS_MIN'] = np.array([lim[0] for lim in consts['U_LIMITS_RAD'].values()])
    consts['U_LIMITS_MAX'] = np.array([lim[1] for lim in consts['U_LIMITS_RAD'].values()])

    return consts


# ############################################################################
# Load Aircraft Parameters into Global Scope
#
# Numba's JIT compiler captures global variables when a function is first
# compiled. By loading our parameters into the global scope, we make them
# available to the performance-critical `RCAM_model` and `control_sat`
# functions without needing to pass them as arguments on every call.
# ############################################################################

try:
    # Unpack the dictionary into global variables
    globals().update(load_aircraft_parameters('rcam_parameters.json'))
except FileNotFoundError:
    print("ERROR: `rcam_parameters.json` not found. Please create it.")
    sys.exit(1)
except (KeyError, json.JSONDecodeError) as e:
    print(f"ERROR: Invalid format in `rcam_parameters.json`: {e}")
    sys.exit(1)


CF34 = Turbofan_Deck('CF34_deck_v4.csv')

def engine_worker(jobs_queue, results_queue):
    """
    This is the worker function that runs in its own PROCESS.
    """
    print("[Engine Process] Worker started and waiting for jobs.")
    while True:
        try:
            job = jobs_queue.get()
            if job is None:
                print("[Engine Process] Shutdown signal received.")
                break
            
            # unpack the arguments
            job_alt, job_MN, job_E1_TLA, job_E2_TLA, job_time = job
            E1_res = CF34.run_deck(job_alt, job_MN, job_E1_TLA, job_time)
            E2_res = CF34.run_deck(job_alt, job_MN, job_E2_TLA, job_time)
            results = (E1_res, E2_res)
            
            # The logic for clearing old results remains the same.
            if not results_queue.empty():
                try:
                    results_queue.get_nowait()
                except mp.queues.Empty:
                    pass
            results_queue.put(results)

        except Exception as e:
            print(f"[Engine Process] Error: {e}")
            break
    print("[Engine Process] Worker has shut down.")



# ############################################################################
# Helper Functions
# ############################################################################

def network_worker(socks, packet_queue, fg_addresses):
    """
    This function runs in a separate thread. It waits for FDM packets to appear
    in the queue and sends them over UDP.
    Inputs:
        socks: list with network open socks
        packet_queue: a Python multithread queue that received the packets to be sent
        fg_addresses: list of tuples with IP address and port
    """
    print("Starting network thread", end="")
    while True:
        try:
            # Block until a packet is available in the queue.
            # A timeout is added to allow for graceful shutdown checks if needed,
            # though using a sentinel value is cleaner.
            packet = packet_queue.get()

            # THREADING: Check for the sentinel value (None) to signal shutdown.
            if packet is None:
                print("Network thread received shutdown signal.")
                break

            # Send the packet to FlightGear.
            for idx, s in enumerate(socks):
                s.sendto(packet, fg_addresses[idx])

        except queue.Empty:
            # This will only happen if a timeout is used in get()
            continue
        except Exception as e:
            print(f"Error in network thread: {e}")
            break
    print("Network thread finished.")

def make_plots(x_data=np.array([0,1,2]), y_data=np.array([0,1,2]), \
                header=['PSim_Time', 'u', 'v', 'w', 'p', 'q', 'r', 'phi', 'theta', 'psi', 'lat', 'lon', 'h', 'V_N', 'V_E', 'V_D', 'dA', 'dE', 'dR', 'dT1', 'dT2'], skip=0):

    '''
    Function to plot results.
    Inputs: 
        x_data - time vector
        y_data - n-dimentional array with parameters to be plotted
        header has standard sequence of parameters/labels as generated by simulator
        skip: number of header items to skip
    '''
    plotlist = []

    plt.ioff()
    plt.clf()
    counter = 1
    myfig = plt.figure(figsize = (16,(y_data.shape[1]*4)))
    myfig.patch.set_edgecolor('w')
    plt.subplots_adjust(hspace = 0.0)
    for y_data_idx in range(y_data.shape[1]):
        strip_chart_y_data = y_data[:,y_data_idx]
        ax = myfig.add_subplot(y_data.shape[1], 1, counter)
        ax.plot(x_data, strip_chart_y_data)
        plt.ylabel(header[y_data_idx+skip])
        plt.grid(True)
        counter += 1
    return myfig


def save2disk(filename, x_data=np.array([0,1,2]), y_data=np.array([0,1,2]), \
                header=['u', 'v', 'w', 'p', 'q', 'r', 'phi', 'theta', 'psi', 'lat', 'lon', 'h', 'V_N', 'V_E', 'V_D', 'dA', 'dE', 'dR', 'dT1', 'dT2'], skip=0):
    '''
    saves data to disk
    '''
    with open(filename, 'w') as f:
        y_dim = y_data.shape[1]
        data_header = header[skip:y_dim]
        data_header.insert(0, 'PSim_Time')
        writer = csv.writer(f)
        writer.writerow(data_header)
        for idx, row in enumerate(y_data):
            row_list = row.tolist()
            row_list.insert(0, x_data[idx].astype('float'))
            writer.writerow(row_list)

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


def get_rho(altitude:float)->float:
    '''
    calculate the air density given an altitude in meters
    '''
    return ISA.rho_SL * ISA.sigma(altitude * M2FT) # ISA expects alt in ft

@jit(nopython=True)
def fpa(V_NED)->float:
    '''
    returns flight path angle
    input is a vector with North, East and Down velocities
    '''
    return np.arctan2(-V_NED[2], np.sqrt(V_NED[0]**2 + V_NED[1]**2))


def course(V_NED)->float:
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


def get_doublet(t_vector, t=0, duration=1, amplitude=0.1):
    '''
    calculates a doublet input
    inputs:
        t_vector: time vector
        t: value at which the doublet should start
        duration: duration of the high/low input states
        amplitude: multiplication factor to set amplitude
    returns:
        doublet vector
    '''
    rise_idx = np.argmax(t_vector>=t)
    drop_idx = np.argmax(t_vector >=(t+duration/2))
    zero_idx = np.argmax(t_vector >=(t+duration))
    res = np.zeros(t_vector.shape)
    res[rise_idx:drop_idx] = 1 * amplitude
    res[drop_idx:zero_idx] = -1 * amplitude
    return res


def get_step(t_vector, t=0, amplitude=0.1):
    '''
    calculates a step input
    inputs:
        t_vector: time vector
        t: value at which the doublet should start
        amplitude: multiplication factor to set amplitude
    returns:
        step vector
    '''
    rise_idx = np.argmax(t_vector>=t)
    res = np.zeros(t_vector.shape)
    res[rise_idx:] = 1 * amplitude
    return res


def create_cmd(t_vector=np.zeros(5), input_channel='ail', cmd_type='doublet', at_time=0.0, duration=1.0, amplitude=0.0):
    '''
    helper function to create a doublet or step in a channel
    inputs:
        t_vector: time vector
        input_channel: selector for axis (see if/else below)
        cmd_type: selector for doublet or step
        at_time: value at which the inpute should start
        duration: duration of the high/low input states
        amplitude: multiplication factor to set amplitude
    returns:
        input_ch_number: integer with the index of command to be added to integration loop
        cmd: vector with command
    '''
    
    if input_channel == 'ail':
        input_ch_num = 0
    elif input_channel == 'elev':
        input_ch_num = 1
    elif input_channel == 'rud':
        input_ch_num = 2
    elif input_channel == 'thru':
        input_ch_num = 3
    elif input_channel == 'none' or input_channel == 'None':
        cmd = np.zeros(t_vector.shape)
        input_ch_num = 0
    else:
        input_ch_num = -1
    
    
    if cmd_type=='doublet' and input_ch_num>=0:
        cmd = get_doublet(t_vector, t=at_time, duration=duration, amplitude=amplitude)
    elif cmd_type=='step' and input_ch_num>=0:
        cmd = get_step(t_vector, t=at_time, amplitude=amplitude)
    else:
        print('error - command type not recognized')
        cmd = np.zeros(t_vector.shape)
        input_ch_num = 0
        
    return input_ch_num, cmd

def set_FDM(this_fgFDM, X, U_norm, latlon, alt, body_accels):
    '''
    function to set the current time step data to be sent to FlightGear
    inputs are:
    X - states
    U - controls
    latlon - in radians
    alt - in meters
    NED - velocities in m/s
    '''
    this_fgFDM.set('phi', X[6])
    this_fgFDM.set('theta', X[7])
    this_fgFDM.set('psi', X[8])

    this_fgFDM.set('phidot', X[3])
    this_fgFDM.set('thetadot', X[4])
    this_fgFDM.set('psidot', X[5])
    
    # this sets units to kts because the HUD does not apply any conversions to the speed
    # if we send speed in fps as the API requires, the HUD displays wrong value
    this_fgFDM.set('vcas', ISA.Vt2Vc(VA(X[:3]), alt*M2FT) * MS2KT) 
    this_fgFDM.set('cur_time', int(time.perf_counter() ), units='seconds')
    this_fgFDM.set('latitude', latlon[0], units='radians')
    this_fgFDM.set('longitude', latlon[1], units='radians')
    this_fgFDM.set('altitude', alt, units='meters')

    this_fgFDM.set('left_aileron', -U_norm[0])
    this_fgFDM.set('right_aileron', +U_norm[0])
    this_fgFDM.set('elevator', U_norm[1])
    this_fgFDM.set('rudder', -U_norm[2])

    this_fgFDM.set('A_X_pilot', body_accels[0], units='mpss')
    this_fgFDM.set('A_Y_pilot', body_accels[1], units='mpss')
    this_fgFDM.set('A_Z_pilot', body_accels[2], units='mpss')



def get_joy_inputs(joystick, U_trim, fr, trim_params, joy_factors):
    '''
    function that will read joystick positions and adjust controls:
    1. joy will change controls on top of trim point
    2. trim settings (buttons) will change trim point
    3. engine does not have trim function, but depending on
    button pressed, throttle should be commanded left/right/both
    '''
    U = np.zeros(U_trim.shape)

    # # # TRIM

    # multipliers to adjust how much trim is added per integration step.
    # --- TRIM ---
    pitch_trim_step = trim_params['pitch'] / fr
    aileron_trim_step = trim_params['aileron'] / fr
    throttle_trim_step = trim_params['throttle'] / fr

    # read joystick button states for trimming
    zero_ail_rud_thr = joystick.get_button(0)
    pitch_dn = joystick.get_button(4)
    pitch_up = joystick.get_button(2)
    roll_rt = joystick.get_button(7)
    roll_lt = joystick.get_button(6)
    T1_fd = joystick.get_button(8)
    T1_af = joystick.get_button(10)
    T2_fd = joystick.get_button(9)
    T2_af = joystick.get_button(11)
    exit_signal = joystick.get_button(1)

    # if trigger is pressed, then zero out aileron, rudder states and make thrust equal on both sides
    if zero_ail_rud_thr == 1:
        U_trim[0] = 0.0
        U_trim[2] = 0.0
        U_trim[3] = U_trim[4]
    

    U_trim[0] = U_trim[0] - aileron_trim_step * roll_rt + aileron_trim_step * roll_lt
    U_trim[1] = U_trim[1] - pitch_trim_step * pitch_up  + pitch_trim_step * pitch_dn
    #U_trim[2] = U_trim[2] + rudder_trim_step *  - rudder_trim_step * roll_lt
    U_trim[3] = U_trim[3] - throttle_trim_step * T1_af + throttle_trim_step * T1_fd
    U_trim[4] = U_trim[4] - throttle_trim_step * T2_af + throttle_trim_step * T2_fd

    # # # JOYSTICK COMMAND

    # joystick constants/multipliers to adjust correct movement and amplitude
    U[0] = U_trim[0] + joystick.get_axis(0) * joy_factors['aileron']
    U[1] = U_trim[1] + joystick.get_axis(1) * joy_factors['elevator']
    U[2] = U_trim[2] + joystick.get_axis(2) * joy_factors['rudder']
    throttle_cmd = joystick.get_axis(3) * joy_factors['throttle_m'] + joy_factors['throttle_b'] # linearly map joystick inputs to RCAM
    U[3] = U_trim[3] + throttle_cmd
    U[4] = U_trim[4] + throttle_cmd


    return U, U_trim, exit_signal



# geodsy
# https://www.youtube.com/watch?v=4BJ-GpYbZlU
@jit(nopython=True)
def WGS84_MN(lat:float):
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
def latlonh_dot(V_NED, lat, h):
    '''
    V_NED: m/s
    lat: latitude in degrees (decimal)
    h: altitude in meters
    '''
    M, N = WGS84_MN(lat)
    return np.array([(V_NED[0]) / (M + h), 
                     (V_NED[1]) / ((N + h) * np.cos(lat)),
                     -V_NED[2]])


# controls
def control_norm(U:np.array) -> np.array:
    '''
    normalizes controls to be sent to FG
    inputs:
        U controls: positions (in radians)
        [U_lim: control limits (in radians) moved to global variable for speed]
    returns:
        vector with control positions normalized between 1 and -1
    '''
    U_norm = np.zeros_like(U)
    for i in range(len(U)):
        u_min, u_max = U_LIMITS_MIN[i], U_LIMITS_MAX[i]
        if U[i] < 0:
            U_norm[i] = U[i] / abs(u_min) if u_min != 0 else 0
        else:
            U_norm[i] = U[i] / u_max if u_max != 0 else 0
    return U_norm[:3] # Only return first 3 for FG FDM (ail, elev, rud)

@jit(nopython=True)
def control_sat(U:np.ndarray) -> np.ndarray:
    '''
    saturates the control inputs to maximum allowable in RCAM model
    '''
    return np.clip(U, U_LIMITS_MIN, U_LIMITS_MAX)



# ############################################################################
# RCAM flight dynamics model
# ############################################################################

@jit(nopython=True)
def RCAM_model(X:np.ndarray, U:np.ndarray, rho:float) -> np.ndarray:
    '''
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
            3: throttle 1 (%) ->>>E1 THRUST (N)
            4: throttle 2 (%) ->>>E2 THRUST (N)
        rho: density for current altitude (kg/m3)
    outputs:
        X_dot: derivatives of states (same order)
    '''
   
    # ------------------------- states ----------------------------------
    u, v, w = X[0], X[1], X[2] # m/s
    p, q, r = X[3], X[4], X[5] # rad/s
    phi, theta, psi = X[6], X[7], X[8] # rad

    # ----------------------- controls ----------------------------------
    da, de, dr, dt1, dt2 = U[0], U[1], U[2], U[3], U[4]
    
    #----------------- intermediate variables ---------------------------
    # airspeed
    Va = np.sqrt(u**2 + v**2 + w**2) # m/s
    
    # alpha and beta
    #np.arctan2 --> y, x
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
    CL_wb = N * (alpha - ALPHA_L0) if alpha <= ALPHA_SWITCH else A3 * alpha**3 + A2 * alpha**2 + A1 * alpha + A0
    
    # CL thrust
    epsilon = DEPSDA * (alpha - ALPHA_L0)
    alpha_t = alpha - epsilon + de + EPSILON_DOT * q * LT / Va
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
    
    dCMdx = (CBAR / Va) * np.array([[C_l_P, C_l_Q, C_l_R], 
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
    #F1 = dt1 * M * G
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
    
    #---------------------- state derivatives --------------------------------
    
    # form F_b and calculate u, v, w dot
    F_b = Fg_b + FE_b + FA_b
    
    u_v_w_dot  = (1 / M) * F_b - np.cross(wbe_b, V_b)
    
    # form Mcg_b and calc p, q r dot
    Mcg_b = MAcg_b + MEcg_b
    
    p_q_r_dot = np.dot(INV_INERTIA_TENSOR_b, (Mcg_b - np.cross(wbe_b, np.dot(INERTIA_TENSOR_b , wbe_b))))
    
    # phi, theta, psi dot
    H_phi = np.array([[1.0, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                      [0.0, np.cos(phi), -np.sin(phi)],
                      [0.0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]], dtype=np.dtype('f8'))
    
    phi_theta_psi_dot = np.dot(H_phi, wbe_b)
    
    #--------------------- place in first order form --------------------------
    X_dot = np.concatenate((u_v_w_dot, p_q_r_dot, phi_theta_psi_dot))
    
    return X_dot


# ############################################################################
# Navigation Equations
# ############################################################################

# source:
# Christopher Lum - "The Naviation Equations: Computing Position North, East and Down"
# https://www.youtube.com/watch?v=XQZV-YZ7asE


@jit(nopython=True)
def NED(uvw, phithetapsi):
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
    

# ############################################################################
# Model Integration
# ############################################################################

# # # wrappers
    # Scipy's "integrate.ode" does not accept a numba/@jit(nopython=True) compiled function
    # therefore, we need to create dummy wrappers

def RCAM_model_wrapper(t, X, U, rho):
    return RCAM_model(X, U, rho)

def NED_wrapper(t, X, NED):
    return NED

def latlonh_dot_wrapper(t, X, V_NED, lat, h):
    return latlonh_dot(V_NED, lat, h)


# # # integrators
def ss_integrator(t_ini:float, X0:np.ndarray, U:np.ndarray, rho:float):
    
    '''
    single step integrator
    returns scipy object, initialized
    '''
    
    RK_integrator = integrate.ode(RCAM_model_wrapper)
    RK_integrator.set_integrator('dopri5') # Dormand-Prince explicit Runge-Kutta 5th order method
    RK_integrator.set_f_params(U, rho)
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

def trim_functional2(Z:np.ndarray, VA_trim, gamma_trim, side_speed_trim, phi_trim, psi_trim, rho_trim) -> np.dtype('f8'):
    '''
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

    ****
    method
    Q.T*H*Q
    with H = diagonal matrix of "1"s (equal weights for all states)
    
    returns:
        cost [float]
    '''

    X = Z[:9]
    U = Z[9:]
    
    X_dot = RCAM_model(X, control_sat(U), rho_trim)
    V_NED_current = NED(X_dot[:3], X_dot[3:6])
    
    VA_current = VA(X[:3])
    
    gamma_current = X[7] - np.arctan2(X[2], X[0]) # only valid for wings level case
     
    Q = np.concatenate((X_dot, [VA_current - VA_trim], [gamma_current - gamma_trim], [X[1] - side_speed_trim], [X[6] - phi_trim], [X[8] - psi_trim]))
    diag_ones = np.ones(Q.shape[0])
    H = np.diag(diag_ones)
    
    return np.dot(np.dot(Q.T, H), Q)

def trim_model(VA_trim=85.0, gamma_trim=0.0, side_speed_trim=0.0, phi_trim=0.0, psi_trim=0.0, rho_trim=1.225, 
               X0=np.array([85, 0, 0, 0, 0, 0, 0, 0.1, 0]), 
               U0=np.array([0, -0.1, 0, 0.08, 0.08])) -> np.ndarray:
    '''
    uses scipy minimize on functional to find trim point
    '''

    print(f'trimming with X0 = {X0}')
    print(f'trimming with U0 = {U0}')
    X0[0] = VA_trim
    Z0 = np.concatenate((X0, U0))
 
    result = minimize(trim_functional2, Z0, args=(VA_trim, gamma_trim, side_speed_trim, phi_trim, psi_trim, rho_trim),
               method='L-BFGS-B', options={'maxiter':5000,\
                                            'gtol':1e-25, 'ftol':1e-25, \
                                            'maxls':4000})

    return result.x, result.message



# ############################################################################
# Model Initialization
# ############################################################################

def initialize(VA_t=85.0, gamma_t=0.0, latlon=np.zeros(2), altitude=10000, psi_t=0.0):
    '''
    this initializes the integrators at a straight and level flight condition
    inputs:
        VA_t: true airspeed at trim (m/s)
        gamma_t: flight path angle at trim (rad)
        latlon: initial lat and long (rad)
        altitude: trim altitude (ft)
        psi_t: initial heading (rad)
    outputs:
        AC_integrator: aircraft integrator object
        X0: initial states found at trim point
        U0: initial commands found at trim point
        latlonh_integrator: navigation equation scipy object integrator
    '''
    t0 = 0.0 #intial time for integrators
    alt_m = altitude * FT2M
    rho_trim = get_rho(alt_m)

    print(f'initializing model with altitude {altitude} ft, rho={rho_trim}')
    

    latlonh0 = np.array([latlon[0]*DEG2RAD, latlon[1]*DEG2RAD, alt_m])

    # trim model
    res4, res4_status = trim_model(VA_trim=VA_t, gamma_trim=gamma_t, side_speed_trim=0, 
                                   phi_trim=0.0, psi_trim=psi_t*DEG2RAD, rho_trim=rho_trim)
    print(res4_status)
    X0 = res4[:9]
    U0 = res4[9:]
    print(f'initial states: {X0}')
    print(f'initial inputs: {U0}')

    # initialize integrators
    AC_integrator = ss_integrator(t0, X0, U0, rho_trim)
    
    NED0 = NED(X0[:3], X0[6:]) #uvw and phithetapsi
    
    latlonh_integrator = latlonh_int(t0, latlonh0, NED0)
    
    return AC_integrator, X0, U0, latlonh_integrator
    



# ############################################################################
# MAIN
# ############################################################################

if __name__ == "__main__":


############################################################################
    # INITIAL CONDITIONS (for trim)
    INIT_ALT_FT = 2100 #ft
    V_TRIM_MPS = 140 * KT2MS # m/s
    GAMMA_TRIM_RAD = 0.0 * DEG2RAD # RAD
    INIT_HDG_DEG = 82.0 # DEG
    # Lat/Lon
    #INIT_LATLON_DEG = np.array([37.6213, -122.3790]) #in degrees - the func initialize transforms to radians internally
    #INIT_LATLON_DEG = np.array([-21.7632, -48.4051]) #in degrees - SBGP
    INIT_LATLON_DEG = np.array([47.2548, 11.2963]) #in degrees - LOWI short final TFB
    # wind
    WIND_NED_MPS = np.array([0, 0, 0]) # (m/s), NED
    WIND_STDDEV_MPS = np.array([1, 1, 0]) # wind standard deviation, NED

###########################################################################
    # SIMULATION OPTIONS
    SIM_TOTAL_TIME_S = 10 * 60 # (s) total simulation time
    SIM_LOOP_HZ = 400 # (Hz) simulation loop frame rate throttling
    FG_OUTPUT_LOOP_HZ = 60 # (Hz) frames per second to be sent out to FG
    ENGINE_TRIGGER_S = 0.1
    PRINT_TRIGGER_S = 1.0 # trigger printing for awareness


###########################################################################
    # JOYSTICK SCALING FACTORS
    TRIM_PARAMS = { 'pitch': 0.01, 'aileron': 0.003, 'throttle': 0.01 } # Trim adjustment per second
    # linearly map the joystick input to the RCAM limits
    JOY_THROTTLE_LIMITS = [1, -1] # these are the limits for the throttle input for this specific joystick - full fwd = -1
    THROTTLE_MAP_M = (U_LIMITS_MAX[3] - U_LIMITS_MIN[3]) / (JOY_THROTTLE_LIMITS[1] - JOY_THROTTLE_LIMITS[0])
    THROTTLE_MAP_b = U_LIMITS_MAX[3] - THROTTLE_MAP_M * (JOY_THROTTLE_LIMITS[1]) # y = mx + b - equation for a line
    print(f'Joystick mapping: M={THROTTLE_MAP_M}, b={THROTTLE_MAP_b}, from JSON:{U_LIMITS_MAX[3]=}, {U_LIMITS_MIN[3]=} ')
    JOY_FACTORS = { 'aileron': -0.7, 'elevator': -0.5, 'rudder': -0.52, 'throttle_m': THROTTLE_MAP_M, 'throttle_b': THROTTLE_MAP_b } # specific for this joystick model


############################################################################
    # JOYSTICK INIT AND CHECK
    pygame.init() # automatically initializes joystick also

    # check if joystick is connected
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        print('Will run OFFLINE simulation, no joystick detected')
        OFFLINE = True
    else:
        print(f'found {joystick_count} joysticks connected.')
        this_joy = pygame.joystick.Joystick(0)
        print(f'{this_joy.get_name()}, axes={this_joy.get_numaxes()}')
        OFFLINE = False

    signals_header = ['u', 'v', 'w', 'p', 'q', 'r', 'phi', 'theta', 'psi', 'lat', 'lon', 'h', 'V_N', 'V_E', 'V_D', 'dA', 'dE', 'dR', 'dT1', 'dT2']

    
    # we only start the network and multiprocessing if doing online sim, at least for now
    if OFFLINE == False:
    ############################################################################
        # FLIGHTGEAR SOCKS
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

        fdm_packet_queue = queue.Queue() # async queue that will receive the packets

        # THREADING: Create and start the network worker thread.
        # It's a daemon thread, so it will exit automatically if the main program exits.
        network_thread = threading.Thread(
            target=network_worker,
            args=(socks, fdm_packet_queue, fg_addresses),
            daemon=True
        )
        try:
            network_thread.start()
            print("... started!")
        except Exception as e:
            print(f"Error in network thread: {e}")
            exit()


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
        #CF34 = Turbofan_Deck('deck_file_name', initial_time=time.perf_counter())
        # --- Multiprocessing Setup ---
        # MULTIPROCESSING: Use Queues from the multiprocessing module.
        # These queues handle the necessary serialization (pickling) to pass
        # data between process memory spaces.
        jobs_queue = mp.Queue(maxsize=1)
        results_queue = mp.Queue(maxsize=1)

        # MULTIPROCESSING: Create and start the engine as a Process, not a Thread.
        engine_process = mp.Process(
            target=engine_worker,
            args=(jobs_queue, results_queue),
            daemon=True  # Daemon processes are terminated when the parent exits
        )
        engine_process.start()

        current_thrust = 0 # FOR DEBUG ONLY


    
    # initializations
    data_collector, t_vector_collector = [], [] # data collectors
    
    prev_uvw = np.array([0,0,0])
    current_uvw = np.array([0,0,0])



    # aircraft initialization (includes trimming)
    this_AC_int, X_trim, U1, this_latlonh_int = initialize(VA_t=V_TRIM_MPS, gamma_t=GAMMA_TRIM_RAD, latlon=INIT_LATLON_DEG, altitude=INIT_ALT_FT, psi_t=INIT_HDG_DEG)
    U_man = U1.copy()
    # FOR DEBUGGING ONLY - TEMP FIX FOR NOW
    #e1_thrust = U_man[3] * 13000 * LBF2N# simple TLA * max thrust-ish
    #e2_thrust = U_man[4] * 13000 * LBF2N
    e1_thrust = 0
    e2_thrust = 0


    # frame variables
    current_alt_m = INIT_ALT_FT * FT2M # m
    current_latlon_rad = INIT_LATLON_DEG
    frame_count = 0

    last_frame_time = 0 # holds the time from last 100 frame to calc frame rate at print statement
    
    send_frame_trigger = False
    run_sim_loop = False # this is a semaphore. it will wait for the clock to reach the next "simdt" and run the simulation
    calc_eng_trigger = True

    fgdt = 1.0 / FG_OUTPUT_LOOP_HZ # (s) fg frame period
    simdt = 1 / SIM_LOOP_HZ # (s) desired simulation time step
    
    sim_time_adder, fg_time_adder = 0, 0 # counts the time between integration steps to trigger next simulation frame and FG dispatch
    eng_time_adder = 0 # loop to calculate engine
    
    dt = 0 # actual integration time step
    prev_dt = dt

    exit_signal = 0 # if joystick button #1 is pressed, ends simulation
    
    if OFFLINE:
        # code for offline simulation
        # create time vector
        t_vector = np.arange(0, SIM_TOTAL_TIME_S, simdt)
        print(f'Offline sim time vector: {t_vector[0]:.2f}s to {t_vector[-1]:.2f}s')

        # create control inputs
        sim_U = np.zeros((U_man.shape[0],t_vector.shape[0]))
        pitch_doublet = get_doublet(t_vector,t=5, duration=2, amplitude=0.2)
        roll_doublet = get_doublet(t_vector,t=15, duration=2, amplitude=0.2)
        yaw_doublet = get_doublet(t_vector,t=25, duration=4, amplitude=0.2)
        sim_U[0,:] = roll_doublet
        sim_U[1,:] = pitch_doublet
        sim_U[2,:] = yaw_doublet

        
        # single step integrate through each time step
        for idx, t in enumerate(t_vector):
            current_rho = get_rho(current_alt_m)
            
            # integrate 6-DOF
            this_AC_int.set_f_params(sim_U[:,idx], current_rho)
            this_AC_int.integrate(this_AC_int.t + simdt)

            # integrate navigation equations
            current_NED = NED(this_AC_int.y[:3], this_AC_int.y[6:])
            this_wind = add_wind(WIND_NED_MPS, WIND_STDDEV_MPS)
            this_latlonh_int.set_f_params(current_NED + this_wind, current_latlon_rad[0], current_alt_m)
            this_latlonh_int.integrate(this_latlonh_int.t + simdt) #in radians and alt in meters
            
            # store current state and time vector
            current_latlon_rad = this_latlonh_int.y[0:2] # store lat and long (RAD)
            current_alt_m = this_latlonh_int.y[2] # store altitude (m)
            data_collector.append(np.concatenate((this_AC_int.y, this_latlonh_int.y, current_NED + this_wind, sim_U[:,idx])))
            t_vector_collector.append(this_AC_int.t)

        print(f'Enf of simulation; {len(t_vector_collector)} time steps!')
        
    
    else:

        ##### SIMULATION LOOP #####

        while this_AC_int.t <= SIM_TOTAL_TIME_S and exit_signal == 0:
            # get clock
            start = time.perf_counter()

            if run_sim_loop:

                _ = pygame.event.get()
                
                # get density, inputs
                current_throttle = [U_man[3], U_man[4]] # keep track of throttle to zero-out the trim bias
                current_rho = get_rho(current_alt_m)
            
                U_man, U1, exit_signal = get_joy_inputs(this_joy, U1, SIM_LOOP_HZ, TRIM_PARAMS, JOY_FACTORS)
                
                # trim bias is always positive, so we washout if throttles move back
                delta_throttle_1 = U_man[3] - current_throttle[0]
                if delta_throttle_1 < 0 and U1[3] > 0: 
                    if delta_throttle_1 > U1[3]:
                        U1[3] = 0
                        U1[4] = 0
                    else:
                        U1[3] = U1[3] + delta_throttle_1
                        U1[4] = U1[4] + delta_throttle_1
                        if U1[3] < 0 : U1[3] = 0
                        if U1[4] < 0 : U1[4] = 0

                U_man = control_sat(U_man)

                # check if we have thrust etc from engine deck
                try:
                    # MULTIPROCESSING: The API is the same, but we import mp.queues for the exception.
                    eng_vals = results_queue.get(block=False) # block=False is equivalent to get_nowait()
                    e1_thrust = eng_vals[0]['Fn'] * LBF2N
                    e2_thrust = eng_vals[1]['Fn'] * LBF2N
                    #print(f"[Main Process] Updated engine results at t={t:.2f}s.")
                    #print(f"[Main Process] Updated engine results")
                except mp.queues.Empty:
                    pass

                # set current integration step commands, density and integrate aircraft states
                prev_uvw = current_uvw
                #change U_man to instead of having TLA, pass thrust
                U_man_plus_thrust = U_man.copy()

                # DEBUG ONLY - ARTIFICIALLY INCREASING THRUST TO MATCH 757
                # WEE NEED ABOUT 0.35 * 120000 * 9.81 Newtons
                # that is 3.5 ish
                U_man_plus_thrust[3] = e1_thrust * 3.5
                U_man_plus_thrust[4] = e2_thrust * 3.5

                this_AC_int.set_f_params(U_man_plus_thrust, current_rho)
                this_AC_int.integrate(this_AC_int.t + dt)
                current_uvw = this_AC_int.y[0:3]

                # integrate navigation equations
                current_NED = NED(this_AC_int.y[:3], this_AC_int.y[6:])
                this_wind = add_wind(WIND_NED_MPS, WIND_STDDEV_MPS)
                this_latlonh_int.set_f_params(current_NED + this_wind, current_latlon_rad[0], current_alt_m)
                this_latlonh_int.integrate(this_latlonh_int.t + dt) #in radians and alt in meters
                
                # store current state and time vector
                current_latlon_rad = this_latlonh_int.y[0:2] # store lat and long (RAD)
                current_alt_m = this_latlonh_int.y[2] # store altitude (m)
                data_collector.append(np.concatenate((this_AC_int.y, this_latlonh_int.y, current_NED + this_wind, U_man)))
                t_vector_collector.append(this_AC_int.t)
                
                # check for FG frame trigger
                if send_frame_trigger:
                    # it is easier to calculate body accelerations instead of reaching into the RCAM function
                    if dt == 0:
                        body_accels = (current_uvw - prev_uvw) / prev_dt
                    else:
                        body_accels = (current_uvw - prev_uvw) / dt
                    # add gravity
                    g_b = np.array([-G * np.sin(this_AC_int.y[7]),
                                    G * np.cos(this_AC_int.y[7]) * np.sin(this_AC_int.y[6]),
                                    G * np.cos(this_AC_int.y[7]) * np.cos(this_AC_int.y[6])])
                    body_accels = body_accels + g_b
                    body_accels[2] = -body_accels[2] # FG expects Z-up

                    # set values and send frames
                    set_FDM(my_fgFDM, this_AC_int.y, 
                            control_norm(U_man), 
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

                if calc_eng_trigger:
                    if jobs_queue.empty():
                        #print(f"[Main Process] Triggering new engine calculation...{VA(current_uvw)*MS2KT:.2f}, {current_alt_m*M2FT:.1f}")
                        new_job = (current_alt_m*M2FT, ISA.Vt2M(VA(current_uvw)*MS2KT, current_alt_m*M2FT), U_man[3], U_man[4], time.perf_counter())
                        try:
                            jobs_queue.put(new_job, block=False)
                            eng_time_adder = 0
                        except mp.queues.Full:
                            #print("[Main Process] Engine is busy, skipping this trigger.")
                            pass
                    else:
                        #print("[Main Process] Engine is still busy with a pending job, skipping this trigger.")
                        print("Ebusy.",end="")
                    calc_eng_trigger = False

                
                frame_count += 1
                # DEBUG ONLY - 
                # print out stuff every so often
                if (frame_count % 100) == 0:
                    #print(f'frame: {frame_count}, time: {this_AC_int.t:0.2f}, theta:{this_AC_int.y[7]:0.6f}, Elev:{this_joy.get_axis(1) * elev_factor}')
                    #print(f'frame: {frame_count}, time: {this_AC_int.t:0.2f}, lat:{current_latlon_rad[0]:0.6f}, lon:{current_latlon_rad[1]:0.6f}')
                    #print(f'time: {this_AC_int.t:0.2f}, N:{current_NED[0]:0.3f}, E:{current_NED[1]:0.3f}, D:{current_NED[2]:0.3f}')
                    print(f'time: {this_AC_int.t:0.1f}s, dt: {this_AC_int.t - last_frame_time:0.2f}s Vcas_2fg:{my_fgFDM.get("vcas"):0.1f}KCAS, elev={U1[1]:0.3f}  ail={U1[0]:0.3f}, U_man={U_man[3]:0.3f},{U_man[4]:0.3f}, U1={U1[3]:0.3f},{U1[4]:0.3f}, E1T={e1_thrust}N')
                    last_frame_time = this_AC_int.t
                if (frame_count % 1000) == 0:
                    print(eng_vals[0])
                    
                

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
            if eng_time_adder >= ENGINE_TRIGGER_S:
                eng_time_adder = 0
                calc_eng_trigger = True

            # parking lot
            # it will keep off the simulation loop above while time does not catch up
            # with the desired "simdt".
            # keeps adding time until that point, then releases the semaphore to run the sim
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


    if OFFLINE == False:
        # close threads
        print("Shutting down network thread...")
        fdm_packet_queue.put(None)  # Send the shutdown signal
        network_thread.join(timeout=1.0) # Wait for the thread to finish
        for s in socks:
            s.close()

        jobs_queue.put(None)
        engine_process.join(timeout=2.0) # Wait for the worker process to finish
        # It's good practice to terminate if it doesn't join cleanly
        if engine_process.is_alive():
            print("[Main Process] Worker did not shut down cleanly. Terminating.")
            engine_process.terminate()
        
    # save data to disk
    save2disk('test_data.csv', x_data=np.array(t_vector_collector), y_data=np.array(data_collector), header=signals_header, skip=0)
    fig1 = make_plots(x_data=np.array(t_vector_collector), y_data=np.array(data_collector), header=signals_header, skip=0)
    plt.show();
