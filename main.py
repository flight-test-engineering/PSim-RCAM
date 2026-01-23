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
    2) add atmospheric wind [DONE - wind]
    3) add other actuator dynamics [DONE]
    4) save/read trim point
    5) fuel detot / inertia update
    6) add engine cut logic with dynamics per RCAM and controls from joystick [DONE]
    7) update wind/turbulence per RCAM
    8) add flaps (delta CL, CM, CD) with controls from joystick [DONE]
    9) add landing ger (delta CM, CD) with controls from joystick [DONE]
    10) add ground effect (delta CL) with radalt/height


'''

# imports
import time
import numpy as np


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import pygame #joystick interface
import socket
import os

import ISA_module as ISA # International Standard Atmosphere library

import sys
sys.path.insert(1, '../')

# threading for FG comms
import threading
import queue

# multiprocessing for engine deck
import multiprocessing as mp


from psim.constants import *
from psim.config import load_aircraft_parameters
import psim.environment as env
import psim.propulsion as prop
import psim.helpers as helpers
from psim.helpers import logger # use the logger without the helper namespace across all modules
import psim.io.joystick as joy
import psim.io.network as net
from psim.io.fgFDM import * # FlightGear comm class
import psim.physics as physics


  


# ############################################################################
# Model Initialization
# ############################################################################

def initialize(VA_t=85.0, gamma_t=0.0, latlon=np.zeros(2), altitude=10000.0, psi_t=0.0, height=0.0, flap_pos=0, gear=0):
    '''
    this initializes the integrators
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
        AC_integrator: scipy aircraft integrator object
        X0: initial states at trim point
        U0: initial commands at trim point
        latlonh_integrator: navigation equation scipy object integrator
    '''
    t0 = 0.0 #intial time for integrators
    alt_m = altitude * FT2M
    rho_trim = env.get_rho(alt_m)

    print()
    print(f'initializing model with {VA_t*MS2KT:.0f} KIAS, {altitude} ft, rho={rho_trim:.4f} kg/m3, flaps={flap_pos}')
    

    latlonh0 = np.array([latlon[0] * DEG2RAD, latlon[1] * DEG2RAD, alt_m])

    if VA_t > 15:
        # we are flying
        # trim model
        res4, res4_status = physics.trim_model(VA_trim=VA_t, gamma_trim=gamma_t, side_speed_trim=0, 
                                    phi_trim=0.0, psi_trim=psi_t*DEG2RAD, rho_trim=rho_trim, h_trim=height,
                                    flap_pos=flap_pos, gear=gear)
        print()
        print('Trimming',res4_status)
        print()
        X0 = res4[:9] # separate states and controls
        U0 = np.concatenate((res4[9:], np.array([flap_pos, gear, 0.0, 0.0]))) # add back ground spoiler and brakes to control vector
        print(f'initial states: {X0}')
        print(f'initial inputs: {U0}')
        print()
    else:
        # we are on the ground
        X0=np.array([VA_t, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, INIT_HDG_DEG * DEG2RAD])
        U0=np.array([0.0, 0.0, 0.0, 0.0, 0.0, flap_pos, gear, 0.0, 0.0])

    # interpolate high lift devices effect
    dcl_dcd_dcm_dalpha = physics.array_interp(flap_pos, HIGH_LIFT_COEFFS, MAX_FLAP)
    # interpolate for landing gear delta CD
    ldg_dcd = physics.array_interp(gear, LDG_DCD, MAX_LDG)
    dcl_dcd_dcm_dalpha[IDX_DCD] += ldg_dcd[0] # add additional drag from landing gear

    # initialize integrators
    AC_integrator = physics.ss_integrator(t0, X0, U0, rho_trim, height, dcl_dcd_dcm_dalpha[IDX_DCL], dcl_dcd_dcm_dalpha[IDX_DCD], dcl_dcd_dcm_dalpha[IDX_DCM], dcl_dcd_dcm_dalpha[IDX_DALPHA])
    
    NED0 = env.NED(X0[:3], X0[6:]) #uvw and phithetapsi
    
    latlonh_integrator = physics.latlonh_int(t0, latlonh0, NED0)
    
    return AC_integrator, X0, U0, latlonh_integrator    



# ############################################################################
# MAIN
# ############################################################################

if __name__ == "__main__":


    logger.info('Starting')

############################################################################
    # SELECT AIRCRAFT CONFIGURATION FILE
    AIRCRAFT_CONFIG_FILE = 'rcam_parameters.json'
    
    # SELECT STARTING POINT: ON GROUND OR IN AIR
    TRIM_ON_GROUND = False

    # INITIAL CONDITIONS (for trim)
    if TRIM_ON_GROUND:
        # ON GROUND  
        INIT_ALT_FT = 586.0 * M2FT #ft
        V_TRIM_MPS = 0 * KT2MS # m/s
        INIT_LATLON_DEG = np.array([.8248243303439*RAD2DEG, 0.1977872426444*RAD2DEG]) #LOWI, RWY 08
        FLAPS_INIT = 0
        INIT_GEAR = 1
    else:
        INIT_ALT_FT = 2400 #ft
        V_TRIM_MPS = 160 * KT2MS # m/s
        INIT_LATLON_DEG = np.array([47.2548, 11.2963]) #in degrees - LOWI short final TFB
        FLAPS_INIT = 0
        INIT_GEAR = 0

    
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
    SIM_VISUAL_OFFSET = 0 # Simulator Z-Axis Visual offset so that landing is on the runway. Difference in Sim and SRTM values for ground elevation
    USE_FG_AS_TERRAIN_DB = True # if False, use SRTM database instead
    DATA_LOGGING_HZ = 10 # frames per second to be logged
    ENG_LOG_PARAMETERS = ['Fn', 'Fg', 'F_ram', 'TSFC', 'Wf', 'N1','N2']

    RESULTS_FILE = 'test_data.csv' # name of file where data will be saved
    LOG2DISK_INTERVAL_S = 30.0 # interval in seconds to save data to disk

    

###########################################################################
    # Load Aircraft Parameters into Global Scope
    #
    # Numba's JIT compiler captures global variables when a function is first
    # compiled. By loading our parameters into the global scope, we make them
    # available to the performance-critical functions without needing to pass 
    # them as arguments on every call.

    # we first need the joystick name, to load the correct parameters...
    # JOYSTICK INIT AND CHECK
    # Explicitly restart the joystick module to clear internal SDL flags
    pygame.init()
    if pygame.joystick.get_init():
        pygame.joystick.quit()
    pygame.joystick.init()    
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        joy_name = None
    else:
        this_joy = pygame.joystick.Joystick(0)
        this_joy.init()
        # --- FLUSH GHOST INPUTS ---
        # Pump the event loop multiple times to clear buffered events 
        # from the previous crash.
        print("Flushing Joystick Buffer...", end="", flush=True)
        for _ in range(15):
            pygame.event.pump()
            time.sleep(0.01) # Small delay to allow OS driver to poll
            print(".",end="")
        print(" done.")
        # --------------------------

        joy_name = this_joy.get_name()


    try:
        # Unpack the dictionary into global variables
        consts = load_aircraft_parameters(AIRCRAFT_CONFIG_FILE, joy_name)
        globals().update(consts)
        joy.initialize_constants(consts) # send constants to joystick function as well
        physics.initialize_constants(consts) # send to physics module as well
    except FileNotFoundError:
        logger.error("ERROR: `rcam_parameters.json` not found. Please provide a valid config file.")
        #print("ERROR: `rcam_parameters.json` not found. Please provide a valid config file.")
        sys.exit(1)
    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"ERROR: Invalid format in {AIRCRAFT_CONFIG_FILE}: {e}")
        sys.exit(1)


    if OFFLINE:
        if joy_name == None:
            print()
            print('Will run OFFLINE simulation, no joystick detected!')
        else:
            print()
            print(f'Will run OFFLINE simulation, joystick model {joy_name} not in JSON config file!')
            logger.warning(f'Will run OFFLINE simulation, joystick model {joy_name} not in JSON config file!')
    else:
        print()
        print(f'found {joystick_count} joysticks connected: {joy_name}, axes={this_joy.get_numaxes()}')



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


#### MULTI THREADING / MULTI PROCESSING ####
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
        
        UDP_IP2 = "192.168.0.26" # set to a remote computer on the same network
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
        tx_thread = threading.Thread(
            target=net.network_worker,
            args=(socks, fdm_packet_queue, fg_addresses),
            daemon=True
        )
        try:
            tx_thread.start()
            print("... started!")
        except Exception as e:
            logging.error(f"...Error in network thread: {e}")
            print(f"...Error in network thread: {e}")
            exit()

        # INCOMING DATA (from FG to Python)
        # ... UDP RX Setup ...
        # --- TERRAIN UDP RECEIVER ---
        TERRAIN_RX_IP = "127.0.0.1" 
        TERRAIN_RX_PORT = 5502 # Port we listen ON

         # Queue used only for shutdown signal
        terrain_shutdown_queue = queue.Queue()
        
        rx_thread = threading.Thread(
            target=net.terrain_udp_worker,
            args=(TERRAIN_RX_IP, TERRAIN_RX_PORT, terrain_shared_data, terrain_shutdown_queue),
            daemon=True
        )
        rx_thread.start()



        # instantiate FG comms object and initialize it
        my_fgFDM = fgFDM()
        my_fgFDM.set('latitude', INIT_LATLON_DEG[0] * DEG2RAD) # in rad
        my_fgFDM.set('longitude', INIT_LATLON_DEG[1] * DEG2RAD) # in rad
        my_fgFDM.set('altitude', INIT_ALT_FT * FT2M) # in m
        my_fgFDM.set('num_engines', 2)
        my_fgFDM.set('num_tanks', 1)
        my_fgFDM.set('num_wheels', 3)
        my_fgFDM.set('cur_time', int(time.perf_counter())) # in seconds



    #######################################################################################
        # Engine Deck Multiprocessing
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
    
    
    ########################################################################################
    # Save to disk thread
    disk_log_queue = queue.Queue()
    disk_log_thread = threading.Thread(
        target=helpers.disk_logging_worker,
        args=(disk_log_queue, RESULTS_FILE, full_header),
        daemon=True
    )
    disk_log_thread.start()

##############################################################################
    # Numba/JIT warm-up
    physics.compile_numba_functions()

###########################################################################
    # SIMULATION VARIABLES INITIALIZATION
    data_collector, t_vector_collector = [], [] # data collectors
    
    prev_uvw = np.array([0,0,0])
    current_uvw = np.array([0,0,0])

    # aircraft initialization (includes trimming)
    this_AC_int, X_trim, U1, this_latlonh_int = initialize(VA_t=V_TRIM_MPS, gamma_t=GAMMA_TRIM_RAD, latlon=INIT_LATLON_DEG, altitude=INIT_ALT_FT, 
                                                            psi_t=INIT_HDG_DEG, height=100.0, flap_pos=FLAPS_INIT)
    # Vector U1 has the controls for the trimmed state
    U1[IDX_GEAR] = INIT_GEAR
    U_trim = U1.copy() # we set U_trim (for trim state, or steady state of the controls) as a copy of the trimmed control states first.


    # Initialize Actual Surface Positions
    # We start with actual = commanded (assuming stable trim)
    U_actual = U1.copy() # U_actual will be the controls after applying the actuator dynamics

    e1_thrust = U1[3]
    e2_thrust = U1[4]


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
    is_first_data_write = True
    
    # time tracking
    sim_time_adder, fg_time_adder = 0.0, 0.0 # counts the time between integration steps to trigger next simulation frame and FG dispatch
    eng_time_adder = 0.0 # loop to calculate engine
    datalog_time_adder = 0.0
    log2disk_time_adder = 0.0
    
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
        sim_U = np.zeros((U_trim.shape[0],t_vector.shape[0]))
        for i in range(sim_U.shape[0]):
            sim_U[i,:] = sim_U[i,:] + U_trim[i]
        
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
            U_actual = physics.update_actuators(sim_U[:,idx], U_actual, simdt, ACT_TAU)

            # set highlift deltas (setting to zero for now)
            dcl_dcd_dcm_dalpha = np.zeros(4)
            
            # integrate 6-DOF
            this_AC_int.set_f_params(U_actual, current_rho, current_AGL_m, dcl_dcd_dcm_dalpha[IDX_DCL], dcl_dcd_dcm_dalpha[IDX_DCD], dcl_dcd_dcm_dalpha[IDX_DCM], dcl_dcd_dcm_dalpha[IDX_DALPHA])
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
        print(f'running inverse deck with alt: {INIT_ALT_FT:.1f} ft, Mach: {ISA.Vc2M(V_TRIM_MPS*MS2KT, INIT_ALT_FT):.3f}, Thrust: {U1[3]:.0f} N')

        U1[IDX_THR1] = prop.E1_deck.interp_altMNFN(INIT_ALT_FT, ISA.Vc2M(V_TRIM_MPS*MS2KT, INIT_ALT_FT), e1_thrust*N2LBF)['PC'] # deck takes lbf
        U1[IDX_THR2] = prop.E2_deck.interp_altMNFN(INIT_ALT_FT, ISA.Vc2M(V_TRIM_MPS*MS2KT, INIT_ALT_FT), e2_thrust*N2LBF)['PC']
        U_trim[IDX_THR1] = U1[IDX_THR1] # percent power
        U_trim[IDX_THR2] = U1[IDX_THR2]

        print(f'this is the inverse deck response: E1:{U1[IDX_THR1]:.4f}; E2:{U1[IDX_THR2]:.4f} % power')
        print()

        # run deck preemptively
        print("Adding engine deck initial job...", end="")
        new_job = (current_alt_m*M2FT, ISA.Vt2M(V_TRIM_MPS*MS2KT, current_alt_m*M2FT), U_trim[IDX_THR1], U_trim[IDX_THR2], TRIM_ON_GROUND, time.perf_counter())
        jobs_queue.put(new_job, block=False)
        # need to give time for deck to run
        time.sleep(.2)
        print(' done.')

        ##### SIMULATION LOOP #####
        while this_AC_int.t <= SIM_TOTAL_TIME_S and exit_signal == 0:
            # get clock
            start = time.perf_counter()

            if run_sim_loop:

                #pygame.event.pump() # More efficient than event.get() if just reading axes
                joy_events = pygame.event.get()

                # -- Inputs & Actuators
                current_throttle = [U_trim[IDX_THR1], U_trim[IDX_THR2]] # keep track of throttle to zero-out the trim bias
                U_trim, U1, exit_signal = joy.get_joy_inputs(this_joy, joy_events, U1, SIM_LOOP_HZ, JOY_TRIM_PARAMS, JOY_FACTORS)

                # U_trim is the manual control inputs (as the joystick is moved)
                # U1 is the trim state, or the zero input values for each control.
                
                # for throtlle, initial trim state is always positive, so we washout if throttles move back
                # if engine trim state is negative, it means engine is OFF
                delta_throttle_1 = U_trim[IDX_THR1] - current_throttle[0] #we look only at #1 engine for simplicity
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
                if (physics.get_air_ground_state(physics.calculate_gear_compression(this_AC_int.y[:9], current_AGL_m)) and (U1[IDX_GNDSP] == 1)):
                    U_trim[IDX_GNDSP] = GND_SPOILERS_DCL # 40% lift dump
                else:
                    U_trim[IDX_GNDSP] = 0.0 # close
                

                U_trim = physics.control_sat(U_trim) # saturate commands


                # Calculate the time step for this specific loop iteration
                # If this is the first step, prev_dt might be 0, so guard against it
                actuator_dt = dt if dt > 0 else simdt 
                
                # Update the physical position of the surfaces with actuator dynamics
                U_actual = physics.update_actuators(U_trim, U_actual, actuator_dt, ACT_TAU)


                # Update thrust values (Engine deck results)
                U_actual[IDX_THR1] = e1_thrust
                U_actual[IDX_THR2] = e2_thrust

                # Interpolate for high lift devices influence
                #hi_lift = high_lift_interp(U_actual[IDX_FLAP])
                dcl_dcd_dcm_dalpha = physics.array_interp(U_actual[IDX_FLAP], HIGH_LIFT_COEFFS, MAX_FLAP)


                # interpolate for landing gear delta CD
                ldg_dcd = physics.array_interp(U_actual[IDX_GEAR], LDG_DCD, MAX_LDG)
                dcl_dcd_dcm_dalpha[IDX_DCD] += ldg_dcd[0] # add additional drag from landing gear


                # -- Engines - multiprocessing
                # if there are new deck values, fetch them,
                # if not, keep what we have
                try:
                    eng_vals = results_queue.get(block=False) # block=False is equivalent to get_nowait()
                    if U1[IDX_THR1] < -0.5:
                        # TODO: make time constants variable???
                        e1_thrust = physics.update_actuators(-eng_vals[0]['F_ram'] * LBF2N, U_actual[IDX_THR1], 0.1, 1.5) # FOR NOW, FIXED TIME CONSTANT AND DT
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
                this_AC_int.set_f_params(U_actual, current_rho, current_AGL_m, dcl_dcd_dcm_dalpha[IDX_DCL], dcl_dcd_dcm_dalpha[IDX_DCD], dcl_dcd_dcm_dalpha[IDX_DCM], dcl_dcd_dcm_dalpha[IDX_DALPHA])
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
                            physics.control_norm(U_actual), 
                            current_latlon_rad, 
                            current_alt_m,
                            body_accels,
                            current_AGL_m)
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
                    on_ground = physics.get_air_ground_state(physics.calculate_gear_compression(this_AC_int.y[:9], current_AGL_m))
                    if jobs_queue.empty():
                        #print(f"[Main Process] Triggering new engine calculation...{VA(current_uvw)*MS2KT:.2f}, {current_alt_m*M2FT:.1f}")
                        new_job = (current_alt_m*M2FT, ISA.Vt2M(env.VA(current_uvw)*MS2KT, current_alt_m*M2FT), U_trim[IDX_THR1], U_trim[IDX_THR2], on_ground, time.perf_counter())
                        try:
                            jobs_queue.put(new_job, block=False)
                            eng_time_adder = 0
                        except mp.queues.Full:
                            #print("[Main Process] Engine Worker is busy, skipping this trigger.")
                            logger.warning("[Main Process] Engine Worker is busy, skipping this trigger.")
                            pass
                    else:
                        #print("[Main Process] Engine Worker is still busy with a pending job, skipping this trigger.")
                        logger.warning("[Main Process] Engine Worker is still busy with a pending job, skipping this trigger.")
                        pass
                    calc_eng_trigger = False

                
                if datalog_trigger:
                    # -- Data Logging
                    internals = physics.RCAM_observe(this_AC_int.y, U_actual, current_rho, current_AGL_m, dcl_dcd_dcm_dalpha[IDX_DCL], dcl_dcd_dcm_dalpha[IDX_DCD], dcl_dcd_dcm_dalpha[IDX_DCM], dcl_dcd_dcm_dalpha[IDX_DALPHA]) # get internal FDM states
                    # engine parameters:
                    eng1_states = np.zeros(len(ENG_LOG_PARAMETERS))
                    eng2_states = np.zeros(len(ENG_LOG_PARAMETERS))
                    if eng_vals:
                        for idx, p in enumerate(ENG_LOG_PARAMETERS):
                            eng1_states[idx] = eng_vals[0][p]
                            eng2_states[idx] = eng_vals[1][p]
                    data_collector.append(np.concatenate((this_AC_int.y, this_latlonh_int.y, current_NED + this_wind, U_trim, internals, eng1_states, eng2_states)))
                    t_vector_collector.append(this_AC_int.t)
                    datalog_trigger = False

                
                # -- Next frame setup
                frame_count += 1

                # print out stuff every so often
                if (frame_count % 100) == 0:
                    #print(f'frame: {frame_count}, time: {this_AC_int.t:0.2f}, theta:{this_AC_int.y[7]:0.6f}, Elev:{this_joy.get_axis(1) * elev_factor}')
                    #print(f'frame: {frame_count}, time: {this_AC_int.t:0.2f}, lat:{current_latlon_rad[0]:0.6f}, lon:{current_latlon_rad[1]:0.6f}')
                    #print(f'time: {this_AC_int.t:0.2f}, N:{current_NED[0]:0.3f}, E:{current_NED[1]:0.3f}, D:{current_NED[2]:0.3f}')
                    #print(f'time: {this_AC_int.t:0.1f}s, dt: {this_AC_int.t - last_frame_time:0.2f}s Vcas_2fg:{my_fgFDM.get("vcas"):0.1f}KCAS, U_trim={U_trim[3]:0.3f},{U_trim[4]:0.3f}, U1={U1[3]:0.3f},{U1[4]:0.3f}, E12T={U_actual[IDX_THR1]:0.0f},{U_actual[IDX_THR2]:0.0f}N, AGL={current_AGL_m*M2FT:0.0f}ft, alt={current_alt_m*M2FT:0.1f}, gnd_sp_arm:{gnd_spoilers_armed}')
                    #print(f'fr#:{frame_count}, time: {this_AC_int.t:0.1f}s, alt={current_alt_m*M2FT:0.0f}, E12T={e1_thrust:0.0f},{e2_thrust:0.0f}N, AGL={current_AGL_m*M2FT:0.0f}, {open_gnd_spoiler=}, {gnd_spoilers_armed=}, {toggle_gnd_spoiler_debounce=}, {U_trim[IDX_GNDSP]=}')
                    #print(f'time: {this_AC_int.t:0.1f}s, alt={current_alt_m*M2FT:0.0f}, U_trim={U_trim[3]:0.3f},{U_trim[4]:0.3f}, U1={U1[3]:0.3f},{U1[4]:0.3f}, E12T={U_actual[IDX_THR1]:0.0f},{U_actual[IDX_THR2]:0.0f}N, Flap_U1={U1[IDX_FLAP]}, U1GNDSP={U1[IDX_GNDSP]:0.4f}, UmanGNDSP={U_trim[IDX_GNDSP]:0.4f}, UactualGNDSP={U_actual[IDX_GNDSP]}')
                    #print(f'time: {this_AC_int.t:0.1f}s, alt={current_alt_m*M2FT:0.0f}, U_trim={U_trim[3]:0.3f},{U_trim[4]:0.3f}, U1={U1[3]:0.3f},{U1[4]:0.3f}, E12T={U_actual[IDX_THR1]:0.0f},{U_actual[IDX_THR2]:0.0f}N, Flap_U1={U1[IDX_FLAP]}, U1GEAR={U1[IDX_GEAR]:0.4f}, UmanGEAR={U_trim[IDX_GEAR]:0.4f}, UactualGEAR={U_actual[IDX_GEAR]}')
                    #print(f'time: {this_AC_int.t:0.1f}s, alt={current_alt_m*M2FT:0.0f}, U_trim={U_trim[3]:0.3f},{U_trim[4]:0.3f}, U1={U1[3]:0.3f},{U1[4]:0.3f}, E12T={U_actual[IDX_THR1]:0.0f},{U_actual[IDX_THR2]:0.0f}N, Flap_U1={U1[IDX_FLAP]}, U1THR1={U1[IDX_THR1]:0.4f}, UmanTHR1={U_trim[IDX_THR1]:0.4f}, UactualTHR1={U_actual[IDX_THR1]}')
                    #print(f'time: {this_AC_int.t:0.1f}s, alt={current_alt_m*M2FT:0.0f}, E12T={U_actual[IDX_THR1]:0.0f}, U1GEAR={U1[IDX_GEAR]:0.4f}, UmanGEAR={U_trim[IDX_GEAR]:0.4f}, UactualGEAR={U_actual[IDX_GEAR]}, U1FLAP={U1[IDX_FLAP]:0.4f}, UmanFLAP={U_trim[IDX_FLAP]:0.4f}, UactualFLAP={U_actual[IDX_FLAP]}, HLDeltas={hi_lift}')
                    #print(f'time: {this_AC_int.t:0.1f}s, E12T={U_actual[IDX_THR1]:0.0f}/{U_actual[IDX_THR2]:0.0f}, UactualFLAP={U_actual[IDX_FLAP]}, HLDeltas={dcl_dcd_dcm_dalpha}, ALPHA={internals[1]}, CL={internals[3]}')
                    #print(f'ldg_pos: {U_actual[IDX_GEAR]}, ldg dcd: {ldg_dcd}, gnd_sp_armed? {U1[IDX_GNDSP]}, gnd_spoilers_dcl: {U_actual[IDX_GNDSP]}')
                    #print(f'U_norm: {control_norm(U_actual)}')
                    print(f'time: {this_AC_int.t:0.1f}s, TLA: {U_trim[IDX_THR1]:.3f}, E12T={U_actual[IDX_THR1]:0.0f}/{U_actual[IDX_THR2]:0.0f} N, FLAP={U_actual[IDX_FLAP]:.1f}, GEAR={U_actual[IDX_GEAR]:.1f}, GndSpoilerArmed={int(U1[IDX_GNDSP])}, ALPHA={internals[1]:.1f}, CL={internals[3]:.2f}, Nz={-body_accels[2]/G:.2f}')
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

            # check/set log2disk trigger
            if log2disk_time_adder >= LOG2DISK_INTERVAL_S:
                #helpers.log_chunk_to_disk(RESULTS_FILE,
                #np.array(t_vector_collector),
                #np.array(data_collector),
                #full_header,
                #is_new_file=is_first_data_write
                #)
                disk_log_queue.put((np.array(t_vector_collector), np.array(data_collector)))
                t_vector_collector = []
                data_collector = []
                log2disk_time_adder = 0.0
                is_first_data_write = False


            # parking lot
            # it will keep off the simulation loop while time does not catch up with the desired "simdt".
            # continuously adds time until that point, then releases the semaphore to run the sim
            if sim_time_adder >= simdt:
                dt = sim_time_adder
                # clamp dt if the OS hangs
                if dt > MAX_INTEG_TIMESTEP_S: dt = MAX_INTEG_TIMESTEP_S
                sim_time_adder = 0
                run_sim_loop = True

            # end-of-frame 
            end = time.perf_counter()
            this_frame_dt = end - start
            fg_time_adder += this_frame_dt
            sim_time_adder += this_frame_dt
            eng_time_adder += this_frame_dt
            datalog_time_adder += this_frame_dt
            log2disk_time_adder += this_frame_dt


    if OFFLINE == False:
        # close threads
        # -- Stop TX threads
        print()
        print("Shutting down network threads...")
        logger.info("Shutting down network threads...")
        fdm_packet_queue.put(None)  # Send the shutdown signal
        tx_thread.join(timeout=1.0) # Wait for the thread to finish
        for s in socks:
            s.close()
                
        # -- Stop RX thread
        terrain_shutdown_queue.put(True)
        rx_thread.join(timeout=1.0)

        # close engine process
        jobs_queue.put(None)
        engine_process.join(timeout=2.0) # Wait for the worker process to finish
        # It's good practice to terminate if it doesn't join cleanly
        if engine_process.is_alive():
            print("[Main Process] Worker did not shut down cleanly. Terminating.")
            logger.warning("[Main Process] Worker did not shut down cleanly. Terminating.")
            engine_process.terminate()
        
    # save data to disk
    if len(t_vector_collector) > 0: # flush rest of data still in memory:
        disk_log_queue.put((np.array(t_vector_collector), np.array(data_collector)))
        print("Waiting for Disk Logger to finish...", end="")
        disk_log_queue.put(None) # Sentinel
        disk_log_thread.join()   # Wait for writing to complete
        print("finished")


    try:
        print(f"Reloading {RESULTS_FILE} for plotting...")
        t_full, data_full = helpers.load_from_disk(RESULTS_FILE)
        
        if len(t_full) > 0:
            print("Generating Plots...", end="")
            helpers.make_plots(t_full, data_full, header=full_header, skip=0)
            plt.show();
            print("done")
        else:
            print("No data found to plot.")
            
    except MemoryError:
        print("Log file too large for RAM plotting.")
    except Exception as e:
        print(f"Plotting error: {e}")
