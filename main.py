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
The FG interface uses code based on the class implemented by Andrew Tridgel (fgFDM):
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
    10) add ground effect (delta CL) with radalt/height [DONE]


'''

# imports
import time
import numpy as np
from numba.experimental import jitclass # dataclass for FDM



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

# module imports
from psim.constants import *                        # basic global constants
from psim.config import load_aircraft_parameters    # reads json and loads parameters
import psim.environment as env                      # ISA atmosphere and geodsy
import psim.propulsion as prop                      # engine deck
import psim.helpers as helpers                      # plotting and helpers
from psim.helpers import logger                     # use the logger without "helper" namespace
import psim.io.joystick as joy                      # joystcik stuff
import psim.io.network as net                       # FG network comms
from psim.io.fgFDM import *                         # FlightGear comm class
import psim.physics as physics                      # actual FDM math is here


  

# ############################################################################
# Model Initialization
# ############################################################################

def initialize(VA_t=85.0, gamma_t=0.0, latlon=np.zeros(2), altitude=10000.0, psi_t=0.0, height=0.0, flap_pos=0, gear=0, acp=None):
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


    logger.info(f'[initialize] initializing model with {VA_t*MS2KT:.0f} KIAS, {altitude} ft, rho={rho_trim:.4f} kg/m3, flaps={flap_pos}')
    

    latlonh0 = np.array([latlon[0] * DEG2RAD, latlon[1] * DEG2RAD, alt_m])

    if VA_t > 15:
        # we are flying
        # trim model
        res4, res4_status = physics.trim_model(VA_trim=VA_t, gamma_trim=gamma_t, side_speed_trim=0, 
                                    phi_trim=0.0, psi_trim=psi_t*DEG2RAD, rho_trim=rho_trim, h_trim=height,
                                    flap_pos=flap_pos, gear=gear, acp=acp)
        logger.info(f'[initialize] Trimming {res4_status}')
        X0 = res4[:9] # separate states and controls
        U0 = np.concatenate((res4[9:], np.array([flap_pos, gear, 0.0, 0.0]))) # add back ground spoiler and brakes to control vector
        logger.debug(f'initial states: {X0}')
        logger.debug(f'initial inputs: {U0}')
    else:
        # we are on the ground
        X0=np.array([VA_t, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, INIT_HDG_DEG * DEG2RAD])
        U0=np.array([0.0, 0.0, 0.0, 0.0, 0.0, flap_pos, gear, 0.0, 0.0])

    # interpolate high lift devices effect
    dcl_dcd_dcm_dalpha = physics.array_interp(flap_pos, acp.HIGH_LIFT_COEFFS, acp.MAX_FLAP)

    # interpolate for landing gear delta CD
    ldg_dcd = physics.array_interp(gear, acp.LDG_DCD, acp.MAX_LDG)
    dcl_dcd_dcm_dalpha[IDX_DCD] += ldg_dcd[0] # add additional drag from landing gear

    # initialize integrators
    AC_integrator = physics.ss_integrator(t0, X0, U0, rho_trim, height, dcl_dcd_dcm_dalpha[IDX_DCL], 
                                            dcl_dcd_dcm_dalpha[IDX_DCD], dcl_dcd_dcm_dalpha[IDX_DCM], dcl_dcd_dcm_dalpha[IDX_DALPHA], acp)
    
    NED0 = env.NED(X0[:3], X0[6:]) #uvw and phithetapsi
    
    latlonh_integrator = physics.latlonh_int(t0, latlonh0, NED0)
    
    return AC_integrator, X0, U0, latlonh_integrator    

# NUMBA/JIT WARM-UP
def compile_numba_functions(acp:jitclass)->None:
        '''
        Runs Numba functions with dummy data to force JIT compilation 
        before the real-time loop begins.
        '''
        logger.info('[compile_numba_functions] Compiling Numba functions (Warm-up)...')
        
        # Dummy Data
        t = 0.0
        X = np.zeros(9)
        U = np.zeros(9)
        NED = np.zeros(3)
        rho = 1.225
        h = 0.0
        dcl = 0.0
        dcd = 0.0
        dcm = 0.0
        dalpha = 0.0
        lat = 0.59
        lon = 0.59
        latlonh0 = np.array([lat, lon, h])

        
        # 1. Physics Core
        _ = physics.array_interp(0, acp.HIGH_LIFT_COEFFS, acp.MAX_FLAP)
        _ = physics.control_sat(U, acp)
        _ = physics.update_actuators(U, U, 0.01, np.ones(9))
        _ = physics.calculate_gear_compression(X, h, acp)
        _ = physics.get_air_ground_state(np.ones(3))
        _ = physics.calculate_ground_forces(X, h, 0.0, acp)
        _ = physics.RCAM_model(X, U, rho, h, dcl, dcd, dcm, dalpha, acp)
        _ = physics.RCAM_observe(X, U, rho, h, dcl, dcd, dcm, dalpha, acp)
        _ = physics.RCAM_model_wrapper(t, X, U, rho, h, dcl, dcd, dcm, dalpha, acp)
        _ = physics.NED_wrapper(t, X, NED)
        _ = physics.latlonh_dot_wrapper(t, X, NED, lat, h)
        _ = physics.ss_integrator(t, X, U, rho, h, dcl, dcd, dcm, dalpha, acp)
        _ = physics.latlonh_int(t, latlonh0, NED)
        _ = physics.calc_ground_effect(h, acp)


        # 2. Environment
        _ = env.VA(np.array([10.,0.,0.]))
        _ = env.fpa(np.ones(3))
        _ = env.add_wind(np.ones(3), np.ones(3))
        _ = env.NED(np.array([10.,0.,0.]), np.array([0.,0.,0.]))
        _ = env.latlonh_dot(np.array([10.,0.,0.]), 0.0, 0.0)
        _ = env.WGS84_MN(lat)
        _ = env.get_rho(h)
        
        logger.info('[compile_numba_functions] ...done compiling.')


# ############################################################################
# MAIN
# ############################################################################

if __name__ == "__main__":



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
    # Lat/Lon - alternate locations
    #INIT_LATLON_DEG = np.array([37.6213, -122.3790]) #in degrees - the func initialize transforms to radians internally
    #INIT_LATLON_DEG = np.array([-21.7632, -48.4051]) #in degrees - SBGP
    #INIT_LATLON_DEG = np.array([47.2548, 11.2963]) #in degrees - LOWI short final TFB
    # wind
    WIND_NED_MPS = np.array([0, 0, 0]) # average wind (m/s), NED
    WIND_STDDEV_MPS = np.array([1, 1, 0]) # wind standard deviation, NED

###########################################################################
    # SIMULATION OPTIONS
    SIM_TOTAL_TIME_S = 10 * 60  # (s) total simulation time
    SIM_LOOP_HZ = 400           # (Hz) simulation loop frame rate throttling
    FG_OUTPUT_LOOP_HZ = 60      # (Hz) frames per second to be sent out to FlightGear AND for recording data
    DECK_LOOP_HZ = 10           # (Hz) fra1me rate to calculate engine deck
    SIM_VISUAL_OFFSET = 0       # Simulator Z-Axis Visual offset so that landing is on the runway. Difference in Sim and SRTM values for ground elevation
    USE_FG_AS_TERRAIN_DB = True # if False, use SRTM database instead
    DATA_LOGGING_HZ = 10        # frames per second to be logged
    ENG_LOG_PARAMETERS = ['Fn', 'Fg', 'F_ram', 'TSFC', 'Wf', 'N1','N2']

    RESULTS_FILE = 'test_data.csv'  # name of file where data will be saved
    LOG2DISK_INTERVAL_S = 30.0      # interval in seconds to save data to disk

    

###########################################################################
    # Load Aircraft Parameters into Global Scope
    #
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
        logger.info("[main] Flushing Joystick Buffer...")
        for _ in range(15):
            pygame.event.pump()
            time.sleep(0.01) # Small delay to allow OS driver to poll
        # --------------------------

        joy_name = this_joy.get_name()

    try:
        # Unpack the dictionary into global variables
        
        consts, acp = load_aircraft_parameters(AIRCRAFT_CONFIG_FILE, joy_name)

        globals().update(consts)
        joy.initialize_constants(consts) # send constants to joystick function as well
        physics.initialize_constants(consts) # send to physics module as well
    except FileNotFoundError:
        logger.error("[main] ERROR: `rcam_parameters.json` not found. Please provide a valid config file.")
        sys.exit(1)
    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"[main] ERROR: Invalid format in {AIRCRAFT_CONFIG_FILE}: {e}")
        sys.exit(1)


    if OFFLINE:
        if joy_name == None:
            logger.info('[main] Will run OFFLINE simulation, no joystick detected!')
        else:
            logger.warning(f'[main] Will run OFFLINE simulation, joystick model {joy_name} not in JSON config file!')
    else:
        logger.info(f'[main] found {joystick_count} joysticks connected: {joy_name}, axes={this_joy.get_numaxes()}')



###########################################################################
    # TERRAIN SHARED DATA
    terrain_shared_data = {'ground_alt': 0.0} # this variable receives terrain height from the FG thread

##########################################################################

    # header for saving data to disk
    signals_header = ['u', 'v', 'w', 'p', 'q', 'r', 'phi', 'theta', 'psi', 'lat', 'lon', 'h', 'V_N', 'V_E', 'V_D', 'dA_actual', 
                        'dE_actual', 'dR_actual', 'dT1_actual', 'dT2_actual','flap_pos_actual', 'gear_pos_actual', 'dgsp_actual', 'brake_actual',
                        'dA_cmd', 'dE_cmd', 'dR_cmd', 'dT1_cmd', 'dT2_cmd', 'flap_pos_cmd', 'gear_pos', 'dgsp', 'brake']
    
    internals_header = ['Va', 'alpha_deg', 'beta_deg', 'CL', 'CD', 'CY', 'Gnd_Fx', 'Gnd_Fy', 'Gnd_Fz'] # internal FDM states
    engine_header = []
    for eng_prefix in ['E1', 'E2']:
        for param in ENG_LOG_PARAMETERS:
            engine_header.append(eng_prefix+param)

    full_header = signals_header + internals_header + engine_header

############################################
#### MULTI THREADING / MULTI PROCESSING ####
############################################

###########################################################################
    # FlightGear Threads and Engine Deck Process Initialization
    # we only start the network and multiprocessing if doing online sim, at least for now
    if OFFLINE == False:
    ############################################################################
        # FLIGHTGEAR SOCKS
        # OUTGOING data (from Python to FG)
        UDP_IP1 = "127.0.0.1" # set to localhost
        UDP_PORT1 = 5500
        
        UDP_IP2 = "192.168.0.26" # set to a remote computer, for additional visuals
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
            logger.info('[main]...started')
        except Exception as e:
            logging.error(f"[main]...Error in network thread: {e}")
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
    # on first run, Numba needs to compile functions
    # this takes time and creates stutter
    # so we warm it up before we start crunching
    
    compile_numba_functions(acp)

###########################################################################
    # SIMULATION VARIABLES INITIALIZATION
    data_collector, t_vector_collector = [], [] # data collectors, for saving data to disk
    
    prev_uvw = np.array([0,0,0]) # velocity vectors
    current_uvw = np.array([0,0,0])

    # aircraft initialization (includes trimming)
    this_AC_int, X_trim, trim_point, this_latlonh_int = initialize(VA_t=V_TRIM_MPS, gamma_t=GAMMA_TRIM_RAD, latlon=INIT_LATLON_DEG, altitude=INIT_ALT_FT, 
                                                            psi_t=INIT_HDG_DEG, height=100.0, flap_pos=FLAPS_INIT, acp=acp)
    trim_point[IDX_GEAR] = INIT_GEAR # controls for the trimmed state
    inceptor_cmd = trim_point.copy() # we set inceptor_cmd as a copy of the trimmed control states first.


    # Initialize Actual Surface Positions
    # We start with actual = trimmed state (assuming stable trim)
    U_actual = trim_point.copy() # U_actual will be the controls vector after applying actuator dynamics

    e1_thrust = trim_point[3] # trimming routine returns thrust, not thrust lever position
    e2_thrust = trim_point[4]


    # aircraft position variables
    current_alt_m = INIT_ALT_FT * FT2M # m
    current_latlon_rad = INIT_LATLON_DEG * DEG2RAD
    current_AGL_m = env.get_AGL(INIT_LATLON_DEG, current_alt_m, SIM_VISUAL_OFFSET)

    
    # frame variables
    frame_count = 0

    fgdt = 1.0 / FG_OUTPUT_LOOP_HZ # (s) fg frame OUTPUT period
    simdt = 1 / SIM_LOOP_HZ # (s) desired simulation time step
    deckdt = 1 / DECK_LOOP_HZ # (s) engine deck call interval
    datalogdt = 1 / DATA_LOGGING_HZ # (s) data logging time interval

    
    # semaphores
    send_frame_trigger = False # send frame to FG
    run_sim_loop = False # main simulation semaphore. it will wait for the clock to reach the next "simdt" and run the simulation
    calc_eng_trigger = True
    datalog_trigger = True
    
    # time tracking
    sim_time_adder, fg_time_adder = 0.0, 0.0 # counts the time between integration steps to trigger next simulation frame and FG dispatch
    eng_time_adder = 0.0 # timer to calculate engine deck
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
        sim_U = np.zeros((inceptor_cmd.shape[0],t_vector.shape[0]))
        for i in range(sim_U.shape[0]):
            sim_U[i,:] = sim_U[i,:] + inceptor_cmd[i]
        
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
            U_actual = physics.update_actuators(sim_U[:,idx], U_actual, simdt, acp.ACT_TAU)

            # set highlift deltas (setting to zero for now)
            dcl_dcd_dcm_dalpha = np.zeros(4)
            
            # integrate 6-DOF
            this_AC_int.set_f_params(U_actual, current_rho, current_AGL_m, dcl_dcd_dcm_dalpha[IDX_DCL], dcl_dcd_dcm_dalpha[IDX_DCD], dcl_dcd_dcm_dalpha[IDX_DCM], dcl_dcd_dcm_dalpha[IDX_DALPHA], acp)
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
            # save data to disk
        if len(t_vector_collector) > 0: # flush rest of data still in memory:
            disk_log_queue.put((np.array(t_vector_collector), np.array(data_collector)))
            logger.info('[main] Waiting for Disk Logger to finish...')
            disk_log_queue.put(None) # Sentinel
            disk_log_thread.join()   # Wait for writing to complete
            logger.info('[main] disk logging finished...')
        
    # with joystick attached, run online
    else:
        ##### ONLINE #####

        # adjust engine command
        # what comes out of the trimming function is thrust directly
        # for online sim, we can't use it
        # let's run the reverse deck to get the thrust lever angle:
        logger.info(f'[main] running inverse deck with alt: {INIT_ALT_FT:.1f} ft, Mach: {ISA.Vc2M(V_TRIM_MPS*MS2KT, INIT_ALT_FT):.3f}, Thrust: {trim_point[3]:.0f} N')

        trim_point[IDX_THR1] = prop.E1_deck.interp_altMNFN(INIT_ALT_FT, ISA.Vc2M(V_TRIM_MPS*MS2KT, INIT_ALT_FT), e1_thrust*N2LBF)['PC'] # deck takes lbf
        trim_point[IDX_THR2] = prop.E2_deck.interp_altMNFN(INIT_ALT_FT, ISA.Vc2M(V_TRIM_MPS*MS2KT, INIT_ALT_FT), e2_thrust*N2LBF)['PC']
        inceptor_cmd[IDX_THR1] = trim_point[IDX_THR1] # percent power
        inceptor_cmd[IDX_THR2] = trim_point[IDX_THR2]

        logger.info(f'[main] this is the inverse deck response: E1:{trim_point[IDX_THR1]:.4f}; E2:{trim_point[IDX_THR2]:.4f} % power')

        # run deck preemptively, so we have data when we need it
        logger.info('[main] Adding engine deck initial job...')
        new_job = (current_alt_m*M2FT, ISA.Vt2M(V_TRIM_MPS*MS2KT, current_alt_m*M2FT), inceptor_cmd[IDX_THR1], inceptor_cmd[IDX_THR2], TRIM_ON_GROUND, time.perf_counter())
        jobs_queue.put(new_job, block=False)


        # joystick needs to read zero (or close to zero)
        # if not, it means pygame is reading the zero wrong
        # to reset, only if joystick is moved
        print('Move joystick to start sim...', end='')
        joy_not_moved = True
        while joy_not_moved:
            dummy = this_joy.get_axis(JOY_PITCH_AXIS) 
            pygame.event.pump()
            if abs(dummy) < 0.05:
                joy_not_moved = False
                print('movement detecte! OK!') # now we can proceed
            else:
                print(".", end='')
                time.sleep(0.1)


        ##### SIMULATION LOOP #####
        while this_AC_int.t <= SIM_TOTAL_TIME_S and exit_signal == 0:
            # get clock
            start = time.perf_counter()


            if run_sim_loop:

                joy_events = pygame.event.get()

                # -- Inputs & Actuators
                current_throttle = [inceptor_cmd[IDX_THR1], inceptor_cmd[IDX_THR2]] # keep track of throttle to zero-out the trim bias
                inceptor_cmd, trim_point, exit_signal = joy.get_joy_inputs(this_joy, joy_events, trim_point, SIM_LOOP_HZ, JOY_TRIM_PARAMS, JOY_FACTORS, acp)

                # inceptor_cmd is the manual control inputs (as the joystick is moved)
                # trim_point is the trim state, or the zero input values for each control.
                
                # for throtlle, initial trim state is always positive, so we washout if throttles move back
                # if engine trim state is negative, it means engine is OFF
                delta_throttle_1 = inceptor_cmd[IDX_THR1] - current_throttle[0] #we look only at #1 engine for simplicity
                if delta_throttle_1 < 0 and trim_point[IDX_THR1] > 0: # if we retard throttle and have positive trim bias
                    if delta_throttle_1 > trim_point[IDX_THR1]: # if we move the throttle a lot, limit washout to zero
                        trim_point[IDX_THR1] = 0
                        trim_point[IDX_THR2] = 0
                    else: #washout from the trim bias, the amount we moved the throttle lever
                        trim_point[IDX_THR1] += delta_throttle_1
                        trim_point[IDX_THR2] += delta_throttle_1
                        if trim_point[IDX_THR1] < 0 : trim_point[IDX_THR1] = 0 # ensure it is never less than zero
                        if trim_point[IDX_THR2] < 0 : trim_point[IDX_THR2] = 0

                # toggle ground spoilers if button is pressed -> this is done in joystick submodule
                # ground spoiler arm/disarm state is passaed through inceptor_cmd[IDX_GNDSP]
                # if spoilers are armed and we are on ground, set ground spoilers to open
                if (physics.get_air_ground_state(physics.calculate_gear_compression(this_AC_int.y[:9], current_AGL_m, acp)) and (trim_point[IDX_GNDSP] == 1)):
                    inceptor_cmd[IDX_GNDSP] = acp.GND_SPOILERS_DCL # 40% lift dump
                else:
                    inceptor_cmd[IDX_GNDSP] = 0.0 # close
                
                # Apply control saturation
                inceptor_cmd = physics.control_sat(inceptor_cmd, acp)

                
                # Update the physical position of the surfaces with actuator dynamics
                U_actual = physics.update_actuators(inceptor_cmd, U_actual, dt, acp.ACT_TAU)

                # Update thrust values (Engine deck results)
                U_actual[IDX_THR1] = e1_thrust
                U_actual[IDX_THR2] = e2_thrust 


                # Interpolate for high lift devices influence
                dcl_dcd_dcm_dalpha = physics.array_interp(U_actual[IDX_FLAP], acp.HIGH_LIFT_COEFFS, acp.MAX_FLAP)

                # interpolate for landing gear delta CD
                ldg_dcd = physics.array_interp(U_actual[IDX_GEAR], acp.LDG_DCD, acp.MAX_LDG)
                dcl_dcd_dcm_dalpha[IDX_DCD] += ldg_dcd[0] # add additional drag from landing gear


                # -- Engines - multiprocessing
                # if there are new deck values, fetch them,
                # if not, keep what we have
                # note, we need to keep this here, after the update thrust values above,
                # otherwise, on engine cut, there is no dynamics
                try:
                    eng_vals = results_queue.get(block=False) # block=False is equivalent to get_nowait()
                    if trim_point[IDX_THR1] < -0.5:
                        # TODO: make time constants variable???
                        e1_thrust = physics.update_actuators(-eng_vals[0]['F_ram'] * LBF2N, U_actual[IDX_THR1], 0.1, 1.5) # FOR NOW, FIXED TIME CONSTANT AND DT
                        U_actual[IDX_THR1] = e1_thrust
                    else:
                        e1_thrust = eng_vals[0]['Fn'] * LBF2N # deck returns lbf, need to convert to N
                    e2_thrust = eng_vals[1]['Fn'] * LBF2N # engine 2 is always what the deck returns, no engine cut yet...
                    U_actual[IDX_THR2] = e2_thrust 
                except mp.queues.Empty:
                    pass             


                # -------------------------------------------------------
                # PREPARE FOR INTEGRATOR
                # -------------------------------------------------------
                prev_uvw = current_uvw
                current_rho = env.get_rho(current_alt_m)


                # -- Integrate Physics
                this_AC_int.set_f_params(U_actual, current_rho, current_AGL_m, dcl_dcd_dcm_dalpha[IDX_DCL], dcl_dcd_dcm_dalpha[IDX_DCD], dcl_dcd_dcm_dalpha[IDX_DCM], dcl_dcd_dcm_dalpha[IDX_DALPHA], acp)
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
                    current_AGL_m = current_alt_m - terrain_shared_data['ground_alt'] # this in meters. subtracting landing gear height
                    
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
                            physics.control_norm(U_actual, acp), 
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
                    on_ground = physics.get_air_ground_state(physics.calculate_gear_compression(this_AC_int.y[:9], current_AGL_m, acp))
                    if jobs_queue.empty():
                        new_job = (current_alt_m*M2FT, ISA.Vt2M(env.VA(current_uvw)*MS2KT, current_alt_m*M2FT), inceptor_cmd[IDX_THR1], inceptor_cmd[IDX_THR2], on_ground, time.perf_counter())
                        try:
                            jobs_queue.put(new_job, block=False)
                            eng_time_adder = 0
                        except mp.queues.Full:
                            logger.warning("[Main Process] Engine Worker is busy, skipping this trigger.")
                            pass
                    else:
                        logger.warning("[Main Process] Engine Worker is still busy with a pending job, skipping this trigger.")
                        pass
                    calc_eng_trigger = False

                
                if datalog_trigger:
                    # -- Data Logging
                    internals = physics.RCAM_observe(this_AC_int.y, U_actual, current_rho, current_AGL_m, dcl_dcd_dcm_dalpha[IDX_DCL], dcl_dcd_dcm_dalpha[IDX_DCD], dcl_dcd_dcm_dalpha[IDX_DCM], dcl_dcd_dcm_dalpha[IDX_DALPHA], acp) # get internal FDM states
                    # engine parameters:
                    eng1_states = np.zeros(len(ENG_LOG_PARAMETERS))
                    eng2_states = np.zeros(len(ENG_LOG_PARAMETERS))
                    if eng_vals:
                        for idx, p in enumerate(ENG_LOG_PARAMETERS):
                            eng1_states[idx] = eng_vals[0][p]
                            eng2_states[idx] = eng_vals[1][p]
                    data_collector.append(np.concatenate((this_AC_int.y, this_latlonh_int.y, current_NED, U_actual, inceptor_cmd, internals, eng1_states, eng2_states)))
                    t_vector_collector.append(this_AC_int.t)
                    datalog_trigger = False

                
                # -- Next frame setup
                frame_count += 1

                # print out stuff every so often
                if (frame_count % 100) == 0:
                    print(f'time: {this_AC_int.t:0.1f}s, TLA: {inceptor_cmd[IDX_THR1]:.3f}, E12T={U_actual[IDX_THR1]:0.0f}/{U_actual[IDX_THR2]:0.0f} N, FLAP={U_actual[IDX_FLAP]:.1f}, GEAR={U_actual[IDX_GEAR]:.1f}, GndSpArmed={int(trim_point[IDX_GNDSP])},Elev={trim_point[IDX_ELE]:.3f}, ALPHA={internals[1]:.1f}, CL={internals[3]:.2f}, Nz={-body_accels[2]/G:.2f}, RADALT={current_AGL_m*M2FT:.0f}ft')
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
                disk_log_queue.put((np.array(t_vector_collector), np.array(data_collector)))
                # zero out variables
                t_vector_collector = []
                data_collector = []
                log2disk_time_adder = 0.0



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
        logger.info("[main] Shutting down network threads...")
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
            logger.warning("[main] Worker did not shut down cleanly. Terminating.")
            engine_process.terminate()
        
    # save data to disk
    if len(t_vector_collector) > 0: # flush rest of data still in memory:
        disk_log_queue.put((np.array(t_vector_collector), np.array(data_collector)))
        logger.info('[main] Waiting for Disk Logger to finish...')
        disk_log_queue.put(None) # Sentinel
        disk_log_thread.join()   # Wait for writing to complete


    try:
        logger.info(f"Reloading {RESULTS_FILE} for plotting...")
        t_full, data_full = helpers.load_from_disk(RESULTS_FILE)
        
        if len(t_full) > 0:
            print("Generating Plots...", end="")
            helpers.make_plots(t_full, data_full, header=full_header, skip=0)
            plt.show();
            print("done")
        else:
            logger.error('[main] No data found to plot.')
            
    except MemoryError:
        logger.error('[main] Log file too large for RAM plotting.')
    except Exception as e:
        logger.error(f"[main] Plotting error: {e}")
