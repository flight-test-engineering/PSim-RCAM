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
--> for best results, start the throttles at idle

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

def initialize(VA_t, gamma_t, latlon, altitude, psi_t, height, flap_pos, gear_pos, acp, fdm_state):
    '''
    this initializes the integrators
    inputs:
        VA_t: true airspeed at trim (m/s)
        gamma_t: flight path angle at trim (rad)
        latlon: initial lat and long (deg)
        altitude: trim altitude (ft)
        psi_t: initial heading (deg)
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
                                    flap_pos=flap_pos, gear=gear_pos, acp=acp)
        logger.info(f'[initialize] Trimming {res4_status}')
        X0 = res4[:9] # separate states from controls
        U0 = np.concatenate((res4[9:], np.array([flap_pos, gear_pos, 0.0, 0.0]))) # add back gear, flaps, ground spoiler and brakes to control vector
        logger.debug(f'initial states: {X0}')
        logger.debug(f'initial inputs: {U0}')
    else:
        # we are on the ground
        X0=np.array([VA_t, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, INIT_HDG_DEG * DEG2RAD])
        U0=np.array([0.0, 0.0, 0.0, 0.0, 0.0, flap_pos, gear_pos, 0.0, 0.0])

    # interpolate high lift devices effects
    aero_delta_coeffs = physics.array_interp(flap_pos, acp.HIGH_LIFT_COEFFS, acp.MAX_FLAP)

    # interpolate for landing gear delta CD and delta CM
    ldg_dcd_dcm = physics.array_interp(gear_pos, acp.LDG_DCD_DCM, acp.MAX_LDG)
    aero_delta_coeffs[IDX_DCD] += ldg_dcd_dcm[0] # add additional CD from landing gear
    aero_delta_coeffs[IDX_DCM] += ldg_dcd_dcm[1] # add additional CM from landing gear

    # initialize integrators
    fdm_state.X = X0
    fdm_state.U = U0
    fdm_state.rho = rho_trim
    fdm_state.h = height
    fdm_state.dcl = aero_delta_coeffs[IDX_DCL]
    fdm_state.dcd = aero_delta_coeffs[IDX_DCD]
    fdm_state.dcm = aero_delta_coeffs[IDX_DCM]
    fdm_state.dalpha = aero_delta_coeffs[IDX_DALPHA]
    fdm_state.dN = aero_delta_coeffs[IDX_DN]

    # 2. Call the updated integrator with just 4 arguments
    AC_integrator = physics.ss_integrator(t0, X0, fdm_state, acp)
    
    NED0 = env.NED(X0[:3], X0[6:]) #uvw and phithetapsi
    
    latlonh_integrator = physics.latlonh_int(t0, latlonh0, NED0)
    
    return AC_integrator, X0, U0, latlonh_integrator    

# Numba/JIT warm-up
def compile_numba_functions(acp:jitclass)->None:
        '''
        Runs Numba functions with dummy data to force JIT compilation 
        before the real-time loop begins.
        '''
        logger.info('[compile_numba_functions] Compiling Numba functions (Warm-up)...')
        
        # Dummy Data
        t = 0.0
        dummy_state = physics.FDMState()
        #X = np.zeros(9)
        #U = np.zeros(9)
        NED = np.zeros(3)
        #rho = 1.225
        h = 0.0
        #dcl = 0.0
        #dcd = 0.0
        #dcm = 0.0
        #dalpha = 0.0
        #dN = 0.0
        lat = 0.59
        lon = 0.59
        latlonh0 = np.array([lat, lon, h])
        gear_compression = np.zeros(3)

        
        # 1. Physics Core
        _ = physics.array_interp(0, acp.HIGH_LIFT_COEFFS, acp.MAX_FLAP)
        _ = physics.control_sat(dummy_state.U, acp)
        _ = physics.update_actuators(dummy_state.U, dummy_state.U, 0.01, np.ones(9))
        _ = physics.calculate_gear_compression(dummy_state.X, h, acp)
        _ = physics.get_air_ground_state(np.ones(3))
        _ = physics.calculate_ground_forces(dummy_state.X, gear_compression, 0.0, acp)
        _ = physics.RCAM_model(dummy_state, acp) 
        _ = physics.RCAM_model_wrapper(t, dummy_state.X, dummy_state, acp)
        _ = physics.NED_wrapper(t, dummy_state.X, NED)
        _ = physics.latlonh_dot_wrapper(t, dummy_state.X, NED, lat, h)
        _ = physics.ss_integrator(t, dummy_state.X, dummy_state, acp)
        _ = physics.latlonh_int(t, latlonh0, NED)
        _ = physics.calc_ground_effect(h, acp)


        # 2. Environment
        _ = env.VA(np.array([10.,0.,0.]))
        _ = env.fpa(np.ones(3))
        _ = env.add_wind(np.ones(3), np.ones(3))
        _ = env.NED(np.array([10.,0.,0.]), np.array([0.,0.,0.]))
        _ = env.latlonh_dot(np.array([ 10.,0.,0.]), 0.0, 0.0)
        _ = env.WGS84_MN(lat)
        _ = env.get_rho(h)
        
        logger.info('[compile_numba_functions] ...done compiling.')


# ############################################################################
# MAIN
# ############################################################################

if __name__ == "__main__":



############################################################################
#                                                                          #
#              SIMULATION CONFIGURATION IN THIS SECTION                    #
#                                                                          #
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
        INIT_LATLON_DEG = np.array([47.258953, 11.332374]) #LOWI, RWY 08
        FLAPS_INIT = 0
        INIT_GEAR = 1
        GAMMA_TRIM_RAD = 0.0 * DEG2RAD # RAD
        INIT_HDG_DEG = 82.0 # DEG
    else:
        # FLYING
        '''
        INIT_ALT_FT = 2400 #ft
        V_TRIM_MPS = 160 * KT2MS # m/s
        INIT_LATLON_DEG = np.array([47.2548, 11.2963]) #in degrees - LOWI short final TFB
        FLAPS_INIT = 0
        INIT_GEAR = 0
        GAMMA_TRIM_RAD = 0.0 * DEG2RAD # RAD
        INIT_HDG_DEG = 82.0 # DEG
        '''
        INIT_ALT_FT = 3000 #ft
        V_TRIM_MPS = 160 * KT2MS # m/s
        INIT_LATLON_DEG = np.array([47.27, 11.43]) #in degrees - LOWI short final TFB
        FLAPS_INIT = 0
        INIT_GEAR = 0
        GAMMA_TRIM_RAD = -3.0 * DEG2RAD # RAD
        INIT_HDG_DEG = 260.0 # DEG
    
    
    


    # wind
    WIND_NED_MPS = np.array([0, 0, 0]) # average wind (m/s), NED
    WIND_STDDEV_MPS = np.array([1, 1, 0]) # wind standard deviation, NED


    # SIMULATION TIMING OPTIONS
    SIM_TOTAL_TIME_S = 10 * 60  # (s) total simulation time
    SIM_LOOP_HZ = 400           # (Hz) simulation loop frame rate throttling
    FG_OUTPUT_LOOP_HZ = 60      # (Hz) frames per second to be sent out to FlightGear AND for recording data
    DECK_LOOP_HZ = 10           # (Hz) fra1me rate to calculate engine deck
    SIM_VISUAL_OFFSET = 0       # Simulator Z-Axis Visual offset so that landing is on the runway. Difference in Sim and SRTM values for ground elevation
    USE_FG_AS_TERRAIN_DB = True # if False, use SRTM database instead
    DATA_LOGGING_HZ = 30        # frames per second to be logged
    ENG_LOG_PARAMETERS = ['Fn', 'Fg', 'F_ram', 'TSFC', 'Wf', 'N1','N2']

    RESULTS_FILE = 'test_data.csv'              # name of file where data will be saved
    LOG2DISK_INTERVAL_S = 30.0                  # interval in seconds to save data to disk
    RESULTS_PLOT_FILE = 'test_data_plot.png'    # name of file of quick plot

############################################################################
# END OF SIMULATION CONFIGURATION SECTION                                  #
#                                                                          #
############################################################################



    # Load Aircraft Parameters into Global Scope
    #
    # we first need the joystick name, to load the correct parameters...

    # Explicitly restart the joystick module to clear internal SDL flags
    pygame.init()
    if pygame.joystick.get_init():
        pygame.joystick.quit()
    pygame.joystick.init()    
    joystick_count = pygame.joystick.get_count()
    if joystick_count == 0:
        joy_name = None
        joy_n_buttons = 0
    else:
        pygame.joystick.init()
        this_joy = pygame.joystick.Joystick(0)
        joy_n_buttons = this_joy.get_numbuttons() # checks how many buttons this joystick has
        # flush ghost inputs
        # Pump the event loop multiple times to clear buffered events 
        logger.info("[main] Flushing Joystick Buffer...")
        for _ in range(15):
            pygame.event.pump()
            time.sleep(0.01) # Small delay to allow OS driver to poll

        joy_name = this_joy.get_name()

    try:
        # Unpack the dictionary into global variables
        
        consts, acp = load_aircraft_parameters(AIRCRAFT_CONFIG_FILE, joy_name, joy_n_buttons)

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
            logger.info(f'[main] Will run OFFLINE simulation, joystick model {joy_name} not in JSON config file!')
    else:
        logger.info(f'[main] found {joystick_count} joysticks connected: {joy_name}, axes={this_joy.get_numaxes()}')



###########################################################################
    # TERRAIN SHARED DATA
    terrain_shared_data = {'ground_alt': 0.0} # this variable receives terrain height from the FG thread

##########################################################################
    # header for saving data to disk

    signals_header = ['u [m/s]', 'v [m/s]', 'w [m/s]', 
                      'p [rad/s]', 'q [rad/s]', 'r [rad/s]', 
                      'phi [rad]', 'theta [rad]', 'psi [rad]', 
                      'lat [deg]', 'lon [deg]', 'Alt [m]', 
                      'V_N [m/s]', 'V_E [m/s]', 'V_D [m/s]',
                      'dA_actual [%]', 'dE_actual [%]', 'dR_actual [%]',
                      'dT1_actual [N]', 'dT2_actual [N]',
                      'flap_pos_actual', 'gear_pos_actual', 'dgsp_actual', 'brake_actual',
                      'dA_cmd [%]', 'dE_cmd [%]', 'dR_cmd [%]', 
                      'dT1_cmd [%]', 'dT2_cmd [%]', 
                      'flap_pos_cmd', 'gear_pos', 'dgsp', 'brake']
    
    internals_header = ['Va [m/s]', 'alpha [rad]', 'beta [rad]',
                        'CL', 'CD', 'CY',
                        'Gnd_Fx [N]', 'Gnd_Fy [N]', 'Gnd_Fz [N]',
                        'Radalt [m]', 'Air_Ground'] # internal FDM states
    engine_header = []
    for eng_prefix in ['E1', 'E2']:
        for param in ENG_LOG_PARAMETERS:
            engine_header.append(eng_prefix+param)

    full_header = signals_header + internals_header + engine_header

    # vectors to log engine data to disk
    eng1_states = np.zeros(len(ENG_LOG_PARAMETERS))
    eng2_states = np.zeros(len(ENG_LOG_PARAMETERS))

############################################
#### MULTI THREADING / MULTI PROCESSING ####
############################################

###########################################################################
    # FlightGear Threads and Engine Deck Process Initialization
    # we only start the network and multiprocessing if doing online sim
    if OFFLINE == False:
    ############################################################################
        # FLIGHTGEAR

        # OUTGOING data (from Python to FG) for Visuals
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

        # THREADING: Create and start the FG network worker thread.
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
            logging.error(f"[main]...Error in FG/outbound network thread: {e}")
            exit()

        # OUTGOING data2 (from Python to Telemetry Viewer)
        UDP_IP3 = "127.0.0.1" # set to localhost, but you can change
        UDP_PORT3 = 5555

        telemetry_thread = net.TelemetrySender(ip=UDP_IP3, port=UDP_PORT3)
        try:
            telemetry_thread.start()
            logger.info('[main]...started')
        except Exception as e:
            logging.error(f"[main]...Error in TM network thread: {e}")
            exit()
        
        

        # INCOMING DATA (from FG to Python) - for height above ground data
        TERRAIN_RX_IP = "127.0.0.1" 
        TERRAIN_RX_PORT = 5502 # Port we listen ON

         # Queue used only for shutdown signal
        terrain_shutdown_queue = queue.Queue()
        
        rx_thread = threading.Thread(
            target=net.terrain_udp_worker,
            args=(TERRAIN_RX_IP, TERRAIN_RX_PORT, terrain_shared_data, terrain_shutdown_queue),
            daemon=True
        )
        try:
            rx_thread.start()
            logger.info('[main]...started')
        except Exception as e:
            logging.error(f"[main]...Error in FG/inbound network thread: {e}")
            exit()
        


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
        # Engine Deck
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
    # instantiate the main dataclass that holds states and controls
    fdm_state = physics.FDMState()

    data_collector, t_vector_collector = [], [] # data collectors, for saving data to disk

    # aircraft initialization (includes trimming)
    this_AC_int, X_trim, trim_point, this_latlonh_int = initialize(
        VA_t=V_TRIM_MPS, gamma_t=GAMMA_TRIM_RAD, latlon=INIT_LATLON_DEG, 
        altitude=INIT_ALT_FT, psi_t=INIT_HDG_DEG, height=100.0, 
        flap_pos=FLAPS_INIT, gear_pos=INIT_GEAR, acp=acp, fdm_state=fdm_state
    )
    inceptor_cmd = trim_point.copy() # we set inceptor_cmd as a copy of the trimmed control states first.


    # Initialize Actual Surface Positions
    # We start with actual = trimmed state (assuming stable trim)
    U_actual = trim_point.copy() # U_actual will be the controls vector after applying actuator dynamics

    e1_thrust = trim_point[IDX_THR1] # trimming routine returns thrust, not thrust lever position
    e2_thrust = trim_point[IDX_THR2]


    # aircraft position variables
    current_alt_m = INIT_ALT_FT * FT2M # m
    current_latlon_rad = INIT_LATLON_DEG * DEG2RAD
    fdm_state.h = env.get_AGL(INIT_LATLON_DEG, current_alt_m, SIM_VISUAL_OFFSET)

    
    # frame variables
    frame_count = 0

    fgdt = 1.0 / FG_OUTPUT_LOOP_HZ # (s) fg frame OUTPUT period
    simdt = 1.0 / SIM_LOOP_HZ # (s) desired simulation time step
    deckdt = 1.0 / DECK_LOOP_HZ # (s) engine deck call interval
    datalogdt = 1.0 / DATA_LOGGING_HZ # (s) data logging time interval

    
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
    
    dt = 0 # actual (measured) integration time step
    prev_dt = dt

    exit_signal = 0 # if joystick button #1 is pressed, end simulation
    
###########################################################################
    # RUN SIMULATION
    # if no joystick detected, run offline
    if OFFLINE:
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
            fdm_state.rho = env.get_rho(current_alt_m)

            # add actuator dynamics to control inputs:
            U_actual = physics.update_actuators(sim_U[:,idx], U_actual, simdt, acp.ACT_TAU)

            # set highlift deltas (setting to zero for now)
            aero_delta_coeffs = np.zeros(5)
            
            # integrate 6-DOF
            fdm_state.U = U_actual
            fdm_state.dcl = aero_delta_coeffs[IDX_DCL]
            fdm_state.dcd = aero_delta_coeffs[IDX_DCD]
            fdm_state.dcm = aero_delta_coeffs[IDX_DCM]
            fdm_state.dalpha = aero_delta_coeffs[IDX_DALPHA]
            fdm_state.dN = aero_delta_coeffs[IDX_DN]

            this_AC_int.integrate(this_AC_int.t + simdt)
            fdm_state.X = this_AC_int.y.copy()

            # integrate navigation equations
            current_NED = env.NED(fdm_state.X[:3], fdm_state.X[6:])
            this_wind = env.add_wind(WIND_NED_MPS, WIND_STDDEV_MPS)
            this_latlonh_int.set_f_params(current_NED + this_wind, current_latlon_rad[0], current_alt_m)
            this_latlonh_int.integrate(this_latlonh_int.t + simdt) #in radians and alt in meters
            
            # store current state and time vector
            current_latlon_rad = this_latlonh_int.y[0:2] # store lat and long (RAD)
            
            if fdm_state.h != 0 : 
                fdm_state.h = this_latlonh_int.y[2] # store altitude (m)
            else:
                this_latlonh_int.y[2] = fdm_state.h

            data_collector.append(np.concatenate((this_AC_int.y, this_latlonh_int.y, current_NED + this_wind, U_actual)))
            current_alt_m = this_latlonh_int.y[2] # store altitude (m)
            
            t_vector_collector.append(this_AC_int.t)

        print(f'Enf of simulation; {len(t_vector_collector)} time steps!')
            # save data to disk
        if len(t_vector_collector) > 0: # if we have data, send it out to disk:
            disk_log_queue.put((np.array(t_vector_collector), np.array(data_collector)))
            logger.info('[main] Waiting for Disk Logger to finish...')
            disk_log_queue.put(None) # Sentinel
            disk_log_thread.join()   # Wait for writing to complete
            logger.info('[main] disk logging finished...')
        else:
            logger.warning('[main] something went wrong and no simultion was calculated... not saving to disk...')
        
    # with joystick attached, run online
    else:
        ##### ONLINE #####

        # adjust engine control (U)
        # what comes out of the trimming function is thrust directly
        # for online sim, we can't use it
        # let's run the reverse deck to get the thrust lever angle:
        logger.info(f'[main] running inverse deck with alt: {INIT_ALT_FT:.1f} ft, Mach: {ISA.Vc2M(V_TRIM_MPS*MS2KT, INIT_ALT_FT):.3f}, Thrust: {trim_point[3]:.0f} N')

        trim_point[IDX_THR1] = prop.E1_deck.interp_altMNFN(INIT_ALT_FT, ISA.Vc2M(V_TRIM_MPS*MS2KT, INIT_ALT_FT), e1_thrust*N2LBF)['PC'] # deck takes lbf
        trim_point[IDX_THR2] = prop.E2_deck.interp_altMNFN(INIT_ALT_FT, ISA.Vc2M(V_TRIM_MPS*MS2KT, INIT_ALT_FT), e2_thrust*N2LBF)['PC']
        # store TLA as our inceptor_cmd
        inceptor_cmd[IDX_THR1] = trim_point[IDX_THR1] # percent power
        inceptor_cmd[IDX_THR2] = trim_point[IDX_THR2]

        logger.info(f'[main] this is the inverse deck response: E1:{trim_point[IDX_THR1]:.4f}; E2:{trim_point[IDX_THR2]:.4f} % power')

        # run deck preemptively, so we have data when we need it shortly...
        logger.info('[main] Adding engine deck initial job...')
        for i in range(10):
            new_job = (current_alt_m*M2FT, ISA.Vt2M(V_TRIM_MPS*MS2KT, current_alt_m*M2FT), inceptor_cmd[IDX_THR1], inceptor_cmd[IDX_THR2], TRIM_ON_GROUND, time.perf_counter())
            jobs_queue.put(new_job, block=False)
            time.sleep(1)


        # pygame sometimes does not initialize the joystick correctly and
        # until it is moved, returns non-zero position,
        # even with stick in neutral.
        # here we ask for user help by moving the stick...
        logger.info('[main] waiting for joystick to be moved or return a valid value...')
        print('Move joystick to start sim -> ', end='')
        joy_not_moved = True
        joy_check_axes = [JOY_ROLL_AXIS, JOY_PITCH_AXIS, JOY_YAW_AXIS] # main joystick axes we want to check

        while joy_not_moved:
            pygame.event.pump()
            joy_mvmt_amount = 0
            for axis in joy_check_axes:
                joy_mvmt_amount += abs(this_joy.get_axis(axis)) # sum up all axes movements
            
            if abs(joy_mvmt_amount) < (0.1 * len(joy_check_axes)):
                # if we get some movement...
                # then check if throttle position is OK:
                joy_events = pygame.event.get()
                dummy_inceptor_cmd, dummy_trim_point, _ = joy.get_joy_inputs(this_joy, joy_n_buttons, joy_events, trim_point, SIM_LOOP_HZ, JOY_TRIM_PARAMS, JOY_FACTORS, acp)
                if (trim_point[IDX_THR1] - (dummy_inceptor_cmd[IDX_THR1] - dummy_trim_point[IDX_THR1])) < 0:
                    print(f'move throttle back: {(trim_point[IDX_THR1] - (dummy_inceptor_cmd[IDX_THR1] - dummy_trim_point[IDX_THR1])):.4f}')
                    time.sleep(0.5)
                else: 
                    trim_point[IDX_THR1] = (trim_point[IDX_THR1] - (dummy_inceptor_cmd[IDX_THR1] - dummy_trim_point[IDX_THR1])) # set bias according to throttle position
                    # note that we need to assume some bias will be left in
                    trim_point[IDX_THR2] = trim_point[IDX_THR1] # set both engines to same throttle
                    joy_not_moved = False # let program continue
            else:
                # if there is no movement... sleep a little
                print(".", end='')
                time.sleep(0.5)

        print('movement detecte! OK!') # now we can proceed


        ##### SIMULATION LOOP #####
        while this_AC_int.t <= SIM_TOTAL_TIME_S and exit_signal == 0:
            # get clock
            start = time.perf_counter()

            if run_sim_loop:

                # -- Inputs & Actuators
                joy_events = pygame.event.get()
                
                current_throttle = [inceptor_cmd[IDX_THR1], inceptor_cmd[IDX_THR2]] # keep track of throttle to zero-out the trim bias
                inceptor_cmd, trim_point, exit_signal = joy.get_joy_inputs(this_joy, joy_n_buttons, joy_events, trim_point, SIM_LOOP_HZ, JOY_TRIM_PARAMS, JOY_FACTORS, acp)

                # inceptor_cmd is the manual control inputs (as the joystick is moved)
                # trim_point is the trim state for each control.
                
                # for throtlle, initial trim state is always positive, so we washout if throttles move back
                # if engine trim state is negative, it means engine is OFF
                delta_throttle_1 = inceptor_cmd[IDX_THR1] - current_throttle[0] #we look only at #1 engine for simplicity
                if delta_throttle_1 < 0 and trim_point[IDX_THR1] > 0: # if we retard throttle and have positive trim bias
                    trim_point[IDX_THR1] = max(0.0, (trim_point[IDX_THR1] + delta_throttle_1))
                    trim_point[IDX_THR2] = trim_point[IDX_THR1]

                # to toggle ground spoilers armed/disarmed (if button is pressed) -> done in joystick submodule
                # ground spoiler arm/disarm state comes in from trim_point[IDX_GNDSP]
                # the actual inceptor command is inserted here:
                # if spoilers are armed and we are on ground, set ground spoilers to open
                if ((fdm_state.AG) and (trim_point[IDX_GNDSP] == 1)):
                    inceptor_cmd[IDX_GNDSP] = acp.GND_SPOILERS_DCL # % lift dump
                else:
                    inceptor_cmd[IDX_GNDSP] = 0.0 # closed, no lift dump
                
                # Apply control saturation
                inceptor_cmd = physics.control_sat(inceptor_cmd, acp)

                
                # Update the physical position of the surfaces with actuator dynamics
                U_actual = physics.update_actuators(inceptor_cmd, U_actual, dt, acp.ACT_TAU)

                # Update thrust values (Engine deck results)
                U_actual[IDX_THR1] = e1_thrust
                U_actual[IDX_THR2] = e2_thrust 


                # Interpolate for high lift devices influence
                # key: delta_CL, delta_CD, delta_CM, delta_alpha, delta_N
                aero_delta_coeffs = physics.array_interp(U_actual[IDX_FLAP], acp.HIGH_LIFT_COEFFS, acp.MAX_FLAP)

                # interpolate for landing gear delta CD and delta CM
                ldg_dcd_dcm = physics.array_interp(U_actual[IDX_GEAR], acp.LDG_DCD_DCM, acp.MAX_LDG)
                aero_delta_coeffs[IDX_DCD] += ldg_dcd_dcm[0] # add additional CD from landing gear
                aero_delta_coeffs[IDX_DCM] += ldg_dcd_dcm[1] # add additional CM from landing gear


                # -- Engines - multiprocessing
                # if there are new deck values, fetch them,
                # if not, keep what we have
                # note, we need to keep this here, after the update thrust values above,
                # otherwise, on engine cut, there are no engine dynamics (spooldown)
                try:
                    eng_vals = results_queue.get(block=False) # block=False is equivalent to get_nowait()
                    # engine cut logic
                    if trim_point[IDX_THR1] < -0.5: # engine was cut
                        # TODO: make time constants variable???
                        e1_thrust = physics.update_actuators(-eng_vals[0]['F_ram'] * LBF2N, U_actual[IDX_THR1], 0.1, 1.5) # FOR NOW, FIXED TIME CONSTANT AND DT
                        
                    else: # engine is running
                        e1_thrust = eng_vals[0]['Fn'] * LBF2N # deck returns lbf, need to convert to N
                    e2_thrust = eng_vals[1]['Fn'] * LBF2N # engine 2 is always what the deck returns, no engine cut yet...
                    U_actual[IDX_THR1] = e1_thrust
                    U_actual[IDX_THR2] = e2_thrust 
                except mp.queues.Empty:
                    pass             


                # -------------------------------------------------------
                # INTEGRATION
                # -------------------------------------------------------
                fdm_state.rho = env.get_rho(current_alt_m)

                # Update control inputs into the state object
                fdm_state.U = U_actual
                fdm_state.dcl = aero_delta_coeffs[IDX_DCL]
                fdm_state.dcd = aero_delta_coeffs[IDX_DCD]
                fdm_state.dcm = aero_delta_coeffs[IDX_DCM]
                fdm_state.dalpha = aero_delta_coeffs[IDX_DALPHA]
                fdm_state.dN = aero_delta_coeffs[IDX_DN]

                # -- Integrate Physics
                this_AC_int.integrate(this_AC_int.t + dt)
                # CRITICAL: Re-sync state observables to the final step
                fdm_state.X = this_AC_int.y.copy()
                
                # -- Integrate navigation equations
                current_NED = env.NED(fdm_state.X[:3], fdm_state.X[6:])
                this_wind = env.add_wind(WIND_NED_MPS, WIND_STDDEV_MPS)

                this_latlonh_int.set_f_params(current_NED + this_wind, current_latlon_rad[0], current_alt_m)
                this_latlonh_int.integrate(this_latlonh_int.t + dt) #in radians and alt in meters
                
                # store current state and time vector for next iteration
                current_latlon_rad = this_latlonh_int.y[0:2]
                current_alt_m = this_latlonh_int.y[2]

                # height above terrain
                if USE_FG_AS_TERRAIN_DB:
                    # use FlightGear as a terrain database...
                    fdm_state.h = current_alt_m - terrain_shared_data['ground_alt'] # this in meters. subtracting landing gear height
                else:
                    # alternate: use SRTM instead...
                    fdm_state.h = env.get_AGL(current_latlon_rad*RAD2DEG, current_alt_m, SIM_VISUAL_OFFSET) # in meters as well
                

                # SEMAPHORES TRIGGER CHECKS

                # -- FlightGear Output
                if send_frame_trigger:
                    # -- Send data to FlightGear
                    # it is easier to calculate body accelerations instead of reaching into the RCAM function
                    body_accels = fdm_state.body_accels.copy()
                    body_accels[2] = -body_accels[2] # FG expects Z-up

                    # set values and send frames
                    net.set_FDM(my_fgFDM, fdm_state.X, 
                            physics.control_norm(U_actual, acp), 
                            current_latlon_rad, 
                            current_alt_m,
                            body_accels,
                            fdm_state.h)
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
                    on_ground = physics.get_air_ground_state(physics.calculate_gear_compression(fdm_state.X[:9], fdm_state.h, acp))
                    if jobs_queue.empty():
                        new_job = (current_alt_m*M2FT, ISA.Vt2M(fdm_state.Va*MS2KT, current_alt_m*M2FT), inceptor_cmd[IDX_THR1], inceptor_cmd[IDX_THR2], on_ground, time.perf_counter())
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

                
                # -- Data logging and Telemetry (they will run at same frequency)
                if datalog_trigger:
                    _ = physics.RCAM_model(fdm_state, acp) # re-sync internal states before storing
                    internals = np.array([
                            fdm_state.Va, fdm_state.alpha, fdm_state.beta,
                            fdm_state.CL, fdm_state.CD, fdm_state.CY,
                            fdm_state.F_gnd_x, fdm_state.F_gnd_y, fdm_state.F_gnd_z,
                            fdm_state.h - acp.LG_MAIN_R_POS[2],
                            fdm_state.AG
                        ])
                    # engine parameters:
                    if eng_vals:
                        for idx, p in enumerate(ENG_LOG_PARAMETERS):
                            eng1_states[idx] = eng_vals[0][p]
                            eng2_states[idx] = eng_vals[1][p]

                    datalog_data = np.block([fdm_state.X, this_latlonh_int.y, current_NED, U_actual, inceptor_cmd, internals, eng1_states, eng2_states])
                    data_collector.append(datalog_data)
                    t_vector_collector.append(this_AC_int.t)
                    

                    # send data out to telemetry
                    tm_data = [
                        this_AC_int.t,
                        fdm_state.Va * MS2KT,
                        fdm_state.h * M2FT,
                        fdm_state.X[physics.IDX_THETA] * RAD2DEG,
                        fdm_state.alpha * RAD2DEG,
                        fdm_state.X[physics.IDX_THETA] * RAD2DEG - fdm_state.alpha * RAD2DEG, # gamma
                        fdm_state.AG
                    ]
                    telemetry_thread.send_data(tm_data)

                    datalog_trigger = False

                
                # -- Next frame setup
                frame_count += 1
                prev_dt = dt
                dt = 0
                run_sim_loop = False

                # print out stuff every so often
                if (frame_count % 100) == 0:
                    _ = physics.RCAM_model(fdm_state, acp) # re-sync internal states
                    print(f'time: {this_AC_int.t:0.1f}s, TLA: {inceptor_cmd[IDX_THR1]:.3f}, E12T={U_actual[IDX_THR1]:0.0f}/{U_actual[IDX_THR2]:0.0f} N, FLAP={U_actual[IDX_FLAP]:.1f}, GEAR={U_actual[IDX_GEAR]:.1f}, GndSpArmed={int(trim_point[IDX_GNDSP])},Elev={trim_point[IDX_ELE]:.3f}, ALPHA={internals[1]*RAD2DEG:.1f}, CL={internals[3]:.2f}, Nz={fdm_state.load_factor:.2f}, RADALT={(fdm_state.h - acp.LG_MAIN_R_POS[2])*M2FT:.0f}ft, CD={internals[4]:.2f}')
                #################################################################################################################################################################

                # reset integrator timestep counter
                # performance check
                # if you want to check for dropped frames, uncomment below
                #if dt > simdt:
                    #actual_fps = 1.0 / dt
                    #logger.warning(f"[Main Process] Simulation Loop took longer than planned:Target {SIM_LOOP_HZ}Hz | Actual {actual_fps:.1f}Hz | Calc Time {calc_duration*1000:.2f}ms")
            
            
            # TIMERS CHECKS

            # parking lot
            
            # physics loop trigger
            if sim_time_adder >= simdt:
            # it will keep off the simulation loop until time catches up with the desired "simdt",
            # then releases the semaphore to run the sim
                dt = sim_time_adder
                sim_time_adder = 0
                run_sim_loop = True

            # FlightGear send frame trigger
            if fg_time_adder >= fgdt:
                fg_time_adder = 0
                send_frame_trigger = True

            # engine deck trigger
            if eng_time_adder >= deckdt:
                eng_time_adder = 0
                calc_eng_trigger = True

            # data log to memory trigger
            if datalog_time_adder >= datalogdt:
                datalog_time_adder = 0
                datalog_trigger = True

            # data log to disk trigger
            if log2disk_time_adder >= LOG2DISK_INTERVAL_S:
                disk_log_queue.put((np.array(t_vector_collector), np.array(data_collector)))
                # zero out variables
                t_vector_collector = []
                data_collector = []
                log2disk_time_adder = 0.0


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
        telemetry_thread.running = False
        telemetry_thread.join(timeout=1.0)
                
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
        
    # save last data to disk
    if len(t_vector_collector) > 0: # flush rest of data still in memory:
        disk_log_queue.put((np.array(t_vector_collector), np.array(data_collector)))
        logger.info('[main] Waiting for Disk Logger to finish...')
        disk_log_queue.put(None) # Sentinel
        disk_log_thread.join()   # Wait for writing to complete


    # plotting results
    try:
        logger.info(f"Reloading {RESULTS_FILE} for plotting...")
        t_full, data_full = helpers.load_from_disk(RESULTS_FILE)
        
        if len(t_full) > 0:
            print("Generating Plots...", end="")
            full_plot = helpers.make_plots(t_full, data_full, header=full_header, skip=0)
            plt.savefig(RESULTS_PLOT_FILE)
            print("done")
        else:
            logger.error('[main] No data found to plot.')
            
    except MemoryError:
        logger.error('[main] Log file too large for RAM plotting.')
    except Exception as e:
        logger.error(f"[main] Plotting error: {e}")
