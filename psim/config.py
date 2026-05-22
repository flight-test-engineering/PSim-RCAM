import json
import sys
import numpy as np
from psim.constants import DEG2RAD, RAD2DEG
from psim.helpers import logger
from numba import jit, float64, int32
from numba.experimental import jitclass



# ############################################################################
# Aircraft Parameter Loader
# ############################################################################



# define the data and types for Numba jitclass
spec = [
    # Mass & Geometry
    ('M', float64),
    ('CBAR', float64),
    ('S', float64),
    ('ST', float64),
    ('LT', float64),
    ('WINGSPAN', float64),
    
    # CG Positions
    ('XCG', float64),
    ('YCG', float64),
    ('ZCG', float64),
    ('XAC', float64),
    ('YAC', float64),
    ('ZAC', float64),
    
    # Engines
    ('XAPT1', float64), ('YAPT1', float64), ('ZAPT1', float64),
    ('XAPT2', float64), ('YAPT2', float64), ('ZAPT2', float64),
    
    # Aero Coefficients (Scalar)
    ('DEPSDA', float64),
    ('ALPHA_L0', float64),
    ('ALPHA_SWITCH', float64),
    ('N', float64),
    ('A3', float64), ('A2', float64), ('A1', float64), ('A0', float64),
    ('NT', float64),
    ('EPSILON_DOT', float64),
    ('HIGH_LIFT_COEFFS', float64[:,:]),
    ('MAX_FLAP', int32),
    ('LDG_DCD_DCM', float64[:,:]),
    ('MAX_LDG', int32),
    ('GND_SPOILERS_DCL', float64),
    ('CDMIN', float64), ('D1', float64), ('D0', float64),
    ('CY_BETA', float64), ('CY_DR', float64),
    ('C_l_BETA', float64), ('C_m_ZERO', float64), ('C_m_ALPHA', float64), ('C_n_BETA', float64),
    
    # Rate & Control Derivatives (Matrix/Array would be cleaner, but scalars are faster)
    ('C_l_P', float64), ('C_l_Q', float64), ('C_l_R', float64),
    ('C_m_P', float64), ('C_m_Q', float64), ('C_m_R', float64),
    ('C_n_P', float64), ('C_n_Q', float64), ('C_n_R', float64),
    
    ('C_l_DA', float64), ('C_l_DE', float64), ('C_l_DR', float64),
    ('C_m_DA', float64), ('C_m_DE', float64), ('C_m_DR', float64),
    ('C_n_DA', float64), ('C_n_DE', float64), ('C_n_DR', float64),

    # Arrays (Must define as float64[:])
    ('INERTIA_TENSOR_b', float64[:,::1]),
    ('INV_INERTIA_TENSOR_b', float64[:,::1]),
    
    # Gear (Vectors)
    ('LG_NOSE_POS', float64[:]),
    ('LG_MAIN_L_POS', float64[:]),
    ('LG_MAIN_R_POS', float64[:]),
    
    # Gear Dynamics
    ('LG_SPRING_K', float64),
    ('LG_DAMP_COMPRESSION', float64),
    ('LG_DAMP_REBOUND', float64),
    ('LG_ROLLING_FRICTION_MU', float64),
    ('LG_SIDE_FRICTION_MU', float64),
    ('LG_MU_BRAKE', float64),
    ('LG_FRICTION_STIFFNESS', float64),

    # Actuator Limits
    ('U_LIMITS_MIN', float64[:]),
    ('U_LIMITS_MAX', float64[:]),

    # Actuator Dynamics
    ('ACT_TAU', float64[:]),
]



@jitclass(spec)
class AircraftParams:
    def __init__(self):
        # We leave this empty. 
        # Fields are initialized to 0.0 by default in Numba.
        # We will populate them using the factory function.
        pass


# ############################################################################
# Aircraft Parameter Loader
# JIT version
# ############################################################################

def load_aircraft_parameters(filepath: str, joy_name: str|None, joy_n_buttons: int) -> (dict, jitclass):
    """
    Loads aircraft parameters from a JSON file, processes them, and returns
    them as a dictionary of constants ready for the simulation.
    """
    with open(filepath, 'r') as f:
        params = json.load(f)

    logger.info(f"[load_aircraft_parameters] Loading aircraft model: {params['aircraft_name']}")

    consts = {}
    acp = AircraftParams() # create an empty instance of the jitclass

    # .. Nominal vehicle constants ..
    # (TP-088-3, p. 9, para 2.2, table 2.4)
    mg = params['mass_and_geometry']

    # (TP-088-3, p. 9, para 2.2, table 2.5)
    acp.M = float(mg['mass']) # kg - total mass

    # (TP-088-3, p. 9, para 2.2, table 2.4)
    acp.CBAR = mg['wing_mean_aerod_chord'] # m - mean aerodynamic chord
    acp.S = mg['wing_area'] # m^2 - wing area
    acp.ST = mg['tail_area'] # m^2 - tail area
    acp.LT = mg['tail_arm'] # m - tail aerodynamic center distance to CG
    acp.WINGSPAN = mg['wing_area'] / mg['wing_mean_aerod_chord'] # m - calculated wing span

    # Derived Geometry (CG, AC)
    cgap = params['cg_and_ac_positions']
    # .. centre of gravity position ..
    acp.XCG = cgap['xcg'] * mg['wing_mean_aerod_chord'] # m - x pos of CG
    acp.YCG = cgap['ycg'] # m - y pos of CG
    acp.ZCG = cgap['zcg'] * mg['wing_mean_aerod_chord'] # m - z pos of CG
    # .. aerodynamic centre position .. (TP-088-3, p. 9, para 2.2, table 2.4)
    acp.XAC = cgap['xac'] * mg['wing_mean_aerod_chord']
    acp.YAC = cgap['yac']
    acp.ZAC = cgap['zac']

    # .. engines point of thrust application .. (TP-088-3, p. 9, para 2.2, table 2.4)
    ep = params['engine_positions']
    acp.XAPT1, acp.YAPT1, acp.ZAPT1 = ep[0]['x'], ep[0]['y'], ep[0]['z']
    acp.XAPT2, acp.YAPT2, acp.ZAPT2 = ep[1]['x'], ep[1]['y'], ep[1]['z']

    # .. aerodynamic properties 
    # ... wing lift ...
    ac = params['lift_coeffs']
    acp.DEPSDA = ac['depsda'] # rad/rad - change in downwash wrt alpha # (TP-088-3, p. 14, para 2.3.4, eq 2.30)
    acp.ALPHA_L0 = ac['alpha_l0_deg'] * DEG2RAD # rad - zero lift AOA
    acp.ALPHA_SWITCH = ac['alpha_switch_deg'] * DEG2RAD # rad - kink point of lift slope
    # these values are from the 1997 RCAM revision
    acp.N = ac['lift_slope_n'] # adm - slope of linear region of lift slope # (TP-088-3, p. 14, para 2.3.4, eq 2.25)
    acp.A3 = ac['lift_poly_coeffs']['a3'] # adm - coeff of alpha^3
    acp.A2 = ac['lift_poly_coeffs']['a2'] # adm - coeff of alpha^2
    acp.A1 = ac['lift_poly_coeffs']['a1'] # adm - coeff of alpha^1
    acp.A0 = ac['lift_poly_coeffs']['a0'] # adm - coeff of alpha^0
    # ... tail ...
    # (TP-088-3, p. 15, eq 2.27)
    acp.NT = ac['htail_coeffs']['nt']                   # adm - slope of linear region of TAIL lift slope
    # (TP-088-3, p. 15, eq 2.28)
    acp.EPSILON_DOT = ac['htail_coeffs']['epsilon_dot'] # adm multiplier for tail dynamic downwash response wrt pitch rate

    # ... high lift aero delta coefficients ...
    # key: delta_CL, delta_CD, delta_CM, delta_alpha, delta_N (lift curve slope)
    high_lift_dict = ac['high_lift_coeffs']
    acp.HIGH_LIFT_COEFFS = np.array([high_lift_dict[str(i)] for i in range(len(high_lift_dict))])
    
    # transform delta_alpha to radians
    for i in range(acp.HIGH_LIFT_COEFFS.shape[0]):
        acp.HIGH_LIFT_COEFFS[i][3] = acp.HIGH_LIFT_COEFFS[i][3] * DEG2RAD # RCAM model uses alpha in radians
    
    acp.MAX_FLAP = int(acp.HIGH_LIFT_COEFFS.shape[0] - 1) # maximum flap setting (note: integer)
    
    # ... landing gear aerodynamics delta coefficients...
    # key: delta_CD, delta_CM
    ldg_drag_dict = ac['landing_gear_aero']
    acp.LDG_DCD_DCM = np.array([ldg_drag_dict[str(i)] for i in range(len(ldg_drag_dict))])   
    acp.MAX_LDG = int(acp.LDG_DCD_DCM.shape[0] - 1)
    
    # ... ground spoilers lift dump ...
    acp.GND_SPOILERS_DCL = ac['gnd_spoilers_dcl']

    # .. aerodynamic properties - drag ..
    drag_coeffs = params['drag_coeffs']
    # (TP-088-3, p. 14, para 2.3.4, eq 2.31)
    acp.CDMIN = drag_coeffs['cdmin'] # adm - CD min - bottom of CDxALpha curve
    acp.D1 = drag_coeffs['d1'] # adm - coeff of alpha^2
    acp.D0 = drag_coeffs['d0'] # adm - coeff of alpha^0

    # .. aerodynamic properties - side force ..
    # (TP-088-3, p. 14, para 2.3.4, eq 2.32)
    side_force_coeffs = params['side_force_coeffs']
    acp.CY_BETA = side_force_coeffs['cy_beta'] # adm - side force coeff with sideslip
    acp.CY_DR = side_force_coeffs['cy_dr'] # adm - side force coeff with rudder deflection

    # .. aerodynamic properties - moment coefficients ..
    # (TP-088-3, p. 14, para 2.3.4, eq 2.33)
    moment_coeffs = params['moment_coeffs']
    acp.C_l_BETA = moment_coeffs['c_l_beta'] # adm - roll moment due to beta
    acp.C_m_ZERO = moment_coeffs['c_m_zero'] # adm - pitch moment at zero alpha
    acp.C_m_ALPHA = moment_coeffs['c_m_alpha'] # adm - pitch moment due to alpha
    acp.C_n_BETA = moment_coeffs['c_n_beta'] * RAD2DEG # per RCAM doc, need to mult by 180/pi

    # ... roll, pitch, yaw moments with rates ..,
    # (TP-088-3, p. 14, para 2.3.4, eq 2.33)
    moment_rate_coeffs = params["pqr_moment_coeffs"]
    acp.C_l_P = moment_rate_coeffs['c_l_p']
    acp.C_l_Q = moment_rate_coeffs['c_l_q']
    acp.C_l_R = moment_rate_coeffs['c_l_r']
    acp.C_m_P = moment_rate_coeffs['c_m_p']
    acp.C_m_Q = moment_rate_coeffs['c_m_q']
    acp.C_m_R = moment_rate_coeffs['c_m_r']
    acp.C_n_P = moment_rate_coeffs['c_n_p']
    acp.C_n_Q = moment_rate_coeffs['c_n_q']
    acp.C_n_R = moment_rate_coeffs['c_n_r']

    # ... roll, pitch, yaw moments with controls ...
    # (TP-088-3, p. 14, para 2.3.4, eq 2.33)
    moment_controls_coeffs = params["controls_moment_coeffs"]
    acp.C_l_DA = moment_controls_coeffs['c_l_da']
    acp.C_l_DE = moment_controls_coeffs['c_l_de']
    acp.C_l_DR = moment_controls_coeffs['c_l_dr']
    acp.C_m_DA = moment_controls_coeffs['c_m_da']
    acp.C_m_DE = moment_controls_coeffs['c_m_de']
    acp.C_m_DR = moment_controls_coeffs['c_m_dr']
    acp.C_n_DA = moment_controls_coeffs['c_n_da']
    acp.C_n_DE = moment_controls_coeffs['c_n_de']
    acp.C_n_DR = moment_controls_coeffs['c_n_dr']

    # .. inertia tensor ..
    mass = float(mg['mass'])
    tensor_per_unit_mass = np.array(params['inertia']['tensor_per_unit_mass'])
    acp.INERTIA_TENSOR_b = np.ascontiguousarray(mass * tensor_per_unit_mass) # each element in m2 - (TP-088-3, p. 12, para 2.3.1.2, eq 2.11)
    acp.INV_INERTIA_TENSOR_b = np.ascontiguousarray(np.linalg.inv(acp.INERTIA_TENSOR_b))

    # .. Control Surface and Throttle Limits ..
    # (TP-088-3, p. 19, para 2.5)
    cl_deg = params['control_limits_deg']
    U_LIMITS_RAD = {k: (v[0] * DEG2RAD, v[1] * DEG2RAD) for k, v in cl_deg.items()} # transform from degrees to radians
    acp.U_LIMITS_MIN = np.array([lim[0] for lim in U_LIMITS_RAD.values()]) # separate mins
    acp.U_LIMITS_MAX = np.array([lim[1] for lim in U_LIMITS_RAD.values()]) # from maximums for efficiency

    # Landing Gear
    # .. Geometry
    ldg_geom = params['landing_gear_geometry']
    acp.LG_NOSE_POS = np.array(ldg_geom['nose_pos']) # Nose Gear: ~10-12m forward of CG, centerline, ~2.5m below CG
    acp.LG_MAIN_L_POS = np.array(ldg_geom['left_mains_pos']) # Main Gear Left: ~4-5m behind CG, ~4m left, ~3.5m below CG
    acp.LG_MAIN_R_POS = np.array(ldg_geom['right_mains_pos'])

    # .. Dynamics
    ldg_dynamics = params['landing_gear_dynamics']
    acp.LG_SPRING_K = ldg_dynamics['lg_spring_k']
    acp.LG_DAMP_COMPRESSION = ldg_dynamics['lg_damp_compression']
    acp.LG_DAMP_REBOUND = ldg_dynamics['lg_damp_rebound']
    acp.LG_FRICTION_STIFFNESS = ldg_dynamics['lg_friction_stiffness'] # "Virtual spring" to hold plane still when stopped
    acp.LG_SIDE_FRICTION_MU = ldg_dynamics['lg_side_friction_mu']
    acp.LG_ROLLING_FRICTION_MU = ldg_dynamics['lg_rolling_friction_mu'] * 10 # DEBUG
    acp.LG_MU_BRAKE = ldg_dynamics['lg_mu_brake']

    # Actuator Dynamics
    act_dyn = params['actuator_dynamics']
    acp.ACT_TAU = np.array([act_dyn["tau_aileron"], 
                                  act_dyn["tau_elevator"], 
                                  act_dyn["tau_rudder"],
                                  act_dyn["tau_engine"],
                                  act_dyn["tau_engine"],
                                  act_dyn["tau_flaps"],
                                  act_dyn["tau_gear"], 
                                  act_dyn["tau_gnd_spoiler"],
                                  act_dyn["tau_brakes"]])

    # Joystick Mappings
    # the function order is has the sequence of functions
    # that will be checked for button presses
    consts['JOY_BUTTONS_MAP'] = {'exit_signal':-1,
                      'brake':-1,
                      'pitch_dn':-1,
                      'pitch_up':-1,
                      'E1_cycle_cut':-1,
                      'ldg_cycle':-1,
                      'flaps_command_dn':-1,
                      'flaps_command_up':-1,
                      'arm_disarm_gnd_spoiler':-1,
                      'T1_fd':-1,
                      'T1_af':-1,
                      'zero_ail_rud_thr':-1}
    # available models:
    # - Logitech Extreme 3D
    joystick_library = params['joystick_maps']
    if joy_name in joystick_library.keys():
        logger.info(f'[load_aircraft_parameters] joystick {joy_name} in database/JSON config file...will run online loop.')
        joy_map = joystick_library[joy_name]
        # axes config
        joy_map_axes = joy_map["axes"]
        consts['JOY_ROLL_AXIS'] = joy_map_axes["roll_axis"] # axis number that controls roll
        consts['JOY_PITCH_AXIS'] = joy_map_axes["pitch_axis"] # axis number that controls pitch
        consts['JOY_YAW_AXIS'] = joy_map_axes["yaw_axis"]  # axis number for yaw
        consts['JOY_THROTTLE_AXIS'] = joy_map_axes["throttle_axis"] # axis number for throttle control
        # buttons config
        joy_map_buttons = joy_map["buttons"]
        for i in range(joy_n_buttons):
            consts['JOY_BUTTONS_MAP'][list(consts['JOY_BUTTONS_MAP'].keys())[i]] = joy_map_buttons[list(consts['JOY_BUTTONS_MAP'].keys())[i]]

        consts['JOY_TRIM_PARAMS'] = joy_map["trim_params"]  # Trim rate adjustment (amount per second)
        consts['JOY_THROTTLE_LIMITS'] = joy_map["joy_throttle_limits"] # these are the limits for the throttle input for this specific joystick - full fwd = -1
        
        # linearly map the joystick input to the RCAM limits
        throttle_map_m = (acp.U_LIMITS_MAX[3] - acp.U_LIMITS_MIN[3]) / (consts['JOY_THROTTLE_LIMITS'][1] - consts['JOY_THROTTLE_LIMITS'][0])
        throttle_map_b = acp.U_LIMITS_MAX[3] - throttle_map_m * (consts['JOY_THROTTLE_LIMITS'][1]) # y = mx + b - equation for a line
        consts['JOY_FACTORS'] = joy_map["partial_joy_factors"]
        consts['JOY_FACTORS']['throttle_m'] = throttle_map_m
        consts['JOY_FACTORS']['throttle_b'] = throttle_map_b
        consts['OFFLINE'] = False # if a joytick is present, then run online


    else:
        logger.error('[load_aircraft_parameters] no joystick or joystick model not in databas/JSON config file...will run offline loop.')
        consts['OFFLINE'] = True

    return (consts, acp)