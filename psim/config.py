import json
import sys
import numpy as np
from .constants import DEG2RAD, RAD2DEG



# ############################################################################
# Aircraft Parameter Loader
# ############################################################################

def load_aircraft_parameters(filepath: str, joy_name: str|None) -> dict:
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
    # ... high lift coefficients ...
    # key: delta_CD, delta_CD, delta_CM, delta_alpha
    high_lift_dict = ac['high_lift_coeffs']
    consts['HIGH_LIFT_COEFFS'] = np.array([high_lift_dict[str(i)] for i in range(len(high_lift_dict))])
    for i in range(consts['HIGH_LIFT_COEFFS'].shape[0]):
        consts['HIGH_LIFT_COEFFS'][i][3] = consts['HIGH_LIFT_COEFFS'][i][3] * DEG2RAD # RCAM model uses alpha in radians
    consts['MAX_FLAP'] = int(consts['HIGH_LIFT_COEFFS'].shape[0] - 1)
    # ... landing gear drag increase ...
    ldg_drag_dict = ac['landing_gear_drag']
    consts['LDG_DCD'] = np.array([ldg_drag_dict[str(i)] for i in range(len(ldg_drag_dict))])   
    consts['MAX_LDG'] = int(consts['LDG_DCD'].shape[0] - 1)
    # ... ground spoilers lift dump ...
    consts['GND_SPOILERS_DCL'] = ac['gnd_spoilers_dcl']
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

    # Landing Gear
    # .. Geometry
    ldg_geom = params['landing_gear_geometry']
    consts['LG_NOSE_POS'] = np.array(ldg_geom['nose_pos']) # Nose Gear: ~10-12m forward of CG, centerline, ~2.5m below CG
    consts['LG_MAIN_L_POS'] = np.array(ldg_geom['left_mains_pos']) # Main Gear Left: ~4-5m behind CG, ~4m left, ~3.5m below CG
    consts['LG_MAIN_R_POS'] = np.array(ldg_geom['right_mains_pos'])

    # .. Dynamics
    ldg_dynamics = params['landing_gear_dynamics']
    consts['LG_SPRING_K'] = ldg_dynamics['lg_spring_k']
    consts['LG_DAMP_COMPRESSION'] = ldg_dynamics['lg_damp_compression']
    consts['LG_DAMP_REBOUND'] = ldg_dynamics['lg_damp_rebound']
    consts['LG_FRICTION_STIFFNESS'] = ldg_dynamics['lg_friction_stiffness'] # "Virtual spring" to hold plane still when stopped
    consts['LG_SIDE_FRICTION_MU'] = ldg_dynamics['lg_side_friction_mu']
    consts['LG_ROLLING_FRICTION_MU'] = ldg_dynamics['lg_rolling_friction_mu'] * 10 # DEBUG
    consts['LG_MU_BRAKE'] = ldg_dynamics['lg_mu_brake']

    # Actuator Dynamics
    act_dyn = params['actuator_dynamics']
    consts['ACT_TAU'] = np.array([act_dyn["tau_aileron"], 
                                  act_dyn["tau_elevator"], 
                                  act_dyn["tau_rudder"],
                                  act_dyn["tau_engine"],
                                  act_dyn["tau_engine"],
                                  act_dyn["tau_flaps"],
                                  act_dyn["tau_gear"], 
                                  act_dyn["tau_gnd_spoiler"],
                                  act_dyn["tau_brakes"]])

    # Joystick Mappings
    # available models:
    # - Logitech Extreme 3D
    joystick_library = params['joystick_maps']
    if joy_name in joystick_library.keys():
        print('yes')
        joy_map = joystick_library[joy_name]
        consts['JOY_ROLL_AXIS'] = joy_map["roll_axis"] # axis number that controls roll
        consts['JOY_PITCH_AXIS'] = joy_map["pitch_axis"] # axis number tht controls pitch
        consts['JOY_YAW_AXIS'] = joy_map["yaw_axis"]  # axis number for yaw
        consts['JOY_ZERO_AIL_RUD_THR'] = joy_map["zero_ail_rud_thr"] # convenience function to zero ail,rud and thrust trim points
        consts['JOY_ARM_DIS_GND_SPOILER'] = joy_map["arm_disarm_gnd_spoiler"] # toggles ARM/DISARM of ground spoilers
        consts['JOY_PITCH_TRIM_DN'] = joy_map["pitch_dn"] # pitch trim nose down button
        consts['JOY_PITCH_TRIM_UP'] = joy_map["pitch_up"] # pitch trim nose up button
        #consts['JOY_ROLL_TRIM_RH'] = joy_map["roll_rt"] # roll trim right wing down button
        #consts['JOY_ROLL_TRIM_LH'] = joy_map["roll_lt"] # roll trim left wing down button
        consts['JOY_E1_CYCLE_CUT'] = joy_map["E1_cycle_cut"] # cut/restart engine 1
        consts['JOY_LDG_CYCLE'] = joy_map["ldg_cycle"] # cycle landing gear down/up (default is up)
        consts['JOY_E1_THR_TRIM_FWD'] = joy_map["T1_fd"] # E1 trim forward (adds incremental thrust)
        consts['JOY_E1_THR_TRIM_AFT'] = joy_map["T1_af"] # E1 trim aft (subtracts incremental thrust)
        #consts['JOY_E2_THR_TRIM_FWD'] = joy_map["T2_fd"] # same for E2
        #consts['JOY_E2_THR_TRIM_AFT'] = joy_map["T2_af"] # same for E2
        consts['JOY_FLAP_CMD_UP'] = joy_map['flaps_command_up'] # command flaps one notch up
        consts['JOY_FLAP_CMD_DN'] = joy_map['flaps_command_dn'] # command flaps one notch down
        consts['JOY_EXIT_SIGNAL'] = joy_map["exit_signal"] # ends the simulation
        consts['JOY_BRAKE'] = joy_map["brake"] # ends the simulation
        consts['JOY_TRIM_PARAMS'] = joy_map["trim_params"]  # Trim rate adjustment (amount per second)
        consts['JOY_THROTTLE_LIMITS'] = joy_map["joy_throttle_limits"] # these are the limits for the throttle input for this specific joystick - full fwd = -1
        
        # linearly map the joystick input to the RCAM limits
        throttle_map_m = (consts['U_LIMITS_MAX'][3] - consts['U_LIMITS_MIN'][3]) / (consts['JOY_THROTTLE_LIMITS'][1] - consts['JOY_THROTTLE_LIMITS'][0])
        throttle_map_b = consts['U_LIMITS_MAX'][3] - throttle_map_m * (consts['JOY_THROTTLE_LIMITS'][1]) # y = mx + b - equation for a line
        consts['JOY_FACTORS'] = joy_map["partial_joy_factors"]
        consts['JOY_FACTORS']['throttle_m'] = throttle_map_m
        consts['JOY_FACTORS']['throttle_b'] = throttle_map_b
        consts['OFFLINE'] = False # if a joytick is present, then run online


    else:
        print('no')
        consts['OFFLINE'] = True

    return consts