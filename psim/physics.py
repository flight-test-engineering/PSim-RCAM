# Physics
# this file contains the core physics related functions

import numpy as np
from scipy import integrate
from scipy.optimize import minimize # for trimming routine
from numba import jit
from .constants import *
import psim.environment as env
from psim.helpers import logger

def initialize_constants(params: dict)->None:
    """
    Injects aircraft parameters into this module's global namespace 
    """
    globals().update(params)


# ############################################################################
# High Lift Devices Interpolator
# ############################################################################

@jit(nopython=True)
def array_interp(x:float, data_array:np.array, data_array_len:int) -> np.array:
    '''
    interpolator for flaps, gear
    interpolates between the two closest data points
    returns the interpolated array
    '''
    x = max(0.0, min(float(data_array_len), x))
    idx = int(x)
    frac = x - idx
    if idx >= data_array_len: return data_array[data_array_len]
    return data_array[idx] + (data_array[idx+1] - data_array[idx]) * frac 


# ############################################################################
# Controls and Actuators
# ############################################################################

# NUMBA DOES NOT LIKE THIS FUNCTION
# When Numba is enabled, trim function converges, but aircraft is not stable
#@jit(nopython=True) # DO NOT ENABLE Numba FOR THIS FUNCTION
def control_norm(U:np.array) -> np.array:
    '''
    normalizes controls to be sent to FG
    inputs:
        U controls: positions (in radians)
        [U_lim: control limits (in radians) moved to global variable for speed]
    returns:
        vector with control positions normalized between 1 and -1
    '''
    # Create local copy
    mins = U_LIMITS_MIN
    maxs = U_LIMITS_MAX
    
    # Avoid divide by zero
    mins = np.where(mins == 0, 1.0, U_LIMITS_MIN)
    maxs = np.where(maxs == 0, 1.0, U_LIMITS_MAX)
    
    # Vectorized normalization
    U_norm = np.where(U < 0, 
                      U / np.abs(mins), 
                      U / maxs)
    
    return U_norm


@jit(nopython=True)
def control_sat(U:np.ndarray) -> np.ndarray:
    '''
    saturates the control inputs to maximum allowable in RCAM model
    '''
    return np.clip(U, U_LIMITS_MIN, U_LIMITS_MAX)



@jit(nopython=True)
def update_actuators(U_cmd:np.ndarray, U_actual:np.ndarray, dt:float, tau:np.ndarray) -> np.ndarray:
    '''
    Simulates first order lag for control surfaces.
    y_dot = (u_cmd - y_current) / tau
    y_new = y_current + y_dot * dt
    Includes a 'snap-to-target' threshold to ensure values reach exactly 0.0 or 1.0.
    '''
    
    # 1. Standard First Order Lag Calculation
    # y_dot = (u_cmd - y_current) / tau
    rate = (U_cmd - U_actual) / tau 
    U_new = U_actual + rate * dt
    
    # 2. Snap-to-target Logic
    # Calculate the distance between the new calculated position and the commanded target
    dist_to_target = np.abs(U_new - U_cmd)
    
    # If the distance is smaller than epsilon, force the value to be exactly U_cmd.
    # Otherwise, use the calculated U_new.
    U_final = np.where(dist_to_target < FIRST_ORDER_EPSILON, U_cmd, U_new)
             
    return U_final


# ############################################################################
# Ground Reactions and Detection Logic
# ############################################################################

@jit(nopython=True)
def calculate_gear_compression(X:np.ndarray, h_cg:float) -> np.ndarray:
    '''
    Calculates the vertical compression of each landing gear strut.
    Positive value = Gear is in contact with ground (compressed).
    Negative value = Gear is in the air.
    
    Inputs:
        X: State vector (needs phi [6] and theta [7])
        h_cg: Current altitude of the Center of Gravity (meters)
        rwy_alt: height of the ground/runway (meters)
        
    Returns:
        compressions: np.array([nose_comp, main_l_comp, main_r_comp])
    '''
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
    '''
    Returns True (Ground) if ANY gear is compressed, False (Air) otherwise.
    This acts as the requested 'air_ground' variable.
    '''
    if compressions[0] > 0 or compressions[1] > 0 or compressions[2] > 0:
        return True
    else:
        return False


@jit(nopython=True)
def calculate_ground_forces(X:np.ndarray, h_cg:float, brake:float) -> np.ndarray:
    '''
    Calculates total Forces and Moments (Body Frame) from all 3 landing gears.
    h_cg is height AGL (RADALT) in (m)
    brake is a float between 0 and 1 - represents brake percent
    Returns: 6-element array [Fx, Fy, Fz, Mx, My, Mz]

    '''
    
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

            
            # Cap friction so it doesn't exceed mu * Normal Force (Coulomb friction)
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
def RCAM_model(X:np.ndarray, U:np.ndarray, rho:float, h:float, dcl:float, dcd:float, dcm:float, dalpha:float) -> np.ndarray:
    '''
    Modified RCAM model implementation
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
        dcl: Delta CL (High Lift / Ldg / Gnd Spoiler)
        dcd: Delta CD (High Lift / Ldg / Gnd Spoiler)
        dcm: Delta CM (High Lift / Ldg / Gnd Spoiler)
        dalpha: Delta alpha (High Lift / Ldg / Gnd Spoiler)
    outputs:
        X_dot: derivatives of states (same order)
    '''
   
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
    # RCAM modified to include dalpha and dcl and dgsp
    if (alpha + dalpha) <= ALPHA_SWITCH:
        CL_wb = N * (alpha - ALPHA_L0 + dalpha) * (1 - dgsp)
    else: 
        (A3 * (alpha + dalpha)**3 + A2 * (alpha + dalpha)**2 + A1 * (alpha + dalpha) + A0) * (1 - dgsp) + dcl


    # CL thrust
    epsilon = DEPSDA * (alpha - ALPHA_L0)
    # Prevent divide by zero in q_term, if speed is too low
    q_term = (EPSILON_DOT * q * LT / Va) if Va > 0.1 else 0.0
    alpha_t = alpha - epsilon + de + q_term
    CL_t = NT * (ST / S) * alpha_t

    # Total CL
    CL = CL_wb + CL_t

    # Total CD (in stability frame)
    CD = CDMIN + D1 * (N * alpha + D0)**2 + dcd # RCAM modified to include dcd

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
    eta21 = C_m_ALPHA - (NT * (ST * LT) / (S * CBAR)) * (alpha - epsilon) + dcm # RCAM Modified
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
    CMac_b = eta + np.dot(dCMdx, wbe_b) + np.dot(dCMdu, np.array([da, de, dr])) # RCAM original

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
    F1 = dt1 # modified RCAM, now we have thrust from cycle deck
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
    # This is extra to RCAM model
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
def RCAM_observe(X:np.ndarray, U:np.ndarray, rho:float, h:float, dcl:float, dcd:float, dcm:float, dalpha:float) -> np.ndarray:
    '''
    Performs the same calculations as RCAM_model but returns internal variables
    for logging purposes instead of state derivatives.
    
    Returns array:
    [0:Va, 1:alpha, 2:beta, 3:CL, 4:CD, 5:CY, 6:Gnd_Fx, 7:Gnd_Fy, 8:Gnd_Fz]
    -> you can add more variables if required
    '''
   
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
    if Va < MIN_AIRSPEED_FOR_ALPHA_BETA_M_S:
        alpha = 0.0
        beta = 0.0
    else:
        alpha = np.arctan2(w, u)
        beta = np.arcsin(v / Va)
       
    #----------------- aerodynamic force coefficients ---------------------
        # this is only available in the newer RCAM document (rev Feb 1997)
    # which is not availble to the public
    # CL - wing + body
    # RCAM modified to include dalpha and dcl and dgsp
    if (alpha + dalpha) <= ALPHA_SWITCH:
        CL_wb = N * (alpha - ALPHA_L0 + dalpha) * (1 - dgsp)
    else: 
        CL_wb = (A3 * (alpha + dalpha)**3 + A2 * (alpha + dalpha)**2 + A1 * (alpha + dalpha) + A0) * (1 - dgsp) + dcl

    # CL thrust
    epsilon = DEPSDA * (alpha - ALPHA_L0)
    # Prevent divide by zero
    q_term = (EPSILON_DOT * q * LT / Va) if Va > 0.1 else 0.0
    alpha_t = alpha - epsilon + de + q_term
    CL_t = NT * (ST / S) * alpha_t

    # Total CL
    CL = CL_wb + CL_t

    # Total CD (in stability frame)
    CD = CDMIN + D1 * (N * alpha + D0)**2 + dcd

    # Total side force CY (stability frame)
    CY = CY_BETA * beta + CY_DR * dr

    #---------------------- gravity effects ----------------------------------
    g_b = np.array([-G * np.sin(theta), G * np.cos(theta) * np.sin(phi), G * np.cos(theta) * np.cos(phi)])
    

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

def RCAM_model_wrapper(t, X, U, rho, h, dcl, dcd, dcm, dalpha):
    return RCAM_model(X, U, rho, h, dcl, dcd, dcm, dalpha)

def NED_wrapper(t, X, NED):
    return env.NED

def latlonh_dot_wrapper(t, X, V_NED, lat, h):
    return env.latlonh_dot(V_NED, lat, h)


# # # integrators
def ss_integrator(t_ini:float, X0:np.ndarray, U:np.ndarray, rho:float, h:float, dcl:float, dcd:float, dcm:float, dalpha:float):
    '''
    single step integrator for FDM
    returns scipy object, initialized
    '''

    RK_integrator = integrate.ode(RCAM_model_wrapper)
    RK_integrator.set_integrator('dopri5') 
    RK_integrator.set_f_params(U, rho, h, dcl, dcd, dcm, dalpha)

    RK_integrator.set_initial_value(X0, t_ini)
    return RK_integrator


def latlonh_int(t_ini:float, latlonh0:np.ndarray, V_NED:np.ndarray):
        
    '''
    single step integrator for lat/long/height
    returns scipy object, initialized
    '''
    
    RK_integrator = integrate.ode(latlonh_dot_wrapper)
    RK_integrator.set_integrator('dopri5')
    RK_integrator.set_f_params(V_NED, latlonh0[0], latlonh0[2])
    RK_integrator.set_initial_value(latlonh0, t_ini)
    return RK_integrator


# NUMBA/JIT WARM-UP
def compile_numba_functions():
        '''
        Runs Numba functions with dummy data to force JIT compilation 
        before the real-time loop begins.
        '''
        print('Compiling Numba functions (Warm-up)...', end="")
        
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
        _ = array_interp(0, HIGH_LIFT_COEFFS, MAX_FLAP)
        _ = control_sat(U)
        _ = control_norm(U)
        _ = update_actuators(U, U, 0.01, np.ones(9))
        _ = calculate_gear_compression(X, h)
        _ = get_air_ground_state(np.ones(3))
        _ = calculate_ground_forces(X, h, 0.0)
        _ = RCAM_model(X, U, rho, h, dcl, dcd, dcm, dalpha)
        _ = RCAM_observe(X, U, rho, h, dcl, dcd, dcm, dalpha)
        _ = RCAM_model_wrapper(t, X, U, rho, h, dcl, dcd, dcm, dalpha)
        _ = NED_wrapper(t, X, NED)
        _ = latlonh_dot_wrapper(t, X, NED, lat, h)
        _ = ss_integrator(t, X, U, rho, h, dcl, dcd, dcm, dalpha)
        _ = latlonh_int(t, latlonh0, NED)


        # 2. Environment
        _ = env.VA(np.array([10.,0.,0.]))
        _ = env.fpa(np.ones(3))
        _ = env.add_wind(np.ones(3), np.ones(3))
        _ = env.NED(np.array([10.,0.,0.]), np.array([0.,0.,0.]))
        _ = env.latlonh_dot(np.array([10.,0.,0.]), 0.0, 0.0)
        _ = env.WGS84_MN(lat)
        _ = env.get_rho(h)
        _ = env.get_AGL(latlonh0, h, 0.0)
        
        print(' Done.')




# ############################################################################
# Model Trimming Function
# ############################################################################

# define partial function with fixed flaps, gear position, ground spoilers and brake
# this partial function leaves "floating" only the parameters that the optimizer can vary
# because we trim only once, no big deal defining these special functions

def trim_functional3(Z:np.ndarray, VA_trim:float, gamma_trim:float, side_speed_trim:float,
                     phi_trim:float, psi_trim:float, rho_trim:float, h_trim:float, 
                     flap_pos:float, gear_pos:float, gnd_sp_pos:float, brakes_pos:float) -> np.dtype('f8'):
    '''
    functional to calculate a cost for minimizer (used to find trim point)
    inputs:
        Z: lumped vector of X (states) and U (control)
        trim targets:
        VA_trim: airspeed [m/s]
        gamma_trim: climb gradient [rad]
        side_speed_trim: lateral (v) speed [m/s]
        phi_trim: roll angle [rad]
        psi_trim: course angle [rad]
        rho_trim: density of air at trim condition [kg/m3]
        h_trim: height above the ground (m) - for ground proximity checks,
        flap_pos: number between 0 and MAX_FLAP
        gear_pos: 0 for gear up, 1 for down
        gnd_sp_pos: 0 for closed, 1 for open
        brakes_pos: 0 for not applied, 1 applied

    ****
    method
    Q.T*H*Q
    with H = diagonal matrix of "1"s (equal weights for all states)
    this returns the squares of the elements in Q
    
    returns:
        cost [float]
    '''

    X = Z[:9] # extract states
    
    # create controls vector with size of Z, minus 9 states, plus 4 extra controls
    U = np.zeros(Z.shape[0] - 9 + 4) 

    for i in range (Z.shape[0] - 9):
        U[i] = Z[9+i]

    # .. add extra control values
    U[Z.shape[0] -9 + 0] = flap_pos
    U[Z.shape[0] -9 + 1] = gear_pos
    U[Z.shape[0] -9 + 2] = gnd_sp_pos
    U[Z.shape[0] -9 + 3] = brakes_pos

    # add CL, CD, CM and delta Alpha modifiers due to gear and flaps
    dcl_dcd_dcm_dalpha = array_interp(flap_pos, HIGH_LIFT_COEFFS, MAX_FLAP)

    # interpolate for landing gear delta CD
    ldg_dcd = array_interp(gear_pos, LDG_DCD, MAX_LDG)
    dcl_dcd_dcm_dalpha[IDX_DCD] += ldg_dcd[0] # add additional drag from landing gear
    
    # calculate model
    X_dot = RCAM_model(X, U, rho_trim, h_trim, dcl_dcd_dcm_dalpha[IDX_DCL], dcl_dcd_dcm_dalpha[IDX_DCD], dcl_dcd_dcm_dalpha[IDX_DCM], dcl_dcd_dcm_dalpha[IDX_DALPHA])
    
    # calculate speed and gamma
    VA_current = env.VA(X[:3])
    gamma_current = X[IDX_THETA] - np.arctan2(X[IDX_W], X[IDX_U]) 
     
    Q = np.concatenate((X_dot, [VA_current - VA_trim], [gamma_current - gamma_trim], [X[IDX_V] - side_speed_trim], [X[IDX_PHI] - phi_trim], [X[IDX_PSI] - psi_trim]))
    square_ones = np.ones(Q.shape[0])
    H = np.diag(square_ones)
    
    return np.dot(np.dot(Q.T, H), Q)


def trim_model(VA_trim=85.0, gamma_trim=0.0, side_speed_trim=0.0, phi_trim=0.0, psi_trim=0.0, rho_trim=1.225, 
               h_trim=100.0, flap_pos=0, gear=0, gnd_sp=0, brakes=0,
               X0=np.array([85.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
               U0=np.array([1.0, 1.0, 1.0, 0.08, 0.08, 0.0, 0.0, 0.0, 0.0])) -> np.ndarray:
    '''
    uses scipy minimize on functional to find trim point
    X0 states:
        u, v, w, p, q, r, phi, theta, psi
    U0 controls:
        ail, ele, rud, thr1, thr2, flaps position, gear position, gnd spoiler, brake
    h_trim is passed on to check ground proximity

    '''

    # add target trim values to X0 vector, as a better initial guess for the states
    X0[IDX_U] = VA_trim
    X0[IDX_PHI] = phi_trim
    X0[IDX_PSI] = psi_trim

    # Add additional control values
    U0[IDX_FLAP] = flap_pos
    U0[IDX_GEAR] = gear
    U0[IDX_GNDSP] = gnd_sp
    U0[IDX_BRAKE] = brakes

    # loop control variables
    MAX_ITER = 10 
    iter_counter = 0
    epsilon = 1E-9
    converge = False

    # concatenate states and inputs into single vector
    # for trimming, ground spoilers and brakes are not a valid control
    # removing additional control values from trim variables
    # because we do not want the optimizer to play with them,
    # we add them separately:
    Z0 = np.concatenate((X0, U0[:-4])) 

    print(f'initial functional cost: {trim_functional3(Z0, VA_trim, gamma_trim, side_speed_trim, phi_trim, psi_trim, rho_trim, h_trim,
                             flap_pos, gear, gnd_sp, brakes):.3e}')


    while iter_counter <= MAX_ITER and not converge:
        # Updated args tuple to include h_trim
        result = minimize(trim_functional3, Z0, args=(VA_trim, gamma_trim, side_speed_trim, phi_trim, psi_trim, rho_trim, h_trim,
                          flap_pos, gear, gnd_sp, brakes),
                method='Nelder-Mead', options={'maxiter':50000,\
                                               'maxfev':40000})
        
        # Updated cost check with h_trim
        current_cost = trim_functional3(result.x, env.VA(result.x[:3]), result.x[IDX_THETA] - np.arctan2(result.x[IDX_W], result.x[IDX_U]), result.x[IDX_V], result.x[IDX_PHI], result.x[IDX_PSI], rho_trim, h_trim,
                                         flap_pos, gear, gnd_sp, brakes)
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
        print(f'check gamma {result.x[IDX_THETA] - np.arctan2(result.x[IDX_W], result.x[IDX_U])} RAD')
        print(f'check side vel {result.x[IDX_V]:.1f} m/s')
        print(f'check phi {result.x[IDX_PHI] * RAD2DEG:.1f} Deg')
        print(f'check psi {result.x[IDX_PSI]* RAD2DEG:.1f} Deg')
        logger.info("Trim converged")
    else:
        print('FAILED TO CONVERGE')
        logger.warning('Trim FAILED to converge')


    return result.x, result.message #remember that the control vector is missing ground spoilers now


