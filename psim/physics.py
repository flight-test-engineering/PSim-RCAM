# Physics
# this file contains the core physics related functions

import numpy as np
from scipy import integrate
from scipy.optimize import minimize # for trimming routine
from numba import jit, float64, int32
from numba.experimental import jitclass
from .constants import *
import psim.environment as env
from psim.helpers import logger

def initialize_constants(params: dict)->None:
    """
    Injects aircraft parameters into this module's global namespace 
    """
    globals().update(params)

# Define the data and types for the FDM State
# see RCAM_model() for details on each quantity
state_spec =[
    # Inputs / Configurations
    ('X', float64[:]),
    ('U', float64[:]),
    ('rho', float64),
    ('h', float64),
    ('dcl', float64),
    ('dcd', float64),
    ('dcm', float64),
    ('dalpha', float64),
    ('dN', float64),
    
    # Outputs / Intermediates (previously from RCAM_observe)
    ('dX', float64[:]),
    ('Va', float64),
    ('alpha', float64),
    ('beta', float64),
    ('CL', float64),
    ('CD', float64),
    ('CY', float64),
    ('F_gnd_x', float64),
    ('F_gnd_y', float64),
    ('F_gnd_z', float64),
    
    # Body Accelerations for FlightGear
    ('body_accels', float64[:]),
    ('load_factor', float64),
]

@jitclass(state_spec)
class FDMState:
    def __init__(self):
        # Initialize arrays and scalars
        self.X = np.zeros(9)
        self.U = np.zeros(9)
        self.rho = 1.225
        self.h = 0.0
        self.dcl = 0.0
        self.dcd = 0.0
        self.dcm = 0.0
        self.dalpha = 0.0
        self.dN = 0.0
        
        self.dX = np.zeros(9)
        self.body_accels = np.zeros(3)
        self.Va = 0.0
        self.alpha = 0.0
        self.beta = 0.0
        self.CL = 0.0
        self.CD = 0.0
        self.CY = 0.0
        self.F_gnd_x = 0.0
        self.F_gnd_y = 0.0
        self.F_gnd_z = 0.0





# ############################################################################
# General Array Interpolator
# ############################################################################

@jit(nopython=True)
def array_interp(x:float, data_array:np.array, data_array_len:int) -> np.array:
    '''
    interpolator for flaps, gear
    interpolates between the two closest data points
    returns the interpolated array
    '''
    local_data_array = np.copy(data_array)
    x = max(0.0, min(float(data_array_len), x))
    idx = int(x)
    frac = x - idx
    if idx >= data_array_len: return local_data_array[data_array_len]
    return local_data_array[idx] + (local_data_array[idx+1] - local_data_array[idx]) * frac


# ############################################################################
# Controls and Actuators
# ############################################################################

# NUMBA DOES NOT LIKE THIS FUNCTION
# When Numba is enabled, trim function converges, but aircraft is not stable
#@jit(nopython=True) # DO NOT ENABLE Numba FOR THIS FUNCTION
def control_norm(U:np.array, acp:jitclass) -> np.array:
    '''
    normalizes controls to be sent to FG
    inputs:
        U controls: positions (in radians)
        [U_lim: control limits (in radians) moved to global variable for speed]
    returns:
        vector with control positions normalized between 1 and -1
    '''
    # Create local copy
    mins = acp.U_LIMITS_MIN
    maxs = acp.U_LIMITS_MAX
    
    # Avoid divide by zero
    mins = np.where(mins == 0, 1.0, acp.U_LIMITS_MIN)
    maxs = np.where(maxs == 0, 1.0, acp.U_LIMITS_MAX)
    
    # Vectorized normalization
    U_norm = np.where(U < 0, 
                      U / np.abs(mins), 
                      U / maxs)
    
    return U_norm


@jit(nopython=True)
def control_sat(U:np.ndarray, acp:jitclass) -> np.ndarray:
    '''
    saturates the control inputs to maximum allowable in RCAM model
    '''
    return np.clip(U, acp.U_LIMITS_MIN, acp.U_LIMITS_MAX)



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
def calculate_gear_compression(X:np.ndarray, h_cg:float, acp:jitclass) -> np.ndarray:
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
    gear_positions[0, :] = acp.LG_NOSE_POS
    gear_positions[1, :] = acp.LG_MAIN_L_POS
    gear_positions[2, :] = acp.LG_MAIN_R_POS
    
    # Calculate vertical distance (dz) for all 3 gears at once
    # Result is a vector of 3 elements
    dz = gear_positions @ DCM_z_row 
    
    # Calculate tips
    h_tips = h_cg - dz
    
    # Compression is simply -h_tips (assuming ground is 0 relative to h_cg passed in)
    # We clip negative values (air) to 0.0
    compressions = np.maximum(-h_tips, 0.0)
    
    # Clip max compression
    max_travel = np.array([acp.LG_NOSE_POS[2], acp.LG_MAIN_L_POS[2], acp.LG_MAIN_R_POS[2]])
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
def calculate_ground_forces(X:np.ndarray, h_cg:float, brake:float, acp:jitclass) -> np.ndarray:
    '''
    Calculates total Forces and Moments (Body Frame) from all 3 landing gears.
    h_cg is height AGL (RADALT) in (m)
    brake is a float between 0 and 1 - represents brake percent
    Returns: 6-element array [Fx, Fy, Fz, Mx, My, Mz]

    '''
    
    # Get current compressions
    compressions = calculate_gear_compression(X, h_cg, acp)

    
    # State extractions for velocity calc
    u, v, w = X[IDX_U], X[IDX_V], X[IDX_W]
    p, q, r = X[IDX_P], X[IDX_Q], X[IDX_R]
    wbe_b = np.array([p, q, r])
    V_cg_b = np.array([u, v, w])
    
    total_F = np.zeros(3)
    total_M = np.zeros(3)
    
    # Loop through the 3 gears
    # 0=Nose, 1=MainL, 2=MainR
    gears_pos = [acp.LG_NOSE_POS, acp.LG_MAIN_L_POS, acp.LG_MAIN_R_POS]
    
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
                damping_force = acp.LG_DAMP_COMPRESSION * v_z_gear
            else:
                # Rebounding (spring pushing plane up)
                # Use HIGH damping to prevent the spring from shooting the plane up
                damping_force = acp.LG_DAMP_REBOUND * v_z_gear
            
            # Normal Force = Spring Force - Damping Force
            # Note: We use max(0, ...) on the spring to ensure it pushes up.
            F_normal = -max(0.0, acp.LG_SPRING_K * delta_z) - damping_force
            
            # Sanity Check: The ground cannot pull the plane down. 
            # If the damping force is so high it overcomes the spring during rebound, 
            # clamp force to zero.
            if F_normal > 0: 
                F_normal = 0 
            
            # Friction
            # A simple "stiffness" approach to friction to stop sliding
            # Side force
            F_y = -V_gear[1] * 50000.0 
            raw_fy = -V_gear[1] * acp.LG_FRICTION_STIFFNESS
            # for longitudinal force, we have the brakes
            F_x = F_normal * (acp.LG_MU_BRAKE * brake + acp.LG_ROLLING_FRICTION_MU)
            raw_fx = -V_gear[0] * acp.LG_FRICTION_STIFFNESS

            
            # Cap friction so it doesn't exceed mu * Normal Force (Coulomb friction)
            # This prevents numerical instability if sliding sideways fast
            max_fx = abs(F_normal * (acp.LG_MU_BRAKE + acp.LG_ROLLING_FRICTION_MU))
            max_fy = abs(F_normal * acp.LG_SIDE_FRICTION_MU)
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
def RCAM_model(state: FDMState, acp: jitclass) -> None:
    '''
    Modified RCAM model implementation
    sources: RCAM docs and Christopher Lum
    Group for Aeronautical Research and Technology Europe (GARTEUR) - Research Civil Aircraft Model (RCAM)
    http://garteur.org/wp-content/reports/FM/FM_AG-08_TP-088-3.pdf

    Christopher Lum - Equations/Modeling
    https://www.youtube.com/watch?v=bFFAL9lI2IQ
    Christopher Lum - Matlab implementation
    https://www.youtube.com/watch?v=m5sEln5bWuM

    inputs (via dataclass):
        FDMState.X: states (TP-088-3, p. 6, para 2.2, table 2.2)
            0: u (m/s)
            1: v (m/s)
            2: w (m/s)
            3: p (rad/s)
            4: q (rad/s)
            5: r (rad/s)
            6: phi (rad)
            7: theta (rad)
            8: psi (rad)
        FDMState.U: controls (TP-088-3, p. 6, para 2.2, table 2.1)
            0: aileron (rad)
            1: stabilator (rad)
            2: rudder (rad)
            3: E1 THRUST (N) (original RCAM was throttle 1 in %)
            4: E2 THRUST (N)  (original RCAM was throttle 2 in %)
            5: Flaps position
            6: Landing gear position (0=up / 1=dn)
            7: spoilers (%) (not included in original RCAM)
            8: wheel brakes (%) (not included in original RCAM)
        FDMState.rho: density for current altitude (kg/m3)
        FDMState.h: height above ground (m)
        FDMState.dcl: Delta CL (High Lift / Ldg / Gnd Spoiler)
        FDMState.dcd: Delta CD (High Lift / Ldg / Gnd Spoiler)
        FDMState.dcm: Delta CM (High Lift / Ldg / Gnd Spoiler)
        FDMState.dalpha: Delta alpha (High Lift / Ldg / Gnd Spoiler)
        FDMState.dN: Delta lift curve slope
    outputs (via dataclass)
        FDMState.dX: derivatives of states (same order)
        FDMState.Va: true airspeed (m/s)
        FDMState.alpha (rad)
        FDMState.beta (rad)
        FDMState.CL
        FDMState.CD
        FDMState.CY
        FDMState.F_gnd_x: ground force x direction (N)
        FDMState.F_gnd_y: ground force y direction (N)
        FDMState.F_gnd_z: ground force z direction (N)
        FDMState.body_accels: in body frame, (m/s2)
        FDMState.load_factor
    '''
   
    # --------------- extract data from dataclass, to local variables ----
    rho = state.rho
    h = state.h

    # ------------------------- states ----------------------------------
    u, v, w = state.X[IDX_U], state.X[IDX_V], state.X[IDX_W]
    p, q, r = state.X[IDX_P], state.X[IDX_Q], state.X[IDX_R]
    phi, theta, psi = state.X[IDX_PHI], state.X[IDX_THETA], state.X[IDX_PSI]

    # ----------------------- controls & env ----------------------------
    da, de, dr = state.U[IDX_AIL], state.U[IDX_ELE], state.U[IDX_RUD]
    dt1, dt2 = state.U[IDX_THR1], state.U[IDX_THR2]
    flap_pos, gear_pos = state.U[IDX_FLAP], state.U[IDX_GEAR] # not part of RCAM original doc
    dgsp, brake = state.U[IDX_GNDSP], state.U[IDX_BRAKE] # not part of RCAM original doc
    
    dcl, dcd, dcm, dalpha, dN = state.dcl, state.dcd, state.dcm, state.dalpha, state.dN # not part of RCAM original doc

    # step 2
    #----------------- intermediate variables ---------------------------
    # airspeed
    state.Va = np.sqrt(u**2 + v**2 + w**2) # m/s
    
    # alpha and beta
    # Protect against divide by zero if Va is very small (on ground)
    if state.Va < MIN_AIRSPEED_FOR_ALPHA_BETA_M_S:
        alpha = 0.0
        beta = 0.0
    else:
        alpha = np.arctan2(w, u)
        beta = np.arcsin(v / state.Va)
    
    # dynamic pressure
    Q = 0.5 * state.rho * state.Va**2
    
    # define vectors wbe_b and V_b
    wbe_b = np.array([p, q, r])
    V_b = np.array([u, v, w])
    
    # step 3
    #----------------- aerodynamic force coefficients ---------------------
    # this is only available in the newer RCAM document (rev Feb 1997)
    # however this version is modified to include high lift devices
    # CL - wing + body
    # dalpha, dcl, dN and dgsp - not part of RCAM original doc
    alpha_corr = alpha - dalpha # correct alpha for high lift devices
    if alpha_corr <= acp.ALPHA_SWITCH:
        CL_wb = ((acp.N + dN) * (alpha_corr - acp.ALPHA_L0) + dcl) * (1 - dgsp)
    else:
        CL_wb = ((acp.A3 * alpha_corr**3 + acp.A2 * alpha_corr**2 + acp.A1 * alpha_corr + acp.A0)
                  + dcl + (dN) * (acp.ALPHA_SWITCH - acp.ALPHA_L0)) * (1 - dgsp)

    # clip CL_wb # not part of RCAM original doc
    if CL_wb < -1.0: CL_wb = -1.0


    # CL thrust
    epsilon = acp.DEPSDA * (alpha - acp.ALPHA_L0)
    # Prevent divide by zero in q_term, if speed is too low
    q_term = (acp.EPSILON_DOT * q * acp.LT / state.Va) if state.Va > 0.1 else 0.0
    alpha_t = alpha - epsilon + de + q_term
    CL_t = acp.NT * (acp.ST / acp.S) * alpha_t

    # Total CL
    CL = CL_wb + CL_t

    # Total CD (in stability frame)
    CD = acp.CDMIN + acp.D1 * (acp.N * alpha + acp.D0)**2 + dcd # dcd - not part of RCAM original doc


    # Total side force CY (stability frame)
    CY = acp.CY_BETA * beta + acp.CY_DR * dr

    # Step 4
    #------------------- dimensional aerodynamic forces --------------------
    # forces in F_s
    FA_s = np.array([-CD * Q * acp.S, CY * Q * acp.S, -CL * Q * acp.S])

    # rotate forces to body axis (F_b)
    C_bs = np.array([[np.cos(alpha), 0.0, -np.sin(alpha)],
                     [0.0, 1.0, 0.0],
                     [np.sin(alpha), 0.0, np.cos(alpha)]], dtype=np.dtype('f8'))

    FA_b = np.dot(C_bs, FA_s)   
    
    # Step 5
    #------------------ aerodynamic moment coefficients about AC -----------
    # moments in F_b
    eta11 = acp.C_l_BETA * beta
    eta21 = acp.C_m_ZERO + (acp.C_m_ALPHA * (acp.ST * acp.LT) / (acp.S * acp.CBAR)) * (alpha - epsilon) + dcm # RCAM Modified
    eta31 = (1 - alpha * acp.C_n_BETA) * beta

    eta = np.array([eta11, eta21, eta31])
    
    # Prevent divide by zero in damping terms
    inv_Va = (acp.CBAR / state.Va) if state.Va > 0.1 else 0.0

    dCMdx = inv_Va * np.array([[acp.C_l_P, acp.C_l_Q, acp.C_l_R], # moment change with respect to states
                                    [acp.C_m_P, (acp.C_m_Q * (acp.ST * acp.LT**2) / (acp.S * acp.CBAR**2)), acp.C_m_R], 
                                    [acp.C_n_P, acp.C_n_Q, acp.C_n_R]], dtype=np.dtype('f8'))
    dCMdu = np.array([[acp.C_l_DA , acp.C_l_DE, acp.C_l_DR], # control authority/effectiveness matrix
                      [acp.C_m_DA, (acp.C_m_DE * (acp.ST * acp.LT) / (acp.S * acp.CBAR)), acp.C_m_DR],
                      [acp.C_n_DA, acp.C_n_DE, acp.C_n_DR]], dtype=np.dtype('f8'))
    
    # CM about AC in Fb
    CMac_b = eta + np.dot(dCMdx, wbe_b) + np.dot(dCMdu, np.array([da, de, dr]))

    # Step 5A
    # Add ground effect
    ground_effect_deltas = calc_ground_effect((state.h - acp.LG_MAIN_R_POS[2]) , acp) # state.h and gear are in meters
    CL_wb += ground_effect_deltas[0]
    CD += ground_effect_deltas[1]
    CMac_b += ground_effect_deltas[2]

    # Step 6
    #------------------- aerodynamic moment about AC -------------------------
    # normalize to aerodynamic moment
    MAac_b = CMac_b * Q * acp.S * acp.CBAR

    # Step 7
    #-------------------- aerodynamic moment about CG ------------------------
    rcg_b = np.array([acp.XCG, acp.YCG, acp.ZCG])
    rac_b = np.array([acp.XAC, acp.YAC, acp.ZAC])

    MAcg_b = MAac_b + np.cross(FA_b, rcg_b - rac_b)
    
    # Step 8
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
    mew1 = np.array([acp.XCG - acp.XAPT1, acp.YAPT1 - acp.YCG, acp.ZCG - acp.ZAPT1])
    mew2 = np.array([acp.XCG - acp.XAPT2, acp.YAPT2 - acp.YCG, acp.ZCG - acp.ZAPT2])

    MEcg1_b = np.cross(mew1, FE1_b)
    MEcg2_b = np.cross(mew2, FE2_b)

    MEcg_b = MEcg1_b + MEcg2_b
    
    # Step 9
    #---------------------- gravity effects ----------------------------------
    g_b = np.array([-G * np.sin(theta), G * np.cos(theta) * np.sin(phi), G * np.cos(theta) * np.cos(phi)])

    Fg_b = acp.M * g_b

    #---------------------- GROUND REACTION ----------------------------------
    # not part of RCAM original doc
    Gnd_Reac = calculate_ground_forces(state.X, h, brake, acp)
    F_gnd_b = Gnd_Reac[:3]
    M_gnd_b = Gnd_Reac[3:]
    
    # Step 10
    #---------------------- state derivatives --------------------------------
    # form F_b and calculate u, v, w dot
    # Added F_gnd_b to the sum
    F_b = Fg_b + FE_b + FA_b + F_gnd_b
    
    u_v_w_dot  = (1 / acp.M) * F_b - np.cross(wbe_b, V_b)
    
    # form Mcg_b and calc p, q r dot
    # Added M_gnd_b to the sum
    Mcg_b = MAcg_b + MEcg_b + M_gnd_b
    
    p_q_r_dot = np.dot(acp.INV_INERTIA_TENSOR_b, (Mcg_b - np.cross(wbe_b, np.dot(acp.INERTIA_TENSOR_b , wbe_b))))
    
    # phi, theta, psi dot -> Euler
    H_phi = np.array([[1.0, np.sin(phi) * np.tan(theta), np.cos(phi) * np.tan(theta)],
                      [0.0, np.cos(phi), -np.sin(phi)],
                      [0.0, np.sin(phi) / np.cos(theta), np.cos(phi) / np.cos(theta)]], dtype=np.dtype('f8'))
    
    phi_theta_psi_dot = np.dot(H_phi, wbe_b)

    # Calculate Body Accelerations for FlightGear natively
    # Specific Force = (Total Forces excluding Gravity) / Mass
    # OR simply: u_v_w_dot + g_b (matches your current calculation perfectly)
    state.body_accels = u_v_w_dot + np.dot(np.array([[0, -r, q],[r, 0, -p],[-q, p, 0]]), np.array([u, v, w]))
    state.load_factor =  (g_b[IDX_W] - state.body_accels[IDX_W]) / G
    
    # Store outputs to the state object
    state.dX = np.concatenate((u_v_w_dot, p_q_r_dot, phi_theta_psi_dot))
    
    state.alpha = alpha
    state.beta = beta
    state.CL = CL
    state.CD = CD
    state.CY = CY
    state.F_gnd_x = F_gnd_b[0]
    state.F_gnd_y = F_gnd_b[1]
    state.F_gnd_z = F_gnd_b[2]
        
    return None





# ############################################################################
# Model Integration
# ############################################################################

# # # wrappers
    # Scipy's "integrate.ode" does not accept a numba/@jit(nopython=True) compiled function
    # therefore, we need to create dummy wrappers

def RCAM_model_wrapper(t, X, state, acp):
    state.X = X  # Feed the current RK evaluation state into the dataclass
    RCAM_model(state, acp)
    return state.dX # Return the derivative array

def NED_wrapper(t, X, NED):
    return env.NED

def latlonh_dot_wrapper(t, X, V_NED, lat, h):
    return env.latlonh_dot(V_NED, lat, h)


# # # integrators
def ss_integrator(t_ini:float, X0:np.ndarray, state:FDMState, acp:jitclass):
    '''
    single step integrator for FDM
    returns scipy object, initialized
    '''
    RK_integrator = integrate.ode(RCAM_model_wrapper)
    RK_integrator.set_integrator('dopri5') 
    RK_integrator.set_f_params(state, acp) # Pass the state object pointer
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






# ############################################################################
# Model Trimming Function
# ############################################################################

# define partial function with fixed flaps, gear position, ground spoilers and brake
# this partial function leaves "floating" only the parameters that the optimizer can vary
# because we trim only once, no big deal defining these special functions

def trim_functional3(Z:np.ndarray, VA_trim:float, gamma_trim:float, side_speed_trim:float,
                     phi_trim:float, psi_trim:float, rho_trim:float, h_trim:float, 
                     flap_pos:float, gear_pos:float, gnd_sp_pos:float, brakes_pos:float, 
                     trim_state, acp:jitclass) -> float:
    '''
    functional to calculate a cost for minimizer (used to find trim point)
    inputs:
        Z: lumped vector of X (states) and U (control sub-set)
            states:
                u, v, w, p, q, r, phi, theta, psi
            controls sub-set:
                ail, ele, rud, thr1, thr2
        
        controls *NOT USED* by trimming function (and not part of Z):
        (because we do not want the optimizer to play with these)
            flap_pos: number between 0 and MAX_FLAP
            gear_pos: 0 for gear up, 1 for down
            gnd_sp_pos: 0 for closed, 1 for open
            brakes_pos: 0 for not applied, 1 applied

        trim targets:
            VA_trim: airspeed [m/s]
            gamma_trim: climb gradient [rad]
            side_speed_trim: lateral (v) speed [m/s]
            phi_trim: roll angle [rad]
            psi_trim: course angle [rad]
        conditions to trim at:
            rho_trim: density of air at trim condition [kg/m3]
            h_trim: height above the ground (m) - for ground proximity checks,


    method:
        Q.T*H*Q
        with H = diagonal matrix of "1"s (equal weights for all states)
        this returns the squares of the elements in Q
    
    returns:
        cost [float]
    '''

    # re-create the variables (states and controls)
    # in form and shape that the RCAM_model function expects

    X = Z[:9] # extract states
    
    # create controls vector with size of Z, minus 9 states, plus 4 extra controls
    # (flaps position, gear position, gnd spoiler, brake)
    U = np.zeros(Z.shape[0] - 9 + 4) 

    # copy control sub-set values over to U
    for i in range (Z.shape[0] - 9):
        U[i] = Z[9+i]

    # .. add extra control values
    U[Z.shape[0] -9 + 0] = flap_pos
    U[Z.shape[0] -9 + 1] = gear_pos
    U[Z.shape[0] -9 + 2] = gnd_sp_pos
    U[Z.shape[0] -9 + 3] = brakes_pos

    # add CL, CD, CM and delta Alpha modifiers due to gear and flaps
    aero_delta_coeffs = array_interp(flap_pos, acp.HIGH_LIFT_COEFFS, acp.MAX_FLAP)

    # interpolate for landing gear delta CD and delta CM
    ldg_dcd_dcm = array_interp(gear_pos, acp.LDG_DCD_DCM, acp.MAX_LDG)
    aero_delta_coeffs[IDX_DCD] += ldg_dcd_dcm[0] # add additional cd from landing gear
    aero_delta_coeffs[IDX_DCM] += ldg_dcd_dcm[1] # add additional cm from landing gear
    
    trim_state.X = X
    trim_state.U = U
    trim_state.rho = rho_trim
    trim_state.h = h_trim
    trim_state.dcl = aero_delta_coeffs[IDX_DCL]
    trim_state.dcd = aero_delta_coeffs[IDX_DCD]
    trim_state.dcm = aero_delta_coeffs[IDX_DCM]
    trim_state.dalpha = aero_delta_coeffs[IDX_DALPHA]
    
    # calculate model (mutates trim_state in place)
    RCAM_model(trim_state, acp)
    X_dot = trim_state.dX

    # calculate speed and gamma
    VA_current = trim_state.Va
    gamma_current = X[IDX_THETA] - np.arctan2(X[IDX_W], X[IDX_U]) 
     
    Q = np.concatenate((X_dot, [VA_current - VA_trim], [gamma_current - gamma_trim], [X[IDX_V] - side_speed_trim], [X[IDX_PHI] - phi_trim], [X[IDX_PSI] - psi_trim]))
    square_ones = np.ones(Q.shape[0])
    H = np.diag(square_ones)
    
    return np.dot(np.dot(Q.T, H), Q)


def trim_model(VA_trim=85.0, gamma_trim=0.0, side_speed_trim=0.0, phi_trim=0.0, psi_trim=0.0, rho_trim=1.225, 
               h_trim=100.0, flap_pos=0, gear=0, gnd_sp=0, brakes=0,
               X0=np.array([85.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 
               U0=np.array([1.0, 1.0, 1.0, 0.08, 0.08, 0.0, 0.0, 0.0, 0.0]),
               acp=None) -> tuple[np.ndarray, str]:
    '''
    uses scipy minimize on functional to find trim point
    X0 states:
        u, v, w, p, q, r, phi, theta, psi
    U0 controls:
        ail, ele, rud, thr1, thr2, flaps position, gear position, gnd spoiler, brake
    with the caveat that the following controls are locked for the optimzer:
        flaps position, gear position, gnd spoiler, brake

    '''

    trim_state = FDMState()

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
    # for trimming, flaps, gear, ground spoilers and brakes are not a valid control
    # removing additional control values from trim variables
    # because we do not want the optimizer to play with them,
    # we add them separately:
    Z0 = np.concatenate((X0, U0[:-4])) 

    logger.info(f'[trim_model] initial functional cost: {trim_functional3(Z0, VA_trim, gamma_trim, side_speed_trim, phi_trim, psi_trim, rho_trim, h_trim, flap_pos, gear, gnd_sp, brakes, trim_state, acp):.3e}')


    while iter_counter <= MAX_ITER and not converge:
        # Updated args tuple to include h_trim
        result = minimize(trim_functional3, Z0, 
                          args=(VA_trim, gamma_trim, side_speed_trim, phi_trim, psi_trim, rho_trim, h_trim, flap_pos, gear, gnd_sp, brakes, trim_state, acp),
                          method='Nelder-Mead', 
                          options={'maxiter':50000, 'maxfev':40000})
        
        # Cost check
        current_cost = trim_functional3(result.x, env.VA(result.x[:3]), result.x[IDX_THETA] - np.arctan2(result.x[IDX_W], result.x[IDX_U]), result.x[IDX_V], result.x[IDX_PHI], result.x[IDX_PSI], rho_trim, h_trim, flap_pos, gear, gnd_sp, brakes, trim_state, acp)
        logger.info(f'[trim_model] iter: {iter_counter}, functional cost: {current_cost:.3e}')

        if current_cost < epsilon:
            converge = True
        else:
            iter_counter += 1
            Z0 = result.x.copy()


    if converge:
        logger.info(f'[trim_model] Trim converged! Speed: {env.VA(Z0[:3]):.1f} m/s, Gamma: {result.x[IDX_THETA] - np.arctan2(result.x[IDX_W], result.x[IDX_U])} RAD')
    else:
        logger.warning('[trim_model] Trim FAILED to converge')


    return result.x, result.message #remember that the control vector is missing flaps position, gear position, gnd spoiler, brake now

@jit(nopython=True)
def calc_ground_effect(h_agl: float, acp:jitclass) -> np.ndarray:
    '''
    Calculates aerodynamic multipliers for Ground Effect.
    Based on relative height to wingspan (h/b).
    Inputs:
        h_agl : [m] - height above terrain
        acp: aircraft parameters dataclass (wingspan is stored in [m])
    
    Returns:
        np.array with:
            delta_CL_IGE: delta CL in ground effect
            delta_CD_IGE: ...
            delta_CM_IGE: ...
    '''
    # 1. Calculate Ratio
    # We use max(h, 0) to handle slight underground numeric errors
    ratio = max(h_agl, 0.0) / acp.WINGSPAN
    
    # 2. Check Range (Effect vanishes above 1.0 span)
    if ratio >= 1.0:
        return np.zeros(3)
        
    # 3. Calculate Exponential values
    accel_factor = 8
    deltas = np.array([acp.delta_CL_IGE, acp.delta_CD_IGE, acp.delta_CM_IGE])

    return np.exp(accel_factor * (1 - ratio)) * deltas / np.exp(accel_factor) 
