import numpy as np

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
N2LBF = 1 / LBF2N


# .. States and Controls Indices ..
# STATE INDICES
IDX_U, IDX_V, IDX_W = 0, 1, 2
IDX_P, IDX_Q, IDX_R = 3, 4, 5
IDX_PHI, IDX_THETA, IDX_PSI = 6, 7, 8

# CONTROL INDICES
IDX_AIL, IDX_ELE, IDX_RUD = 0, 1, 2
IDX_THR1, IDX_THR2 = 3, 4
IDX_FLAP, IDX_GEAR = 5, 6
IDX_GNDSP, IDX_BRAKE = 7, 8

# LANDING GEAR INDICES
IDX_MLG_X, IDX_MLG_Y, IDX_MLG_Z = 0, 1, 2

# DELTA CL, CD, CM & ALPHA
IDX_DCL, IDX_DCD, IDX_DCM, IDX_DALPHA = 0, 1, 2, 3

# Epsilon for controls first order clamping to 0.0 and 1.0
FIRST_ORDER_EPSILON = 1E-3

# Integration limits
MAX_INTEG_TIMESTEP_S = 0.05 # if the OS hangs, this is the max time step we will allow
MIN_AIRSPEED_FOR_ALPHA_BETA_M_S = 0.1 # below this airspeed, we set alpha and beta to zero

# Network timing
TERRAIN_POLL_INTERVAL_S = 0.01 # sleep time to prevent CPU hogging when polling for terrain packets
