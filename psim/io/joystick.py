import numpy as np
import pygame
from psim.constants import *

# ############################################################################
# Module Initialization
# ############################################################################
def initialize_constants(params: dict):
    """
    Injects aircraft parameters into this module's global namespace 
    """
    globals().update(params)

def get_joy_inputs(joystick, U_trim, fr, trim_params, joy_factors):
    '''
    function that will read joystick positions and adjust controls:
    1. joy will change controls on top of trim point
    2. trim settings (buttons) will change trim point
    3. engine does not have trim function, but depending on
    button pressed, throttle should be commanded left/right/both
    '''
    U = np.zeros(U_trim.shape)

    # multipliers to adjust how much trim is added per integration step.
    # --- TRIM ---
    pitch_trim_step = trim_params['pitch'] / fr
    aileron_trim_step = trim_params['aileron'] / fr
    throttle_trim_step = trim_params['throttle'] / fr

    # read joystick button states for trimming
    zero_ail_rud_thr = joystick.get_button(JOY_ZERO_AIL_RUD_THR)
    pitch_dn = joystick.get_button(JOY_PITCH_TRIM_DN)
    pitch_up = joystick.get_button(JOY_PITCH_TRIM_UP)
    roll_rt = joystick.get_button(JOY_ROLL_TRIM_RH)
    roll_lt = joystick.get_button(JOY_ROLL_TRIM_LH)
    T1_fd = joystick.get_button(JOY_E1_THR_TRIM_FWD)
    T1_af = joystick.get_button(JOY_E1_THR_TRIM_AFT)
    T2_fd = joystick.get_button(JOY_E2_THR_TRIM_FWD)
    T2_af = joystick.get_button(JOY_E2_THR_TRIM_AFT)
    exit_signal = joystick.get_button(JOY_EXIT_SIGNAL)
    brake_applied = joystick.get_button(JOY_BRAKE)
    toggle_gnd_spoiler = joystick.get_button(JOY_ARM_DIS_GND_SPOILER)

    # if trigger is pressed, then zero out aileron, rudder states and make thrust equal on both sides
    if zero_ail_rud_thr == 1:
        U_trim[IDX_AIL] = 0.0
        U_trim[IDX_RUD] = 0.0
        U_trim[IDX_THR1] = U_trim[IDX_THR2]
    

    U_trim[IDX_AIL] += aileron_trim_step * roll_lt - aileron_trim_step * roll_rt
    U_trim[IDX_ELE] += pitch_trim_step * pitch_dn - pitch_trim_step * pitch_up
    #U_trim[IDX_RUD] = U_trim[IDX_RUD] + rudder_trim_step *  - rudder_trim_step * roll_lt  # no rudder trim buttons available
    U_trim[IDX_THR1] += throttle_trim_step * T1_fd - throttle_trim_step * T1_af
    U_trim[IDX_THR2] += throttle_trim_step * T2_fd - throttle_trim_step * T2_af

    # # # JOYSTICK COMMAND
    # joystick constants/multipliers to adjust correct movement and amplitude
    U[IDX_AIL] = U_trim[IDX_AIL] + joystick.get_axis(JOY_ROLL_AXIS) * joy_factors['aileron']
    U[IDX_ELE] = U_trim[IDX_ELE] + joystick.get_axis(JOY_PITCH_AXIS) * joy_factors['elevator']
    U[IDX_RUD] = U_trim[IDX_RUD] + joystick.get_axis(JOY_YAW_AXIS) * joy_factors['rudder']
    throttle_cmd = joystick.get_axis(3) * joy_factors['throttle_m'] + joy_factors['throttle_b'] # linearly map joystick inputs to RCAM
    U[IDX_THR1] = U_trim[IDX_THR1] + throttle_cmd
    U[IDX_THR2] = U_trim[IDX_THR2] + throttle_cmd
    U[IDX_BRAKE] = float(brake_applied)
    U[IDX_GNDSP] = float(toggle_gnd_spoiler)


    return U, U_trim, exit_signal