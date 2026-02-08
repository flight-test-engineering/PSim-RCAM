import numpy as np
from numba.experimental import jitclass
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

def get_joy_inputs(joystick, joy_events, U_trim, fr, trim_params, joy_factors, acp):
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
    E1_cycle_cut = joystick.get_button(JOY_E1_CYCLE_CUT)
    T1_fd = joystick.get_button(JOY_E1_THR_TRIM_FWD)
    T1_af = joystick.get_button(JOY_E1_THR_TRIM_AFT)

    flap_cmd_dn = 0
    flap_cmd_up = 0

    for event in joy_events:
        if event.type == pygame.JOYBUTTONDOWN:
            if event.button == JOY_FLAP_CMD_UP:
                flap_cmd_up = 1
            if event.button == JOY_FLAP_CMD_DN:
                flap_cmd_dn = 1
            if event.button == JOY_ARM_DIS_GND_SPOILER:
                if U_trim[IDX_GNDSP] > 0.5: 
                    U_trim[IDX_GNDSP] = 0
                else:
                    U_trim[IDX_GNDSP] = 1
            if event.button == JOY_LDG_CYCLE:
                if U_trim[IDX_GEAR] > 0.5: 
                    U_trim[IDX_GEAR] = 0
                else:
                    U_trim[IDX_GEAR] = 1
            if event.button == JOY_E1_CYCLE_CUT:
                if U_trim[IDX_THR1] > -0.5: 
                    U_trim[IDX_THR1] = -1
                    # because we have only one throttle lever,
                    # when we cut the engine, we need to zero out the trim on E2
                    U_trim[IDX_THR2] = 0
                else:
                    U_trim[IDX_THR1] = 0
    exit_signal = joystick.get_button(JOY_EXIT_SIGNAL)
    brake_applied = joystick.get_button(JOY_BRAKE)


    # if trigger is pressed, then zero out aileron, rudder states and make thrust equal on both sides
    if zero_ail_rud_thr == 1:
        U_trim[IDX_AIL] = 0.0
        U_trim[IDX_RUD] = 0.0
        U_trim[IDX_THR2] = U_trim[IDX_THR1]
    

    U_trim[IDX_ELE] += pitch_trim_step * pitch_dn - pitch_trim_step * pitch_up
    U_trim[IDX_THR1] += throttle_trim_step * T1_fd - throttle_trim_step * T1_af


    # # # JOYSTICK COMMAND
    # joystick constants/multipliers to adjust correct movement and amplitude
    U[IDX_AIL] = U_trim[IDX_AIL] + joystick.get_axis(JOY_ROLL_AXIS) * joy_factors['aileron']
    U[IDX_ELE] = U_trim[IDX_ELE] + joystick.get_axis(JOY_PITCH_AXIS) * joy_factors['elevator']
    U[IDX_RUD] = U_trim[IDX_RUD] + joystick.get_axis(JOY_YAW_AXIS) * joy_factors['rudder']
    throttle_cmd = joystick.get_axis(3) * joy_factors['throttle_m'] + joy_factors['throttle_b'] # linearly map joystick inputs to RCAM
    U[IDX_THR1] = U_trim[IDX_THR1] + throttle_cmd
    U[IDX_THR2] = U_trim[IDX_THR2] + throttle_cmd
    U[IDX_FLAP] = max(0, min(acp.MAX_FLAP, U_trim[IDX_FLAP] + flap_cmd_dn - flap_cmd_up))
    U_trim[IDX_FLAP] = U[IDX_FLAP]
    U[IDX_BRAKE] = float(brake_applied)
    U[IDX_GEAR] = U_trim[IDX_GEAR] # gear command is lever state



    return U, U_trim, exit_signal