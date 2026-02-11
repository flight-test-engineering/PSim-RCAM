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

def get_joy_inputs(joystick: pygame.joystick, joy_events: pygame.event, trim_point: np.ndarray, fr:float,
                  trim_params: np.ndarray, joy_factors: np.ndarray, acp:jitclass):
    '''
    function that will read joystick positions and adjust controls:
    1. joy will change controls on top of trim point
    2. trim settings (buttons) will change trim point
    3. engine does not have trim function, but depending on
    button pressed, throttle should be commanded left/right/both
    '''
    inceptor_cmd = np.zeros(trim_point.shape)

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
                if trim_point[IDX_GNDSP] > 0.5: 
                    trim_point[IDX_GNDSP] = 0
                else:
                    trim_point[IDX_GNDSP] = 1
            if event.button == JOY_LDG_CYCLE:
                if trim_point[IDX_GEAR] > 0.5: 
                    trim_point[IDX_GEAR] = 0
                else:
                    trim_point[IDX_GEAR] = 1
            if event.button == JOY_E1_CYCLE_CUT:
                if trim_point[IDX_THR1] > -0.5: 
                    trim_point[IDX_THR1] = -1
                    # because we have only one throttle lever,
                    # when we cut the engine, we need to zero out the trim on E2
                    trim_point[IDX_THR2] = 0
                else:
                    trim_point[IDX_THR1] = 0
    exit_signal = joystick.get_button(JOY_EXIT_SIGNAL)
    brake_applied = joystick.get_button(JOY_BRAKE)


    # if trigger is pressed, then zero out aileron, rudder states and make thrust equal on both sides
    if zero_ail_rud_thr == 1:
        trim_point[IDX_AIL] = 0.0
        trim_point[IDX_RUD] = 0.0
        trim_point[IDX_THR2] = trim_point[IDX_THR1]
    

    trim_point[IDX_ELE] += pitch_trim_step * pitch_dn - pitch_trim_step * pitch_up
    trim_point[IDX_THR1] += throttle_trim_step * T1_fd - throttle_trim_step * T1_af


    # # # JOYSTICK COMMAND
    # joystick constants/multipliers to adjust correct movement and amplitude
    inceptor_cmd[IDX_AIL] = trim_point[IDX_AIL] + joystick.get_axis(JOY_ROLL_AXIS) * joy_factors['aileron']
    inceptor_cmd[IDX_ELE] = trim_point[IDX_ELE] + joystick.get_axis(JOY_PITCH_AXIS) * joy_factors['elevator']
    inceptor_cmd[IDX_RUD] = trim_point[IDX_RUD] + joystick.get_axis(JOY_YAW_AXIS) * joy_factors['rudder']
    throttle_cmd = joystick.get_axis(3) * joy_factors['throttle_m'] + joy_factors['throttle_b'] # linearly map joystick inputs to RCAM
    inceptor_cmd[IDX_THR1] = trim_point[IDX_THR1] + throttle_cmd
    inceptor_cmd[IDX_THR2] = trim_point[IDX_THR2] + throttle_cmd
    inceptor_cmd[IDX_FLAP] = max(0, min(acp.MAX_FLAP, trim_point[IDX_FLAP] + flap_cmd_dn - flap_cmd_up))
    trim_point[IDX_FLAP] = inceptor_cmd[IDX_FLAP]
    inceptor_cmd[IDX_BRAKE] = float(brake_applied)
    inceptor_cmd[IDX_GEAR] = trim_point[IDX_GEAR] # gear command is lever state



    return inceptor_cmd, trim_point, exit_signal