import socket
import queue
import time
import threading
import numpy as np

import ISA_module as ISA
from psim.constants import *
import psim.environment as env





# Threads for communication with FlightGear
def network_worker(socks, packet_queue, fg_addresses):
    """
    This function runs in a separate thread. It waits for FDM packets to appear
    in the queue and sends them over UDP.
    Inputs:
        socks: list with network open socks
        packet_queue: a Python multithread queue that received the packets to be sent
        fg_addresses: list of tuples with IP address and port
    """
    print("Starting FlightGear output network thread", end="")
    while True:
        try:
            # Block until a packet is available in the queue.
            # A timeout is added to allow for graceful shutdown checks if needed,
            # though using a sentinel value is cleaner.
            packet = packet_queue.get()

            # THREADING: Check for the sentinel value (None) to signal shutdown.
            if packet is None:
                print("Network thread received shutdown signal.")
                break

            # Send the packet to FlightGear.
            for idx, s in enumerate(socks):
                s.sendto(packet, fg_addresses[idx])

        except queue.Empty:
            # This will only happen if a timeout is used in get()
            continue
        except Exception as e:
            print(f"Error in network thread: {e}")
            break
    print("Network thread finished.")


def terrain_udp_worker(ip, port, shared_data, shutdown_queue):
    """
    Listens for UDP packets from FlightGear containing ground elevation.
    Updates shared_data['ground_alt'] with the latest received value.
    """
    print(f"Starting Terrain RX Worker on {ip}:{port}...", end="")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Bind to the interface/port we expect FG to send TO
    sock.bind((ip, port))
    
    # Set a timeout so the loop can check the shutdown queue periodically
    sock.settimeout(0.5) 
    sock.setblocking(0) # Non-blocking mode
    
    print(" Listening.")
    
    while True:
        # Check for kill signal
        if not shutdown_queue.empty():
            break
            
        try:
            # Loop to drain buffer and get the absolutely latest packet
            data = None
            while True:
                try:
                    chunk, _ = sock.recvfrom(1024)
                    data = chunk
                except BlockingIOError:
                    # Buffer empty, we have the latest 'data' (if any)
                    break
            
            if data:
                decoded_str = data.decode('utf-8').strip()
                if decoded_str:
                    val_ft = float(decoded_str)
                    shared_data['ground_alt'] = val_ft * FT2M
                    
            time.sleep(0.01) # Slight rest to prevent CPU hogging
            
        except Exception:
            pass
            
    sock.close()
    print("Terrain RX Worker finished.")






def set_FDM(this_fgFDM, X, U_norm, latlon, alt, body_accels, agl_m=0.0):
    '''
    function to set the current time step data to be sent to FlightGear
    inputs are:
    this_fgFDM - network comms object
    X - states
    U_norm - normalized controls
    latlon - in radians
    alt - in meters
    body_accels - m/s2
    agl_m -  in m
    '''
    # Unpack State
    phi, theta, psi = X[IDX_PHI], X[IDX_THETA], X[IDX_PSI]
    p, q, r = X[IDX_P], X[IDX_Q], X[IDX_R]
    uvw = X[IDX_U : IDX_W+1]
    
    # --- SPEEDS ---
    # FlightGear needs body velocities in Feet Per Second (FPS) for instruments
    this_fgFDM.set('u_body', uvw[0] * M2FT)
    this_fgFDM.set('v_body', uvw[1] * M2FT)
    this_fgFDM.set('w_body', uvw[2] * M2FT)
    
    # Vcas for HUD
    Va = env.VA(uvw)
    this_fgFDM.set('vcas', ISA.Vt2Vc(Va, alt*M2FT) * MS2KT)
    
    # --- ATTITUDE & RATES ---
    this_fgFDM.set('phi', phi)
    this_fgFDM.set('theta', theta)
    this_fgFDM.set('psi', psi)
    this_fgFDM.set('phidot', p)
    this_fgFDM.set('thetadot', q)
    this_fgFDM.set('psidot', r)
    
    # --- POSITION ---
    this_fgFDM.set('latitude', latlon[0])   # radians
    this_fgFDM.set('longitude', latlon[1])  # radians
    this_fgFDM.set('altitude', alt)         # meters
    this_fgFDM.set('agl', agl_m)            # meters 
    
    # --- CONTROLS ---
    this_fgFDM.set('left_aileron', -U_norm[IDX_AIL])
    this_fgFDM.set('right_aileron', +U_norm[IDX_AIL])
    this_fgFDM.set('elevator', U_norm[IDX_ELE])
    this_fgFDM.set('rudder', -U_norm[IDX_RUD])
    
    # Check if these indices exist in your constants.py; if not, remove or define them
    
    this_fgFDM.set('left_flap', U_norm[IDX_FLAP])
    this_fgFDM.set('right_flap', U_norm[IDX_FLAP])
        
    this_fgFDM.set('spoilers', U_norm[IDX_GNDSP])


    # Gear (Assuming IDX_GEAR exists, otherwise default to 1.0 down)
    gear_val = U_norm[IDX_GEAR]
    gear_compression = max(0, 10 - agl_m) * gear_val / abs(10 - agl_m) / 10
    this_fgFDM.set('gear_pos', gear_val, idx=0)
    this_fgFDM.set('gear_pos', gear_val, idx=1)
    this_fgFDM.set('gear_pos', gear_val, idx=2)
    this_fgFDM.set('gear_comp', gear_compression, idx=0)
    this_fgFDM.set('gear_comp', gear_compression, idx=1)
    this_fgFDM.set('gear_comp', gear_compression, idx=2)

    # --- ACCELS ---
    this_fgFDM.set('A_X_pilot', body_accels[0])
    this_fgFDM.set('A_Y_pilot', body_accels[1])
    this_fgFDM.set('A_Z_pilot', body_accels[2])
    
    # --- TIME ---
    # Use time.time() for Unix Epoch, likely preferred by FG over perf_counter
    this_fgFDM.set('cur_time', int(time.time()))