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





def set_FDM(this_fgFDM, X, U_norm, latlon, alt, body_accels):
    '''
    function to set the current time step data to be sent to FlightGear
    inputs are:
    X - states
    U - controls
    latlon - in radians
    alt - in meters
    NED - velocities in m/s
    '''
    this_fgFDM.set('phi', X[IDX_PHI])
    this_fgFDM.set('theta', X[IDX_THETA])
    this_fgFDM.set('psi', X[IDX_PSI])

    this_fgFDM.set('phidot', X[IDX_P])
    this_fgFDM.set('thetadot', X[IDX_Q])
    this_fgFDM.set('psidot', X[IDX_R])
    
    # this sets units to kts because the HUD does not apply any conversions to the speed
    # if we send speed in fps as the API requires, the HUD displays wrong value
    this_fgFDM.set('vcas', ISA.Vt2Vc(env.VA(X[:3]), alt*M2FT) * MS2KT) 
    this_fgFDM.set('cur_time', int(time.perf_counter() ), units='seconds')
    this_fgFDM.set('latitude', latlon[0], units='radians')
    this_fgFDM.set('longitude', latlon[1], units='radians')
    this_fgFDM.set('altitude', alt, units='meters')

    this_fgFDM.set('left_aileron', -U_norm[IDX_AIL])
    this_fgFDM.set('right_aileron', +U_norm[IDX_AIL])
    this_fgFDM.set('elevator', U_norm[IDX_ELE])
    this_fgFDM.set('rudder', -U_norm[IDX_RUD])
    this_fgFDM.set('left_flap', U_norm[IDX_FLAP])
    this_fgFDM.set('right_flap', U_norm[IDX_FLAP])
    this_fgFDM.set('spoilers', U_norm[IDX_GNDSP])
    this_fgFDM.set('gear_pos', U_norm[IDX_GEAR], idx=0)
    this_fgFDM.set('gear_pos', U_norm[IDX_GEAR], idx=1)
    this_fgFDM.set('gear_pos', U_norm[IDX_GEAR], idx=2)


    this_fgFDM.set('A_X_pilot', body_accels[0], units='mpss')
    this_fgFDM.set('A_Y_pilot', body_accels[1], units='mpss')
    this_fgFDM.set('A_Z_pilot', body_accels[2], units='mpss')









