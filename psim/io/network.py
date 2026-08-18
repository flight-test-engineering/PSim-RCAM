import socket
import queue
import time
import threading
import numpy as np
import struct
import logging

import ISA_module as ISA
from psim.constants import *
import psim.environment as env
from psim.helpers import logger





# Threads for communication with FlightGear
def network_worker(socks:list[socket.socket], packet_queue:queue.Queue, fg_addresses:list[str]):
    '''
    This function runs in a separate thread. It waits for FDM packets to appear
    in the queue and sends them over UDP.
    Inputs:
        socks: list with network open socks
        packet_queue: a Python multithread queue that received the packets to be sent
        fg_addresses: list of tuples with IP address and port
    '''
    logger.info('[network_worker] Starting FlightGear output network thread')
    while True:
        try:
            # Block until a packet is available in the queue.
            # A timeout is added to allow for graceful shutdown checks if needed,
            # though using a sentinel value is cleaner.
            packet = packet_queue.get()

            # THREADING: Check for the sentinel value (None) to signal shutdown.
            if packet is None:
                logger.info('[network_worker]Network thread received shutdown signal.')
                break

            # Send the packet to FlightGear.
            for idx, s in enumerate(socks):
                s.sendto(packet, fg_addresses[idx])

        except queue.Empty:
            # This will only happen if a timeout is used in get()
            continue
        except Exception as e:
            logger.error(f'[network_worker] Error in network thread: {e}')
            break
    logger.info("[network_worker] Network thread finished.")


def terrain_udp_worker(ip, port, shared_data, shutdown_queue):
    '''
    Listens for UDP packets from FlightGear containing ground elevation.
    Updates shared_data['ground_alt'] with the latest received value.
    '''
    logger.info(f"[terrain_udp_worker] Starting Terrain RX Worker on {ip}:{port}...")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Bind to the interface/port we expect FG to send TO
    try:
        sock.bind((ip, port))
        
        # Set a timeout so the loop can check the shutdown queue periodically
        sock.settimeout(0.5) 
        sock.setblocking(0) # Non-blocking mode
        
        logger.info("[terrain_udp_worker] Listening")
        
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
                        # this is not an error...
                        break
                
                if data:
                    decoded_str = data.decode('utf-8').strip()
                    if decoded_str:
                        val_ft = float(decoded_str)
                        shared_data['ground_alt'] = val_ft * FT2M
                        
                time.sleep(TERRAIN_POLL_INTERVAL_S) # Slight rest to prevent CPU hogging
                
            except socket.timeout:
                continue  # Expected, keep going
            except (ConnectionError, OSError) as e:
                logger.error(f"[terrain_udp_worker] Fatal network error: {e}")
                break
            except Exception as e:
                logger.error(f"[terrain_udp_worker] Unexpected error: {e}")
                continue
    finally:
        sock.close()
        logger.info("[terrain_udp_worker] Terrain RX Worker finished.")






def set_FDM(this_fgFDM, X:np.ndarray, U_norm:np.ndarray, latlon:np.ndarray, alt:float, body_accels:np.ndarray, agl_m:float=0.0)->None:
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


# logger = logging.getLogger(__name__)

class TelemetrySender(threading.Thread):
    def __init__(self, ip="127.0.0.1", port=5555):
        super().__init__(daemon=True) # Daemon thread exits automatically when main.py closes
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.q = queue.Queue(maxsize=1)
        self.running = True

    def run(self):
        logger.info(f"[Telemetry Sender] Telemetry Sender thread started on {self.ip}:{self.port}")
        while self.running:
            try:
                # Wait for data. Timeout allows the thread to check self.running gracefully
                data = self.q.get(timeout=0.1) 
                
                # Pack the list of floats into a fast binary C-struct.
                # '<' means little-endian, 'd' means double precision float (8 bytes each).
                packed_data = struct.pack('<' + 'd' * len(data), *data)
                
                self.sock.sendto(packed_data, (self.ip, self.port))
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"[Telemetry Sender]  sender error: {e}")
                logger.debug(data)

    def send_data(self, data_list):
        """Called by the main simulation loop to push new data."""
        try:
            # Try to push new data immediately
            self.q.put_nowait(data_list)
        except queue.Full:
            # If the queue is full, discard the old frame and put the new one
            try:
                self.q.get_nowait()
                self.q.put_nowait(data_list)
            except queue.Empty:
                pass