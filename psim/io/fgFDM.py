import struct
import time


class fgFDM:
    """
    FlightGear NetFDM Protocol (v24) implementation.
    """
    def __init__(self):
        self.values = {}
        
        # ---------------------------------------------------------
        # Build the Packet Format String (Big Endian '>')
        # ---------------------------------------------------------
        self.pack_str = (
            '> '            # Big Endian
            'I I '          # Header (Version, Padding)
            'd d d '        # Position (Lon, Lat, Alt)
            'f '            # AGL (Above Ground Level)
            'f f f '        # Attitude (Phi, Theta, Psi)
            'f f '          # Aero (Alpha, Beta)
            'f f f '        # Rates (PhiDot, ThetaDot, PsiDot)
            'f f '          # Velocities (Vcas, ClimbRate)
            'f f f '        # Velocities NED (N, E, D)
            'f f f '        # Velocities Body (U, V, W)
            'f f f '        # Accels (Ax, Ay, Az)
            'f f '          # Stall, Slip
            
            # --- Engines (45 items) ---
            'I '            # Num Engines
            'I I I I '      # Engine State [4]
            'f f f f '      # RPM [4]
            'f f f f '      # Fuel Flow [4]
            'f f f f '      # Fuel PX [4]
            'f f f f '      # EGT [4]
            'f f f f '      # CHT [4]
            'f f f f '      # MP [4]
            'f f f f '      # TIT [4]
            'f f f f '      # Oil Temp [4]
            'f f f f '      # Oil PX [4]
            
            # --- Consumables (5 items) ---
            'I '            # Num Tanks
            'f f f f '      # Fuel Qty [4]
            
            # --- Gear (13 items) ---
            'I '            # Num Wheels
            'I I I '        # WoW [3]
            'f f f '        # Gear Pos [3]
            'f f f '        # Gear Steer [3]
            'f f f '        # Gear Comp [3]
            
            # --- Environment (3 items) ---
            'I i f '        # CurTime, Warp, Visibility
            
            # --- Controls (10 items) ---
            'f f f f f f f f f f' # Elev, Trim, Flaps(2), Ail(2), Rud, Nose, Brk, Spoil
        )

    # --- UPDATED SET METHOD ---
    def set(self, name, value, idx=None):
        """
        Set a value. 
        idx: Appends suffix to name (e.g. gear_pos + 0 -> gear_pos_0)
        units: Ignored (compatibility)
        """
        if idx is not None:
            key = f"{name}_{idx}"
        else:
            key = name
        self.values[key] = value

    def get(self, name, default=0):
        return self.values.get(name, default)

    def pack(self):
        """
        Populate the list of 99 arguments and pack into binary.
        """
        cur_time = int(self.get('cur_time', time.time()))
        
        args = [
            # 1. Header (2)
            24, 0,
            
            # 2. Position (3)
            self.get('longitude'), self.get('latitude'), self.get('altitude'),
            
            # 2b. AGL (1)
            self.get('agl', 0),
            
            # 3. Attitude (3)
            self.get('phi'), self.get('theta'), self.get('psi'),
            
            # 4. Aero (2)
            self.get('alpha'), self.get('beta'),
            
            # 5. Rates (3)
            self.get('phidot'), self.get('thetadot'), self.get('psidot'),
            
            # 6. Velocities Nav (2)
            self.get('vcas'), self.get('climb_rate'),
            
            # 7. Velocities NED (3)
            self.get('v_north'), self.get('v_east'), self.get('v_down'),
            
            # 8. Velocities Body (3)
            self.get('u_body'), self.get('v_body'), self.get('w_body'),
            
            # 9. Accels (3)
            self.get('A_X_pilot'), self.get('A_Y_pilot'), self.get('A_Z_pilot'),
            
            # 10. Stall/Slip (2)
            self.get('stall_warning'), self.get('slip_deg'),
            
            # --- ENGINES (45) ---
            self.get('num_engines', 2),
            self.get('eng_state_0', 2), self.get('eng_state_1', 2), 0, 0,
            self.get('rpm_0'), self.get('rpm_1'), 0, 0,
            # Unused
            0,0,0,0,  0,0,0,0,  0,0,0,0,  0,0,0,0,
            0,0,0,0,  0,0,0,0,  0,0,0,0,  0,0,0,0,
            
            # --- TANKS (5) ---
            self.get('num_tanks', 1),
            self.get('fuel_qty_0'), 0, 0, 0,
            
            # --- GEAR (13) ---
            self.get('num_wheels', 3),
            # WoW
            self.get('wow_0'), self.get('wow_1'), self.get('wow_2'),
            # Position (0=Up, 1=Down)
            # The 'set' method with idx=0 maps to 'gear_pos_0', which matches these calls:
            self.get('gear_pos_0', 1.0), self.get('gear_pos_1', 1.0), self.get('gear_pos_2', 1.0),
            # Steer
            self.get('gear_steer_0'), 0, 0,
            # Compression
            self.get('gear_comp_0'), self.get('gear_comp_1'), self.get('gear_comp_2'),
            
            # --- ENV (3) ---
            cur_time,
            self.get('warp', 0),
            self.get('visibility', 5000.0),
            
            # --- CONTROLS (10) ---
            self.get('elevator'),
            self.get('elevator_trim_tab'),
            self.get('left_flap'), 
            self.get('right_flap'),
            self.get('left_aileron'), 
            self.get('right_aileron'),
            self.get('rudder'),
            self.get('nose_wheel'),
            self.get('speedbrake'),
            self.get('spoilers')
        ]
        
        return struct.pack(self.pack_str, *args)