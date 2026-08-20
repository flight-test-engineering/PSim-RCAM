import sys
import socket
import struct
import threading
import queue
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QButtonGroup)
from PyQt5.QtCore import QTimer

# ---------------- CONFIGURATION ----------------
UDP_IP = "0.0.0.0" # Listen on all network interfaces
UDP_PORT = 5555
# Must match the order of variables sent from main.py. Index 0 is assumed to be Time.
PARAM_NAMES = ["Time", "Airspeed (kts)", "Altitude (ft)", "Pitch (deg)", "Alpha (deg)", "Gamma (deg)", "A/G (-)"]
MAX_HISTORY_SEC = 60
SIM_HZ = 60
BUFFER_SIZE = MAX_HISTORY_SEC * SIM_HZ * 2
# -----------------------------------------------

class TelemetryReceiver(threading.Thread):
    def __init__(self, data_queue):
        super().__init__(daemon=True)
        self.q = data_queue
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.running = True

    def run(self):
        print(f"Listening for telemetry on UDP {UDP_PORT}...")
        while self.running:
            try:
                data, _ = self.sock.recvfrom(1024)
                num_doubles = len(data) // 8
                if num_doubles > 0:
                    unpacked = struct.unpack(f'<{num_doubles}d', data)
                    self.q.put(unpacked)
            except Exception as e:
                print(f"UDP Error: {e}")

class RealTimePlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PSim Real-Time Telemetry")
        self.resize(1200, 800)

        # Buffer setup
        self.num_vars = len(PARAM_NAMES)
        self.data_buffer = np.zeros((self.num_vars, BUFFER_SIZE))
        self.ptr = 0 # Keeps track of how many items we've received
        self.current_time_window = 10.0 # Default to 10 seconds

        # Data Queue from UDP thread
        self.data_queue = queue.Queue()

        self.init_ui()

        # Start UDP Thread
        self.receiver_thread = TelemetryReceiver(self.data_queue)
        self.receiver_thread.start()

        # GUI Update Timer (Updates screen at ~30 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(30)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # --- TOP ROW BUTTONS ---
        btn_layout = QHBoxLayout()
        
        # 1. Parameter Toggle Buttons
        self.param_buttons = []
        for i in range(1, self.num_vars): # Skip 'Time' (index 0)
            btn = QPushButton(PARAM_NAMES[i])
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.clicked.connect(self.update_layout)
            btn_layout.addWidget(btn)
            self.param_buttons.append(btn)

        btn_layout.addStretch() # Spacer

        # 2. Time Window Buttons
        time_group = QButtonGroup(self)
        for t_val in [5, 10, 60]:
            btn = QPushButton(f"{t_val}s")
            btn.setCheckable(True)
            time_group.addButton(btn)
            btn_layout.addWidget(btn)
            btn.clicked.connect(lambda checked, t=t_val: self.set_time_window(t))
            if t_val == 10:
                btn.setChecked(True)

        main_layout.addLayout(btn_layout)

        # --- PLOT AREA ---
        # GraphicsLayoutWidget handles tightly stacked plots perfectly
        self.plot_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.plot_widget)

        self.plots = []
        self.curves = []

        for i in range(1, self.num_vars):
            p = self.plot_widget.addPlot(row=i-1, col=0)

            # --- Move Y-Axis to the Right ---
            p.hideAxis('left')                 # Hide the default left axis
            p.showAxis('right')                # Enable the right axis
            p.setLabel('right', PARAM_NAMES[i]) # Set the text label on the right
            # --------------------------------

            p.showGrid(x=True, y=True, alpha=0.3)
            
            # Remove margin/spacing to make them tightly stacked
            p.layout.setContentsMargins(0, 0, 0, 0)
            
            # Create the curve line
            curve = p.plot(pen=pg.mkPen(color=(0, 200, 255), width=2))
            
            self.plots.append(p)
            self.curves.append(curve)

        self.plot_widget.ci.layout.setSpacing(0) # Zero spacing between stacked plots
        self.update_layout() # Setup initial visibility

    def set_time_window(self, seconds):
        self.current_time_window = seconds

    def update_layout(self):
        """Hides or shows plots based on button toggles and fixes the X-axes."""
        visible_plots = []
        for i, btn in enumerate(self.param_buttons):
            if btn.isChecked():
                self.plots[i].show()
                visible_plots.append(self.plots[i])
            else:
                self.plots[i].hide()

        # Only the bottom-most visible plot should show the X-axis (Time)
        for p in self.plots:
            p.hideAxis('bottom')
        if visible_plots:
            visible_plots[-1].showAxis('bottom')
            visible_plots[-1].setLabel('bottom', 'Time (s)')

    def update_plots(self):
        """Drains the UDP queue, updates numpy buffers, and updates screen."""
        new_data_list = []
        # Drain the queue to get all packets received since last GUI tick
        while not self.data_queue.empty():
            new_data_list.append(self.data_queue.get())

        if not new_data_list:
            return

        # Convert list of tuples to a numpy array for fast column extraction
        new_data_arr = np.array(new_data_list).T # Transpose so rows are variables, cols are time steps
        n_new = new_data_arr.shape[1]

        # Shift the buffer and append new data (Rolling buffer)
        self.data_buffer = np.roll(self.data_buffer, -n_new, axis=1)
        self.data_buffer[:, -n_new:] = new_data_arr
        self.ptr += n_new

        # Only plot the valid data we've received so far (prevents drawing zeros at startup)
        valid_points = min(self.ptr, BUFFER_SIZE)
        
        # Get the X-axis (Time) valid slice
        time_data = self.data_buffer[0, -valid_points:]
        current_time = time_data[-1]

        for i in range(1, self.num_vars):
            if self.param_buttons[i-1].isChecked():
                # Set X and Y data for this curve
                y_data = self.data_buffer[i, -valid_points:]
                self.curves[i-1].setData(time_data, y_data)
                
                # Lock the X-Axis to the selected time window (e.g., last 10 seconds)
                self.plots[i-1].setXRange(current_time - self.current_time_window, current_time, padding=0)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True) # Makes the lines look smooth
    window = RealTimePlotter()
    window.show()
    sys.exit(app.exec_())