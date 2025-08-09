import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QScrollArea, QPushButton, QCheckBox, QComboBox, QHBoxLayout, QGridLayout, QLabel
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, QTimer
from PyQt5.QtGui import QIcon
import pyqtgraph as pg
from datetime import datetime
import scipy.signal as signal
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TabularViewSettings:
    def __init__(self, project_id):
        self.project_id = project_id
        self.bandpass_selection = "None"
        self.channel_name_visible = True
        self.unit_visible = True
        self.datetime_visible = True
        self.rpm_visible = True
        self.gap_visible = True
        self.direct_visible = True
        self.bandpass_visible = True
        self.one_xa_visible = True
        self.one_xp_visible = True
        self.two_xa_visible = True
        self.two_xp_visible = True
        self.nx_amp_visible = True
        self.nx_phase_visible = True
        self.updated_at = datetime.utcnow()

class TabularViewWorker(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    initialized = pyqtSignal(list, int, str, dict, str)

    def __init__(self, parent, project_name, model_name, db):
        super().__init__()
        self.parent = parent
        self.project_name = project_name
        self.model_name = model_name
        self.db = db

    def run(self):
        try:
            database = self.db.client.get_database("changed_db")
            projects_collection = database.get_collection("projects")
            project = projects_collection.find_one({"project_name": self.project_name, "email": self.db.email})
            if not project:
                self.error.emit(f"Project {self.project_name} not found for email {self.db.email}.")
                self.initialized.emit(["Channel 1"], 1, "", {}, None)
                return
            project_id = project["_id"]
            model = next((m for m in project["models"] if m["name"] == self.model_name), None)
            if not model or not model.get("channels"):
                self.error.emit(f"Model {self.model_name} or channels not found in project {self.project_name}.")
                self.initialized.emit(["Channel 1"], 1, "", {}, None)
                return
            channel_names = [c.get("channelName", f"Channel {i+1}") for i, c in enumerate(model["channels"])]
            num_channels = len(channel_names)
            if not channel_names:
                self.error.emit("No channels found in model.")
                self.initialized.emit(["Channel 1"], 1, "", {}, None)
                return
            channel_properties = {}
            for channel in model["channels"]:
                channel_name = channel.get("channelName", "Unknown")
                correction_value = float(channel.get("correctionValue", "1.0")) if channel.get("correctionValue") else 1.0
                gain = float(channel.get("gain", "1.0")) if channel.get("gain") else 1.0
                sensitivity = float(channel.get("sensitivity", "1.0")) if channel.get("sensitivity") and float(channel.get("sensitivity")) != 0 else 1.0
                unit = channel.get("unit", "mil").lower().strip()
                channel_properties[channel_name] = {
                    "Unit": unit,
                    "CorrectionValue": correction_value,
                    "Gain": gain,
                    "Sensitivity": sensitivity
                }
            tag_name = model.get("tagName", "")
            logging.debug(f"Worker initialized: {num_channels} channels, names: {channel_names}")
            self.initialized.emit(channel_names, num_channels, tag_name, channel_properties, project_id)
        except Exception as ex:
            self.error.emit(f"Error initializing TabularView: {str(ex)}")
            self.initialized.emit(["Channel 1"], 1, "", {}, None)
        finally:
            self.finished.emit()

class TabularViewFeature:
    def __init__(self, parent, db, project_name, channel=None, model_name=None, console=None):
        self.parent = parent
        self.db = db
        self.project_name = project_name
        self.model_name = model_name
        self.console = console
        self.widget = None
        self.data = None
        self.sample_rate = 4096
        self.num_channels = 1
        self.channel_names = ["Channel 1"]
        self.channel_properties = {}
        self.project_id = None
        self.selected_channel = 0  # Fixed to Channel 1
        self.raw_data = [np.zeros(4096)]
        self.low_pass_data = [np.zeros(4096)]
        self.high_pass_data = [np.zeros(4096)]
        self.band_pass_data = [np.zeros(4096)]
        self.time_points = np.arange(4096) / self.sample_rate
        self.band_pass_peak_to_peak_history = [[]]
        self.band_pass_peak_to_peak_times = [[]]
        self.average_frequency = [0.0]
        self.band_pass_peak_to_peak = [0.0]
        self.one_x_amps = [[]]
        self.one_x_phases = [[]]
        self.two_x_amps = [[]]
        self.two_x_phases = [[]]
        self.three_x_amps = [[]]
        self.three_x_phases = [[]]
        self.start_time = datetime.now()
        self.column_visibility = {
            "Channel Name": True, "Unit": True, "DateTime": True, "RPM": True, "Gap": True,
            "Direct": True, "Bandpass": True, "1xA": True, "1xP": True, "2xA": True,
            "2xP": True, "NXAmp": True, "NXPhase": True
        }
        self.bandpass_selection = "None"
        self.plot_initialized = False
        self.table = None
        self.plot_widgets = []
        self.plots = []
        self.tag_name = ""
        self.scroll_content = None
        self.scroll_layout = None
        self.mongo_client = self.db.client
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(1000)
        self.table_initialized = False
        self.data_buffer = []  # Buffer for incoming data
        self.last_update_time = datetime.now()
        self.update_interval = 0.5  # Update every 0.5 seconds
        self.initUI()
        self.initialize_thread()

    def initUI(self):
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        self.widget = QWidget()
        layout = QVBoxLayout()
        self.widget.setLayout(layout)

        top_layout = QHBoxLayout()
        top_layout.addStretch()
        self.settings_button = QPushButton("⚙️ Settings")
        self.settings_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        self.settings_button.clicked.connect(self.toggle_settings)
        top_layout.addWidget(self.settings_button)
        layout.addLayout(top_layout)

        self.settings_panel = QWidget()
        self.settings_panel.setStyleSheet("""
            QWidget {
                background-color: #f5f5f5;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 10px;
            }
        """)
        self.settings_panel.setVisible(False)
        settings_layout = QGridLayout()
        self.settings_panel.setLayout(settings_layout)

        bandpass_label = QLabel("Bandpass Selection:")
        bandpass_label.setStyleSheet("font-size: 14px;")
        settings_layout.addWidget(bandpass_label, 0, 0)
        self.bandpass_combo = QComboBox()
        self.bandpass_combo.addItems(["None", "50-200 Hz", "100-300 Hz"])
        self.bandpass_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                background-color: white;
                min-width: 100px;
            }
        """)
        settings_layout.addWidget(self.bandpass_combo, 0, 1)

        headers = ["Channel Name", "Unit", "DateTime", "RPM", "Gap", "Direct", "Bandpass", "1xA", "1xP", "2xA", "2xP", "NXAmp", "NXPhase"]
        self.checkbox_dict = {}
        for i, header in enumerate(headers):
            cb = QCheckBox(header)
            cb.setChecked(True)
            cb.setStyleSheet("font-size: 14px;")
            self.checkbox_dict[header] = cb
            settings_layout.addWidget(cb, (i // 3) + 1, i % 3)

        self.save_settings_button = QPushButton("Save")
        self.save_settings_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #1e88e5;
            }
            QPushButton:pressed {
                background-color: #1976d2;
            }
        """)
        self.save_settings_button.clicked.connect(self.save_settings)

        self.close_settings_button = QPushButton("Close")
        self.close_settings_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 14px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #e53935;
            }
            QPushButton:pressed {
                background-color: #d32f2f;
            }
        """)
        self.close_settings_button.clicked.connect(self.close_settings)

        settings_layout.addWidget(self.save_settings_button, len(headers) // 3 + 1, 0)
        settings_layout.addWidget(self.close_settings_button, len(headers) // 3 + 1, 1)
        settings_layout.addWidget(QLabel(""), len(headers) // 3 + 1, 2)
        layout.addWidget(self.settings_panel)

        self.table = QTableWidget()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setFixedHeight(200)
        layout.addWidget(self.table)

        self.table_initialized = True
        if self.console:
            self.console.append_to_console(f"Initialized table with {self.num_channels} rows for channels: {self.channel_names}")

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        scroll_area.setWidget(self.scroll_content)
        layout.addWidget(scroll_area)

        self.initialize_plots()
        if self.console:
            self.console.append_to_console(f"Initialized UI with channel: {self.channel_names[self.selected_channel]}")

    def initialize_thread(self):
        self.worker = TabularViewWorker(self, self.project_name, self.model_name, self.db)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.error.connect(self.log_and_set_status)
        self.worker.initialized.connect(self.complete_initialization)
        self.thread.start()

    def complete_initialization(self, channel_names, num_channels, tag_name, channel_properties, project_id):
        try:
            self.channel_names = channel_names
            self.num_channels = num_channels
            self.tag_name = tag_name
            self.channel_properties = channel_properties
            self.project_id = project_id
            self.table.setRowCount(self.num_channels)
            self.initialize_data_arrays()
            self.update_table_defaults()
            self.load_settings_from_database()
            self.initialize_plots()
            if self.console:
                self.console.append_to_console(f"Completed initialization: TagName: {self.tag_name}, Model: {self.model_name}, Channels: {self.num_channels}, Names: {self.channel_names}")
        except Exception as ex:
            self.log_and_set_status(f"Error completing initialization: {str(ex)}")
            self.channel_names = ["Channel 1"]
            self.num_channels = 1
            self.table.setRowCount(1)
            self.initialize_data_arrays()
            self.update_table_defaults()
            self.initialize_plots()

    def initialize_plots(self):
        for widget in self.plot_widgets:
            self.scroll_layout.removeWidget(widget)
            widget.deleteLater()
        self.plot_widgets = []
        self.plots = []

        plot_titles = [
            "Raw Data", "Low-Pass Filtered Data (20 Hz)", "High-Pass Filtered Data (200 Hz)",
            "Band-Pass Filtered Data (50-200 Hz)", "Bandpass Peak-to-Peak Over Time"
        ]
        for title in plot_titles:
            plot_widget = pg.PlotWidget(title=title)
            plot_widget.showGrid(x=True, y=True)
            plot_widget.setLabel('bottom', 'Time (s)' if title != "Bandpass Peak-to-Peak Over Time" else 'Time (s)')
            unit_label = self.get_unit_label()
            plot_widget.setLabel('left', f'Amplitude ({unit_label})' if title != "Bandpass Peak-to-Peak Over Time" else f'Peak-to-Peak Value ({unit_label})')
            plot_widget.setFixedHeight(250)
            self.scroll_layout.addWidget(plot_widget)
            self.plot_widgets.append(plot_widget)
            plot = plot_widget.plot(pen='b')
            self.plots.append(plot)
        self.scroll_layout.addStretch()
        self.plot_initialized = True
        self.update_plots()

    def get_unit_label(self):
        channel_name = self.channel_names[self.selected_channel] if self.selected_channel < len(self.channel_names) else "Channel 1"
        unit = self.channel_properties.get(channel_name, {"Unit": "mil"})["Unit"].lower()
        return unit

    def initialize_data_arrays(self):
        self.raw_data = [np.zeros(4096) for _ in range(self.num_channels)]
        self.low_pass_data = [np.zeros(4096) for _ in range(self.num_channels)]
        self.high_pass_data = [np.zeros(4096) for _ in range(self.num_channels)]
        self.band_pass_data = [np.zeros(4096) for _ in range(self.num_channels)]
        self.band_pass_peak_to_peak_history = [[] for _ in range(self.num_channels)]
        self.band_pass_peak_to_peak_times = [[] for _ in range(self.num_channels)]
        self.average_frequency = [0.0 for _ in range(self.num_channels)]
        self.band_pass_peak_to_peak = [0.0 for _ in range(self.num_channels)]
        self.one_x_amps = [[] for _ in range(self.num_channels)]
        self.one_x_phases = [[] for _ in range(self.num_channels)]
        self.two_x_amps = [[] for _ in range(self.num_channels)]
        self.two_x_phases = [[] for _ in range(self.num_channels)]
        self.three_x_amps = [[] for _ in range(self.num_channels)]
        self.three_x_phases = [[] for _ in range(self.num_channels)]
        self.time_points = np.arange(4096) / self.sample_rate
        if self.console:
            self.console.append_to_console(f"Initialized data arrays for {self.num_channels} channels: {self.channel_names}")

    def update_table_defaults(self):
        if not self.table or not self.table_initialized:
            self.log_and_set_status("Table not initialized, skipping update_table_defaults")
            return
        headers = ["Channel Name", "Unit", "DateTime", "RPM", "Gap", "Direct", "Bandpass", "1xA", "1xP", "2xA", "2xP", "NXAmp", "NXPhase"]
        self.table.setRowCount(self.num_channels)
        for row in range(self.num_channels):
            channel_name = self.channel_names[row] if row < len(self.channel_names) else f"Channel {row+1}"
            unit = self.channel_properties.get(channel_name, {"Unit": "mil"})["Unit"].lower()
            logging.debug(f"Setting unit for channel {channel_name}: {unit}")
            default_data = {
                "Channel Name": channel_name,
                "Unit": unit,
                "DateTime": datetime.now().strftime("%d-%b-%Y %I:%M:%S %p"),
                "RPM": "0.00", "Gap": "0.00", "Direct": "0.00", "Bandpass": "0.00",
                "1xA": "0.00", "1xP": "0.00", "2xA": "0.00", "2xP": "0.00",
                "NXAmp": "0.00", "NXPhase": "0.00"
            }
            for col, header in enumerate(headers):
                item = QTableWidgetItem(default_data[header])
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, col, item)
        self.table.setFixedHeight(200)
        self.update_column_visibility()
        if self.console:
            self.console.append_to_console(f"Updated table defaults with units for {self.num_channels} channels: {self.channel_names}")

    def load_settings_from_database(self):
        try:
            database = self.mongo_client.get_database("changed_db")
            settings_collection = database.get_collection("TabularViewSettings")
            setting = settings_collection.find_one({"projectId": self.project_id}, sort=[("updated_at", -1)])
            if setting:
                self.bandpass_selection = setting.get("bandpassSelection", "None")
                self.column_visibility = {
                    "Channel Name": setting.get("channelNameVisible", True),
                    "Unit": setting.get("unitVisible", True),
                    "DateTime": setting.get("datetimeVisible", True),
                    "RPM": setting.get("rpmVisible", True),
                    "Gap": setting.get("gapVisible", True),
                    "Direct": setting.get("directVisible", True),
                    "Bandpass": setting.get("bandpassVisible", True),
                    "1xA": setting.get("oneXaVisible", True),
                    "1xP": setting.get("oneXpVisible", True),
                    "2xA": setting.get("twoXaVisible", True),
                    "2xP": setting.get("twoXpVisible", True),
                    "NXAmp": setting.get("nxAmpVisible", True),
                    "NXPhase": setting.get("nxPhaseVisible", True)
                }
                for header, cb in self.checkbox_dict.items():
                    cb.setChecked(self.column_visibility.get(header, True))
                self.bandpass_combo.setCurrentText(self.bandpass_selection)
            self.update_column_visibility()
        except Exception as ex:
            self.log_and_set_status(f"Error loading settings: {str(ex)}")

    def save_settings(self):
        try:
            self.bandpass_selection = self.bandpass_combo.currentText()
            for header, cb in self.checkbox_dict.items():
                self.column_visibility[header] = cb.isChecked()
            settings = TabularViewSettings(self.project_id)
            settings.bandpass_selection = self.bandpass_selection
            settings.channel_name_visible = self.column_visibility["Channel Name"]
            settings.unit_visible = self.column_visibility["Unit"]
            settings.datetime_visible = self.column_visibility["DateTime"]
            settings.rpm_visible = self.column_visibility["RPM"]
            settings.gap_visible = self.column_visibility["Gap"]
            settings.direct_visible = self.column_visibility["Direct"]
            settings.bandpass_visible = self.column_visibility["Bandpass"]
            settings.one_xa_visible = self.column_visibility["1xA"]
            settings.one_xp_visible = self.column_visibility["1xP"]
            settings.two_xa_visible = self.column_visibility["2xA"]
            settings.two_xp_visible = self.column_visibility["2xP"]
            settings.nx_amp_visible = self.column_visibility["NXAmp"]
            settings.nx_phase_visible = self.column_visibility["NXPhase"]
            database = self.mongo_client.get_database("changed_db")
            settings_collection = database.get_collection("TabularViewSettings")
            settings_collection.insert_one({
                "projectId": self.project_id,
                "bandpassSelection": settings.bandpass_selection,
                "channelNameVisible": settings.channel_name_visible,
                "unitVisible": settings.unit_visible,
                "datetimeVisible": settings.datetime_visible,
                "rpmVisible": settings.rpm_visible,
                "gapVisible": settings.gap_visible,
                "directVisible": settings.direct_visible,
                "bandpassVisible": settings.bandpass_visible,
                "oneXaVisible": settings.one_xa_visible,
                "oneXpVisible": settings.one_xp_visible,
                "twoXaVisible": settings.two_xa_visible,
                "twoXpVisible": settings.two_xp_visible,
                "nxAmpVisible": settings.nx_amp_visible,
                "nxPhaseVisible": settings.nx_phase_visible,
                "updated_at": settings.updated_at
            })
            self.update_column_visibility()
            if self.console:
                self.console.append_to_console("Settings saved successfully")
        except Exception as ex:
            self.log_and_set_status(f"Error saving settings: {str(ex)}")

    def toggle_settings(self):
        self.settings_panel.setVisible(not self.settings_panel.isVisible())

    def close_settings(self):
        self.settings_panel.setVisible(False)

    def update_column_visibility(self):
        headers = ["Channel Name", "Unit", "DateTime", "RPM", "Gap", "Direct", "Bandpass", "1xA", "1xP", "2xA", "2xP", "NXAmp", "NXPhase"]
        for col, header in enumerate(headers):
            self.table.setColumnHidden(col, not self.column_visibility[header])

    def compute_harmonics(self, data, start_idx, segment_length, order):
        try:
            if segment_length <= 0 or start_idx >= len(data) or start_idx + segment_length > len(data):
                return 0.0, 0.0
            segment = data[start_idx:start_idx + segment_length]
            if len(segment) < 2:
                return 0.0, 0.0
            freqs = np.fft.fftfreq(len(segment), 1.0 / self.sample_rate)
            fft_vals = np.fft.fft(segment)
            pos_mask = freqs > 0
            freqs = freqs[pos_mask]
            fft_vals = fft_vals[pos_mask]
            if len(freqs) == 0 or self.average_frequency[self.selected_channel] == 0:
                return 0.0, 0.0
            fundamental_freq = self.average_frequency[self.selected_channel]
            target_freq = fundamental_freq * order
            idx = np.argmin(np.abs(freqs - target_freq))
            amplitude = np.abs(fft_vals[idx]) * 2 / len(segment)
            phase = np.angle(fft_vals[idx], deg=True)
            return amplitude, phase
        except Exception as ex:
            self.log_and_set_status(f"Error computing harmonics: {str(ex)}")
            return 0.0, 0.0

    def process_calibrated_data(self, values, ch):
        channel_name = self.channel_names[ch] if ch < len(self.channel_names) else f"Channel {ch+1}"
        props = self.channel_properties.get(channel_name, {
            "CorrectionValue": 1.0,
            "Gain": 1.0,
            "Sensitivity": 1.0,
            "Unit": "mil"
        })
        try:
            channel_data = np.array(values, dtype=float) * (3.3 / 65535.0) * (props["CorrectionValue"] * props["Gain"]) / props["Sensitivity"]
            unit = props["Unit"].lower()
            if unit == "mm":
                channel_data /= 25.4  # Convert from mil to mm
            elif unit == "um":
                channel_data *= 25.4 * 1000  # Convert from mil to um
            logging.debug(f"Processed data for {channel_name} with unit {unit}, shape: {channel_data.shape}")
            return channel_data
        except Exception as ex:
            self.log_and_set_status(f"Error processing calibrated data for {channel_name}: {str(ex)}")
            return np.zeros(4096)

    def format_direct_value(self, values, unit):
        if not values or len(values) == 0:
            return "0.00"
        avg = np.mean(values)
        unit = unit.lower()
        if unit == "mil":
            return f"{avg:.2f}"
        elif unit == "um":
            return f"{avg * 25.4 * 1000:.0f}"  # Convert mil to um
        elif unit == "mm":
            return f"{avg / 25.4:.3f}"  # Convert mil to mm
        logging.debug(f"Formatted direct value: {avg} in unit {unit}")
        return f"{avg:.2f}"

    def on_data_received(self, tag_name, model_name, values, sample_rate, frame_index):
        if not values or len(values) < 1:
            self.log_and_set_status(f"Insufficient data received for frame {frame_index}: {len(values)} channels")
            return
        self.data_buffer.append((values, sample_rate, frame_index))
        current_time = datetime.now()
        if (current_time - self.last_update_time).total_seconds() >= self.update_interval:
            self.process_buffered_data()
            self.last_update_time = current_time

    def process_buffered_data(self):
        if not self.data_buffer:
            return
        try:
            self.refresh_channel_properties()
            values, sample_rate, frame_index = self.data_buffer[-1]  # Process the latest data
            self.data_buffer = []  # Clear buffer after processing
            values = values[:self.num_channels] + [np.zeros(4096).tolist() for _ in range(self.num_channels - len(values))] if len(values) < self.num_channels else values[:self.num_channels]
            for i in range(len(values)):
                if len(values[i]) < 4096:
                    values[i] = list(np.pad(values[i], (0, 4096 - len(values[i])), 'constant'))[:4096]
                elif len(values[i]) > 4096:
                    values[i] = values[i][:4096]
            self.sample_rate = sample_rate if sample_rate > 0 else 4096
            self.data = values
            if self.console:
                self.console.append_to_console(f"Processing buffered data for frame {frame_index}, {len(values)} channels")

            channel_data_list = []
            for ch in range(self.num_channels):
                channel_name = self.channel_names[ch] if ch < len(self.channel_names) else f"Channel {ch+1}"
                props = self.channel_properties.get(channel_name, {"Unit": "mil"})
                unit = props["Unit"].lower()
                self.raw_data[ch] = self.process_calibrated_data(values[ch], ch)
                nyquist = self.sample_rate / 2.0
                tap_num = 31
                if self.bandpass_selection == "50-200 Hz":
                    band = [50 / nyquist, 200 / nyquist]
                elif self.bandpass_selection == "100-300 Hz":
                    band = [100 / nyquist, 300 / nyquist]
                else:
                    band = [50 / nyquist, 200 / nyquist]
                low_pass_coeffs = signal.firwin(tap_num, 20 / nyquist, window='hamming')
                high_pass_coeffs = signal.firwin(tap_num, 200 / nyquist, window='hamming', pass_zero=False)
                band_pass_coeffs = signal.firwin(tap_num, band, window='hamming', pass_zero=False)
                self.low_pass_data[ch] = signal.lfilter(low_pass_coeffs, 1.0, self.raw_data[ch])
                self.high_pass_data[ch] = signal.lfilter(high_pass_coeffs, 1.0, self.raw_data[ch])
                self.band_pass_data[ch] = signal.lfilter(band_pass_coeffs, 1.0, self.raw_data[ch])
                tacho_freq = 0.0
                if len(values) > self.num_channels:
                    tacho_data = values[self.num_channels]
                    if len(tacho_data) > 1:
                        peaks, _ = signal.find_peaks(tacho_data)
                        if len(peaks) > 1:
                            tacho_freq = self.sample_rate / np.mean(np.diff(peaks))
                self.average_frequency[ch] = tacho_freq
                band_pass_peak_to_peak = np.ptp(self.band_pass_data[ch]) if np.any(self.band_pass_data[ch]) else 0.0
                self.band_pass_peak_to_peak[ch] = band_pass_peak_to_peak
                self.band_pass_peak_to_peak_history[ch].append(band_pass_peak_to_peak)
                self.band_pass_peak_to_peak_times[ch].append((datetime.now() - self.start_time).total_seconds())
                if len(self.band_pass_peak_to_peak_history[ch]) > 50:
                    self.band_pass_peak_to_peak_history[ch] = self.band_pass_peak_to_peak_history[ch][-50:]
                    self.band_pass_peak_to_peak_times[ch] = self.band_pass_peak_to_peak_times[ch][-50:]
                amp1, phase1 = self.compute_harmonics(self.raw_data[ch], 0, len(self.raw_data[ch]), 1)
                amp2, phase2 = self.compute_harmonics(self.raw_data[ch], 0, len(self.raw_data[ch]), 2)
                amp3, phase3 = self.compute_harmonics(self.raw_data[ch], 0, len(self.raw_data[ch]), 3)
                self.one_x_amps[ch].append(amp1)
                self.one_x_phases[ch].append(phase1)
                self.two_x_amps[ch].append(amp2)
                self.two_x_phases[ch].append(phase2)
                self.three_x_amps[ch].append(amp3)
                self.three_x_phases[ch].append(phase3)
                if len(self.one_x_amps[ch]) > 50:
                    self.one_x_amps[ch] = self.one_x_amps[ch][-50:]
                    self.one_x_phases[ch] = self.one_x_phases[ch][-50:]
                    self.two_x_amps[ch] = self.two_x_amps[ch][-50:]
                    self.two_x_phases[ch] = self.two_x_phases[ch][-50:]
                    self.three_x_amps[ch] = self.three_x_amps[ch][-50:]
                    self.three_x_phases[ch] = self.three_x_phases[ch][-50:]
                direct_values = [np.ptp(self.raw_data[ch])] if np.any(self.raw_data[ch]) else []
                channel_data = {
                    "Channel Name": channel_name,
                    "Unit": unit,
                    "DateTime": datetime.now().strftime("%d-%b-%Y %I:%M:%S %p"),
                    "RPM": f"{self.average_frequency[ch] * 60.0:.2f}" if self.average_frequency[ch] > 0 else "0.00",
                    "Gap": "0.00",
                    "Direct": self.format_direct_value(direct_values, unit),
                    "Bandpass": self.format_direct_value([self.band_pass_peak_to_peak[ch]], unit),
                    "1xA": self.format_direct_value([np.mean(self.one_x_amps[ch])], unit) if self.one_x_amps[ch] else "0.00",
                    "1xP": f"{np.mean(self.one_x_phases[ch]):.2f}" if self.one_x_phases[ch] else "0.00",
                    "2xA": self.format_direct_value([np.mean(self.two_x_amps[ch])], unit) if self.two_x_amps[ch] else "0.00",
                    "2xP": f"{np.mean(self.two_x_phases[ch]):.2f}" if self.two_x_phases[ch] else "0.00",
                    "NXAmp": self.format_direct_value([np.mean(self.three_x_amps[ch])], unit) if self.three_x_amps[ch] else "0.00",
                    "NXPhase": f"{np.mean(self.three_x_phases[ch]):.2f}" if self.three_x_phases[ch] else "0.00"
                }
                channel_data_list.append(channel_data)
                self.update_table_row(ch, channel_data)
            QTimer.singleShot(0, self.update_plots)
            if self.console:
                self.console.append_to_console(f"Processed buffered data for frame {frame_index}, {self.num_channels} channels")
        except Exception as ex:
            self.log_and_set_status(f"Error processing buffered data for frame {frame_index}: {str(ex)}")

    def update_table_row(self, row, channel_data):
        if not self.table or not self.table_initialized:
            self.log_and_set_status("Table not initialized, skipping update_table_row")
            return
        headers = ["Channel Name", "Unit", "DateTime", "RPM", "Gap", "Direct", "Bandpass", "1xA", "1xP", "2xA", "2xP", "NXAmp", "NXPhase"]
        try:
            for col, header in enumerate(headers):
                item = QTableWidgetItem(channel_data[header])
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, col, item)
            logging.debug(f"Updated table row {row} with unit: {channel_data['Unit']}")
        except Exception as ex:
            self.log_and_set_status(f"Error updating table row {row}: {str(ex)}")

    def update_display(self):
        if not self.table or not self.table_initialized:
            self.log_and_set_status("Table not initialized, skipping update_display")
            self.initialize_plots()
            return
        try:
            self.process_buffered_data()  # Process any buffered data
            for ch in range(self.num_channels):
                channel_name = self.channel_names[ch] if ch < len(self.channel_names) else f"Channel {ch+1}"
                props = self.channel_properties.get(channel_name, {"Unit": "mil"})
                unit = props["Unit"].lower()
                direct_values = [np.ptp(self.raw_data[ch])] if np.any(self.raw_data[ch]) else []
                channel_data = {
                    "Channel Name": channel_name,
                    "Unit": unit,
                    "DateTime": datetime.now().strftime("%d-%b-%Y %I:%M:%S %p"),
                    "RPM": f"{self.average_frequency[ch] * 60.0:.2f}" if self.average_frequency[ch] > 0 else "0.00",
                    "Gap": "0.00",
                    "Direct": self.format_direct_value(direct_values, unit),
                    "Bandpass": self.format_direct_value([self.band_pass_peak_to_peak[ch]], unit),
                    "1xA": self.format_direct_value([np.mean(self.one_x_amps[ch])], unit) if self.one_x_amps[ch] else "0.00",
                    "1xP": f"{np.mean(self.one_x_phases[ch]):.2f}" if self.one_x_phases[ch] else "0.00",
                    "2xA": self.format_direct_value([np.mean(self.two_x_amps[ch])], unit) if self.two_x_amps[ch] else "0.00",
                    "2xP": f"{np.mean(self.two_x_phases[ch]):.2f}" if self.two_x_phases[ch] else "0.00",
                    "NXAmp": self.format_direct_value([np.mean(self.three_x_amps[ch])], unit) if self.three_x_amps[ch] else "0.00",
                    "NXPhase": f"{np.mean(self.three_x_phases[ch]):.2f}" if self.three_x_phases[ch] else "0.00"
                }
                self.update_table_row(ch, channel_data)
            QTimer.singleShot(0, self.update_plots)
            if self.console:
                self.console.append_to_console(f"Updated display for all {self.num_channels} channels")
        except Exception as ex:
            self.log_and_set_status(f"Error in update_display: {str(ex)}")

    def update_plots(self):
        if not self.plot_initialized or not self.plot_widgets or not self.plots:
            self.initialize_plots()
            return
        ch = self.selected_channel
        if ch >= self.num_channels:
            self.log_and_set_status("Selected channel not available, skipping plot update")
            return
        trim_samples = 47
        low_pass_trim = 0
        high_pass_trim = 110
        band_pass_trim = 110
        raw_trim = trim_samples
        if len(self.raw_data[ch]) <= trim_samples:
            raw_trim = low_pass_trim = high_pass_trim = band_pass_trim = 0
        channel_name = self.channel_names[ch] if ch < len(self.channel_names) else f"Channel {ch+1}"
        unit = self.channel_properties.get(channel_name, {"Unit": "mil"})["Unit"].lower()
        data_sets = [
            (self.raw_data[ch], raw_trim, "Raw Data"),
            (self.low_pass_data[ch], low_pass_trim, "Low-Pass Filtered Data (20 Hz)"),
            (self.high_pass_data[ch], high_pass_trim, "High-Pass Filtered Data (200 Hz)"),
            (self.band_pass_data[ch], band_pass_trim, "Band-Pass Filtered Data (50-200 Hz)")
        ]
        for i, (data, trim, title) in enumerate(data_sets):
            try:
                if len(data) <= trim:
                    data = np.array([0])
                    time_data = np.array([0])
                else:
                    data = data[trim:][:4096]  # Limit to 4096 samples for plotting
                    time_data = self.time_points[:len(data)]
                if i < len(self.plots):
                    self.plots[i].setData(time_data, data)
                    self.plot_widgets[i].setTitle(f"{title} (Channel: {self.channel_names[ch]}, Freq: {self.average_frequency[ch]:.2f} Hz, Unit: {unit})")
                    y_min = np.min(data) * 1.1 if data.size > 0 else -1.0
                    y_max = np.max(data) * 1.1 if data.size > 0 else 1.0
                    self.plot_widgets[i].setYRange(y_min, y_max, padding=0.1)
            except Exception as ex:
                self.log_and_set_status(f"Error updating plot {i}: {str(ex)}")
        try:
            if self.band_pass_peak_to_peak_times[ch] and self.band_pass_peak_to_peak_history[ch]:
                peak_data = np.array(self.band_pass_peak_to_peak_history[ch])
                if unit == "mm":
                    peak_data /= 25.4
                elif unit == "um":
                    peak_data *= 25.4 * 1000
                self.plots[4].setData(self.band_pass_peak_to_peak_times[ch], peak_data)
                y_max = max(0.01, np.max(peak_data) * 1.1) if peak_data.size > 0 else 0.01
                self.plot_widgets[4].setYRange(0, y_max, padding=0.1)
            else:
                self.plots[4].setData(np.array([0]), np.array([0]))
                self.plot_widgets[4].setYRange(0, 0.01, padding=0.1)
            self.plot_widgets[4].setTitle(f"Bandpass Peak-to-Peak Over Time (Channel: {self.channel_names[ch]}, Unit: {unit})")
        except Exception as ex:
            self.log_and_set_status(f"Error updating peak-to-peak plot: {str(ex)}")

    def refresh_channel_properties(self):
        try:
            project_data = self.db.get_project_data(self.project_name)
            if not project_data:
                self.log_and_set_status(f"Project {self.project_name} not found")
                return
            model = next((m for m in project_data["models"] if m["name"] == self.model_name), None)
            if not model:
                self.log_and_set_status(f"Model {self.model_name} not found")
                return
            self.channel_names = [c.get("channelName", f"Channel {i+1}") for i, c in enumerate(model["channels"])]
            self.num_channels = len(self.channel_names)
            self.channel_properties = {}
            for channel in model["channels"]:
                channel_name = channel.get("channelName", "Unknown")
                unit = channel.get("unit", "mil").lower().strip()
                self.channel_properties[channel_name] = {
                    "Unit": unit,
                    "CorrectionValue": float(channel.get("correctionValue", "1.0")) if channel.get("correctionValue") else 1.0,
                    "Gain": float(channel.get("gain", "1.0")) if channel.get("gain") else 1.0,
                    "Sensitivity": float(channel.get("sensitivity", "1.0")) if channel.get("sensitivity") else 1.0
                }
            self.table.setRowCount(self.num_channels)
            self.initialize_data_arrays()
            self.update_table_defaults()
            self.load_settings_from_database()
            if self.console:
                self.console.append_to_console(f"Refreshed channel properties: {self.num_channels} channels, Names: {self.channel_names}, Units: {[self.channel_properties[name]['Unit'] for name in self.channel_names]}")
        except Exception as ex:
            self.log_and_set_status(f"Error refreshing channel properties: {str(ex)}")

    def log_and_set_status(self, message):
        logging.error(message)
        if self.console:
            self.console.append_to_console(message)

    def close(self):
        self.timer.stop()
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        for widget in self.plot_widgets:
            self.scroll_layout.removeWidget(widget)
            widget.deleteLater()
        self.plot_widgets = []
        self.plots = []
        if self.table:
            self.table.deleteLater()
            self.table = None
            self.table_initialized = False
        if self.widget:
            self.widget.deleteLater()
            self.widget = None

    def get_widget(self):
        if not self.widget:
            self.log_and_set_status("Widget not initialized, recreating UI")
            self.initUI()
        return self.widget
