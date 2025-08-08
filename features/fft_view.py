from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QGridLayout, QComboBox
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg
import numpy as np
import logging
from scipy.fft import fft
from scipy.signal import get_window
from datetime import datetime

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class FFTSettings:
    def __init__(self, project_id):
        self.project_id = project_id
        self.window_type = "Hamming"
        self.start_frequency = 10.0
        self.stop_frequency = 2000.0
        self.number_of_lines = 1600
        self.overlap_percentage = 0.0
        self.averaging_mode = "No Averaging"
        self.number_of_averages = 10
        self.weighting_mode = "Linear"
        self.linear_mode = "Continuous"
        self.updated_at = datetime.utcnow()

class FFTViewFeature:
    def __init__(self, parent, db, project_name, channel=None, model_name=None, console=None, layout="vertical", channel_count=4):
        self.parent = parent
        self.db = db
        self.project_name = project_name
        self.channel_name = channel
        self.model_name = model_name
        self.console = console
        self.widget = None
        self.magnitude_plot_widget = None
        self.phase_plot_widget = None
        self.magnitude_plot_item = None
        self.phase_plot_item = None
        self.sample_rate = 1000
        self.channel_index = self.resolve_channel_index(channel) if channel is not None else None
        self.latest_data = None
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plot)
        self.update_interval = 200
        self.max_samples = 4096
        self.layout_type = layout
        self.mongo_client = self.db.client  # Use existing client from db
        self.project_id = None
        self.settings = FFTSettings(None)
        self.data_buffer = []
        self.settings_panel = None
        self.settings_button = None
        self.channel_count = channel_count
        self.last_frame_index = -1
        self.is_saving = False
        self.current_filename = None
        self.initUI()
        self.initialize_async()
        if self.console:
            self.console.append_to_console(f"Initialized FFTViewFeature for {self.model_name}/{self.channel_name or 'No Channel'} with {self.channel_count} channels")

    def resolve_channel_index(self, channel):
        try:
            if isinstance(channel, str):
                project_data = self.db.get_project_data(self.project_name) if self.db else {}
                models = project_data.get("models", [])
                for m_data in models:
                    if m_data.get("name") == self.model_name:
                        channels = m_data.get("channels", [])
                        for idx, ch in enumerate(channels):
                            if ch.get("channelName") == channel:
                                logging.debug(f"Resolved channel {channel} to index {idx} in model {self.model_name}")
                                return idx
                        logging.warning(f"Channel {channel} not found in model {self.model_name}. Available channels: {[ch.get('channelName') for ch in channels]}")
                        if self.console:
                            self.console.append_to_console(f"Warning: Channel {channel} not found in model {self.model_name}")
                        return None
                logging.warning(f"Model {self.model_name} not found in project {self.project_name}")
                if self.console:
                    self.console.append_to_console(f"Warning: Model {self.model_name} not found in project {self.project_name}")
                return None
            elif isinstance(channel, int):
                if channel >= 0:
                    return channel
                else:
                    logging.warning(f"Invalid channel index: {channel}")
                    if self.console:
                        self.console.append_to_console(f"Warning: Invalid channel index: {channel}")
                    return None
            else:
                logging.warning(f"Invalid channel type: {type(channel)}")
                if self.console:
                    self.console.append_to_console(f"Warning: Invalid channel type: {type(channel)}")
                return None
        except Exception as e:
            logging.error(f"Failed to resolve channel index for {channel}: {e}")
            if self.console:
                self.console.append_to_console(f"Error: Failed to resolve channel index for {channel}: {e}")
            return None

    def initUI(self):
        self.widget = QWidget()
        main_layout = QVBoxLayout()
        self.widget.setLayout(main_layout)

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
                min-width: 代表px;
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
        main_layout.addLayout(top_layout)

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
        settings_layout.setSpacing(10)
        self.settings_panel.setLayout(settings_layout)

        settings_labels = [
            "Window Type", "Start Frequency (Hz)", "Stop Frequency (Hz)",
            "Number of Lines", "Overlap Percentage (%)", "Averaging Mode",
            "Number of Averages", "Weighting Mode", "Linear Mode"
        ]
        self.settings_widgets = {}

        window_label = QLabel("Window Type")
        window_label.setStyleSheet("font-size: 14px;")
        settings_layout.addWidget(window_label, 0, 0)
        window_combo = QComboBox()
        window_combo.addItems(["Hamming", "Hanning", "Blackman", "Flat-top", "None"])
        window_combo.setCurrentText(self.settings.window_type)
        window_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                background-color: white;
                min-width: 100px;
            }
        """)
        settings_layout.addWidget(window_combo, 0, 1)
        self.settings_widgets["WindowType"] = window_combo

        start_freq_label = QLabel("Start Frequency (Hz)")
        start_freq_label.setStyleSheet("font-size: 14px;")
        settings_layout.addWidget(start_freq_label, 1, 0)
        start_freq_edit = QLineEdit(str(self.settings.start_frequency))
        start_freq_edit.setValidator(QDoubleValidator(0.0, 10000.0, 2))
        start_freq_edit.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                background-color: white;
                min-width: 100px;
            }
        """)
        settings_layout.addWidget(start_freq_edit, 1, 1)
        self.settings_widgets["StartFrequency"] = start_freq_edit

        stop_freq_label = QLabel("Stop Frequency (Hz)")
        stop_freq_label.setStyleSheet("font-size: 14px;")
        settings_layout.addWidget(stop_freq_label, 2, 0)
        stop_freq_edit = QLineEdit(str(self.settings.stop_frequency))
        stop_freq_edit.setValidator(QDoubleValidator(0.0, 10000.0, 2))
        stop_freq_edit.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                background-color: white;
                min-width: 100px;
            }
        """)
        settings_layout.addWidget(stop_freq_edit, 2, 1)
        self.settings_widgets["StopFrequency"] = stop_freq_edit

        lines_label = QLabel("Number of Lines")
        lines_label.setStyleSheet("font-size: 14px;")
        settings_layout.addWidget(lines_label, 3, 0)
        lines_edit = QLineEdit(str(self.settings.number_of_lines))
        lines_edit.setValidator(QIntValidator(100, 3200))
        lines_edit.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                background-color: white;
                min-width: 100px;
            }
        """)
        settings_layout.addWidget(lines_edit, 3, 1)
        self.settings_widgets["NumberOfLines"] = lines_edit

        overlap_label = QLabel("Overlap Percentage (%)")
        overlap_label.setStyleSheet("font-size: 14px;")
        settings_layout.addWidget(overlap_label, 4, 0)
        overlap_edit = QLineEdit(str(self.settings.overlap_percentage))
        overlap_edit.setValidator(QDoubleValidator(0.0, 99.9, 2))
        overlap_edit.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                background-color: white;
                min-width: 100px;
            }
        """)
        settings_layout.addWidget(overlap_edit, 4, 1)
        self.settings_widgets["OverlapPercentage"] = overlap_edit

        avg_mode_label = QLabel("Averaging Mode")
        avg_mode_label.setStyleSheet("font-size: 14px;")
        settings_layout.addWidget(avg_mode_label, 5, 0)
        avg_mode_combo = QComboBox()
        avg_mode_combo.addItems(["No Averaging", "Linear", "Exponential"])
        avg_mode_combo.setCurrentText(self.settings.averaging_mode)
        avg_mode_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                background-color: white;
                min-width: 100px;
            }
        """)
        settings_layout.addWidget(avg_mode_combo, 5, 1)
        self.settings_widgets["AveragingMode"] = avg_mode_combo

        avg_num_label = QLabel("Number of Averages")
        avg_num_label.setStyleSheet("font-size: 14px;")
        settings_layout.addWidget(avg_num_label, 6, 0)
        avg_num_edit = QLineEdit(str(self.settings.number_of_averages))
        avg_num_edit.setValidator(QIntValidator(1, 100))
        avg_num_edit.setStyleSheet("""
            QLineEdit {
                padding: 5px;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                background-color: white;
                min-width: 100px;
            }
        """)
        settings_layout.addWidget(avg_num_edit, 6, 1)
        self.settings_widgets["NumberOfAverages"] = avg_num_edit

        weight_label = QLabel("Weighting Mode")
        weight_label.setStyleSheet("font-size: 14px;")
        settings_layout.addWidget(weight_label, 7, 0)
        weight_combo = QComboBox()
        weight_combo.addItems(["Linear", "A-Weighting", "B-Weighting", "C-Weighting"])
        weight_combo.setCurrentText(self.settings.weighting_mode)
        weight_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                background-color: white;
                min-width: 100px;
            }
        """)
        settings_layout.addWidget(weight_combo, 7, 1)
        self.settings_widgets["WeightingMode"] = weight_combo

        linear_label = QLabel("Linear Mode")
        linear_label.setStyleSheet("font-size: 14px;")
        settings_layout.addWidget(linear_label, 8, 0)
        linear_combo = QComboBox()
        linear_combo.addItems(["Continuous", "Peak Hold", "Time Synchronous"])
        linear_combo.setCurrentText(self.settings.linear_mode)
        linear_combo.setStyleSheet("""
            QComboBox {
                padding: 5px;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                background-color: white;
                min-width: 100px;
            }
        """)
        settings_layout.addWidget(linear_combo, 8, 1)
        self.settings_widgets["LinearMode"] = linear_combo

        save_button = QPushButton("Save")
        save_button.setStyleSheet("""
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
        save_button.clicked.connect(self.save_settings)

        close_button = QPushButton("Close")
        close_button.setStyleSheet("""
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
        close_button.clicked.connect(self.close_settings)

        settings_layout.addWidget(save_button, 9, 0)
        settings_layout.addWidget(close_button, 9, 1)

        main_layout.addWidget(self.settings_panel)

        plot_layout = QHBoxLayout() if self.layout_type == "horizontal" else QVBoxLayout()
        pg.setConfigOptions(antialias=False)

        self.magnitude_plot_widget = pg.PlotWidget()
        self.magnitude_plot_widget.setBackground("white")
        display_channel = self.channel_name if self.channel_name else f"Channel_{self.channel_index + 1}" if self.channel_index is not None else "Unknown"
        self.magnitude_plot_widget.setTitle(f"Magnitude Spectrum - {self.model_name or 'Unknown'} - {display_channel}", color="black", size="12pt")
        self.magnitude_plot_widget.setLabel('left', 'Amplitude', color='#000000')
        self.magnitude_plot_widget.setLabel('bottom', 'Frequency (Hz)', color='#000000')
        self.magnitude_plot_widget.showGrid(x=True, y=True)
        self.magnitude_plot_widget.setXRange(self.settings.start_frequency, self.settings.stop_frequency, padding=0.02)
        self.magnitude_plot_widget.enableAutoRange('y', True)
        self.magnitude_plot_item = self.magnitude_plot_widget.plot(pen=pg.mkPen(color='#4a90e2', width=2))
        plot_layout.addWidget(self.magnitude_plot_widget)

        self.phase_plot_widget = pg.PlotWidget()
        self.phase_plot_widget.setBackground("white")
        self.phase_plot_widget.setTitle(f"Phase Spectrum - {self.model_name or 'Unknown'} - {display_channel}", color="black", size="12pt")
        self.phase_plot_widget.setLabel('left', 'Phase (degrees)', color='#000000')
        self.phase_plot_widget.setLabel('bottom', 'Frequency (Hz)', color='#000000')
        self.phase_plot_widget.showGrid(x=True, y=True)
        self.phase_plot_widget.setXRange(self.settings.start_frequency, self.settings.stop_frequency, padding=0.02)
        self.phase_plot_widget.setYRange(-180, 180, padding=0.02)
        self.phase_plot_item = self.phase_plot_widget.plot(pen=pg.mkPen(color='#e74c3c', width=2))
        plot_layout.addWidget(self.phase_plot_widget)

        main_layout.addLayout(plot_layout)
        self.update_timer.start(self.update_interval)

    def initialize_async(self):
        try:
            database = self.mongo_client.get_database("changed_db")
            projects_collection = database.get_collection("projects")
            project = projects_collection.find_one({"project_name": self.project_name, "email": self.db.email})
            if not project:
                self.log_and_set_status(f"Project {self.project_name} not found for email {self.db.email}.")
                return

            self.project_id = project["_id"]
            model = next((m for m in project["models"] if m["name"] == self.model_name), None)
            if not model:
                self.log_and_set_status(f"Model {self.model_name} not found in project {self.project_name}.")
                return

            channels = model.get("channels", [])
            if len(channels) != self.channel_count:
                self.log_and_set_status(f"Warning: Model {self.model_name} has {len(channels)} channels, expected {self.channel_count}")
            
            self.load_settings_from_database()
            if self.console:
                self.console.append_to_console(f"Initialized FFTViewFeature with project_id: {self.project_id}, channel_index: {self.channel_index}")
        except Exception as e:
            self.log_and_set_status(f"Error initializing FFTViewFeature: {str(e)}")

    def load_settings_from_database(self):
        try:
            database = self.mongo_client.get_database("changed_db")
            settings_collection = database.get_collection("FFTSettings")
            setting = settings_collection.find_one({"projectId": self.project_id}, sort=[("updatedAt", -1)])
            
            if setting:
                self.settings.window_type = setting.get("windowType", "Hamming")
                self.settings.start_frequency = float(setting.get("startFrequency", 10.0))
                self.settings.stop_frequency = float(setting.get("stopFrequency", 2000.0))
                self.settings.number_of_lines = int(setting.get("numberOfLines", 1600))
                self.settings.overlap_percentage = float(setting.get("overlapPercentage", 0.0))
                self.settings.averaging_mode = setting.get("averagingMode", "No Averaging")
                self.settings.number_of_averages = int(setting.get("numberOfAverages", 10))
                self.settings.weighting_mode = setting.get("weightingMode", "Linear")
                self.settings.linear_mode = setting.get("linearMode", "Continuous")
                
                self.settings_widgets["WindowType"].setCurrentText(self.settings.window_type)
                self.settings_widgets["StartFrequency"].setText(str(self.settings.start_frequency))
                self.settings_widgets["StopFrequency"].setText(str(self.settings.stop_frequency))
                self.settings_widgets["NumberOfLines"].setText(str(self.settings.number_of_lines))
                self.settings_widgets["OverlapPercentage"].setText(str(self.settings.overlap_percentage))
                self.settings_widgets["AveragingMode"].setCurrentText(self.settings.averaging_mode)
                self.settings_widgets["NumberOfAverages"].setText(str(self.settings.number_of_averages))
                self.settings_widgets["WeightingMode"].setCurrentText(self.settings.weighting_mode)
                self.settings_widgets["LinearMode"].setCurrentText(self.settings.linear_mode)
                
                self.magnitude_plot_widget.setXRange(self.settings.start_frequency, self.settings.stop_frequency, padding=0.02)
                self.phase_plot_widget.setXRange(self.settings.start_frequency, self.settings.stop_frequency, padding=0.02)
                
                if self.console:
                    self.console.append_to_console(f"Loaded FFT settings for project ID: {self.project_id}")
            else:
                if self.console:
                    self.console.append_to_console(f"No FFT settings found for project ID: {self.project_id}. Using defaults.")
        except Exception as e:
            self.log_and_set_status(f"Error loading FFT settings: {str(e)}")

    def save_settings_to_database(self):
        if not self.project_id:
            logging.error("Project ID is not set, cannot save settings")
            if self.console:
                self.console.append_to_console("Error: Project ID is not set, cannot save settings")
            return
        try:
            database = self.mongo_client.get_database("changed_db")
            settings_collection = database.get_collection("FFTSettings")
            setting = {
                "projectId": self.project_id,
                "windowType": self.settings.window_type,
                "startFrequency": self.settings.start_frequency,
                "stopFrequency": self.settings.stop_frequency,
                "numberOfLines": self.settings.number_of_lines,
                "overlapPercentage": self.settings.overlap_percentage,
                "averagingMode": self.settings.averaging_mode,
                "numberOfAverages": self.settings.number_of_averages,
                "weightingMode": self.settings.weighting_mode,
                "linearMode": self.settings.linear_mode,
                "updatedAt": datetime.utcnow()
            }
            result = settings_collection.update_one(
                {"projectId": self.project_id},
                {"$set": setting},
                upsert=True
            )
            if result.upserted_id:
                logging.info(f"Inserted new FFT settings document with ID: {result.upserted_id}")
            else:
                logging.info(f"Updated existing FFT settings document")
            if self.console:
                self.console.append_to_console(f"Saved FFT settings for project ID: {self.project_id}")
        except Exception as e:
            logging.error(f"Error saving FFT settings: {str(e)}")
            if self.console:
                self.console.append_to_console(f"Error saving FFT settings: {str(e)}")

    def start_saving(self):
        if self.is_saving:
            return
        try:
            self.current_filename = datetime.utcnow().strftime("%Y%m%d_%H%M%S_FFT.bin")
            self.is_saving = True
            if self.console:
                self.console.append_to_console(f"Started saving FFT data to {self.current_filename}")
        except Exception as e:
            self.log_and_set_status(f"Error starting FFT data saving: {str(e)}")

    def stop_saving(self):
        if not self.is_saving:
            return
        try:
            self.is_saving = False
            self.current_filename = None
            if self.console:
                self.console.append_to_console("Stopped saving FFT data")
        except Exception as e:
            self.log_and_set_status(f"Error stopping FFT data saving: {str(e)}")

    def save_data_to_database(self, tag_name, values, sample_rate, frame_index):
        try:
            database = self.mongo_client.get_database("changed_db")
            collection = database.get_collection("messages")
            message_data = {
                "topic": tag_name,
                "filename": self.current_filename,
                "frameIndex": frame_index,
                "message": {
                    "channel_data": [values[self.channel_index].tolist()] if self.channel_index is not None else [],
                    "sample_rate": sample_rate
                },
                "projectId": self.project_id,
                "createdAt": datetime.utcnow(),
                "modelName": self.model_name
            }
            collection.insert_one(message_data)
            if self.console:
                self.console.append_to_console(f"Saved FFT data to database, frame {frame_index}")
        except Exception as e:
            self.log_and_set_status(f"Error saving FFT data to database: {str(e)}")

    def toggle_settings(self):
        self.settings_panel.setVisible(not self.settings_panel.isVisible())
        self.settings_button.setVisible(not self.settings_panel.isVisible())

    def save_settings(self):
        try:
            self.settings.window_type = self.settings_widgets["WindowType"].currentText()
            self.settings.start_frequency = float(self.settings_widgets["StartFrequency"].text() or 10.0)
            self.settings.stop_frequency = float(self.settings_widgets["StopFrequency"].text() or 2000.0)
            self.settings.number_of_lines = int(self.settings_widgets["NumberOfLines"].text() or 1600)
            self.settings.overlap_percentage = float(self.settings_widgets["OverlapPercentage"].text() or 0.0)
            self.settings.averaging_mode = self.settings_widgets["AveragingMode"].currentText()
            self.settings.number_of_averages = int(self.settings_widgets["NumberOfAverages"].text() or 10)
            self.settings.weighting_mode = self.settings_widgets["WeightingMode"].currentText()
            self.settings.linear_mode = self.settings_widgets["LinearMode"].currentText()

            if self.settings.start_frequency >= self.settings.stop_frequency:
                self.settings.start_frequency = 10.0
                self.settings.stop_frequency = 2000.0
                self.settings_widgets["StartFrequency"].setText(str(self.settings.start_frequency))
                self.settings_widgets["StopFrequency"].setText(str(self.settings.stop_frequency))
                self.log_and_set_status("Invalid frequency range, reset to defaults.")
            if self.settings.number_of_lines < 100 or self.settings.number_of_lines > 3200:
                self.settings.number_of_lines = 1600
                self.settings_widgets["NumberOfLines"].setText(str(self.settings.number_of_lines))
                self.log_and_set_status("Invalid number of lines, reset to default.")
            if self.settings.overlap_percentage < 0 or self.settings.overlap_percentage > 99.9:
                self.settings.overlap_percentage = 0.0
                self.settings_widgets["OverlapPercentage"].setText(str(self.settings.overlap_percentage))
                self.log_and_set_status("Invalid overlap percentage, reset to default.")
            if self.settings.number_of_averages < 1 or self.settings.number_of_averages > 100:
                self.settings.number_of_averages = 10
                self.settings_widgets["NumberOfAverages"].setText(str(self.settings.number_of_averages))
                self.log_and_set_status("Invalid number of averages, reset to default.")

            self.save_settings_to_database()
            self.magnitude_plot_widget.setXRange(self.settings.start_frequency, self.settings.stop_frequency, padding=0.02)
            self.phase_plot_widget.setXRange(self.settings.start_frequency, self.settings.stop_frequency, padding=0.02)
            self.settings_panel.setVisible(False)
            self.settings_button.setVisible(True)
            if self.console:
                self.console.append_to_console("FFT settings updated and saved.")
            self.update_plot()
        except Exception as e:
            self.log_and_set_status(f"Error saving FFT settings: {str(e)}")

    def close_settings(self):
        self.settings_widgets["WindowType"].setCurrentText(self.settings.window_type)
        self.settings_widgets["StartFrequency"].setText(str(self.settings.start_frequency))
        self.settings_widgets["StopFrequency"].setText(str(self.settings.stop_frequency))
        self.settings_widgets["NumberOfLines"].setText(str(self.settings.number_of_lines))
        self.settings_widgets["OverlapPercentage"].setText(str(self.settings.overlap_percentage))
        self.settings_widgets["AveragingMode"].setCurrentText(self.settings.averaging_mode)
        self.settings_widgets["NumberOfAverages"].setText(str(self.settings.number_of_averages))
        self.settings_widgets["WeightingMode"].setCurrentText(self.settings.weighting_mode)
        self.settings_widgets["LinearMode"].setCurrentText(self.settings.linear_mode)
        self.settings_panel.setVisible(False)
        self.settings_button.setVisible(True)

    def get_widget(self):
        return self.widget

    def on_data_received(self, tag_name, model_name, values, sample_rate, frame_index):
        if self.model_name != model_name or self.channel_index is None:
            if self.console:
                self.console.append_to_console(
                    f"FFT View: Skipped data - model_name={model_name} (expected {self.model_name}), "
                    f"channel_index={self.channel_index}, frame {frame_index}"
                )
            return

        try:
            if frame_index != self.last_frame_index + 1 and self.last_frame_index != -1:
                logging.warning(f"Non-sequential frame index: expected {self.last_frame_index + 1}, got {frame_index}")
                if self.console:
                    self.console.append_to_console(f"Warning: Non-sequential frame index: expected {self.last_frame_index + 1}, got {frame_index}")
            self.last_frame_index = frame_index

            if len(values) < self.channel_count:
                self.log_and_set_status(f"Received {len(values)} channels, expected at least {self.channel_count}, frame {frame_index}")
                return
            if self.channel_index >= len(values):
                self.log_and_set_status(f"Channel index {self.channel_index} out of range for {len(values)} channels, frame {frame_index}")
                return

            self.sample_rate = sample_rate if sample_rate > 0 else 1000
            scaling_factor = 3.3 / 65535.0
            raw_data = np.array(values[self.channel_index][:self.max_samples], dtype=np.float32)
            self.latest_data = raw_data * scaling_factor
            self.data_buffer.append(self.latest_data.copy())

            if len(self.data_buffer) > self.settings.number_of_averages:
                self.data_buffer = self.data_buffer[-self.settings.number_of_averages:]

            if self.is_saving and self.current_filename:
                self.save_data_to_database(tag_name, values, sample_rate, frame_index)

            if self.console:
                self.console.append_to_console(
                    f"FFT View: Received data for channel {self.channel_name or self.channel_index}, "
                    f"samples={len(self.latest_data)}, Fs={self.sample_rate}Hz, frame {frame_index}"
                )
        except Exception as e:
            self.log_and_set_status(f"Error in on_data_received, frame {frame_index}: {str(e)}")

    def update_plot(self):
        if not self.data_buffer:
            return

        try:
            data = self.data_buffer[-1] if self.settings.averaging_mode == "No Averaging" else np.mean(self.data_buffer, axis=0)
            n = len(data)
            if n < 2:
                self.log_and_set_status(f"Insufficient data length: {n}")
                return

            window_name = self.settings.window_type.lower() if self.settings.window_type != "None" else "rectangular"
            window = get_window(window_name, n)
            windowed_data = data * window

            target_length = 2 ** int(np.ceil(np.log2(n)))
            padded_data = np.zeros(target_length)
            padded_data[:n] = windowed_data
            fft_result = fft(padded_data)
            half = target_length // 2

            frequencies = np.linspace(0, self.sample_rate / 2, half)
            freq_mask = (frequencies >= self.settings.start_frequency) & (frequencies <= self.settings.stop_frequency)
            filtered_frequencies = frequencies[freq_mask]

            magnitudes = np.abs(fft_result[:half]) / target_length
            phases = np.degrees(np.angle(fft_result[:half]))
            filtered_magnitudes = magnitudes[freq_mask]
            filtered_phases = phases[freq_mask]

            if self.settings.weighting_mode != "Linear":
                weights = np.ones_like(filtered_frequencies)
                if self.settings.weighting_mode == "A-Weighting":
                    weights = 1.0 / (1.0 + (filtered_frequencies / 1000) ** 2)
                elif self.settings.weighting_mode == "B-Weighting":
                    weights = 1.0 / (1.0 + (filtered_frequencies / 500) ** 2)
                elif self.settings.weighting_mode == "C-Weighting":
                    weights = 1.0 / (1.0 + (filtered_frequencies / 200) ** 2)
                filtered_magnitudes *= weights

            if self.settings.averaging_mode == "Linear" and len(self.data_buffer) > 1:
                avg_magnitudes = np.mean([np.abs(fft(np.zeros(target_length)[:n] + d * window)[:half]) / target_length for d in self.data_buffer], axis=0)
                avg_phases = np.mean([np.degrees(np.angle(fft(np.zeros(target_length)[:n] + d * window)[:half])) for d in self.data_buffer], axis=0)
                filtered_magnitudes = avg_magnitudes[freq_mask]
                filtered_phases = avg_phases[freq_mask]
            elif self.settings.averaging_mode == "Exponential" and len(self.data_buffer) > 1:
                alpha = 2.0 / (self.settings.number_of_averages + 1)
                avg_magnitudes = np.zeros(half)
                avg_phases = np.zeros(half)
                for d in self.data_buffer:
                    fft_d = fft(np.zeros(target_length)[:n] + d * window)
                    avg_magnitudes = alpha * (np.abs(fft_d[:half]) / target_length) + (1 - alpha) * avg_magnitudes
                    avg_phases = alpha * np.degrees(np.angle(fft_d[:half])) + (1 - alpha) * avg_phases
                filtered_magnitudes = avg_magnitudes[freq_mask]
                filtered_phases = avg_phases[freq_mask]

            if len(filtered_frequencies) > self.settings.number_of_lines:
                indices = np.linspace(0, len(filtered_frequencies) - 1, self.settings.number_of_lines, dtype=int)
                filtered_frequencies = filtered_frequencies[indices]
                filtered_magnitudes = filtered_magnitudes[indices]
                filtered_phases = filtered_phases[indices]

            self.magnitude_plot_item.setData(filtered_frequencies, filtered_magnitudes)
            self.phase_plot_item.setData(filtered_frequencies, filtered_phases)
            self.magnitude_plot_widget.setXRange(self.settings.start_frequency, self.settings.stop_frequency, padding=0.02)
            self.phase_plot_widget.setXRange(self.settings.start_frequency, self.settings.stop_frequency, padding=0.02)

            if self.console:
                self.console.append_to_console(
                    f"FFT Updated: Samples={n}, FFT Size={target_length}, "
                    f"Fs={self.sample_rate}Hz, Lines={len(filtered_frequencies)}, "
                    f"Range={self.settings.start_frequency}-{self.settings.stop_frequency}Hz"
                )
        except Exception as e:
            self.log_and_set_status(f"Error updating FFT: {str(e)}")

    def log_and_set_status(self, message):
        logging.error(message)
        if self.console:
            self.console.append_to_console(message)

    def close(self):
        self.update_timer.stop()

    def cleanup(self):
        self.close()

    def refresh_channel_properties(self):
        self.initialize_async()