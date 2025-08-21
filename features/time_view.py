import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QPushButton, QComboBox, QGridLayout
from PyQt5.QtCore import QObject, QEvent, Qt, QTimer
from PyQt5.QtGui import QIcon
from pyqtgraph import PlotWidget, mkPen, AxisItem, SignalProxy, InfiniteLine
from datetime import datetime, timedelta
import time
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeAxisItem(AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        return [datetime.fromtimestamp(v).strftime('%Y-%m-%d\n%H:%M:%S') for v in values]

class MouseTracker(QObject):
    def __init__(self, parent, idx, feature):
        super().__init__(parent)
        self.idx = idx
        self.feature = feature

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Enter:
            self.feature.mouse_enter(self.idx)
        elif event.type() == QEvent.Leave:
            self.feature.mouse_leave(self.idx)
        return False

class TimeViewFeature:
    def __init__(self, parent, db, project_name, channel=None, model_name=None, console=None):
        super().__init__()
        self.parent = parent
        self.db = db
        self.project_name = project_name
        self.channel = channel
        self.model_name = model_name
        self.console = console

        self.widget = None
        self.plot_widgets = []
        self.plots = []
        self.fifo_data = []
        self.fifo_times = []
        self.vlines = []
        self.proxies = []
        self.trackers = []

        self.sample_rate = None
        self.main_channels = None
        self.tacho_channels_count = 2
        self.total_channels = None
        self.scaling_factor = 3.3 / 65535
        self.num_plots = None
        self.samples_per_channel = None
        self.window_seconds = 1
        self.previous_window_seconds = 1
        self.fifo_window_samples = None

        self.settings_panel = None
        self.settings_button = None
        self.refresh_timer = None
        self.needs_refresh = []
        self.is_initialized = False

        self.channel_properties = {}
        self.channel_names = []
        self.is_scrolling = False
        self.active_line_idx = None

        self.plot_colors = [
            '#0000FF', '#FF0000', '#00FF00', '#800080', '#FFA500', '#A52A2A', '#FFC0CB', '#008080',
            '#FF4500', '#32CD32', '#00CED1', '#FFD700', '#FF69B4', '#8A2BE2', '#FF6347', '#20B2AA',
            '#ADFF2F', '#9932CC', '#FF7F50', '#00FA9A', '#9400D3'
        ]

        self.initUI()
        self.load_channel_properties()

    def initUI(self):
        self.widget = QWidget()
        main_layout = QVBoxLayout()

        header = QLabel(f"TIME VIEW FOR {self.project_name.upper()}")
        header.setStyleSheet("color: black; font-size: 26px; font-weight: bold; padding: 8px;")
        main_layout.addWidget(header, alignment=Qt.AlignCenter)

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
        QPushButton:hover { background-color: #45a049; }
        QPushButton:pressed { background-color: #3d8b40; }
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

        window_label = QLabel("Window Size (seconds)")
        window_label.setStyleSheet("font-size: 14px;")
        settings_layout.addWidget(window_label, 0, 0)

        window_combo = QComboBox()
        window_combo.addItems([str(i) for i in range(1, 11)])
        window_combo.setCurrentText(str(self.window_seconds))
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
        self.settings_widgets = {"WindowSeconds": window_combo}

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
        QPushButton:hover { background-color: #1e88e5; }
        QPushButton:pressed { background-color: #1976d2; }
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
        QPushButton:hover { background-color: #e53935; }
        QPushButton:pressed { background-color: #d32f2f; }
        """)
        close_button.clicked.connect(self.close_settings)

        settings_layout.addWidget(save_button, 1, 0)
        settings_layout.addWidget(close_button, 1, 1)
        main_layout.addWidget(self.settings_panel)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
        QScrollArea { border-radius: 8px; padding: 5px; }
        QScrollBar:vertical { background: white; width: 10px; margin: 0px; border-radius: 5px; }
        QScrollBar::handle:vertical { background: black; border-radius: 5px; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
        """)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_content.setStyleSheet("background-color: #d1d6d9; border-radius: 5px; padding: 10px;")
        self.scroll_area.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll_area)

        self.widget.setLayout(main_layout)

        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_plots)
        self.refresh_timer.start(100)

        self.scroll_debounce_timer = QTimer()
        self.scroll_debounce_timer.setInterval(200)
        self.scroll_debounce_timer.timeout.connect(self.stop_scrolling)
        self.scroll_area.verticalScrollBar().valueChanged.connect(self.on_scroll_changed)

        if not self.model_name and self.console:
            self.console.append_to_console("No model selected in TimeViewFeature.")
        if not self.channel and self.console:
            self.console.append_to_console("No channel selected in TimeViewFeature.")
        logging.debug("UI initialized, waiting for data to start plotting")

    def load_channel_properties(self):
        try:
            project_data = self.db.get_project_data(self.project_name)
            if not project_data:
                self.log_and_set_status(f"Project {self.project_name} not found")
                return
            for model in project_data.get("models", []):
                if model.get("name") == self.model_name:
                    self.channel_names = [ch.get("channelName") for ch in model.get("channels", [])]
                    for channel in model.get("channels", []):
                        channel_name = channel.get("channelName")
                        self.channel_properties[channel_name] = {
                            "type": channel.get("type", "Displacement"),
                            "unit": channel.get("unit", "mil").lower(),
                            "correctionValue": float(channel.get("correctionValue", "1.0") or "1.0"),
                            "gain": float(channel.get("gain", "1.0") or "1.0"),
                            "sensitivity": float(channel.get("sensitivity", "1.0") or "1.0"),
                            "convertedSensitivity": float(channel.get("ConvertedSensitivity", channel.get("sensitivity", "1.0")) or "1.0")
                        }
                    break
            logging.debug(f"Loaded channel properties: {self.channel_properties}")
            logging.debug(f"Channel names: {self.channel_names}")
        except Exception as e:
            self.log_and_set_status(f"Error loading channel properties: {str(e)}")

    def on_scroll_changed(self):
        self.is_scrolling = True
        self.scroll_debounce_timer.stop()
        self.scroll_debounce_timer.start()

    def stop_scrolling(self):
        self.is_scrolling = False
        self.scroll_debounce_timer.stop()

    def initialize_plots(self, channel_count):
        if not channel_count:
            self.log_and_set_status("Cannot initialize plots: channel count not set")
            return

        self.plot_widgets = []
        self.plots = []
        self.fifo_data = []
        self.fifo_times = []
        self.vlines = []
        self.proxies = []
        self.trackers = []
        self.needs_refresh = []

        self.num_plots = channel_count
        self.total_channels = channel_count
        self.main_channels = channel_count - self.tacho_channels_count

        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_content.setStyleSheet("background-color: #d1d6d9; border-radius: 5px; padding: 10px;")

        for i in range(self.num_plots):
            plot_widget = PlotWidget()
            plot_widget.setMinimumHeight(200)
            plot_widget.setBackground('#d1d6d9')
            plot_widget.showGrid(x=True, y=True)
            plot_widget.addLegend()

            axis = TimeAxisItem(orientation='bottom')
            plot_widget.setAxisItems({'bottom': axis})

            channel_name = self.channel_names[i] if i < len(self.channel_names) else f"Channel {i + 1}"
            unit = self.channel_properties.get(channel_name, {}).get("unit", "mil")
            y_label = f"Amplitude ({unit})" if i < self.main_channels else "Value"

            if i >= self.main_channels:
                channel_name = "Frequency" if i == self.main_channels else "Trigger"
                plot_widget.setYRange(-0.5, 1.5, padding=0)
                plot_widget.getAxis('left').setLabel(f"{channel_name} ({y_label})")

            plot = plot_widget.plot([], [], pen=mkPen(color=self.plot_colors[i % len(self.plot_colors)], width=1))
            self.plot_widgets.append(plot_widget)
            self.plots.append(plot)

            self.fifo_data.append([])
            self.fifo_times.append([])
            self.needs_refresh.append(True)

            self.scroll_layout.addWidget(plot_widget)

            vline = InfiniteLine(pos=0, angle=90, movable=False, pen=mkPen('k', width=1, style=Qt.DashLine))
            vline.setVisible(False)
            plot_widget.addItem(vline)
            self.vlines.append(vline)

            tracker = MouseTracker(plot_widget, i, self)
            plot_widget.installEventFilter(tracker)
            self.trackers.append(tracker)

            proxy = SignalProxy(plot_widget.scene().sigMouseMoved, rateLimit=60, slot=lambda evt, idx=i: self.mouse_moved(evt, idx))
            self.proxies.append(proxy)

        self.scroll_area.setWidget(self.scroll_content)
        self.initialize_buffers()
        logging.debug(f"Initialized {self.num_plots} plots with {self.window_seconds}-second window")

    def initialize_buffers(self):
        if not self.sample_rate or not self.num_plots:
            self.log_and_set_status("Cannot initialize buffers: sample_rate or num_plots not set")
            return

        self.fifo_window_samples = int(self.sample_rate * self.window_seconds)
        current_time = datetime.now()
        time_step = 1.0 / self.sample_rate

        for i in range(self.num_plots):
            self.fifo_data[i] = np.zeros(self.fifo_window_samples)
            self.fifo_times[i] = np.array([current_time - timedelta(seconds=(self.fifo_window_samples - 1 - j) * time_step) for j in range(self.fifo_window_samples)])
            self.needs_refresh[i] = True

        self.is_initialized = True
        logging.debug(f"Initialized FIFO buffers: {self.num_plots} channels, {self.fifo_window_samples} samples each")

    def toggle_settings(self):
        self.settings_panel.setVisible(not self.settings_panel.isVisible())
        self.settings_button.setVisible(not self.settings_panel.isVisible())

    def save_settings(self):
        try:
            selected_seconds = int(self.settings_widgets["WindowSeconds"].currentText())
            if 1 <= selected_seconds <= 10:
                self.window_seconds = selected_seconds
                self.update_window_size()
                self.log_and_set_status(f"Applied window size: {self.window_seconds} seconds")
                self.refresh_plots()
            else:
                self.log_and_set_status(f"Invalid window seconds selected: {selected_seconds}. Must be 1-10.")
            self.settings_panel.setVisible(False)
            self.settings_button.setVisible(True)
        except Exception as e:
            self.log_and_set_status(f"Error saving TimeView settings: {str(e)}")

    def close_settings(self):
        self.settings_widgets["WindowSeconds"].setCurrentText(str(self.window_seconds))
        self.settings_panel.setVisible(False)
        self.settings_button.setVisible(True)

    def update_window_size(self):
        if not self.sample_rate or not self.num_plots or not self.is_initialized:
            self.log_and_set_status("Cannot update window size: sample_rate, num_plots, or initialization not set")
            return

        if self.window_seconds == self.previous_window_seconds:
            logging.debug("No change in window size, skipping update")
            return

        new_fifo_window_samples = int(self.sample_rate * self.window_seconds)
        current_time = datetime.now()
        time_step = 1.0 / self.sample_rate

        for i in range(self.num_plots):
            current_data = self.fifo_data[i]
            current_times = self.fifo_times[i]
            new_data = np.zeros(new_fifo_window_samples)
            new_times = np.array([current_time - timedelta(seconds=(new_fifo_window_samples - 1 - j) * time_step) for j in range(new_fifo_window_samples)])

            copy_length = min(len(current_data), new_fifo_window_samples)
            if copy_length > 0:
                new_data[-copy_length:] = current_data[-copy_length:]
                new_times[-copy_length:] = current_times[-copy_length:] if len(current_times) >= copy_length else new_times[-copy_length:]

            self.fifo_data[i] = new_data
            self.fifo_times[i] = new_times
            self.needs_refresh[i] = True

        self.fifo_window_samples = new_fifo_window_samples
        self.previous_window_seconds = self.window_seconds
        logging.debug(f"Updated FIFO buffers to {self.window_seconds} seconds, {self.fifo_window_samples} samples")
        self.initialize_plots(self.num_plots)

    def get_widget(self):
        return self.widget

    def on_data_received(self, tag_name, model_name, values, sample_rate, frame_index):
        logging.debug(f"on_data_received called with tag_name={tag_name}, model_name={model_name}, values_len={len(values) if values else 0}, sample_rate={sample_rate}, frame_index={frame_index}")
        if self.model_name != model_name:
            logging.debug(f"Ignoring data for model {model_name}, expected {self.model_name}")
            return
        try:
            if not values or not sample_rate or sample_rate <= 0:
                self.log_and_set_status(f"Invalid MQTT data: values={values}, sample_rate={sample_rate}")
                return

            expected_channels = len(values)
            if self.main_channels is None:
                self.main_channels = expected_channels - self.tacho_channels_count
                if self.main_channels < 0:
                    self.log_and_set_status(f"Channel mismatch: received {expected_channels}, expected at least {self.tacho_channels_count} tacho channels")
                    return

            self.total_channels = expected_channels
            self.sample_rate = sample_rate
            self.samples_per_channel = len(values[0]) if values else 0

            if not all(len(values[i]) == self.samples_per_channel for i in range(expected_channels)):
                self.log_and_set_status(f"Channel data length mismatch: expected {self.samples_per_channel} samples")
                return

            if not self.is_initialized or len(self.fifo_data) != self.total_channels:
                self.num_plots = self.total_channels
                self.initialize_plots(self.total_channels)

            current_time = datetime.now()
            time_step = 1.0 / sample_rate
            new_times = np.array([current_time - timedelta(seconds=(self.samples_per_channel - 1 - i) * time_step) for i in range(self.samples_per_channel)])

            for ch in range(self.total_channels):
                channel_name = self.channel_names[ch] if ch < len(self.channel_names) else f"Channel {ch + 1}"
                props = self.channel_properties.get(channel_name, {
                    "type": "Displacement",
                    "unit": "mil",
                    "correctionValue": 1.0,
                    "gain": 1.0,
                    "sensitivity": 1.0,
                    "convertedSensitivity": 1.0
                })
                volts = np.array(values[ch]) * self.scaling_factor

                if ch < self.main_channels:
                    new_data = volts * (props["correctionValue"] * props["gain"]) / props["sensitivity"]
                    if props["type"] == "Displacement":
                        if props["unit"] == "mm":
                            new_data *= 0.0254
                        elif props["unit"] == "um":
                            new_data *= 25.4
                elif ch == self.main_channels:
                    new_data = volts * 10
                else:
                    new_data = volts

                if len(self.fifo_data[ch]) != self.fifo_window_samples:
                    self.fifo_data[ch] = np.zeros(self.fifo_window_samples)
                    self.fifo_times[ch] = np.array([current_time - timedelta(seconds=(self.fifo_window_samples - 1 - j) * time_step) for j in range(self.fifo_window_samples)])

                self.fifo_data[ch] = np.roll(self.fifo_data[ch], -self.samples_per_channel)
                self.fifo_data[ch][-self.samples_per_channel:] = new_data
                self.fifo_times[ch] = np.roll(self.fifo_times[ch], -self.samples_per_channel)
                self.fifo_times[ch][-self.samples_per_channel:] = new_times
                self.needs_refresh[ch] = True

            for ch in range(self.total_channels):
                if len(self.fifo_times[ch]) > 1:
                    sort_indices = np.argsort([t.timestamp() for t in self.fifo_times[ch]])
                    self.fifo_times[ch] = self.fifo_times[ch][sort_indices]
                    self.fifo_data[ch] = self.fifo_data[ch][sort_indices]
                    self.needs_refresh[ch] = True

            self.refresh_plots()
        except Exception as e:
            logging.error(f"Error processing data: {str(e)}")
            self.log_and_set_status(f"Error processing data: {str(e)}")

    def refresh_plots(self):
        if self.is_scrolling:
            return
        try:
            for i in range(self.num_plots):
                if self.needs_refresh[i] and len(self.fifo_data[i]) > 0 and len(self.fifo_times[i]) > 0:
                    self.plot_widgets[i].clear()
                    self.plot_widgets[i].addLegend()
                    self.plot_widgets[i].showGrid(x=True, y=True)

                    channel_name = self.channel_names[i] if i < len(self.channel_names) else f"Channel {i + 1}"
                    unit = self.channel_properties.get(channel_name, {}).get("unit", "mil")
                    y_label = f"Amplitude ({unit})" if i < self.main_channels else "Value"
                    if i >= self.main_channels:
                        ylabel_name = "Frequency" if i == self.main_channels else "Trigger"
                        self.plot_widgets[i].setYRange(-0.5, 1.5, padding=0)
                        self.plot_widgets[i].getAxis('left').setLabel(f"{ylabel_name} ({y_label})")

                    time_data = np.array([t.timestamp() for t in self.fifo_times[i]])
                    pen = mkPen(color=self.plot_colors[i % len(self.plot_colors)], width=1)
                    self.plots[i] = self.plot_widgets[i].plot(time_data, self.fifo_data[i], pen=pen, name=channel_name)

                    if len(time_data) > 0:
                        self.plot_widgets[i].setXRange(min(time_data), max(time_data), padding=0.1)
                    self.plot_widgets[i].enableAutoRange(axis='y')
                    self.plot_widgets[i].getAxis('bottom').setStyle(tickTextOffset=10)

                    self.needs_refresh[i] = False
        except Exception as e:
            logging.error(f"Error refreshing plots: {str(e)}")
            self.log_and_set_status(f"Error refreshing plots: {str(e)}")

    def load_file(self, filename):
        """Existing loader: kept intact to support 'Open saved files' flow."""
        try:
            messages = self.db.get_history_messages(self.project_name, self.model_name, filename=filename)
            if not messages:
                self.log_and_set_status(f"No data found for filename {filename}")
                return

            message = messages[-1]
            main_channels = message.get("numberOfChannels", 0)
            tacho_channels = message.get("tacoChannelCount", 0)
            samples_per_channel = message.get("samplingSize", 0)
            sample_rate = message.get("samplingRate", 0)
            frame_index = message.get("frameIndex", 0)
            flattened_data = message.get("message", [])

            if not flattened_data or not sample_rate or not samples_per_channel:
                self.log_and_set_status(f"Invalid data in file {filename}")
                return

            total_channels = main_channels + tacho_channels
            if samples_per_channel * total_channels != len(flattened_data):
                self.log_and_set_status(f"Data length mismatch in file {filename}")
                return

            values = []
            for ch in range(total_channels):
                start_idx = ch * samples_per_channel
                end_idx = (ch + 1) * samples_per_channel
                values.append(flattened_data[start_idx:end_idx])

            self.main_channels = main_channels
            self.tacho_channels_count = tacho_channels
            self.total_channels = total_channels
            self.sample_rate = sample_rate
            self.samples_per_channel = samples_per_channel

            if not self.is_initialized or len(self.fifo_data) != self.total_channels:
                self.initialize_plots(total_channels)

            created_at = datetime.fromisoformat(message['createdAt'].replace('Z', '+00:00'))
            time_step = 1.0 / sample_rate
            new_times = np.array([created_at + timedelta(seconds=i * time_step) for i in range(samples_per_channel)])

            for ch in range(self.total_channels):
                channel_name = self.channel_names[ch] if ch < len(self.channel_names) else f"Channel {ch + 1}"
                props = self.channel_properties.get(channel_name, {
                    "type": "Displacement",
                    "unit": "mil",
                    "correctionValue": 1.0,
                    "gain": 1.0,
                    "sensitivity": 1.0,
                    "convertedSensitivity": 1.0
                })
                volts = np.array(values[ch]) * self.scaling_factor
                if ch < self.main_channels:
                    new_data = volts * (props["correctionValue"] * props["gain"]) / props["sensitivity"]
                    if props["type"] == "Displacement":
                        if props["unit"] == "mm":
                            new_data *= 0.0254
                        elif props["unit"] == "um":
                            new_data *= 25.4
                elif ch == self.main_channels:
                    new_data = volts * 10
                else:
                    new_data = volts

                self.fifo_data[ch] = new_data
                self.fifo_times[ch] = new_times
                self.needs_refresh[ch] = True

            self.refresh_plots()
            if self.console:
                self.console.append_to_console(f"Loaded data from {filename}, frame {frame_index}")
        except Exception as e:
            logging.error(f"Error loading file {filename}: {str(e)}")
            self.log_and_set_status(f"Error loading file {filename}: {str(e)}")

    # ADDED: direct payload loader from FrequencyPlot selection
    def load_selected_frame(self, payload: dict):
        """
        payload fields:
          - filename, model, frameIndex, timestamp, channelData (flattened),
          - project_name, numberOfChannels, tacoChannelCount, samplingRate, samplingSize
        """
        try:
            main_channels = int(payload.get("numberOfChannels", 0))
            tacho_channels = int(payload.get("tacoChannelCount", 0))
            samples_per_channel = int(payload.get("samplingSize", 0))
            sample_rate = float(payload.get("samplingRate", 0))
            flattened_data = payload.get("channelData", [])
            created_at_str = payload.get("timestamp")

            if not flattened_data or not sample_rate or not samples_per_channel:
                self.log_and_set_status("Invalid payload: missing channelData/sample_rate/samples_per_channel")
                return

            total_channels = main_channels + tacho_channels
            if samples_per_channel * total_channels != len(flattened_data):
                self.log_and_set_status(f"Payload data length mismatch: expected {samples_per_channel * total_channels}, got {len(flattened_data)}")
                return

            values = []
            for ch in range(total_channels):
                start_idx = ch * samples_per_channel
                end_idx = (ch + 1) * samples_per_channel
                values.append(flattened_data[start_idx:end_idx])

            # Initialize plot structure
            self.main_channels = main_channels
            self.tacho_channels_count = tacho_channels
            self.total_channels = total_channels
            self.sample_rate = sample_rate
            self.samples_per_channel = samples_per_channel

            if not self.is_initialized or len(self.fifo_data) != self.total_channels:
                self.initialize_plots(total_channels)

            # Use the actual creation time to reconstruct exact timestamps
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(str(created_at_str).replace('Z', '+00:00'))
                except Exception:
                    created_at = datetime.now()
            else:
                created_at = datetime.now()

            time_step = 1.0 / sample_rate
            new_times = np.array([created_at + timedelta(seconds=i * time_step) for i in range(samples_per_channel)])

            for ch in range(self.total_channels):
                channel_name = self.channel_names[ch] if ch < len(self.channel_names) else f"Channel {ch + 1}"
                props = self.channel_properties.get(channel_name, {
                    "type": "Displacement",
                    "unit": "mil",
                    "correctionValue": 1.0,
                    "gain": 1.0,
                    "sensitivity": 1.0,
                    "convertedSensitivity": 1.0
                })

                volts = np.array(values[ch]) * self.scaling_factor
                if ch < self.main_channels:
                    new_data = volts * (props["correctionValue"] * props["gain"]) / props["sensitivity"]
                    if props["type"] == "Displacement":
                        if props["unit"] == "mm":
                            new_data *= 0.0254
                        elif props["unit"] == "um":
                            new_data *= 25.4
                elif ch == self.main_channels:
                    new_data = volts * 10
                else:
                    new_data = volts

                self.fifo_data[ch] = np.array(new_data)
                self.fifo_times[ch] = np.array(new_times)
                self.needs_refresh[ch] = True

            self.refresh_plots()
            if self.console:
                self.console.append_to_console(f"Loaded selected frame {payload.get('frameIndex')} from {payload.get('filename')}")
        except Exception as e:
            logging.error(f"Error loading selected frame: {str(e)}")
            self.log_and_set_status(f"Error loading selected frame: {str(e)}")

    def mouse_enter(self, idx):
        self.active_line_idx = idx
        self.vlines[idx].setVisible(True)

    def mouse_leave(self, idx):
        self.active_line_idx = None
        for vline in self.vlines:
            vline.setVisible(False)

    def mouse_moved(self, evt, idx):
        if self.active_line_idx is None:
            return
        pos = evt[0]
        if not self.plot_widgets[idx].sceneBoundingRect().contains(pos):
            return
        mouse_point = self.plot_widgets[idx].plotItem.vb.mapSceneToView(pos)
        x = mouse_point.x()
        times = self.fifo_times[idx]
        if len(times) > 0:
            time_stamps = np.array([t.timestamp() for t in times])
            if x < time_stamps:
                x = time_stamps
            elif x > time_stamps[-1]:
                x = time_stamps[-1]
            for vline in self.vlines:
                vline.setPos(x)
                vline.setVisible(True)

    def log_and_set_status(self, message):
        logging.error(message)
        if self.console:
            self.console.append_to_console(message)

    def cleanup(self):
        try:
            if self.refresh_timer.isActive():
                self.refresh_timer.stop()
            for plot in self.plots:
                plot.setData([], [])
            for widget in self.plot_widgets:
                widget.clear()
            while self.scroll_layout.count():
                item = self.scroll_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.plot_widgets = []
            self.plots = []
            self.fifo_data = []
            self.fifo_times = []
            self.vlines = []
            self.proxies = []
            self.trackers = []
            self.num_plots = 0
            if self.widget:
                self.widget.setParent(None)
                self.widget.deleteLater()
            logging.debug("TimeViewFeature cleaned up")
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
