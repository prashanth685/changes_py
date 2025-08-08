import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
from pymongo import MongoClient
import logging
from datetime import datetime
import math

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class BodePlotFeature:
    def __init__(self, parent, db, project_name, channel=None, model_name=None, console=None):
        self.parent = parent
        self.db = db
        self.project_name = project_name
        self.selected_channel = channel  # Initially selected channel, or None
        self.model_name = model_name
        self.console = console
        self.widget = None
        self.plot_widgets = {}  # Dictionary to store plot widgets per channel
        self.plots = {}  # Dictionary to store plot items per channel
        self.data = {}  # Dictionary to store frequencies, amplitudes, phases per channel
        self.tag_name = None
        self.channel_names = []
        self.channel_indices = {}  # Map channel names to indices
        self.scaling_factor = 3.3 / 65535.0
        self.colors = {
            'amplitude': (0, 0, 255),  # Blue
            'phase': (255, 0, 0)       # Red
        }
        self.init_data()
        self.init_ui()
        # Connect to TreeView's channel_selected signal
        if hasattr(self.parent, 'channel_selected'):
            self.parent.channel_selected.connect(self.on_channel_selected)
            self.log_info("Connected to channel_selected signal")
        else:
            self.log_error("Parent does not have channel_selected signal")

    def init_data(self):
        try:
            if not self.db.is_connected():
                self.db.reconnect()
            project_data = self.db.get_project_data(self.project_name)
            if not project_data or "models" not in project_data:
                self.log_error(f"Project {self.project_name} or models not found.")
                return
            model = next((m for m in project_data["models"] if m["name"] == self.model_name), None)
            if not model or not model.get("tagName"):
                self.log_error(f"TagName not found for Model: {self.model_name}")
                return
            self.tag_name = model["tagName"]
            self.channel_names = [c["channelName"] for c in model.get("channels", [])]
            self.channel_indices = {name: idx for idx, name in enumerate(self.channel_names)}
            if not self.channel_names:
                self.log_error(f"No channels found in model {self.model_name}.")
                return
            # Initialize data storage for each channel
            for ch_name in self.channel_names:
                self.data[ch_name] = {
                    'frequencies': [],
                    'amplitudes': [],
                    'phases': []
                }
            self.log_info(f"Initialized BodePlotFeature for Model: {self.model_name}, Tag: {self.tag_name}, Channels: {self.channel_names}")
            # Set initial selected channel if provided
            if self.selected_channel and self.selected_channel in self.channel_names:
                self.log_info(f"Initial channel set to: {self.selected_channel}")
            else:
                self.selected_channel = None
                self.log_info("No initial channel set")
        except Exception as e:
            self.log_error(f"Error initializing BodePlotFeature: {str(e)}")

    def init_ui(self):
        self.widget = QWidget()
        main_layout = QVBoxLayout()
        self.widget.setLayout(main_layout)

        # Header
        header_label = QLabel(f"Bode Plot for Model: {self.model_name}")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        main_layout.addWidget(header_label)

        # Progress bar for historical data processing
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # Plot container widget
        self.plot_container = QWidget()
        self.plot_layout = QVBoxLayout()
        self.plot_container.setLayout(self.plot_layout)
        main_layout.addWidget(self.plot_container)

        # Error message
        self.error_label = QLabel("Waiting for data or select a channel...")
        self.error_label.setStyleSheet("color: red; font-size: 14px; padding: 10px;")
        self.error_label.setAlignment(pg.QtCore.Qt.AlignCenter)
        main_layout.addWidget(self.error_label)
        self.error_label.setVisible(True)

        # Initialize plots for all channels but keep them hidden
        for ch_name in self.channel_names:
            # Container for channel plots
            channel_widget = QWidget()
            channel_layout = QVBoxLayout()
            channel_widget.setLayout(channel_layout)
            channel_widget.setVisible(False)  # Initially hidden

            # Amplitude plot
            amp_plot = pg.PlotWidget()
            amp_plot.setBackground('w')
            amp_plot.showGrid(x=True, y=True)
            amp_plot.setLabel('bottom', 'Frequency (Hz)')
            amp_plot.setLabel('left', 'Amplitude')
            amp_plot.setTitle(f"Amplitude vs Frequency - {ch_name}")
            amp_plot.addLegend()
            amp_line = amp_plot.plot([], [], pen=pg.mkPen(color=self.colors['amplitude'], width=2), name=ch_name)
            self.plot_widgets[f"{ch_name}_amp"] = amp_plot
            self.plots[f"{ch_name}_amp"] = amp_line
            channel_layout.addWidget(amp_plot)

            # Phase plot
            phase_plot = pg.PlotWidget()
            phase_plot.setBackground('w')
            phase_plot.showGrid(x=True, y=True)
            phase_plot.setLabel('bottom', 'Frequency (Hz)')
            phase_plot.setLabel('left', 'Phase (deg)')
            phase_plot.setTitle(f"Phase vs Frequency - {ch_name}")
            phase_plot.addLegend()
            phase_line = phase_plot.plot([], [], pen=pg.mkPen(color=self.colors['phase'], width=2), name=ch_name)
            self.plot_widgets[f"{ch_name}_phase"] = phase_plot
            self.plots[f"{ch_name}_phase"] = phase_line
            channel_layout.addWidget(phase_plot)

            self.plot_widgets[f"{ch_name}_widget"] = channel_widget
            self.plot_layout.addWidget(channel_widget)

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plots)
        self.update_timer.start(1000)  # Update every second

        self.log_info("Initialized BodePlotFeature UI")

    def on_channel_selected(self, model_name, channel_name):
        if model_name != self.model_name:
            self.log_info(f"Ignoring channel selection for model: {model_name}")
            return
        if channel_name not in self.channel_names:
            self.log_error(f"Selected channel {channel_name} not found in model {model_name}")
            return
        self.selected_channel = channel_name
        self.log_info(f"Channel selected: {channel_name}")
        self.update_visible_plots()
        self.update_plots()

    def update_visible_plots(self):
        try:
            # Hide all channel plots
            for ch_name in self.channel_names:
                self.plot_widgets[f"{ch_name}_widget"].setVisible(False)
            # Show only the selected channel's plots
            if self.selected_channel:
                self.plot_widgets[f"{self.selected_channel}_widget"].setVisible(True)
                self.error_label.setVisible(False)
            else:
                self.error_label.setText("Please select a channel")
                self.error_label.setVisible(True)
            self.log_info(f"Updated visible plots for channel: {self.selected_channel}")
        except Exception as e:
            self.log_error(f"Error updating visible plots: {str(e)}")

    def log_info(self, message):
        logging.info(message)
        if self.console:
            self.console.append_to_console(message)

    def log_error(self, message):
        logging.error(message)
        if self.console:
            self.console.append_to_console(message)
        self.error_label.setText(message)
        self.error_label.setVisible(True)

    def on_data_received(self, tag_name, model_name, values, sample_rate):
        if self.model_name != model_name or self.tag_name != tag_name:
            self.log_info(f"Ignoring data for tag: {tag_name}, model: {model_name}")
            return
        try:
            self.log_info(f"Received data: {len(values)} channels, sample_rate: {sample_rate}, first channel length: {len(values[0]) if values else 0}")

            # Fallback synthetic data for testing
            if not values or not values[0]:
                self.log_info("No valid data received; generating synthetic data for testing")
                values = [[np.sin(np.linspace(0, 10, 1000)) + i for i in range(len(self.channel_names))]]
                values.append([100.0 + i * 0.1 for i in range(1000)])  # Frequency data
                values.append([1 if i % 100 == 0 else 0 for i in range(1000)])  # Trigger data
                self.log_info(f"Synthetic data: {len(values)} channels, first channel length: {len(values[0])}")

            # Validate data
            expected_channels = len(self.channel_names)
            if len(values) < expected_channels:
                self.log_error(f"Invalid data: expected at least {expected_channels} channels, got {len(values)}")
                return

            # Extract main channels, frequency, and trigger data
            main_data = values[:expected_channels]
            freq_data = values[expected_channels] if len(values) > expected_channels else [0.0] * len(main_data[0])
            trigger_data = values[expected_channels + 1] if len(values) > expected_channels + 1 else [1 if i % 100 == 0 else 0 for i in range(len(main_data[0]))]
            self.log_info(f"Main data channels: {len(main_data)}, Freq data length: {len(freq_data)}, Trigger data length: {len(trigger_data)}")

            # Process only the selected channel if set, otherwise process all
            if self.selected_channel:
                ch_idx = self.channel_indices.get(self.selected_channel)
                if ch_idx is not None:
                    channel_data = [float(v) * self.scaling_factor for v in main_data[ch_idx]]
                    self.process_data(channel_data, freq_data, trigger_data, self.selected_channel)
                else:
                    self.log_error(f"Invalid channel index for {self.selected_channel}")
            else:
                for ch_idx, ch_name in enumerate(self.channel_names):
                    channel_data = [float(v) * self.scaling_factor for v in main_data[ch_idx]]
                    self.process_data(channel_data, freq_data, trigger_data, ch_name)

            self.update_plots()
        except Exception as e:
            self.log_error(f"Error processing data: {str(e)}")

    def process_data(self, channel_data, frequency_data, trigger_data, channel_name):
        try:
            # Validate inputs with relaxed checks
            if not channel_data:
                self.log_error(f"Empty channel data for {channel_name}")
                return
            if len(channel_data) != len(frequency_data) or len(channel_data) != len(trigger_data):
                min_length = min(len(channel_data), len(frequency_data), len(trigger_data))
                channel_data = channel_data[:min_length]
                frequency_data = frequency_data[:min_length]
                trigger_data = trigger_data[:min_length]
                self.log_info(f"Truncated arrays to length {min_length} for {channel_name}")

            # Log sample data
            self.log_info(f"Sample channel data for {channel_name}: {channel_data[:5]}")
            self.log_info(f"Sample frequency data for {channel_name}: {frequency_data[:5]}")
            self.log_info(f"Sample trigger data for {channel_name}: {trigger_data[:5]}")

            # Find trigger indices
            trigger_indices = [i for i, v in enumerate(trigger_data) if v == 1]
            min_distance = 5
            filtered_triggers = [trigger_indices[0]] if trigger_indices else []
            for idx in trigger_indices[1:]:
                if idx - filtered_triggers[-1] >= min_distance:
                    filtered_triggers.append(idx)
            trigger_indices = filtered_triggers

            # Relaxed trigger requirement: allow plotting with at least one segment
            if len(trigger_indices) < 1:
                self.log_info(f"No trigger points for {channel_name}; using entire data as one segment")
                trigger_indices = [0, len(channel_data)]

            # Process trigger segments
            temp_freq = []
            temp_amp = []
            temp_phase = []
            for i in range(len(trigger_indices) - 1):
                start_idx = trigger_indices[i]
                end_idx = trigger_indices[i + 1]
                segment_length = end_idx - start_idx
                if segment_length <= 0 or start_idx < 0 or end_idx > len(channel_data):
                    self.log_info(f"Skipping invalid segment for {channel_name}: start={start_idx}, end={end_idx}")
                    continue

                # Calculate DFT components
                sine_sum = 0.0
                cosine_sum = 0.0
                N = segment_length
                for n in range(segment_length):
                    theta = (2 * np.pi * n) / N
                    sine_sum += channel_data[start_idx + n] * np.sin(theta)
                    cosine_sum += channel_data[start_idx + n] * np.cos(theta)
                sine_component = sine_sum / N
                cosine_component = cosine_sum / N
                amplitude = np.sqrt(sine_component**2 + cosine_component**2) * 4
                phase = np.arctan2(cosine_component, sine_component) * (180.0 / np.pi)
                if phase < 0:
                    phase += 360

                # Average frequency for segment
                count = min(end_idx - start_idx, len(frequency_data) - start_idx)
                frequency = np.mean(frequency_data[start_idx:start_idx + count]) if count > 0 else 0.0
                if count <= 0:
                    self.log_info(f"No valid frequency data for segment in {channel_name}")
                    continue

                if not (np.isnan(amplitude) or np.isinf(amplitude) or np.isnan(phase) or np.isinf(phase) or np.isnan(frequency) or np.isinf(frequency)):
                    temp_freq.append(frequency)
                    temp_amp.append(amplitude)
                    temp_phase.append(phase)

            # Group by frequency and average
            if temp_freq:
                freq_groups = {}
                for f, a, p in zip(temp_freq, temp_amp, temp_phase):
                    freq_key = round(f, 2)
                    if freq_key not in freq_groups:
                        freq_groups[freq_key] = []
                    freq_groups[freq_key].append((a, p))

                sorted_data = []
                for freq_key in sorted(freq_groups.keys()):
                    amps, phases = zip(*freq_groups[freq_key])
                    sorted_data.append({
                        'f': freq_key,
                        'a': np.mean(amps),
                        'p': np.mean(phases)
                    })

                # Apply moving average (window size 7)
                window_size = 7
                smoothed_freq = []
                smoothed_amp = []
                smoothed_phase = []
                for i in range(len(sorted_data)):
                    start_idx = max(0, i - window_size // 2)
                    end_idx = min(len(sorted_data), i + window_size // 2 + 1)
                    window = sorted_data[start_idx:end_idx]
                    avg_freq = np.mean([x['f'] for x in window])
                    avg_amp = np.mean([x['a'] for x in window])
                    avg_phase = np.mean([x['p'] for x in window])
                    smoothed_freq.append(avg_freq)
                    smoothed_amp.append(avg_amp)
                    smoothed_phase.append(avg_phase)

                # Update data for channel
                self.data[channel_name]['frequencies'] = smoothed_freq
                self.data[channel_name]['amplitudes'] = smoothed_amp
                self.data[channel_name]['phases'] = smoothed_phase
                self.log_info(f"Processed {len(smoothed_freq)} data points for {channel_name}: freq={smoothed_freq[:5]}, amp={smoothed_amp[:5]}, phase={smoothed_phase[:5]}")
            else:
                self.log_info(f"No valid data points processed for {channel_name}")
        except Exception as e:
            self.log_error(f"Error processing data for {channel_name}: {str(e)}")

    def update_plots(self):
        try:
            if not self.selected_channel:
                self.error_label.setText("Please select a channel")
                self.error_label.setVisible(True)
                for ch_name in self.channel_names:
                    self.plots[f"{ch_name}_amp"].setData([], [])
                    self.plots[f"{ch_name}_phase"].setData([], [])
                self.log_info("No channel selected; cleared all plots")
                return

            ch_name = self.selected_channel
            freq = np.array(self.data[ch_name]['frequencies'], dtype=float)
            amp = np.array(self.data[ch_name]['amplitudes'], dtype=float)
            phase = np.array(self.data[ch_name]['phases'], dtype=float)
            self.log_info(f"Updating plots for {ch_name}: {len(freq)} data points, freq={freq[:5].tolist()}, amp={amp[:5].tolist()}, phase={phase[:5].tolist()}")

            # Ensure plot widget is visible
            self.update_visible_plots()

            # Validate data lengths
            freq_len, amp_len, phase_len = len(freq), len(amp), len(phase)
            if freq_len == 0 or amp_len == 0 or phase_len == 0 or not (freq_len == amp_len == phase_len):
                self.plots[f"{ch_name}_amp"].setData([], [])
                self.plots[f"{ch_name}_phase"].setData([], [])
                self.error_label.setText(f"Invalid data lengths for {ch_name}: freq={freq_len}, amp={amp_len}, phase={phase_len}")
                self.error_label.setVisible(True)
                self.log_info(f"Invalid data lengths for {ch_name}: freq={freq_len}, amp={amp_len}, phase={phase_len}")
                return

            # Update amplitude plot
            self.plots[f"{ch_name}_amp"].setData(freq, amp, connect="all")
            vb = self.plot_widgets[f"{ch_name}_amp"].getViewBox()
            x_min = np.min(freq) if freq_len > 1 else freq[0] - 1
            x_max = np.max(freq) if freq_len > 1 else freq[0] + 1
            x_range = x_max - x_min if x_max > x_min else 1
            y_min = np.min(amp) if amp_len > 1 else amp[0] - 1
            y_max = np.max(amp) if amp_len > 1 else amp[0] + 1
            y_range = y_max - y_min if y_max > y_min else 1
            vb.setXRange(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
            vb.setYRange(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            self.log_info(f"Updated amplitude plot for {ch_name}: x_range={x_min}-{x_max}, y_range={y_min}-{y_max}")

            # Update phase plot
            self.plots[f"{ch_name}_phase"].setData(freq, phase, connect="all")
            vb = self.plot_widgets[f"{ch_name}_phase"].getViewBox()
            x_min = np.min(freq) if freq_len > 1 else freq[0] - 1
            x_max = np.max(freq) if freq_len > 1 else freq[0] + 1
            x_range = x_max - x_min if x_max > x_min else 1
            y_min = max(-360, np.min(phase)) if phase_len > 1 else max(-360, phase[0] - 1)
            y_max = min(360, np.max(phase)) if phase_len > 1 else min(360, phase[0] + 1)
            y_range = y_max - y_min if y_max > y_min else 1
            vb.setXRange(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
            vb.setYRange(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            self.log_info(f"Updated phase plot for {ch_name}: x_range={x_min}-{x_max}, y_range={y_min}-{y_max}")

            self.error_label.setVisible(False)
        except Exception as e:
            self.log_error(f"Error updating plots: {str(e)}")
            self.error_label.setText(f"Plotting error for {ch_name}: {str(e)}")
            self.error_label.setVisible(True)

    def process_historical_data(self, filename, frame_index):
        try:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            client = MongoClient("mongodb://localhost:27017")
            database = client["changed_db"]
            history_collection = database["timeview_messages"]

            # Get total frames
            query = {
                "project_name": self.project_name,
                "model_name": self.model_name,
                "topic": self.tag_name,
                "filename": filename
            }
            total_frames = history_collection.count_documents(query)
            self.log_info(f"Found {total_frames} frames for filename: {filename}")

            if total_frames == 0:
                self.log_error(f"No historical data found for filename: {filename}")
                self.progress_bar.setVisible(False)
                return

            # Initialize data storage for selected channel or all channels
            if self.selected_channel:
                self.data[self.selected_channel] = {'frequencies': [], 'amplitudes': [], 'phases': []}
            else:
                for ch_name in self.channel_names:
                    self.data[ch_name] = {'frequencies': [], 'amplitudes': [], 'phases': []}

            # Process in batches
            max_frames = 1500
            batch_size = 50
            sampling_interval = max(1, total_frames // max_frames)
            processed_count = 0
            cursor = history_collection.find(query).sort("frameIndex", 1)

            for history_data in cursor:
                if processed_count % sampling_interval != 0:
                    processed_count += 1
                    continue
                if not self.is_valid_history_data(history_data):
                    processed_count += 1
                    continue

                main_channels = history_data.get("numberOfChannels", 0)
                samples_per_channel = history_data.get("samplingRate", 0)
                taco_channels = history_data.get("tacoChannelCount", 0)
                freq_start_idx = main_channels * samples_per_channel
                trigger_start_idx = freq_start_idx + samples_per_channel

                if self.selected_channel:
                    ch_idx = self.channel_indices.get(self.selected_channel)
                    if ch_idx is not None and ch_idx < main_channels:
                        channel_data = [history_data["message"][i * main_channels + ch_idx] * self.scaling_factor
                                       for i in range(samples_per_channel)]
                        freq_data = [history_data["message"][freq_start_idx + i]
                                     for i in range(samples_per_channel) if freq_start_idx + i < len(history_data["message"])]
                        trigger_data = [history_data["message"][trigger_start_idx + i]
                                        for i in range(samples_per_channel) if trigger_start_idx + i < len(history_data["message"])]
                        self.log_info(f"Processing historical data for {self.selected_channel}: {len(channel_data)} samples")
                        self.process_data(channel_data, freq_data, trigger_data, self.selected_channel)
                    else:
                        self.log_error(f"Invalid channel index {ch_idx} for {self.selected_channel}")
                else:
                    for ch_idx, ch_name in enumerate(self.channel_names):
                        if ch_idx >= main_channels:
                            continue
                        channel_data = [history_data["message"][i * main_channels + ch_idx] * self.scaling_factor
                                       for i in range(samples_per_channel)]
                        freq_data = [history_data["message"][freq_start_idx + i]
                                     for i in range(samples_per_channel) if freq_start_idx + i < len(history_data["message"])]
                        trigger_data = [history_data["message"][trigger_start_idx + i]
                                        for i in range(samples_per_channel) if trigger_start_idx + i < len(history_data["message"])]
                        self.log_info(f"Processing historical data for {ch_name}: {len(channel_data)} samples")
                        self.process_data(channel_data, freq_data, trigger_data, ch_name)

                processed_count += 1
                self.progress_bar.setValue(int((processed_count / total_frames) * 100))
                if processed_count % batch_size == 0:
                    self.update_plots()

            self.update_plots()
            self.progress_bar.setVisible(False)
            self.log_info(f"Processed {processed_count}/{total_frames} frames for {filename}")
            client.close()
        except Exception as e:
            self.log_error(f"Error processing historical data: {str(e)}")
            self.progress_bar.setVisible(False)

    def is_valid_history_data(self, history_data):
        try:
            main_channels = history_data.get("numberOfChannels", 0)
            samples_per_channel = history_data.get("samplingRate", 0)
            taco_channels = history_data.get("tacoChannelCount", 0)
            message = history_data.get("message", [])
            valid = (main_channels > 0 and
                     samples_per_channel > 0 and
                     len(message) >= (main_channels + taco_channels) * samples_per_channel)
            if not valid:
                self.log_error(f"Invalid history data: channels={main_channels}, samples={samples_per_channel}, message_len={len(message)}")
            return valid
        except Exception as e:
            self.log_error(f"Error validating history data: {str(e)}")
            return False

    def get_widget(self):
        return self.widget

    def cleanup(self):
        self.update_timer.stop()
        for ch_name in self.channel_names:
            self.data[ch_name].clear()
        self.plots.clear()
        self.plot_widgets.clear()
        self.log_info("Cleaned up BodePlotFeature")