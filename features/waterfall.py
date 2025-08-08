from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import numpy as np
import math
import logging
from datetime import datetime

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class WaterfallFeature:
    def __init__(self, parent, db, project_name, channel=None, model_name=None, console=None, channel_count=None):
        self.parent = parent
        self.db = db
        self.project_name = project_name
        self.model_name = model_name
        self.console = console
        self.widget = None
        try:
            self.channel_count = int(channel_count) if channel_count is not None else self.get_channel_count_from_db()
            if self.channel_count <= 0:
                raise ValueError(f"Invalid channel count: {self.channel_count}")
        except (ValueError, TypeError) as e:
            self.channel_count = self.get_channel_count_from_db()
            if self.console:
                self.console.append_to_console(f"Invalid channel_count {channel_count}: {str(e)}. Using {self.channel_count} from database.")
            logging.error(f"Invalid channel_count {channel_count}: {str(e)}. Using {self.channel_count} from database.")
        self.max_lines = 1
        self.data_history = [[] for _ in range(self.channel_count)]
        self.phase_history = [[] for _ in range(self.channel_count)]
        self.scaling_factor = 3.3 / 65535.0
        self.sample_rate = 4096
        self.samples_per_channel = 4096
        self.last_frame_index = -1
        self.frequency_range = (0, 2000)
        self.channel_names = self.get_channel_names()
        self.initUI()
        if self.console:
            self.console.append_to_console(
                f"Initialized WaterfallFeature for {self.model_name or 'No Model'} with {self.channel_count} channels: {self.channel_names}"
            )

    def get_channel_count_from_db(self):
        try:
            if not self.db.is_connected():
                self.db.reconnect()
            project_data = self.db.get_project_data(self.project_name)
            if not project_data:
                if self.console:
                    self.console.append_to_console(f"Project {self.project_name} not found in database")
                return 1
            model = next((m for m in project_data.get("models", []) if m["name"] == self.model_name), None)
            if not model:
                if self.console:
                    self.console.append_to_console(f"Model {self.model_name} not found")
                return 1
            channels = model.get("channels", [])
            return max(1, len(channels))
        except Exception as e:
            if self.console:
                self.console.append_to_console(f"Error retrieving channel count from database: {str(e)}")
            logging.error(f"Error retrieving channel count from database: {str(e)}")
            return 1

    def get_channel_names(self):
        try:
            project_data = self.db.get_project_data(self.project_name) if self.db else {}
            model = next((m for m in project_data.get("models", []) if m["name"] == self.model_name), None)
            if model:
                return [c.get("channelName", f"Channel_{i+1}") for i, c in enumerate(model.get("channels", []))]
            return [f"Channel_{i+1}" for i in range(self.channel_count)]
        except Exception as e:
            if self.console:
                self.console.append_to_console(f"Error retrieving channel names: {str(e)}")
            logging.error(f"Error retrieving channel names: {str(e)}")
            return [f"Channel_{i+1}" for i in range(self.channel_count)]

    def initUI(self):
        self.widget = QWidget()
        layout = QVBoxLayout()
        self.widget.setLayout(layout)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111, projection='3d')
        layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self.widget)
        layout.addWidget(self.toolbar)
        if not self.model_name and self.console:
            self.console.append_to_console("No model selected in WaterfallFeature.")

    def get_widget(self):
        return self.widget

    def on_data_received(self, tag_name, model_name, values, sample_rate, frame_index):
        if self.model_name != model_name:
            if self.console:
                self.console.append_to_console(f"WaterfallFeature: Ignored data for model {model_name}, expected {self.model_name}, frame {frame_index}")
            return
        try:
            if frame_index != self.last_frame_index + 1 and self.last_frame_index != -1:
                logging.warning(f"Non-sequential frame index: expected {self.last_frame_index + 1}, got {frame_index}")
                if self.console:
                    self.console.append_to_console(f"Warning: Non-sequential frame index: expected {self.last_frame_index + 1}, got {frame_index}")
            self.last_frame_index = frame_index
            if len(values) < self.channel_count:
                if self.console:
                    self.console.append_to_console(
                        f"WaterfallFeature: Received {len(values)} channels, expected {self.channel_count}, frame {frame_index}"
                    )
                return
            self.sample_rate = sample_rate if sample_rate > 0 else 4096
            self.samples_per_channel = len(values[0]) if values and values[0] else 4096
            sample_count = self.samples_per_channel
            target_length = 2 ** math.ceil(math.log2(sample_count))
            fft_magnitudes = []
            fft_phases = []
            frequencies = np.fft.fftfreq(target_length, 1.0 / self.sample_rate)[:target_length // 2]
            freq_mask = (frequencies >= self.frequency_range[0]) & (frequencies <= self.frequency_range[1])
            filtered_frequencies = frequencies[freq_mask]
            if len(filtered_frequencies) == 0:
                if self.console:
                    self.console.append_to_console(f"Error: No valid frequencies in range {self.frequency_range}, frame {frame_index}")
                return
            for ch_idx in range(self.channel_count):
                if len(values[ch_idx]) != self.samples_per_channel:
                    if self.console:
                        self.console.append_to_console(
                            f"Invalid data length for channel {self.channel_names[ch_idx]}: got {len(values[ch_idx])}, expected {self.samples_per_channel}, frame {frame_index}"
                        )
                    continue
                channel_data = np.array(values[ch_idx], dtype=np.float32) * self.scaling_factor
                if not np.any(channel_data):
                    if self.console:
                        self.console.append_to_console(
                            f"Warning: Zero data for channel {self.channel_names[ch_idx]}, frame {frame_index}"
                        )
                    continue
                padded_data = np.pad(channel_data, (0, target_length - sample_count), mode='constant') if target_length > sample_count else channel_data
                fft_result = np.fft.fft(padded_data)
                half = target_length // 2
                magnitudes = (2.0 / target_length) * np.abs(fft_result[:half])
                magnitudes[0] /= 2
                if target_length % 2 == 0:
                    magnitudes[-1] /= 2
                phases = np.angle(fft_result[:half], deg=True)
                filtered_magnitudes = magnitudes[freq_mask]
                filtered_phases = phases[freq_mask]
                if len(filtered_frequencies) > 1600:
                    indices = np.linspace(0, len(filtered_frequencies) - 1, 1600, dtype=int)
                    filtered_frequencies_subset = filtered_frequencies[indices]
                    filtered_magnitudes = filtered_magnitudes[indices]
                    filtered_phases = filtered_phases[indices]
                else:
                    filtered_frequencies_subset = filtered_frequencies
                if len(filtered_magnitudes) == 0 or len(filtered_frequencies_subset) == 0:
                    if self.console:
                        self.console.append_to_console(
                            f"Error: Empty FFT data for channel {self.channel_names[ch_idx]}, frame {frame_index}"
                        )
                    continue
                self.data_history[ch_idx].append(filtered_magnitudes)
                self.phase_history[ch_idx].append(filtered_phases)
                if len(self.data_history[ch_idx]) > self.max_lines:
                    self.data_history[ch_idx].pop(0)
                    self.phase_history[ch_idx].pop(0)
                fft_magnitudes.append(filtered_magnitudes)
                fft_phases.append(filtered_phases)
                if self.console:
                    self.console.append_to_console(
                        f"WaterfallFeature: Processed FFT for channel {self.channel_names[ch_idx]}, "
                        f"samples={len(channel_data)}, Fs={self.sample_rate}Hz, FFT points={len(filtered_magnitudes)}, frame {frame_index}"
                    )
            if fft_magnitudes:
                self.update_waterfall_plot(filtered_frequencies_subset if fft_magnitudes else None)
            else:
                if self.console:
                    self.console.append_to_console(f"No valid FFT data to plot, frame {frame_index}")
        except Exception as e:
            if self.console:
                self.console.append_to_console(f"WaterfallFeature: Error processing data, frame {frame_index}: {str(e)}")
            logging.error(f"WaterfallFeature: Error processing data, frame {frame_index}: {str(e)}")

    def update_waterfall_plot(self, frequencies):
        try:
            self.ax.clear()
            self.ax.set_title(f"Waterfall FFT Plot (Model: {self.model_name}, {self.channel_count} Channels)")
            self.ax.set_xlabel("Frequency (Hz)")
            self.ax.set_ylabel("Channel")
            self.ax.set_zlabel("Amplitude (V)")
            self.ax.grid(True)
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black', 'brown']
            max_amplitude = 0
            plotted = False
            for ch_idx in range(self.channel_count):
                if not self.data_history[ch_idx]:
                    if self.console:
                        self.console.append_to_console(f"No data to plot for channel {self.channel_names[ch_idx]}")
                    continue
                num_lines = len(self.data_history[ch_idx])
                for idx, fft_line in enumerate(self.data_history[ch_idx]):
                    if len(fft_line) == 0:
                        if self.console:
                            self.console.append_to_console(f"Empty FFT data for channel {self.channel_names[ch_idx]}, line {idx}")
                        continue
                    x = frequencies if frequencies is not None and len(frequencies) == len(fft_line) else np.arange(len(fft_line))
                    y = np.full_like(x, ch_idx * (self.max_lines + 2))
                    z = fft_line
                    self.ax.plot(x, y, z, color=colors[ch_idx % len(colors)], label=self.channel_names[ch_idx] if idx == num_lines - 1 else None)
                    max_amplitude = max(max_amplitude, np.max(z) if len(z) > 0 else 0)
                    plotted = True
                    if self.console:
                        self.console.append_to_console(
                            f"Plotted channel {self.channel_names[ch_idx]}, FFT points={len(fft_line)}, max amplitude={np.max(z):.4f}"
                        )
            if not plotted:
                if self.console:
                    self.console.append_to_console("No valid data plotted, drawing empty plot")
                x = np.array([0, 1])
                y = np.array([0, 0])
                z = np.array([0, 0])
                self.ax.plot(x, y, z, color='gray', label='No Data')
            self.ax.set_ylim(-1, self.channel_count * (self.max_lines + 2))
            self.ax.set_xlim(self.frequency_range[0], self.frequency_range[1] if frequencies is not None else 1000)
            self.ax.set_zlim(0, max_amplitude * 1.1 if max_amplitude > 0 else 1.0)
            self.ax.legend(loc='upper right')
            self.ax.view_init(elev=20, azim=-45)
            self.figure.tight_layout()
            self.canvas.draw_idle()
            self.canvas.flush_events()
            if self.console:
                self.console.append_to_console(f"WaterfallFeature: Updated plot for {self.channel_count} channels, plotted={plotted}")
        except Exception as e:
            if self.console:
                self.console.append_to_console(f"WaterfallFeature: Error updating plot: {str(e)}")
            logging.error(f"WaterfallFeature: Error updating plot: {str(e)}")

    def cleanup(self):
        try:
            self.canvas.figure.clear()
            self.canvas.deleteLater()
            self.toolbar.deleteLater()
            self.widget.deleteLater()
            self.data_history = [[] for _ in range(self.channel_count)]
            self.phase_history = [[] for _ in range(self.channel_count)]
            if self.console:
                self.console.append_to_console(f"WaterfallFeature: Cleaned up resources")
        except Exception as e:
            if self.console:
                self.console.append_to_console(f"Error cleaning up WaterfallFeature: {str(e)}")
            logging.error(f"Error cleaning up WaterfallFeature: {str(e)}")

    def refresh_channel_properties(self):
        try:
            if not self.db.is_connected():
                self.db.reconnect()
            project_data = self.db.get_project_data(self.project_name)
            model = next((m for m in project_data.get("models", []) if m["name"] == self.model_name), None)
            if model:
                self.channel_names = [ch.get("channelName", f"Channel_{i+1}") for i, ch in enumerate(model.get("channels", []))]
                new_channel_count = len(self.channel_names)
                if new_channel_count != self.channel_count:
                    if self.console:
                        self.console.append_to_console(
                            f"Channel count updated from {self.channel_count} to {new_channel_count} for model {self.model_name}"
                        )
                    self.channel_count = new_channel_count
                    self.data_history = [self.data_history[i] if i < len(self.data_history) else [] for i in range(self.channel_count)]
                    self.phase_history = [self.phase_history[i] if i < len(self.phase_history) else [] for i in range(self.channel_count)]
                if self.console:
                    self.console.append_to_console(f"Refreshed channel properties: {self.channel_count} channels: {self.channel_names}")
        except Exception as e:
            if self.console:
                self.console.append_to_console(f"Error refreshing channel properties: {str(e)}")
            logging.error(f"Error refreshing channel properties: {str(e)}")