from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class TimeAxisItem(pg.AxisItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        return [datetime.fromtimestamp(val).strftime('%H:%M:%S') for val in values]

class TrendViewFeature:
    def __init__(self, parent, db, project_name, channel=None, model_name=None, console=None):
        self.parent = parent
        self.db = db
        self.project_name = project_name
        self.model_name = model_name
        self.console = console
        self.scaling_factor = 3.3 / 65535.0
        self.display_window_seconds = 60.0
        self.channel_name = channel
        self.channel = self.resolve_channel_index(channel) if channel is not None else None
        self.sample_rate = None
        self.plot_data = []
        self.user_interacted = False
        self.last_right_limit = None
        self.last_frame_index = -1
        self.widget = None
        self.initUI()

    def resolve_channel_index(self, channel):
        try:
            if isinstance(channel, str):
                project_data = self.db.get_project_data(self.project_name) if self.db else {}
                models = project_data.get("models", [])
                model_found = False
                for m_data in models:
                    if m_data.get("name") == self.model_name:
                        model_found = True
                        channels = m_data.get("channels", [])
                        for idx, ch in enumerate(channels):
                            if ch.get("channelName") == channel:
                                logging.debug(f"Resolved channel {channel} to index {idx + 1} in model {self.model_name}")
                                return idx + 1
                        logging.warning(f"Channel {channel} not found in model {self.model_name}. Available channels: {[ch.get('channelName') for ch in channels]}")
                        if self.console:
                            self.console.append_to_console(f"Warning: Channel {channel} not found in model {self.model_name}")
                        return None
                if not model_found:
                    logging.warning(f"Model {self.model_name} not found in project {self.project_name}")
                    if self.console:
                        self.console.append_to_console(f"Warning: Model {self.model_name} not found in project {self.project_name}")
                    return None
            elif isinstance(channel, int):
                if channel >= 0:
                    return channel + 1
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
        layout = QVBoxLayout()
        self.widget.setLayout(layout)

        display_channel = self.channel_name if self.channel_name else f"Channel_{self.channel}" if self.channel else "Unknown"
        self.label = QLabel(f"Trend View for Model: {self.model_name or 'Unknown'}, Channel: {display_channel}")
        layout.addWidget(self.label)

        self.plot_widget = pg.PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')})
        self.plot_widget.setTitle(f"Trend for {self.model_name or 'Unknown'} - {display_channel}")
        self.plot_widget.setLabel('left', 'Direct (Peak-to-Peak Voltage, V)')
        self.plot_widget.setLabel('bottom', 'Time (hh:mm:ss)')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setBackground('w')
        self.plot_widget.setXRange(-self.display_window_seconds, 0, padding=0.02)
        self.plot_widget.enableAutoRange('y', True)
        layout.addWidget(self.plot_widget)

        self.curve = self.plot_widget.plot(pen=pg.mkPen('b', width=1))
        self.curve.setSymbol('o')
        self.curve.setSymbolSize(5)

        self.plot_widget.scene().sigMouseClicked.connect(self.on_mouse_interaction)
        self.plot_widget.getViewBox().sigRangeChangedManually.connect(self.on_range_changed)

    def on_mouse_interaction(self, event):
        self.user_interacted = True

    def on_range_changed(self, view_box, ranges):
        self.user_interacted = True
        self.last_right_limit = ranges[0][1]

    def get_widget(self):
        return self.widget

    def on_data_received(self, tag_name, model_name, values, sample_rate, frame_index):
        if self.model_name != model_name or self.channel is None:
            return

        try:
            if frame_index != self.last_frame_index + 1 and self.last_frame_index != -1:
                logging.warning(f"Non-sequential frame index: expected {self.last_frame_index + 1}, got {frame_index}")
                if self.console:
                    self.console.append_to_console(f"Warning: Non-sequential frame index: expected {self.last_frame_index + 1}, got {frame_index}")
            self.last_frame_index = frame_index

            channel_idx = self.channel - 1
            if not values or len(values) <= channel_idx:
                logging.warning(f"Invalid data: {len(values)} channels, expected at least {channel_idx + 1}, frame {frame_index}")
                if self.console:
                    self.console.append_to_console(f"Invalid data: {len(values)} channels, expected at least {channel_idx + 1}, frame {frame_index}")
                return

            main_channels = len(values) - 2 if len(values) >= 2 else len(values)
            if main_channels < 1 or channel_idx >= main_channels:
                logging.warning(f"Channel index {self.channel} out of range for {main_channels} main channels, frame {frame_index}")
                if self.console:
                    self.console.append_to_console(f"Channel index {self.channel} out of range for {main_channels} channels, frame {frame_index}")
                return

            self.sample_rate = sample_rate
            channel_data = np.array(values[channel_idx], dtype=np.float32) * self.scaling_factor
            trigger_data = np.array(values[-1], dtype=np.float32) if len(values) >= 2 else np.zeros_like(channel_data)

            trigger_indices = np.where(trigger_data == 1)[0].tolist()
            min_distance_between_triggers = 5
            filtered_trigger_indices = [trigger_indices[0]] if trigger_indices else []
            for i in range(1, len(trigger_indices)):
                if trigger_indices[i] - filtered_trigger_indices[-1] >= min_distance_between_triggers:
                    filtered_trigger_indices.append(trigger_indices[i])

            if len(filtered_trigger_indices) < 2:
                logging.warning(f"Not enough trigger points detected, frame {frame_index}")
                if self.console:
                    self.console.append_to_console(f"Not enough trigger points detected, frame {frame_index}")
                return

            direct_values = []
            for i in range(len(filtered_trigger_indices) - 1):
                start_idx = filtered_trigger_indices[i]
                end_idx = filtered_trigger_indices[i + 1]
                if end_idx <= start_idx:
                    continue
                segment_data = channel_data[start_idx:end_idx]
                if len(segment_data) == 0:
                    continue
                peak_to_peak = segment_data.max() - segment_data.min()
                direct_values.append(peak_to_peak)

            if not direct_values:
                logging.warning(f"No valid segments for peak-to-peak calculation, frame {frame_index}")
                if self.console:
                    self.console.append_to_console(f"No valid segments for calculation, frame {frame_index}")
                return

            direct_average = np.mean(direct_values)
            timestamp = datetime.now().timestamp()
            self.plot_data.append((timestamp, direct_average))

            self.trim_old_data()
            self.update_plot()

            logging.debug(f"Processed TrendView for {tag_name}, Channel {self.channel_name or self.channel}: Direct value {direct_average:.4f} at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}, frame {frame_index}")
            if self.console:
                self.console.append_to_console(f"{tag_name}: Direct={direct_average:.4f} V at {datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')}, frame {frame_index}")

        except Exception as e:
            logging.error(f"Data processing error for channel {self.channel_name or self.channel}, frame {frame_index}: {e}")
            if self.console:
                self.console.append_to_console(f"Data processing error for channel {self.channel_name or self.channel}, frame {frame_index}: {e}")

    def trim_old_data(self):
        now = datetime.now().timestamp()
        self.plot_data = [(t, v) for t, v in self.plot_data if (now - t) <= self.display_window_seconds]

    def update_plot(self):
        if not self.plot_data:
            return

        timestamps, voltages = zip(*self.plot_data)
        timestamps = np.array(timestamps)
        voltages = np.array(voltages)

        if self.user_interacted and self.last_right_limit is not None:
            max_time = self.last_right_limit
            min_time = max_time - self.display_window_seconds
        else:
            max_time = timestamps.max() if len(timestamps) > 0 else datetime.now().timestamp()
            min_time = max_time - self.display_window_seconds
            if len(timestamps) > 0 and (timestamps.max() - timestamps.min()) < self.display_window_seconds:
                min_time = timestamps.min()

        plot_width = self.plot_widget.width() or 600
        total_span = max_time - min_time
        padding_time = (40.0 / plot_width) * total_span

        self.plot_widget.setXRange(min_time, max_time + padding_time, padding=0.0)
        if len(voltages) > 0:
            min_y = voltages.min() * 0.9
            max_y = voltages.max() * 1.1
            self.plot_widget.setYRange(min_y, max_y, padding=0.02)

        self.curve.setData(timestamps, voltages)