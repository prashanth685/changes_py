from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QHBoxLayout
from PyQt5.QtCore import QObject, pyqtSignal
import pyqtgraph as pg
import numpy as np
import logging
from datetime import datetime

class OrbitFeature(QObject):
    primary_channel_changed = pyqtSignal(int)
    secondary_channel_changed = pyqtSignal(int)

    def __init__(self, parent, db, project_name, channel=None, model_name=None, console=None, channel_count=None):
        super().__init__(parent)
        self.parent = parent
        self.db = db
        self.project_name = project_name
        self.selected_channel = channel
        self.model_name = model_name
        self.console = console
        self.channel_count = channel_count if channel_count is not None else 0
        self.widget = None
        self.plot_widgets = []
        self.plot_items = []
        self.data_plots = []
        self.time_plot_widgets = []
        self.time_plots = []
        self.channel_data = []
        self.primary_channel = 0
        self.secondary_channel = 1
        self.sample_rate = None
        self.samples_per_channel = None
        self.current_time = 0.0
        self.available_channels = []
        self.is_updating = False
        self.last_frame_index = -1
        self.window_seconds = 1.0
        self.initUI()
        self.parent.tree_view.model_selected.connect(self.update_model)
        self.parent.tree_view.channel_selected.connect(self.update_channel)
        self.load_channel_data()
        if self.console:
            self.console.append_to_console(
                f"Initialized OrbitFeature for {self.model_name or 'No Model'}/{self.selected_channel or 'No Channel'} "
                f"with {self.channel_count} channels"
            )

    def initUI(self):
        self.widget = QWidget()
        main_layout = QVBoxLayout()
        self.widget.setLayout(main_layout)

        primary_label = QLabel("Primary Channel:")
        self.primary_combo = QComboBox()
        self.primary_combo.setStyleSheet("""
            QComboBox {
                font-size: 16px;
                padding: 5px;
                background-color: white;
            }
            QComboBox QAbstractItemView {
                font-size: 16px;
                background-color: white;
            }
        """)
        self.primary_combo.currentIndexChanged.connect(self.on_primary_combo_changed)

        secondary_label = QLabel("Secondary Channel:")
        self.secondary_combo = QComboBox()
        self.secondary_combo.setStyleSheet("""
            QComboBox {
                font-size: 16px;
                padding: 5px;
                background-color: white;
            }
            QComboBox QAbstractItemView {
                font-size: 16px;
                background-color: white;
            }
        """)
        self.secondary_combo.currentIndexChanged.connect(self.on_secondary_combo_changed)

        combo_layout = QHBoxLayout()
        combo_layout.addWidget(primary_label)
        combo_layout.addWidget(self.primary_combo)
        combo_layout.addWidget(secondary_label)
        combo_layout.addWidget(self.secondary_combo)
        main_layout.addLayout(combo_layout)

        self.plot_layout = QHBoxLayout()
        main_layout.addLayout(self.plot_layout)

        self.create_plots()

    def update_model(self, model_name):
        if self.model_name != model_name:
            self.model_name = model_name
            self.selected_channel = None
            self.primary_channel = 0
            self.secondary_channel = 1
            self.channel_data = []
            self.available_channels = []
            self.load_channel_data()
            if self.console:
                self.console.append_to_console(f"OrbitFeature: Updated model to {model_name}")

    def update_channel(self, model_name, channel_name):
        if self.model_name == model_name and channel_name in self.available_channels:
            self.update_selected_channel(channel_name)
            if self.console:
                self.console.append_to_console(f"OrbitFeature: Updated channel to {channel_name}")
        else:
            if self.console:
                self.console.append_to_console(
                    f"OrbitFeature: Skipped channel update - model mismatch ({self.model_name} vs {model_name}) "
                    f"or channel {channel_name} not in {self.available_channels}"
                )

    def load_channel_data(self):
        try:
            if not self.project_name or not self.model_name:
                if self.console:
                    self.console.append_to_console("OrbitFeature: No project or model selected")
                self.channel_count = 0
                self.available_channels = []
                self.channel_data = []
                self.primary_combo.clear()
                self.secondary_combo.clear()
                self.clear_plots()
                return
            if not self.db.is_connected():
                self.db.reconnect()
            project_data = self.db.get_project_data(self.project_name)
            if not project_data:
                if self.console:
                    self.console.append_to_console(f"OrbitFeature: Project {self.project_name} not found")
                self.channel_count = 0
                self.available_channels = []
                self.channel_data = []
                self.primary_combo.clear()
                self.secondary_combo.clear()
                self.clear_plots()
                return
            model = next((m for m in project_data.get("models", []) if m["name"] == self.model_name), None)
            if not model:
                if self.console:
                    self.console.append_to_console(f"OrbitFeature: Model {self.model_name} not found")
                self.channel_count = 0
                self.available_channels = []
                self.channel_data = []
                self.primary_combo.clear()
                self.secondary_combo.clear()
                self.clear_plots()
                return
            self.available_channels = [ch.get("channelName", f"Channel_{i+1}") for i, ch in enumerate(model.get("channels", []))]
            self.channel_count = len(self.available_channels)
            self.channel_data = [[] for _ in range(self.channel_count)]
            self.primary_combo.clear()
            self.secondary_combo.clear()
            self.primary_combo.addItems(self.available_channels)
            self.secondary_combo.addItems(self.available_channels)
            if self.selected_channel and self.selected_channel in self.available_channels:
                idx = self.available_channels.index(self.selected_channel)
                self.primary_channel = idx
                self.secondary_channel = (idx + 1) % self.channel_count if self.channel_count > 1 else idx
            else:
                self.primary_channel = 0
                self.secondary_channel = 1 if self.channel_count > 1 else 0
                self.selected_channel = self.available_channels[self.primary_channel] if self.available_channels else None
            self.primary_combo.setCurrentIndex(self.primary_channel)
            self.secondary_combo.setCurrentIndex(self.secondary_channel)
            if self.console:
                self.console.append_to_console(
                    f"OrbitFeature: Loaded {self.channel_count} channels. "
                    f"Set primary channel to index {self.primary_channel} ({self.available_channels[self.primary_channel] if self.available_channels else 'None'}), "
                    f"secondary to index {self.secondary_channel} ({self.available_channels[self.secondary_channel] if self.channel_count > 1 else 'None'})"
                )
            self.create_plots()
        except Exception as e:
            if self.console:
                self.console.append_to_console(f"OrbitFeature: Error loading channel data: {str(e)}")
            logging.error(f"OrbitFeature: Error loading channel data: {str(e)}")
            self.channel_count = 0
            self.available_channels = []
            self.channel_data = []
            self.primary_combo.clear()
            self.secondary_combo.clear()
            self.clear_plots()

    def get_channel_index(self, channel_name):
        try:
            if not channel_name:
                if self.console:
                    self.console.append_to_console("OrbitFeature: get_channel_index: No channel name provided")
                return None
            if channel_name in self.available_channels:
                idx = self.available_channels.index(channel_name)
                if self.console:
                    self.console.append_to_console(f"OrbitFeature: get_channel_index: Found channel {channel_name} at index {idx}")
                return idx
            if self.console:
                self.console.append_to_console(f"OrbitFeature: get_channel_index: Channel {channel_name} not found in {self.available_channels}")
            return None
        except Exception as e:
            if self.console:
                self.console.append_to_console(f"OrbitFeature: Error in get_channel_index for {channel_name}: {str(e)}")
            logging.error(f"OrbitFeature: Error in get_channel_index for {channel_name}: {str(e)}")
            return None

    def on_primary_combo_changed(self, index):
        if self.is_updating:
            return
        self.is_updating = True
        try:
            if 0 <= index < self.channel_count:
                self.primary_channel = index
                self.selected_channel = self.available_channels[index]
                self.primary_channel_changed.emit(index)
                self.update_plot_labels()
                self.update_plots()
                if self.console:
                    self.console.append_to_console(
                        f"OrbitFeature: Selected primary channel: {self.available_channels[index]} (index {index})"
                    )
            else:
                if self.console:
                    self.console.append_to_console(f"OrbitFeature: Invalid primary channel index {index}")
        finally:
            self.is_updating = False

    def on_secondary_combo_changed(self, index):
        if self.is_updating:
            return
        self.is_updating = True
        try:
            if 0 <= index < self.channel_count:
                self.secondary_channel = index
                self.secondary_channel_changed.emit(index)
                self.update_plot_labels()
                self.update_plots()
                if self.console:
                    self.console.append_to_console(
                        f"OrbitFeature: Selected secondary channel: {self.available_channels[index]} (index {index})"
                    )
            else:
                if self.console:
                    self.console.append_to_console(f"OrbitFeature: Invalid secondary channel index {index}")
        finally:
            self.is_updating = False

    def create_plots(self):
        if self.plot_widgets or self.time_plot_widgets:
            return

        while self.plot_layout.count():
            item = self.plot_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                item.layout().deleteLater()

        if not self.available_channels or self.channel_count == 0:
            if self.console:
                self.console.append_to_console("OrbitFeature: No channels available, cannot create plots")
            return

        plot_widget = pg.PlotWidget()
        plot_widget.setBackground('w')
        plot_widget.setFixedSize(500, 500)
        self.plot_layout.addWidget(plot_widget)
        plot_item = plot_widget.getPlotItem()
        plot_item.setTitle(f"Orbit Plot (Ch {self.available_channels[self.secondary_channel]} vs Ch {self.available_channels[self.primary_channel]})")
        plot_item.setLabel('bottom', f"Channel {self.available_channels[self.primary_channel]}")
        plot_item.setLabel('left', f"Channel {self.available_channels[self.secondary_channel]}")
        plot_item.showGrid(x=True, y=True)
        plot_item.setAspectLocked(True)
        plot_item.enableAutoRange('xy', True)  # Enable auto-ranging for orbit plot
        data_plot = plot_item.plot(pen=pg.mkPen('b', width=2))
        self.plot_widgets.append(plot_widget)
        self.plot_items.append(plot_item)
        self.data_plots.append(data_plot)

        time_colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', '#FF4500', '#32CD32', '#00CED1', '#FFD700',
                       '#FF69B4', '#8A2BE2', '#FF6347', '#20B2AA', '#ADFF2F', '#9932CC', '#FF7F50', '#00FA9A', '#9400D3']
        for ch in [self.primary_channel, self.secondary_channel]:
            if ch < self.channel_count:
                time_plot_widget = pg.PlotWidget()
                time_plot_widget.setBackground('w')
                time_plot_widget.setFixedSize(500, 500)
                self.plot_layout.addWidget(time_plot_widget)
                time_plot_item = time_plot_widget.getPlotItem()
                time_plot_item.setTitle(f"Channel {self.available_channels[ch]} Time Domain")
                time_plot_item.setLabel('bottom', "Time (s)")
                time_plot_item.setLabel('left', f"Channel {self.available_channels[ch]} Value")
                time_plot_item.showGrid(x=True, y=True)
                time_plot_item.setXRange(self.current_time - self.window_seconds, self.current_time, padding=0.02)
                time_plot_item.enableAutoRange('y', True)
                time_plot = time_plot_item.plot(pen=pg.mkPen(time_colors[ch % len(time_colors)], width=2))
                self.time_plot_widgets.append(time_plot_widget)
                self.time_plots.append(time_plot)

        if self.console:
            self.console.append_to_console(
                f"OrbitFeature: Created plots for primary channel {self.available_channels[self.primary_channel]} (index {self.primary_channel}), "
                f"secondary channel {self.available_channels[self.secondary_channel]} (index {self.secondary_channel}), "
                f"plot widgets: {len(self.plot_widgets)} orbit, {len(self.time_plot_widgets)} time"
            )

        self.update_plots()

    def update_plot_labels(self):
        if self.plot_items and self.time_plot_widgets:
            self.plot_items[0].setTitle(f"Orbit Plot (Ch {self.available_channels[self.secondary_channel]} vs Ch {self.available_channels[self.primary_channel]})")
            self.plot_items[0].setLabel('bottom', f"Channel {self.available_channels[self.primary_channel]}")
            self.plot_items[0].setLabel('left', f"Channel {self.available_channels[self.secondary_channel]}")
            for i, ch in enumerate([self.primary_channel, self.secondary_channel]):
                if i < len(self.time_plot_widgets) and ch < self.channel_count:
                    self.time_plot_widgets[i].getPlotItem().setTitle(f"Channel {self.available_channels[ch]} Time Domain")
                    self.time_plot_widgets[i].getPlotItem().setLabel('left', f"Channel {self.available_channels[ch]} Value")
            if self.console:
                self.console.append_to_console("OrbitFeature: Updated plot labels")

    def clear_plots(self):
        if self.data_plots:
            self.data_plots[0].clear()
        for plot in self.time_plots:
            plot.clear()
        for widget in self.plot_widgets + self.time_plot_widgets:
            widget.getPlotItem().enableAutoRange('xy', True)
            widget.getViewBox().update()
        if self.console:
            self.console.append_to_console("OrbitFeature: Cleared all plots")

    def update_plots(self):
        if not self.data_plots or not self.time_plots:
            if self.console:
                self.console.append_to_console("OrbitFeature: No plots available for update")
            return

        data_lengths = [len(d) if isinstance(d, (list, np.ndarray)) else 0 for d in self.channel_data]
        if self.console:
            self.console.append_to_console(
                f"OrbitFeature: Updating plots with channel data lengths: {data_lengths}, "
                f"primary: {self.primary_channel}, secondary: {self.secondary_channel}"
            )

        if (self.primary_channel >= len(self.channel_data) or 
            self.secondary_channel >= len(self.channel_data) or
            len(self.channel_data[self.primary_channel]) == 0 or 
            len(self.channel_data[self.secondary_channel]) == 0):
            if self.console:
                self.console.append_to_console(
                    f"OrbitFeature: Cannot update plots - invalid channel indices or empty data "
                    f"(primary: {self.primary_channel}, secondary: {self.secondary_channel}, "
                    f"data lengths: {data_lengths})"
                )
            self.clear_plots()
            return

        x_data = np.array(self.channel_data[self.primary_channel])
        y_data = np.array(self.channel_data[self.secondary_channel])
        if x_data.size > 0 and y_data.size > 0 and x_data.size == y_data.size:
            self.data_plots[0].clear()
            self.data_plots[0].setData(x_data, y_data)
            self.plot_items[0].enableAutoRange('xy', True)  # Auto-range for orbit plot
            self.plot_items[0].getViewBox().update()
            if self.console:
                self.console.append_to_console(
                    f"OrbitFeature: Updated orbit plot with {x_data.size} samples for "
                    f"Ch {self.available_channels[self.primary_channel]} vs Ch {self.available_channels[self.secondary_channel]}, "
                    f"data sample: {x_data[:5].tolist()}"
                )
        else:
            self.data_plots[0].clear()
            if self.console:
                self.console.append_to_console(
                    f"OrbitFeature: Cannot update orbit plot - invalid data sizes (x: {x_data.size}, y: {y_data.size})"
                )

        if self.sample_rate and self.samples_per_channel:
            time = np.linspace(self.current_time - self.samples_per_channel / self.sample_rate, 
                             self.current_time, self.samples_per_channel, endpoint=False)
            for i, ch in enumerate([self.primary_channel, self.secondary_channel]):
                if i < len(self.time_plots) and ch < len(self.channel_data) and len(self.channel_data[ch]) > 0:
                    ch_data = np.array(self.channel_data[ch])
                    if ch_data.size > 0:
                        self.time_plots[i].clear()
                        self.time_plots[i].setData(time, ch_data)
                        self.time_plot_widgets[i].getPlotItem().setXRange(
                            self.current_time - self.window_seconds, self.current_time, padding=0.02
                        )
                        self.time_plot_widgets[i].getPlotItem().enableAutoRange('y', True)
                        self.time_plot_widgets[i].getViewBox().update()
                        if self.console:
                            self.console.append_to_console(
                                f"OrbitFeature: Updated time plot {i} for channel {self.available_channels[ch]} with {ch_data.size} samples, "
                                f"data sample: {ch_data[:5].tolist()}"
                            )
                    else:
                        self.time_plots[i].clear()
                        if self.console:
                            self.console.append_to_console(
                                f"OrbitFeature: Cannot update time plot {i} - empty data for channel {self.available_channels[ch]}"
                            )
        else:
            for plot in self.time_plots:
                plot.clear()
            if self.console:
                self.console.append_to_console(
                    f"OrbitFeature: Cannot update time plots - missing sample rate ({self.sample_rate}) or sample count ({self.samples_per_channel})"
                )

    def on_data_received(self, tag_name, model_name, values, sample_rate, frame_index):
        if self.model_name != model_name:
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
                        f"OrbitFeature: Received {len(values)} channels, expected at least {self.channel_count}, frame {frame_index}"
                    )
                return
            self.sample_rate = sample_rate
            self.samples_per_channel = len(values[0])
            self.current_time = datetime.now().timestamp()
            for i in range(min(self.channel_count, len(values))):
                if len(values[i]) == self.samples_per_channel:
                    self.channel_data[i] = np.array(values[i])
                else:
                    if self.console:
                        self.console.append_to_console(
                            f"OrbitFeature: Channel {i} has {len(values[i])} samples, expected {self.samples_per_channel}, frame {frame_index}"
                        )
                    return
            if self.console:
                self.console.append_to_console(
                    f"OrbitFeature ({self.model_name}): Received {self.samples_per_channel} samples for {self.channel_count} channels, "
                    f"data lengths: {[len(d) for d in self.channel_data]}, frame {frame_index}"
                )
            self.update_plots()
        except Exception as e:
            if self.console:
                self.console.append_to_console(f"OrbitFeature: Error processing data, frame {frame_index}: {str(e)}")
            logging.error(f"OrbitFeature: Error processing data, frame {frame_index}: {str(e)}")

    def update_selected_channel(self, channel_name):
        if self.is_updating:
            return
        self.is_updating = True
        try:
            channel_idx = self.get_channel_index(channel_name)
            if channel_idx is not None and channel_idx < self.channel_count:
                self.selected_channel = channel_name
                self.primary_channel = channel_idx
                self.secondary_channel = (channel_idx + 1) % self.channel_count if self.channel_count > 1 else channel_idx
                self.primary_combo.setCurrentIndex(self.primary_channel)
                self.secondary_combo.setCurrentIndex(self.secondary_channel)
                self.primary_channel_changed.emit(self.primary_channel)
                self.secondary_channel_changed.emit(self.secondary_channel)
                self.update_plot_labels()
                self.update_plots()
                if self.console:
                    self.console.append_to_console(
                        f"OrbitFeature: Updated selected channel to {channel_name} "
                        f"(primary index {self.primary_channel}, secondary index {self.secondary_channel})"
                    )
            else:
                if self.console:
                    self.console.append_to_console(
                        f"OrbitFeature: Channel {channel_name} not found or invalid index, keeping current selection"
                    )
        except Exception as e:
            if self.console:
                self.console.append_to_console(f"OrbitFeature: Error updating channel {channel_name}: {str(e)}")
            logging.error(f"OrbitFeature: Error updating channel {channel_name}: {str(e)}")
        finally:
            self.is_updating = False

    def get_widget(self):
        return self.widget

    def cleanup(self):
        for plot_widget in self.plot_widgets + self.time_plot_widgets:
            plot_widget.deleteLater()
        self.plot_widgets = []
        self.time_plot_widgets = []
        self.data_plots = []
        self.time_plots = []
        self.channel_data = []
        self.channel_count = 0
        if self.console:
            self.console.append_to_console("OrbitFeature: Cleaned up resources")

    def refresh_channel_properties(self):
        self.load_channel_data()
        if self.console:
            self.console.append_to_console("OrbitFeature: Refreshed channel properties")