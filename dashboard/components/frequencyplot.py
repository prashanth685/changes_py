from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QSlider, QWidget, QHBoxLayout, QApplication
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QMouseEvent
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from pymongo import MongoClient
import numpy as np
import datetime
import logging
from database import Database
import uuid
import sys
from datetime import datetime as dt

class FrequencyPlot(QDialog):
    time_range_selected = pyqtSignal(dict)

    def __init__(self, parent=None, project_name=None, model_name=None, filename=None, start_time=None, end_time=None, email="user@example.com"):
        super().__init__(parent)
        self.setWindowTitle(f"Frequency Plot - {filename}")
        self.setMinimumSize(800, 600)
        self.project_name = project_name
        self.model_name = model_name
        self.filename = filename
        self.start_time = self.parse_time(start_time) if start_time else None
        self.end_time = self.parse_time(end_time) if end_time else None
        self.email = email
        self.db = Database(connection_string="mongodb://localhost:27017/", email=email)
        self.current_records = []
        self.filtered_records = []
        self.lower_time_percentage = 0
        self.upper_time_percentage = 100
        self.time_data = None
        self.frequency_data = None
        self.selected_record = None
        self.is_crosshair_visible = False
        self.is_crosshair_locked = False
        self.locked_crosshair_position = None
        self.last_mouse_move = datetime.datetime.now()
        self.mouse_move_debounce_ms = 50
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.filter_and_plot_data)
        self.debounce_delay = 200  # ms
        self.crosshair_state_saved = False
        self.saved_crosshair_visible = False
        self.saved_crosshair_locked = False
        self.saved_crosshair_position = None
        self.is_dragging_range = False
        self.drag_start_x = 0
        self.initial_lower_x = 0
        self.initial_upper_x = 0
        self.initUI()
        self.initialize_data()

    def parse_time(self, time_str):
        try:
            return datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        except Exception as e:
            logging.error(f"Error parsing time {time_str}: {str(e)}")
            return None

    def initUI(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)

        # Title
        self.title_label = QLabel(f"Frequency Analysis for {self.filename}")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        self.layout.addWidget(self.title_label)

        # Matplotlib canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.layout.addWidget(self.canvas)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.canvas.mpl_connect('axes_leave_event', self.on_mouse_leave)

        # Range slider widget
        self.slider_widget = QWidget()
        self.slider_layout = QHBoxLayout()
        self.slider_widget.setLayout(self.slider_layout)
        self.slider_widget.setFixedHeight(50)

        self.start_label = QLabel("Start: ")
        self.start_label.setStyleSheet("font-size: 14px; color: #333;")
        self.slider_layout.addWidget(self.start_label)

        self.start_slider = QSlider(Qt.Horizontal)
        self.start_slider.setMinimum(0)
        self.start_slider.setMaximum(100)
        self.start_slider.setValue(0)
        self.start_slider.valueChanged.connect(self.update_labels)
        self.slider_layout.addWidget(self.start_slider)

        self.end_label = QLabel("End: ")
        self.end_label.setStyleSheet("font-size: 14px; color: #333;")
        self.slider_layout.addWidget(self.end_label)

        self.end_slider = QSlider(Qt.Horizontal)
        self.end_slider.setMinimum(0)
        self.end_slider.setMaximum(100)
        self.end_slider.setValue(100)
        self.end_slider.valueChanged.connect(self.update_labels)
        self.slider_layout.addWidget(self.end_slider)

        self.layout.addWidget(self.slider_widget)

        # Range indicator (draggable)
        self.range_indicator = QPushButton("Drag Range")
        self.range_indicator.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #357abd; }
            QPushButton:pressed { background-color: #2c5d9b; }
        """)
        self.range_indicator.pressed.connect(self.start_range_drag)
        self.range_indicator.released.connect(self.stop_range_drag)
        self.slider_layout.addWidget(self.range_indicator)
        self.slider_widget.mouseMoveEvent = self.range_mouse_move

        # Select button
        self.select_button = QPushButton("Select")
        self.select_button.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #357abd; }
            QPushButton:pressed { background-color: #2c5d9b; }
        """)
        self.select_button.clicked.connect(self.select_button_click)
        self.layout.addWidget(self.select_button)

        self.layout.addStretch()
        self.setLayout(self.layout)
        self.update_labels()

    def initialize_data(self):
        try:
            self.current_records.clear()
            self.filtered_records.clear()
            self.fetch_all_records()
        except Exception as e:
            logging.error(f"Error initializing data: {str(e)}")
            self.show_message_box("Error initializing view.", "Error", "error")

    def fetch_all_records(self):
        try:
            query = {
                "filename": self.filename,
                "model_name": self.model_name,
                "project_name": self.project_name,
                "email": self.email
            }
            messages = self.db.get_timeview_messages(
                project_name=self.project_name,
                model_name=self.model_name,
                topic=None,
                filename=self.filename
            )
            if not messages:
                logging.info("No records found for selected recording.")
                self.show_message_box("No records found for selected recording.", "Information", "info")
                return

            self.current_records = sorted(messages, key=lambda x: x.get("frameIndex", 0))
            logging.info(f"Loaded {len(self.current_records)} records for {self.filename}")

            recording_duration = (self.end_time - self.start_time).total_seconds() / 60
            if recording_duration > 6:
                logging.info("Recording > 6 minutes, plotting initial 5%")
                self.lower_time_percentage = 0
                self.upper_time_percentage = 5
                self.start_slider.setValue(0)
                self.end_slider.setValue(5)
                self.update_labels()
                self.filter_records_by_frame_index_range()
                self.plot_frequency_data()
            else:
                self.filtered_records = self.current_records.copy()
                self.plot_frequency_data()
        except Exception as e:
            logging.error(f"Error fetching records: {str(e)}")
            self.show_message_box(f"Error fetching records: {str(e)}", "Error", "error")

    def filter_records_by_frame_index_range(self):
        try:
            if not self.current_records:
                self.filtered_records.clear()
                return

            min_frame_index = min(r.get("frameIndex", 0) for r in self.current_records)
            max_frame_index = max(r.get("frameIndex", 0) for r in self.current_records)
            total_frame_range = max_frame_index - min_frame_index

            start_frame_index = min_frame_index + int(total_frame_range * self.lower_time_percentage / 100.0)
            end_frame_index = min_frame_index + int(total_frame_range * self.upper_time_percentage / 100.0)

            start_frame_index = max(min_frame_index, start_frame_index)
            end_frame_index = min(max_frame_index, end_frame_index)

            self.filtered_records = [
                r for r in self.current_records
                if start_frame_index <= r.get("frameIndex", 0) <= end_frame_index
            ]
            self.filtered_records.sort(key=lambda x: x.get("frameIndex", 0))
            logging.info(f"Filtered {len(self.filtered_records)} records from {len(self.current_records)} for frame range {start_frame_index} to {end_frame_index}")
        except Exception as e:
            logging.error(f"Error filtering records: {str(e)}")
            self.filtered_records.clear()

    def save_crosshair_state(self):
        try:
            if self.is_crosshair_visible or self.is_crosshair_locked:
                self.saved_crosshair_visible = self.is_crosshair_visible
                self.saved_crosshair_locked = self.is_crosshair_locked
                self.saved_crosshair_position = self.locked_crosshair_position
                self.crosshair_state_saved = True
                logging.info(f"Crosshair state saved - Visible: {self.saved_crosshair_visible}, Locked: {self.saved_crosshair_locked}")
        except Exception as e:
            logging.error(f"Error saving crosshair state: {str(e)}")

    def restore_crosshair_state(self):
        try:
            if not self.crosshair_state_saved or self.time_data is None or self.frequency_data is None:
                return

            self.is_crosshair_visible = self.saved_crosshair_visible
            self.is_crosshair_locked = self.saved_crosshair_locked

            if self.saved_crosshair_locked and self.saved_crosshair_position:
                x, y = self.saved_crosshair_position
                if self.time_data.min() <= x <= self.time_data.max():
                    interpolated_y = self.interpolate_y_value(x)
                    self.locked_crosshair_position = (x, interpolated_y)
                    self.update_crosshair(self.locked_crosshair_position, is_locked=True)
                    logging.info("Restored locked crosshair state")
                else:
                    self.is_crosshair_visible = False
                    self.is_crosshair_locked = False
                    self.update_crosshair(None, is_visible=False)
                    logging.info("Locked crosshair position outside bounds, hiding")
            elif self.saved_crosshair_visible and self.saved_crosshair_position:
                x, y = self.saved_crosshair_position
                if self.time_data.min() <= x <= self.time_data.max():
                    interpolated_y = self.interpolate_y_value(x)
                    self.update_crosshair((x, interpolated_y), is_locked=False)
                    logging.info("Restored crosshair position within bounds")
                else:
                    self.is_crosshair_visible = False
                    self.update_crosshair(None, is_visible=False)
                    logging.info("Crosshair position outside bounds, hiding")

            self.crosshair_state_saved = False
        except Exception as e:
            logging.error(f"Error restoring crosshair state: {str(e)}")

    def plot_frequency_data(self):
        try:
            if not self.filtered_records:
                logging.info("No filtered records available for plotting")
                self.ax.clear()
                self.canvas.draw()
                return

            first_record = self.filtered_records[0]
            channel_count = first_record.get("numberOfChannels", 1)
            taco_channel_count = first_record.get("tacoChannelCount", 0)
            samples_per_channel = first_record.get("samplingRate", 1)

            if channel_count != 4 or taco_channel_count != 2:
                self.show_message_box(
                    f"Expected 4 main channels and 2 tacho channels, got {channel_count} main and {taco_channel_count} tacho channels.",
                    "Error", "error"
                )
                return

            plot_start_time = min(r.get("createdAt", self.start_time) for r in self.filtered_records)
            plot_end_time = max(r.get("createdAt", self.start_time) for r in self.filtered_records)
            plot_start_time = self.parse_time(plot_start_time) if isinstance(plot_start_time, str) else plot_start_time
            plot_end_time = self.parse_time(plot_end_time) if isinstance(plot_end_time, str) else plot_end_time
            plot_duration = (plot_end_time - plot_start_time).total_seconds()

            if plot_duration <= 0:
                plot_duration = 1.0

            records_with_message = [r for r in self.filtered_records if r.get("message") is not None]
            total_samples = len(records_with_message) * samples_per_channel
            main_channel_data_length = channel_count * samples_per_channel

            if total_samples == 0:
                self.show_message_box("No valid message data found in selected range.", "Information", "info")
                return

            max_points_to_plot = 100000
            needs_downsampling = total_samples > max_points_to_plot
            downsample_factor = int(np.ceil(total_samples / max_points_to_plot)) if needs_downsampling else 1

            self.frequency_data = []
            sample_index = 0

            for record in records_with_message:
                message = record.get("message", [])
                expected_message_length = main_channel_data_length + (taco_channel_count * samples_per_channel)
                if len(message) < expected_message_length:
                    logging.warning(f"Invalid message length in record FrameIndex {record.get('frameIndex')}: expected {expected_message_length}, got {len(message)}")
                    continue
                start_index = main_channel_data_length
                self.frequency_data.extend(message[start_index:start_index + samples_per_channel])
                sample_index += samples_per_channel

            self.frequency_data = np.array(self.frequency_data[:sample_index])
            total_samples = len(self.frequency_data)

            self.time_data = np.linspace(
                mdates.date2num(plot_start_time),
                mdates.date2num(plot_end_time),
                total_samples
            )

            if needs_downsampling and total_samples > 0:
                self.frequency_data = self.downsample_array(self.frequency_data, downsample_factor)
                self.time_data = self.downsample_array(self.time_data, downsample_factor)
                logging.info(f"Downsampled data to {len(self.time_data)} points")

            self.ax.clear()
            self.ax.plot(self.time_data, self.frequency_data, color='orange', linewidth=1)
            self.ax.set_ylabel("Frequency Value")
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            self.ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            self.figure.autofmt_xdate()

            if self.time_data.size > 0:
                self.ax.set_xlim(self.time_data.min(), self.time_data.max())
                self.ax.set_ylim(np.min(self.frequency_data), np.max(self.frequency_data))

            self.save_crosshair_state()
            self.initialize_crosshair()
            self.restore_crosshair_state()

            self.canvas.draw()
            logging.info(f"Plotted frequency data for {self.filename}: {total_samples} samples, duration {plot_duration:.2f} seconds")
        except Exception as e:
            logging.error(f"Error plotting frequency data: {str(e)}")
            self.show_message_box(f"Error plotting data: {str(e)}", "Error", "error")

    def initialize_crosshair(self):
        try:
            if self.time_data is None or self.frequency_data is None or len(self.time_data) == 0:
                return

            self.ax.lines = [line for line in self.ax.lines if not hasattr(line, 'is_crosshair')]
            mid_time = (self.time_data.min() + self.time_data.max()) / 2
            mid_freq = (self.frequency_data.min() + self.frequency_data.max()) / 2

            self.crosshair_vline = self.ax.axvline(mid_time, color='gray', linestyle='--', linewidth=1.5)
            self.crosshair_hline = self.ax.axhline(mid_freq, color='gray', linestyle='--', linewidth=1.5)
            self.crosshair_vline.is_crosshair = True
            self.crosshair_hline.is_crosshair = True
            self.crosshair_vline.set_visible(False)
            self.crosshair_hline.set_visible(False)

            self.intersection_point, = self.ax.plot([mid_time], [mid_freq], 'ro', markersize=6)
            self.intersection_point.set_visible(False)

            self.update_crosshair_style()
            logging.info("Crosshair and intersection point initialized")
        except Exception as e:
            logging.error(f"Error initializing crosshair: {str(e)}")

    def update_crosshair_style(self):
        try:
            if not hasattr(self, 'crosshair_vline') or not hasattr(self, 'crosshair_hline'):
                return

            if self.is_crosshair_locked:
                self.crosshair_vline.set_linewidth(2)
                self.crosshair_hline.set_linewidth(2)
                self.crosshair_vline.set_color('gray')
                self.crosshair_hline.set_color('gray')
                self.intersection_point.set_color('green')
                self.intersection_point.set_markersize(8)
            else:
                self.crosshair_vline.set_linewidth(1.5)
                self.crosshair_hline.set_linewidth(1.5)
                self.crosshair_vline.set_color('gray')
                self.crosshair_hline.set_color('gray')
                self.intersection_point.set_color('red')
                self.intersection_point.set_markersize(6)
        except Exception as e:
            logging.error(f"Error updating crosshair style: {str(e)}")

    def update_crosshair(self, position, is_locked=False, is_visible=True):
        try:
            if not hasattr(self, 'crosshair_vline') or not hasattr(self, 'crosshair_hline'):
                return

            if not is_visible:
                self.crosshair_vline.set_visible(False)
                self.crosshair_hline.set_visible(False)
                self.intersection_point.set_visible(False)
                return

            x, y = position
            self.crosshair_vline.set_xdata([x, x])
            self.crosshair_hline.set_ydata([y, y])
            self.crosshair_vline.set_visible(True)
            self.crosshair_hline.set_visible(True)
            self.intersection_point.set_data([x], [y])
            self.intersection_point.set_visible(True)
            self.update_crosshair_style()
            self.canvas.draw()
            logging.info(f"Crosshair updated at ({x:.2f}, {y:.2f}), Locked: {is_locked}")
        except Exception as e:
            logging.error(f"Error updating crosshair: {str(e)}")

    def interpolate_y_value(self, target_x):
        try:
            if self.time_data is None or self.frequency_data is None or len(self.time_data) == 0:
                return 0

            if target_x <= self.time_data[0]:
                return self.frequency_data[0]
            if target_x >= self.time_data[-1]:
                return self.frequency_data[-1]

            left_index = np.searchsorted(self.time_data, target_x, side='left') - 1
            if left_index < 0:
                closest_index = np.argmin(np.abs(self.time_data - target_x))
                return self.frequency_data[closest_index]

            right_index = left_index + 1
            x1, x2 = self.time_data[left_index], self.time_data[right_index]
            y1, y2 = self.frequency_data[left_index], self.frequency_data[right_index]
            interpolated_y = y1 + (y2 - y1) * (target_x - x1) / (x2 - x1)
            return interpolated_y
        except Exception as e:
            logging.error(f"Error interpolating Y value: {str(e)}")
            return 0

    def on_mouse_move(self, event):
        if (datetime.datetime.now() - self.last_mouse_move).total_seconds() * 1000 < self.mouse_move_debounce_ms:
            return
        if self.is_crosshair_locked or not event.inaxes:
            return

        try:
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                self.is_crosshair_visible = False
                self.update_crosshair(None, is_visible=False)
                self.canvas.draw()
                return

            if self.time_data.min() <= x <= self.time_data.max():
                interpolated_y = self.interpolate_y_value(x)
                self.is_crosshair_visible = True
                self.update_crosshair((x, interpolated_y), is_locked=False)
                logging.info(f"Crosshair moved to Time: {mdates.num2date(x):%H:%M:%S.%f}, Frequency: {interpolated_y:.2f}")
            else:
                self.is_crosshair_visible = False
                self.update_crosshair(None, is_visible=False)
            self.last_mouse_move = datetime.datetime.now()
        except Exception as e:
            logging.error(f"Error in mouse move: {str(e)}")

    def on_mouse_click(self, event):
        if not event.inaxes or event.button != 1:
            return

        try:
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return

            if self.time_data.min() <= x <= self.time_data.max():
                interpolated_y = self.interpolate_y_value(x)
                if self.is_crosshair_locked:
                    self.is_crosshair_locked = False
                    logging.info("Crosshair unlocked")
                else:
                    self.is_crosshair_locked = True
                    self.locked_crosshair_position = (x, interpolated_y)
                    logging.info(f"Crosshair locked at Time: {mdates.num2date(x):%H:%M:%S.%f}, Frequency: {interpolated_y:.2f}")
                self.update_crosshair(self.locked_crosshair_position or (x, interpolated_y), is_locked=self.is_crosshair_locked)
        except Exception as e:
            logging.error(f"Error in mouse click: {str(e)}")

    def on_mouse_leave(self, event):
        try:
            if self.is_crosshair_visible and not self.is_crosshair_locked:
                self.is_crosshair_visible = False
                self.update_crosshair(None, is_visible=False)
                self.canvas.draw()
                logging.info("Crosshair hidden - mouse left plot area")
        except Exception as e:
            logging.error(f"Error in mouse leave: {str(e)}")

    def start_range_drag(self):
        self.is_dragging_range = True
        self.drag_start_x = self.range_indicator.mapFromGlobal(self.cursor().pos()).x()
        self.initial_lower_x = self.start_slider.value()
        self.initial_upper_x = self.end_slider.value()

    def stop_range_drag(self):
        self.is_dragging_range = False
        self.debounce_timer.start(self.debounce_delay)

    def range_mouse_move(self, event: QMouseEvent):
        if not self.is_dragging_range:
            return

        try:
            current_x = event.pos().x()
            delta_x = int(((current_x - self.drag_start_x) / self.slider_widget.width()) * 100)

            new_lower_x = self.initial_lower_x + delta_x
            new_upper_x = self.initial_upper_x + delta_x

            min_range_percentage = self.get_minimum_range_percentage()
            max_lower_x = 100 - min_range_percentage
            new_lower_x = max(0, min(new_lower_x, max_lower_x))
            new_upper_x = max(new_lower_x + min_range_percentage, min(new_upper_x, 100))

            self.start_slider.setValue(new_lower_x)
            self.end_slider.setValue(new_upper_x)
            self.lower_time_percentage = new_lower_x
            self.upper_time_percentage = new_upper_x
            self.update_labels()
            self.debounce_timer.start(self.debounce_delay)
        except Exception as e:
            logging.error(f"Error in range mouse move: {str(e)}")

    def get_minimum_range_percentage(self):
        try:
            if not self.start_time or not self.end_time:
                return 1.0

            total_duration_seconds = (self.end_time - self.start_time).total_seconds()
            if total_duration_seconds <= 0:
                return 1.0

            min_duration_seconds = 1.0
            if self.current_records:
                sampling_rate = self.current_records[0].get("samplingRate", 1)
                if sampling_rate > 0:
                    min_duration_seconds = max(1.0 / sampling_rate, 0.1)

            return min(min_duration_seconds / total_duration_seconds * 100, 100.0)
        except Exception as e:
            logging.error(f"Error calculating minimum range percentage: {str(e)}")
            return 1.0

    def update_labels(self):
        try:
            total_duration = (self.end_timestamp - self.start_timestamp)
            start_value = self.start_slider.value() / 100.0
            end_value = self.end_slider.value() / 100.0
            start_ts = self.start_timestamp + (total_duration * start_value)
            end_ts = self.start_timestamp + (total_duration * end_value)
            start_time_str = datetime.datetime.fromtimestamp(start_ts).strftime('%d-%m-%Y %H:%M:%S')
            end_time_str = datetime.datetime.fromtimestamp(end_ts).strftime('%d-%m-%Y %H:%M:%S')
            self.start_label.setText(f"Start: {start_time_str}")
            self.end_label.setText(f"End: {end_time_str}")
            self.lower_time_percentage = self.start_slider.value()
            self.upper_time_percentage = self.end_slider.value()
            self.debounce_timer.start(self.debounce_delay)
        except Exception as e:
            logging.error(f"Error updating labels: {str(e)}")

    def filter_and_plot_data(self):
        try:
            if not self.current_records:
                return
            self.filter_records_by_frame_index_range()
            self.plot_frequency_data()
        except Exception as e:
            logging.error(f"Error in filter and plot: {str(e)}")

    def downsample_array(self, array, factor):
        if factor <= 1:
            return array
        output_length = int(np.ceil(len(array) / factor))
        output = np.zeros(output_length)
        for i in range(output_length):
            start_index = i * factor
            end_index = min(start_index + factor, len(array))
            output[i] = np.mean(array[start_index:end_index])
        return output

    def get_current_frame_index_range(self):
        try:
            if not self.current_records:
                return (0, 0)
            min_frame_index = min(r.get("frameIndex", 0) for r in self.current_records)
            max_frame_index = max(r.get("frameIndex", 0) for r in self.current_records)
            total_frame_range = max_frame_index - min_frame_index
            start_frame_index = min_frame_index + int(total_frame_range * self.lower_time_percentage / 100.0)
            end_frame_index = min_frame_index + int(total_frame_range * self.upper_time_percentage / 100.0)
            start_frame_index = max(min_frame_index, start_frame_index)
            end_frame_index = min(max_frame_index, end_frame_index)
            return (start_frame_index, end_frame_index)
        except Exception as e:
            logging.error(f"Error getting frame index range: {str(e)}")
            return (0, 0)

    def find_closest_record(self, clicked_time):
        try:
            clicked_time = dt.fromtimestamp(clicked_time)
            closest_record = min(
                self.current_records,
                key=lambda r: abs((self.parse_time(r.get("createdAt", self.start_time)) - clicked_time).total_seconds()),
                default=None
            )
            if not closest_record:
                logging.info("No matching record found for clicked time")
                return None
            if not closest_record.get("message"):
                query = {
                    "filename": self.filename,
                    "model_name": self.model_name,
                    "project_name": self.project_name,
                    "frameIndex": closest_record.get("frameIndex"),
                    "email": self.email
                }
                closest_record = self.db.timeview_collection.find_one(query)
            return closest_record
        except Exception as e:
            logging.error(f"Error finding closest record: {str(e)}")
            return None

    def select_button_click(self):
        try:
            if not self.is_crosshair_locked or not self.locked_crosshair_position:
                self.show_message_box(
                    "Please click on the plot to lock the crosshair at desired position first, then click Select.",
                    "Information", "info"
                )
                logging.info("Select button clicked but crosshair not locked")
                return

            x, y = self.locked_crosshair_position
            selected_time = mdates.num2date(x).timestamp()
            self.selected_record = self.find_closest_record(selected_time)

            if not self.selected_record:
                self.show_message_box(
                    "No record found for the locked crosshair position.",
                    "Warning", "warning"
                )
                logging.info("No record found for locked crosshair position")
                return

            start_frame_index, end_frame_index = self.get_current_frame_index_range()
            selected_time_local = dt.fromtimestamp(
                self.parse_time(self.selected_record.get("createdAt")).timestamp(),
                tz=timezone('Asia/Kolkata')
            ).strftime('%d-%m-%Y %H:%M:%S.%f')[:-3]

            confirmation_message = (
                f"Final Confirmation - Range Selection Details:\n\n"
                f"📊 Selected Frame Index: {self.selected_record.get('frameIndex')}\n"
                f"🕐 Timestamp: {selected_time_local}\n"
                f"📁 Filename: {self.filename}\n"
                f"🔧 Model: {self.model_name}\n"
                f"📈 Frequency Value: {y:.2f}\n\n"
                f"📈 Current Range Selection:\n"
                f"   📍 Start Frame Index: {start_frame_index}\n"
                f"   📍 End Frame Index: {end_frame_index}\n"
                f"   📊 Range: {self.lower_time_percentage:.1f}% to {self.upper_time_percentage:.1f}%\n\n"
                f"✅ Confirm final selection?\n"
                f"The frequency plot will close after confirmation."
            )

            from PyQt5.QtWidgets import QMessageBox
            msg = QMessageBox()
            msg.setWindowTitle("Final Confirmation - Frame Range Information")
            msg.setText(confirmation_message)
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg.setDefaultButton(QMessageBox.No)
            result = msg.exec_()

            if result == QMessageBox.Yes:
                selected_data = {
                    "filename": self.filename,
                    "model": self.model_name,
                    "frameIndex": self.selected_record.get("frameIndex"),
                    "timestamp": self.selected_record.get("createdAt"),
                    "channelData": self.selected_record.get("message", [])
                }
                self.time_range_selected.emit(selected_data)
                logging.info(f"Data confirmed for FrameIndex: {self.selected_record.get('frameIndex')}, Range: {start_frame_index} to {end_frame_index}")
                self.show_message_box(
                    f"✅ Selection Confirmed Successfully!\n\n"
                    f"Frame Index {self.selected_record.get('frameIndex')} has been selected.\n"
                    f"Range: {start_frame_index} to {end_frame_index}\n\n"
                    f"The frequency plot will now close.",
                    "Selection Complete", "info"
                )
                self.accept()
            else:
                logging.info(f"User cancelled confirmation for FrameIndex: {self.selected_record.get('frameIndex')}")
        except Exception as e:
            logging.error(f"Error in select button click: {str(e)}")
            self.show_message_box(f"Error during selection: {str(e)}", "Error", "error")

    def show_message_box(self, message, title, icon_type):
        from PyQt5.QtWidgets import QMessageBox
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        if icon_type == "error":
            msg.setIcon(QMessageBox.Critical)
        elif icon_type == "info":
            msg.setIcon(QMessageBox.Information)
        elif icon_type == "warning":
            msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    @property
    def start_timestamp(self):
        return self.start_time.timestamp() if self.start_time else 0

    @property
    def end_timestamp(self):
        return self.end_time.timestamp() if self.end_time else 0

if __name__ == "__main__":
    from datetime import timezone
    app = QApplication(sys.argv)
    window = FrequencyPlot(
        project_name="TestProject",
        model_name="Model1",
        filename="data1",
        start_time="2023-01-01T00:00:00Z",
        end_time="2023-01-01T01:00:00Z"
    )
    window.show()
    sys.exit(app.exec_())