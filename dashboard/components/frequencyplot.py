# frequencyplot.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QSlider, QHBoxLayout, QApplication, QMessageBox
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
from datetime import datetime as dt, timezone

class FrequencyPlot(QWidget):
    time_range_selected = pyqtSignal(dict)

    def __init__(self, parent=None, project_name=None, model_name=None, filename=None, start_time=None, end_time=None, email="user@example.com"):
        super().__init__(parent)
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
        self.debounce_delay = 200
        self.crosshair_state_saved = False
        self.saved_crosshair_visible = False
        self.saved_crosshair_locked = False
        self.saved_crosshair_position = None
        self.is_dragging_range = False
        self.drag_start_x = 0
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

        self.title_label = QLabel(f"Frequency Analysis for {self.filename}")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        self.layout.addWidget(self.title_label)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.layout.addWidget(self.canvas)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.canvas.mpl_connect('button_press_event', self.on_mouse_click)
        self.canvas.mpl_connect('axes_leave_event', self.on_mouse_leave)

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
        self.slider_layout.addWidget(self.select_button)

        self.setLayout(self.layout)

    def initialize_data(self):
        try:
            messages = self.db.get_history_messages(self.project_name, self.model_name, filename=self.filename)
            if not messages:
                logging.error(f"No history messages found for {self.filename}")
                return
            self.current_records = messages
            self.filtered_records = self.current_records.copy()
            
            self.current_records.sort(key=lambda x: x.get("frameIndex", 0))
            self.filtered_records = self.current_records.copy()
            
            self.time_data = [record.get("frameIndex", 0) for record in self.current_records]
            self.frequency_data = [record.get("messageFrequency", 0) for record in self.current_records]
            
            if not self.start_time:
                first_record = min(self.current_records, key=lambda x: self.parse_time(x.get("createdAt")).timestamp() if self.parse_time(x.get("createdAt")) else 0)
                self.start_time = self.parse_time(first_record.get("createdAt"))
            if not self.end_time:
                last_record = max(self.current_records, key=lambda x: self.parse_time(x.get("createdAt")).timestamp() if self.parse_time(x.get("createdAt")) else 0)
                self.end_time = self.parse_time(last_record.get("createdAt"))
                
            self.filter_and_plot_data()
        except Exception as e:
            logging.error(f"Error initializing: {str(e)}")

    def filter_and_plot_data(self):
        try:
            if not self.current_records:
                return
                
            # Calculate filtered records for selection purposes
            all_frame_indices = [r.get("frameIndex", 0) for r in self.current_records]
            min_frame = min(all_frame_indices)
            max_frame = max(all_frame_indices)
            
            frame_range = max_frame - min_frame
            lower_frame = min_frame + (frame_range * self.lower_time_percentage / 100)
            upper_frame = min_frame + (frame_range * self.upper_time_percentage / 100)
            
            self.filtered_records = [record for record in self.current_records 
                                   if lower_frame <= record.get("frameIndex", 0) <= upper_frame]
            
            # Plot ALL data from current_records
            self.time_data = [record.get("frameIndex", 0) for record in self.current_records]
            self.frequency_data = [record.get("messageFrequency", 0) for record in self.current_records]
            
            self.ax.clear()
            self.ax.plot(self.time_data, self.frequency_data, marker='o', linestyle='-', color='b', label='Frequency')
            self.ax.set_xlabel('Frame Index')
            self.ax.set_ylabel('Frequency')
            self.ax.set_title('Frequency vs Frame Index')
            self.ax.legend()
            
            # Redraw locked crosshair if it exists
            if self.is_crosshair_locked and self.locked_crosshair_position:
                self.draw_crosshair(self.locked_crosshair_position[0], self.locked_crosshair_position[1])
            
            self.canvas.draw()
            logging.debug(f"Plotted {len(self.current_records)} data points")
        except Exception as e:
            logging.error(f"Error filtering and plotting: {str(e)}")

    def update_labels(self):
        self.lower_time_percentage = self.start_slider.value()
        self.upper_time_percentage = self.end_slider.value()
        
        if self.current_records:
            all_frame_indices = [r.get("frameIndex", 0) for r in self.current_records]
            min_frame = min(all_frame_indices)
            max_frame = max(all_frame_indices)
            frame_range = max_frame - min_frame
            
            lower_frame = int(min_frame + (frame_range * self.lower_time_percentage / 100))
            upper_frame = int(min_frame + (frame_range * self.upper_time_percentage / 100))
            
            self.start_label.setText(f"Start: {lower_frame}")
            self.end_label.setText(f"End: {upper_frame}")
        else:
            self.start_label.setText(f"Start: 0")
            self.end_label.setText(f"End: 0")
            
        self.debounce_timer.start(self.debounce_delay)

    def on_mouse_move(self, event):
        if not event.inaxes:
            return
        current_time = datetime.datetime.now()
        if (current_time - self.last_mouse_move).total_seconds() * 1000 < self.mouse_move_debounce_ms:
            return
        self.last_mouse_move = current_time
        
        if not self.is_crosshair_locked:
            self.is_crosshair_visible = True
            self.draw_crosshair(event.xdata, event.ydata)
        elif self.is_crosshair_locked and self.locked_crosshair_position:
            self.draw_crosshair(self.locked_crosshair_position[0], self.locked_crosshair_position[1])
        
        if self.is_dragging_range:
            self.update_range_on_drag(event.xdata)

    def on_mouse_click(self, event):
        if not event.inaxes:
            return
        if not self.is_crosshair_locked:
            self.is_crosshair_locked = True
            self.locked_crosshair_position = (event.xdata, event.ydata)
            self.draw_crosshair(event.xdata, event.ydata)
            logging.debug(f"Crosshair locked at ({event.xdata}, {event.ydata})")
        else:
            self.is_crosshair_locked = False
            self.locked_crosshair_position = None
            self.is_crosshair_visible = True
            self.draw_crosshair(event.xdata, event.ydata)
            logging.debug("Crosshair unlocked")

    def on_mouse_leave(self, event):
        if not self.is_crosshair_locked:
            self.is_crosshair_visible = False
            self.remove_crosshair()

    def draw_crosshair(self, x, y):
        if not x or not y:
            return
        # Only remove lines if they exist and are in the axes
        if hasattr(self, 'crosshair_vline') and self.crosshair_vline in self.ax.lines:
            self.crosshair_vline.remove()
        if hasattr(self, 'crosshair_hline') and self.crosshair_hline in self.ax.lines:
            self.crosshair_hline.remove()
        
        # Draw new crosshair lines
        self.crosshair_vline = Line2D([x, x], [self.ax.get_ylim()[0], self.ax.get_ylim()[1]], 
                                    color='red', linestyle='--', linewidth=1)
        self.crosshair_hline = Line2D([self.ax.get_xlim()[0], self.ax.get_xlim()[1]], [y, y], 
                                    color='red', linestyle='--', linewidth=1)
        self.ax.add_line(self.crosshair_vline)
        self.ax.add_line(self.crosshair_hline)
        self.canvas.draw_idle()

    def remove_crosshair(self):
        # Only remove lines if they exist and are in the axes
        if hasattr(self, 'crosshair_vline') and self.crosshair_vline in self.ax.lines:
            self.crosshair_vline.remove()
        if hasattr(self, 'crosshair_hline') and self.crosshair_hline in self.ax.lines:
            self.crosshair_hline.remove()
        self.canvas.draw_idle()

    def start_range_drag(self):
        self.is_dragging_range = True
        if self.time_data:
            self.drag_start_x = self.time_data[0] + (self.time_data[-1] - self.time_data[0]) * (self.lower_time_percentage / 100)

    def stop_range_drag(self):
        self.is_dragging_range = False

    def range_mouse_move(self, event):
        if self.is_dragging_range:
            self.update_range_on_drag(event.xdata)

    def update_range_on_drag(self, x):
        if not x or not self.time_data:
            return
        delta_x = x - self.drag_start_x
        delta_percentage = (delta_x / (self.time_data[-1] - self.time_data[0])) * 100
        new_lower = max(0, min(100, self.lower_time_percentage + delta_percentage))
        new_upper = max(0, min(100, self.upper_time_percentage + delta_percentage))
        if new_lower < new_upper:
            self.lower_time_percentage = new_lower
            self.upper_time_percentage = new_upper
            self.start_slider.setValue(int(new_lower))
            self.end_slider.setValue(int(new_upper))
            self.filter_and_plot_data()

    def find_closest_record(self, selected_frame_index):
        try:
            closest_record = min(self.filtered_records, key=lambda r: abs(r.get("frameIndex", 0) - selected_frame_index))
            if not closest_record.get("message"):
                query = {
                    "filename": self.filename,
                    "moduleName": self.model_name,
                    "projectName": self.project_name,
                    "frameIndex": closest_record.get("frameIndex"),
                    "email": self.email
                }
                full_records = self.db.history_collection.find(query)
                full_record = list(full_records)
                if full_record:
                    closest_record = full_record[0]
            return closest_record
        except Exception as e:
            logging.error(f"Error finding closest record: {str(e)}")
            return None

    def get_current_frame_index_range(self):
        if not self.current_records:
            return 0, 0
        all_frame_indices = [r.get("frameIndex", 0) for r in self.current_records]
        min_frame = min(all_frame_indices)
        max_frame = max(all_frame_indices)
        frame_range = max_frame - min_frame
        start_frame_index = int(min_frame + (frame_range * self.lower_time_percentage / 100))
        end_frame_index = int(min_frame + (frame_range * self.upper_time_percentage / 100))
        return start_frame_index, end_frame_index

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
            selected_frame_index = int(x)
            self.selected_record = self.find_closest_record(selected_frame_index)

            if not self.selected_record:
                self.show_message_box(
                    "No record found for the locked crosshair position.",
                    "Warning", "warning"
                )
                logging.info("No record found for locked crosshair position")
                return

            start_frame_index, end_frame_index = self.get_current_frame_index_range()
            
            selected_time_local = "N/A"
            if self.selected_record.get("createdAt"):
                try:
                    selected_time_local = dt.fromtimestamp(
                        self.parse_time(self.selected_record.get("createdAt")).timestamp(),
                        tz=timezone('Asia/Kolkata')
                    ).strftime('%d-%m-%Y %H:%M:%S.%f')[:-3]
                except:
                    selected_time_local = self.selected_record.get("createdAt", "N/A")

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
                    "channelData": self.selected_record.get("message", []),
                    "project_name": self.project_name,
                    "numberOfChannels": self.selected_record.get("numberOfChannels", 0),
                    "tacoChannelCount": self.selected_record.get("tacoChannelCount", 0),
                    "samplingRate": self.selected_record.get("samplingRate", 0),
                    "samplingSize": self.selected_record.get("samplingSize", 0),
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
                self.parent().close()  # Close the parent subwindow
            else:
                logging.info(f"User cancelled confirmation for FrameIndex: {self.selected_record.get('frameIndex')}")
        except Exception as e:
            logging.error(f"Error in select button click: {str(e)}")
            self.show_message_box(f"Error during selection: {str(e)}", "Error", "error")

    def show_message_box(self, message, title, icon_type):
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