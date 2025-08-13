# time_report.py
import asyncio
import platform
import logging
import math
import numpy as np
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton,
    QScrollArea, QDateTimeEdit, QGridLayout, QProgressDialog, QApplication, QMessageBox
)
from PyQt5.QtCore import (
    Qt, QDateTime, QRect, pyqtSignal, QEvent, QObject, QTimer
)
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor
import pyqtgraph as pg
from pyqtgraph import PlotWidget, mkPen, AxisItem, InfiniteLine, SignalProxy

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class QRangeSlider(QWidget):
    """Custom dual slider widget for selecting a time range."""
    valueChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(30)
        self.setMinimumWidth(300)
        self.min_value = 0
        self.max_value = 1000
        self.left_value = 0
        self.right_value = 1000
        self.dragging = None
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #d1d6d9;")

    def setRange(self, min_val, max_val):
        self.min_value = min_val
        self.max_value = max_val
        self.left_value = max(self.min_value, min(self.left_value, self.max_value))
        self.right_value = max(self.left_value + 1, min(self.right_value, self.max_value))
        self.update()

    def setValues(self, left, right):
        self.left_value = max(self.min_value, min(left, self.max_value))
        self.right_value = max(self.left_value + 1, min(right, self.max_value))
        self.update()
        self.valueChanged.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        groove_rect = QRect(10, 10, self.width() - 20, 8)
        painter.setPen(QPen(QColor("#1a73e8")))
        painter.setBrush(QColor("#34495e"))
        painter.drawRoundedRect(groove_rect, 4, 4)
        left_pos = int(self._value_to_pos(self.left_value))
        right_pos = int(self._value_to_pos(self.right_value))
        selected_rect = QRect(left_pos, 10, right_pos - left_pos, 8)
        painter.setBrush(QColor("#90caf9"))
        painter.drawRoundedRect(selected_rect, 4, 4)
        painter.setPen(QPen(QColor("#1a73e8")))
        painter.setBrush(QColor("#42a5f5" if self.dragging == 'left' else "#1a73e8"))
        painter.drawEllipse(left_pos - 9, 6, 18, 18)
        painter.setBrush(QColor("#42a5f5" if self.dragging == 'right' else "#1a73e8"))
        painter.drawEllipse(right_pos - 9, 6, 18, 18)

    def _value_to_pos(self, value):
        if self.max_value == self.min_value:
            return 10
        return 10 + (self.width() - 20) * (value - self.min_value) / (self.max_value - self.min_value)

    def _pos_to_value(self, pos):
        if self.width() <= 20:
            return self.min_value
        value = self.min_value + (pos - 10) / (self.width() - 20) * (self.max_value - self.min_value)
        return max(self.min_value, min(self.max_value, value))

    def mousePressEvent(self, event):
        pos = event.pos().x()
        left_pos = self._value_to_pos(self.left_value)
        right_pos = self._value_to_pos(self.right_value)
        if abs(pos - left_pos) < abs(pos - right_pos) and abs(pos - left_pos) < 10:
            self.dragging = 'left'
        elif abs(pos - right_pos) <= abs(pos - left_pos) and abs(pos - right_pos) < 10:
            self.dragging = 'right'
        self.update()

    def mouseMoveEvent(self, event):
        if self.dragging:
            pos = event.pos().x()
            value = self._pos_to_value(pos)
            if self.dragging == 'left':
                self.left_value = max(self.min_value, min(value, self.right_value - 1))
            elif self.dragging == 'right':
                self.right_value = max(self.left_value + 1, min(value, self.max_value))
            self.update()
            self.valueChanged.emit()

    def mouseReleaseEvent(self, event):
        self.dragging = None
        self.update()

    def getValues(self):
        return self.left_value, self.right_value

class TimeAxisItem(pg.AxisItem):
    """Custom axis to display datetime on x-axis."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tickStrings(self, values, scale, spacing):
        result = []
        for v in values:
            try:
                if isinstance(v, (int, float)) and v > 0:
                    dt = datetime.fromtimestamp(v)
                    result.append(dt.strftime('%Y-%m-%d\n%H:%M:%S'))
                else:
                    result.append("")
            except (ValueError, OSError, OverflowError) as e:
                logging.warning(f"Error formatting timestamp {v}: {e}")
                result.append("")
        return result

class MouseTracker(QObject):
    """Event filter to track mouse enter/leave on plot viewport."""
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

class TimeReportFeature:
    def __init__(self, parent, db, project_name, channel=None, model_name=None, console=None, filename=None):
        self.parent = parent
        self.db = db
        self.project_name = project_name
        self.channel = channel
        self.model_name = model_name
        self.console = console
        self.widget = QWidget(self.parent)
        self.plot_widgets = []
        self.plots = [] # Stores the actual PlotDataItem objects
        self.data = [] # List of np arrays for each channel's Y data
        self.channel_times = np.array([]) # Single np array for X time data (shared by all)
        self.vlines = []
        self.proxies = []
        self.trackers = []
        self.trigger_lines = []
        self.active_line_idx = None
        self.num_channels = 0 # Main channels
        self.num_plots = 0 # Total plots (main + tacho)
        self.tacho_channels_count = 0
        self.sample_rate = None
        self.samples_per_channel = None
        self.filenames = []
        self.selected_filename = filename
        self.file_start_time = None
        self.file_end_time = None
        self.start_time = None
        self.end_time = None
        self.scaling_factor = 3.3 / 65535
        self.channel_properties = {}
        self.channel_names = [] # Store channel names from DB
        self.max_points_to_plot = 100000
        self.plot_colors = [
            '#0000FF', '#FF0000', '#00FF00', '#800080', '#FFA500', '#A52A2A',
            '#FFC0CB', '#008080', '#FF4500', '#32CD32', '#00CED1', '#FFD700',
            '#FF69B4', '#8A2BE2', '#FF6347', '#20B2AA', '#ADFF2F', '#9932CC',
            '#FF7F50', '#00FA9A', '#9400D3'
        ]
        self.init_ui_deferred()
        self.load_channel_properties()

    def load_channel_properties(self):
        try:
            project_data = self.db.get_project_data(self.project_name)
            if not project_data:
                logging.error(f"Project {self.project_name} not found")
                if self.console:
                    self.console.append_to_console(f"Project {self.project_name} not found")
                return
            for model in project_data.get("models", []):
                if model.get("name") == self.model_name:
                    self.channel_names = [ch.get("channelName") for ch in model.get("channels", [])]
                    for channel in model.get("channels", []):
                        channel_name = channel.get("channelName")
                        # Use .get() with defaults and handle potential None/empty string values
                        unit = (channel.get("unit", "mil") or "mil").lower()
                        correction_value_str = channel.get("correctionValue", "1.0") or "1.0"
                        gain_str = channel.get("gain", "1.0") or "1.0"
                        sensitivity_str = channel.get("sensitivity", "1.0") or "1.0"
                        converted_sensitivity_str = channel.get("ConvertedSensitivity", sensitivity_str) or sensitivity_str

                        try:
                            correction_value = float(correction_value_str)
                        except ValueError:
                            logging.warning(f"Invalid CorrectionValue '{correction_value_str}' for {channel_name}, defaulting to 1.0")
                            correction_value = 1.0

                        try:
                            gain = float(gain_str)
                        except ValueError:
                            logging.warning(f"Invalid Gain '{gain_str}' for {channel_name}, defaulting to 1.0")
                            gain = 1.0

                        try:
                            sensitivity = float(sensitivity_str)
                        except ValueError:
                            logging.warning(f"Invalid Sensitivity '{sensitivity_str}' for {channel_name}, defaulting to 1.0")
                            sensitivity = 1.0

                        try:
                            converted_sensitivity = float(converted_sensitivity_str)
                        except ValueError:
                            logging.warning(f"Invalid ConvertedSensitivity '{converted_sensitivity_str}' for {channel_name}, defaulting to Sensitivity ({sensitivity})")
                            converted_sensitivity = sensitivity # Fallback

                        self.channel_properties[channel_name] = {
                            "unit": unit,
                            "correctionValue": correction_value,
                            "gain": gain,
                            "sensitivity": sensitivity, # Keep original for reference if needed
                            "ConvertedSensitivity": converted_sensitivity
                        }
                    break
            logging.debug(f"Loaded channel names: {self.channel_names}")
            logging.debug(f"Loaded channel properties: {self.channel_properties}")
        except Exception as e:
            logging.error(f"Error loading channel properties: {str(e)}", exc_info=True)
            if self.console:
                self.console.append_to_console(f"Error loading channel properties: {str(e)}")

    def init_ui_deferred(self):
        self.setup_basic_ui()
        QTimer.singleShot(0, self.load_data_async)

    def setup_basic_ui(self):
        layout = QVBoxLayout()
        self.widget.setLayout(layout)

        header = QLabel(f"TIME REPORT FOR {self.project_name.upper()}")
        header.setStyleSheet("color: black; font-size: 26px; font-weight: bold; padding: 8px;")
        layout.addWidget(header, alignment=Qt.AlignCenter)

        controls_widget = QWidget()
        controls_widget.setStyleSheet("background-color: #d1d6d9; border-radius: 5px; padding: 10px;")
        controls_layout = QVBoxLayout()
        controls_widget.setLayout(controls_layout)

        file_layout = QHBoxLayout()
        file_label = QLabel(f"Select Saved File (Model: {self.model_name or 'None'}, Channel: {self.channel or 'All'}):")
        file_label.setStyleSheet("color: black; font-size: 16px; font: bold")
        self.file_combo = QComboBox()
        self.file_combo.addItem("Loading files...")
        self.file_combo.setStyleSheet("""
            QComboBox {
                background-color: #fdfdfd;
                color: #212121;
                border: 2px solid #90caf9;
                border-radius: 8px;
                padding: 10px 40px 10px 14px;
                font-size: 16px;
                font-weight: 600;
                min-width: 220px;
                box-shadow: inset 0 0 5px rgba(0, 0, 0, 0.05);
            }
            QComboBox:hover {
                border: 2px solid #42a5f5;
                background-color: #f5faff;
            }
            QComboBox:focus {
                border: 2px solid #1e88e5;
                background-color: #ffffff;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 36px;
                border-left: 1px solid #e0e0e0;
                background-color: #e3f2fd;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                border: 1px solid #90caf9;
                border-radius: 4px;
                padding: 5px;
                selection-background-color: #e3f2fd;
                selection-color: #0d47a1;
                font-size: 15px;
                outline: 0;
            }
            QComboBox::item {
                padding: 10px 8px;
                border: none;
            }
            QComboBox::item:selected {
                background-color: #bbdefb;
                color: #0d47a1;
            }
        """)
        self.file_combo.currentTextChanged.connect(self.on_filename_selected)

        self.ok_button = QPushButton("OK")
        self.ok_button.setStyleSheet("""
            QPushButton {
                background-color: #1a73e8;
                color: white;
                padding: 15px;
                font-size: 15px;
                width: 100px;
                border-radius: 50%;
                font-weight: bold;
            }
            QPushButton:pressed {
                background-color: #155ab6;
            }
            QPushButton:disabled {
                background-color: #546e7a;
                color: #b0bec5;
            }
        """)
        self.ok_button.clicked.connect(self.plot_data)
        self.ok_button.setEnabled(False)

        file_layout.addWidget(file_label)
        file_layout.addWidget(self.file_combo)
        file_layout.addWidget(self.ok_button)
        file_layout.addStretch()
        controls_layout.addLayout(file_layout)

        time_range_layout = QHBoxLayout()
        start_time_label = QLabel("Select Start Time:")
        start_time_label.setStyleSheet("color: black; font-size: 14px; font: bold")
        self.start_time_edit = QDateTimeEdit()
        self.start_time_edit.setStyleSheet("background-color: #34495e; color: white; border: 2px solid black; padding: 15px; font: bold; width: 200px")
        self.start_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.start_time_edit.dateTimeChanged.connect(self.validate_time_range)

        end_time_label = QLabel("Select End Time:")
        end_time_label.setStyleSheet("color: black; font-size: 14px; font: bold")
        self.end_time_edit = QDateTimeEdit()
        self.end_time_edit.setStyleSheet("background-color: #34495e; color: white; border: 2px solid black; padding: 15px; font: bold; width: 200px")
        self.end_time_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.end_time_edit.dateTimeChanged.connect(self.validate_time_range)

        time_range_layout.addWidget(start_time_label)
        time_range_layout.addWidget(self.start_time_edit)
        time_range_layout.addWidget(end_time_label)
        time_range_layout.addWidget(self.end_time_edit)
        time_range_layout.addStretch()
        controls_layout.addLayout(time_range_layout)

        slider_layout = QGridLayout()
        slider_label = QLabel("Drag Time Range:")
        slider_label.setStyleSheet("color: black; font-size: 14px; font: bold")
        slider_label.setFixedWidth(150)
        self.time_slider = QRangeSlider(self.widget)
        self.time_slider.valueChanged.connect(self.update_time_from_slider)
        slider_layout.addWidget(slider_label, 0, 0, 1, 1, Qt.AlignLeft | Qt.AlignVCenter)
        slider_layout.addWidget(self.time_slider, 0, 1, 1, 1)
        slider_layout.setColumnStretch(1, 1)
        controls_layout.addLayout(slider_layout)

        time_info_layout = QHBoxLayout()
        self.start_time_label = QLabel("File Start Time: Loading...")
        self.start_time_label.setStyleSheet("color: black; font-size: 14px; font: bold")
        self.stop_time_label = QLabel("File Stop Time: Loading...")
        self.stop_time_label.setStyleSheet("color: black; font-size: 14px; font: bold")
        time_info_layout.addWidget(self.start_time_label)
        time_info_layout.addWidget(self.stop_time_label)
        time_info_layout.addStretch()
        controls_layout.addLayout(time_info_layout)

        layout.addWidget(controls_widget)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border-radius: 8px;
                padding: 5px;
            }
            QScrollBar:vertical {
                background: white;
                width: 10px;
                margin: 0px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: black;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: none;
            }
        """)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_content.setStyleSheet("background-color: #d1d6d9; border-radius: 5px; padding: 10px;")
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area, stretch=1)

        self.file_combo.setEnabled(False)
        self.start_time_edit.setEnabled(False)
        self.end_time_edit.setEnabled(False)

    def load_data_async(self):
        try:
            # Fetch filenames using the correct DB method
            self.filenames = self.db.get_distinct_filenames(self.project_name, self.model_name)
            self.file_combo.clear()
            if not self.filenames:
                self.file_combo.addItem("No Files Available")
                self.ok_button.setEnabled(False)
            else:
                self.file_combo.addItem("Select File")
                self.file_combo.addItems(self.filenames)
                self.ok_button.setEnabled(True)
            if self.selected_filename and self.selected_filename in self.filenames:
                self.file_combo.setCurrentText(self.selected_filename)
                # Trigger load automatically if filename is pre-selected
                # self.on_filename_selected(self.selected_filename) # Let user click OK
            else:
                self.file_combo.setCurrentIndex(0)
            self.file_combo.setEnabled(True)
            logging.debug(f"Loaded {len(self.filenames)} files for {self.project_name}/{self.model_name}")
        except Exception as e:
            logging.error(f"Error loading files: {e}", exc_info=True)
            self.file_combo.clear()
            self.file_combo.addItem("Error Loading Files")
            self.ok_button.setEnabled(False)
            self.file_combo.setEnabled(True)
            if self.console:
                self.console.append_to_console(f"Error loading files: {e}")

    def on_filename_selected(self, filename):
        self.selected_filename = filename
        if filename and filename not in ["Loading files...", "No Files Available", "Error Loading Files", "Select File"]:
            self.ok_button.setEnabled(True)
            # Load time labels and enable time controls
            self.update_time_labels(filename)
        else:
            self.ok_button.setEnabled(False)
            self.start_time_edit.setEnabled(False)
            self.end_time_edit.setEnabled(False)
            self.start_time_label.setText("File Start Time: N/A")
            self.stop_time_label.setText("File Stop Time: N/A")
            self.clear_plots()

    def validate_time_range(self):
        if not self.start_time_edit.isEnabled() or not self.end_time_edit.isEnabled():
             return
        start = self.start_time_edit.dateTime().toPython().timestamp()
        end = self.end_time_edit.dateTime().toPython().timestamp()
        if start > end:
            # Correct the start time to match end time if it's later
            self.start_time_edit.setDateTime(QDateTime.fromSecsSinceEpoch(int(end)))

    def update_time_from_slider(self):
        left, right = self.time_slider.getValues()
        self.start_time = left
        self.end_time = right
        # Block signals to prevent recursive calls
        self.start_time_edit.blockSignals(True)
        self.end_time_edit.blockSignals(True)
        self.start_time_edit.setDateTime(QDateTime.fromSecsSinceEpoch(int(left)))
        self.end_time_edit.setDateTime(QDateTime.fromSecsSinceEpoch(int(right)))
        self.start_time_edit.blockSignals(False)
        self.end_time_edit.blockSignals(False)

    def update_time_labels(self, filename):
        try:
            messages = self.db.get_history_messages(self.project_name, self.model_name, filename=filename)
            if not messages:
                self.start_time_label.setText("File Start Time: N/A")
                self.stop_time_label.setText("File Stop Time: N/A")
                self.start_time_edit.setEnabled(False)
                self.end_time_edit.setEnabled(False)
                return

            # Sort messages by creation time to get accurate start/end
            sorted_messages = sorted(messages, key=lambda x: datetime.fromisoformat(x['createdAt'].replace('Z', '+00:00')))
            first_message = sorted_messages[0]
            last_message = sorted_messages[-1]

            first_created_at = datetime.fromisoformat(first_message['createdAt'].replace('Z', '+00:00'))
            last_created_at = datetime.fromisoformat(last_message['createdAt'].replace('Z', '+00:00'))

            # Duration calculation based on the *last* message's parameters
            sampling_size = last_message.get("samplingSize", 0)
            sampling_rate = last_message.get("samplingRate", 1)
            if sampling_rate <= 0:
                 raise ValueError(f"Invalid sampling rate {sampling_rate} in message")
            duration = sampling_size / sampling_rate

            file_start = first_created_at
            file_end = last_created_at + timedelta(seconds=duration)

            self.file_start_time = file_start.timestamp()
            self.file_end_time = file_end.timestamp()
            self.start_time = self.file_start_time
            self.end_time = self.file_end_time
            self.start_time_label.setText(f"File Start Time: {file_start.strftime('%Y-%m-%d %H:%M:%S')}")
            self.stop_time_label.setText(f"File Stop Time: {file_end.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Block signals to prevent triggering validate_time_range
            self.start_time_edit.blockSignals(True)
            self.end_time_edit.blockSignals(True)
            self.start_time_edit.setDateTime(QDateTime.fromString(file_start.isoformat(), Qt.ISODate))
            self.end_time_edit.setDateTime(QDateTime.fromString(file_end.isoformat(), Qt.ISODate))
            self.start_time_edit.blockSignals(False)
            self.end_time_edit.blockSignals(False)
            
            self.time_slider.setRange(self.file_start_time, self.file_end_time)
            self.time_slider.setValues(self.start_time, self.end_time)
            self.start_time_edit.setEnabled(True)
            self.end_time_edit.setEnabled(True)
            logging.debug(f"Time labels updated for {filename}")

        except Exception as e:
            logging.error(f"Error updating time labels for {filename}: {e}", exc_info=True)
            self.start_time_label.setText("File Start Time: Error")
            self.stop_time_label.setText("File Stop Time: Error")
            self.start_time_edit.setEnabled(False)
            self.end_time_edit.setEnabled(False)
            if self.console:
                self.console.append_to_console(f"Error updating time labels for {filename}: {e}")

    def init_plots(self, main_channels, tacho_channels):
        try:
            total_channels = main_channels + tacho_channels
            logging.debug(f"Initializing {total_channels} plots: {main_channels} main, {tacho_channels} tacho")

            # Clear existing plots
            self.clear_plots()

            self.num_channels = main_channels
            self.tacho_channels_count = tacho_channels
            self.num_plots = total_channels

            for ch in range(total_channels):
                plot_widget = PlotWidget()
                plot_widget.setMinimumHeight(200)
                plot_widget.setBackground('#d1d6d9')
                plot_widget.showGrid(x=True, y=True)
                plot_widget.addLegend()
                axis = TimeAxisItem(orientation='bottom')
                plot_widget.setAxisItems({'bottom': axis})

                # Determine channel name and y-label
                if ch < len(self.channel_names):
                    channel_name = self.channel_names[ch]
                elif ch == len(self.channel_names) and tacho_channels >= 1: # First tacho
                    channel_name = "Frequency"
                elif ch == len(self.channel_names) + 1 and tacho_channels >= 2: # Second tacho
                    channel_name = "Trigger"
                else:
                    channel_name = f"Channel {ch + 1}"

                unit = self.channel_properties.get(channel_name, {}).get("unit", "mil")
                y_label = f"Amplitude ({unit})" if ch < main_channels else "Value"

                # Set Y range for tacho channels
                if ch >= main_channels:
                    plot_widget.setYRange(-0.5, 1.5, padding=0)

                plot_widget.getAxis('left').setLabel(f"{channel_name} ({y_label})")
                # Initial empty plot - store the PlotDataItem
                plot_data_item = plot_widget.plot([], [], pen=mkPen(color=self.plot_colors[ch % len(self.plot_colors)], width=2), name=channel_name)
                self.plot_widgets.append(plot_widget)
                self.plots.append(plot_data_item) # Store the PlotDataItem, not the widget
                self.scroll_layout.addWidget(plot_widget)

                vline = InfiniteLine(pos=0, angle=90, movable=False, pen=mkPen('k', width=1, style=Qt.DashLine))
                vline.setVisible(False)
                plot_widget.addItem(vline)
                self.vlines.append(vline)

                tracker = MouseTracker(plot_widget, ch, self)
                plot_widget.installEventFilter(tracker)
                self.trackers.append(tracker)

                proxy = SignalProxy(plot_widget.scene().sigMouseMoved, rateLimit=60, slot=lambda evt, idx=ch: self.mouse_moved(evt, idx))
                self.proxies.append(proxy)

            logging.debug(f"Initialized {self.num_plots} plots successfully")
        except Exception as e:
            logging.error(f"Error initializing plots: {str(e)}", exc_info=True)
            if self.console:
                self.console.append_to_console(f"Error initializing plots: {str(e)}")

    def downsample_array(self, data, factor):
        if factor <= 1 or len(data) == 0:
            return data
        # Ensure data is a numpy array for efficient operations
        data = np.asarray(data)
        output_length = int(np.ceil(len(data) / factor))
        # Reshape data to group elements for averaging
        # Truncate data to make its length divisible by factor
        truncated_len = (len(data) // factor) * factor
        if truncated_len == 0:
            # If not enough data, return the mean
            return np.array([np.mean(data)]) if len(data) > 0 else np.array([])
        truncated_data = data[:truncated_len]
        reshaped_data = truncated_data.reshape(-1, factor)
        downsampled = np.mean(reshaped_data, axis=1)

        # If there were leftover elements, average them and append
        if truncated_len < len(data):
            leftover = data[truncated_len:]
            downsampled = np.append(downsampled, np.mean(leftover))

        return downsampled

    def plot_data(self):
        filename = self.selected_filename or self.file_combo.currentText()
        if not filename or filename in ["No Files Available", "Error Loading Files", "Loading files...", "Select File"]:
            self.clear_plots()
            if self.console:
                self.console.append_to_console("No valid file selected to plot.")
            return

        progress = QProgressDialog("Loading and plotting data...", "Cancel", 0, 100, self.widget)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        try:
            # --- 1. Fetch Data ---
            progress.setLabelText("Fetching data from database...")
            progress.setValue(10)
            messages = self.db.get_history_messages(self.project_name, self.model_name, filename=filename)
            if not messages:
                raise ValueError(f"No data found for filename {filename}")

            # --- 2. Sort and Filter Messages ---
            progress.setLabelText("Sorting messages...")
            progress.setValue(15)
            try:
                sorted_messages = sorted(messages, key=lambda x: datetime.fromisoformat(x['createdAt'].replace('Z', '+00:00')))
            except (ValueError, KeyError) as sort_error:
                logging.error(f"Error sorting messages by 'createdAt': {sort_error}")
                raise ValueError(f"Could not sort messages by timestamp: {sort_error}")

            progress.setLabelText("Filtering messages by time range and validating data...")
            progress.setValue(20)
            # Filter messages whose *start time* (createdAt) is before the selected end_time
            # and whose *end time* (createdAt + duration) is after the selected start_time.
            # This ensures we get messages that overlap with the selected range.
            filtered_messages = []
            for msg in sorted_messages:
                 try:
                     msg_created_at_dt = datetime.fromisoformat(msg['createdAt'].replace('Z', '+00:00'))
                     msg_created_at_ts = msg_created_at_dt.timestamp()
                     # --- CRITICAL: Strict Validation of samplingSize and samplingRate ---
                     sampling_size_raw = msg.get("samplingSize")
                     sampling_rate_raw = msg.get("samplingRate")
                     message_data_raw = msg.get("message")

                     # Check 1: Is the field present and not None?
                     if sampling_size_raw is None:
                         logging.warning(f"Skipping message (FrameIndex: {msg.get('frameIndex', 'N/A')}): 'samplingSize' field is missing or None.")
                         continue
                     if sampling_rate_raw is None:
                         logging.warning(f"Skipping message (FrameIndex: {msg.get('frameIndex', 'N/A')}): 'samplingRate' field is missing or None.")
                         continue

                     # Check 2: Is it the correct type?
                     if not isinstance(sampling_size_raw, int):
                          # Handle potential string representations of integers from DB
                          if isinstance(sampling_size_raw, str) and sampling_size_raw.isdigit():
                              try:
                                  sampling_size_raw = int(sampling_size_raw)
                                  logging.info(f"Converted string 'samplingSize' to int: {sampling_size_raw}")
                              except ValueError:
                                  pass # Conversion failed, will be caught below
                          if not isinstance(sampling_size_raw, int):
                              logging.warning(f"Skipping message (FrameIndex: {msg.get('frameIndex', 'N/A')}): 'samplingSize' is not an integer ({type(sampling_size_raw)}: {sampling_size_raw})")
                              continue

                     if not isinstance(sampling_rate_raw, (int, float)):
                          # Handle potential string representations
                          if isinstance(sampling_rate_raw, str):
                              try:
                                  sampling_rate_raw = float(sampling_rate_raw)
                                  logging.info(f"Converted string 'samplingRate' to float: {sampling_rate_raw}")
                              except ValueError:
                                  pass # Conversion failed, will be caught below
                          if not isinstance(sampling_rate_raw, (int, float)):
                              logging.warning(f"Skipping message (FrameIndex: {msg.get('frameIndex', 'N/A')}): 'samplingRate' is not a number ({type(sampling_rate_raw)}: {sampling_rate_raw})")
                              continue

                     # Check 3: Is it a positive value?
                     if sampling_size_raw <= 0:
                         logging.warning(f"Skipping message (FrameIndex: {msg.get('frameIndex', 'N/A')}): 'samplingSize' is not positive ({sampling_size_raw})")
                         continue
                     if sampling_rate_raw <= 0:
                         logging.warning(f"Skipping message (FrameIndex: {msg.get('frameIndex', 'N/A')}): 'samplingRate' is not positive ({sampling_rate_raw})")
                         continue

                     # Check 4: Is message data present?
                     if message_data_raw is None or not isinstance(message_data_raw, (list, np.ndarray)):
                          logging.warning(f"Skipping message (FrameIndex: {msg.get('frameIndex', 'N/A')}): Invalid or missing 'message' data")
                          continue

                     duration = sampling_size_raw / sampling_rate_raw
                     msg_end_time_ts = msg_created_at_ts + duration

                     # Check 5: Does it overlap with the selected time range?
                     if msg_created_at_ts <= self.end_time and msg_end_time_ts >= self.start_time:
                         # --- If ALL checks pass, add to filtered list ---
                         # Store the validated values to avoid .get() later
                         msg['_validated_samplingSize'] = sampling_size_raw
                         msg['_validated_samplingRate'] = sampling_rate_raw
                         filtered_messages.append(msg)
                     # --- END CRITICAL VALIDATION ---
                 except (ValueError, TypeError, KeyError) as e:
                     logging.warning(f"Skipping message due to error parsing fields: {e}")
                     continue

            if not filtered_messages:
                error_msg = (f"No valid data messages found within the selected time range for filename {filename}. "
                             f"Check database for correct 'samplingSize' (must be positive int), "
                             f"'samplingRate' (must be positive number), and 'message' fields.")
                raise ValueError(error_msg)

            # --- 3. Get Structure from First VALID Filtered Message ---
            # CRITICAL: Set sample_rate and samples_per_channel HERE from the first valid message
            # BEFORE using them in any calculations.
            progress.setLabelText("Getting data structure from first valid message...")
            progress.setValue(25)
            
            # --- CRITICAL FIX: Validate and assign from first message ---
            first_valid_msg = filtered_messages[0] # Use the first *filtered* message
            main_channels = first_valid_msg.get("numberOfChannels", 0)
            # Note: DB field name discrepancy with C# (tachoChannelCount vs tacoChannelCount)
            tacho_channels = first_valid_msg.get("tachoChannelCount", 0) or first_valid_msg.get("tacoChannelCount", 0)
            
            # --- CRITICAL: Use the validated values stored during filtering ---
            # This ensures we are using the confirmed good values.
            validated_samples_per_channel = first_valid_msg.get('_validated_samplingSize')
            validated_sample_rate = first_valid_msg.get('_validated_samplingRate')
            # --- END CRITICAL ASSIGNMENT ---

            # --- ULTIMATE SANITY CHECK ---
            # This is the final, paranoid check before the calculation.
            if validated_samples_per_channel is None or not isinstance(validated_samples_per_channel, int) or validated_samples_per_channel <= 0:
                 raise ValueError(f"FATAL: First valid message's validated 'samplingSize' is invalid: {validated_samples_per_channel} (Type: {type(validated_samples_per_channel)})")
            if validated_sample_rate is None or not isinstance(validated_sample_rate, (int, float)) or validated_sample_rate <= 0:
                 raise ValueError(f"FATAL: First valid message's validated 'samplingRate' is invalid: {validated_sample_rate} (Type: {type(validated_sample_rate)})")
            # --- END ULTIMATE SANITY CHECK ---

            # Now it's safe to assign to instance variables
            self.samples_per_channel = validated_samples_per_channel
            self.sample_rate = validated_sample_rate

            total_channels = main_channels + tacho_channels
            # --- FIXED LINE: Now guaranteed safe by the checks above ---
            expected_length_per_msg = self.samples_per_channel * total_channels # <-- Should not fail now
            # --- END FIXED LINE ---
            
            logging.debug(f"First valid message structure: main_channels={main_channels}, tacho_channels={tacho_channels}, "
                          f"samples_per_channel={self.samples_per_channel}, sample_rate={self.sample_rate}, "
                          f"total_channels={total_channels}, expected_length_per_msg={expected_length_per_msg}")

            # --- 4. Initialize Plots ---
            progress.setLabelText("Initializing plots...")
            progress.setValue(30)
            self.init_plots(main_channels, tacho_channels)

            # --- 5. Process Data ---
            progress.setLabelText("Processing data...")
            progress.setValue(40)

            # Pre-allocate lists for channel data
            channel_data_buffers = [[] for _ in range(total_channels)]
            all_timestamps_list = [] # List of arrays for timestamps from each message

            # Iterate through filtered messages and process data
            num_msgs = len(filtered_messages)
            for i, msg in enumerate(filtered_messages):
                if progress.wasCanceled():
                    break
                # Update progress from 40% to 80%
                progress.setValue(40 + int(35 * i / max(num_msgs, 1)))
                QApplication.processEvents()

                # --- CRITICAL: Validate message data AGAIN using validated values ---
                # Use the validated values stored in the message dict
                validated_samples_per_channel_msg = msg.get('_validated_samplingSize')
                validated_sample_rate_msg = msg.get('_validated_samplingRate')

                # Re-check message length based on the confirmed structure
                # Use the validated values from THIS specific message
                expected_len_for_this_msg = validated_samples_per_channel_msg * total_channels
                flattened_data = msg.get("message", [])
                if not isinstance(flattened_data, (list, np.ndarray)) or len(flattened_data) != expected_len_for_this_msg:
                    logging.warning(f"Skipping message {msg.get('frameIndex')} during processing due to data length mismatch. "
                                    f"Expected (validated) {expected_len_for_this_msg}, got {len(flattened_data)}")
                    continue # Skip this message, continue with others

                msg_created_at_dt = datetime.fromisoformat(msg['createdAt'].replace('Z', '+00:00'))
                msg_created_at_ts = msg_created_at_dt.timestamp()
                # --- END CRITICAL VALIDATION ---

                # --- Unflatten and Process ---
                # 1. Main Channels (Interleaved)
                main_data_start_idx = 0
                # --- FIXED LINE: Now guaranteed safe for THIS message ---
                main_data_end_idx = main_channels * validated_samples_per_channel_msg # <-- Use validated value
                # --- END FIXED LINE ---
                main_data_flat = flattened_data[main_data_start_idx:main_data_end_idx]

                for ch in range(main_channels):
                    # Extract interleaved data for channel `ch`
                    ch_data = main_data_flat[ch::main_channels] # Every `main_channels`-th element starting at `ch`
                    channel_data_buffers[ch].extend(ch_data)

                # 2. Tacho Channels (Non-interleaved, after main)
                tacho_data_start_idx = main_data_end_idx
                for tch in range(tacho_channels):
                    tch_start = tacho_data_start_idx + tch * validated_samples_per_channel_msg # <-- Use validated value
                    tch_end = tch_start + validated_samples_per_channel_msg # <-- Use validated value
                    tacho_data = flattened_data[tch_start:tch_end]
                    channel_data_buffers[main_channels + tch].extend(tacho_data)

                # 3. Generate timestamps for this message block (using validated sample rate)
                # np.arange is good for large arrays and precision
                msg_times = msg_created_at_ts + np.arange(validated_samples_per_channel_msg) / validated_sample_rate_msg # <-- Use validated value
                all_timestamps_list.append(msg_times)

            # --- 6. Combine and Filter Data ---
            progress.setLabelText("Combining and filtering data...")
            progress.setValue(80)
            if not all_timestamps_list:
                 raise ValueError("No valid data found in selected messages after processing")

            # Concatenate all data and times
            combined_times = np.concatenate(all_timestamps_list)
            combined_data = [np.array(buf) for buf in channel_data_buffers]

            # Create a mask for the overall selected time range
            combined_time_mask = (combined_times >= self.start_time) & (combined_times <= self.end_time)
            if not np.any(combined_time_mask):
                 raise ValueError("No data points found within the final combined time range")

            # Apply mask
            filtered_times = combined_times[combined_time_mask]
            filtered_data = [d[combined_time_mask] for d in combined_data]

            # --- 7. Calibration and Downsampling ---
            progress.setLabelText("Applying calibration and downsampling...")
            progress.setValue(90)

            processed_data = []
            total_points = len(filtered_times)
            needs_downsampling = total_points > self.max_points_to_plot
            downsample_factor = int(np.ceil(total_points / self.max_points_to_plot)) if needs_downsampling else 1

            if needs_downsampling:
                logging.debug(f"Downsampling data by factor of {downsample_factor} (from {total_points} to ~{total_points // downsample_factor} points)")

            # Calibrate Main Channels (Aligning with C# logic)
            for ch in range(main_channels):
                raw_data = filtered_data[ch] # This should be raw counts or volts
                # If data is raw ushort (0-65535), convert to volts first:
                # volts = raw_data * self.scaling_factor
                # However, if the DB already provides volts or scaled values, use them directly.
                # Assuming data from DB message is already in a form suitable for calibration (e.g., volts).
                volts = raw_data

                channel_name = self.channel_names[ch] if ch < len(self.channel_names) else f"Channel {ch + 1}"
                props = self.channel_properties.get(channel_name, {
                    "unit": "mil", "correctionValue": 1.0, "gain": 1.0, "ConvertedSensitivity": 1.0
                })

                try:
                    # C# logic: baseValue = volts * Corr * Gain / Sensitivity -> Result is in mm.
                    # Python: Use ConvertedSensitivity as per your DB structure.
                    # The C# code uses props.Sensitivity directly. Your DB has ConvertedSensitivity.
                    # Assuming ConvertedSensitivity is the correct value to use here (could be Sensitivity or adjusted).
                    calibrated_data = volts * (props["correctionValue"] * props["gain"]) / props["ConvertedSensitivity"]
                except (ZeroDivisionError, TypeError) as cal_error:
                    logging.error(f"Calibration error for channel {channel_name}: {cal_error}. Using raw volts.")
                    calibrated_data = volts # Fallback

                # --- CORRECTED UNIT CONVERSION (Align with C#) ---
                # C# logic inside switch(props.Unit.ToLower()):
                # case "mil": calibratedData[j] = baseValue / 25.4; (mm to mil)
                # case "um": calibratedData[j] = baseValue * 1000; (mm to um)
                # case "mm": calibratedData[j] = baseValue; (already mm)
                # This implies `baseValue` (our `calibrated_data` before unit conversion) is in mm.
                unit = props["unit"]
                if unit == "mil":
                    calibrated_data = calibrated_data / 25.4 # mm to mil
                elif unit == "um":
                    calibrated_data = calibrated_data * 1000 # mm to um
                # elif unit == "mm": pass # Already mm
                # --- END CORRECTED UNIT CONVERSION ---

                if needs_downsampling and len(calibrated_data) > 0:
                    calibrated_data = self.downsample_array(calibrated_data, downsample_factor)
                processed_data.append(calibrated_data)

            # Handle Tacho Channels (No calibration, potentially scaling if needed, but C# doesn't)
            for ch in range(main_channels, total_channels):
                 tacho_data = filtered_data[ch]
                 # C# applies no scaling to tacho channels. They are raw.
                 # Your previous code had `data = data * 10` for Frequency (ch == main_channels).
                 # This is likely incorrect unless there's a specific reason.
                 # Let's remove arbitrary scaling and match C#.
                 processed_tacho_data = tacho_data # Raw tacho data
                 if needs_downsampling and len(processed_tacho_data) > 0:
                     processed_tacho_data = self.downsample_array(processed_tacho_data, downsample_factor)
                 processed_data.append(processed_tacho_data)

            # Downsample times if needed
            if needs_downsampling and len(filtered_times) > 0:
                filtered_times = self.downsample_array(filtered_times, downsample_factor)

            # --- 8. Plotting ---
            progress.setLabelText("Updating plots...")
            progress.setValue(95)

            # Clear plots before adding new data
            # Note: We clear the PlotDataItem, not the widget, and re-add data.
            # Or we can just call setData. Let's use setData for clarity and efficiency.
            # for widget in self.plot_widgets:
            #     widget.clear() # This clears everything including axes, legend, items
            #     # Re-add legend, grid, axes if needed (but they are usually persistent)
            #     # widget.addLegend()
            #     # widget.showGrid(x=True, y=True)
            #     # axis = TimeAxisItem(orientation='bottom')
            #     # widget.setAxisItems({'bottom': axis})
            #     # Re-apply Y range for tacho if needed
            #     y_axis_label = widget.getAxis('left').labelText
            #     if "Frequency" in y_axis_label or "Trigger" in y_axis_label:
            #          widget.setYRange(-0.5, 1.5, padding=0)

            # --- CRITICAL FIX: Ensure data and times are NumPy arrays ---
            # Assign processed data and times
            self.data = [np.asarray(d, dtype=np.float64) for d in processed_data] # Ensure float64
            self.channel_times = np.asarray(filtered_times, dtype=np.float64) # Ensure float64

            # --- CRITICAL FIX: Check for matching lengths before plotting ---
            if len(self.channel_times) == 0:
                 raise ValueError("No valid time data to plot after filtering/downsampling.")

            # Plot each channel
            for ch in range(self.num_plots):
                # Check if we have data for this channel and times
                if ch < len(self.data) and len(self.data[ch]) > 0 and len(self.channel_times) > 0:
                    # --- CRITICAL CHECK: Lengths must match ---
                    if len(self.data[ch]) != len(self.channel_times):
                        logging.error(f"Data length mismatch for plot {ch}: data={len(self.data[ch])}, times={len(self.channel_times)}")
                        # Optionally, truncate/pad or skip
                        min_len = min(len(self.data[ch]), len(self.channel_times))
                        plot_y_data = self.data[ch][:min_len]
                        plot_x_times = self.channel_times[:min_len]
                        logging.warning(f"Truncated data for plot {ch} to length {min_len} to match.")
                    else:
                        plot_y_data = self.data[ch]
                        plot_x_times = self.channel_times

                    # Determine channel name for legend
                    if ch < len(self.channel_names):
                        channel_name = self.channel_names[ch]
                    elif ch == len(self.channel_names):
                        channel_name = "Frequency"
                    elif ch == len(self.channel_names) + 1:
                        channel_name = "Trigger"
                    else:
                        channel_name = f"Channel {ch + 1}"

                    # --- CRITICAL FIX: Use the correct PlotDataItem (self.plots[ch]) ---
                    # And ensure data is NumPy array of correct type
                    pen = mkPen(color=self.plot_colors[ch % len(self.plot_colors)], width=2)
                    
                    # --- Ensure data is float64 for pyqtgraph ---
                    plot_x_times_f64 = np.asarray(plot_x_times, dtype=np.float64)
                    plot_y_data_f64 = np.asarray(plot_y_data, dtype=np.float64)
                    
                    # --- Use setData for efficient update ---
                    self.plots[ch].setData(plot_x_times_f64, plot_y_data_f64, pen=pen, name=channel_name)
                    
                    # --- Update the plot widget's axes and ranges ---
                    plot_widget = self.plot_widgets[ch]
                    # Set X range to selected time window
                    plot_widget.setXRange(self.start_time, self.end_time, padding=0.02)
                    # Enable auto-range for Y to fit the data
                    plot_widget.enableAutoRange(axis='y')
                    logging.debug(f"Plotted channel {ch} ({channel_name}): {len(plot_y_data_f64)} points")
                else:
                    logging.warning(f"Skipping plot {ch}: data length={len(self.data[ch]) if ch < len(self.data) else 'N/A'}, times length={len(self.channel_times)}")
                    # Clear the plot if no data
                    if ch < len(self.plots):
                         self.plots[ch].setData([], []) # Clear plot data


            # --- 9. Trigger Lines (Optional) ---
            # Add vertical lines on Trigger plot where Trigger == 1
            # Find the Trigger plot index (should be the last one if tacho_channels >= 2)
            if self.tacho_channels_count >= 2 and self.num_plots > 1:
                 trigger_plot_idx = self.num_plots - 1 # Assuming last plot is Trigger
                 # Ensure we have data for the trigger plot
                 if trigger_plot_idx < len(self.data) and len(self.data[trigger_plot_idx]) > 0 and len(self.channel_times) > 0:
                     trigger_data = self.data[trigger_plot_idx]
                     trigger_times = self.channel_times
                     # Check lengths again for trigger data
                     if len(trigger_data) == len(trigger_times):
                         # Clear previous trigger lines
                         for line in self.trigger_lines:
                             if line.scene() is not None:
                                 line.scene().removeItem(line)
                         self.trigger_lines = []

                         # Find indices where trigger is high (assuming 1 or > 0.5 for robustness)
                         # Use numpy for efficient finding
                         trigger_indices = np.where(trigger_data >= 0.5)[0]
                         logging.debug(f"Found {len(trigger_indices)} trigger events.")
                         for idx in trigger_indices:
                             if idx < len(trigger_times):
                                 line = InfiniteLine(
                                     pos=trigger_times[idx],
                                     angle=90,
                                     movable=False,
                                     pen=mkPen('k', width=1, style=Qt.SolidLine) # Solid line for triggers
                                 )
                                 self.plot_widgets[trigger_plot_idx].addItem(line)
                                 self.trigger_lines.append(line) # Keep reference
                     else:
                         logging.error(f"Trigger data length mismatch: data={len(trigger_data)}, times={len(trigger_times)}")


            progress.setValue(100)
            progress.close()
            success_msg = f"Time Report ({self.model_name}): Successfully plotted {self.num_plots} plots for {filename}"
            logging.info(success_msg)
            if self.console:
                self.console.append_to_console(success_msg)

        except Exception as e:
            logging.error(f"Error plotting data for {filename}: {str(e)}", exc_info=True) # Log full traceback
            self.clear_plots()
            progress.setValue(100)
            progress.close()
            error_msg = f"Error plotting data for {filename}: {str(e)}"
            QMessageBox.critical(self.widget, "Plot Error", error_msg)
            if self.console:
                self.console.append_to_console(error_msg)

    def clear_plots(self):
        # Clear data lists/arrays
        self.data = []
        self.channel_times = np.array([])
        # Clear trigger lines and remove them from the scene
        for line in self.trigger_lines:
             if line.scene() is not None:
                 line.scene().removeItem(line)
        self.trigger_lines = []

        # Clear plot data items (more efficient than clearing widgets)
        for plot_data_item in self.plots:
             plot_data_item.setData([], []) # Clear the data from the plot item

        # Clear and delete plot widgets from the layout (if re-initializing)
        while self.scroll_layout.count():
            child = self.scroll_layout.takeAt(0)
            if child.widget():
                widget = child.widget()
                widget.setParent(None) # Important for cleanup in Qt
                widget.deleteLater() # Schedule for deletion

        # Clear references to prevent memory leaks and errors
        self.plot_widgets = []
        self.plots = [] # Clear the list of PlotDataItems
        self.vlines = []
        self.proxies = []
        self.trackers = []
        self.num_plots = 0
        self.num_channels = 0
        self.tacho_channels_count = 0
        # Reset data-dependent variables
        self.sample_rate = None
        self.samples_per_channel = None
        logging.debug("Cleared all plots and associated data")

    def mouse_enter(self, idx):
        self.active_line_idx = idx
        # Ensure idx is valid before accessing vlines
        if 0 <= idx < len(self.vlines):
            self.vlines[idx].setVisible(True)

    def mouse_leave(self, idx):
        self.active_line_idx = None
        for vline in self.vlines:
            vline.setVisible(False)

    def mouse_moved(self, evt, idx):
        if self.active_line_idx is None:
            return
        # Ensure idx is valid
        if not (0 <= idx < len(self.plot_widgets)):
             return
        pos = evt[0]
        if not self.plot_widgets[idx].sceneBoundingRect().contains(pos):
            return
        mouse_point = self.plot_widgets[idx].plotItem.vb.mapSceneToView(pos)
        x = mouse_point.x() # This is the timestamp

        # Clamp x to the actual data range for better UX
        if len(self.channel_times) > 0:
            x = np.clip(x, self.channel_times[0], self.channel_times[-1])
        else:
            # If no data, don't show line
            for vline in self.vlines:
                vline.setVisible(False)
            return

        # Update all vertical lines
        for vline in self.vlines:
            vline.setPos(x)
            vline.setVisible(True)

    def get_widget(self):
        return self.widget

    def cleanup(self):
        try:
            self.clear_plots()
            if self.widget:
                self.widget.setParent(None)
                self.widget.deleteLater()
            logging.debug("TimeReportFeature cleaned up")
        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}", exc_info=True)
            if self.console:
                self.console.append_to_console(f"Error during cleanup: {str(e)}")
