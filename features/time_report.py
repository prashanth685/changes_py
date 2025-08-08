from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton, QScrollArea, QDateTimeEdit, QGridLayout, QProgressDialog)
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QPropertyAnimation, QEasingCurve, Qt, QDateTime, QRect, pyqtSignal, QEvent, QObject, QTimer
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor
import pyqtgraph as pg
from pyqtgraph import PlotWidget, mkPen, AxisItem, InfiniteLine, SignalProxy
import numpy as np
from datetime import datetime, timedelta
import logging

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
        self.setStyleSheet("""
            QWidget {
                background-color: #d1d6d9;
            }
        """)

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
        groove_rect = QRect(int(10), int(10), int(self.width() - 20), int(8))
        painter.setPen(QPen(QColor("#1a73e8")))
        painter.setBrush(QColor("#34495e"))
        painter.drawRoundedRect(groove_rect, 4, 4)
        left_pos = int(self._value_to_pos(self.left_value))
        right_pos = int(self._value_to_pos(self.right_value))
        selected_rect = QRect(left_pos, int(10), int(right_pos - left_pos), int(8))
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
        return [datetime.fromtimestamp(v).strftime('%Y-%m-%d\n%H:%M:%S') for v in values]

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
    def __init__(self, parent, db, project_name, channel=None, model_name=None, console=None):
        self.parent = parent
        self.db = db
        self.project_name = project_name
        self.channel = channel
        self.model_name = model_name
        self.console = console
        self.widget = QWidget(self.parent)
        self.plot_widgets = []
        self.plots = []
        self.data = []
        self.channel_times = []
        self.tacho_times = []
        self.vlines = []
        self.proxies = []
        self.trackers = []
        self.trigger_lines = []
        self.active_line_idx = None
        self.num_channels = 0  # Will be set dynamically
        self.num_plots = 0     # Will be set dynamically
        self.tacho_channels_count = 2  # Matches time_view.py
        self.sample_rate = None
        self.filenames = []
        self.selected_filename = None
        self.file_start_time = None
        self.file_end_time = None
        self.start_time = None
        self.end_time = None
        self.use_full_range = True
        self.scaling_factor = 3.3 / 65535  # Matches time_view.py
        self.init_ui_deferred()

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
        self.start_time_edit.setDisplayFormat("HH:mm:ss")
        self.start_time_edit.dateTimeChanged.connect(self.validate_time_range)

        end_time_label = QLabel("Select End Time:")
        end_time_label.setStyleSheet("color: black; font-size: 14px; font: bold")
        self.end_time_edit = QDateTimeEdit()
        self.end_time_edit.setStyleSheet("background-color: #34495e; color: white; border: 2px solid black; padding: 15px; font: bold; width: 200px")
        self.end_time_edit.setDisplayFormat("HH:mm:ss")
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
        self.time_slider.setEnabled(False)
        self.ok_button.setEnabled(False)

    def load_data_async(self):
        try:
            self.refresh_filenames()
        except Exception as e:
            logging.error(f"Error in async data loading: {str(e)}")
            if self.console:
                self.console.append_to_console(f"Error loading Time Report data: {str(e)}")

    def init_plots(self, num_channels, tacho_channels_count):
        self.num_channels = num_channels
        self.tacho_channels_count = tacho_channels_count
        self.num_plots = num_channels + tacho_channels_count

        # Clear existing plots
        self.plot_widgets = []
        self.plots = []
        self.data = []
        self.vlines = []
        self.proxies = []
        self.trackers = []
        self.trigger_lines = []

        # Clear scroll content
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)

        colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'b', '#FF4500', '#32CD32', '#00CED1', '#FFD700', '#FF69B4', '#8A2BE2', '#FF6347', '#20B2AA', '#ADFF2F', '#9932CC', '#FF7F50', '#00FA9A', '#9400D3']
        for i in range(self.num_plots):
            plot_widget = PlotWidget(axisItems={'bottom': TimeAxisItem(orientation='bottom')}, background='w')
            plot_widget.setFixedHeight(250)
            plot_widget.setMinimumWidth(0)
            if i < self.num_channels:
                plot_widget.setLabel('left', f'CH{i+1} Value')
            elif i == self.num_channels:
                plot_widget.setLabel('left', 'Tacho Frequency')
            else:
                plot_widget.setLabel('left', f'Tacho Trigger {i - self.num_channels}')
                plot_widget.setYRange(-0.5, 1.5, padding=0)
            plot_widget.setLabel('bottom', 'Time')
            plot_widget.showGrid(x=True, y=True)
            plot_widget.addLegend()
            pen = mkPen(color=colors[i % len(colors)], width=2)
            plot = plot_widget.plot([], [], pen=pen)
            self.plots.append(plot)
            self.plot_widgets.append(plot_widget)
            self.data.append([])

            vline = InfiniteLine(angle=90, movable=False, pen=mkPen('r', width=2))
            vline.setVisible(False)
            plot_widget.addItem(vline)
            self.vlines.append(vline)

            if i == self.num_plots - 1:
                self.trigger_lines = []
            else:
                self.trigger_lines.append(None)

            proxy = SignalProxy(plot_widget.scene().sigMouseMoved, rateLimit=60, slot=lambda evt, idx=i: self.mouse_moved(evt, idx))
            self.proxies.append(proxy)

            tracker = MouseTracker(plot_widget.viewport(), i, self)
            plot_widget.viewport().installEventFilter(tracker)
            self.trackers.append(tracker)

            self.scroll_layout.addWidget(plot_widget)
            QApplication.processEvents()

    def animate_button_press(self):
        animation = QPropertyAnimation(self.ok_button, b"styleSheet")
        animation.setDuration(200)
        animation.setStartValue("background-color: #1a73e8;")
        animation.setEndValue("background-color: #155ab6;")
        animation.setEasingCurve(QEasingCurve.InOutQuad)
        animation.start()

    def refresh_filenames(self):
        try:
            self.filenames = self.db.get_distinct_filenames(self.project_name, self.model_name)
            self.file_combo.clear()
            if not self.filenames:
                self.file_combo.addItem("No Files Available")
                self.start_time_label.setText("File Start Time: N/A")
                self.stop_time_label.setText("File Stop Time: N/A")
                self.start_time_edit.setEnabled(False)
                self.end_time_edit.setEnabled(False)
                self.time_slider.setEnabled(False)
                self.ok_button.setEnabled(False)
                if self.console:
                    self.console.append_to_console("No saved files found for this project.")
            else:
                self.file_combo.addItems(self.filenames)
                self.start_time_edit.setEnabled(True)
                self.end_time_edit.setEnabled(True)
                self.time_slider.setEnabled(True)
                self.ok_button.setEnabled(True)
                self.file_combo.setEnabled(True)
                self.update_time_labels(self.file_combo.currentText())
                if self.console:
                    self.console.append_to_console(f"Refreshed filenames: {len(self.filenames)} found")
        except Exception as e:
            logging.error(f"Error refreshing filenames: {str(e)}")
            self.file_combo.clear()
            self.file_combo.addItem("Error Loading Files")
            self.start_time_label.setText("File Start Time: N/A")
            self.stop_time_label.setText("File Stop Time: N/A")
            self.start_time_edit.setEnabled(False)
            self.end_time_edit.setEnabled(False)
            self.time_slider.setEnabled(False)
            self.ok_button.setEnabled(False)
            if self.console:
                self.console.append_to_console(f"Error refreshing filenames: {str(e)}")

    def on_filename_selected(self, filename):
        self.selected_filename = filename
        self.use_full_range = True
        self.update_time_labels(filename)
        self.clear_plots()
        if filename and filename not in ["No Files Available", "Error Loading Files"]:
            self.ok_button.setEnabled(True)
        else:
            self.ok_button.setEnabled(False)
            self.file_start_time = None
            self.file_end_time = None
            self.start_time = None
            self.end_time = None

    def update_time_labels(self, filename):
        if not filename or filename in ["No Files Available", "Error Loading Files"]:
            self.start_time_label.setText("File Start Time: N/A")
            self.stop_time_label.setText("File Stop Time: N/A")
            self.start_time_edit.setEnabled(False)
            self.end_time_edit.setEnabled(False)
            self.time_slider.setEnabled(False)
            self.ok_button.setEnabled(False)
            self.file_start_time = None
            self.file_end_time = None
            return

        try:
            messages = self.db.get_timeview_messages(
                self.project_name,
                model_name=self.model_name,
                filename=filename
            )
            if not messages:
                self.start_time_label.setText("File Start Time: N/A")
                self.stop_time_label.setText("File Stop Time: N/A")
                self.start_time_edit.setEnabled(False)
                self.end_time_edit.setEnabled(False)
                self.time_slider.setEnabled(False)
                self.ok_button.setEnabled(False)
                self.file_start_time = None
                self.file_end_time = None
                if self.console:
                    self.console.append_to_console(f"No data found for file: {filename}")
                return

            timestamps = []
            for msg in messages:
                try:
                    created_at = datetime.fromisoformat(msg['createdAt'].replace('Z', '+00:00')).timestamp()
                    timestamps.append(created_at)
                except Exception as e:
                    logging.warning(f"Invalid timestamp in {filename}: {e}")
                    if self.console:
                        self.console.append_to_console(f"Invalid timestamp in {filename}: {e}")
                    continue

            if timestamps:
                self.file_start_time = datetime.fromtimestamp(min(timestamps))
                self.file_end_time = datetime.fromtimestamp(max(timestamps))
                self.start_time = min(timestamps)
                self.end_time = max(timestamps)
                self.start_time_label.setText(f"File Start Time: {self.file_start_time.strftime('%H:%M:%S')}")
                self.stop_time_label.setText(f"File Stop Time: {self.file_end_time.strftime('%H:%M:%S')}")
                self.start_time_edit.setDateTime(QDateTime(self.file_start_time))
                self.end_time_edit.setDateTime(QDateTime(self.file_end_time))
                self.time_slider.setRange(0, 1000)
                self.time_slider.setValues(0, 1000)
                self.start_time_edit.setEnabled(True)
                self.end_time_edit.setEnabled(True)
                self.time_slider.setEnabled(True)
                self.ok_button.setEnabled(True)
            else:
                self.start_time_label.setText("File Start Time: N/A")
                self.stop_time_label.setText("File Stop Time: N/A")
                self.start_time_edit.setEnabled(False)
                self.end_time_edit.setEnabled(False)
                self.time_slider.setEnabled(False)
                self.ok_button.setEnabled(False)
                self.file_start_time = None
                self.file_end_time = None
        except Exception as e:
            logging.error(f"Error updating time labels for {filename}: {e}")
            self.start_time_label.setText("File Start Time: N/A")
            self.stop_time_label.setText("File Stop Time: N/A")
            self.start_time_edit.setEnabled(False)
            self.end_time_edit.setEnabled(False)
            self.time_slider.setEnabled(False)
            self.ok_button.setEnabled(False)
            if self.console:
                self.console.append_to_console(f"Error loading time data for {filename}: {str(e)}")
            self.file_start_time = None
            self.file_end_time = None

    def update_time_from_slider(self):
        if not self.file_start_time or not self.file_end_time:
            return

        total_duration = (self.file_end_time - self.file_start_time).total_seconds()
        if total_duration <= 0:
            return

        left_pos, right_pos = self.time_slider.getValues()
        if left_pos > right_pos:
            left_pos, right_pos = right_pos, left_pos
            self.time_slider.setValues(left_pos, right_pos)

        left_fraction = left_pos / 1000.0
        right_fraction = right_pos / 1000.0

        start_seconds = left_fraction * total_duration
        end_seconds = right_fraction * total_duration
        start_time = self.file_start_time + timedelta(seconds=start_seconds)
        end_time = self.file_start_time + timedelta(seconds=end_seconds)

        self.start_time = start_time.timestamp()
        self.end_time = end_time.timestamp()

        self.start_time_edit.blockSignals(True)
        self.end_time_edit.blockSignals(True)
        self.start_time_edit.setDateTime(QDateTime(start_time))
        self.end_time_edit.setDateTime(QDateTime(end_time))
        self.start_time_edit.blockSignals(False)
        self.end_time_edit.blockSignals(False)

        self.use_full_range = (left_pos == 0 and right_pos == 1000)
        self.validate_time_range()

    def validate_time_range(self):
        start_time = self.start_time_edit.dateTime().toPyDateTime()
        end_time = self.end_time_edit.dateTime().toPyDateTime()

        self.start_time = start_time.timestamp()
        self.end_time = end_time.timestamp()

        if start_time >= end_time:
            self.ok_button.setEnabled(False)
            if self.console:
                self.console.append_to_console("Error: Start time must be before end time.")
        else:
            self.ok_button.setEnabled(True)
            if self.file_start_time and self.file_end_time:
                total_duration = (self.file_end_time - self.file_start_time).total_seconds()
                if total_duration > 0:
                    start_offset = (start_time - self.file_start_time).total_seconds()
                    end_offset = (end_time - self.file_start_time).total_seconds()
                    left_pos = (start_offset / total_duration) * 1000
                    right_pos = (end_offset / total_duration) * 1000
                    self.time_slider.blockSignals(True)
                    self.time_slider.setValues(left_pos, right_pos)
                    self.time_slider.blockSignals(False)

    def clear_plots(self):
        for plot in self.plots:
            plot.setData([], [])
        for widget in self.plot_widgets:
            widget.clear()
            widget.addLegend()
            widget.showGrid(x=True, y=True)
            if widget.getAxis('left').labelText.startswith('Tacho Trigger'):
                widget.setYRange(-0.5, 1.5, padding=0)
        self.data = [[] for _ in range(self.num_plots)]
        self.channel_times = []
        self.tacho_times = []
        self.trigger_lines = [None] * (self.num_plots - 1) + [[]]
        logging.debug("Cleared all plots")

    def plot_data(self):
        """Plot data for the selected file and time range with a progress dialog."""
        filename = self.file_combo.currentText()
        if not filename or filename in ["No Files Available", "Error Loading Files"]:
            self.clear_plots()
            if self.console:
                self.console.append_to_console("No valid file selected to plot.")
            return

        # Initialize progress dialog
        progress = QProgressDialog("Loading and plotting data...", "Cancel", 0, 100, self.widget)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        try:
            # Fetch messages
            progress.setLabelText("Fetching data from database...")
            progress.setValue(10)
            messages = self.db.get_timeview_messages(
                self.project_name,
                model_name=self.model_name,
                filename=filename
            )
            if not messages:
                self.clear_plots()
                progress.setValue(100)
                progress.close()
                if self.console:
                    self.console.append_to_console(f"No data found for filename {filename}")
                return

            # Sort messages by timestamp to ensure chronological order
            progress.setLabelText("Sorting messages...")
            progress.setValue(15)
            messages.sort(key=lambda x: datetime.fromisoformat(x['createdAt'].replace('Z', '+00:00')).timestamp())

            # Determine number of channels dynamically
            first_msg = messages[0]
            num_channels = first_msg.get('numberOfChannels', 0)
            tacho_channels_count = self.tacho_channels_count
            if not num_channels or len(first_msg['message']['channel_data']) != num_channels:
                self.clear_plots()
                progress.setValue(100)
                progress.close()
                if self.console:
                    self.console.append_to_console(f"Invalid channel data in {filename}")
                return

            # Initialize plots with dynamic channel counts
            progress.setLabelText("Initializing plots...")
            progress.setValue(20)
            self.init_plots(num_channels, tacho_channels_count)

            if self.use_full_range:
                self.start_time = self.file_start_time.timestamp()
                self.end_time = self.file_end_time.timestamp()
                self.start_time_edit.setDateTime(QDateTime(self.file_start_time))
                self.end_time_edit.setDateTime(QDateTime(self.file_end_time))
                self.time_slider.setValues(0, 1000)

            if self.start_time >= self.end_time:
                self.clear_plots()
                progress.setValue(100)
                progress.close()
                if self.console:
                    self.console.append_to_console("Error: Start time must be before end time.")
                return

            # Filter messages by time range
            progress.setLabelText("Filtering messages by time range...")
            progress.setValue(25)
            filtered_messages = [
                msg for msg in messages
                if self.start_time <= datetime.fromisoformat(msg['createdAt'].replace('Z', '+00:00')).timestamp() <= self.end_time
            ]

            if not filtered_messages:
                self.clear_plots()
                progress.setValue(100)
                progress.close()
                if self.console:
                    self.console.append_to_console(f"No data within time range for filename {filename}")
                return

            # Initialize data arrays
            progress.setLabelText("Preparing data arrays...")
            progress.setValue(30)
            channel_data_agg = [[] for _ in range(self.num_channels)]
            tacho_freq_agg = []
            tacho_trigger_agg = []
            channel_times_agg = []
            tacho_times_agg = []
            self.sample_rate = filtered_messages[0].get('samplingRate', 4096)
            total_messages = len(filtered_messages)

            # Process messages
            progress.setLabelText("Processing messages...")
            for i, msg in enumerate(filtered_messages):
                progress.setValue(30 + int((i / total_messages) * 50))  # 30% to 80%
                QApplication.processEvents()
                if progress.wasCanceled():
                    self.clear_plots()
                    progress.close()
                    return

                created_at = datetime.fromisoformat(msg['createdAt'].replace('Z', '+00:00')).timestamp()
                channel_data = msg['message']['channel_data']
                tacho_freq = msg['message']['tacho_freq']
                tacho_trigger = msg['message']['tacho_trigger']
                channel_samples = msg.get('samplingSize', 4096)
                tacho_samples = len(tacho_freq)

                if len(channel_data) != self.num_channels or len(tacho_freq) != tacho_samples or len(tacho_trigger) != tacho_samples:
                    if self.console:
                        self.console.append_to_console(f"Data length mismatch in message at {created_at} for {filename}")
                    continue

                # Calculate time arrays in ascending order
                channel_time_step = 1.0 / self.sample_rate
                tacho_time_step = 1.0 / self.sample_rate
                channel_times = np.array([
                    created_at + i * channel_time_step
                    for i in range(channel_samples)
                ])
                tacho_times = np.array([
                    created_at + i * tacho_time_step
                    for i in range(tacho_samples)
                ])

                # Filter by time range
                channel_mask = (channel_times >= self.start_time) & (channel_times <= self.end_time)
                tacho_mask = (tacho_times >= self.start_time) & (tacho_times <= self.end_time)

                # Ensure data is appended in chronological order
                for ch in range(self.num_channels):
                    filtered_data = np.array(channel_data[ch])[channel_mask] * self.scaling_factor
                    if len(filtered_data) > 0:
                        channel_data_agg[ch].extend(filtered_data)
                filtered_tacho_freq = np.array(tacho_freq)[tacho_mask] / 100  # Matches time_view.py scaling
                if len(filtered_tacho_freq) > 0:
                    tacho_freq_agg.extend(filtered_tacho_freq)
                filtered_tacho_trigger = np.array(tacho_trigger)[tacho_mask]
                if len(filtered_tacho_trigger) > 0:
                    tacho_trigger_agg.extend(filtered_tacho_trigger)
                filtered_channel_times = channel_times[channel_mask]
                if len(filtered_channel_times) > 0:
                    channel_times_agg.extend(filtered_channel_times)
                filtered_tacho_times = tacho_times[tacho_mask]
                if len(filtered_tacho_times) > 0:
                    tacho_times_agg.extend(filtered_tacho_times)

            # Sort aggregated times to ensure chronological order
            if channel_times_agg:
                channel_sort_indices = np.argsort(channel_times_agg)
                channel_times_agg = np.array(channel_times_agg)[channel_sort_indices]
                for ch in range(self.num_channels):
                    if channel_data_agg[ch]:
                        channel_data_agg[ch] = np.array(channel_data_agg[ch])[channel_sort_indices]
            if tacho_times_agg:
                tacho_sort_indices = np.argsort(tacho_times_agg)
                tacho_times_agg = np.array(tacho_times_agg)[tacho_sort_indices]
                tacho_freq_agg = np.array(tacho_freq_agg)[tacho_sort_indices]
                tacho_trigger_agg = np.array(tacho_trigger_agg)[tacho_sort_indices]

            # Assign data to plots
            progress.setLabelText("Assigning data to plots...")
            progress.setValue(80)
            for ch in range(self.num_channels):
                self.data[ch] = np.array(channel_data_agg[ch])
            self.data[self.num_channels] = np.array(tacho_freq_agg)
            if self.tacho_channels_count >= 2:
                self.data[self.num_plots - 1] = np.array(tacho_trigger_agg)
            self.channel_times = np.array(channel_times_agg)
            self.tacho_times = np.array(tacho_times_agg)

            # Clear and update plots
            progress.setLabelText("Updating plots...")
            progress.setValue(90)
            for widget in self.plot_widgets:
                widget.clear()
                widget.addLegend()
                widget.showGrid(x=True, y=True)
                if widget.getAxis('left').labelText.startswith('Tacho Trigger'):
                    widget.setYRange(-0.5, 1.5, padding=0)

            colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w', '#FF4500', '#32CD32', '#00CED1', '#FFD700', '#FF69B4', '#8A2BE2', '#FF6347', '#20B2AA', '#ADFF2F', '#9932CC', '#FF7F50', '#00FA9A', '#9400D3']
            for ch in range(self.num_plots):
                times = self.tacho_times if ch >= self.num_channels else self.channel_times
                if len(self.data[ch]) > 0 and len(times) > 0:
                    pen = mkPen(color=colors[ch % len(colors)], width=2)
                    self.plots[ch] = self.plot_widgets[ch].plot(times, self.data[ch], pen=pen)
                    self.plot_widgets[ch].setXRange(self.start_time, self.end_time, padding=0)
                    if ch < self.num_channels:
                        self.plot_widgets[ch].enableAutoRange(axis='y')
                    elif ch == self.num_channels:
                        self.plot_widgets[ch].enableAutoRange(axis='y')
                else:
                    if self.console:
                        self.console.append_to_console(f"No data for plot {ch} in time range")

            # Add trigger lines
            if self.tacho_channels_count >= 2 and len(self.data[self.num_plots - 1]) > 0 and len(self.tacho_times) > 0:
                trigger_indices = np.where(self.data[self.num_plots - 1] == 1)[0]
                self.trigger_lines = [None] * (self.num_plots - 1) + [[]]
                for idx in trigger_indices:
                    if idx < len(self.tacho_times):
                        line = InfiniteLine(
                            pos=self.tacho_times[idx],
                            angle=90,
                            movable=False,
                            pen=mkPen('k', width=2, style=Qt.SolidLine)
                        )
                        self.plot_widgets[self.num_plots - 1].addItem(line)
                        self.trigger_lines[self.num_plots - 1].append(line)

            progress.setValue(100)
            progress.close()
            if self.console:
                self.console.append_to_console(f"Time Report ({self.model_name}): Plotted {self.num_plots} plots for {filename}")
        except Exception as e:
            logging.error(f"Error plotting data: {str(e)}")
            self.clear_plots()
            progress.setValue(100)
            progress.close()
            if self.console:
                self.console.append_to_console(f"Error plotting data: {str(e)}")

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
        times = self.tacho_times if idx >= self.num_channels else self.channel_times
        if len(times) > 0:
            if x < times[0]:
                x = times[0]
            elif x > times[-1]:
                x = times[-1]
        for vline in self.vlines:
            vline.setPos(x)
            vline.setVisible(True)

    def get_widget(self):
        return self.widget