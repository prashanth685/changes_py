from PyQt5.QtWidgets import (
    QToolBar, QAction, QWidget, QHBoxLayout, QSizePolicy, QLineEdit,
    QLabel, QDialog, QVBoxLayout, QPushButton, QGridLayout, QComboBox
)
from PyQt5.QtCore import QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon
import logging
import re
import time

class LayoutSelectionDialog(QDialog):
    def __init__(self, parent=None, current_layout=None):
        super().__init__(parent)
        self.setWindowTitle("Select Layout")
        self.setFixedSize(400, 400)
        self.setWindowFlags(Qt.Popup)
        self.selected_layout = current_layout
        self.layout_buttons = {}
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        label = QLabel("Choose a layout:")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: bold;
                color: #333;
                margin-bottom: 10px;
            }
        """)
        layout.addWidget(label)
        grid = QGridLayout()
        layouts = {
            "1x2": "⬛⬛",
            "2x2": "⬛⬛\n⬛⬛",
            "3x3": "⬛⬛⬛\n⬛⬛⬛\n⬛⬛⬛"
        }
        row, col = 0, 0
        for layout_name, icon in layouts.items():
            btn = QPushButton(icon)
            btn.setFixedSize(80, 80)
            btn.setToolTip(layout_name)
            self.layout_buttons[layout_name] = btn
            btn.clicked.connect(lambda _, l=layout_name: self.select_layout(l))
            grid.addWidget(btn, row, col)
            col += 1
            if col >= 3:
                row += 1
                col = 0
        layout.addLayout(grid)
        self.setLayout(layout)
        self.update_button_styles()

    def update_button_styles(self):
        for layout_name, btn in self.layout_buttons.items():
            btn.setStyleSheet(
                "background-color: #4a90e2; color: white; font-weight: bold;"
                if layout_name == self.selected_layout
                else "background-color: #cfd8dc;"
            )

    def select_layout(self, layout):
        self.selected_layout = layout
        self.update_button_styles()
        self.accept()

class SubToolBar(QWidget):
    # Signals to communicate with DashboardWindow
    start_saving_triggered = pyqtSignal()
    stop_saving_triggered = pyqtSignal()
    connect_mqtt_triggered = pyqtSignal()
    disconnect_mqtt_triggered = pyqtSignal()
    layout_selected = pyqtSignal(str)
    open_file_triggered = pyqtSignal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.selected_layout = "2x2"
        self.filename_edit = None
        self.saving_indicator = None
        self.timer_label = None
        self.files_combo = None
        self.open_action = None
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self.toggle_saving_indicator)
        self.blink_state = False
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.start_time = None
        self.current_project = None
        self.mqtt_connected = False
        self.is_saving = False
        self.initUI()
        self.parent.mqtt_status_changed.connect(self.update_mqtt_status)
        self.parent.project_changed.connect(self.update_project_status)
        self.parent.saving_state_changed.connect(self.update_saving_state)
        self.stop_saving_triggered.connect(self.schedule_files_combo_update)
        logging.debug("SubToolBar: Initialized with signal connections")

    def initUI(self):
        self.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #eceff1, stop:1 #cfd8dc);")
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.setLayout(layout)
        self.toolbar = QToolBar("Controls")
        self.toolbar.setFixedHeight(100)
        layout.addWidget(self.toolbar)
        self.update_subtoolbar()

    def toggle_saving_indicator(self):
        if self.saving_indicator:
            self.blink_state = not self.blink_state
            text = "rec 🔴" if self.blink_state else "rec ⚪"
            self.saving_indicator.setText(text)
            logging.debug(f"SubToolBar: Toggled saving indicator to {text}")

    def update_timer(self):
        if self.start_time and self.timer_label:
            elapsed = int(time.time() - self.start_time)
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = elapsed % 60
            self.timer_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            logging.debug(f"SubToolBar: Updated timer to {hours:02d}:{minutes:02d}:{seconds:02d}")

    def start_blinking(self):
        if self.is_saving and not self.blink_timer.isActive():
            self.blink_timer.start(500)
            self.start_time = time.time()
            self.timer.start(1000)
            if self.timer_label:
                self.timer_label.setText("00:00:00")
            if self.saving_indicator:
                self.saving_indicator.setText("rec 🔴")
            logging.debug("SubToolBar: Started blinking and timer")
        else:
            logging.debug(f"SubToolBar: Skipped starting blinking (is_saving={self.is_saving}, timer_active={self.blink_timer.isActive()})")

    def stop_blinking(self):
        if not self.is_saving and self.blink_timer.isActive():
            self.blink_timer.stop()
            self.timer.stop()
            if self.saving_indicator:
                self.saving_indicator.setText("")
            if self.timer_label:
                self.timer_label.setText("")
            self.start_time = None
            logging.debug("SubToolBar: Stopped blinking and timer")
        else:
            logging.debug(f"SubToolBar: Skipped stopping blinking (is_saving={self.is_saving}, timer_active={self.blink_timer.isActive()})")

    def update_saving_state(self, is_saving):
        if self.is_saving != is_saving:
            self.is_saving = is_saving
            if is_saving:
                self.start_blinking()
            else:
                self.stop_blinking()
            self.update_subtoolbar()
            logging.debug(f"SubToolBar: Updated saving state to {is_saving}")
        else:
            logging.debug(f"SubToolBar: Saving state unchanged (is_saving={is_saving})")
        self.refresh_filename()

    def update_mqtt_status(self, connected):
        self.mqtt_connected = connected
        self.update_subtoolbar()
        self.schedule_files_combo_update()
        logging.debug(f"SubToolBar: Updated MQTT status to {connected}")

    def update_project_status(self, project_name):
        self.current_project = project_name
        self.refresh_filename()
        self.schedule_files_combo_update()
        self.update_subtoolbar()
        logging.debug(f"SubToolBar: Updated project to {project_name}")

    def schedule_files_combo_update(self):
        """Schedule an update for the files combo with a slight delay to ensure DB commit."""
        QTimer.singleShot(1000, self.update_files_combo)  # Increased delay to 1000ms
        logging.debug("SubToolBar: Scheduled files combo update")

    def update_files_combo(self):
        if not self.files_combo:
            logging.debug("SubToolBar: Files combo not initialized yet")
            return
        self.files_combo.clear()
        try:
            if not self.current_project:
                self.files_combo.addItem("No project selected")
                self.files_combo.setEnabled(False)
                self.open_action.setEnabled(False)
                logging.debug("SubToolBar: No project selected, disabled files combo")
                return

            if not self.parent.db.is_connected():
                self.parent.db.reconnect()
                logging.debug("SubToolBar: Reconnected to database")

            model_name = self.parent.tree_view.get_selected_model()
            if not model_name:
                self.files_combo.addItem("No model selected")
                self.files_combo.setEnabled(False)
                self.open_action.setEnabled(False)
                logging.debug("SubToolBar: No model selected, disabled files combo")
                return

            # Retry fetching filenames up to 5 times with increasing delay
            for attempt in range(5):
                filenames = self.parent.db.get_distinct_filenames(self.current_project, model_name)
                if filenames:
                    break
                logging.debug(f"SubToolBar: Attempt {attempt + 1} failed to retrieve filenames, retrying after {0.2 * (attempt + 1)}s...")
                time.sleep(0.2 * (attempt + 1))  # Exponential backoff: 0.2s, 0.4s, 0.6s, 0.8s, 1.0s
            else:
                self.files_combo.addItem("No files available")
                self.files_combo.setEnabled(False)
                self.open_action.setEnabled(False)
                logging.debug("SubToolBar: No filenames found after retries, disabled files combo")
                return

            # Sort filenames numerically based on the number in "dataX"
            sorted_filenames = sorted(
                filenames,
                key=lambda x: int(re.match(r"data(\d+)", x).group(1)) if re.match(r"data(\d+)", x) else 0
            )
            self.files_combo.addItems(sorted_filenames)
            self.files_combo.setEnabled(not self.mqtt_connected)
            self.open_action.setEnabled(not self.mqtt_connected and sorted_filenames)
            logging.debug(f"SubToolBar: Populated files combo with {len(sorted_filenames)} items, enabled={not self.mqtt_connected}")
        except Exception as e:
            self.files_combo.addItem("Error loading files")
            self.files_combo.setEnabled(False)
            self.open_action.setEnabled(False)
            logging.error(f"SubToolBar: Error updating files combo: {str(e)}")

    def update_subtoolbar(self):
        logging.debug(f"SubToolBar: Updating toolbar, project: {self.current_project}, MQTT: {self.mqtt_connected}, Saving: {self.is_saving}")
        self.toolbar.clear()
        self.toolbar.setStyleSheet("""
            QToolBar { border: none; padding: 5px; spacing: 10px; }
            QToolButton { border: none; padding: 8px; border-radius: 5px; font-size: 24px; color: white; }
            QToolButton:hover { background-color: #4a90e2; }
            QToolButton:pressed { background-color: #357abd; }
            QToolButton:focus { outline: none; border: 1px solid #4a90e2; }
            QToolButton:disabled { background-color: #546e7a; color: #b0bec5; }
        """)
        self.toolbar.setIconSize(QSize(25, 25))
        self.toolbar.setMovable(False)
        self.toolbar.setFloatable(False)

        self.filename_edit = QLineEdit()
        self.filename_edit.setStyleSheet("""
            QLineEdit {
                background-color: #ffffff;
                color: #212121;
                border: 1px solid #90caf9;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 14px;
                font-weight: 500;
                min-width: 200px;
                max-width: 250px;
            }
            QLineEdit:hover { border: 1px solid #42a5f5; background-color: #f5faff; }
            QLineEdit:focus { border: 1px solid #1e88e5; background-color: #ffffff; }
            QLineEdit[readOnly="true"] { background-color: #e0e0e0; color: #616161; border: 1px solid #b0bec5; }
        """)
        is_time_view = self.parent.current_feature == "Time View"
        self.filename_edit.setReadOnly(not is_time_view)
        self.filename_edit.setEnabled(self.current_project is not None)
        self.refresh_filename()
        self.toolbar.addWidget(self.filename_edit)

        self.saving_indicator = QLabel("")
        self.saving_indicator.setStyleSheet("font-size: 20px; padding: 0px 8px;")
        self.toolbar.addWidget(self.saving_indicator)

        self.timer_label = QLabel("")
        self.timer_label.setStyleSheet("font-size: 20px; padding: 0px 8px;")
        self.toolbar.addWidget(self.timer_label)

        # Ensure blinking and timer reflect the current is_saving state
        if self.is_saving:
            self.start_blinking()
        else:
            self.stop_blinking()

        self.toolbar.addSeparator()

        def add_action(text_icon, color, callback, tooltip, enabled, background_color):
            action = QAction(text_icon, self)
            action.triggered.connect(callback)
            action.setToolTip(tooltip)
            action.setEnabled(enabled)
            self.toolbar.addAction(action)
            button = self.toolbar.widgetForAction(action)
            if button:
                button.setStyleSheet(f"""
                    QToolButton {{
                        color: {color};
                        font-size: 24px;
                        border: none;
                        padding: 8px;
                        border-radius: 5px;
                        background-color: {background_color if enabled else '#546e7a'};
                    }}
                    QToolButton:hover {{ background-color: #4a90e2; }}
                    QToolButton:pressed {{ background-color: #357abd; }}
                    QToolButton:disabled {{ background-color: #546e7a; color: #b0bec5; }}
                """)

        add_action("▶", "#ffffff", self.start_saving_triggered, "Start Saving Data", not self.is_saving and self.current_project is not None, "#43a047")
        add_action("⏸", "#ffffff", self.stop_saving_triggered, "Stop Saving Data", self.is_saving, "#d8291d")
        self.toolbar.addSeparator()

        connect_enabled = not self.mqtt_connected
        disconnect_enabled = self.mqtt_connected
        connect_bg = "#43a047" if connect_enabled else "#546e7a"
        disconnect_bg = "#ef5350" if disconnect_enabled else "#546e7a"
        add_action("🔗", "#ffffff", self.connect_mqtt_triggered, "Connect to MQTT", connect_enabled, connect_bg)
        add_action("🔌", "#ffffff", self.disconnect_mqtt_triggered, "Disconnect from MQTT", disconnect_enabled, disconnect_bg)
        self.toolbar.addSeparator()

        self.files_combo = QComboBox()
        self.files_combo.setStyleSheet("""
            QComboBox {
                background-color: #ffffff;
                color: #212121;
                border: 1px solid #90caf9;
                border-radius: 4px;
                padding: 4px 8px;
                font-size: 14px;
                font-weight: 500;
                min-width: 150px;
                max-width: 200px;
            }
            QComboBox:hover { border: 1px solid #42a5f5; background-color: #f5faff; }
            QComboBox:disabled { background-color: #e0e0e0; color: #616161; border: 1px solid #b0bec5; }
        """)
        self.update_files_combo()
        self.toolbar.addWidget(self.files_combo)

        self.open_action = QAction("📂", self)
        self.open_action.setToolTip("Open Selected File")
        self.open_action.triggered.connect(self.open_selected_file)
        self.open_action.setEnabled(not self.mqtt_connected and self.files_combo.count() > 0 and self.files_combo.currentText() not in ["No files available", "No project selected", "Error loading files"])
        self.toolbar.addAction(self.open_action)
        open_button = self.toolbar.widgetForAction(self.open_action)
        if open_button:
            open_button.setStyleSheet(f"""
                QToolButton {{
                    color: #ffffff;
                    font-size: 24px;
                    border: none;
                    padding: 8px;
                    border-radius: 5px;
                    background-color: {'#43a047' if self.open_action.isEnabled() else '#546e7a'};
                }}
                QToolButton:hover {{ background-color: #4a90e2; }}
                QToolButton:pressed {{ background-color: #357abd; }}
                QToolButton:disabled {{ background-color: #546e7a; color: #b0bec5; }}
            """)

        self.toolbar.addSeparator()

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)

        layout_action = QAction("🖼️", self)
        layout_action.setToolTip("Select Layout")
        layout_action.triggered.connect(self.show_layout_menu)
        self.toolbar.addAction(layout_action)
        layout_button = self.toolbar.widgetForAction(layout_action)
        if layout_button:
            layout_button.setStyleSheet("""
                QToolButton {
                    color: #ffffff;
                    font-size: 24px;
                    border: none;
                    padding: 8px;
                    border-radius: 5px;
                }
                QToolButton:hover { background-color: #4a90e2; }
                QToolButton:pressed { background-color: #357abd; }
            """)
        self.toolbar.repaint()

    def open_selected_file(self):
        selected_file = self.files_combo.currentText()
        if selected_file and selected_file not in ["No files available", "No project selected", "Error loading files"]:
            self.open_file_triggered.emit(selected_file)
            logging.debug(f"SubToolBar: Open file triggered for {selected_file}")
        else:
            logging.debug(f"SubToolBar: Invalid file selection: {selected_file}")

    def refresh_filename(self):
        if not self.filename_edit:
            return
        try:
            next_filename = "data1"
            filename_counter = 1
            if self.current_project:
                model_name = self.parent.tree_view.get_selected_model()
                if model_name:
                    filenames = self.parent.db.get_distinct_filenames(self.current_project, model_name)
                    if filenames:
                        numbers = [int(re.match(r"data(\d+)", f).group(1)) for f in filenames if re.match(r"data(\d+)", f)]
                        filename_counter = max(numbers, default=0) + 1
                    next_filename = f"data{filename_counter}"
                else:
                    logging.debug("SubToolBar: No model selected for filename refresh")
            self.filename_edit.setText(next_filename)
            logging.debug(f"SubToolBar: Refreshed filename to {next_filename}")
        except Exception as e:
            logging.error(f"SubToolBar: Error refreshing filename: {str(e)}")
            self.filename_edit.setText("data1")

    def show_layout_menu(self):
        dialog = LayoutSelectionDialog(self, current_layout=self.selected_layout)
        parent_geom = self.parent.geometry()
        dialog.move(
            parent_geom.x() + (parent_geom.width() - dialog.width()) // 2,
            parent_geom.y() + (parent_geom.height() - dialog.height()) // 2
        )
        if dialog.exec_() == QDialog.Accepted:
            self.layout_selected.emit(dialog.selected_layout)