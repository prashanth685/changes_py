from PyQt5.QtWidgets import QToolBar, QAction
from PyQt5.QtCore import Qt, pyqtSignal
import logging

class FileBar(QToolBar):
    # Signals to communicate with DashboardWindow
    home_triggered = pyqtSignal()
    open_triggered = pyqtSignal()
    edit_triggered = pyqtSignal()
    new_triggered = pyqtSignal()
    save_triggered = pyqtSignal()
    settings_triggered = pyqtSignal()
    refresh_triggered = pyqtSignal()
    exit_triggered = pyqtSignal()

    def __init__(self, parent):
        super().__init__("File", parent)
        self.parent = parent
        self.current_project = None
        self.mqtt_connected = False
        self.initUI()
        # Connect to parent's MQTT status signal
        self.parent.mqtt_status_changed.connect(self.update_mqtt_status)

    def initUI(self):
        self.setStyleSheet("""
            QToolBar {
                background: #2D2F33;
                border: none;
                padding: 0;
                spacing: 5px;
            }
            QToolBar QToolButton {
                font-size: 18px;
                font-weight: bold;
                color: #fff;
                padding: 8px 12px;
                border-radius: 4px;
                background-color: transparent;
            }
            QToolBar QToolButton:hover {
                background-color: #4a90e2;
                color: white;
            }
            QToolBar QToolButton:disabled {
                color: #666;
            }
        """)
        self.setFixedHeight(40)
        self.setMovable(False)
        self.setFloatable(False)

        # Define actions with their signals
        self.actions = {
            "Home": QAction("Home", self),
            "Open": QAction("Open", self),
            "Edit": QAction("Edit", self),
            "New": QAction("New", self),
            "Save": QAction("Save", self),
            "Settings": QAction("Settings", self),
            "Refresh": QAction("Refresh", self),
            "Exit": QAction("Exit", self)
        }
        action_configs = [
            ("Home", "Go to Dashboard Home", self.home_triggered),
            ("Open", "Open an Existing Project", self.open_triggered),
            ("Edit", "Edit an Existing Project", self.edit_triggered),
            ("New", "Create a New Project", self.new_triggered),
            ("Save", "Save Current Project Data", self.save_triggered),
            ("Settings", "Open Application Settings", self.settings_triggered),
            ("Refresh", "Refresh Current View", self.refresh_triggered),
            ("Exit", "Exit Application", self.exit_triggered)
        ]
        for action_name, tooltip, signal in action_configs:
            action = self.actions[action_name]
            action.setToolTip(tooltip)
            action.triggered.connect(signal.emit)
            self.addAction(action)

        # Initial state update
        self.update_state()

    def update_state(self, project_name=None, mqtt_connected=None):
        """Update action states and toolbar appearance based on application state."""
        try:
            if project_name is not None:
                self.current_project = project_name
            if mqtt_connected is not None:
                self.mqtt_connected = mqtt_connected

            # Enable/disable actions based on state
            self.actions["Home"].setEnabled(True)
            self.actions["Open"].setEnabled(True)
            self.actions["New"].setEnabled(True)
            self.actions["Settings"].setEnabled(True)
            self.actions["Exit"].setEnabled(True)
            self.actions["Save"].setEnabled(self.current_project is not None)
            self.actions["Edit"].setEnabled(self.current_project is not None)
            self.actions["Refresh"].setEnabled(self.current_project is not None)

            # Update stylesheet based on project state
            background = "#2D2F33" if self.current_project else "#f5f5f5"
            text_color = "#fff" if self.current_project else "#333"
            self.setStyleSheet(f"""
                QToolBar {{
                    background: {background};
                    border: none;
                    padding: 0;
                    spacing: 5px;
                }}
                QToolBar QToolButton {{
                    font-size: 18px;
                    font-weight: bold;
                    color: {text_color};
                    padding: 8px 12px;
                    border-radius: 4px;
                    background-color: transparent;
                }}
                QToolBar QToolButton:hover {{
                    background-color: #4a90e2;
                    color: white;
                }}
                QToolBar QToolButton:disabled {{
                    color: #666;
                }}
            """)
            logging.debug(f"FileBar updated: project={self.current_project}, mqtt_connected={self.mqtt_connected}")
        except Exception as e:
            logging.error(f"Error updating FileBar state: {str(e)}")

    def update_mqtt_status(self, connected):
        """Update MQTT connection status."""
        self.update_state(mqtt_connected=connected)