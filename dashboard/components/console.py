from PyQt5.QtWidgets import QWidget, QHBoxLayout, QTextEdit, QPushButton, QSizePolicy
import logging

class Console(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.initUI()
        self.minimize_console()

    def initUI(self):
        self.button_container = QWidget()
        self.button_container.setFixedHeight(40)
        self.button_container.setStyleSheet("background-color: #0c0c0f;")
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(5, 0, 5, 0)
        button_layout.setSpacing(5)
        self.button_container.setLayout(button_layout)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        button_layout.addWidget(spacer)
        self.minimize_button = QPushButton("-")
        self.minimize_button.setToolTip("Minimize Console")
        self.minimize_button.clicked.connect(self.minimize_console)
        self.minimize_button.setStyleSheet("""
            QPushButton { 
                color: white; 
                font-size: 16px; 
                padding: 2px 8px; 
                border-radius: 4px; 
                background-color: #34495e; 
                border: none;
            }
            QPushButton:hover { background-color: #4a90e2; }
            QPushButton:pressed { background-color: #357abd; }
        """)
        button_layout.addWidget(self.minimize_button)
        self.maximize_button = QPushButton("🗖")
        self.maximize_button.setToolTip("Maximize Console")
        self.maximize_button.clicked.connect(self.maximize_console)
        self.maximize_button.setStyleSheet("""
            QPushButton { 
                color: white; 
                font-size: 16px; 
                padding: 2px 8px; 
                border-radius: 4px; 
                background-color: #34495e; 
                border: none;
            }
            QPushButton:hover { background-color: #4a90e2; }
            QPushButton:pressed { background-color: #357abd; }
        """)
        button_layout.addWidget(self.maximize_button)
        self.console_message_area = QTextEdit()
        self.console_message_area.setReadOnly(True)
        self.console_message_area.setFixedHeight(200)
        self.console_message_area.setStyleSheet("""
            QTextEdit { 
                background-color: #0a0a0a; 
                color: #e0e0e0; 
                border: none; 
                font-family: Consolas, monospace; 
                font-size: 14px; 
                padding: 10px; 
            }
        """)
        self.maximize_button.show()
        self.minimize_button.hide()

    def append_to_console(self, text):
        if "MQTT" in text or "mqtt" in text or "layout" in text.lower():
            logging.info(text)
            if self.console_message_area.isVisible():
                self.console_message_area.append(text)
                self.console_message_area.ensureCursorVisible()

    def clear_console(self):
        try:
            self.console_message_area.clear()
            logging.info("Console cleared")
        except Exception as e:
            logging.error(f"Error clearing console: {str(e)}")

    def minimize_console(self):
        try:
            self.console_message_area.setFixedHeight(0)
            self.console_message_area.hide()
            layout = self.parent.console_layout
            layout.removeWidget(self.button_container)
            layout.removeWidget(self.console_message_area)
            layout.removeWidget(self.parent.mqtt_status)
            layout.addWidget(self.button_container)
            layout.addWidget(self.console_message_area)
            layout.addWidget(self.parent.mqtt_status)
            self.minimize_button.hide()
            self.maximize_button.show()
            self.parent.console_container.setFixedHeight(80)
            logging.info("Console minimized")
        except Exception as e:
            logging.error(f"Error minimizing console: {str(e)}")

    def maximize_console(self):
        try:
            self.console_message_area.setFixedHeight(200)
            self.console_message_area.show()
            layout = self.parent.console_layout
            layout.removeWidget(self.button_container)
            layout.removeWidget(self.console_message_area)
            layout.removeWidget(self.parent.mqtt_status)
            layout.addWidget(self.button_container)
            layout.addWidget(self.console_message_area)
            layout.addWidget(self.parent.mqtt_status)
            self.minimize_button.show()
            self.maximize_button.hide()
            self.parent.console_container.setFixedHeight(200)
            logging.info("Console maximized")
        except Exception as e:
            logging.error(f"Error maximizing console: {str(e)}")