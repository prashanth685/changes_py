from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt

class MQTTStatus(QLabel):
    def __init__(self, parent):
        super().__init__("MQTT Status: Disconnected 🔴", parent)
        self.parent = parent
        self.initUI()
        self.parent.mqtt_status_changed.connect(self.update_mqtt_status_indicator)

    def initUI(self):
        self.setToolTip("MQTT Connection Status")
        self.setFixedHeight(40)
        self.setStyleSheet("""
            QLabel {
                background-color: black;
                color: #FFFFFF;
                font-size: 14px;
                font: bold;
                padding: 2px 8px;
                border-radius: 0px;
            }
        """)

    def update_mqtt_status_indicator(self, connected=None):
        status_text = "MQTT Status: Connected 🟢" if (connected if connected is not None else self.parent.mqtt_connected) else "MQTT Status: Disconnected 🔴"
        self.setText(status_text)