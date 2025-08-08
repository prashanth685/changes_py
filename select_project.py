from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont

class SelectProjectWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.initUI()

    def initUI(self):
        # Main background
        self.setStyleSheet("background-color: #f8f9fa;")  # Soft gray like React dashboards

        # Main layout
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        # Card container
        card_widget = QWidget()
        card_widget.setFixedSize(500, 500)
        card_widget.setStyleSheet("""
            QWidget {
                background-color: #ffffff;
                border-radius: 16px;
            }
        """)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setXOffset(0)
        shadow.setYOffset(8)
        shadow.setColor(QColor(0, 0, 0, 30))  # Light shadow
        card_widget.setGraphicsEffect(shadow)

        # Card layout
        card_layout = QVBoxLayout()
        card_layout.setContentsMargins(40, 40, 40, 40)
        card_layout.setSpacing(30)
        card_widget.setLayout(card_layout)
        layout.addWidget(card_widget, alignment=Qt.AlignCenter)

        # Title
        title_label = QLabel("Select an option")
        title_label.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title_label.setStyleSheet("color: #212529;")
        card_layout.addWidget(title_label, alignment=Qt.AlignCenter)

        # Button style
        button_style = """
            QPushButton {
                background-color: #0d6efd;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 14px 20px;
                font-size: 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #0b5ed7;
            }
            QPushButton:pressed {
                background-color: #0a58ca;
            }
        """

        # Create Project Button
        create_button = QPushButton("➕  Create New Project")
        create_button.setStyleSheet(button_style)
        create_button.clicked.connect(self.parent.create_project)
        card_layout.addWidget(create_button, alignment=Qt.AlignCenter)

        # Open Existing Project Button (green)
        open_button = QPushButton("📂  Open Existing Project")
        open_button.setStyleSheet(button_style.replace("#0d6efd", "#198754").replace("#0b5ed7", "#157347").replace("#0a58ca", "#146c43"))
        open_button.clicked.connect(self.parent.display_project_structure)
        card_layout.addWidget(open_button, alignment=Qt.AlignCenter)
