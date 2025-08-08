from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, QMessageBox
from PyQt5.QtCore import Qt
from project_structure import ProjectStructureWidget
import logging


class ExistingProjectWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.db = parent.db
        self.initUI()

    def initUI(self):
        self.setStyleSheet("background-color: #f5f7fa;")

        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(40, 40, 40, 40)
        self.setLayout(main_layout)

        card_widget = QWidget()
        card_widget.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 15px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                padding: 30px;
            }
        """)
        card_layout = QVBoxLayout()
        card_layout.setSpacing(20)
        card_widget.setLayout(card_layout)
        main_layout.addWidget(card_widget)

        title_label = QLabel("Open Existing Project")
        title_label.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #343a40;
            margin-bottom: 20px;
        """)
        card_layout.addWidget(title_label, alignment=Qt.AlignCenter)

        # Project selection dropdown
        project_layout = QHBoxLayout()
        project_layout.setSpacing(15)
        project_label = QLabel("Select Project:")
        project_label.setStyleSheet("font-size: 16px; color: #343a40; font-weight: bold;")
        project_layout.addWidget(project_label)

        self.project_combo = QComboBox()
        self.project_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #ced4da;
                border-radius: 8px;
                padding: 10px;
                font-size: 16px;
                min-width: 350px;
                background-color: #ffffff;
            }
            QComboBox:focus {
                border: 1px solid #007bff;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 14px;
                height: 14px;
            }
        """)
        self.load_projects()
        self.project_combo.currentTextChanged.connect(self.update_project_structure)
        project_layout.addWidget(self.project_combo)
        project_layout.addStretch()
        card_layout.addLayout(project_layout)

        # Project structure view
        self.structure_widget = ProjectStructureWidget(self)
        card_layout.addWidget(self.structure_widget)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)

        open_button = QPushButton("Open Project")
        open_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border-radius: 8px;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
        """)
        open_button.clicked.connect(self.open_project)
        button_layout.addWidget(open_button)

        back_button = QPushButton("Back")
        back_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border-radius: 8px;
                padding: 12px;
                font-size: 16px;
                font-weight: bold;
                min-width: 150px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:pressed {
                background-color: #4b5359;
            }
        """)
        back_button.clicked.connect(self.back_to_select)
        button_layout.addWidget(back_button)

        card_layout.addLayout(button_layout)

    def load_projects(self):
        try:
            projects = self.db.load_projects()
            self.project_combo.clear()
            if not projects:
                self.project_combo.addItem("No projects available")
                self.project_combo.setEnabled(False)
                self.structure_widget.update_structure("", [])
            else:
                self.project_combo.addItems(projects)
                self.project_combo.setEnabled(True)
                self.update_project_structure()
        except Exception as e:
            logging.error(f"Error loading projects: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to load projects: {str(e)}")

    def update_project_structure(self):
        project_name = self.project_combo.currentText()
        if project_name and project_name != "No projects available":
            try:
                project_data = self.db.get_project_data(project_name)
                self.structure_widget.update_structure(project_name, project_data.get("models", []))
            except Exception as e:
                logging.error(f"Error updating project structure for {project_name}: {str(e)}")
                QMessageBox.warning(self, "Error", f"Failed to load project structure: {str(e)}")

    def open_project(self):
        project_name = self.project_combo.currentText()
        if not project_name or project_name == "No projects available":
            QMessageBox.warning(self, "Error", "Please select a project to open!")
            return
        if project_name in self.parent.open_dashboards:
            self.parent.open_dashboards[project_name].raise_()
            self.parent.open_dashboards[project_name].activateWindow()
            return
        self.parent.load_project(project_name)

    def back_to_select(self):
        self.parent.display_select_project()