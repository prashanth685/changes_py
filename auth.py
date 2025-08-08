import sys
from PyQt5.QtGui import QPixmap, QColor
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QLineEdit,
                             QPushButton, QMessageBox, QFormLayout, QApplication,
                             QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import bcrypt
import os
from database import Database
from project_selection import ProjectSelectionWindow

class AuthWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.client = None
        self.db = None
        self.user_collection = None
        self.is_login_mode = True
        self.initDB()
        self.initUI()
        self.setWindowState(Qt.WindowMaximized)

    def initDB(self):
        try:
            self.client = MongoClient("mongodb://localhost:27017/")
            self.db = self.client["changed_db"]
            self.user_collection = self.db["users"]
            print("Connected to MongoDB successfully!")
        except ConnectionFailure as e:
            print(f"Could not connect to MongoDB: {e}")
            QMessageBox.critical(self, "Database Error", "Failed to connect to the database.")
            sys.exit(1)

    def initUI(self):
        self.setWindowTitle('Sarayu Infotech Solutions Pvt. Ltd.')
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AlignCenter)
        main_layout.setSpacing(10)
        self.setLayout(main_layout)

        # Logo
        logo_label = QLabel(self)
        logo_path = "logo.png" if os.path.exists("logo.png") else "icons/placeholder.png"
        pixmap = QPixmap(logo_path)
        if pixmap.isNull():
            print(f"Warning: Could not load logo at {logo_path}")
            pixmap = QPixmap("icons/placeholder.png")
        logo_label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio))
        logo_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(logo_label)

        # Company name
        company_label = QLabel('Sarayu Infotech Solutions Pvt. Ltd.')
        company_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #007bff;")
        main_layout.addWidget(company_label, alignment=Qt.AlignCenter)

        # Welcome text
        welcome_label = QLabel('Welcome')
        welcome_label.setStyleSheet("font-size: 20px; color: #007bff;")
        main_layout.addWidget(welcome_label, alignment=Qt.AlignCenter)

        # Form container
        self.form_container = QWidget()
        self.form_layout = QVBoxLayout()
        self.form_layout.setAlignment(Qt.AlignCenter)
        self.form_layout.setSpacing(15)
        self.form_container.setLayout(self.form_layout)

        # Heading
        self.heading = QLabel("Sign In")
        self.heading.setStyleSheet("font-size: 30px; font-weight: bold; color: rgb(16, 137, 211);")
        self.heading.setAlignment(Qt.AlignCenter)
        self.form_layout.addWidget(self.heading)

        # Form inputs
        self.form_fields = QFormLayout()
        self.form_fields.setAlignment(Qt.AlignCenter)
        self.form_fields.setSpacing(10)

        # Email input
        email_label = QLabel('Email')
        email_label.setStyleSheet("font-size: 18px; color: #333; font-weight: bold;")
        self.email_input = self.create_input_field('Enter your email')
        self.email_input.setText('raj@gmail.com')
        self.email_input.setStyleSheet('font-size:16px;font:bold')
        self.form_fields.addRow(email_label, self.email_input)

        # Password input
        password_label = QLabel('Password')
        password_label.setStyleSheet("font-size: 18px; color: #333; font-weight: bold;")
        self.password_input = self.create_input_field('Enter your password')
        self.password_input.setText('12345678')
        self.password_input.setStyleSheet('font-size:16px;font:bold')
        self.password_input.setEchoMode(QLineEdit.Password)
        self.form_fields.addRow(password_label, self.password_input)

        # Confirm password input (only for signup)
        self.confirm_password_label = QLabel('Confirm Password')
        self.confirm_password_label.setStyleSheet("font-size: 18px; color: #333; font-weight: bold;")
        self.confirm_password_input = self.create_input_field('Confirm your password')
        self.confirm_password_input.setStyleSheet('font-size:16px;font:bold')
        self.confirm_password_input.setEchoMode(QLineEdit.Password)
        self.form_fields.addRow(self.confirm_password_label, self.confirm_password_input)
        self.confirm_password_label.hide()
        self.confirm_password_input.hide()

        self.form_layout.addLayout(self.form_fields)

        # Forgot password link (only for login)
        self.forgot_link = QLabel('<a href="#" style="color: #0099ff; text-decoration: none; font-size: 14px;">Forgot Password?</a>')
        self.forgot_link.setOpenExternalLinks(True)
        self.form_fields.addRow("", self.forgot_link)

        # Action button
        self.action_button = QPushButton('Sign In')
        self.action_button.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgb(16, 137, 211), stop:1 rgb(18, 177, 209));
                color: white;
                border-radius: 20px;
                padding: 15px;
                font-weight: bold;
                border: none;
                width: 290px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgb(14, 123, 190), stop:1 rgb(16, 159, 188));
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgb(12, 110, 170), stop:1 rgb(14, 141, 168));
            }
        """)
        shadow_button = QGraphicsDropShadowEffect()
        shadow_button.setOffset(0, 20)
        shadow_button.setBlurRadius(10)
        shadow_button.setColor(QColor(133, 189, 215, 223))
        self.action_button.setGraphicsEffect(shadow_button)
        self.action_button.clicked.connect(self.handle_action)
        self.form_layout.addWidget(self.action_button, alignment=Qt.AlignCenter)

        # Toggle link
        self.toggle_link = QLabel('<a href="#" style="color: #0099ff; text-decoration: none; font-size: 14px;">Don\'t have an account? Sign Up</a>')
        self.toggle_link.setOpenExternalLinks(False)
        self.toggle_link.linkActivated.connect(self.toggle_mode)
        self.form_layout.addWidget(self.toggle_link, alignment=Qt.AlignCenter)

        # Style form container
        self.form_container.setStyleSheet("""
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #FFFFFF, stop:1 #F4F7FB);
            border-radius: 40px;
            border: 5px solid white;
            padding: 25px 35px;
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setOffset(0, 10)
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 0, 0, 60))
        self.form_container.setGraphicsEffect(shadow)

        main_layout.addWidget(self.form_container, alignment=Qt.AlignCenter)
        self.setStyleSheet("background-color: white;")

    def create_input_field(self, placeholder):
        input_field = QLineEdit()
        input_field.setPlaceholderText(placeholder)
        input_field.setStyleSheet("""
            QLineEdit {
                background: white;
                border: none;
                padding: 12px 20px;
                border-radius: 20px;
                border: 2px solid transparent;
                color: rgb(170, 170, 170);
                font-size: 14px;
                width: 290px;
            }
            QLineEdit:focus {
                border: 2px solid #12B1D1;
            }
        """)
        shadow = QGraphicsDropShadowEffect()
        shadow.setOffset(0, 10)
        shadow.setBlurRadius(10)
        shadow.setColor(QColor("#cff0ff"))
        input_field.setGraphicsEffect(shadow)
        return input_field

    def toggle_mode(self):
        self.is_login_mode = not self.is_login_mode
        if self.is_login_mode:
            self.heading.setText("Sign In")
            self.action_button.setText("Sign In")
            self.action_button.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 rgb(16, 137, 211), stop:1 rgb(18, 177, 209));
                    color: white;
                    border-radius: 20px;
                    padding: 15px;
                    font-weight: bold;
                    border: none;
                    width: 290px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 rgb(14, 123, 190), stop:1 rgb(16, 159, 188));
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 rgb(12, 110, 170), stop:1 rgb(14, 141, 168));
                }
            """)
            self.toggle_link.setText('<a href="#" style="color: #0099ff; text-decoration: none; font-size: 14px;">Don\'t have an account? Sign Up</a>')
            self.confirm_password_label.hide()
            self.confirm_password_input.hide()
            self.forgot_link.show()
        else:
            self.heading.setText("Sign Up")
            self.action_button.setText("Sign Up")
            self.action_button.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #28a745, stop:1 #218838);
                    color: white;
                    border-radius: 20px;
                    padding: 15px;
                    font-weight: bold;
                    border: none;
                    width: 290px;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #218838, stop:1 #1e7e34);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                        stop:0 #1e7e34, stop:1 #1a6b2d);
                }
            """)
            self.toggle_link.setText('<a href="#" style="color: #0099ff; text-decoration: none; font-size: 14px;">Already have an account? Sign In</a>')
            self.confirm_password_label.show()
            self.confirm_password_input.show()
            self.forgot_link.hide()
        self.email_input.clear()
        self.password_input.clear()
        self.confirm_password_input.clear()

    def handle_action(self):
        if self.is_login_mode:
            self.login()
        else:
            self.signup()

    def login(self):
        email = self.email_input.text().strip()
        password = self.password_input.text().strip()

        if not email or not password:
            QMessageBox.warning(self, "Input Error", "Please enter both email and password.")
            return

        user = self.user_collection.find_one({"email": email})
        if user and bcrypt.checkpw(password.encode('utf-8'), user["password"]):
            try:
                db = Database(connection_string="mongodb://localhost:27017/", email=email)
                ProjectSelectionWindow(db, email, self)

                self.hide()
            except Exception as e:
                print(f"Error opening Project Selection: {e}")
                QMessageBox.critical(self, "Error", f"Failed to open project selection: {e}")
        else:
            QMessageBox.warning(self, "Login Failed", "Incorrect email or password.")

    def signup(self):
        email = self.email_input.text().strip()
        password = self.password_input.text().strip()
        confirm_password = self.confirm_password_input.text().strip()

        if not email or not password or not confirm_password:
            QMessageBox.warning(self, "Input Error", "Please fill in all fields.")
            return

        if password != confirm_password:
            QMessageBox.warning(self, "Input Error", "Passwords do not match.")
            return

        if self.user_collection.find_one({"email": email}):
            QMessageBox.warning(self, "Signup Failed", "User with this email already exists. Please log in.")
            return

        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        user_data = {"email": email, "password": hashed_password}
        try:
            self.user_collection.insert_one(user_data)
            QMessageBox.information(self, "Success", "Signup successful! Proceeding to project selection.")
            db = Database(connection_string="mongodb://localhost:27017/", email=email)
            ProjectSelectionWindow(db, email, self)

            self.hide()
        except Exception as e:
            print(f"Error inserting user: {e}")
            QMessageBox.critical(self, "Database Error", "Failed to sign up.")

    def closeEvent(self, event):
        if self.client:
            self.client.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AuthWindow()
    window.show()
    sys.exit(app.exec_())