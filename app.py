import sys
from PyQt5.QtWidgets import QApplication
from auth import AuthWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    auth_window = AuthWindow()
    auth_window.show()
    sys.exit(app.exec_())