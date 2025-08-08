from dashboard.dashboard_window import DashboardWindow

class ProjectSelectionWindow:

    def __init__(self, db, email, auth_window=None):
        self.db = db
        self.email = email
        self.auth_window = auth_window
        self.dashboard_window = None
        self.open_dashboard()

    def open_dashboard(self):
        self.dashboard_window = DashboardWindow(self.db, self.email, self.auth_window)
        self.dashboard_window.show()
    
        