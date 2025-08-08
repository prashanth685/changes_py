from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

class HistoryPlotFeature:
    def __init__(self, parent, db, project_name, channel=None, model_name=None, console=None):
        self.parent = parent
        self.db = db
        self.project_name = project_name
        self.channel = channel
        self.model_name = model_name
        self.console = console  # Store the console instance
        self.widget = None
        self.initUI()

    def initUI(self):
        self.widget = QWidget()
        layout = QVBoxLayout()
        self.widget.setLayout(layout)
        label = QLabel(f"FFT View for Model: {self.model_name}, Channel: {self.channel}")
        layout.addWidget(label)

        if not self.model_name and self.console:
            self.console.append_to_console("No model selected in FFTViewFeature.")
        if not self.channel and self.console:
            self.console.append_to_console("No channel selected in FFTViewFeature.")

    def get_widget(self):
        return self.widget

    def on_data_received(self, tag_name, model_name, values):
        if self.model_name != model_name:
            return  # Ignore data for other models
        if self.console:
            self.console.append_to_console(f"FFT View ({self.model_name} - {self.channel}): Received data for {tag_name} - {values}")