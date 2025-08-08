from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
import pyqtgraph as pg
import numpy as np
from PyQt5.QtCore import QTimer

class PolarPlotFeature:
    def __init__(self, parent=None, db=None, project_name='', channel=0, model_name=None, console=None):
        self.parent = parent
        self.db = db
        self.project_name = project_name
        try:
            self.channel = int(channel)  # Convert channel to integer
        except (ValueError, TypeError):
            self.channel = 0  # Default to channel 0 if conversion fails
            if console:
                console.append_to_console(f"Invalid channel input '{channel}', defaulting to 0")
        self.model_name = model_name
        self.console = console
        self.widget = None
        self.plot_widget = None
        self.curve = None
        self.grid_curves = []
        self.initUI()

    def initUI(self):
        self.widget = QWidget()
        layout = QVBoxLayout()
        self.widget.setLayout(layout)

        # Display label
        label = QLabel(f"Polar Plot View for Model: {self.model_name}, Channel: {self.channel}")
        layout.addWidget(label)

        # Create pyqtgraph plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('k')  # White background
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setAspectLocked(True)  # Lock aspect ratio for circular appearance
        self.plot_widget.setRange(xRange=[-1.5, 1.5], yRange=[-1.5, 1.5])  # Fixed range
        self.plot_widget.setMinimumSize(800, 800)  # Increase plot size
        self.plot_widget.setMouseEnabled(x=False, y=False)  # Disable zooming/panning

        # Add circular grid
        for r in [0.5, 1.0, 1.5]:
            theta = np.linspace(0, 2 * np.pi, 100)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            circle = self.plot_widget.plot(x, y, pen=pg.mkPen('k', width=0.5, style=pg.QtCore.Qt.DashLine))
            self.grid_curves.append(circle)

        # Add radial lines
        for theta in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            x = [0, 1.5 * np.cos(theta)]
            y = [0, 1.5 * np.sin(theta)]
            line = self.plot_widget.plot(x, y, pen=pg.mkPen('k', width=0.5, style=pg.QtCore.Qt.DashLine))
            self.grid_curves.append(line)

        # Create data curve
        self.curve = self.plot_widget.plot(pen=pg.mkPen('b', width=2), symbol='o', symbolSize=5, symbolPen='b', symbolBrush='b')
        layout.addWidget(self.plot_widget)

        if not self.model_name and self.console:
            self.console.append_to_console("No model selected in PolarPlotFeature.")
        if self.channel is None and self.console:
            self.console.append_to_console("No channel selected in PolarPlotFeature.")

    def get_widget(self):
        return self.widget

    def on_data_received(self, tag_name, model_name, values, sample_rate):
        if self.model_name != model_name:
            if self.console:
                self.console.append_to_console(f"Ignoring data for model {model_name}, expected {self.model_name}")
            return

        if self.console:
            self.console.append_to_console(
                f"Polar Plot View ({self.model_name} - Channel {self.channel}): Received data for {tag_name}, {len(values)} channels"
            )

        # Validate and extract data
        if not isinstance(values, list) or self.channel >= len(values):
            if self.console:
                self.console.append_to_console(f"Invalid channel {self.channel} or values for {tag_name}")
            return

        data = np.asarray(values[self.channel], dtype=float)
        if data.size == 0:
            if self.console:
                self.console.append_to_console(f"No data for channel {self.channel} in {tag_name}")
            return

        # Prepare polar plot data
        theta = np.linspace(0, 2 * np.pi, len(data), endpoint=False)
        r = data / (np.max(np.abs(data)) + 1e-12)  # Normalize to avoid division by zero

        # Convert to Cartesian coordinates
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        # Clear previous data and update plot
        self.curve.clear()
        self.curve = self.plot_widget.plot(x, y, pen=pg.mkPen('b', width=2), symbol='o', symbolSize=5, symbolPen='b', symbolBrush='b')
        self.plot_widget.setRange(xRange=[-1.5, 1.5], yRange=[-1.5, 1.5])  # Reset range to prevent zoom issues
        self.plot_widget.setTitle(f"Polar Plot - {tag_name} (Channel {self.channel})")
        if self.console:
            self.console.append_to_console(f"Plotted {len(data)} points for channel {self.channel}")