from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class CenterLineFeature:
    def __init__(self, parent, db, project_name, channel=None, model_name=None, console=None):
        self.parent = parent
        self.db = db
        self.project_name = project_name
        self.channel = channel
        self.model_name = model_name
        self.console = console
        self.widget = None
        self.plot_widget = None
        self.plot_item = None
        self.primary_gap_values = []
        self.secondary_gap_values = []
        self.channel_names = []
        self.channel_index = None
        self.secondary_channel_index = None
        self.tag_name = None
        self.main_channels = 0
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plot)
        self.update_interval = 200  # ms
        self.initUI()
        self.cache_channel_data()
        # Add dummy data to test plotting
        self.add_dummy_data()
        logging.debug(f"Initialized CenterLineFeature with project_name: {project_name}, model_name: {model_name}, channel: {channel}")

    def initUI(self):
        self.widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        self.widget.setLayout(main_layout)

        # Label for primary channel
        self.primary_label = QLabel(f"Primary Channel: {self.channel or 'Unknown'}")
        self.primary_label.setStyleSheet("color: #ecf0f1; font-size: 16px; padding: 10px;")
        main_layout.addWidget(self.primary_label)

        # Secondary channel selection
        self.secondary_channel_combo = QComboBox()
        self.secondary_channel_combo.setStyleSheet("""
            QComboBox {
                background-color: #2c3e50;
                color: white;
                border: 1px solid #4a90e2;
                padding: 5px;
                border-radius: 4px;
                font-size: 14px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2c3e50;
                color: white;
                selection-background-color: #4a90e2;
            }
        """)
        self.secondary_channel_combo.currentIndexChanged.connect(self.secondary_channel_changed)
        main_layout.addWidget(self.secondary_channel_combo)

        # Plot setup using pyqtgraph
        pg.setConfigOptions(antialias=True)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("white")
        self.plot_widget.setTitle("Centerline Plot", color="black", size="12pt")
        self.plot_widget.setLabel('left', 'Secondary Channel Gap', color='black')
        self.plot_widget.setLabel('bottom', 'Primary Channel Gap', color='black')
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_item = self.plot_widget.plot(
            x=[], y=[],
            symbol='o',
            symbolSize=5,
            pen=None,
            symbolPen=pg.mkPen(color=(0, 128, 0), width=2),  # Green, matching C# ScottPlot.Color(0, 128, 0)
            symbolBrush=pg.mkBrush(color=(0, 128, 0)),
            name="Gap Data"
        )
        main_layout.addWidget(self.plot_widget)

        # Waiting message label
        self.waiting_message = QLabel("Waiting for data...")
        self.waiting_message.setStyleSheet("color: #ecf0f1; font-size: 14px; padding: 10px;")
        self.waiting_message.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.waiting_message)

        self.update_timer.start(self.update_interval)

    def add_dummy_data(self):
        """Add dummy data to test plotting functionality."""
        if not self.primary_gap_values and not self.secondary_gap_values:
            self.primary_gap_values = [1.0, 2.0, 3.0, 4.0, 5.0]
            self.secondary_gap_values = [2.0, 3.0, 4.0, 5.0, 6.0]
            logging.debug("Added dummy data for testing plot")
            if self.console:
                self.console.append_to_console("Added dummy data for testing plot")
            self.waiting_message.setVisible(False)
            self.update_plot()

    def cache_channel_data(self):
        try:
            if not self.db.is_connected():
                self.db.reconnect()
            project_data = self.db.get_project_data(self.project_name)
            if not project_data or "models" not in project_data:
                logging.error(f"Project {self.project_name} or models not found")
                if self.console:
                    self.console.append_to_console(f"Project {self.project_name} or models not found.")
                self.waiting_message.setText("Project or models not found.")
                return

            model = next((m for m in project_data["models"] if m.get("name") == self.model_name), None)
            if not model or not model.get("channels"):
                logging.error(f"Model {self.model_name} or channels not found")
                if self.console:
                    self.console.append_to_console(f"Model {self.model_name} or channels not found.")
                self.waiting_message.setText("Model or channels not found.")
                return

            self.channel_names = [ch.get("channelName") for ch in model.get("channels", [])]
            self.main_channels = len(self.channel_names)

            # Check for duplicate channel names
            channel_name_counts = {}
            for ch in self.channel_names:
                channel_name_counts[ch] = channel_name_counts.get(ch, 0) + 1
            for ch_name, count in channel_name_counts.items():
                if count > 1:
                    logging.warning(f"Channel name '{ch_name}' appears {count} times in the model")
                    if self.console:
                        self.console.append_to_console(f"Warning: Channel name '{ch_name}' appears {count} times in the model.")

            self.tag_name = model.get("tagName")
            if not self.tag_name:
                logging.error(f"TagName is empty for model {self.model_name}")
                if self.console:
                    self.console.append_to_console(f"TagName not found for model {self.model_name}.")
                self.waiting_message.setText("TagName not found for selected model.")
                return

            # Find primary channel index (0-based)
            self.channel_index = self.channel_names.index(self.channel) if self.channel in self.channel_names else -1
            if self.channel_index == -1:
                logging.error(f"Selected channel {self.channel} not found in model {self.model_name}. Available channels: {', '.join(self.channel_names)}")
                if self.console:
                    self.console.append_to_console(f"Selected channel {self.channel} not found.")
                self.waiting_message.setText("Selected channel not found.")
                return

            # Populate secondary channel combo box
            self.secondary_channel_combo.clear()
            for channel_name in self.channel_names:
                if channel_name != self.channel:
                    self.secondary_channel_combo.addItem(channel_name)

            # Set default secondary channel
            default_secondary_index = (self.channel_index + 1) % self.main_channels
            if default_secondary_index == self.channel_index:
                default_secondary_index = (default_secondary_index + 1) % self.main_channels
            if default_secondary_index < len(self.channel_names):
                self.secondary_channel_combo.setCurrentIndex(
                    self.channel_names.index(self.channel_names[default_secondary_index])
                )
                self.secondary_channel_index = default_secondary_index
            elif self.secondary_channel_combo.count() > 0:
                self.secondary_channel_combo.setCurrentIndex(0)
                self.secondary_channel_index = self.channel_names.index(self.secondary_channel_combo.currentText())

            self.primary_label.setText(f"Primary Channel: {self.channel_names[self.channel_index]}")
            logging.debug(f"Channel {self.channel} index (0-based): {self.channel_index}, TagName: {self.tag_name}, Secondary channel: {self.secondary_channel_combo.currentText()}")
            if self.console:
                self.console.append_to_console(f"Set primary channel to: {self.channel_names[self.channel_index]}")
                self.console.append_to_console(f"Secondary channel options: {', '.join(self.secondary_channel_combo.itemText(i) for i in range(self.secondary_channel_combo.count()))}")
                if self.secondary_channel_combo.currentText():
                    self.console.append_to_console(f"Selected default secondary channel: {self.secondary_channel_combo.currentText()}")

        except Exception as e:
            logging.error(f"Error caching channel data: {str(e)}")
            if self.console:
                self.console.append_to_console(f"Error caching channel data: {str(e)}")
            self.waiting_message.setText("Error initializing view.")

    def get_widget(self):
        return self.widget

    def on_data_received(self, tag_name, model_name, values, sample_rate):
        if self.model_name != model_name or self.tag_name != tag_name:
            logging.debug(f"Ignoring data for model {model_name}/tag {tag_name}, expected {self.model_name}/{self.tag_name}")
            return

        try:
            logging.debug(f"Received data: tag_name={tag_name}, model_name={model_name}, values_length={len(values)}")
            if len(values) < 200:  # Minimum 100 ushort values (200 bytes) for header
                logging.warning("Received invalid MQTT payload: too short.")
                if self.console:
                    self.console.append_to_console("Received invalid MQTT payload: too short.")
                return

            # Parse header (100 ushort values)
            header = np.frombuffer(values[:200], dtype=np.uint16)
            logging.debug(f"Raw bytes for header[10]: [{values[20]}, {values[21]}], header[11]: [{values[22]}, {values[23]}], "
                         f"header[12]: [{values[24]}, {values[25]}], header[13]: [{values[26]}, {values[27]}]")
            logging.debug(f"Header values [10-13]: {header[10:14].tolist()}")

            main_channels = header[2]
            if main_channels != self.main_channels:
                logging.warning(f"Mismatch in channel count: expected {self.main_channels}, got {main_channels}")
                if self.console:
                    self.console.append_to_console(f"Mismatch in channel count: expected {self.main_channels}, got {main_channels}")
                return

            # Get primary and secondary gap values
            primary_gap = float(header[10 + self.channel_index])
            secondary_gap = float(header[10 + self.secondary_channel_index])

            # Validate gap values
            if primary_gap > 1000 or secondary_gap > 1000:
                logging.warning(f"Ignoring unreasonable gap values - Primary ({self.channel_names[self.channel_index]}): {primary_gap}, "
                               f"Secondary ({self.channel_names[self.secondary_channel_index]}): {secondary_gap}")
                if self.console:
                    self.console.append_to_console(f"Ignoring unreasonable gap values - Primary: {primary_gap}, Secondary: {secondary_gap}")
                return

            # Append gap values
            self.primary_gap_values.append(primary_gap)
            self.secondary_gap_values.append(secondary_gap)
            self.waiting_message.setVisible(False)

            logging.debug(f"Received data for {tag_name}: Primary Gap ({self.channel_names[self.channel_index]}): {primary_gap}, "
                         f"Secondary Gap ({self.channel_names[self.secondary_channel_index]}): {secondary_gap}")
            if self.console:
                self.console.append_to_console(f"Received: Primary Gap ({self.channel_names[self.channel_index]}): {primary_gap}, "
                                             f"Secondary Gap ({self.channel_names[self.secondary_channel_index]}): {secondary_gap}")

            # Clear dummy data if present
            if len(self.primary_gap_values) > 5 and self.primary_gap_values[:5] == [1.0, 2.0, 3.0, 4.0, 5.0]:
                self.primary_gap_values = self.primary_gap_values[5:]
                self.secondary_gap_values = self.secondary_gap_values[5:]
                logging.debug("Cleared dummy data after receiving real data")
                if self.console:
                    self.console.append_to_console("Cleared dummy data after receiving real data")

            # Immediate plot update
            self.update_plot()

        except Exception as e:
            logging.error(f"Error in on_data_received: {str(e)}")
            if self.console:
                self.console.append_to_console(f"Error in Centerline View: {str(e)}")
            self.waiting_message.setText("Error processing data.")

    def update_plot(self):
        try:
            if not self.primary_gap_values or not self.secondary_gap_values:
                logging.debug("No data to plot")
                return

            # Ensure data is in NumPy arrays
            x_data = np.array(self.primary_gap_values, dtype=np.float64)
            y_data = np.array(self.secondary_gap_values, dtype=np.float64)

            # Update the scatter plot
            self.plot_item.setData(x=x_data, y=y_data)
            self.plot_widget.setTitle(f"{self.channel_names[self.channel_index]} vs {self.channel_names[self.secondary_channel_index]}")
            self.plot_widget.setLabel('bottom', f"{self.channel_names[self.channel_index]} Gap")
            self.plot_widget.setLabel('left', f"{self.channel_names[self.secondary_channel_index]} Gap")
            self.plot_widget.getPlotItem().autoRange()

            logging.debug(f"Updated plot with {len(self.primary_gap_values)} points: "
                         f"Primary Gaps = {self.primary_gap_values[:5]}, Secondary Gaps = {self.secondary_gap_values[:5]}")
            if self.console:
                self.console.append_to_console(f"Updated plot with {len(self.primary_gap_values)} points")

        except Exception as e:
            logging.error(f"Error updating Centerline plot: {str(e)}")
            if self.console:
                self.console.append_to_console(f"Error updating Centerline plot: {str(e)}")

    def secondary_channel_changed(self):
        try:
            selected_channel = self.secondary_channel_combo.currentText()
            if selected_channel:
                self.secondary_channel_index = self.channel_names.index(selected_channel)
                self.primary_gap_values.clear()
                self.secondary_gap_values.clear()
                self.plot_item.clear()
                self.waiting_message.setVisible(True)
                self.waiting_message.setText("Waiting for data...")
                # Add dummy data to ensure plot is visible
                self.add_dummy_data()
                logging.debug(f"Secondary channel changed to {selected_channel}. Plot data reset.")
                if self.console:
                    self.console.append_to_console(f"Secondary channel changed to {selected_channel}. Plot data reset.")
        except Exception as e:
            logging.error(f"Error changing secondary channel: {str(e)}")
            if self.console:
                self.console.append_to_console(f"Error changing secondary channel: {str(e)}")

    def cleanup(self):
        self.update_timer.stop()
        self.primary_gap_values.clear()
        self.secondary_gap_values.clear()
        self.plot_item.clear()
        logging.debug("Cleaned up CenterLineFeature resources")
        if self.console:
            self.console.append_to_console("Cleaned up Centerline View resources")