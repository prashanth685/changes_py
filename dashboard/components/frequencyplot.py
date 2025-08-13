from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QApplication, QMessageBox, QSpinBox
from PyQt5.QtCore import Qt, pyqtSignal, QRectF
import pyqtgraph as pg
import numpy as np
import datetime
import logging
from database import Database

class DraggableLine(pg.InfiniteLine):
    """A draggable InfiniteLine that emits its position when moved."""
    positionChanged = pyqtSignal(float)

    def __init__(self, pos, *args, **kwargs):
        super().__init__(pos=pos, *args, **kwargs)
        self.setMovable(True)
        self.addMarker('<|>', position=0.5) # Add a marker in the middle
        self.sigDragged.connect(self._on_dragged)
        self.sigPositionChangeFinished.connect(self._on_drag_finished)

    def _on_dragged(self):
        """Emit signal continuously while dragging."""
        # Round to nearest integer for frame index
        pos = int(round(self.value()))
        self.positionChanged.emit(float(pos)) # Ensure it's a float for consistency

    def _on_drag_finished(self):
        """Ensure the final position is an integer and update the line."""
        pos = int(round(self.value()))
        # Block signals temporarily to avoid recursive updates
        self.blockSignals(True)
        self.setValue(float(pos))
        self.blockSignals(False)
        self.positionChanged.emit(float(pos))

class FrequencyPlot(QWidget):
    time_range_selected = pyqtSignal(dict)

    def __init__(self, parent=None, project_name=None, model_name=None, filename=None, start_time=None, end_time=None, email="user@example.com"):
        super().__init__(parent)
        self.setMinimumSize(900, 700)
        self.project_name = project_name
        self.model_name = model_name
        self.filename = filename
        self.email = email
        self.db = Database(connection_string="mongodb://localhost:27017/", email=email)
        
        # Data storage
        self.current_records = []
        self.frame_indices = [] # Will store frame indices (int)
        self.frequencies = []   # Will store message frequencies (float)
        
        # Selection lines
        self.start_line = None
        self.end_line = None
        self.selection_region = None

        # Crosshair lines
        self.v_line = None
        self.h_line = None
        self.crosshair_enabled = True # Flag to enable/disable crosshairs

        # Selected record info
        self.selected_frame_index = None

        self.initUI()
        self.initialize_data()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.layout.setSpacing(10)

        self.title_label = QLabel(f"Frequency Analysis for {self.filename}")
        self.title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #333;")
        self.layout.addWidget(self.title_label)

        # Create the pyqtgraph PlotWidget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w') # White background
        self.plot_widget.setLabel('left', 'Frequency')
        self.plot_widget.setLabel('bottom', 'Frame Index')
        self.plot_widget.setTitle('Frequency vs Frame Index')
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        # Disable auto-ranging initially to prevent errors with bad data
        self.plot_widget.disableAutoRange()
        self.layout.addWidget(self.plot_widget)

        # --- Controls Layout ---
        controls_layout = QHBoxLayout()
        
        # Start Index Input
        self.start_index_label = QLabel("Start Index:")
        self.start_index_input = QSpinBox()
        self.start_index_input.setMinimum(0)
        self.start_index_input.setMaximum(999999999) # Large number
        self.start_index_input.setSingleStep(1)
        self.start_index_input.valueChanged.connect(self.on_start_index_changed)
        controls_layout.addWidget(self.start_index_label)
        controls_layout.addWidget(self.start_index_input)

        # End Index Input
        self.end_index_label = QLabel("End Index:")
        self.end_index_input = QSpinBox()
        self.end_index_input.setMinimum(0)
        self.end_index_input.setMaximum(999999999)
        self.end_index_input.setSingleStep(1)
        self.end_index_input.valueChanged.connect(self.on_end_index_changed)
        controls_layout.addWidget(self.end_index_label)
        controls_layout.addWidget(self.end_index_input)

        # Spacer
        controls_layout.addStretch()

        # Selected Index Display
        self.selected_index_label = QLabel("Selected Index:")
        self.selected_index_display = QLabel("N/A")
        self.selected_index_display.setStyleSheet("font-weight: bold; color: #4a90e2;")
        controls_layout.addWidget(self.selected_index_label)
        controls_layout.addWidget(self.selected_index_display)

        # Select Button
        self.select_button = QPushButton("Select")
        self.select_button.setStyleSheet("""
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #357abd; }
            QPushButton:pressed { background-color: #2c5d9b; }
        """)
        self.select_button.clicked.connect(self.select_button_click)
        controls_layout.addWidget(self.select_button)

        self.layout.addLayout(controls_layout)
        self.setLayout(self.layout)

    def initialize_data(self):
        try:
            if not all([self.project_name, self.model_name, self.filename]):
                logging.error("Missing project, model, or filename for FrequencyPlot")
                self.show_message_box("Missing project, model, or filename.", "Error", "error")
                return

            messages = self.db.get_history_messages(
                self.project_name,
                self.model_name,
                filename=self.filename
            )
            if not messages:
                logging.warning(f"No history messages found for {self.filename}")
                self.show_message_box(f"No data found for file {self.filename}", "No Data", "warning")
                return

            # --- Data Cleaning and Preparation ---
            raw_data = []
            for record in messages:
                try:
                    frame_idx = record.get("frameIndex")
                    freq = record.get("messageFrequency")
                    # Ensure frameIndex is an integer and frequency is a float
                    if frame_idx is not None and freq is not None:
                        frame_idx = int(frame_idx)
                        freq = float(freq)
                        raw_data.append((frame_idx, freq))
                    else:
                        logging.debug(f"Skipping record with missing frameIndex or messageFrequency: {record}")
                except (ValueError, TypeError) as e:
                    logging.debug(f"Skipping record with invalid data types: {record}, Error: {e}")

            if not raw_data:
                logging.error("No valid data found after cleaning.")
                self.show_message_box("No valid data found in the file.", "No Valid Data", "warning")
                return

            # Sort by frame index
            raw_data.sort(key=lambda x: x[0])
            self.frame_indices, self.frequencies = zip(*raw_data)
            self.frame_indices = list(self.frame_indices)
            self.frequencies = list(self.frequencies)

            logging.debug(f"Initialized  {len(self.frame_indices)} points")
            self.plot_full_data()
        except Exception as e:
            logging.error(f"Error initializing  {e}", exc_info=True)
            self.show_message_box(f"Error loading  {str(e)}", "Error", "error")

    def plot_full_data(self):
        """Plot all data and initialize draggable lines."""
        if not self.frame_indices or not self.frequencies:
            logging.warning("No data to plot in plot_full_data.")
            return

        self.plot_widget.clear()

        # Plot the data
        pen = pg.mkPen(color=(0, 0, 255), width=2) # Blue line
        # Use scatter plot for points and line for connection
        self.plot_widget.plot(self.frame_indices, self.frequencies, pen=pen, symbol='o', symbolSize=5, symbolBrush='b', name='Frequency')

        # Get min and max frame indices
        min_frame = min(self.frame_indices)
        max_frame = max(self.frame_indices)

        # Create and add draggable lines
        self.start_line = DraggableLine(pos=float(min_frame), angle=90, movable=True, pen=pg.mkPen('g', width=3), label='Start', labelOpts={'position': 0.95})
        self.start_line.positionChanged.connect(self.on_start_line_moved)
        self.plot_widget.addItem(self.start_line)

        self.end_line = DraggableLine(pos=float(max_frame), angle=90, movable=True, pen=pg.mkPen('r', width=3), label='End', labelOpts={'position': 0.95})
        self.end_line.positionChanged.connect(self.on_end_line_moved)
        self.plot_widget.addItem(self.end_line)

        # Create a LinearRegionItem for visual selection feedback
        self.selection_region = pg.LinearRegionItem(
            values=[float(min_frame), float(max_frame)],
            orientation=pg.LinearRegionItem.Vertical,
            brush=pg.mkBrush(0, 100, 255, 50), # Semi-transparent blue
            movable=False
        )
        self.plot_widget.addItem(self.selection_region)
        self.start_line.positionChanged.connect(self.update_selection_region)
        self.end_line.positionChanged.connect(self.update_selection_region)

        # --- Initialize Crosshair Lines ---
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('gray', width=1, style=Qt.DashLine))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('gray', width=1, style=Qt.DashLine))
        self.v_line.hide() # Hide initially
        self.h_line.hide()
        self.plot_widget.addItem(self.v_line, ignoreBounds=True)
        self.plot_widget.addItem(self.h_line, ignoreBounds=True)

        # --- Connect Mouse Events for Crosshairs ---
        self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)
        self.plot_widget.scene().sigMouseClicked.connect(self.on_plot_clicked) # Keep existing click handler

        # Set initial values in input boxes
        self.start_index_input.setValue(min_frame)
        self.end_index_input.setValue(max_frame)
        # Set range for spinboxes
        self.start_index_input.setRange(min_frame, max_frame)
        self.end_index_input.setRange(min_frame, max_frame)

        # Set plot range to show all data
        x_range = max_frame - min_frame
        if x_range == 0: x_range = 1 # Avoid division by zero
        self.plot_widget.setXRange(min_frame - 0.02 * x_range, max_frame + 0.02 * x_range, padding=0)
        
        y_min, y_max = min(self.frequencies), max(self.frequencies)
        if y_min == y_max:
             y_min -= 0.5
             y_max += 0.5
        y_range = y_max - y_min
        if y_range == 0: y_range = 1 # Avoid division by zero
        self.plot_widget.setYRange(y_min - 0.05 * y_range, y_max + 0.05 * y_range, padding=0)
        
        # Re-enable autorange if needed for future updates, or keep manual control
        # self.plot_widget.enableAutoRange()

        logging.debug(f"Plotted {len(self.frame_indices)} data points")

    def on_mouse_moved(self, evt):
        """Handle mouse movement to show crosshairs."""
        if not self.crosshair_enabled:
            return

        pos = evt # using 'pos' because the event *is* the position in newer pyqtgraph versions
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            x_val = mouse_point.x()
            y_val = mouse_point.y()

            # Update crosshair lines
            if self.v_line:
                self.v_line.setPos(x_val)
                self.v_line.show()
            if self.h_line:
                self.h_line.setPos(y_val)
                self.h_line.show()
        else:
            # Hide crosshairs if mouse leaves the plot area
            if self.v_line:
                self.v_line.hide()
            if self.h_line:
                self.h_line.hide()

    def update_selection_region(self):
        """Update the visual selection region based on line positions."""
        if self.selection_region and self.start_line and self.end_line:
            start_val = int(round(self.start_line.value()))
            end_val = int(round(self.end_line.value()))
            low_val, high_val = sorted([start_val, end_val])
            # Block signals to prevent recursive updates
            self.selection_region.blockSignals(True)
            self.selection_region.setRegion([float(low_val), float(high_val)])
            self.selection_region.blockSignals(False)

    def constrain_line_positions(self):
        """Ensure start line <= end line."""
        if not self.start_line or not self.end_line:
            return
        start_val = int(round(self.start_line.value()))
        end_val = int(round(self.end_line.value()))
        
        if start_val > end_val:
            # Swap positions
            self.start_line.blockSignals(True)
            self.end_line.blockSignals(True)
            self.start_line.setValue(float(end_val))
            self.end_line.setValue(float(start_val))
            self.start_line.blockSignals(False)
            self.end_line.blockSignals(False)
            # Update inputs after swap
            self.start_index_input.setValue(end_val)
            self.end_index_input.setValue(start_val)
        else:
            # Just update inputs to match lines
            self.start_index_input.setValue(start_val)
            self.end_index_input.setValue(end_val)

    def on_start_line_moved(self, new_pos):
        """Handle start line movement."""
        self.constrain_line_positions()
        self.update_selection_region()

    def on_end_line_moved(self, new_pos):
        """Handle end line movement."""
        self.constrain_line_positions()
        self.update_selection_region()

    def on_start_index_changed(self, new_value):
        """Handle start index input change."""
        if self.start_line:
            # Block signal to prevent recursion
            self.start_line.blockSignals(True)
            self.start_line.setValue(float(new_value))
            self.start_line.blockSignals(False)
            self.constrain_line_positions()
            self.update_selection_region()

    def on_end_index_changed(self, new_value):
        """Handle end index input change."""
        if self.end_line:
            # Block signal to prevent recursion
            self.end_line.blockSignals(True)
            self.end_line.setValue(float(new_value))
            self.end_line.blockSignals(False)
            self.constrain_line_positions()
            self.update_selection_region()

    def on_plot_clicked(self, event):
        """Handle mouse click on the plot to select a point."""
        if event.button() == Qt.LeftButton:
            pos = event.scenePos()
            if self.plot_widget.sceneBoundingRect().contains(pos):
                mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
                x_val = mouse_point.x()
                
                if self.frame_indices:
                    # Find index of closest x (frame index)
                    distances = [abs(f - x_val) for f in self.frame_indices]
                    closest_index = distances.index(min(distances))
                    self.selected_frame_index = self.frame_indices[closest_index]
                    
                    # Update the display
                    self.selected_index_display.setText(str(self.selected_frame_index))
                    logging.debug(f"Selected point at index {self.selected_frame_index}")

    def find_closest_record(self, selected_frame_index):
        """Find the record corresponding to the selected frame index."""
        try:
            # Find record with matching frame index
            closest_record = next((r for r in self.current_records if r.get("frameIndex") == selected_frame_index), None)
            
            if closest_record and not closest_record.get("message"):
                try:
                    query = {
                        "filename": self.filename,
                        "moduleName": self.model_name,
                        "projectName": self.project_name,
                        "frameIndex": selected_frame_index,
                        "email": self.email
                    }
                    full_records_cursor = self.db.history_collection.find(query)
                    full_records = list(full_records_cursor)
                    if full_records:
                        closest_record = full_records[0]
                except Exception as db_e:
                    logging.error(f"Error fetching full record from DB: {str(db_e)}")
                    
            return closest_record
        except Exception as e:
            logging.error(f"Error finding closest record: {str(e)}")
            return None

    def select_button_click(self):
        """Handle the Select button click."""
        try:
            if self.selected_frame_index is None:
                self.show_message_box(
                    "Please click on the plot to select a data point first, then click Select.",
                    "Information", "info"
                )
                logging.info("Select button clicked but no point selected")
                return

            start_frame_index = int(round(self.start_line.value())) if self.start_line else 0
            end_frame_index = int(round(self.end_line.value())) if self.end_line else 0
            start_frame_index, end_frame_index = sorted([start_frame_index, end_frame_index])

            # Find the record corresponding to the *selected* frame index, not necessarily within the range
            # You might want to validate if selected_frame_index is within [start, end] range
            # For now, we just find the record for the selected index.
            
            # Re-query the database for the specific selected record to get full data
            try:
                query = {
                    "filename": self.filename,
                    "moduleName": self.model_name,
                    "projectName": self.project_name,
                    "frameIndex": self.selected_frame_index,
                    "email": self.email
                }
                # Assuming get_history_messages can take a frame_index parameter or we query directly
                # Let's use direct query for clarity
                selected_record_cursor = self.db.history_collection.find(query)
                selected_records = list(selected_record_cursor)
                
                if not selected_records:
                     self.show_message_box(
                        f"No detailed record found for the selected frame index {self.selected_frame_index}.",
                        "Warning", "warning"
                    )
                     logging.info(f"No detailed record found for selected frame index {self.selected_frame_index}")
                     return
                
                self.selected_record = selected_records[0] # Take the first one

            except Exception as db_e:
                logging.error(f"Database error fetching selected record: {db_e}")
                self.show_message_box(f"Database error: {str(db_e)}", "Error", "error")
                return

            if not self.selected_record:
                # Fallback if self.selected_record wasn't set correctly
                self.show_message_box(
                    f"No record found for the selected frame index {self.selected_frame_index}.",
                    "Warning", "warning"
                )
                logging.info(f"No record found for selected frame index {self.selected_frame_index}")
                return

            selected_time_local = "N/A"
            if self.selected_record.get("createdAt"):
                try:
                    # Parse the createdAt string
                    created_at_str = self.selected_record.get("createdAt")
                    # Handle potential timezone formats
                    if 'Z' in created_at_str:
                        created_at_str = created_at_str.replace('Z', '+00:00')
                    parsed_time = datetime.datetime.fromisoformat(created_at_str)
                    selected_time_local = parsed_time.strftime('%d-%m-%Y %H:%M:%S.%f')[:-3]
                except Exception as ts_e:
                    logging.warning(f"Could not format timestamp: {ts_e}")
                    selected_time_local = self.selected_record.get("createdAt", "N/A")

            confirmation_message = (
                f"Final Confirmation - Selection Details:\n\n"
                f"📊 Selected Frame Index: {self.selected_record.get('frameIndex')}\n"
                f"🕐 Timestamp: {selected_time_local}\n"
                f"📁 Filename: {self.filename}\n"
                f"🔧 Model: {self.model_name}\n"
                f"📈 Frequency Value: {self.selected_record.get('messageFrequency', 'N/A')}\n\n"
                f"📈 Selected Range:\n"
                f"   📍 Start Frame Index: {start_frame_index}\n"
                f"   📍 End Frame Index: {end_frame_index}\n\n"
                f"✅ Confirm final selection?\n"
                f"The frequency plot will close after confirmation."
            )

            msg = QMessageBox()
            msg.setWindowTitle("Final Confirmation - Frame Selection")
            msg.setText(confirmation_message)
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg.setDefaultButton(QMessageBox.No)
            result = msg.exec_()

            if result == QMessageBox.Yes:
                selected_data = {
                    "filename": self.filename,
                    "model": self.model_name,
                    "frameIndex": self.selected_record.get("frameIndex"),
                    "timestamp": self.selected_record.get("createdAt"),
                    "channelData": self.selected_record.get("message", []),
                    "project_name": self.project_name,
                    "range_start": start_frame_index,
                    "range_end": end_frame_index
                }
                self.time_range_selected.emit(selected_data)
                logging.info(f"Data confirmed for FrameIndex: {self.selected_record.get('frameIndex')}, Range: {start_frame_index} to {end_frame_index}")
                self.show_message_box(
                    f"✅ Selection Confirmed Successfully!\n\n"
                    f"Frame Index {self.selected_record.get('frameIndex')} has been selected.\n"
                    f"Range: {start_frame_index} to {end_frame_index}\n\n"
                    f"The frequency plot will now close.",
                    "Selection Complete", "info"
                )
                self.hide()
            else:
                logging.info(f"User cancelled confirmation for FrameIndex: {self.selected_record.get('frameIndex')}")
        except Exception as e:
            logging.error(f"Error in select button click: {e}", exc_info=True)
            self.show_message_box(f"Error during selection: {str(e)}", "Error", "error")

    def show_message_box(self, message, title, icon_type):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(message)
        if icon_type == "error":
            msg.setIcon(QMessageBox.Critical)
        elif icon_type == "info":
            msg.setIcon(QMessageBox.Information)
        elif icon_type == "warning":
            msg.setIcon(QMessageBox.Warning)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
