import sys
import gc
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSplitter, QSizePolicy, QApplication, QMessageBox, QInputDialog
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt5.QtGui import QIcon, QColor
import logging
from dashboard.components.file_bar import FileBar
from dashboard.components.tool_bar import ToolBar
from dashboard.components.sub_tool_bar import SubToolBar
from dashboard.components.main_section import MainSection
from dashboard.components.tree_view import TreeView
from dashboard.components.console import Console
from dashboard.components.mqtt_status import MQTTStatus
from mqtthandler import MQTTHandler
from features.tabular_view import TabularViewFeature
from features.polar import PolarPlotFeature
from features.time_view import TimeViewFeature
from features.fft_view import FFTViewFeature
from features.waterfall import WaterfallFeature
from features.centerline import CenterLineFeature
from features.orbit import OrbitFeature
from features.trend_view import TrendViewFeature
from features.multi_trend import MultiTrendFeature
from features.bode_plot import BodePlotFeature
from features.history_plot import HistoryPlotFeature
from features.time_report import TimeReportFeature
from features.report import ReportFeature
from select_project import SelectProjectWidget
from create_project import CreateProjectWidget
from project_structure import ProjectStructureWidget
import time

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class Worker(QObject):
    finished = pyqtSignal()
    select_project = pyqtSignal()

    def __init__(self, dashboard):
        super().__init__()
        self.dashboard = dashboard

    def run(self):
        try:
            projects = self.dashboard.db.load_projects()
            if projects and self.dashboard.current_project:
                self.dashboard.load_project(self.dashboard.current_project)
            else:
                self.select_project.emit()
        except Exception as e:
            logging.error(f"Error in deferred initialization: {str(e)}")
            self.dashboard.console.append_to_console(f"Error in deferred initialization: {str(e)}")
        finally:
            self.finished.emit()

class DashboardWindow(QWidget):
    mqtt_status_changed = pyqtSignal(bool)
    project_changed = pyqtSignal(str)
    saving_state_changed = pyqtSignal(bool)

    def __init__(self, db, email, auth_window=None):
        super().__init__()
        self.db = db
        self.email = email
        self.auth_window = auth_window
        self.current_project = None
        self.channel_count = None
        self.open_dashboards = {}
        self.current_feature = None
        self.mqtt_handler = None
        self.feature_instances = {}
        self.sub_windows = {}
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.is_saving = False
        self.mqtt_connected = False
        self.select_project_widget = None
        self.create_project_widget = None
        self.project_structure_widget = None
        self.initUI()
        self.deferred_initialization()

    def initUI(self):
        self.setWindowTitle('Sarayu Desktop Application')
        self.setWindowState(Qt.WindowMaximized)
        app = QApplication.instance()
        app.setStyleSheet("""
            QInputDialog, QMessageBox {
                background-color: #1e2937;
                color: white;
                font-size: 16px;
                border: 1px solid #2c3e50;
                border-radius: 8px;
                padding: 15px;
                width:500px;
            }
            QInputDialog QLineEdit {
                background-color: #2c3e50;
                color: white;
                border: 1px solid #4a90e2;
                padding: 8px;
                border-radius: 4px;
                font-size: 15px;
            }
            QInputDialog QLabel,
            QMessageBox QLabel {
                color: #ecf0f1;
                font-size: 16px;
                padding-bottom: 10px;
            }
            QInputDialog QPushButton,
            QMessageBox QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 5px;
                font-size: 15px;
                min-width: 80px;
                transition: background-color 0.2s ease;
            }
            QInputDialog QPushButton:hover,
            QMessageBox QPushButton:hover {
                background-color: #357abd;
            }
            QInputDialog QPushButton:pressed,
            QMessageBox QPushButton:pressed {
                background-color: #2c5d9b;
            }
            QMdiSubWindow {
                background-color: #d1d6d9;
                border: 1px solid #d1d6d9;
                border-radius: 4px;
            }
            QMdiSubWindow > QWidget {
                background-color: #d1d6d9;
                color: #ecf0f1;
            }
        """)
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)
        self.file_bar = FileBar(self)
        self.file_bar.home_triggered.connect(self.display_dashboard)
        self.file_bar.open_triggered.connect(self.open_project)
        self.file_bar.edit_triggered.connect(self.edit_project_dialog)
        self.file_bar.new_triggered.connect(self.create_project)
        self.file_bar.save_triggered.connect(self.save_action)
        self.file_bar.settings_triggered.connect(self.settings_action)
        self.file_bar.refresh_triggered.connect(self.refresh_action)
        self.file_bar.exit_triggered.connect(self.close)
        main_layout.addWidget(self.file_bar)
        self.tool_bar = ToolBar(self)
        self.tool_bar.feature_selected.connect(self.display_feature_content)
        main_layout.addWidget(self.tool_bar)
        self.sub_tool_bar = SubToolBar(self)
        self.sub_tool_bar.start_saving_triggered.connect(self.start_saving)
        self.sub_tool_bar.stop_saving_triggered.connect(self.stop_saving)
        self.sub_tool_bar.connect_mqtt_triggered.connect(self.connect_mqtt)
        self.sub_tool_bar.disconnect_mqtt_triggered.connect(self.disconnect_mqtt)
        central_widget = QWidget()
        central_layout = QVBoxLayout()
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_widget.setLayout(central_layout)
        main_layout.addWidget(central_widget, 1)
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setContentsMargins(0, 0, 0, 0)
        self.main_splitter.setHandleWidth(1)
        self.main_splitter.setStyleSheet("QSplitter::handle { background-color: #2c3e50; }")
        central_layout.addWidget(self.main_splitter)
        self.tree_view = TreeView(self)
        self.tree_view.setVisible(False)
        self.main_splitter.addWidget(self.tree_view)
        right_container = QWidget()
        right_container.setStyleSheet("background-color: #d1d6d9;")
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_container.setLayout(right_layout)
        self.sub_tool_bar.setVisible(False)
        right_layout.addWidget(self.sub_tool_bar)
        self.main_section = MainSection(self)
        right_layout.addWidget(self.main_section, 1)
        self.main_splitter.addWidget(right_container)
        window_width = self.width() if self.width() > 0 else 1200
        tree_view_width = int(window_width * 0.15)
        right_container_width = int(window_width * 0.85)
        self.main_splitter.setSizes([tree_view_width, right_container_width])
        self.console = Console(self)
        self.mqtt_status = MQTTStatus(self)
        self.console_layout = QVBoxLayout()
        self.console_layout.setContentsMargins(0, 0, 0, 0)
        self.console_layout.setSpacing(0)
        self.console_container = QWidget()
        self.console_container.setStyleSheet("background-color: black;")
        self.console_container.setFixedHeight(80)
        self.console_container.setLayout(self.console_layout)
        self.console_layout.addWidget(self.console.button_container)
        self.console_layout.addWidget(self.console.console_message_area)
        self.console_layout.addWidget(self.mqtt_status)
        main_layout.addWidget(self.console_container)

    def deferred_initialization(self):
        self.worker = Worker(self)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.select_project.connect(self.display_select_project)
        self.thread.start()

    def display_select_project(self):
        self.clear_content_layout()
        self.tree_view.setVisible(False)
        self.sub_tool_bar.setVisible(False)
        self.current_project = None
        self.channel_count = None
        self.file_bar.update_state(project_name=None)
        self.project_changed.emit(None)
        self.setWindowTitle('Sarayu Desktop Application')
        self.select_project_widget = SelectProjectWidget(self)
        self.main_section.set_widget(self.select_project_widget)
        logging.debug("Displayed SelectProjectWidget in MainSection")

    def display_create_project(self):
        self.clear_content_layout()
        self.sub_tool_bar.setVisible(False)
        self.create_project_widget = CreateProjectWidget(self)
        self.main_section.set_widget(self.create_project_widget)
        logging.debug("Displayed CreateProjectWidget in MainSection")

    def display_project_structure(self):
        self.clear_content_layout()
        self.tree_view.setVisible(False)
        self.sub_tool_bar.setVisible(False)
        self.project_structure_widget = ProjectStructureWidget(self)
        self.project_structure_widget.project_selected.connect(self.load_project)
        self.main_section.set_widget(self.project_structure_widget)
        self.main_splitter.setSizes([0, 1200])
        logging.debug("Displayed ProjectStructureWidget in MainSection")

    def load_project(self, project_name):
        self.current_project = project_name
        project_data = self.db.get_project_data(project_name)
        if not project_data:
            self.console.append_to_console(f"Error: Project {project_name} not found.")
            logging.error(f"Project {project_name} not found!")
            self.display_select_project()
            return
        channel_count_map = {
            "DAQ4CH": 4,
            "DAQ8CH": 8,
            "DAQ10CH": 10
        }
        raw_channel_count = project_data.get("channel_count", 4)
        try:
            if isinstance(raw_channel_count, str):
                self.channel_count = channel_count_map.get(raw_channel_count, int(raw_channel_count))
            else:
                self.channel_count = int(raw_channel_count)
            if self.channel_count not in [4, 8, 10]:
                raise ValueError(f"Invalid channel count: {self.channel_count}")
        except (ValueError, TypeError) as e:
            self.console.append_to_console(f"Error: Invalid channel count {raw_channel_count} for project {project_name}. Defaulting to 4.")
            logging.error(f"Invalid channel count {raw_channel_count} for project {project_name}: {str(e)}. Defaulting to 4.")
            self.channel_count = 4
        self.setWindowTitle(f'Sarayu Desktop Application - {self.current_project.upper()}')
        self.tree_view.setVisible(True)
        self.sub_tool_bar.setVisible(True)
        window_width = self.width() if self.width() > 0 else 1200
        tree_view_width = int(window_width * 0.15)
        right_container_width = int(window_width * 0.85)
        self.main_splitter.setSizes([tree_view_width, right_container_width])
        logging.debug(f"TreeView visibility: {self.tree_view.isVisible()}")
        logging.debug(f"SubToolBar visibility: {self.sub_tool_bar.isVisible()}")
        logging.debug(f"Loading project: {project_name} with {self.channel_count} channels")
        self.console.append_to_console(f"Loaded project {project_name} with {self.channel_count} channels")
        self.clear_content_layout()
        if self.project_structure_widget:
            self.project_structure_widget.setParent(None)
            self.project_structure_widget = None
        logging.debug("ProjectStructureWidget removed from MainSection")
        self.file_bar.update_state(project_name=project_name)
        self.project_changed.emit(project_name)
        self.load_project_features()
        QTimer.singleShot(0, self.setup_mqtt)

    def setup_mqtt(self):
        if not self.current_project:
            logging.warning("No project selected for MQTT setup")
            self.console.append_to_console("No project selected for MQTT setup")
            return
        self.cleanup_mqtt()
        try:
            tags = self.get_project_tags()
            if tags:
                self.mqtt_handler = MQTTHandler(self.db, self.current_project)
                self.mqtt_handler.data_received.connect(self.on_data_received)
                self.mqtt_handler.connection_status.connect(self.on_mqtt_status)
                self.mqtt_handler.start()
                logging.info(f"MQTT setup initiated for project: {self.current_project}")
                self.console.append_to_console(f"MQTT setup initiated for project: {self.current_project}")
            else:
                logging.warning(f"No tags found for project: {self.current_project}")
                self.mqtt_connected = False
                self.mqtt_status_changed.emit(False)
                self.console.append_to_console(f"No tags found for project: {self.current_project}")
        except Exception as e:
            logging.error(f"Failed to setup MQTT: {str(e)}")
            self.console.append_to_console(f"Failed to setup MQTT: {str(e)}")
            self.mqtt_connected = False
            self.mqtt_status_changed.emit(False)

    def cleanup_mqtt(self):
        if self.mqtt_handler:
            try:
                self.mqtt_handler.data_received.disconnect()
                self.mqtt_handler.connection_status.disconnect()
                self.mqtt_handler.stop()
                self.mqtt_handler.deleteLater()
                logging.info("Previous MQTT handler stopped")
            except Exception as e:
                logging.error(f"Error stopping MQTT handler: {str(e)}")
            finally:
                self.mqtt_handler = None
                self.mqtt_connected = False
                self.mqtt_status_changed.emit(False)

    def get_project_tags(self):
        try:
            if not self.db.is_connected():
                self.db.reconnect()
            project_data = self.db.get_project_data(self.current_project)
            if not project_data or "models" not in project_data:
                logging.warning(f"No models found for project: {self.current_project}")
                return []
            tags = []
            for model in project_data["models"]:
                model_name = model.get("name")
                tag_name = model.get("tagName", "")
                if tag_name and model_name:
                    tags.append({"tag_name": tag_name, "model_name": model_name})
            logging.debug(f"Retrieved tags for project {self.current_project}: {tags}")
            return tags
        except Exception as e:
            logging.error(f"Failed to retrieve project tags: {str(e)}")
            return []

    def on_data_received(self, feature_name, tag_name, model_name, channel_index, values, sample_rate, frame_index):
        try:
            for key, feature_instance in self.feature_instances.items():
                instance_feature, instance_model, instance_channel, _ = key
                if (instance_feature == feature_name and instance_model == model_name and
                    (instance_channel is None or channel_index == -1 or
                     (instance_channel and f"Channel_{channel_index + 1}" == instance_channel))):
                    if hasattr(feature_instance, 'on_data_received'):
                        QTimer.singleShot(0, lambda: self._update_feature(
                            instance_feature, instance_model, instance_channel,
                            feature_instance, tag_name, values, sample_rate, frame_index
                        ))
                        logging.debug(f"Processed data for {feature_name}/{model_name}/channel_{channel_index}, frame {frame_index}")
        except Exception as e:
            logging.error(f"Error in on_data_received for {feature_name}/{model_name}/channel_{channel_index}, frame {frame_index}: {str(e)}")
            self.console.append_to_console(f"Error processing data for {feature_name}: {str(e)}")

    def _update_feature(self, feature_name, model_name, channel, feature_instance, tag_name, values, sample_rate, frame_index):
        try:
            feature_instance.on_data_received(tag_name, model_name, values, sample_rate, frame_index)
            logging.debug(f"Updated feature {feature_name}/{model_name}/{channel or 'No Channel'}, frame {frame_index}")
        except Exception as e:
            logging.error(f"Error updating feature {feature_name}/{model_name}/{channel or 'No Channel'}, frame {frame_index}: {str(e)}")

    def on_mqtt_status(self, message):
        self.mqtt_connected = "Connected" in message
        self.mqtt_status_changed.emit(self.mqtt_connected)
        self.console.append_to_console(f"MQTT Status: {message}")

    def load_project_features(self):
        try:
            if not self.db.is_connected():
                self.db.reconnect()
            self.tree_view.tree.clear()
            self.tree_view.add_project_to_tree(self.current_project)
            for i in range(self.tree_view.tree.topLevelItemCount()):
                item = self.tree_view.tree.topLevelItem(i)
                if item.text(0) == f"📁 {self.current_project}":
                    item.setExpanded(True)
                    self.tree_view.tree.setCurrentItem(item)
                    self.tree_view.tree.scrollToItem(item)
                    break
            logging.debug(f"Loaded project features for: {self.current_project}")
        except Exception as e:
            logging.error(f"Failed to load project features: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to load project features: {str(e)}")

    def open_project(self):
        try:
            if not self.db.is_connected():
                self.db.reconnect()
            projects = self.db.load_projects()
            if not projects:
                QMessageBox.warning(self, "Error", "No projects available to open!")
                return
            self.display_project_structure()
        except Exception as e:
            logging.error(f"Error opening project: {str(e)}")
            QMessageBox.warning(self, "Error", f"Error opening project: {str(e)}")

    def create_project(self):
        self.display_create_project()

    def edit_project_dialog(self):
        if not self.current_project:
            QMessageBox.warning(self, "Error", "No project selected to edit!")
            return
        project_data = self.db.get_project_data(self.current_project)
        if not project_data:
            QMessageBox.warning(self, "Error", "Project data not found!")
            return
        self.clear_content_layout()
        self.tree_view.setVisible(False)
        self.sub_tool_bar.setVisible(False)
        self.create_project_widget = CreateProjectWidget(
            parent=self,
            edit_mode=True,
            existing_project_name=self.current_project,
            existing_models=project_data.get("models", []),
            existing_channel_count=project_data.get("channel_count", "DAQ4CH")
        )
        self.create_project_widget.project_edited.connect(self.handle_project_edited)
        self.main_section.set_widget(self.create_project_widget)
        logging.debug(f"Opened CreateProjectWidget in edit mode for {self.current_project}")

    def handle_project_edited(self, new_project_name, updated_models, channel_count):
        try:
            if not self.db.is_connected():
                self.db.reconnect()
            valid_channel_counts = {
                "DAQ4CH": 4,
                "DAQ8CH": 8,
                "DAQ10CH": 10
            }
            required_channels = valid_channel_counts.get(channel_count)
            if not required_channels:
                try:
                    required_channels = int(channel_count)
                    if required_channels not in [4, 8, 10]:
                        raise ValueError(f"Invalid channel count: {channel_count}")
                except (ValueError, TypeError):
                    QMessageBox.warning(self, "Error", f"Invalid channel count: {channel_count}")
                    return
            for model in updated_models:
                if len(model.get("channels", [])) != required_channels:
                    model_channels = model.get("channels", [])
                    if len(model_channels) < required_channels:
                        model_channels.extend([
                            {
                                "channelName": f"Channel_{j+1}",
                                "type": "Displacement",
                                "sensitivity": "1.0",
                                "unit": "mil",
                                "correctionValue": "",
                                "gain": "",
                                "unitType": "",
                                "angle": "",
                                "angleDirection": "Right",
                                "shaft": ""
                            } for j in range(len(model_channels), required_channels)
                        ])
                    elif len(model_channels) > required_channels:
                        model_channels = model_channels[:required_channels]
                    model["channels"] = model_channels
            success, message = self.db.edit_project(self.current_project, new_project_name, updated_models, channel_count)
            if success:
                self.current_project = new_project_name
                self.channel_count = required_channels
                self.setWindowTitle(f'Sarayu Desktop Application - {self.current_project.upper()}')
                self.load_project(new_project_name)
                if self.current_feature:
                    self.display_feature_content(self.current_feature)
                for key, instance in self.feature_instances.items():
                    if hasattr(instance, 'refresh_channel_properties'):
                        instance.refresh_channel_properties()
                QMessageBox.information(self, "Success", message)
                self.file_bar.update_state(project_name=new_project_name)
                self.project_changed.emit(new_project_name)
                self.display_select_project()
                self.console.append_to_console(f"Project updated: {new_project_name} with {required_channels} channels")
            else:
                QMessageBox.warning(self, "Error", message)
        except Exception as e:
            logging.error(f"Error handling edited project: {str(e)}")
            QMessageBox.warning(self, "Error", f"Error saving edited project: {str(e)}")

    def edit_channel_dialog(self):
        selected_model = self.tree_view.get_selected_model()
        selected_channel = self.tree_view.get_selected_channel()
        if not selected_model or not selected_channel:
            QMessageBox.warning(self, "Error", "Please select a channel to edit!")
            return
        project_data = self.db.get_project_data(self.current_project)
        if not project_data:
            QMessageBox.warning(self, "Error", "Project data not found!")
            return
        model = next((m for m in project_data["models"] if m["name"] == selected_model), None)
        if not model:
            QMessageBox.warning(self, "Error", f"Model '{selected_model}' not found!")
            return
        channel = next((c for c in model["channels"] if c["channelName"] == selected_channel), None)
        if not channel:
            QMessageBox.warning(self, "Error", f"Channel '{selected_channel}' not found!")
            return
        properties = ["type", "sensitivity", "unit", "correctionValue", "gain", "unitType", "angle", "angleDirection", "shaft"]
        updated_properties = {}
        for prop in properties:
            value, ok = QInputDialog.getText(self, f"Edit {prop.capitalize()}", f"Enter new {prop.capitalize()} (current: {channel.get(prop, 'None')})")
            if ok and value:
                updated_properties[prop] = value
        if updated_properties:
            success, message = self.db.update_channel_properties(self.current_project, selected_model, selected_channel, updated_properties)
            if success:
                QMessageBox.information(self, "Success", message)
                for key, instance in self.feature_instances.items():
                    if hasattr(instance, 'refresh_channel_properties'):
                        instance.refresh_channel_properties()
                self.load_project_features()
            else:
                QMessageBox.warning(self, "Error", message)

    def start_saving(self):
        selected_model = self.tree_view.get_selected_model()
        if not selected_model:
            QMessageBox.warning(self, "Error", "Please select a model to save data!")
            return
        if not self.current_project:
            QMessageBox.warning(self, "Error", "No project selected!")
            return
        time_view_keys = [k for k in self.feature_instances.keys() if k[0] == "Time View" and k[1] == selected_model]
        if time_view_keys:
            key = max(time_view_keys, key=lambda k: k[3])
            feature_instance = self.feature_instances.get(key)
            try:
                feature_instance.start_saving()
                self.is_saving = True
                self.saving_state_changed.emit(True)
                logging.info("Started saving data from existing TimeViewFeature")
            except Exception as e:
                logging.error(f"Failed to start saving: {str(e)}")
                QMessageBox.warning(self, "Error", f"Failed to start saving: {str(e)}")
        else:
            try:
                if not self.db.is_connected():
                    self.db.reconnect()
                unique_id = int(time.time() * 1000)
                key = ("Time View", selected_model, None, unique_id)
                feature_instance = TimeViewFeature(
                    self, self.db, self.current_project, model_name=selected_model, console=self.console
                )
                self.feature_instances[key] = feature_instance
                feature_instance.start_saving()
                self.is_saving = True
                self.saving_state_changed.emit(True)
                logging.info(f"Created new TimeViewFeature for saving data, key: {key}")
            except Exception as e:
                logging.error(f"Failed to create and start saving with new TimeViewFeature: {str(e)}")
                QMessageBox.warning(self, "Error", f"Failed to start saving: {str(e)}")

    def stop_saving(self):
        selected_model = self.tree_view.get_selected_model()
        if not selected_model:
            QMessageBox.warning(self, "Error", "Please select a model to stop saving!")
            return
        time_view_keys = [k for k in self.feature_instances.keys() if k[0] == "Time View" and k[1] == selected_model]
        if not time_view_keys:
            QMessageBox.warning(self, "Error", "No Time View feature initialized for the selected model!")
            return
        key = max(time_view_keys, key=lambda k: k[3])
        feature_instance = self.feature_instances.get(key)
        try:
            feature_instance.stop_saving()
            self.is_saving = False
            self.saving_state_changed.emit(False)
            logging.info("Stopped saving data from dashboard")
        except Exception as e:
            logging.error(f"Failed to stop saving: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to stop saving: {str(e)}")

    def display_feature_content(self, feature_name):
        try:
            logging.debug(f"Attempting to display feature: {feature_name} for project: {self.current_project} with channel_count: {self.channel_count}")
            if not self.current_project:
                self.console.append_to_console(f"No project selected for {feature_name}.")
                logging.warning(f"No project selected for {feature_name}")
                return
            self.current_feature = feature_name
            self.is_saving = False
            self.saving_state_changed.emit(False)
            self.sub_tool_bar.setVisible(True)
            current_console_height = self.console.console_message_area.height()
            selected_model = self.tree_view.get_selected_model()
            if not selected_model:
                self.console.append_to_console(f"Please select a model to view {feature_name}.")
                logging.warning(f"No model selected for {feature_name}")
                return
            project_data = self.db.get_project_data(self.current_project)
            if not project_data:
                self.console.append_to_console(f"Project {self.current_project} not found in database.")
                logging.error(f"Project {self.current_project} not found!")
                return
            model = next((m for m in project_data["models"] if m["name"] == selected_model), None)
            if not model:
                self.console.append_to_console(f"Model {selected_model} not found in project {self.current_project}.")
                logging.error(f"Model {selected_model} not found in project {self.current_project}!")
                return
            selected_channel = self.tree_view.get_selected_channel() if feature_name not in ["Time View", "Time Report", "Tabular View"] else None
            if not selected_channel and feature_name not in ["Time View", "Time Report", "Tabular View"]:
                self.console.append_to_console(f"Please select a channel for {feature_name} in model {selected_model}.")
                logging.warning(f"No channel selected for {feature_name} in model {selected_model}")
                return
            channel_list = [None] if feature_name in ["Time View", "Time Report", "Tabular View"] else [selected_channel]
            feature_classes = {
                "Tabular View": TabularViewFeature,
                "Time View": TimeViewFeature,
                "Time Report": TimeReportFeature,
                "FFT": FFTViewFeature,
                "Waterfall": WaterfallFeature,
                "Centerline": CenterLineFeature,
                "Orbit": OrbitFeature,
                "Trend View": TrendViewFeature,
                "Multiple Trend View": MultiTrendFeature,
                "Bode Plot": BodePlotFeature,
                "History Plot": HistoryPlotFeature,
                "Polar Plot": PolarPlotFeature,
                "Report": ReportFeature
            }
            if feature_name not in feature_classes:
                logging.warning(f"Unknown feature: {feature_name}")
                QMessageBox.warning(self, "Error", f"Unknown feature: {feature_name}")
                return
            for channel in channel_list:
                unique_id = int(time.time() * 1000)
                key = (feature_name, selected_model, channel, unique_id)
                try:
                    if not self.db.is_connected():
                        self.db.reconnect()
                    feature_kwargs = {
                        "parent": self,
                        "db": self.db,
                        "project_name": self.current_project,
                        "channel": channel,
                        "model_name": selected_model,
                        "console": self.console
                    }
                    if feature_name in ["Orbit", "FFT"]:
                        feature_kwargs["channel_count"] = self.channel_count
                    feature_instance = feature_classes[feature_name](**feature_kwargs)
                    if feature_name == "Tabular View":
                        logging.debug(f"TabularViewFeature initialized for model {selected_model}, channel {channel or 'None'}; displays all {self.channel_count} channels")
                    else:
                        logging.debug(f"Initialized {feature_name} for model {selected_model}, channel {channel or 'None'}")
                    if feature_name in ["Orbit", "FFT"] and channel and hasattr(feature_instance, 'update_selected_channel'):
                        feature_instance.update_selected_channel(channel)
                    self.feature_instances[key] = feature_instance
                    widget = feature_instance.get_widget()
                    if widget:
                        sub_window = self.main_section.add_subwindow(
                            widget,
                            feature_name,
                            channel_name=channel,
                            model_name=selected_model
                        )
                        if sub_window:
                            self.sub_windows[key] = sub_window
                            sub_window.closeEvent = lambda event, k=key: self.on_subwindow_closed(event, k)
                            sub_window.show()
                            logging.debug(f"Created new subwindow for {key}, ID: {id(sub_window)}")
                        else:
                            logging.error(f"Failed to create subwindow for {feature_name}/{selected_model}/{channel or 'No Channel'}")
                            QMessageBox.warning(self, "Error", f"Failed to create subwindow for {feature_name}")
                            del self.feature_instances[key]
                    else:
                        logging.error(f"Feature {feature_name} returned invalid widget")
                        QMessageBox.warning(self, "Error", f"Feature {feature_name} failed to initialize")
                        del self.feature_instances[key]
                    self.console.console_message_area.setFixedHeight(current_console_height)
                except Exception as e:
                    logging.error(f"Failed to load feature {feature_name} for channel {channel or 'No Channel'}: {str(e)}")
                    QMessageBox.warning(self, "Error", f"Failed to load {feature_name}: {str(e)}")
                    if key in self.feature_instances:
                        del self.feature_instances[key]
            self.main_section.arrange_layout()
            self.console.console_message_area.setFixedHeight(current_console_height)
        except Exception as e:
            logging.error(f"Error displaying feature content: {str(e)}")
            QMessageBox.warning(self, "Error", f"Error displaying feature: {str(e)}")

    def on_subwindow_closed(self, event, key):
        try:
            feature_name, model_name, channel_name, unique_id = key
            logging.debug(f"Closing subwindow for key: {key}, ID: {id(self.sub_windows.get(key))}")
            sub_window = self.sub_windows.get(key)
            if not sub_window:
                logging.warning(f"No subwindow found for key: {key}")
                event.accept()
                return
            if sub_window.isMaximized():
                sub_window.showNormal()
                logging.debug(f"Restored maximized subwindow for {key}")
            if key in self.feature_instances:
                instance = self.feature_instances[key]
                if hasattr(instance, 'cleanup'):
                    try:
                        instance.cleanup()
                        logging.debug(f"Called cleanup for {feature_name}/{model_name}/{channel_name or 'No Channel'}")
                    except Exception as e:
                        logging.error(f"Error in cleanup for {key}: {str(e)}")
                widget = instance.get_widget()
                if widget:
                    try:
                        widget.hide()
                        widget.setParent(None)
                        widget.deleteLater()
                        logging.debug(f"Cleaned up widget for {key}")
                    except Exception as e:
                        logging.error(f"Error cleaning up widget for {key}: {str(e)}")
                del self.feature_instances[key]
                logging.debug(f"Removed feature instance for {key}")
            try:
                sub_window.close()
                self.main_section.mdi_area.removeSubWindow(sub_window)
                sub_window.setParent(None)
                sub_window.deleteLater()
                logging.debug(f"Removed subwindow from MDI area for {key}, ID: {id(sub_window)}")
            except Exception as e:
                logging.error(f"Error removing subwindow for {key}: {str(e)}")
            del self.sub_windows[key]
            if self.current_feature == feature_name:
                if not any(k[0] == feature_name for k in self.feature_instances.keys()):
                    self.current_feature = None
                    self.is_saving = False
                    self.saving_state_changed.emit(False)
                    logging.debug(f"Reset current_feature as no instances of {feature_name} remain")
            self.main_section.mdi_area.update()
            self.main_section.scroll_area.viewport().update()
            self.main_section.arrange_layout()
            self.main_section.mdi_area.setMinimumSize(0, 0)
            gc.collect()
            logging.debug(f"Completed cleanup for subwindow: {key}")
        except Exception as e:
            logging.error(f"Error cleaning up subwindow for {key}: {str(e)}")
        event.accept()

    def save_action(self):
        if self.current_project:
            try:
                if not self.db.is_connected():
                    self.db.reconnect()
                project_data = self.db.get_project_data(self.current_project)
                if project_data:
                    QMessageBox.information(self, "Save", f"Data for project '{self.current_project}' saved successfully!")
                else:
                    QMessageBox.warning(self, "Save Error", "No data to save for the selected project!")
            except Exception as e:
                logging.error(f"Error saving project: {str(e)}")
                QMessageBox.warning(self, "Error", f"Error saving project: {str(e)}")
        else:
            QMessageBox.warning(self, "Save Error", "No project selected to save!")

    def refresh_action(self):
        try:
            if self.current_project and self.current_feature:
                self.display_feature_content(self.current_feature)
                QMessageBox.information(self, "Refresh", f"Refreshed view for '{self.current_feature}'!")
            else:
                self.display_select_project()
                QMessageBox.information(self, "Refresh", "Refreshed project selection view!")
        except Exception as e:
            logging.error(f"Error refreshing view: {str(e)}")
            QMessageBox.warning(self, "Error", f"Error refreshing view: {str(e)}")

    def display_dashboard(self):
        if not self.current_project:
            self.display_select_project()
            return
        self.current_feature = None
        self.is_saving = False
        self.saving_state_changed.emit(False)

    def clear_content_layout(self):
        try:
            logging.debug("Starting clear_content_layout")
            for key in list(self.sub_windows.keys()):
                sub_window = self.sub_windows.get(key)
                if sub_window:
                    try:
                        if sub_window.isMaximized():
                            sub_window.showNormal()
                        sub_window.close()
                        self.main_section.mdi_area.removeSubWindow(sub_window)
                        sub_window.setParent(None)
                        sub_window.deleteLater()
                        logging.debug(f"Closed subwindow for {key} during clear_content_layout, ID: {id(sub_window)}")
                    except Exception as e:
                        logging.error(f"Error closing subwindow {key}: {str(e)}")
            self.sub_windows.clear()
            logging.debug("Cleared all subwindows")
            for key in list(self.feature_instances.keys()):
                try:
                    instance = self.feature_instances[key]
                    if hasattr(instance, 'cleanup'):
                        instance.cleanup()
                        logging.debug(f"Called cleanup for feature instance {key}")
                    widget = instance.get_widget()
                    if widget:
                        widget.hide()
                        widget.setParent(None)
                        widget.deleteLater()
                        logging.debug(f"Cleaned up widget for {key}")
                    del self.feature_instances[key]
                    logging.debug(f"Removed feature instance for {key}")
                except Exception as e:
                    logging.error(f"Error cleaning up feature instance {key}: {str(e)}")
            self.main_section.clear_widget()
            self.main_section.mdi_area.setMinimumSize(0, 0)
            self.main_section.mdi_area.update()
            self.main_section.scroll_area.viewport().update()
            gc.collect()
            logging.debug("Completed clear_content_layout")
            logging.debug(f"Current widget in MainSection: {self.main_section.current_widget}")
        except Exception as e:
            logging.error(f"Error clearing content layout: {str(e)}")

    def settings_action(self):
        QMessageBox.information(self, "Settings", "Settings functionality not implemented yet.")

    def back_to_login(self):
        try:
            if self.auth_window:
                self.auth_window.show()
                self.auth_window.showMaximized()
                self.close()
        except Exception as e:
            logging.error(f"Error returning to login: {str(e)}")
            QMessageBox.warning(self, "Error", f"Failed to return to login: {str(e)}")

    def closeEvent(self, event):
        try:
            if self.timer.isActive():
                self.timer.stop()
            self.cleanup_mqtt()
            self.clear_content_layout()
            if hasattr(self, 'thread') and self.thread.isRunning():
                self.thread.quit()
                self.thread.wait()
            if self.db and self.db.is_connected():
                self.db.close_connection()
            app = QApplication.instance()
            if app:
                app.quit()
        except Exception as e:
            logging.error(f"Error during closeEvent: {str(e)}")
        finally:
            event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.tree_view.isVisible():
            window_width = self.width()
            tree_view_width = int(window_width * 0.15)
            right_container_width = int(window_width * 0.85)
            self.main_splitter.setSizes([tree_view_width, right_container_width])
        self.main_section.arrange_layout()

    def connect_mqtt(self):
        if self.mqtt_connected:
            self.console.append_to_console("Already connected to MQTT")
            return
        QTimer.singleShot(0, self.setup_mqtt)

    def disconnect_mqtt(self):
        if not self.mqtt_connected:
            self.console.append_to_console("Already disconnected from MQTT")
            return
        try:
            self.cleanup_mqtt()
            self.mqtt_connected = False
            self.mqtt_status_changed.emit(False)
            logging.info(f"MQTT disconnected for project: {self.current_project}")
            self.console.append_to_console(f"MQTT disconnected for project: {self.current_project}")
        except Exception as e:
            logging.error(f"Failed to disconnect MQTT: {str(e)}")
            self.console.append_to_console(f"Failed to disconnect MQTT: {str(e)}")