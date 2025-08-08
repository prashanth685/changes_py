from PyQt5.QtWidgets import QTreeWidget, QTreeWidgetItem, QMessageBox, QWidget, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
import logging

class TreeView(QWidget):
    model_selected = pyqtSignal(str)
    channel_selected = pyqtSignal(str, str)
    feature_requested = pyqtSignal(str, str, str)  # feature_name, model_name, channel_name

    def __init__(self, parent=None):
        super().__init__(parent)
        self.db = parent.db
        self.parent_widget = parent
        self.project_name = None
        self.selected_channel = None
        self.selected_channel_item = None
        self.selected_model = None
        self.initUI()
        self.parent_widget.project_changed.connect(self.update_project)

    def initUI(self):
        self.tree = QTreeWidget()
        self.tree.header().hide()
        self.tree.setStyleSheet("""
            QTreeWidget { background-color: #232629; color: #ecf0f1; border: none; font-size: 16px; }
            QTreeWidget::item { padding: 8px; border-bottom: 1px solid #2c3e50; }
            QTreeWidget::item:hover { background-color: #34495e; }
            QTreeWidget::item:selected { background-color: #4a90e2; color: white; }
        """)
        self.tree.setFixedWidth(300)
        self.setFixedWidth(300)
        self.setMinimumWidth(300)
        self.setMaximumWidth(300)
        self.tree.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.tree.itemClicked.connect(self.handle_item_clicked)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tree)
        self.setLayout(layout)

    def update_project(self, project_name):
        self.project_name = project_name
        self.add_project_to_tree(project_name)

    def add_project_to_tree(self, project_name):
        self.tree.clear()
        if not project_name:
            self.selected_model = None
            self.selected_channel = None
            self.selected_channel_item = None
            return
        project_item = QTreeWidgetItem(self.tree)
        project_item.setText(0, f"📁 {project_name}")
        project_item.setData(0, Qt.UserRole, {"type": "project", "name": project_name})
        try:
            project_data = self.db.get_project_data(project_name)
            if not project_data or "models" not in project_data:
                self.console_message(f"No models found for project: {project_name}")
                return
            models = project_data.get("models", [])
            if not models:
                self.console_message(f"Empty models list for project: {project_name}")
                return
            for model in models:
                model_name = model.get("name", "")
                if not model_name:
                    self.console_message(f"Model without name in project: {project_name}")
                    continue
                model_item = QTreeWidgetItem(project_item)
                model_item.setExpanded(True)
                model_item.setText(0, f"🖥️ {model_name}")
                model_item.setData(0, Qt.UserRole, {
                    "type": "model",
                    "name": model_name,
                    "project": project_name
                })
                channels = model.get("channels", [])
                for idx, channel in enumerate(channels):
                    channel_name = channel.get("channelName", f"Channel_{idx + 1}")
                    tag_name = model.get("tagName", channel_name)
                    channel_item = QTreeWidgetItem(model_item)
                    channel_item.setText(0, f"📡 {channel_name}")
                    channel_item.setData(0, Qt.UserRole, {
                        "type": "channel",
                        "index": idx,
                        "name": tag_name,
                        "channel_name": channel_name,
                        "model": model_name,
                        "project": project_name
                    })
            if models and not self.selected_model:
                self.selected_model = models[0].get("name")
                for i in range(project_item.childCount()):
                    item = project_item.child(i)
                    if item.data(0, Qt.UserRole).get("type") == "model" and item.data(0, Qt.UserRole).get("name") == self.selected_model:
                        self.tree.setCurrentItem(item)
                        item.setBackground(0, QColor("#4a90e2"))
                        self.model_selected.emit(self.selected_model)
                        self.console_message(f"Automatically selected model: {self.selected_model}")
                        break
        except Exception as e:
            logging.error(f"Error adding project to tree: {str(e)}")
            self.console_message(f"Error adding project to tree: {str(e)}")
            QMessageBox.warning(self, "Error", f"Error adding project to tree: {str(e)}")

    def handle_item_clicked(self, item, column):
        data = item.data(0, Qt.UserRole)
        try:
            if self.selected_channel_item and self.selected_channel_item != item:
                self.selected_channel_item.setBackground(0, QColor("#232629"))
            if data["type"] == "project":
                self.selected_channel = None
                self.selected_channel_item = None
                self.selected_model = None
                if item.childCount() > 0:
                    first_child = item.child(0)
                    first_child_data = first_child.data(0, Qt.UserRole)
                    if first_child_data["type"] == "model":
                        self.selected_model = first_child_data["name"]
                        self.tree.setCurrentItem(first_child)
                        first_child.setBackground(0, QColor("#4a90e2"))
                        self.model_selected.emit(self.selected_model)
                        self.console_message(f"Auto-selected model: {self.selected_model}")
            elif data["type"] == "model":
                self.selected_channel = None
                self.selected_channel_item = None
                self.selected_model = data["name"]
                item.setBackground(0, QColor("#4a90e2"))
                self.model_selected.emit(self.selected_model)
            elif data["type"] == "channel":
                self.selected_channel = data["channel_name"]  # Use channel_name instead of name
                self.selected_channel_item = item
                self.selected_model = data["model"]
                item.setBackground(0, QColor("#28a745"))
                self.channel_selected.emit(self.selected_model, self.selected_channel)
                # Trigger feature display for the current feature if set
                if hasattr(self.parent_widget, 'current_feature') and self.parent_widget.current_feature:
                    self.feature_requested.emit(self.parent_widget.current_feature, self.selected_model, self.selected_channel)
            self.console_message(f"Selected: {data['type']} - {data.get('channel_name', data.get('name', 'Unknown'))}")
        except Exception as e:
            logging.error(f"Error handling tree item click: {str(e)}")
            self.console_message(f"Error handling tree item click: {str(e)}")
            QMessageBox.warning(self, "Error", f"Error handling tree item click: {str(e)}")

    def get_selected_channel(self):
        return self.selected_channel

    def get_selected_model(self):
        return self.selected_model

    def console_message(self, message):
        if hasattr(self.parent_widget, 'console') and self.parent_widget.console:
            self.parent_widget.console.append_to_console(message)