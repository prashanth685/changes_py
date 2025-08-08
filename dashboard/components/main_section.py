from PyQt5.QtWidgets import QWidget, QVBoxLayout, QMdiArea, QScrollArea, QMdiSubWindow
from PyQt5.QtCore import Qt
import logging

class MainSection(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.current_widget = None
        self.current_layout = "2x2"
        self.initUI()
        self.parent.project_changed.connect(self.on_project_changed)
        self.parent.sub_tool_bar.layout_selected.connect(self.on_layout_selected)

    def initUI(self):
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea { background-color: #263238; border: none; }
            QScrollBar:vertical { border: none; background: #2c3e50; width: 15px; margin: 0px; }
            QScrollBar::handle:vertical { background: #4a90e2; border-radius: 4px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { background: none; }
            QScrollBar:horizontal { border: none; background: #2c3e50; height: 8px; margin: 0px; }
            QScrollBar::handle:horizontal { background: #4a90e2; border-radius: 4px; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { background: none; }
        """)
        self.mdi_area = QMdiArea()
        self.mdi_area.setStyleSheet("""
            QMdiArea { background-color: #d1d6d9; border: none; }
            QMdiSubWindow { background-color: #d1d6d9; border: 1px solid #d1d6d9; border-radius: 4px; }
            QMdiSubWindow::title { height: 40px; }
        """)
        self.mdi_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.mdi_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.mdi_area.setActivationOrder(QMdiArea.ActivationHistoryOrder)
        self.scroll_area.setWidget(self.mdi_area)
        self.layout.addWidget(self.scroll_area)
        self.setLayout(self.layout)

    def on_project_changed(self, project_name):
        if not project_name:
            self.clear_widget()

    def on_layout_selected(self, layout):
        self.arrange_layout(layout)

    def set_widget(self, widget, feature_name=None, channel_name=None, model_name=None):
        self.clear_widget()
        self.current_widget = widget
        self.layout.addWidget(widget)
        self.scroll_area.hide()
        logging.debug(f"Set widget in MainSection: {type(widget).__name__}")

    def add_subwindow(self, widget, feature_name, channel_name=None, model_name=None):
        try:
            subwindow = QMdiSubWindow()
            subwindow.setWidget(widget)
            subwindow.setOption(QMdiSubWindow.RubberBandMove, False)
            subwindow.setWindowFlags(subwindow.windowFlags() & ~Qt.WindowMinimizeButtonHint)
            title = f"{model_name or ''} - {channel_name or ''} - {feature_name}".strip(" - ")
            subwindow.setWindowTitle(title)
            self.mdi_area.addSubWindow(subwindow)
            subwindow.showNormal()
            subwindow.windowStateChanged.connect(self.on_window_state_changed)
            self.arrange_layout()
            logging.debug(f"Added subwindow with title: {title}")
            return subwindow
        except Exception as e:
            logging.error(f"Failed to add subwindow for {feature_name}: {str(e)}")
            return None

    def clear_widget(self):
        try:
            for subwindow in self.mdi_area.subWindowList():
                try:
                    subwindow.windowStateChanged.disconnect()
                except:
                    pass
                subwindow.close()
                self.mdi_area.removeSubWindow(subwindow)
                widget = subwindow.widget()
                if widget:
                    widget.hide()
                    widget.setParent(None)
                    widget.deleteLater()
                subwindow.setParent(None)
                subwindow.deleteLater()
            if self.current_widget:
                self.layout.removeWidget(self.current_widget)
                self.current_widget.hide()
                self.current_widget.setParent(None)
                self.current_widget.deleteLater()
                self.current_widget = None
            self.scroll_area.show()
            self.mdi_area.update()
            logging.debug("Cleared all subwindows and custom widget")
        except Exception as e:
            logging.error(f"Error in clear_widget: {str(e)}")

    def on_window_state_changed(self, old_state, new_state):
        try:
            if new_state & Qt.WindowMaximized:
                sender = self.sender()
                if isinstance(sender, QMdiSubWindow):
                    self.maximize_subwindow(sender)
                    logging.debug(f"Maximized subwindow: {sender.windowTitle()}")
            elif (old_state & Qt.WindowMaximized) and not (new_state & Qt.WindowMaximized):
                self.arrange_layout()
                logging.debug("Subwindow restored, rearranging layout")
        except Exception as e:
            logging.error(f"Error in on_window_state_changed: {str(e)}")

    def maximize_subwindow(self, subwindow):
        try:
            # Get the viewport dimensions of the scroll area
            viewport = self.scroll_area.viewport()
            viewport_width = viewport.width()
            viewport_height = viewport.height()
            # Adjust for any margins or padding
            margins = self.layout.contentsMargins()
            viewport_width -= (margins.left() + margins.right())
            viewport_height -= (margins.top() + margins.bottom())
            # Set the subwindow geometry to fit within the viewport
            subwindow.setGeometry(0, 0, viewport_width, viewport_height)
            # Ensure the subwindow is visible and active
            subwindow.showMaximized()
            self.mdi_area.setActiveSubWindow(subwindow)
            # Update the scroll area to prevent unnecessary scrollbars
            self.mdi_area.setMinimumSize(viewport_width, viewport_height)
            self.scroll_area.viewport().update()
            logging.debug(f"Maximized subwindow {subwindow.windowTitle()} to {viewport_width}x{viewport_height}")
        except Exception as e:
            logging.error(f"Error maximizing subwindow {subwindow.windowTitle()}: {str(e)}")

    def arrange_layout(self, layout=None):
        try:
            if self.current_widget:
                logging.debug("Skipping MDI arrangement due to custom widget")
                return
            if layout:
                self.current_layout = layout
            subwindows = self.mdi_area.subWindowList()
            if not subwindows:
                self.mdi_area.setMinimumSize(0, 0)
                logging.debug("No subwindows to arrange")
                return
            # Skip arrangement if any subwindow is maximized
            for subwindow in subwindows:
                if subwindow.isMaximized():
                    logging.debug(f"Skipping arrangement due to maximized subwindow: {subwindow.windowTitle()}")
                    return
            rows, cols = map(int, self.current_layout.split('x'))
            viewport_width = self.scroll_area.viewport().width()
            viewport_height = self.scroll_area.viewport().height()
            MIN_SUBWINDOW_WIDTH = 250
            MIN_SUBWINDOW_HEIGHT = 150
            GAP = 5
            subwindow_width = max((viewport_width - (cols + 1) * GAP) // cols, MIN_SUBWINDOW_WIDTH)
            subwindow_height = max((viewport_height - (rows + 1) * GAP) // rows, MIN_SUBWINDOW_HEIGHT)
            total_subwindows = len(subwindows)
            subwindows_per_page = rows * cols
            total_rows_needed = (total_subwindows + cols - 1) // cols
            for idx, subwindow in enumerate(subwindows):
                row = idx // cols
                col = idx % cols
                x = GAP + col * (subwindow_width + GAP)
                y = GAP + row * (subwindow_height + GAP)
                subwindow.setGeometry(x, y, subwindow_width, subwindow_height)
                subwindow.showNormal()
                logging.debug(f"Arranged subwindow {subwindow.windowTitle()} at ({x}, {y}) with size ({subwindow_width}x{subwindow_height})")
            total_width = viewport_width + GAP * 2
            total_height = total_rows_needed * (subwindow_height + GAP) + GAP
            self.mdi_area.setMinimumSize(total_width, total_height)
            self.mdi_area.update()
            logging.info(f"Arranged {len(subwindows)} subwindows in {self.current_layout} grid")
        except Exception as e:
            logging.error(f"Error in arrange_layout: {str(e)}")