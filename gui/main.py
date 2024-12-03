from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QSplitter,
                             QTextEdit, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog,
                             QTreeView, QLabel, QProgressBar, QStatusBar, QMenuBar, QMenu)
from PyQt6.QtGui import QAction, QFont, QFontMetrics, QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any

from translator import CudaTranslator
from optimizer import MetalOptimizer
from utils.logger import get_logger

logger = get_logger(__name__)

class TranslationWorker(QThread):
    """Handles asynchronous CUDA to Metal translation."""
    finished = pyqtSignal(dict)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, cuda_code: str, optimization_level: int):
        super().__init__()
        self.cuda_code = cuda_code
        self.optimization_level = optimization_level
        self.translator = CudaTranslator()
        self.optimizer = MetalOptimizer()

    def run(self):
        try:
            # Translation process with progress updates
            self.progress.emit(20)
            ast = self.translator.parse_cuda(self.cuda_code)

            self.progress.emit(40)
            metal_code = self.translator.translate_to_metal(ast)

            self.progress.emit(60)
            optimized_code = self.optimizer.optimize_metal_code(
                metal_code,
                self.optimization_level
            )

            self.progress.emit(80)
            performance_metrics = self.optimizer.analyze_performance(optimized_code)

            self.progress.emit(100)
            self.finished.emit({
                'metal_code': optimized_code,
                'metrics': performance_metrics
            })
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            self.error.emit(str(e))

class CUDAMMainWindow(QMainWindow):
    """Main window for the CUDAM GUI application."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CUDAM - CUDA to Metal Translator")
        self.setMinimumSize(1200, 800)
        self.current_file: Optional[Path] = None
        self.translation_worker: Optional[TranslationWorker] = None

        self._setup_ui()
        self._setup_menubar()
        self._setup_statusbar()
        self._setup_shortcuts()
        self._apply_styles()

    def _setup_ui(self):
        """Initialize the main UI components."""
        # Main layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create main splitter
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(self.main_splitter)

        # Project explorer
        self.project_tree = QTreeView()
        self.project_model = QStandardItemModel()
        self.project_tree.setModel(self.project_model)
        self.project_tree.setHeaderHidden(True)
        self.main_splitter.addWidget(self.project_tree)

        # Editor section
        editor_widget = QWidget()
        editor_layout = QVBoxLayout(editor_widget)

        # Editor tabs
        self.editor_tabs = QTabWidget()
        self.cuda_editor = QTextEdit()
        self.metal_editor = QTextEdit()
        self.cuda_editor.setFont(QFont("SF Mono", 12))
        self.metal_editor.setFont(QFont("SF Mono", 12))

        self.editor_tabs.addTab(self.cuda_editor, "CUDA Source")
        self.editor_tabs.addTab(self.metal_editor, "Metal Output")
        editor_layout.addWidget(self.editor_tabs)

        # Control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)

        self.translate_button = QPushButton("Translate")
        self.translate_button.clicked.connect(self.start_translation)
        self.optimize_button = QPushButton("Optimize")
        self.optimize_button.clicked.connect(self.optimize_code)

        control_layout.addWidget(self.translate_button)
        control_layout.addWidget(self.optimize_button)
        editor_layout.addWidget(control_panel)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        editor_layout.addWidget(self.progress_bar)

        self.main_splitter.addWidget(editor_widget)

        # Performance metrics panel
        metrics_widget = QWidget()
        metrics_layout = QVBoxLayout(metrics_widget)
        metrics_layout.addWidget(QLabel("Performance Metrics"))
        self.metrics_view = QTextEdit()
        self.metrics_view.setReadOnly(True)
        metrics_layout.addWidget(self.metrics_view)

        self.main_splitter.addWidget(metrics_widget)

        # Set splitter proportions
        self.main_splitter.setStretchFactor(0, 1)  # Project tree
        self.main_splitter.setStretchFactor(1, 3)  # Editor
        self.main_splitter.setStretchFactor(2, 1)  # Metrics

    def _setup_menubar(self):
        """Setup the application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        new_action = QAction("New", self)
        new_action.setShortcut("Cmd+N")
        new_action.triggered.connect(self.new_file)

        open_action = QAction("Open...", self)
        open_action.setShortcut("Cmd+O")
        open_action.triggered.connect(self.open_file)

        save_action = QAction("Save", self)
        save_action.setShortcut("Cmd+S")
        save_action.triggered.connect(self.save_file)

        file_menu.addAction(new_action)
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)

        # Edit menu
        edit_menu = menubar.addMenu("Edit")

        undo_action = QAction("Undo", self)
        undo_action.setShortcut("Cmd+Z")
        undo_action.triggered.connect(self.cuda_editor.undo)

        redo_action = QAction("Redo", self)
        redo_action.setShortcut("Cmd+Shift+Z")
        redo_action.triggered.connect(self.cuda_editor.redo)

        edit_menu.addAction(undo_action)
        edit_menu.addAction(redo_action)

        # Translation menu
        translation_menu = menubar.addMenu("Translation")

        translate_action = QAction("Translate", self)
        translate_action.setShortcut("Cmd+T")
        translate_action.triggered.connect(self.start_translation)

        optimize_action = QAction("Optimize", self)
        optimize_action.setShortcut("Cmd+Shift+O")
        optimize_action.triggered.connect(self.optimize_code)

        translation_menu.addAction(translate_action)
        translation_menu.addAction(optimize_action)

    def _setup_statusbar(self):
        """Setup the status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

    def _setup_shortcuts(self):
        """Setup additional keyboard shortcuts."""
        # Implementation of keyboard shortcuts
        pass

    def _apply_styles(self):
        """Apply macOS-style CSS styling."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
            QTextEdit {
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #ffffff;
                selection-background-color: #b2d7ff;
            }
            QPushButton {
                background-color: #0066cc;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0052cc;
            }
            QPushButton:pressed {
                background-color: #004099;
            }
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0066cc;
                border-radius: 3px;
            }
        """)

    def start_translation(self):
        """Start the CUDA to Metal translation process."""
        cuda_code = self.cuda_editor.toPlainText()
        if not cuda_code.strip():
            self.statusbar.showMessage("No CUDA code to translate", 3000)
            return

        self.translate_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self.translation_worker = TranslationWorker(cuda_code, optimization_level=2)
        self.translation_worker.finished.connect(self._handle_translation_finished)
        self.translation_worker.progress.connect(self.progress_bar.setValue)
        self.translation_worker.error.connect(self._handle_translation_error)
        self.translation_worker.start()

    def _handle_translation_finished(self, result: Dict[str, Any]):
        """Handle completed translation."""
        self.metal_editor.setPlainText(result['metal_code'])
        self.metrics_view.setPlainText(
            "Performance Metrics:\n" +
            "\n".join(f"{k}: {v}" for k, v in result['metrics'].items())
        )

        self.translate_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.statusbar.showMessage("Translation completed successfully", 3000)

    def _handle_translation_error(self, error_msg: str):
        """Handle translation errors."""
        self.statusbar.showMessage(f"Error: {error_msg}", 5000)
        self.translate_button.setEnabled(True)
        self.progress_bar.setVisible(False)

    def optimize_code(self):
        """Optimize the generated Metal code."""
        metal_code = self.metal_editor.toPlainText()
        if not metal_code.strip():
            self.statusbar.showMessage("No Metal code to optimize", 3000)
            return

        try:
            optimizer = MetalOptimizer()
            optimized_code = optimizer.optimize_metal_code(metal_code, optimization_level=3)
            self.metal_editor.setPlainText(optimized_code)
            self.statusbar.showMessage("Optimization completed", 3000)
        except Exception as e:
            self.statusbar.showMessage(f"Optimization error: {str(e)}", 5000)

    def new_file(self):
        """Create a new file."""
        self.current_file = None
        self.cuda_editor.clear()
        self.metal_editor.clear()
        self.statusbar.showMessage("New file created", 3000)

    def open_file(self):
        """Open a CUDA source file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open CUDA File",
            "",
            "CUDA Files (*.cu *.cuh);;All Files (*.*)"
        )

        if filename:
            try:
                with open(filename, 'r') as f:
                    self.cuda_editor.setPlainText(f.read())
                self.current_file = Path(filename)
                self.statusbar.showMessage(f"Opened {filename}", 3000)
            except Exception as e:
                self.statusbar.showMessage(f"Error opening file: {str(e)}", 5000)

    def save_file(self):
        """Save the current file."""
        if not self.current_file:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Save CUDA File",
                "",
                "CUDA Files (*.cu);;Metal Files (*.metal);;All Files (*.*)"
            )
            if not filename:
                return
            self.current_file = Path(filename)

        try:
            with open(self.current_file, 'w') as f:
                if self.current_file.suffix == '.cu':
                    f.write(self.cuda_editor.toPlainText())
                elif self.current_file.suffix == '.metal':
                    f.write(self.metal_editor.toPlainText())
            self.statusbar.showMessage(f"Saved {self.current_file}", 3000)
        except Exception as e:
            self.statusbar.showMessage(f"Error saving file: {str(e)}", 5000)

def main():
    """Main entry point for the CUDAM GUI application."""
    app = QApplication(sys.argv)

    # Set application-wide attributes
    app.setApplicationName("CUDAM")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("CUDAM")
    app.setOrganizationDomain("cudam.dev")

    window = CUDAMMainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == '__main__':
    main()