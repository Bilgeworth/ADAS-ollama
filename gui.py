import sys
import os
import io
import subprocess
from contextlib import redirect_stdout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QResizeEvent
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QCheckBox, QSpinBox, QComboBox, QPushButton, QFileDialog, 
    QScrollArea, QFrame, QGridLayout, QSizePolicy, QTextEdit, QPlainTextEdit
)


class SearchGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.setup_resize_handler()
        self.populate_fields()
        self.select_folder = ""
        self.update_command_preview()

    def init_ui(self):
        self.setWindowTitle("Search Configuration")
        self.setGeometry(100, 100, 1200, 800)

        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout()
        self.main_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.main_widget)

        self.setup_info_box()
        self.setup_folder_selection()
        self.setup_scroll_area()
        self.setup_buttons()
        self.setup_output_display()
        self.setup_command_preview()

        self.apply_styles()

    def setup_info_box(self):
        self.info_box = QTextEdit()
        self.info_box.setReadOnly(True)
        self.info_box.setMaximumHeight(100)
        self.main_layout.addWidget(self.info_box)

    def setup_folder_selection(self):
        folder_layout = QHBoxLayout()
        self.folder_label = QLabel("Selected Folder: None")
        folder_button = QPushButton("Select Folder")
        folder_button.clicked.connect(self.select_folder)
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(folder_button)
        self.main_layout.addLayout(folder_layout)

    def setup_scroll_area(self):
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.grid_layout = QGridLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        self.main_layout.addWidget(self.scroll_area)

    def setup_buttons(self):
        button_layout = QHBoxLayout()
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.run_search)
        evaluate_button = QPushButton("Evaluate")
        evaluate_button.clicked.connect(self.run_evaluate)
        button_layout.addWidget(search_button)
        button_layout.addWidget(evaluate_button)
        self.main_layout.addLayout(button_layout)

    def setup_output_display(self):
        self.output_display = QPlainTextEdit()
        self.output_display.setReadOnly(True)
        self.output_display.setMinimumHeight(200)
        self.main_layout.addWidget(self.output_display)

    def setup_command_preview(self):
        self.command_preview = QTextEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setMaximumHeight(100)
        self.main_layout.addWidget(self.command_preview)

    def setup_resize_handler(self):
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.handle_resize)

    def populate_fields(self):
        self.fields = [
            ("Val Data Path", "val_data_path", "sampled_arc_val_data.pkl", "text"),
            ("Test Data Path", "test_data_path", "sampled_arc_test_data.pkl", "text"),
            ("N Repeat", "n_repeat", "5", "spin"),
            ("Multiprocessing", "multiprocessing", True, "checkbox"),
            ("Max Workers", "max_workers", "32", "spin"),
            ("Debug", "debug", True, "checkbox"),
            ("Save Directory", "save_dir", "results/", "text"),
            ("Experiment Name", "archive_name", "arc_llama3.1_results", "text"),
            ("N Generation", "n_generation", "25", "spin"),
            ("Reflect Max", "reflect_max", "3", "spin"),
            ("Debug Max", "debug_max", "3", "spin"),
            ("Model", "model", "llama3.1", "combo", ["mistral-nemo", "gemma2", "llama3.1"])
        ]
        self.add_fields()
        self.update_command_preview() 

    def add_fields(self):
        for i, (label_text, arg_name, default_value, field_type, *options) in enumerate(self.fields):
            row = i // 2  # Assuming 2 columns
            col = i % 2
            if field_type == "checkbox":
                self.add_checkbox(self.grid_layout, row, col, label_text, arg_name, default_value)
            else:
                self.add_input_field(self.grid_layout, row, col, label_text, arg_name, default_value, field_type, options[0] if options else None)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.select_folder = folder
            self.folder_label.setText(f"Selected Folder: {folder}")
            self.load_instructions(folder)
            self.update_command_preview()

    def load_instructions(self, folder):
        description_path = os.path.join(folder, 'description.txt')
        try:
            with open(description_path, 'r') as file:
                instructions = file.read()
            self.info_box.setPlainText(instructions)
        except FileNotFoundError:
            self.info_box.setPlainText("No description.txt found in the selected folder.")

    def add_input_field(self, layout, row, col, label_text, arg_name, default_value, input_type="text", options=None):
        frame = QFrame()
        field_layout = QVBoxLayout(frame)

        label = QLabel(label_text)
        field_layout.addWidget(label)

        if input_type == "text":
            input_field = QLineEdit(default_value)
        elif input_type == "spin":
            input_field = QSpinBox()
            input_field.setRange(0, 1000)
            input_field.setValue(int(default_value))
        elif input_type == "combo":
            input_field = QComboBox()
            input_field.addItems(options)
            input_field.setCurrentText(default_value)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")

        input_field.setObjectName(arg_name)
        field_layout.addWidget(input_field)
        
        frame.setStyleSheet("background-color: #2b2b2b; border-radius: 5px; padding: 5px; margin: 2px;")
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(frame, row, col)

        if hasattr(input_field, 'textChanged'):
            input_field.textChanged.connect(self.update_command_preview)
        elif hasattr(input_field, 'valueChanged'):
            input_field.valueChanged.connect(self.update_command_preview)
        elif hasattr(input_field, 'currentTextChanged'):
            input_field.currentTextChanged.connect(self.update_command_preview)

    def update_command_preview(self):
        cmd_line_args = self.get_argument_values()
        script_path = 'search.py'
        folder = self.folder_label.text().replace("Selected Folder: ", "")
        if folder:
            script_path = os.path.join(folder, 'search.py')
        
        command = [sys.executable, script_path]
        for key, value in cmd_line_args.items():
            if isinstance(value, bool):
                if value:
                    command.append(f'--{key}')
            else:
                command.append(f'--{key}')
                command.append(str(value))
        
        self.command_preview.setPlainText(' '.join(command))

    def run_search(self):
        command = self.command_preview.toPlainText().split()
        self.output_display.clear()
        self.output_display.appendPlainText(f"Running command: {' '.join(command)}\n")
        
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    self.output_display.appendPlainText(output.strip())
                    QApplication.processEvents()  # Update GUI
            
            rc = process.poll()
            if rc != 0:
                error = process.stderr.read()
                self.output_display.appendPlainText(f"Error: {error}")
        except Exception as e:
            self.output_display.appendPlainText(f"Error running search.py: {str(e)}")

    def get_argument_values(self):
        cmd_line_args = {}
        for child in self.findChildren(QWidget):
            if child.objectName():
                if isinstance(child, QLineEdit):
                    cmd_line_args[child.objectName()] = child.text()
                elif isinstance(child, QSpinBox):
                    cmd_line_args[child.objectName()] = child.value()
                elif isinstance(child, QComboBox):
                    cmd_line_args[child.objectName()] = child.currentText()
                elif isinstance(child, QCheckBox):
                    cmd_line_args[child.objectName()] = child.isChecked()
        return cmd_line_args

    def run_evaluate(self):
        cmd_line_args = self.get_argument_values()
        self.output_display.clear()
        self.output_display.appendPlainText("Running evaluate with arguments:")
        self.output_display.appendPlainText(str(cmd_line_args))
        self.output_display.appendPlainText("\nEvaluation output:")

        folder = self.folder_label.text().replace("Selected Folder: ", "")
        search_script_path = os.path.join(folder, 'search.py')
        if not os.path.exists(search_script_path):
            self.output_display.appendPlainText("search.py not found in the selected folder.")
            return

        output = io.StringIO()
        with redirect_stdout(output):
            try:
                subprocess.run([sys.executable, search_script_path] + self.format_args(cmd_line_args), check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running search.py: {e}")

        self.output_display.appendPlainText(output.getvalue())

    def format_args(self, cmd_line_args):
        formatted_args = []
        for key, value in cmd_line_args.items():
            if isinstance(value, bool):
                if value:
                    formatted_args.append(f"--{key}")
            else:
                formatted_args.append(f"--{key}")
                formatted_args.append(str(value))
        return formatted_args

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QLabel, QLineEdit, QCheckBox, QSpinBox, QComboBox, QPushButton, QTextEdit, QPlainTextEdit {
                color: #ffffff;
                font-family: Arial;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #5a5a5a;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
        """)

    def resizeEvent(self, event: QResizeEvent) -> None:
        self.resize_timer.start(100)

    def handle_resize(self):
        num_columns = max(1, self.width() // 300)
        self.reorganize_layout(num_columns)

    
    def add_input_field(self, layout, row, col, label_text, arg_name, default_value, input_type="text", options=None):
        frame = QFrame()
        field_layout = QVBoxLayout(frame)

        label = QLabel(label_text)
        field_layout.addWidget(label)

        if input_type == "text":
            input_field = QLineEdit(default_value)
        elif input_type == "spin":
            input_field = QSpinBox()
            input_field.setRange(0, 1000)
            input_field.setValue(int(default_value))
        elif input_type == "combo":
            input_field = QComboBox()
            input_field.addItems(options)
            input_field.setCurrentText(default_value)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")

        input_field.setObjectName(arg_name)
        field_layout.addWidget(input_field)
        
        frame.setStyleSheet("background-color: #2b2b2b; border-radius: 5px; padding: 5px; margin: 2px;")
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(frame, row, col)

        if hasattr(input_field, 'textChanged'):
            input_field.textChanged.connect(self.update_command_preview)
        elif hasattr(input_field, 'valueChanged'):
            input_field.valueChanged.connect(self.update_command_preview)
        elif hasattr(input_field, 'currentTextChanged'):
            input_field.currentTextChanged.connect(self.update_command_preview)


    def reorganize_layout(self, num_columns):
        for i in reversed(range(self.grid_layout.count())): 
            self.grid_layout.itemAt(i).widget().setParent(None)

        for i, field in enumerate(self.fields):
            row = i // num_columns
            col = i % num_columns
            if field[3] == "checkbox":
                self.add_checkbox(self.grid_layout, row, col, *field[:-1])
            else:
                self.add_input_field(self.grid_layout, row, col, *field)

    def add_checkbox(self, layout, row, col, label_text, arg_name, default_value):
        frame = QFrame()
        field_layout = QVBoxLayout(frame)

        label = QLabel(label_text)
        field_layout.addWidget(label)

        checkbox = QCheckBox()
        checkbox.setChecked(default_value)
        checkbox.setObjectName(arg_name)
        field_layout.addWidget(checkbox)
        
        frame.setStyleSheet("background-color: #2b2b2b; border-radius: 5px; padding: 5px; margin: 2px;")
        frame.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(frame, row, col)
        
        checkbox.stateChanged.connect(self.update_command_preview)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SearchGUI()
    window.show()
    sys.exit(app.exec())