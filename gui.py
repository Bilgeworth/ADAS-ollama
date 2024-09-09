import sys
import os
import re
import subprocess
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QPushButton, QProgressBar, QFileDialog, QSplitter, QLineEdit
from PyQt6.QtGui import QIcon

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ADAS GUI")
        self.setGeometry(100, 100, 800, 600)
        
        self.setWindowIcon(QIcon('favicon.ico'))

        # Initialize arg_inputs
        self.arg_inputs = {}

        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Layouts
        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # Left side widgets, self. is for instance attributes
        description_label = QLabel("Description:")
        description_textbox = QTextEdit()
        description_textbox.setReadOnly(True)
        folder_path_label = QLabel("Folder Path:")
        self.folder_path_preview = QLabel("")  # Define as instance attribute
        select_folder_button = QPushButton("Select Folder")
        select_folder_button.clicked.connect(self.select_folder)  # Connect button click to select_folder method
        command_args_label = QLabel("Command Line Arguments:")
        self.command_args_preview = QVBoxLayout()  # Use QVBoxLayout for command line arguments
        command_preview_label = QLabel("Command Preview:")
        self.command_preview = QLabel()
        run_command_button = QPushButton("Run Command")
        run_command_button.clicked.connect(self.run_command)  # Connect button click to run_command method
        progress_bar = QProgressBar()
        output_label = QLabel("Output:")
        output_textbox = QTextEdit()
        output_textbox.setReadOnly(True)

        left_layout.addWidget(description_label)
        left_layout.addWidget(description_textbox)
        left_layout.addWidget(folder_path_label)
        left_layout.addWidget(self.folder_path_preview)
        left_layout.addWidget(select_folder_button)
        left_layout.addWidget(command_args_label)
        left_layout.addLayout(self.command_args_preview)  # Add QVBoxLayout to the main layout
        left_layout.addWidget(command_preview_label)
        left_layout.addWidget(self.command_preview)
        left_layout.addWidget(run_command_button)  # Add run command button
        left_layout.addWidget(progress_bar)
        left_layout.addWidget(output_label)
        left_layout.addWidget(output_textbox)

        # Right side widgets
        api_call_label = QLabel("API Call:")
        api_call_textbox = QTextEdit()
        api_call_textbox.setReadOnly(True)
        api_response_label = QLabel("API Response:")
        api_response_textbox = QTextEdit()
        api_response_textbox.setReadOnly(True)

        right_layout.addWidget(api_call_label)
        right_layout.addWidget(api_call_textbox)
        right_layout.addWidget(api_response_label)
        right_layout.addWidget(api_response_textbox)

        # Use QSplitter to divide the space between left and right layouts
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 2)  # Left layout takes 2/3 of the space
        splitter.setStretchFactor(1, 1)  # Right layout takes 1/3 of the space

        # Add splitter to main layout
        main_layout.addWidget(splitter)

        # Set main layout on main widget
        main_widget.setLayout(main_layout)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")  # Open file dialog to select folder
        if folder_path:
            relative_path = os.path.relpath(folder_path, start=os.getcwd())
            self.folder_path_preview.setText(relative_path)  # Update folder path preview label
            self.update_command_preview()
            self.search_and_parse_main_py(relative_path)

    def search_and_parse_main_py(self, folder_path):
        main_py_path = None
        for root, dirs, files in os.walk(folder_path):
            if 'main.py' in files:
                main_py_path = os.path.join(root, 'main.py')
                break

        if main_py_path:
            self.parse_main_py(main_py_path)

    def parse_main_py(self, main_py_path):
        with open(main_py_path, 'r') as file:
            content = file.read()

        # Regular expression to find all parser.add_argument lines
        arg_pattern = re.compile(r'parser.add_argument\(\s*\'(--\w+)\',\s*type=(\w+),\s*default=(.*?)(,|\))')
        args = arg_pattern.findall(content)

        # Clear previous command line arguments
        for i in reversed(range(self.command_args_preview.count())): 
            self.command_args_preview.itemAt(i).widget().setParent(None)

        # Add input boxes for each argument
        self.arg_inputs = {}
        for arg in args:
            arg_name, arg_type, arg_default, _ = arg
            arg_default = arg_default.strip().strip("'\"")
            label = QLabel(arg_name)
            input_box = QLineEdit()
            input_box.setPlaceholderText(arg_default)
            input_box.textChanged.connect(self.update_command_preview)  # Connect textChanged signal to update_command_preview
            self.command_args_preview.addWidget(label)
            self.command_args_preview.addWidget(input_box)
            self.arg_inputs[arg_name] = input_box

    def update_command_preview(self):
        command = "python " + self.folder_path_preview.text() + "/main.py"
        for arg_name, input_box in self.arg_inputs.items():
            arg_value = input_box.text()
            if arg_value:
                command += f" {arg_name} {arg_value}"
        self.command_preview.setText(command)

    def run_command(self):
        command = self.command_preview.text()
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            output = result.stdout + result.stderr
        except Exception as e:
            output = str(e)
        self.findChild(QTextEdit, "output_textbox").setPlainText(output)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())