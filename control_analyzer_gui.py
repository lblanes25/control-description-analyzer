import sys
import os
import pandas as pd
import threading
import webbrowser
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QLineEdit, QTextEdit, QPushButton,
                             QTabWidget, QFileDialog, QProgressBar, QMessageBox,
                             QGroupBox, QFormLayout, QComboBox, QTableWidget,
                             QTableWidgetItem, QHeaderView, QCheckBox, QSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QFont, QPixmap

# Import your analyzer code
from control_analyzer import EnhancedControlAnalyzer
from visualization import generate_core_visualizations


class AnalyzerWorker(QThread):
    """Worker thread to run analysis in background"""
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)

    def __init__(self, analyzer, file_path=None, control_data=None, options=None):
        super().__init__()
        self.analyzer = analyzer
        self.file_path = file_path
        self.control_data = control_data
        self.options = options

    def run(self):
        try:
            if self.file_path:
                # Analyze Excel file
                self.progress_signal.emit(10, "Starting analysis...")

                # Extract options
                id_col = self.options.get('id_column', 'Control_ID')
                desc_col = self.options.get('desc_column', 'Control_Description')
                freq_col = self.options.get('frequency_column')
                type_col = self.options.get('type_column')
                risk_col = self.options.get('risk_column')

                # Read the file to get control count for progress updates
                df = pd.read_excel(self.file_path)
                total_controls = len(df)

                # Create a progress callback
                def progress_callback(current, total, message="Processing"):
                    percentage = int((current / total) * 80) + 10  # Scale to 10-90%
                    self.progress_signal.emit(percentage, message)

                # Patch the analyzer to use our progress callback
                original_analyze = self.analyzer.analyze_control

                results = []
                try:
                    for i, (_, row) in enumerate(df.iterrows()):
                        control_id = row[id_col]
                        description = row[desc_col]

                        # Get optional fields if available
                        frequency = row.get(freq_col) if freq_col and freq_col in row else None
                        control_type = row.get(type_col) if type_col and type_col in row else None
                        risk_description = row.get(risk_col) if risk_col and risk_col in row else None

                        # Analyze this control
                        result = original_analyze(control_id, description, frequency, control_type, risk_description)
                        results.append(result)

                        # Update progress
                        progress_callback(i + 1, total_controls, f"Analyzing control {i + 1} of {total_controls}")

                        # Check if thread has been requested to stop
                        if self.isInterruptionRequested():
                            self.progress_signal.emit(0, "Analysis cancelled")
                            return
                finally:
                    # Restore original method
                    self.analyzer.analyze_control = original_analyze

            else:
                # Analyze single control data
                self.progress_signal.emit(30, "Analyzing control...")
                control_id = self.control_data.get('id', 'CONTROL-1')
                description = self.control_data.get('description', '')
                frequency = self.control_data.get('frequency')
                control_type = self.control_data.get('type')
                risk = self.control_data.get('risk')

                result = self.analyzer.analyze_control(
                    control_id, description, frequency, control_type, risk
                )
                results = [result]

            # Generate visualizations if requested
            if self.options.get('generate_visualizations', False):
                self.progress_signal.emit(90, "Generating visualizations...")
                output_dir = self.options.get('visualization_dir', 'visualizations')
                generate_core_visualizations(results, output_dir)

            self.progress_signal.emit(100, "Analysis complete")
            self.finished_signal.emit(results)

        except Exception as e:
            self.error_signal.emit(str(e))
            import traceback
            traceback.print_exc()


class ControlAnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = None
        self.results = None
        self.worker = None
        self.visualization_dir = None

        # Initialize UI components
        self.init_ui()

        # Initialize analyzer
        self.init_analyzer()

    def init_analyzer(self):
        """Initialize the control analyzer"""
        try:
            # Look for config file in the same directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, "control_analyzer_config.yaml")

            if os.path.exists(config_path):
                self.analyzer = EnhancedControlAnalyzer(config_path)
                self.status_bar.showMessage(f"Loaded configuration from {config_path}")
            else:
                self.analyzer = EnhancedControlAnalyzer()
                self.status_bar.showMessage("Using default configuration")
        except Exception as e:
            self.show_error(f"Error initializing analyzer: {str(e)}")
            self.analyzer = None

    def init_ui(self):
        """Initialize UI components"""
        # Main window setup
        self.setWindowTitle("Control Description Analyzer")
        self.setGeometry(100, 100, 1200, 800)

        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Main layout
        main_layout = QVBoxLayout(self.central_widget)

        # Create tabs
        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.North)

        # Add tabs
        self.single_control_tab = self.create_single_control_tab()
        self.excel_file_tab = self.create_excel_file_tab()
        self.results_tab = self.create_results_tab()
        self.visualization_tab = self.create_visualization_tab()

        self.tabs.addTab(self.single_control_tab, "Single Control")
        self.tabs.addTab(self.excel_file_tab, "Excel File")
        self.tabs.addTab(self.results_tab, "Results")
        self.tabs.addTab(self.visualization_tab, "Visualizations")

        main_layout.addWidget(self.tabs)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label = QLabel("Ready")

        progress_layout = QHBoxLayout()
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress_bar)

        main_layout.addLayout(progress_layout)

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")

    def create_single_control_tab(self):
        """Create the tab for analyzing a single control"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Form layout for control properties
        form_group = QGroupBox("Control Properties")
        form_layout = QFormLayout()

        # Control ID
        self.control_id_input = QLineEdit()
        self.control_id_input.setText("CONTROL-1")
        form_layout.addRow("Control ID:", self.control_id_input)

        # Control description
        self.control_desc_input = QTextEdit()
        self.control_desc_input.setPlaceholderText("Enter control description here...")
        self.control_desc_input.setMinimumHeight(200)
        form_layout.addRow("Description:", self.control_desc_input)

        # Optional properties
        self.control_freq_input = QLineEdit()
        form_layout.addRow("Frequency (optional):", self.control_freq_input)

        self.control_type_input = QComboBox()
        self.control_type_input.addItems(["", "Preventive", "Detective", "Corrective", "Manual", "Automated", "Mixed"])
        form_layout.addRow("Control Type (optional):", self.control_type_input)

        self.control_risk_input = QTextEdit()
        self.control_risk_input.setPlaceholderText("Enter associated risk description...")
        self.control_risk_input.setMaximumHeight(100)
        form_layout.addRow("Risk (optional):", self.control_risk_input)

        form_group.setLayout(form_layout)
        layout.addWidget(form_group)

        # Analyze button
        analyze_btn = QPushButton("Analyze Control")
        analyze_btn.setIcon(QIcon("icons/analyze.png"))
        analyze_btn.clicked.connect(self.analyze_single_control)
        layout.addWidget(analyze_btn)

        # Sample control buttons
        samples_group = QGroupBox("Sample Controls")
        samples_layout = QHBoxLayout()

        excellent_btn = QPushButton("Excellent Example")
        excellent_btn.clicked.connect(lambda: self.load_sample_control("excellent"))

        good_btn = QPushButton("Good Example")
        good_btn.clicked.connect(lambda: self.load_sample_control("good"))

        poor_btn = QPushButton("Poor Example")
        poor_btn.clicked.connect(lambda: self.load_sample_control("poor"))

        samples_layout.addWidget(excellent_btn)
        samples_layout.addWidget(good_btn)
        samples_layout.addWidget(poor_btn)

        samples_group.setLayout(samples_layout)
        layout.addWidget(samples_group)

        layout.addStretch()
        return tab

    def create_excel_file_tab(self):
        """Create the tab for analyzing an Excel file"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # File selection
        file_group = QGroupBox("Excel File")
        file_layout = QHBoxLayout()

        self.excel_file_path = QLineEdit()
        self.excel_file_path.setReadOnly(True)
        self.excel_file_path.setPlaceholderText("Select an Excel file...")

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_excel_file)

        file_layout.addWidget(self.excel_file_path)
        file_layout.addWidget(browse_btn)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Column mapping
        columns_group = QGroupBox("Column Mapping")
        columns_layout = QFormLayout()

        self.id_column_input = QLineEdit("Control_ID")
        columns_layout.addRow("Control ID Column:", self.id_column_input)

        self.desc_column_input = QLineEdit("Control_Description")
        columns_layout.addRow("Description Column:", self.desc_column_input)

        self.freq_column_input = QLineEdit("Frequency")
        columns_layout.addRow("Frequency Column (optional):", self.freq_column_input)

        self.type_column_input = QLineEdit("Control_Type")
        columns_layout.addRow("Control Type Column (optional):", self.type_column_input)

        self.risk_column_input = QLineEdit("Risk_Description")
        columns_layout.addRow("Risk Column (optional):", self.risk_column_input)

        columns_group.setLayout(columns_layout)
        layout.addWidget(columns_group)

        # Options
        options_group = QGroupBox("Analysis Options")
        options_layout = QFormLayout()

        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(10, 1000)
        self.batch_size_input.setValue(100)
        self.batch_size_input.setSingleStep(10)
        options_layout.addRow("Batch Size:", self.batch_size_input)

        self.use_enhanced_checkbox = QCheckBox()
        self.use_enhanced_checkbox.setChecked(True)
        options_layout.addRow("Use Enhanced Detection:", self.use_enhanced_checkbox)

        self.gen_vis_checkbox = QCheckBox()
        self.gen_vis_checkbox.setChecked(True)
        options_layout.addRow("Generate Visualizations:", self.gen_vis_checkbox)

        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Analyze button
        analyze_btn = QPushButton("Analyze Excel File")
        analyze_btn.setIcon(QIcon("icons/analyze.png"))
        analyze_btn.clicked.connect(self.analyze_excel_file)
        layout.addWidget(analyze_btn)

        layout.addStretch()
        return tab

    def create_results_tab(self):
        """Create the tab for showing analysis results"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels([
            "Control ID", "Score", "Category", "WHO", "WHEN", "WHAT", "WHY", "ESCALATION"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.cellClicked.connect(self.show_result_details)

        layout.addWidget(self.results_table)

        # Details section
        details_group = QGroupBox("Control Details")
        details_layout = QVBoxLayout()

        self.control_detail_text = QTextEdit()
        self.control_detail_text.setReadOnly(True)
        details_layout.addWidget(self.control_detail_text)

        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        # Export button
        export_btn = QPushButton("Export Results to Excel")
        export_btn.clicked.connect(self.export_results)
        layout.addWidget(export_btn)

        return tab

    def create_visualization_tab(self):
        """Create the tab for visualizations"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Visualizations list
        vis_group = QGroupBox("Available Visualizations")
        vis_layout = QVBoxLayout()

        self.vis_list = QTableWidget()
        self.vis_list.setColumnCount(2)
        self.vis_list.setHorizontalHeaderLabels(["Visualization", "Description"])
        self.vis_list.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.vis_list.setSelectionBehavior(QTableWidget.SelectRows)

        # Add default visualizations
        self.vis_list.setRowCount(4)

        self.vis_list.setItem(0, 0, QTableWidgetItem("Score Distribution"))
        self.vis_list.setItem(0, 1, QTableWidgetItem("Distribution of control scores by category"))

        self.vis_list.setItem(1, 0, QTableWidgetItem("Element Radar"))
        self.vis_list.setItem(1, 1, QTableWidgetItem("Average element scores by category"))

        self.vis_list.setItem(2, 0, QTableWidgetItem("Missing Elements"))
        self.vis_list.setItem(2, 1, QTableWidgetItem("Frequency of missing elements"))

        self.vis_list.setItem(3, 0, QTableWidgetItem("Vague Terms"))
        self.vis_list.setItem(3, 1, QTableWidgetItem("Frequency of vague terms used"))

        vis_layout.addWidget(self.vis_list)

        # Open visualization button
        open_vis_btn = QPushButton("Open Selected Visualization")
        open_vis_btn.clicked.connect(self.open_visualization)
        vis_layout.addWidget(open_vis_btn)

        vis_group.setLayout(vis_layout)
        layout.addWidget(vis_group)

        # Dashboard button
        dashboard_btn = QPushButton("Open Visualization Dashboard")
        dashboard_btn.clicked.connect(self.open_dashboard)
        layout.addWidget(dashboard_btn)

        return tab

    def browse_excel_file(self):
        """Open file dialog to select Excel file"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Excel File", "", "Excel Files (*.xlsx *.xls *.xlsm);;All Files (*)", options=options
        )

        if file_path:
            self.excel_file_path.setText(file_path)

            # Try to read column names from the file
            try:
                df = pd.read_excel(file_path, nrows=0)
                columns = df.columns.tolist()

                # Set column inputs if we can find matching columns
                id_cols = [col for col in columns if "id" in col.lower()]
                if id_cols:
                    self.id_column_input.setText(id_cols[0])

                desc_cols = [col for col in columns if "desc" in col.lower()]
                if desc_cols:
                    self.desc_column_input.setText(desc_cols[0])

                freq_cols = [col for col in columns if "freq" in col.lower()]
                if freq_cols:
                    self.freq_column_input.setText(freq_cols[0])

                type_cols = [col for col in columns if "type" in col.lower()]
                if type_cols:
                    self.type_column_input.setText(type_cols[0])

                risk_cols = [col for col in columns if "risk" in col.lower()]
                if risk_cols:
                    self.risk_column_input.setText(risk_cols[0])

                self.status_bar.showMessage(f"Loaded Excel file with {len(columns)} columns")
            except Exception as e:
                self.status_bar.showMessage(f"Error reading Excel file: {str(e)}")

    def analyze_single_control(self):
        """Analyze a single control description"""

        # Reset analysis state
        self.reset_analysis_state()

        if not self.analyzer:
            self.show_error("Analyzer not initialized")
            return

        control_id = self.control_id_input.text().strip()
        description = self.control_desc_input.toPlainText().strip()

        if not description:
            self.show_error("Please enter a control description")
            return

        # Get optional properties
        frequency = self.control_freq_input.text().strip()
        control_type = self.control_type_input.currentText()
        risk = self.control_risk_input.toPlainText().strip()

        # Create control data
        control_data = {
            'id': control_id,
            'description': description,
            'frequency': frequency if frequency else None,
            'type': control_type if control_type else None,
            'risk': risk if risk else None
        }

        # Create visualization directory
        vis_dir = self.create_visualization_dir()

        # Analysis options
        options = {
            'generate_visualizations': True,
            'visualization_dir': vis_dir
        }

        # Start worker thread
        self.start_analysis(None, control_data, options)

    def reset_analysis_state(self):
        """Reset the analysis state before processing a new control"""
        # Clear previous results
        self.results = None

        # Reset UI elements
        self.progress_bar.setValue(0)
        self.progress_label.setText("Ready")

        # Clear the results table
        self.results_table.setRowCount(0)

        # Clear the detail text
        self.control_detail_text.clear()

        # Reset status
        self.status_bar.showMessage("Ready")

    def analyze_excel_file(self):
        """Analyze controls from an Excel file"""

        # Reset analysis state
        self.reset_analysis_state()

        if not self.analyzer:
            self.show_error("Analyzer not initialized")
            return

        file_path = self.excel_file_path.text().strip()
        if not file_path:
            self.show_error("Please select an Excel file")
            return

        # Get column mappings
        id_column = self.id_column_input.text().strip()
        desc_column = self.desc_column_input.text().strip()

        if not id_column or not desc_column:
            self.show_error("Control ID and Description columns are required")
            return

        # Get optional columns
        freq_column = self.freq_column_input.text().strip()
        type_column = self.type_column_input.text().strip()
        risk_column = self.risk_column_input.text().strip()

        # Get options
        batch_size = self.batch_size_input.value()
        use_enhanced = self.use_enhanced_checkbox.isChecked()
        generate_vis = self.gen_vis_checkbox.isChecked()

        # Update analyzer settings
        self.analyzer.use_enhanced_detection = use_enhanced

        # Create visualization directory
        vis_dir = self.create_visualization_dir()

        # Analysis options
        options = {
            'id_column': id_column,
            'desc_column': desc_column,
            'frequency_column': freq_column if freq_column else None,
            'type_column': type_column if type_column else None,
            'risk_column': risk_column if risk_column else None,
            'batch_size': batch_size,
            'generate_visualizations': generate_vis,
            'visualization_dir': vis_dir
        }

        # Start worker thread
        self.start_analysis(file_path, None, options)

    def start_analysis(self, file_path, control_data, options):
        """Start the analysis worker thread"""
        # Check if analysis is already running
        if self.worker and self.worker.isRunning():
            self.show_error("Analysis is already running")
            return

        # Create and start worker
        self.worker = AnalyzerWorker(self.analyzer, file_path, control_data, options)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.analysis_finished)
        self.worker.error_signal.connect(self.analysis_error)

        # Reset progress
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting analysis...")

        # Start analysis
        self.worker.start()

        # Update UI state
        self.status_bar.showMessage("Analysis in progress...")

    def update_progress(self, value, message):
        """Update the progress bar"""
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def analysis_finished(self, results):
        """Handle analysis completion"""
        self.results = results
        self.status_bar.showMessage(f"Analysis complete: {len(results)} controls analyzed")

        # Update results table
        self.update_results_table()

        # Update visualizations tab
        self.update_visualizations_tab()

        # Switch to results tab
        self.tabs.setCurrentIndex(2)

    def analysis_error(self, error_message):
        """Handle analysis error"""
        self.show_error(f"Analysis error: {error_message}")
        self.progress_bar.setValue(0)
        self.progress_label.setText("Analysis failed")
        self.status_bar.showMessage("Analysis failed")

    def update_results_table(self):
        """Update the results table with analysis results"""
        if not self.results:
            return

        # Clear existing rows
        self.results_table.setRowCount(0)

        # Add results
        for i, result in enumerate(self.results):
            self.results_table.insertRow(i)

            # Set values
            self.results_table.setItem(i, 0, QTableWidgetItem(str(result.get("control_id", ""))))

            score_item = QTableWidgetItem(f"{result.get('total_score', 0):.1f}")
            self.results_table.setItem(i, 1, score_item)

            category = result.get("category", "Unknown")
            category_item = QTableWidgetItem(category)

            # Set category cell color
            if category == "Excellent":
                category_item.setBackground(Qt.green)
            elif category == "Good":
                category_item.setBackground(Qt.yellow)
            else:
                category_item.setBackground(Qt.red)

            self.results_table.setItem(i, 2, category_item)

            # Element scores
            weighted_scores = result.get("weighted_scores", {})
            self.results_table.setItem(i, 3, QTableWidgetItem(f"{weighted_scores.get('WHO', 0):.1f}"))
            self.results_table.setItem(i, 4, QTableWidgetItem(f"{weighted_scores.get('WHEN', 0):.1f}"))
            self.results_table.setItem(i, 5, QTableWidgetItem(f"{weighted_scores.get('WHAT', 0):.1f}"))
            self.results_table.setItem(i, 6, QTableWidgetItem(f"{weighted_scores.get('WHY', 0):.1f}"))
            self.results_table.setItem(i, 7, QTableWidgetItem(f"{weighted_scores.get('ESCALATION', 0):.1f}"))

    def show_result_details(self, row, column):
        """Show details for the selected control"""
        if not self.results or row >= len(self.results):
            return

        result = self.results[row]

        # Format detailed text
        detail_text = f"<h2>Control {result.get('control_id', '')}</h2>"
        detail_text += f"<p><b>Score:</b> {result.get('total_score', 0):.1f} - {result.get('category', 'Unknown')}</p>"
        detail_text += f"<p><b>Description:</b> {result.get('description', '')}</p>"

        # Missing elements
        missing = result.get("missing_elements", [])
        if missing:
            detail_text += f"<p><b>Missing Elements:</b> {', '.join(missing)}</p>"

        # Vague terms
        vague_terms = result.get("vague_terms_found", [])
        if vague_terms:
            detail_text += f"<p><b>Vague Terms:</b> {', '.join(vague_terms)}</p>"

        # Element details
        detail_text += "<h3>Element Scores</h3>"
        weighted_scores = result.get("weighted_scores", {})
        matched_keywords = result.get("matched_keywords", {})

        for element in ["WHO", "WHEN", "WHAT", "WHY", "ESCALATION"]:
            score = weighted_scores.get(element, 0)
            keywords = matched_keywords.get(element, [])

            detail_text += f"<p><b>{element}:</b> {score:.1f}</p>"
            if keywords:
                detail_text += f"<p>Keywords: {', '.join(keywords)}</p>"

            # Add enhancement feedback if available
            feedback = result.get("enhancement_feedback", {}).get(element)
            if feedback:
                if isinstance(feedback, list):
                    feedback_text = "<br>- ".join(feedback)
                    detail_text += f"<p>Suggestions:<br>- {feedback_text}</p>"
                else:
                    detail_text += f"<p>Feedback: {feedback}</p>"

        # Set the detail text
        self.control_detail_text.setHtml(detail_text)

    def update_visualizations_tab(self):
        """Update visualizations tab with available visualizations"""
        # Check if we have a visualization directory
        if not self.visualization_dir or not os.path.exists(self.visualization_dir):
            return

        # Check for visualization files
        vis_files = []
        for file in os.listdir(self.visualization_dir):
            if file.endswith(".html"):
                vis_files.append(file)

        # Update vis_list if we have files
        if vis_files:
            self.vis_list.setRowCount(len(vis_files))

            for i, file in enumerate(vis_files):
                name = file.replace(".html", "").replace("_", " ").title()
                self.vis_list.setItem(i, 0, QTableWidgetItem(name))

                # Set description based on file name
                description = "Interactive visualization"
                if "score" in file.lower():
                    description = "Distribution of control scores by category"
                elif "element" in file.lower():
                    description = "Average element scores by category"
                elif "missing" in file.lower():
                    description = "Frequency of missing elements"
                elif "vague" in file.lower():
                    description = "Frequency of vague terms used"
                elif "leader" in file.lower():
                    description = "Scores by audit leader"

                self.vis_list.setItem(i, 1, QTableWidgetItem(description))

    def open_visualization(self):
        """Open the selected visualization in the default browser"""
        # Check if we have a visualization directory
        if not self.visualization_dir or not os.path.exists(self.visualization_dir):
            self.show_error("No visualizations available")
            return

        # Get selected visualization
        selected_rows = self.vis_list.selectedIndexes()
        if not selected_rows:
            self.show_error("Please select a visualization")
            return

        row = selected_rows[0].row()
        vis_name = self.vis_list.item(row, 0).text().lower().replace(" ", "_") + ".html"

        # Check if file exists
        vis_path = os.path.join(self.visualization_dir, vis_name)
        if not os.path.exists(vis_path):
            # Try to find a matching file
            for file in os.listdir(self.visualization_dir):
                if file.endswith(".html") and vis_name.replace(".html", "") in file.lower():
                    vis_path = os.path.join(self.visualization_dir, file)
                    break
            else:
                self.show_error(f"Visualization file not found: {vis_name}")
                return

        # Open in browser
        try:
            webbrowser.open(f"file://{os.path.abspath(vis_path)}")
            self.status_bar.showMessage(f"Opened visualization: {vis_name}")
        except Exception as e:
            self.show_error(f"Error opening visualization: {str(e)}")

    def open_dashboard(self):
        """Open the visualization dashboard in the default browser"""
        # Check if we have a visualization directory
        if not self.visualization_dir or not os.path.exists(self.visualization_dir):
            self.show_error("No visualizations available")
            return

        # Look for dashboard file
        dashboard_path = os.path.join(self.visualization_dir, "dashboard.html")

        # If it doesn't exist, try to open the score distribution visualization
        if not os.path.exists(dashboard_path):
            for file in os.listdir(self.visualization_dir):
                if file.endswith(".html"):
                    dashboard_path = os.path.join(self.visualization_dir, file)
                    break
            else:
                self.show_error("No visualizations found")
                return

        # Open in browser
        try:
            webbrowser.open(f"file://{os.path.abspath(dashboard_path)}")
            self.status_bar.showMessage(f"Opened visualization dashboard")
        except Exception as e:
            self.show_error(f"Error opening dashboard: {str(e)}")

    def create_visualization_dir(self):
        """Create a directory for visualizations"""
        # Create a temporary directory for visualizations
        if not self.visualization_dir:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            vis_dir = os.path.join(script_dir, "visualizations")

            # Create directory if it doesn't exist
            os.makedirs(vis_dir, exist_ok=True)
            self.visualization_dir = vis_dir

        return self.visualization_dir

    def export_results(self):
        """Export results to an Excel file"""
        if not self.results:
            self.show_error("No results to export")
            return

        # Open file dialog
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "Excel Files (*.xlsx);;All Files (*)", options=options
        )

        if not file_path:
            return

        # Add extension if missing
        if not file_path.lower().endswith('.xlsx'):
            file_path += '.xlsx'

        try:
            # Convert results to DataFrame
            basic_results = []
            for r in self.results:
                result_dict = {
                    "Control ID": r.get("control_id", ""),
                    "Description": r.get("description", ""),
                    "Total Score": r.get("total_score", 0),
                    "Category": r.get("category", ""),
                    "Missing Elements": ", ".join(r.get("missing_elements", [])) if r.get(
                        "missing_elements") else "None",
                    "Vague Terms": ", ".join(r.get("vague_terms_found", [])) if r.get("vague_terms_found") else "None",
                    "WHO Score": r.get("weighted_scores", {}).get("WHO", 0),
                    "WHEN Score": r.get("weighted_scores", {}).get("WHEN", 0),
                    "WHAT Score": r.get("weighted_scores", {}).get("WHAT", 0),
                    "WHY Score": r.get("weighted_scores", {}).get("WHY", 0),
                    "ESCALATION Score": r.get("weighted_scores", {}).get("ESCALATION", 0),
                }
                basic_results.append(result_dict)

            df_results = pd.DataFrame(basic_results)

            # Create a writer for Excel
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Write main results
                df_results.to_excel(writer, sheet_name='Analysis Results', index=False)

                # Write keywords sheet
                keyword_results = []
                for r in self.results:
                    result_dict = {
                        "Control ID": r.get("control_id", ""),
                        "WHO Keywords": ", ".join(r.get("matched_keywords", {}).get("WHO", [])) if r.get(
                            "matched_keywords", {}).get("WHO") else "None",
                        "WHEN Keywords": ", ".join(r.get("matched_keywords", {}).get("WHEN", [])) if r.get(
                            "matched_keywords", {}).get("WHEN") else "None",
                        "WHAT Keywords": ", ".join(r.get("matched_keywords", {}).get("WHAT", [])) if r.get(
                            "matched_keywords", {}).get("WHAT") else "None",
                        "WHY Keywords": ", ".join(r.get("matched_keywords", {}).get("WHY", [])) if r.get(
                            "matched_keywords", {}).get("WHY") else "None",
                        "ESCALATION Keywords": ", ".join(r.get("matched_keywords", {}).get("ESCALATION", [])) if r.get(
                            "matched_keywords", {}).get("ESCALATION") else "None"
                    }
                    keyword_results.append(result_dict)

                df_keywords = pd.DataFrame(keyword_results)
                df_keywords.to_excel(writer, sheet_name='Keyword Matches', index=False)

                # Write feedback sheet
                feedback_results = []
                for r in self.results:
                    result_dict = {"Control ID": r.get("control_id", "")}

                    # Format each element's feedback
                    for element in ["WHO", "WHEN", "WHAT", "WHY", "ESCALATION"]:
                        feedback = r.get("enhancement_feedback", {}).get(element)

                        if isinstance(feedback, list) and feedback:
                            result_dict[f"{element} Feedback"] = "; ".join(feedback)
                        elif isinstance(feedback, str) and feedback:
                            result_dict[f"{element} Feedback"] = feedback
                        else:
                            result_dict[f"{element} Feedback"] = "None"

                    feedback_results.append(result_dict)

                df_feedback = pd.DataFrame(feedback_results)
                df_feedback.to_excel(writer, sheet_name='Enhancement Feedback', index=False)

            self.status_bar.showMessage(f"Results exported to {file_path}")

        except Exception as e:
            self.show_error(f"Error exporting results: {str(e)}")

    def load_sample_control(self, quality):
        """Load a sample control description"""
        if quality == "excellent":
            self.control_id_input.setText("SAMPLE-EXCELLENT")
            self.control_desc_input.setText(
                "The Accounting Manager reviews the monthly reconciliation between the subledger and general ledger by the 5th business day of the following month. "
                "The reviewer examines supporting documentation, verifies that all reconciling items have been properly identified and resolved, and ensures "
                "compliance with accounting policies. The review is evidenced by electronic sign-off in the financial system. Any discrepancies exceeding $10,000 "
                "are escalated to the Controller and documented in the issue tracking system. The reconciliation and review documentation are stored in the "
                "Finance SharePoint site and retained according to the document retention policy."
            )
            self.control_freq_input.setText("Monthly")
            self.control_type_input.setCurrentText("Detective")
            self.control_risk_input.setText(
                "Risk of financial misstatement due to errors or discrepancies between subledger and general ledger.")

        elif quality == "good":
            self.control_id_input.setText("SAMPLE-GOOD")
            self.control_desc_input.setText(
                "The Accounting Supervisor reviews the monthly journal entries prior to posting to ensure accuracy and completeness. "
                "The reviewer checks supporting documentation and approves entries by signing the journal entry form. "
                "Any errors are returned to the preparer for correction."
            )
            self.control_freq_input.setText("Monthly")
            self.control_type_input.setCurrentText("Preventive")
            self.control_risk_input.setText("Risk of incorrect journal entries being posted to the general ledger.")

        elif quality == "poor":
            self.control_id_input.setText("SAMPLE-POOR")
            self.control_desc_input.setText(
                "Management reviews financial statements periodically and addresses any issues as appropriate."
            )
            self.control_freq_input.setText("Periodically")
            self.control_type_input.setCurrentText("")
            self.control_risk_input.setText("Financial misstatement risk.")

    def show_error(self, message):
        """Show an error message dialog"""
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle("Error")
        error_box.setText(message)
        error_box.exec_()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    window = ControlAnalyzerGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()