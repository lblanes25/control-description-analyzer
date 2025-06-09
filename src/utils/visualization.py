import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from collections import Counter
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
import html

# Plotly's update mechanism allows dynamic filtering without recreating entire charts

# Module-level constants
CATEGORY_COLORS = {
    "Meets Expectations": "#28a745",
    "Requires Attention": "#ffc107",
    "Needs Improvement": "#dc3545"
}

CORE_ELEMENTS = ["WHO", "WHAT", "WHEN"]
# Core scoring elements (WHO/WHAT/WHEN) comprise 100% of the base score

# JavaScript template strings as module-level constants
DASHBOARD_JAVASCRIPT_TEMPLATE = """
// JavaScript functions handle real-time filtering without server round-trips
// Store complete dataset
const fullDataset = {dashboard_data_json};
let currentData = fullDataset.records;

  // Initialize dashboard
  function initializeDashboard() {{
      populateAuditLeaderFilter();
      updateDashboard();
  }}

  // Populate audit leader dropdown
  function populateAuditLeaderFilter() {{
      const select = document.getElementById('auditLeaderFilter');
      fullDataset.auditLeaders.sort().forEach(leader => {{
          const option = document.createElement('option');
          option.value = leader;
          option.textContent = leader;
          select.appendChild(option);
      }});
  }}

  // Filter data by selected audit leader
  function filterByAuditLeader() {{
      const selectedLeader = document.getElementById('auditLeaderFilter').value;

      if (selectedLeader === 'all') {{
          currentData = fullDataset.records;
      }} else {{
          currentData = fullDataset.records.filter(record =>
              record['Audit Leader'] === selectedLeader
          );
      }}

      updateDashboard();
  }}

  // Update entire dashboard with filtered data
  function updateDashboard() {{
      const container = document.querySelector('.container');
      container.classList.add('loading');

      setTimeout(() => {{
          updateMetrics();
          updateCharts();
          container.classList.remove('loading');
      }}, 100);
  }}

  // Update executive metrics in 2x2 grid
  function updateMetrics() {{
      const metricsGrid = document.getElementById('metricsGrid');

      if (currentData.length === 0) {{
          metricsGrid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; color: #7f8c8d;">No data available for selected filter</div>';
          return;
      }}

      const needsImprovementCount = currentData.filter(record => record.Category === 'Needs Improvement').length;
      const needsImprovementPercentage = currentData.length > 0 ? ((needsImprovementCount / currentData.length) * 100).toFixed(0) : 0;
      const quickWinCount = currentData.filter(record => record['Missing Elements Count'] === 1).length;
      const totalControls = currentData.length;

      metricsGrid.innerHTML = `
          <div class="metric-card needs-improvement-percent">
              <div class="metric-value">${{needsImprovementPercentage}}%</div>
              <div class="metric-label">% Controls Rated "Needs Improvement"</div>
          </div>
          <div class="metric-card needs-improvement-count">
              <div class="metric-value">${{needsImprovementCount}}</div>
              <div class="metric-label"># Controls Rated "Needs Improvement"</div>
          </div>
          <div class="metric-card quick-win">
              <div class="metric-value">${{quickWinCount}}</div>
              <div class="metric-label"># Quick Win Controls</div>
              <div class="metric-sublabel">(1 element missing)</div>
          </div>
          <div class="metric-card total">
              <div class="metric-value">${{totalControls}}</div>
              <div class="metric-label">Total Controls Reviewed</div>
          </div>
      `;
  }}


  // Update all charts
  function updateCharts() {{
      updateCompositionChart();
      updatePerformanceChart();
      updateGapAnalysisChart();
  }}

  // Update portfolio composition chart
  function updateCompositionChart() {{
      const categoryCounts = {{
          'Meets Expectations': 0,
          'Requires Attention': 0,
          'Needs Improvement': 0
      }};

      currentData.forEach(record => {{
          if (categoryCounts[record.Category] !== undefined) {{
              categoryCounts[record.Category]++;
          }}
      }});

      const data = [{{
          values: Object.values(categoryCounts),
          labels: Object.keys(categoryCounts),
          type: 'pie',
          marker: {{ colors: ['#27ae60', '#f39c12', '#e74c3c'] }},
          textinfo: 'label+percent',
          textposition: 'auto'
      }}];

      const layout = {{
          margin: {{t: 20, l: 20, r: 20, b: 20}},
          showlegend: false
      }};

      Plotly.react('compositionChart', data, layout);
  }}

  // Update performance heatmap
  function updatePerformanceChart() {{
      const entityScores = {{}};
      currentData.forEach(record => {{
          const entity = record['Audit Entity'];
          if (!entityScores[entity]) {{
              entityScores[entity] = [];
          }}
          entityScores[entity].push(record['Total Score']);
      }});

      const entities = Object.keys(entityScores);
      const avgScores = entities.map(entity =>
          entityScores[entity].reduce((a, b) => a + b, 0) / entityScores[entity].length
      );

      const data = [{{
          x: entities,
          y: avgScores,
          type: 'bar',
          marker: {{
              color: avgScores,
              colorscale: [[0, '#e74c3c'], [0.5, '#f39c12'], [1, '#27ae60']],
              cmin: 0,
              cmax: 100
          }}
      }}];

      const layout = {{
          xaxis: {{title: 'Audit Entity'}},
          yaxis: {{title: 'Average Score', range: [0, 100]}},
          margin: {{t: 20, l: 60, r: 20, b: 120}}
      }};

      Plotly.react('performanceChart', data, layout);
  }}

  // Update gap analysis chart
  function updateGapAnalysisChart() {{
      const elements = ['WHO', 'WHAT', 'WHEN'];
      const missingCounts = elements.map(element => {{
          let count = 0;
          currentData.forEach(record => {{
              if (record['Missing Elements'] !== 'None' &&
                  record['Missing Elements'].includes(element)) {{
                  count++;
              }}
          }});
          return count;
      }});

      const data = [{{
          x: elements,
          y: missingCounts,
          type: 'bar',
          marker: {{ color: missingCounts, colorscale: 'Reds' }}
      }}];

      const layout = {{
          xaxis: {{title: 'Control Element'}},
          yaxis: {{title: 'Number of Controls Missing Element'}},
          margin: {{t: 20, l: 60, r: 20, b: 60}}
      }};

      Plotly.react('gapAnalysisChart', data, layout);
  }}

  window.onload = initializeDashboard;
  """

# CSS styles for dashboard
DASHBOARD_CSS = """
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f8f9fa;
}

  .header {
      background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
      color: white;
      padding: 20px 40px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
  }

  .header h1 {
      margin: 0;
      font-size: 28px;
      font-weight: 300;
  }

  .header .subtitle {
      margin: 5px 0 0 0;
      font-size: 14px;
      opacity: 0.9;
  }

  .container {
      max-width: 1400px;
      margin: 0 auto;
      padding: 30px;
  }

  .filter-section {
      background: white;
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 30px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.05);
      display: flex;
      justify-content: space-between;
      align-items: center;
  }

  .filter-group {
      display: flex;
      align-items: center;
      gap: 15px;
  }

  .filter-group label {
      font-weight: 600;
      color: #2c3e50;
      margin-right: 8px;
  }

  select {
      padding: 8px 12px;
      border: 2px solid #e9ecef;
      border-radius: 6px;
      font-size: 14px;
      min-width: 180px;
      transition: border-color 0.2s;
  }

  select:focus {
      outline: none;
      border-color: #3498db;
  }

  .metrics-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 20px;
      margin-bottom: 30px;
  }

  .metric-card {
      background: white;
      border-radius: 8px;
      padding: 20px;
      text-align: center;
      box-shadow: 0 2px 10px rgba(0,0,0,0.05);
      transition: transform 0.2s, opacity 0.2s;
      border-top: 4px solid #3498db;
  }

  .metric-card.needs-improvement-percent {
      border-top-color: #e74c3c;
      background: #fee;
  }

  .metric-card.needs-improvement-count {
      border-top-color: #e74c3c;
      background: #fee;
  }

  .metric-card.quick-win {
      border-top-color: #f39c12;
      background: #fff9e6;
  }

  .metric-card.total {
      border-top-color: #3498db;
      background: white;
  }

  .metric-sublabel {
      font-size: 12px;
      color: #7f8c8d;
      margin-top: 5px;
      font-style: italic;
  }

  .metric-value {
      font-size: 36px;
      font-weight: 700;
      color: #2c3e50;
      margin: 10px 0;
      transition: opacity 0.2s;
  }

  .metric-label {
      font-size: 14px;
      color: #7f8c8d;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.5px;
  }


  .charts-section {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 30px;
      margin-bottom: 30px;
  }

  .chart-container {
      background: white;
      border-radius: 8px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.05);
      overflow: hidden;
  }

  .chart-header {
      padding: 15px 20px;
      background: #f8f9fa;
      border-bottom: 1px solid #e9ecef;
      font-weight: 600;
      color: #2c3e50;
  }

  .chart {
      height: 400px;
  }

  .full-width {
      grid-column: 1 / -1;
  }

  .footer {
      text-align: center;
      padding: 20px;
      color: #7f8c8d;
      font-size: 12px;
      border-top: 1px solid #e9ecef;
      margin-top: 30px;
  }

  .loading {
      opacity: 0.6;
      pointer-events: none;
  }

  @media (max-width: 768px) {
      .container {
          padding: 15px;
      }

      .filter-section {
          flex-direction: column;
          gap: 15px;
      }

      .charts-section {
          grid-template-columns: 1fr;
      }

      .metrics-grid {
          grid-template-columns: 1fr;
      }
  }
  """


# Encapsulates chart configuration to reduce parameter lists
class ChartConfig:
    """Configuration class for chart settings and layout."""

    def __init__(self, title: str, x_title: str = "", y_title: str = "",
                 margin: Dict[str, int] = None, height: int = 400):
        self.title = title
        self.x_title = x_title
        self.y_title = y_title
        self.margin = margin or {"t": 10, "l": 60, "r": 10, "b": 60}
        self.height = height


class DashboardConfig:
    """Holds all dashboard-related configuration."""

    def __init__(self, title: str = "Senior Leadership Dashboard",
                 subtitle: str = "Control Description Quality Overview",
                 generation_date: str = None):
        self.title = title
        self.subtitle = subtitle
        self.generation_date = generation_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class DropdownMenuBuilder:
    """Handles all dropdown-related logic across different charts."""

    @staticmethod
    def create_histogram_button_data(filtered_data: pd.DataFrame, categories: List[str]) -> List[Any]:
        """Create histogram data for dropdown buttons."""
        button_data = []
        for category in categories:
            category_data = filtered_data[filtered_data["Category"] == category]
            button_data.append(go.Histogram(
                x=category_data["Total Score"],
                name=category,
                marker_color=CATEGORY_COLORS.get(category, "#999999")
            ))
        return button_data

    @staticmethod
    def create_bar_button_data(data: pd.DataFrame, x_col: str, y_col: str, colorscale: str = "Reds") -> go.Bar:
        """Create bar chart data for dropdown buttons."""
        return go.Bar(
            x=data[x_col],
            y=data[y_col],
            marker=dict(
                color=data[y_col],
                colorscale=colorscale
            )
        )

    def create_dropdown_buttons(self, data: pd.DataFrame, filter_field: str,
                              title_prefix: str, chart_type: str = "histogram") -> List[Dict]:
        """Generic dropdown button factory for various chart types."""
        dropdown_buttons = []
        unique_values = data[filter_field].unique().tolist()

        if "All" not in unique_values[0]:
            unique_values.insert(0, f"All {filter_field.split(' ')[-1]}s")

        # Add "All" button
        dropdown_buttons.append({
            "method": "update",
            "label": unique_values[0],
            "args": [{"visible": [True] * len(data["Category"].unique())}]
        })

        # Add filter buttons for each unique value
        for value in unique_values[1:]:
            filtered_data = data[data[filter_field] == value]

            if chart_type == "histogram":
                button_data = self.create_histogram_button_data(filtered_data, data["Category"].unique())
                button = {
                    "method": "update",
                    "label": value,
                    "args": [
                        {"data": button_data},
                        {"title": f"{title_prefix} - {value}"}
                    ]
                }
            else:
                # For other chart types, implement as needed
                button = {
                    "method": "update",
                    "label": value,
                    "args": [{"visible": [True]}]
                }

            dropdown_buttons.append(button)

        return dropdown_buttons


class VisualizationBuilder:
    """Main class encapsulating all visualization logic."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.dropdown_builder = DropdownMenuBuilder()
        os.makedirs(output_dir, exist_ok=True)

    def generate_core_visualizations(self, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate enhanced visualizations with dropdown filters for control description analysis results."""

        # Transform results to DataFrame for easier manipulation
        controls_dataframe = self.prepare_dataframe(results)

        # Create dropdown options from unique audit leaders and entities
        audit_leaders = controls_dataframe["Audit Leader"].unique().tolist()
        audit_leaders.insert(0, "All Leaders")

        audit_entities = controls_dataframe["Audit Entity"].unique().tolist()
        audit_entities.insert(0, "All Entities")

        # Generate all charts and combine file paths
        generated_file_paths = self.generate_all_charts(controls_dataframe, audit_leaders, audit_entities)

        return generated_file_paths

    def generate_all_charts(self, controls_dataframe: pd.DataFrame,
                          audit_leaders: List[str], audit_entities: List[str]) -> Dict[str, str]:
        """Separate chart generation logic from file management."""
        generated_file_paths = {}

        # Generate individual charts
        generated_file_paths.update(self.create_score_distribution_chart(controls_dataframe, audit_leaders,
                                                                        audit_entities))
        generated_file_paths.update(self.create_radar_chart(controls_dataframe, audit_leaders, audit_entities))
        generated_file_paths.update(self.create_missing_elements_chart(controls_dataframe, audit_leaders,
                                                                      audit_entities))
        generated_file_paths.update(self.create_vague_terms_chart(controls_dataframe, audit_leaders,
                                                                 audit_entities))
        generated_file_paths.update(self.create_audit_leader_breakdown(controls_dataframe))
        generated_file_paths.update(self.create_audit_entity_breakdown(controls_dataframe))
        generated_file_paths.update(self.create_worst_controls_table(controls_dataframe, audit_leaders,
                                                                    audit_entities))
        generated_file_paths.update(self.create_combined_dashboard(controls_dataframe, audit_leaders,
                                                                  audit_entities))
        generated_file_paths.update(self.create_leadership_dashboard(controls_dataframe, audit_leaders,
                                                                    audit_entities))

        return generated_file_paths

    def prepare_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Transform results to DataFrame for easier manipulation."""
        return pd.DataFrame([{
            "Control ID": result["control_id"],
            "Total Score": result["total_score"],
            "Category": result["category"],
            "Missing Elements Count": len(result["missing_elements"]),
            "WHO Score": result["weighted_scores"]["WHO"],
            "WHEN Score": result["weighted_scores"]["WHEN"],
            "WHAT Score": result["weighted_scores"]["WHAT"],
            # WHERE is conditional scoring, WHY/ESCALATION are feedback-only
            "WHERE Points": result["weighted_scores"].get("WHERE", 0),
            "WHY Score": result["weighted_scores"].get("WHY", 0),
            "ESCALATION Score": result["weighted_scores"].get("ESCALATION", 0),
            "Missing Elements": ", ".join(result["missing_elements"]) if result["missing_elements"] else "None",
            # Get Audit Leader - it's now consistently set in the result dictionary
            "Audit Leader": result.get("Audit Leader", "Unknown"),
            # Get Audit Entity - similarly to Audit Leader
            "Audit Entity": result.get("Audit Entity", "Unknown"),
            "vague_terms_found": result.get("vague_terms_found", [])
        } for result in results])

    def create_score_distribution_chart(self, controls_dataframe: pd.DataFrame,
                                     audit_leaders: List[str], audit_entities: List[str]) -> Dict[str, str]:
        """Create score distribution chart with audit leader and entity filters."""
        config = ChartConfig(
            title="Distribution of Control Description Scores",
            x_title="Score (0-100)",
            y_title="Number of Controls"
        )

        score_distribution_figure = px.histogram(
            controls_dataframe,
            x="Total Score",
            color="Category",
            nbins=20,
            title=config.title,
            labels={"Total Score": config.x_title, "count": config.y_title},
            color_discrete_map=CATEGORY_COLORS
        )

        # Create dropdown menu buttons using the builder
        audit_leader_dropdown_buttons = self.dropdown_builder.create_dropdown_buttons(
            controls_dataframe, "Audit Leader", "Distribution of Control Description Scores", "histogram"
        )

        audit_entity_dropdown_buttons = self.dropdown_builder.create_dropdown_buttons(
            controls_dataframe, "Audit Entity", "Distribution of Control Description Scores", "histogram"
        )

        # Update layout to include both dropdown menus
        score_distribution_figure.update_layout(
            updatemenus=[
                {
                    "buttons": audit_leader_dropdown_buttons,
                    "direction": "down",
                    "showactive": True,
                    "x": 1.0,
                    "xanchor": "right",
                    "y": 1.15,
                    "yanchor": "top"
                },
                {
                    "buttons": audit_entity_dropdown_buttons,
                    "direction": "down",
                    "showactive": True,
                    "x": 0.7,
                    "xanchor": "right",
                    "y": 1.15,
                    "yanchor": "top"
                }
            ],
            annotations=[
                {
                    "text": "Audit Leader:",
                    "showarrow": False,
                    "x": 1.0,
                    "y": 1.2,
                    "xref": "paper",
                    "yref": "paper",
                    "align": "right"
                },
                {
                    "text": "Audit Entity:",
                    "showarrow": False,
                    "x": 0.7,
                    "y": 1.2,
                    "xref": "paper",
                    "yref": "paper",
                    "align": "right"
                }
            ]
        )

        file_path = os.path.join(self.output_dir, "score_distribution.html")
        score_distribution_figure.write_html(file_path)
        return {"score_distribution": file_path}

    def create_radar_traces_for_filter(self, filtered_data: pd.DataFrame, filter_value: str) -> List[go.Scatterpolar]:
        """Eliminate duplicate trace generation logic for radar charts."""
        if len(filtered_data) == 0:
            return []

        category_avg = self.calculate_category_averages(filtered_data, [f"{e} Score" for e in CORE_ELEMENTS])
        traces = []

        for category in category_avg["Category"]:
            values = [category_avg.loc[category_avg["Category"] == category, f"{e} Score"].values[0]
                    for e in CORE_ELEMENTS]
            values.append(values[0])  # Close the loop
            traces.append(go.Scatterpolar(
                r=values,
                theta=CORE_ELEMENTS + [CORE_ELEMENTS[0]],
                fill='toself',
                name=f"{category} ({filter_value})"
            ))

        return traces

    def calculate_category_averages(self, dataframe: pd.DataFrame, score_columns: List[str]) -> pd.DataFrame:
        """Utility function for common operations like calculating category averages."""
        return dataframe.groupby("Category")[score_columns].mean().reset_index()

    def create_radar_chart(self, controls_dataframe: pd.DataFrame,
                         audit_leaders: List[str], audit_entities: List[str]) -> Dict[str, str]:
        """Create element radar chart with audit leader and entity filters."""

        # Create figure
        radar_figure = go.Figure()

        # Add traces for each category (across all leaders)
        category_avg = self.calculate_category_averages(controls_dataframe, [f"{e} Score" for e in CORE_ELEMENTS])
        for category in category_avg["Category"]:
            values = [category_avg.loc[category_avg["Category"] == category, f"{e} Score"].values[0] for e in CORE_ELEMENTS]
            values.append(values[0])  # Close the loop
            radar_figure.add_trace(go.Scatterpolar(
                r=values,
                theta=CORE_ELEMENTS + [CORE_ELEMENTS[0]],
                fill='toself',
                name=f"{category} (All Leaders)"
            ))

        # Create consolidated dropdown buttons for audit leaders and entities
        radar_audit_leader_buttons = self._create_radar_dropdown_buttons(
            controls_dataframe, audit_leaders, "Audit Leader"
        )

        radar_audit_entity_buttons = self._create_radar_dropdown_buttons(
            controls_dataframe, audit_entities, "Audit Entity"
        )

        # Update layout to include both dropdown menus
        radar_figure.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 30])),
            title="Average Element Scores by Category",
            updatemenus=[
                {
                    "buttons": radar_audit_leader_buttons,
                    "direction": "down",
                    "showactive": True,
                    "x": 1.0,
                    "xanchor": "right",
                    "y": 1.15,
                    "yanchor": "top"
                },
                {
                    "buttons": radar_audit_entity_buttons,
                    "direction": "down",
                    "showactive": True,
                    "x": 0.7,
                    "xanchor": "right",
                    "y": 1.15,
                    "yanchor": "top"
                }
            ],
            annotations=[
                {
                    "text": "Audit Leader:",
                    "showarrow": False,
                    "x": 1.0,
                    "y": 1.2,
                    "xref": "paper",
                    "yref": "paper",
                    "align": "right"
                },
                {
                    "text": "Audit Entity:",
                    "showarrow": False,
                    "x": 0.7,
                    "y": 1.2,
                    "xref": "paper",
                    "yref": "paper",
                    "align": "right"
                }
            ]
        )

        file_path = os.path.join(self.output_dir, "element_radar.html")
        radar_figure.write_html(file_path)
        return {"element_radar": file_path}

    def _create_radar_dropdown_buttons(self, controls_dataframe: pd.DataFrame,
                                      filter_values: List[str], filter_field: str) -> List[Dict]:
        """Consolidate radar chart dropdown button creation logic."""
        dropdown_buttons = []

        # Add "All" button
        dropdown_buttons.append({
            "method": "update",
            "label": f"All {filter_field.split(' ')[-1]}s",
            "args": [{"visible": [True] * len(controls_dataframe["Category"].unique())}]
        })

        # Add filter buttons for each value
        for filter_value in filter_values[1:]:  # Skip "All" option
            filtered_data = controls_dataframe[controls_dataframe[filter_field] == filter_value]
            traces = self.create_radar_traces_for_filter(filtered_data, filter_value)

            if traces:
                button = {
                    "method": "update",
                    "label": filter_value,
                    "args": [
                        {"data": traces},
                        {"title": f"Average Element Scores by Category - {filter_value}"}
                    ]
                }
                dropdown_buttons.append(button)

        return dropdown_buttons

    def count_missing_elements_by_category(self, dataframe: pd.DataFrame, elements_list: List[str]) -> pd.DataFrame:
        """Utility function to count missing elements for specific data."""
        missing_counts = {e: 0 for e in elements_list}
        for missing_elements_str in dataframe["Missing Elements"]:
            if missing_elements_str != "None":
                for element in missing_elements_str.split(", "):
                    if element in missing_counts:
                        missing_counts[element] += 1
        return pd.DataFrame({
            "Element": list(missing_counts.keys()),
            "Missing Count": list(missing_counts.values())
        })

    def create_missing_elements_chart(self, controls_dataframe: pd.DataFrame,
                                     audit_leaders: List[str], audit_entities: List[str]) -> Dict[str, str]:
        """Create missing elements chart with audit leader and entity filters."""

        # Count missing elements for all data
        missing_dataframe = self.count_missing_elements_by_category(controls_dataframe, CORE_ELEMENTS)

        # Create figure with dropdown menus for Audit Leader and Audit Entity
        missing_elements_figure = px.bar(
            missing_dataframe,
            x="Element",
            y="Missing Count",
            title="Frequency of Missing Elements",
            labels={"Element": "Control Element", "Missing Count": "Number of Controls Missing Element"},
            color="Missing Count",
            color_continuous_scale=px.colors.sequential.Reds
        )

        # Create dropdown menu buttons for each audit leader
        missing_audit_leader_buttons = self._create_missing_elements_dropdown_buttons(
            controls_dataframe, audit_leaders, "Audit Leader"
        )

        # Create dropdown menu buttons for each audit entity
        missing_audit_entity_buttons = self._create_missing_elements_dropdown_buttons(
            controls_dataframe, audit_entities, "Audit Entity"
        )

        # Update layout to include both dropdown menus
        missing_elements_figure.update_layout(
            updatemenus=[
                {
                    "buttons": missing_audit_leader_buttons,
                    "direction": "down",
                    "showactive": True,
                    "x": 1.0,
                    "xanchor": "right",
                    "y": 1.15,
                    "yanchor": "top"
                },
                {
                    "buttons": missing_audit_entity_buttons,
                    "direction": "down",
                    "showactive": True,
                    "x": 0.7,
                    "xanchor": "right",
                    "y": 1.15,
                    "yanchor": "top"
                }
            ],
            annotations=[
                {
                    "text": "Audit Leader:",
                    "showarrow": False,
                    "x": 1.0,
                    "y": 1.2,
                    "xref": "paper",
                    "yref": "paper",
                    "align": "right"
                },
                {
                    "text": "Audit Entity:",
                    "showarrow": False,
                    "x": 0.7,
                    "y": 1.2,
                    "xref": "paper",
                    "yref": "paper",
                    "align": "right"
                }
            ]
        )

        file_path = os.path.join(self.output_dir, "missing_elements.html")
        missing_elements_figure.write_html(file_path)
        return {"missing_elements": file_path}

    def _create_missing_elements_dropdown_buttons(self, controls_dataframe: pd.DataFrame,
                                                 filter_values: List[str], filter_field: str) -> List[Dict]:
        """Create dropdown buttons for missing elements charts."""
        dropdown_buttons = []

        # Add "All" button
        all_missing_df = self.count_missing_elements_by_category(controls_dataframe, CORE_ELEMENTS)
        dropdown_buttons.append({
            "method": "update",
            "label": f"All {filter_field.split(' ')[-1]}s",
            "args": [
                {"data": [self.dropdown_builder.create_bar_button_data(all_missing_df, "Element", "Missing Count", "Reds")]},
                {"title": "Frequency of Missing Elements"}
            ]
        })

        # Add filter buttons for each value
        for filter_value in filter_values[1:]:  # Skip "All" option
            filtered_data = controls_dataframe[controls_dataframe[filter_field] == filter_value]
            filtered_missing_df = self.count_missing_elements_by_category(filtered_data, CORE_ELEMENTS)

            button = {
                "method": "update",
                "label": filter_value,
                "args": [
                    {"data": [self.dropdown_builder.create_bar_button_data(filtered_missing_df, "Element", "Missing Count", "Reds")]},
                    {"title": f"Frequency of Missing Elements - {filter_value}"}
                ]
            }
            dropdown_buttons.append(button)

        return dropdown_buttons

    def count_vague_terms(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Count vague terms for specific data."""
        vague_terms = sum(dataframe["vague_terms_found"], [])
        term_counts = Counter(vague_terms)
        if term_counts:
            return pd.DataFrame(term_counts.items(), columns=["Term", "Count"]).sort_values("Count", ascending=False)
        return pd.DataFrame(columns=["Term", "Count"])

    def create_vague_terms_chart(self, controls_dataframe: pd.DataFrame,
                                audit_leaders: List[str], audit_entities: List[str]) -> Dict[str, str]:
        """Create vague term frequency chart with audit leader and entity filters."""

        # Count vague terms for all data
        vague_dataframe = self.count_vague_terms(controls_dataframe)

        if not vague_dataframe.empty:
            # Create figure with dropdown menus for Audit Leader and Audit Entity
            vague_terms_figure = px.bar(
                vague_dataframe,
                x="Term",
                y="Count",
                title="Frequency of Vague Terms",
                labels={"Term": "Vague Term", "Count": "Occurrences"},
                color="Count",
                color_continuous_scale=px.colors.sequential.Oranges
            )

            # Create dropdown menu buttons for each audit leader
            vague_audit_leader_buttons = self._create_vague_terms_dropdown_buttons(
                controls_dataframe, audit_leaders, "Audit Leader"
            )

            # Create dropdown menu buttons for each audit entity
            vague_audit_entity_buttons = self._create_vague_terms_dropdown_buttons(
                controls_dataframe, audit_entities, "Audit Entity"
            )

            # Update layout to include both dropdown menus
            vague_terms_figure.update_layout(
                updatemenus=[
                    {
                        "buttons": vague_audit_leader_buttons,
                        "direction": "down",
                        "showactive": True,
                        "x": 1.0,
                        "xanchor": "right",
                        "y": 1.15,
                        "yanchor": "top"
                    },
                    {
                        "buttons": vague_audit_entity_buttons,
                        "direction": "down",
                        "showactive": True,
                        "x": 0.7,
                        "xanchor": "right",
                        "y": 1.15,
                        "yanchor": "top"
                    }
                ],
                annotations=[
                    {
                        "text": "Audit Leader:",
                        "showarrow": False,
                        "x": 1.0,
                        "y": 1.2,
                        "xref": "paper",
                        "yref": "paper",
                        "align": "right"
                    },
                    {
                        "text": "Audit Entity:",
                        "showarrow": False,
                        "x": 0.7,
                        "y": 1.2,
                        "xref": "paper",
                        "yref": "paper",
                        "align": "right"
                    }
                ]
            )

            file_path = os.path.join(self.output_dir, "vague_terms.html")
            vague_terms_figure.write_html(file_path)
            return {"vague_terms": file_path}

        return {}

    def _create_vague_terms_dropdown_buttons(self, controls_dataframe: pd.DataFrame,
                                            filter_values: List[str], filter_field: str) -> List[Dict]:
        """Create dropdown buttons for vague terms charts."""
        dropdown_buttons = []

        # Add "All" button
        all_vague_df = self.count_vague_terms(controls_dataframe)
        dropdown_buttons.append({
            "method": "update",
            "label": f"All {filter_field.split(' ')[-1]}s",
            "args": [
                {"data": [self.dropdown_builder.create_bar_button_data(all_vague_df, "Term", "Count", "Oranges")]},
                {"title": "Frequency of Vague Terms"}
            ]
        })

        # Add filter buttons for each value
        for filter_value in filter_values[1:]:  # Skip "All" option
            filtered_data = controls_dataframe[controls_dataframe[filter_field] == filter_value]
            filtered_vague_df = self.count_vague_terms(filtered_data)

            if not filtered_vague_df.empty:
                button = {
                    "method": "update",
                    "label": filter_value,
                    "args": [
                        {"data": [self.dropdown_builder.create_bar_button_data(filtered_vague_df, "Term", "Count", "Oranges")]},
                        {"title": f"Frequency of Vague Terms - {filter_value}"}
                    ]
                }
                dropdown_buttons.append(button)

        return dropdown_buttons

    def create_audit_leader_breakdown(self, controls_dataframe: pd.DataFrame) -> Dict[str, str]:
        """Create audit leader breakdown charts."""
        generated_file_paths = {}

        if "Audit Leader" in controls_dataframe.columns:
            # Average score by audit leader
            audit_leader_avg = controls_dataframe.groupby("Audit Leader")["Total Score"].mean().reset_index().sort_values("Total Score", ascending=False)
            audit_leader_avg_figure = px.bar(
                audit_leader_avg,
                x="Audit Leader",
                y="Total Score",
                title="Average Control Score by Audit Leader",
                labels={"Total Score": "Avg Score"},
                color="Total Score",
                color_continuous_scale=px.colors.sequential.Blues
            )

            file_path = os.path.join(self.output_dir, "leader_avg_score.html")
            audit_leader_avg_figure.write_html(file_path)
            generated_file_paths["leader_avg_score"] = file_path

            # Create missing elements by leader chart (stacked bar)
            missing_data = []
            for _, row in controls_dataframe.iterrows():
                audit_leader = row["Audit Leader"]
                if row["Missing Elements"] != "None":
                    for elem in row["Missing Elements"].split(", "):
                        missing_data.append((audit_leader, elem))

            if missing_data:
                missing_dataframe = pd.DataFrame(missing_data, columns=["Audit Leader", "Element"])
                missing_counts = missing_dataframe.groupby(["Audit Leader", "Element"]).size().reset_index(name="Count")

                missing_stack_figure = px.bar(
                    missing_counts,
                    x="Audit Leader",
                    y="Count",
                    color="Element",
                    title="Missing Elements by Audit Leader",
                    barmode="stack"
                )

                file_path = os.path.join(self.output_dir, "leader_missing_elements.html")
                missing_stack_figure.write_html(file_path)
                generated_file_paths["leader_missing_elements"] = file_path

        return generated_file_paths

    def create_audit_entity_breakdown(self, controls_dataframe: pd.DataFrame) -> Dict[str, str]:
        """Create audit entity breakdown charts."""
        generated_file_paths = {}

        if "Audit Entity" in controls_dataframe.columns and len(controls_dataframe["Audit Entity"].unique()) > 1:
            # Average score by audit entity
            audit_entity_avg = controls_dataframe.groupby("Audit Entity")["Total Score"].mean().reset_index().sort_values("Total Score", ascending=False)
            audit_entity_avg_figure = px.bar(
                audit_entity_avg,
                x="Audit Entity",
                y="Total Score",
                title="Average Control Score by Audit Entity",
                labels={"Total Score": "Avg Score"},
                color="Total Score",
                color_continuous_scale=px.colors.sequential.Greens
            )

            file_path = os.path.join(self.output_dir, "entity_avg_score.html")
            audit_entity_avg_figure.write_html(file_path)
            generated_file_paths["entity_avg_score"] = file_path

            # Create missing elements by entity chart (stacked bar)
            missing_data = []
            for _, row in controls_dataframe.iterrows():
                audit_entity = row["Audit Entity"]
                if row["Missing Elements"] != "None":
                    for elem in row["Missing Elements"].split(", "):
                        missing_data.append((audit_entity, elem))

            if missing_data:
                missing_dataframe = pd.DataFrame(missing_data, columns=["Audit Entity", "Element"])
                missing_counts = missing_dataframe.groupby(["Audit Entity", "Element"]).size().reset_index(name="Count")

                missing_stack_figure = px.bar(
                    missing_counts,
                    x="Audit Entity",
                    y="Count",
                    color="Element",
                    title="Missing Elements by Audit Entity",
                    barmode="stack"
                )

                file_path = os.path.join(self.output_dir, "entity_missing_elements.html")
                missing_stack_figure.write_html(file_path)
                generated_file_paths["entity_missing_elements"] = file_path

        return generated_file_paths

    def create_worst_controls_table(self, controls_dataframe: pd.DataFrame,
                                   audit_leaders: List[str], audit_entities: List[str]) -> Dict[str, str]:
        """Create worst controls table with filtering."""
        # Controls missing 3+ elements
        worst_controls = controls_dataframe[controls_dataframe["Missing Elements Count"] >= 3]
        if not worst_controls.empty:
            worst_controls = worst_controls.sort_values("Total Score")

            # Use proper HTML template instead of string concatenation
            html_content = self.generate_worst_controls_html_template(worst_controls, audit_leaders, audit_entities)

            # Write HTML file
            file_path = os.path.join(self.output_dir, "worst_controls.html")
            with open(file_path, 'w') as f:
                f.write(html_content)

            return {"worst_controls": file_path}

        return {}

    def generate_worst_controls_html_template(self, worst_controls: pd.DataFrame,
                                             audit_leaders: List[str], audit_entities: List[str]) -> str:
        """Use proper HTML template instead of string concatenation - consider using a proper templating solution."""

        # TODO: The manual HTML string building is error-prone - consider using a proper templating solution

        # Create safe filter options
        safe_audit_leaders = [html.escape(leader) for leader in audit_leaders[1:] if leader in worst_controls["Audit Leader"].values]
        safe_audit_entities = [html.escape(entity) for entity in audit_entities[1:] if entity in worst_controls["Audit Entity"].values]

        # Create safe table data
        table_rows = []
        for _, row in worst_controls.iterrows():
            safe_row_data = {
                "leader": html.escape(str(row["Audit Leader"])),
                "entity": html.escape(str(row["Audit Entity"])),
                "cells": [html.escape(str(row[col]) if not isinstance(row[col], list) else ", ".join(row[col]))
                         for col in worst_controls.columns]
            }
            table_rows.append(safe_row_data)

        # Build template with safe data
        leader_options = "".join([f'<option value="{leader}">{leader}</option>' for leader in safe_audit_leaders])
        entity_options = "".join([f'<option value="{entity}">{entity}</option>' for entity in safe_audit_entities])

        table_headers = "".join([f'<th>{html.escape(col)}</th>' for col in worst_controls.columns])
        table_body = ""
        for row_data in table_rows:
            cells = "".join([f'<td>{cell}</td>' for cell in row_data["cells"]])
            table_body += f'<tr class="row" data-leader="{row_data["leader"]}" data-entity="{row_data["entity"]}">{cells}</tr>'

        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Controls Missing 3+ Elements</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .dropdown {{ margin-bottom: 20px; }}
                select {{ padding: 5px; font-size: 16px; }}
            </style>
        </head>
        <body>
            <h1>Controls Missing 3+ Elements</h1>
            <div class="dropdown">
                <label for="leaderFilter">Filter by Audit Leader: </label>
                <select id="leaderFilter" onchange="filterTable()">
                    <option value="all">All Leaders</option>
                    {leader_options}
                </select>
                &nbsp;&nbsp;&nbsp;
                <label for="entityFilter">Filter by Audit Entity: </label>
                <select id="entityFilter" onchange="filterTable()">
                    <option value="all">All Entities</option>
                    {entity_options}
                </select>
            </div>
            <table id="controlsTable">
                <thead>
                    <tr>{table_headers}</tr>
                </thead>
                <tbody>{table_body}</tbody>
            </table>

            <script>
                function filterTable() {{
                    const leaderFilter = document.getElementById('leaderFilter').value;
                    const entityFilter = document.getElementById('entityFilter').value;
                    const rows = document.querySelectorAll('#controlsTable tbody tr');

                    rows.forEach(row => {{
                        const leader = row.getAttribute('data-leader');
                        const entity = row.getAttribute('data-entity');

                        let leaderMatch, entityMatch;

                        if (leaderFilter === 'all') {{
                            leaderMatch = true;
                        }} else {{
                            leaderMatch = leaderFilter === leader;
                        }}

                        if (entityFilter === 'all') {{
                            entityMatch = true;
                        }} else {{
                            entityMatch = entityFilter === entity;
                        }}

                        if (leaderMatch && entityMatch) {{
                            row.style.display = '';
                        }} else {{
                            row.style.display = 'none';
                        }}
                    }});
                }}
            </script>
        </body>
        </html>
        """

        return html_template

    def create_combined_dashboard(self, controls_dataframe: pd.DataFrame,
                                 audit_leaders: List[str], audit_entities: List[str]) -> Dict[str, str]:
        """Create combined dashboard with multiple charts and filters."""

        # Create a comprehensive dashboard with multiple charts and filters
        dashboard_html = self.generate_combined_dashboard_html_template(controls_dataframe, audit_leaders, audit_entities)

        # Write dashboard HTML
        file_path = os.path.join(self.output_dir, "dashboard.html")
        with open(file_path, "w") as f:
            f.write(dashboard_html)

        return {"dashboard": file_path}

    def generate_combined_dashboard_html_template(self, controls_dataframe: pd.DataFrame,
                                                 audit_leaders: List[str], audit_entities: List[str]) -> str:
        """Generate HTML template for combined dashboard."""
        # Create JSON-serializable data for the dashboard
        dashboard_data = {
            "auditLeaders": audit_leaders[1:],  # Skip "All Leaders"
            "auditEntities": audit_entities[1:],  # Skip "All Entities"
            "records": controls_dataframe.to_dict(orient="records")
        }

        # Generate JavaScript code separately
        dashboard_javascript = self.generate_dashboard_javascript(dashboard_data)

        # Template uses double curly braces for Python string formatting to avoid conflicts with JavaScript
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Control Analysis Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
                .container {{ width: 100%; max-width: 1400px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #333; color: white; padding: 20px; margin-bottom: 20px; }}
                .filters {{ background-color: #f5f5f5; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
                .filter-group {{ margin-bottom: 10px; }}
                .chart-row {{ display: flex; flex-wrap: wrap; margin: -10px; }}
                .chart-container {{ flex: 1 1 calc(50% - 20px); min-width: 300px; margin: 10px; background-color: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .chart-title {{ padding: 10px; background-color: #f9f9f9; border-bottom: 1px solid #eee; }}
                .chart {{ height: 400px; }}
                select, button {{ padding: 8px; margin-right: 10px; }}
                h1, h2, h3 {{ margin-top: 0; }}
                .footer {{ margin-top: 20px; padding: 10px; background-color: #f5f5f5; text-align: center; font-size: 12px; }}
                .score-summary {{ display: flex; flex-wrap: wrap; margin-bottom: 20px; }}
                .score-card {{ flex: 1 1 200px; margin: 5px; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .excellent {{ background-color: #d4edda; border-left: 5px solid #28a745; }}
                .good {{ background-color: #fff3cd; border-left: 5px solid #ffc107; }}
                .needs-improvement {{ background-color: #f8d7da; border-left: 5px solid #dc3545; }}
                .score-number {{ font-size: 36px; font-weight: bold; margin: 10px 0; }}
                .score-label {{ font-size: 14px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Control Description Analysis Dashboard</h1>
            </div>
            <div class="container">
                <div class="filters">
                    <h3>Filters</h3>
                    <div class="filter-group">
                        <label for="leaderFilter">Audit Leader:</label>
                        <select id="leaderFilter" onchange="applyFilters()">
                            <option value="all">All Leaders</option>
                        </select>

                        <label for="entityFilter">Audit Entity:</label>
                        <select id="entityFilter" onchange="applyFilters()">
                            <option value="all">All Entities</option>
                        </select>

                        <label for="categoryFilter">Category:</label>
                        <select id="categoryFilter" onchange="applyFilters()">
                            <option value="all">All Categories</option>
                            <option value="Meets Expectations">Meets Expectations</option>
                            <option value="Requires Attention">Requires Attention</option>
                            <option value="Needs Improvement">Needs Improvement</option>
                        </select>

                        <button onclick="resetFilters()">Reset Filters</button>
                    </div>
                </div>

                <div class="score-summary" id="scoreSummary">
                    <!-- Score summary cards will be filled by JavaScript -->
                </div>

                <div class="chart-row">
                    <div class="chart-container">
                        <div class="chart-title">
                            <h3>Score Distribution</h3>
                        </div>
                        <div class="chart" id="scoreDistChart"></div>
                    </div>
                    <div class="chart-container">
                        <div class="chart-title">
                            <h3>Element Scores by Category</h3>
                        </div>
                        <div class="chart" id="radarChart"></div>
                    </div>
                </div>

                <div class="chart-row">
                    <div class="chart-container">
                        <div class="chart-title">
                            <h3>Missing Elements</h3>
                        </div>
                        <div class="chart" id="missingElementsChart"></div>
                    </div>
                    <div class="chart-container">
                        <div class="chart-title">
                            <h3>Vague Terms</h3>
                        </div>
                        <div class="chart" id="vagueTermsChart"></div>
                    </div>
                </div>
            </div>

            <div class="footer">
                <p>Enhanced Control Description Analyzer - Generated: <span id="generationDate"></span></p>
            </div>

            <script>
                {dashboard_javascript}
            </script>
        </body>
        </html>
        """

        return dashboard_html

    def generate_dashboard_javascript(self, dashboard_data: Dict[str, Any]) -> str:
        """Split JavaScript code generation into a separate method."""
        # The JSON serialization inside HTML templates could cause XSS vulnerabilities if the data contains user input - needs proper escaping
        safe_dashboard_data = json.dumps(dashboard_data, ensure_ascii=True)

        return f"""
                // Dashboard data (will be filled by Python)
                const dashboardData = {safe_dashboard_data};

                // JavaScript functions handle real-time filtering without server round-trips
                // Initialize filters
                function initializeFilters() {{
                    const leaderFilter = document.getElementById('leaderFilter');
                    const entityFilter = document.getElementById('entityFilter');

                    // Add audit leaders
                    dashboardData.auditLeaders.forEach(leader => {{
                        const option = document.createElement('option');
                        option.value = leader;
                        option.textContent = leader;
                        leaderFilter.appendChild(option);
                    }});

                    // Add audit entities
                    dashboardData.auditEntities.forEach(entity => {{
                        const option = document.createElement('option');
                        option.value = entity;
                        option.textContent = entity;
                        entityFilter.appendChild(option);
                    }});

                    // Update date
                    document.getElementById('generationDate').textContent = new Date().toLocaleDateString();
                }}

                // Create score summary cards
                function createScoreSummary(data) {{
                    const scoreSummary = document.getElementById('scoreSummary');
                    scoreSummary.innerHTML = '';

                    const categories = {{
                        'Meets Expectations': {{ count: 0, class: 'excellent' }},
                        'Requires Attention': {{ count: 0, class: 'good' }},
                        'Needs Improvement': {{ count: 0, class: 'needs-improvement' }}
                    }};

                    // Count records in each category
                    data.forEach(record => {{
                        if (categories[record.Category]) {{
                            categories[record.Category].count++;
                        }}
                    }});

                    // Create cards
                    Object.entries(categories).forEach(([category, info]) => {{
                        const card = document.createElement('div');
                        card.className = `score-card ${{info.class}}`;

                        const scoreNumber = document.createElement('div');
                        scoreNumber.className = 'score-number';
                        scoreNumber.textContent = info.count;

                        const scoreLabel = document.createElement('div');
                        scoreLabel.className = 'score-label';
                        scoreLabel.textContent = category;

                        card.appendChild(scoreNumber);
                        card.appendChild(scoreLabel);
                        scoreSummary.appendChild(card);
                    }});

                    // Add average score card
                    const avgScore = data.reduce((sum, record) => sum + record['Total Score'], 0) / data.length;
                    const avgCard = document.createElement('div');
                    avgCard.className = 'score-card';
                    avgCard.style.backgroundColor = '#e6f7ff';
                    avgCard.style.borderLeft = '5px solid #1890ff';

                    const avgScoreNumber = document.createElement('div');
                    avgScoreNumber.className = 'score-number';
                    avgScoreNumber.textContent = avgScore.toFixed(1);

                    const avgScoreLabel = document.createElement('div');
                    avgScoreLabel.className = 'score-label';
                    avgScoreLabel.textContent = 'Average Score';

                    avgCard.appendChild(avgScoreNumber);
                    avgCard.appendChild(avgScoreLabel);
                    scoreSummary.appendChild(avgCard);
                }}

                // Apply filters to data
                function filterData() {{
                    const leaderFilter = document.getElementById('leaderFilter').value;
                    const entityFilter = document.getElementById('entityFilter').value;
                    const categoryFilter = document.getElementById('categoryFilter').value;

                    return dashboardData.records.filter(record => {{
                        let leaderMatch, entityMatch, categoryMatch;

                        if (leaderFilter === 'all') {{
                            leaderMatch = true;
                        }} else {{
                            leaderMatch = record['Audit Leader'] === leaderFilter;
                        }}

                        if (entityFilter === 'all') {{
                            entityMatch = true;
                        }} else {{
                            entityMatch = record['Audit Entity'] === entityFilter;
                        }}

                        if (categoryFilter === 'all') {{
                            categoryMatch = true;
                        }} else {{
                            categoryMatch = record.Category === categoryFilter;
                        }}

                        return leaderMatch && entityMatch && categoryMatch;
                    }});
                }}

                // Create score distribution chart
                function createScoreDistChart(data) {{
                    const categories = ['Meets Expectations', 'Requires Attention', 'Needs Improvement'];

                    // Group data by category
                    const traces = categories.map((category, i) => {{
                        const categoryData = data.filter(record => record.Category === category);
                        return {{
                            x: categoryData.map(record => record['Total Score']),
                            type: 'histogram',
                            name: category,
                            marker: {{
                                color: ['#28a745', '#ffc107', '#dc3545'][i]
                            }},
                            autobinx: false,
                            xbins: {{
                                start: 0,
                                end: 100,
                                size: 5
                            }}
                        }};
                    }});

                    const layout = {{
                        barmode: 'stack',
                        xaxis: {{
                            title: 'Score (0-100)'
                        }},
                        yaxis: {{
                            title: 'Number of Controls'
                        }},
                        margin: {{
                            t: 10,
                            l: 60,
                            r: 10,
                            b: 60
                        }},
                        legend: {{
                            orientation: 'h',
                            y: -0.2
                        }}
                    }};

                    Plotly.newPlot('scoreDistChart', traces, layout);
                }}

                // Create radar chart
                function createRadarChart(data) {{
                    const elements = ['WHO', 'WHAT', 'WHEN']; // Core scoring elements only
                    const categories = ['Meets Expectations', 'Requires Attention', 'Needs Improvement'];

                    // Group data by category and calculate average scores
                    const traces = categories.map((category, i) => {{
                        const categoryData = data.filter(record => record.Category === category);

                        // Skip if no data for this category
                        if (categoryData.length === 0) {{
                            return {{
                                r: [0, 0, 0, 0],
                                theta: [...elements, elements[0]],
                                fill: 'toself',
                                name: category,
                                type: 'scatterpolar',
                                line: {{color: ['#28a745', '#ffc107', '#dc3545'][i]}}
                            }};
                        }}

                        // Calculate average for each element
                        const values = elements.map(element => {{
                            const scores = categoryData.map(record => record[`${{element}} Score`]);
                            return scores.reduce((sum, score) => sum + score, 0) / scores.length;
                        }});

                        // Close the loop
                        values.push(values[0]);

                        return {{
                            r: values,
                            theta: [...elements, elements[0]],
                            fill: 'toself',
                            name: category,
                            type: 'scatterpolar',
                            line: {{color: ['#28a745', '#ffc107', '#dc3545'][i]}}
                        }};
                    }});

                    const layout = {{
                        polar: {{
                            radialaxis: {{
                                visible: true,
                                range: [0, 30]
                            }}
                        }},
                        showlegend: true,
                        margin: {{
                            t: 10,
                            l: 10,
                            r: 10,
                            b: 10
                        }},
                        legend: {{
                            orientation: 'h',
                            y: -0.2
                        }}
                    }};

                    Plotly.newPlot('radarChart', traces, layout);
                }}

                // Create missing elements chart
                function createMissingElementsChart(data) {{
                    const elements = ['WHO', 'WHAT', 'WHEN']; // Core scoring elements only
                    const missingCounts = {{}};

                    // Initialize counts
                    elements.forEach(element => {{
                        missingCounts[element] = 0;
                    }});

                    // Count missing elements
                    data.forEach(record => {{
                        if (record['Missing Elements'] !== 'None') {{
                            record['Missing Elements'].split(', ').forEach(element => {{
                                if (missingCounts[element] !== undefined) {{
                                    missingCounts[element]++;
                                }}
                            }});
                        }}
                    }});

                    // Create trace
                    const trace = {{
                        x: elements,
                        y: elements.map(element => missingCounts[element]),
                        type: 'bar',
                        marker: {{
                            color: elements.map(element => missingCounts[element]),
                            colorscale: 'Reds'
                        }}
                    }};

                    const layout = {{
                        xaxis: {{
                            title: 'Control Element'
                        }},
                        yaxis: {{
                            title: 'Number of Controls Missing Element'
                        }},
                        margin: {{
                            t: 10,
                            l: 60,
                            r: 10,
                            b: 60
                        }}
                    }};

                    Plotly.newPlot('missingElementsChart', [trace], layout);
                }}

                // Create vague terms chart
                function createVagueTermsChart(data) {{
                    // Collect all vague terms
                    const termCounts = {{}};

                    data.forEach(record => {{
                        if (record.vague_terms_found && record.vague_terms_found.length > 0) {{
                            record.vague_terms_found.forEach(term => {{
                                termCounts[term] = (termCounts[term] || 0) + 1;
                            }});
                        }}
                    }});

                    // Sort terms by count
                    const sortedTerms = Object.entries(termCounts)
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 10); // Top 10 terms

                    if (sortedTerms.length === 0) {{
                        document.getElementById('vagueTermsChart').innerHTML = '<div style="display:flex;justify-content:center;align-items:center;height:100%;color:#666;">No vague terms detected</div>';
                        return;
                    }}

                    const trace = {{
                        x: sortedTerms.map(item => item[0]),
                        y: sortedTerms.map(item => item[1]),
                        type: 'bar',
                        marker: {{
                            color: sortedTerms.map(item => item[1]),
                            colorscale: 'Oranges'
                        }}
                    }};

                    const layout = {{
                        xaxis: {{
                            title: 'Vague Term'
                        }},
                        yaxis: {{
                            title: 'Occurrences'
                        }},
                        margin: {{
                            t: 10,
                            l: 60,
                            r: 10,
                            b: 60
                        }}
                    }};

                    Plotly.newPlot('vagueTermsChart', [trace], layout);
                }}

                // Apply filters and update all charts
                function applyFilters() {{
                    const filteredData = filterData();
                    createScoreSummary(filteredData);
                    createScoreDistChart(filteredData);
                    createRadarChart(filteredData);
                    createMissingElementsChart(filteredData);
                    createVagueTermsChart(filteredData);
                }}

                // Reset all filters
                function resetFilters() {{
                    document.getElementById('leaderFilter').value = 'all';
                    document.getElementById('entityFilter').value = 'all';
                    document.getElementById('categoryFilter').value = 'all';
                    applyFilters();
                }}

                // Initialize dashboard
                function initializeDashboard() {{
                    initializeFilters();
                    applyFilters();
                }}

                // Start dashboard when page loads
                window.onload = initializeDashboard;
        """

    def create_leadership_dashboard(self, controls_dataframe: pd.DataFrame,
                                   audit_leaders: List[str], audit_entities: List[str]) -> Dict[str, str]:
        """Create senior leadership dashboard with executive metrics and interactive Audit Leader filtering."""

        # Prepare data for JavaScript
        dashboard_data = self.prepare_dashboard_data(controls_dataframe, audit_leaders, audit_entities)

        # Create dashboard configuration
        config = DashboardConfig(
            title="Senior Leadership Dashboard",
            subtitle="Control Description Quality Overview - Core Elements: WHO (30%), WHAT (35%), WHEN (35%)"
        )

        # Generate HTML template
        dashboard_html = self.generate_leadership_dashboard_html_template(dashboard_data, config)

        # Write leadership dashboard HTML
        file_path = os.path.join(self.output_dir, "leadership_dashboard.html")
        with open(file_path, "w") as f:
            f.write(dashboard_html)

        return {"leadership_dashboard": file_path}

    def prepare_dashboard_data(self, controls_dataframe: pd.DataFrame,
                              audit_leaders: List[str], audit_entities: List[str]) -> Dict[str, Any]:
        """Prepare data for dashboard consumption."""
        return {
            "auditLeaders": [leader for leader in audit_leaders if leader != "All Leaders"],
            "auditEntities": [entity for entity in audit_entities if entity != "All Entities"],
            "records": controls_dataframe.to_dict(orient="records"),
            "generationDate": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def calculate_executive_metrics(self, controls_dataframe: pd.DataFrame) -> Dict[str, Any]:
        """Calculate executive metrics for dashboard."""
        if len(controls_dataframe) == 0:
            return {
                "averageScore": "0.0",
                "highRiskCount": 0,
                "completePercentage": "0",
                "totalControls": 0
            }

        average_score = controls_dataframe["Total Score"].sum() / len(controls_dataframe)
        high_risk_count = len(controls_dataframe[controls_dataframe["Category"] == "Needs Improvement"])
        complete_controls = len(controls_dataframe[controls_dataframe["Missing Elements Count"] == 0])

        if len(controls_dataframe) > 0:
            complete_percentage = (complete_controls / len(controls_dataframe)) * 100
        else:
            complete_percentage = 0

        total_controls = len(controls_dataframe)

        return {
            "averageScore": f"{average_score:.1f}",
            "highRiskCount": high_risk_count,
            "completePercentage": f"{complete_percentage:.0f}",
            "totalControls": total_controls
        }

    def generate_leadership_dashboard_html_template(self, dashboard_data: Dict[str, Any], config: DashboardConfig) -> str:
        """Generate HTML template for leadership dashboard."""
        # Template uses double curly braces for Python string formatting to avoid conflicts with JavaScript
        safe_dashboard_data = json.dumps(dashboard_data, ensure_ascii=True)
        dashboard_javascript = DASHBOARD_JAVASCRIPT_TEMPLATE.format(dashboard_data_json=safe_dashboard_data)

        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{config.title} - Control Description Analyzer</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                {DASHBOARD_CSS}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{config.title}</h1>
                <div class="subtitle">{config.subtitle}</div>
            </div>

            <div class="container">
                <div class="filter-section">
                    <div class="filter-group">
                        <label for="auditLeaderFilter">View Portfolio for:</label>
                        <select id="auditLeaderFilter" onchange="filterByAuditLeader()">
                            <option value="all">All Audit Leaders</option>
                        </select>
                    </div>
                    <div style="font-size: 12px; color: #7f8c8d;">
                        Generated: {config.generation_date}
                    </div>
                </div>

                <div class="metrics-grid" id="metricsGrid">
                    <!-- Metrics will be populated by JavaScript -->
                </div>

                <div class="charts-section">
                    <div class="chart-container">
                        <div class="chart-header">Portfolio Composition</div>
                        <div class="chart" id="compositionChart"></div>
                    </div>
                    <div class="chart-container">
                        <div class="chart-header">Performance by Audit Entity</div>
                        <div class="chart" id="performanceChart"></div>
                    </div>
                    <div class="chart-container full-width">
                        <div class="chart-header">Gap Analysis - Missing Elements</div>
                        <div class="chart" id="gapAnalysisChart"></div>
                    </div>
                </div>
            </div>

            <div class="footer">
                Control Description Analyzer - Senior Leadership Dashboard<br>
                Scoring: Core Elements (WHO/WHAT/WHEN) + Conditional WHERE + Demerits | WHY/ESCALATION are feedback-only<br>
                For detailed analysis, see individual control reports
            </div>

            <script>
                {dashboard_javascript}
            </script>
        </body>
        </html>
        """

        return dashboard_html


def generate_core_visualizations(results: List[Dict[str, Any]], output_dir: str) -> Dict[str, str]:
    """
    Generate enhanced visualizations with dropdown filters for control description analysis results.

    Args:
        results: List of control analysis result dictionaries
        output_dir: Directory to save visualization HTML files

    Returns:
        Dictionary of output files generated
    """
    builder = VisualizationBuilder(output_dir)
    return builder.generate_core_visualizations(results)


def create_leadership_dashboard(controls_dataframe: pd.DataFrame, audit_leaders: List[str], audit_entities: List[str]) -> str:
    """
    Create a senior leadership dashboard with executive metrics and interactive Audit Leader filtering.

    Args:
        controls_dataframe: DataFrame with control analysis results
        audit_leaders: List of unique audit leaders
        audit_entities: List of unique audit entities

    Returns:
        HTML string for the leadership dashboard
    """
    builder = VisualizationBuilder("")
    dashboard_data = builder.prepare_dashboard_data(controls_dataframe, audit_leaders, audit_entities)
    config = DashboardConfig()
    return builder.generate_leadership_dashboard_html_template(dashboard_data, config)