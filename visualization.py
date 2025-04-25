import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from collections import Counter, defaultdict



def generate_core_visualizations(results, output_dir):
    """
    Generate enhanced visualizations with dropdown filters for control description analysis results

    Args:
        results: List of control analysis result dictionaries
        output_dir: Directory to save visualization HTML files

    Returns:
        Dictionary of output files generated
    """
    os.makedirs(output_dir, exist_ok=True)

    # Transform results to DataFrame for easier manipulation
    df = pd.DataFrame([{
        "Control ID": r["control_id"],
        "Total Score": r["total_score"],
        "Category": r["category"],
        "Missing Elements Count": len(r["missing_elements"]),
        "WHO Score": r["weighted_scores"]["WHO"],
        "WHEN Score": r["weighted_scores"]["WHEN"],
        "WHAT Score": r["weighted_scores"]["WHAT"],
        "WHY Score": r["weighted_scores"]["WHY"],
        "ESCALATION Score": r["weighted_scores"]["ESCALATION"],
        "Missing Elements": ", ".join(r["missing_elements"]) if r["missing_elements"] else "None",
        "Audit Leader": r.get("Audit Leader", r.get("metadata", {}).get("Audit Leader", "Unknown")),
        "vague_terms_found": r.get("vague_terms_found", [])
    } for r in results])

    output_files = {}

    # ========== SCORE DISTRIBUTION WITH AUDIT LEADER FILTER ==========
    # Create dropdown options from unique audit leaders
    audit_leaders = df["Audit Leader"].unique().tolist()
    audit_leaders.insert(0, "All Leaders")  # Add "All" option at beginning

    # Create figure with dropdown menu for Audit Leader
    fig_score_dist = px.histogram(df, x="Total Score", color="Category", nbins=20,
                                  title="Distribution of Control Description Scores",
                                  labels={"Total Score": "Score (0-100)", "count": "Number of Controls"},
                                  color_discrete_map={"Excellent": "#28a745", "Good": "#ffc107",
                                                      "Needs Improvement": "#dc3545"})

    # Create dropdown menu buttons for each audit leader
    dropdown_buttons = []

    # Add "All Leaders" button
    dropdown_buttons.append(dict(
        method="update",
        label="All Leaders",
        args=[{"visible": [True] * len(df["Category"].unique())}]
    ))

    # Add filter buttons for each audit leader
    for leader in audit_leaders[1:]:  # Skip "All Leaders"
        # Create a filter mask for this leader
        leader_data = df[df["Audit Leader"] == leader]

        # Create button for this leader
        button = dict(
            method="update",
            label=leader,
            args=[
                {"data": [
                    go.Histogram(
                        x=leader_data[leader_data["Category"] == cat]["Total Score"],
                        name=cat,
                        marker_color=color
                    )
                    for cat, color in zip(
                        ["Excellent", "Good", "Needs Improvement"],
                        ["#28a745", "#ffc107", "#dc3545"]
                    )
                ]},
                {"title": f"Distribution of Control Description Scores - {leader}"}
            ]
        )
        dropdown_buttons.append(button)

    # Update layout to include dropdown menu
    fig_score_dist.update_layout(
        updatemenus=[dict(
            buttons=dropdown_buttons,
            direction="down",
            showactive=True,
            x=1.0,
            xanchor="right",
            y=1.15,
            yanchor="top"
        )],
        annotations=[dict(
            text="Audit Leader:",
            showarrow=False,
            x=1.0,
            y=1.2,
            xref="paper",
            yref="paper",
            align="right"
        )]
    )

    fig_score_dist.write_html(os.path.join(output_dir, "score_distribution.html"))
    output_files["score_distribution"] = os.path.join(output_dir, "score_distribution.html")

    # ========== ELEMENT RADAR CHART WITH AUDIT LEADER FILTER ==========
    elements = ["WHO", "WHEN", "WHAT", "WHY", "ESCALATION"]

    # Create figure
    fig_radar = go.Figure()

    # Add traces for each category (across all leaders)
    category_avg = df.groupby("Category")[[f"{e} Score" for e in elements]].mean().reset_index()
    for category in category_avg["Category"]:
        values = [category_avg.loc[category_avg["Category"] == category, f"{e} Score"].values[0] for e in elements]
        values.append(values[0])  # Close the loop
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=elements + [elements[0]],
            fill='toself',
            name=f"{category} (All Leaders)"
        ))

    # Create dropdown menu buttons for each audit leader
    dropdown_buttons = []

    # Add "All Leaders" button showing the original traces (already added)
    dropdown_buttons.append(dict(
        method="update",
        label="All Leaders",
        args=[{"visible": [True] * len(category_avg["Category"])}]
    ))

    # Add filter buttons for each audit leader
    for leader in audit_leaders[1:]:  # Skip "All Leaders"
        # Calculate averages for this leader
        leader_data = df[df["Audit Leader"] == leader]
        if len(leader_data) > 0:
            leader_category_avg = leader_data.groupby("Category")[[f"{e} Score" for e in elements]].mean().reset_index()

            # Generate data for each category this leader has
            leader_traces = []
            for category in leader_category_avg["Category"]:
                values = [leader_category_avg.loc[leader_category_avg["Category"] == category, f"{e} Score"].values[0]
                          for e in elements]
                values.append(values[0])  # Close the loop
                leader_traces.append(go.Scatterpolar(
                    r=values,
                    theta=elements + [elements[0]],
                    fill='toself',
                    name=f"{category} ({leader})"
                ))

            # Create update data for this leader's button
            visible_list = [False] * len(category_avg["Category"])
            update_data = []
            update_traces = {}

            for trace in leader_traces:
                update_data.append(trace)

            # Create button for this leader
            button = dict(
                method="update",
                label=leader,
                args=[
                    {"data": update_data},
                    {"title": f"Average Element Scores by Category - {leader}"}
                ]
            )
            dropdown_buttons.append(button)

    # Update layout to include dropdown menu
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 30])),
        title="Average Element Scores by Category",
        updatemenus=[dict(
            buttons=dropdown_buttons,
            direction="down",
            showactive=True,
            x=1.0,
            xanchor="right",
            y=1.15,
            yanchor="top"
        )],
        annotations=[dict(
            text="Audit Leader:",
            showarrow=False,
            x=1.0,
            y=1.2,
            xref="paper",
            yref="paper",
            align="right"
        )]
    )

    fig_radar.write_html(os.path.join(output_dir, "element_radar.html"))
    output_files["element_radar"] = os.path.join(output_dir, "element_radar.html")

    # ========== MISSING ELEMENTS WITH AUDIT LEADER FILTER ==========
    elements = ["WHO", "WHEN", "WHAT", "WHY", "ESCALATION"]

    # Function to count missing elements for specific data
    def count_missing_elements(data_frame, element_list):
        missing_counts = {e: 0 for e in element_list}
        for m in data_frame["Missing Elements"]:
            if m != "None":
                for e in m.split(", "):
                    if e in missing_counts:
                        missing_counts[e] += 1
        return pd.DataFrame({
            "Element": list(missing_counts.keys()),
            "Missing Count": list(missing_counts.values())
        })

    # Count missing elements for all data
    missing_df = count_missing_elements(df, elements)

    # Create figure with dropdown menu for Audit Leader
    fig_missing = px.bar(missing_df, x="Element", y="Missing Count",
                         title="Frequency of Missing Elements",
                         labels={"Element": "Control Element", "Missing Count": "Number of Controls Missing Element"},
                         color="Missing Count", color_continuous_scale=px.colors.sequential.Reds)

    # Create dropdown menu buttons for each audit leader
    dropdown_buttons = []

    # Add "All Leaders" button
    dropdown_buttons.append(dict(
        method="update",
        label="All Leaders",
        args=[
            {"data": [go.Bar(
                x=missing_df["Element"],
                y=missing_df["Missing Count"],
                marker=dict(
                    color=missing_df["Missing Count"],
                    colorscale="Reds"
                )
            )]},
            {"title": "Frequency of Missing Elements"}
        ]
    ))

    # Add filter buttons for each audit leader
    for leader in audit_leaders[1:]:  # Skip "All Leaders"
        # Count missing elements for this leader
        leader_data = df[df["Audit Leader"] == leader]
        leader_missing_df = count_missing_elements(leader_data, elements)

        # Create button for this leader
        button = dict(
            method="update",
            label=leader,
            args=[
                {"data": [go.Bar(
                    x=leader_missing_df["Element"],
                    y=leader_missing_df["Missing Count"],
                    marker=dict(
                        color=leader_missing_df["Missing Count"],
                        colorscale="Reds"
                    )
                )]},
                {"title": f"Frequency of Missing Elements - {leader}"}
            ]
        )
        dropdown_buttons.append(button)

    # Update layout to include dropdown menu
    fig_missing.update_layout(
        updatemenus=[dict(
            buttons=dropdown_buttons,
            direction="down",
            showactive=True,
            x=1.0,
            xanchor="right",
            y=1.15,
            yanchor="top"
        )],
        annotations=[dict(
            text="Audit Leader:",
            showarrow=False,
            x=1.0,
            y=1.2,
            xref="paper",
            yref="paper",
            align="right"
        )]
    )

    fig_missing.write_html(os.path.join(output_dir, "missing_elements.html"))
    output_files["missing_elements"] = os.path.join(output_dir, "missing_elements.html")

    # ========== VAGUE TERM FREQUENCY WITH AUDIT LEADER FILTER ==========
    # Function to count vague terms for specific data
    def count_vague_terms(data_frame):
        vague_terms = sum(data_frame["vague_terms_found"], [])
        term_counts = Counter(vague_terms)
        if term_counts:
            return pd.DataFrame(term_counts.items(), columns=["Term", "Count"]).sort_values("Count", ascending=False)
        return pd.DataFrame(columns=["Term", "Count"])

    # Count vague terms for all data
    vague_df = count_vague_terms(df)

    if not vague_df.empty:
        # Create figure with dropdown menu for Audit Leader
        fig_vague = px.bar(vague_df, x="Term", y="Count",
                           title="Frequency of Vague Terms",
                           labels={"Term": "Vague Term", "Count": "Occurrences"},
                           color="Count", color_continuous_scale=px.colors.sequential.Oranges)

        # Create dropdown menu buttons for each audit leader
        dropdown_buttons = []

        # Add "All Leaders" button
        dropdown_buttons.append(dict(
            method="update",
            label="All Leaders",
            args=[
                {"data": [go.Bar(
                    x=vague_df["Term"],
                    y=vague_df["Count"],
                    marker=dict(
                        color=vague_df["Count"],
                        colorscale="Oranges"
                    )
                )]},
                {"title": "Frequency of Vague Terms"}
            ]
        ))

        # Add filter buttons for each audit leader
        for leader in audit_leaders[1:]:  # Skip "All Leaders"
            # Count vague terms for this leader
            leader_data = df[df["Audit Leader"] == leader]
            leader_vague_df = count_vague_terms(leader_data)

            if not leader_vague_df.empty:
                # Create button for this leader
                button = dict(
                    method="update",
                    label=leader,
                    args=[
                        {"data": [go.Bar(
                            x=leader_vague_df["Term"],
                            y=leader_vague_df["Count"],
                            marker=dict(
                                color=leader_vague_df["Count"],
                                colorscale="Oranges"
                            )
                        )]},
                        {"title": f"Frequency of Vague Terms - {leader}"}
                    ]
                )
                dropdown_buttons.append(button)

        # Update layout to include dropdown menu
        fig_vague.update_layout(
            updatemenus=[dict(
                buttons=dropdown_buttons,
                direction="down",
                showactive=True,
                x=1.0,
                xanchor="right",
                y=1.15,
                yanchor="top"
            )],
            annotations=[dict(
                text="Audit Leader:",
                showarrow=False,
                x=1.0,
                y=1.2,
                xref="paper",
                yref="paper",
                align="right"
            )]
        )

        fig_vague.write_html(os.path.join(output_dir, "vague_terms.html"))
        output_files["vague_terms"] = os.path.join(output_dir, "vague_terms.html")

    # Controls missing 3+ elements
    worst_controls = df[df["Missing Elements Count"] >= 3]
    if not worst_controls.empty:
        worst_controls = worst_controls.sort_values("Total Score")
        # Add a dropdown filter for the HTML table
        html_string = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Controls Missing 3+ Elements</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .dropdown { margin-bottom: 20px; }
                select { padding: 5px; font-size: 16px; }
            </style>
        </head>
        <body>
            <h1>Controls Missing 3+ Elements</h1>
            <div class="dropdown">
                <label for="leaderFilter">Filter by Audit Leader: </label>
                <select id="leaderFilter" onchange="filterTable()">
                    <option value="all">All Leaders</option>
        """

        # Add options for each audit leader
        for leader in audit_leaders[1:]:
            if leader in worst_controls["Audit Leader"].values:
                html_string += f'            <option value="{leader}">{leader}</option>\n'

        html_string += """
                </select>
            </div>
            <table id="controlsTable">
                <thead>
                    <tr>
        """

        # Add table headers
        for column in worst_controls.columns:
            html_string += f'            <th>{column}</th>\n'

        html_string += """
                    </tr>
                </thead>
                <tbody>
        """

        # Add table rows
        for _, row in worst_controls.iterrows():
            html_string += f'        <tr class="row" data-leader="{row["Audit Leader"]}">\n'
            for column in worst_controls.columns:
                cell_value = row[column]
                if isinstance(cell_value, list):
                    cell_value = ", ".join(cell_value)
                html_string += f'            <td>{cell_value}</td>\n'
            html_string += '        </tr>\n'

        html_string += """
                </tbody>
            </table>

            <script>
                function filterTable() {
                    const filter = document.getElementById('leaderFilter').value;
                    const rows = document.querySelectorAll('#controlsTable tbody tr');

                    rows.forEach(row => {
                        const leader = row.getAttribute('data-leader');
                        if (filter === 'all' || filter === leader) {
                            row.style.display = '';
                        } else {
                            row.style.display = 'none';
                        }
                    });
                }
            </script>
        </body>
        </html>
        """

        # Write HTML file
        with open(os.path.join(output_dir, "worst_controls.html"), 'w') as f:
            f.write(html_string)

        output_files["worst_controls"] = os.path.join(output_dir, "worst_controls.html")

    # ========== AUDIT LEADER BREAKDOWN ==========
    if "Audit Leader" in df.columns:
        # Average score by audit leader (already filtered by nature)
        leader_avg = df.groupby("Audit Leader")["Total Score"].mean().reset_index().sort_values("Total Score",
                                                                                                ascending=False)
        fig_leader_avg = px.bar(leader_avg, x="Audit Leader", y="Total Score",
                                title="Average Control Score by Audit Leader",
                                labels={"Total Score": "Avg Score"},
                                color="Total Score",
                                color_continuous_scale=px.colors.sequential.Blues)

        fig_leader_avg.write_html(os.path.join(output_dir, "leader_avg_score.html"))
        output_files["leader_avg_score"] = os.path.join(output_dir, "leader_avg_score.html")

        # Create missing elements by leader chart (stacked bar)
        missing_data = []
        for _, row in df.iterrows():
            leader = row["Audit Leader"]
            if row["Missing Elements"] != "None":
                for elem in row["Missing Elements"].split(", "):
                    missing_data.append((leader, elem))

        if missing_data:
            missing_df = pd.DataFrame(missing_data, columns=["Audit Leader", "Element"])
            missing_counts = missing_df.groupby(["Audit Leader", "Element"]).size().reset_index(name="Count")

            fig_missing_stack = px.bar(missing_counts, x="Audit Leader", y="Count", color="Element",
                                       title="Missing Elements by Audit Leader", barmode="stack")

            fig_missing_stack.write_html(os.path.join(output_dir, "leader_missing_elements.html"))
            output_files["leader_missing_elements"] = os.path.join(output_dir, "leader_missing_elements.html")

    # ========== COMBINED DASHBOARD WITH FILTERS ==========
    # Create a comprehensive dashboard with multiple charts and filters
    dashboard_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Control Analysis Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
            .container { width: 100%; max-width: 1400px; margin: 0 auto; padding: 20px; }
            .header { background-color: #333; color: white; padding: 20px; margin-bottom: 20px; }
            .filters { background-color: #f5f5f5; padding: 15px; margin-bottom: 20px; border-radius: 5px; }
            .filter-group { margin-bottom: 10px; }
            .chart-row { display: flex; flex-wrap: wrap; margin: -10px; }
            .chart-container { flex: 1 1 calc(50% - 20px); min-width: 300px; margin: 10px; background-color: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .chart-title { padding: 10px; background-color: #f9f9f9; border-bottom: 1px solid #eee; }
            .chart { height: 400px; }
            select, button { padding: 8px; margin-right: 10px; }
            h1, h2, h3 { margin-top: 0; }
            .footer { margin-top: 20px; padding: 10px; background-color: #f5f5f5; text-align: center; font-size: 12px; }
            .score-summary { display: flex; flex-wrap: wrap; margin-bottom: 20px; }
            .score-card { flex: 1 1 200px; margin: 5px; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .excellent { background-color: #d4edda; border-left: 5px solid #28a745; }
            .good { background-color: #fff3cd; border-left: 5px solid #ffc107; }
            .needs-improvement { background-color: #f8d7da; border-left: 5px solid #dc3545; }
            .score-number { font-size: 36px; font-weight: bold; margin: 10px 0; }
            .score-label { font-size: 14px; color: #666; }
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
                        <!-- Leader options will be filled by JavaScript -->
                    </select>

                    <label for="categoryFilter">Category:</label>
                    <select id="categoryFilter" onchange="applyFilters()">
                        <option value="all">All Categories</option>
                        <option value="Excellent">Excellent</option>
                        <option value="Good">Good</option>
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
            // Dashboard data (will be filled by Python)
            const dashboardData = {% raw %}{{dashboard_data}}{% endraw %};

            // Initialize filters
            function initializeFilters() {
                const leaderFilter = document.getElementById('leaderFilter');

                // Add audit leaders
                dashboardData.auditLeaders.forEach(leader => {
                    const option = document.createElement('option');
                    option.value = leader;
                    option.textContent = leader;
                    leaderFilter.appendChild(option);
                });

                // Update date
                document.getElementById('generationDate').textContent = new Date().toLocaleDateString();
            }

            // Create score summary cards
            function createScoreSummary(data) {
                const scoreSummary = document.getElementById('scoreSummary');
                scoreSummary.innerHTML = '';

                const categories = {
                    'Excellent': { count: 0, class: 'excellent' },
                    'Good': { count: 0, class: 'good' },
                    'Needs Improvement': { count: 0, class: 'needs-improvement' }
                };

                // Count records in each category
                data.forEach(record => {
                    if (categories[record.Category]) {
                        categories[record.Category].count++;
                    }
                });

                // Create cards
                Object.entries(categories).forEach(([category, info]) => {
                    const card = document.createElement('div');
                    card.className = `score-card ${info.class}`;

                    const scoreNumber = document.createElement('div');
                    scoreNumber.className = 'score-number';
                    scoreNumber.textContent = info.count;

                    const scoreLabel = document.createElement('div');
                    scoreLabel.className = 'score-label';
                    scoreLabel.textContent = category;

                    card.appendChild(scoreNumber);
                    card.appendChild(scoreLabel);
                    scoreSummary.appendChild(card);
                });

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
            }

            // Apply filters to data
            function filterData() {
                const leaderFilter = document.getElementById('leaderFilter').value;
                const categoryFilter = document.getElementById('categoryFilter').value;

                return dashboardData.records.filter(record => {
                    const leaderMatch = leaderFilter === 'all' || record['Audit Leader'] === leaderFilter;
                    const categoryMatch = categoryFilter === 'all' || record.Category === categoryFilter;
                    return leaderMatch && categoryMatch;
                });
            }

            // Create score distribution chart
            function createScoreDistChart(data) {
                const categories = ['Excellent', 'Good', 'Needs Improvement'];
                const colors = ['#28a745', '#ffc107', '#dc3545'];

                // Group data by category
                const traces = categories.map((category, i) => {
                    const categoryData = data.filter(record => record.Category === category);
                    return {
                        x: categoryData.map(record => record['Total Score']),
                        type: 'histogram',
                        name: category,
                        marker: {
                            color: colors[i]
                        },
                        autobinx: false,
                        xbins: {
                            start: 0,
                            end: 100,
                            size: 5
                        }
                    };
                });

                const layout = {
                    barmode: 'stack',
                    xaxis: {
                        title: 'Score (0-100)'
                    },
                    yaxis: {
                        title: 'Number of Controls'
                    },
                    margin: {
                        t: 10,
                        l: 60,
                        r: 10,
                        b: 60
                    },
                    legend: {
                        orientation: 'h',
                        y: -0.2
                    }
                };

                Plotly.newPlot('scoreDistChart', traces, layout);
            }

            // Create radar chart
            function createRadarChart(data) {
                const elements = ['WHO', 'WHEN', 'WHAT', 'WHY', 'ESCALATION'];
                const categories = ['Excellent', 'Good', 'Needs Improvement'];
                const colors = ['#28a745', '#ffc107', '#dc3545'];

                // Group data by category and calculate average scores
                const traces = categories.map((category, i) => {
                    const categoryData = data.filter(record => record.Category === category);

                    // Skip if no data for this category
                    if (categoryData.length === 0) {
                        return {
                            r: [0, 0, 0, 0, 0, 0],
                            theta: [...elements, elements[0]],
                            fill: 'toself',
                            name: category,
                            type: 'scatterpolar',
                            line: {color: colors[i]}
                        };
                    }

                    // Calculate average for each element
                    const values = elements.map(element => {
                        const scores = categoryData.map(record => record[`${element} Score`]);
                        return scores.reduce((sum, score) => sum + score, 0) / scores.length;
                    });

                    // Close the loop
                    values.push(values[0]);

                    return {
                        r: values,
                        theta: [...elements, elements[0]],
                        fill: 'toself',
                        name: category,
                        type: 'scatterpolar',
                        line: {color: colors[i]}
                    };
                });

                const layout = {
                    polar: {
                        radialaxis: {
                            visible: true,
                            range: [0, 30]
                        }
                    },
                    showlegend: true,
                    margin: {
                        t: 10,
                        l: 10,
                        r: 10,
                        b: 10
                    },
                    legend: {
                        orientation: 'h',
                        y: -0.2
                    }
                };

                Plotly.newPlot('radarChart', traces, layout);
            }

            // Create missing elements chart
            function createMissingElementsChart(data) {
                const elements = ['WHO', 'WHEN', 'WHAT', 'WHY', 'ESCALATION'];
                const missingCounts = {};

                // Initialize counts
                elements.forEach(element => {
                    missingCounts[element] = 0;
                });

                // Count missing elements
                data.forEach(record => {
                    if (record['Missing Elements'] !== 'None') {
                        record['Missing Elements'].split(', ').forEach(element => {
                            if (missingCounts[element] !== undefined) {
                                missingCounts[element]++;
                            }
                        });
                    }
                });

                // Create trace
                const trace = {
                    x: elements,
                    y: elements.map(element => missingCounts[element]),
                    type: 'bar',
                    marker: {
                        color: elements.map(element => missingCounts[element]),
                        colorscale: 'Reds'
                    }
                };

                const layout = {
                    xaxis: {
                        title: 'Control Element'
                    },
                    yaxis: {
                        title: 'Number of Controls Missing Element'
                    },
                    margin: {
                        t: 10,
                        l: 60,
                        r: 10,
                        b: 60
                    }
                };

                Plotly.newPlot('missingElementsChart', [trace], layout);
            }

            // Create vague terms chart
            function createVagueTermsChart(data) {
                // Collect all vague terms
                const termCounts = {};

                data.forEach(record => {
                    if (record.vague_terms_found && record.vague_terms_found.length > 0) {
                        record.vague_terms_found.forEach(term => {
                            termCounts[term] = (termCounts[term] || 0) + 1;
                        });
                    }
                });

                // Sort terms by count
                const sortedTerms = Object.entries(termCounts)
                    .sort((a, b) => b[1] - a[1])
                    .slice(0, 10); // Top 10 terms

                if (sortedTerms.length === 0) {
                    document.getElementById('vagueTermsChart').innerHTML = '<div style="display:flex;justify-content:center;align-items:center;height:100%;color:#666;">No vague terms detected</div>';
                    return;
                }

                const trace = {
                    x: sortedTerms.map(item => item[0]),
                    y: sortedTerms.map(item => item[1]),
                    type: 'bar',
                    marker: {
                        color: sortedTerms.map(item => item[1]),
                        colorscale: 'Oranges'
                    }
                };

                const layout = {
                    xaxis: {
                        title: 'Vague Term'
                    },
                    yaxis: {
                        title: 'Occurrences'
                    },
                    margin: {
                        t: 10,
                        l: 60,
                        r: 10,
                        b: 60
                    }
                };

                Plotly.newPlot('vagueTermsChart', [trace], layout);
            }

            // Apply filters and update all charts
            function applyFilters() {
                const filteredData = filterData();
                createScoreSummary(filteredData);
                createScoreDistChart(filteredData);
                createRadarChart(filteredData);
                createMissingElementsChart(filteredData);
                createVagueTermsChart(filteredData);
            }

            // Reset all filters
            function resetFilters() {
                document.getElementById('leaderFilter').value = 'all';
                document.getElementById('categoryFilter').value = 'all';
                applyFilters();
            }

            // Initialize dashboard
            function initializeDashboard() {
                initializeFilters();
                applyFilters();
            }

            // Start dashboard when page loads
            window.onload = initializeDashboard;
        </script>
    </body>
    </html>
    """

    # Create JSON-serializable data for the dashboard
    dashboard_data = {
        "auditLeaders": audit_leaders[1:],  # Skip "All Leaders"
        "records": df.to_dict(orient="records")
    }

    # Replace placeholder with actual JSON data
    import json
    dashboard_html = dashboard_html.replace("{% raw %}{{dashboard_data}}{% endraw %}", json.dumps(dashboard_data))

    # Write dashboard HTML
    with open(os.path.join(output_dir, "dashboard.html"), "w") as f:
        f.write(dashboard_html)

    output_files["dashboard"] = os.path.join(output_dir, "dashboard.html")

    return output_files

