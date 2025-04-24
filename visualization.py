
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from collections import Counter

def generate_core_visualizations(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame([{
        "Control ID": r["control_id"],
        "Total Score": r["total_score"],
        "Category": r["category"],
        "Missing Elements Count": len(r["missing_elements"]),
        "WHO Score": r["weighted_scores"]["WHO"],
        "WHEN Score": r["weighted_scores"]["WHEN"],
        "WHAT Score": r["weighted_scores"]["WHAT"],
        "WHY Score": r["weighted_scores"]["WHY"],
        # "EVIDENCE Score": r["weighted_scores"]["EVIDENCE"],
        # "STORAGE Score": r["weighted_scores"]["STORAGE"],
        "ESCALATION Score": r["weighted_scores"]["ESCALATION"],
        "Missing Elements": ", ".join(r["missing_elements"]) if r["missing_elements"] else "None",
        "Audit Leader": r.get("Audit Leader", r.get("metadata", {}).get("Audit Leader", "Unknown")),
        "vague_terms_found": r.get("vague_terms_found", [])
    } for r in results])

    output_files = {}

    # Score distribution
    fig_score_dist = px.histogram(df, x="Total Score", color="Category", nbins=20,
        title="Distribution of Control Description Scores",
        labels={"Total Score": "Score (0-100)", "count": "Number of Controls"},
        color_discrete_map={"Excellent": "#28a745", "Good": "#ffc107", "Needs Improvement": "#dc3545"})
    fig_score_dist.write_html(os.path.join(output_dir, "score_distribution.html"))

    # Element radar
    elements = ["WHO", "WHEN", "WHAT", "WHY", "ESCALATION"]
    category_avg = df.groupby("Category")[[f"{e} Score" for e in elements]].mean().reset_index()
    fig_radar = go.Figure()
    for category in category_avg["Category"]:
        values = [category_avg.loc[category_avg["Category"] == category, f"{e} Score"].values[0] for e in elements]
        values.append(values[0])
        fig_radar.add_trace(go.Scatterpolar(r=values, theta=elements + [elements[0]], fill='toself', name=category))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 30])), title="Average Element Scores by Category")
    fig_radar.write_html(os.path.join(output_dir, "element_radar.html"))

    # Missing elements
    missing_counts = {e: 0 for e in elements}
    for m in df["Missing Elements"]:
        if m != "None":
            for e in m.split(", "):
                if e in missing_counts:
                    missing_counts[e] += 1
    missing_df = pd.DataFrame({"Element": list(missing_counts.keys()), "Missing Count": list(missing_counts.values())})
    fig_missing = px.bar(missing_df, x="Element", y="Missing Count", title="Frequency of Missing Elements",
                         labels={"Element": "Control Element", "Missing Count": "Number of Controls Missing Element"},
                         color="Missing Count", color_continuous_scale=px.colors.sequential.Reds)
    fig_missing.write_html(os.path.join(output_dir, "missing_elements.html"))

    # Vague term frequency
    vague_terms = sum(df["vague_terms_found"], [])
    term_counts = Counter(vague_terms)
    if term_counts:
        vague_df = pd.DataFrame(term_counts.items(), columns=["Term", "Count"]).sort_values("Count", ascending=False)
        fig_vague = px.bar(vague_df, x="Term", y="Count", title="Frequency of Vague Terms",
                           labels={"Term": "Vague Term", "Count": "Occurrences"},
                           color="Count", color_continuous_scale=px.colors.sequential.Oranges)
        fig_vague.write_html(os.path.join(output_dir, "vague_terms.html"))

    # Audit Leader breakdown
    if "Audit Leader" in df.columns:
        leader_avg = df.groupby("Audit Leader")["Total Score"].mean().reset_index().sort_values("Total Score", ascending=False)
        fig_leader_avg = px.bar(leader_avg, x="Audit Leader", y="Total Score", title="Average Control Score by Audit Leader",
                                labels={"Total Score": "Avg Score"}, color="Total Score",
                                color_continuous_scale=px.colors.sequential.Blues)
        fig_leader_avg.write_html(os.path.join(output_dir, "leader_avg_score.html"))

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

    return output_files
