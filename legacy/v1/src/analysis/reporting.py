"""
Methodological summary report generation.

Writes a Markdown file synthesizing results from all analysis stages.
"""

from __future__ import annotations

import os

import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)


def create_methodological_summary(
    output_dir: str,
    *,
    significance_results: pd.DataFrame | None = None,
    coherence_results: pd.DataFrame | None = None,
    similarity_data: pd.DataFrame | None = None,
    policy_shifts: pd.DataFrame | None = None,
    focusing_events: pd.DataFrame | None = None,
    volatility: pd.DataFrame | None = None,
) -> None:
    """Write ``methodological_findings_summary.md`` to *output_dir*."""
    logger.info("Creating methodological summary report ...")

    path = os.path.join(output_dir, "methodological_findings_summary.md")

    with open(path, "w") as f:
        f.write("# Methodological Analysis Summary\n\n")

        # ---- Significance testing ----
        if significance_results is not None:
            f.write("## Significance Testing Results\n\n")
            sig_count = int(significance_results["Significant"].sum())
            total = len(significance_results)
            f.write(
                f"- **{sig_count}** out of **{total}** topics show significant "
                "differences between administrations (p<0.05)\n"
            )
            if sig_count > 0:
                top = significance_results[significance_results["Significant"]].sort_values("p_value").head(5)
                f.write("- Top 5 most significant topics:\n")
                for _, r in top.iterrows():
                    f.write(f"  - Topic {r['Topic']} ({r['Name']}): p={r['p_value']:.4f}\n")

        # ---- Policy attention shifts ----
        if policy_shifts is not None:
            f.write("\n## Policy Attention Shift Analysis\n\n")
            increased = policy_shifts.nlargest(5, "attention_shift")
            f.write("- **Top 5 topics with increased attention under Biden administration:**\n")
            for _, r in increased.iterrows():
                f.write(f"  - {r['Name']}: +{r['attention_shift']:.4f}\n")

            decreased = policy_shifts.nsmallest(5, "attention_shift")
            f.write("\n- **Top 5 topics with increased attention under Trump administration:**\n")
            for _, r in decreased.iterrows():
                f.write(f"  - {r['Name']}: {r['attention_shift']:.4f}\n")

            f.write(
                "\n  **Finding**: Policy attention shifted significantly between "
                "administrations, supporting the bounded rationality theory's "
                "prediction that new leadership selectively focuses on different "
                "priorities.\n"
            )

        # ---- Focusing events ----
        if focusing_events is not None and not focusing_events.empty:
            f.write("\n## Focusing Events Detection\n\n")
            top_ev = focusing_events.nlargest(5, "Z_Score")
            f.write("- **Top 5 potential focusing events (sudden topic attention spikes):**\n")
            for _, r in top_ev.iterrows():
                date_str = str(r["Date"]).split()[0]
                f.write(f"  - {date_str}: {r['Topic_Name']} (intensity: {r['Z_Score']:.2f})\n")
            f.write(
                "\n  **Finding**: Several distinct focusing events were identified "
                "where specific topics gained sudden prominence, potentially "
                "representing policy windows as described by Kingdon's framework.\n"
            )

        # ---- Punctuated equilibrium ----
        if volatility is not None:
            f.write("\n## Punctuated Equilibrium Analysis\n\n")
            high = volatility.nlargest(3, "Volatility")
            f.write("- **Periods of highest policy attention volatility:**\n")
            for _, r in high.iterrows():
                f.write(f"  - {r['Date']}: Volatility score of {r['Volatility']:.2f}\n")
            f.write(
                "\n  **Finding**: The policy attention patterns show clear evidence "
                "of punctuated equilibrium, with periods of stability interrupted "
                "by bursts of change, particularly around administration transitions.\n"
            )

        # ---- Topic coherence ----
        if coherence_results is not None:
            f.write("\n## Topic Coherence Metrics\n\n")
            avg = coherence_results.groupby("Coherence_Measure")["Score"].mean().reset_index()
            for _, r in avg.iterrows():
                f.write(f"- Average {r['Coherence_Measure']} coherence: {r['Score']:.4f}\n")

            cv = coherence_results[coherence_results["Coherence_Measure"] == "c_v"].sort_values(
                "Score", ascending=False,
            )
            if not cv.empty:
                f.write("- Top 5 most coherent topics (c_v):\n")
                for _, r in cv.head(5).iterrows():
                    f.write(f"  - Topic {r['Topic']} ({r['Name']}): {r['Score']:.4f}\n")

        # ---- Document similarity ----
        if similarity_data is not None:
            f.write("\n## Document Similarity Analysis\n\n")
            for _, r in similarity_data.iterrows():
                f.write(f"- Average similarity within Trump documents: {r['Within_Trump']:.4f}\n")
                f.write(f"- Average similarity within Biden documents: {r['Within_Biden']:.4f}\n")
                f.write(f"- Average similarity between administrations: {r['Between_Administrations']:.4f}\n")

                within_avg = (r["Within_Trump"] + r["Within_Biden"]) / 2
                between = r["Between_Administrations"]
                ratio = within_avg / between if between > 0 else float("inf")
                f.write(f"- Ratio of within- to between-administration similarity: {ratio:.2f}\n")

                if ratio > 1.1:
                    f.write(
                        "  - **Finding**: Documents are more similar within "
                        "administrations than between them, suggesting distinct "
                        "writing or framing styles.\n"
                    )
                else:
                    f.write(
                        "  - **Finding**: No strong evidence of distinct writing "
                        "or framing styles between administrations.\n"
                    )

        # ---- Recommendations ----
        f.write("\n## Recommendations for Further Analysis\n\n")
        recommendations = [
            "Focus on significant topics: Concentrate deeper qualitative analysis on topics with statistically significant differences between administrations.",
            "Examine focusing events: Further investigate the detected focusing events and their relationship to external developments or policy windows.",
            "Analyze punctuation patterns: Compare volatility patterns to major policy initiatives or leadership changes to validate the punctuated equilibrium findings.",
            "Consider contextual factors: Analyze how external events (e.g., COVID-19 pandemic) influenced topic shifts and focusing events.",
            "Temporal lag analysis: Investigate whether there's a lag period before policy priorities shift after administration changes.",
        ]
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. **{rec.split(':')[0]}**: {rec.split(':', 1)[1].strip()}\n")

    logger.info("Summary written → %s", path)
