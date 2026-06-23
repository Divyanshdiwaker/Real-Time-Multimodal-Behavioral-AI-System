from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

from .visualizer import plot_metric

import os
from datetime import datetime
import numpy as np


def generate_timeline_insights(stress, align, confidence):

    insights = []

    n = len(stress)
    if n < 10:
        return ["Not enough data for timeline analysis"]

    # ---------------- SEGMENTS ----------------
    start = slice(0, n//3)
    mid = slice(n//3, 2*n//3)
    end = slice(2*n//3, n)

    def avg(arr, sl):
        return round(float(arr[sl].mean()), 3)

    insights.append(
        f"Start Phase: Stress={avg(stress, start)}, "
        f"Incongruence={avg(align, start)}, Confidence={avg(confidence, start)}"
    )

    insights.append(
        f"Mid Phase: Stress={avg(stress, mid)}, "
        f"Incongruence={avg(align, mid)}, Confidence={avg(confidence, mid)}"
    )

    insights.append(
        f"End Phase: Stress={avg(stress, end)}, "
        f"Incongruence={avg(align, end)}, Confidence={avg(confidence, end)}"
    )

    # ---------------- PEAK EVENTS ----------------
    stress_peaks = (stress > 0.75).nonzero()[0]
    align_peaks = (align > 0.75).nonzero()[0]
    conf_drops = (confidence < 0.3).nonzero()[0]

    if len(stress_peaks) > 0:
        insights.append(f"Stress spikes detected around frames {stress_peaks[:5].tolist()}")

    if len(align_peaks) > 0:
        insights.append(f"Incongruence peaks detected around frames {align_peaks[:5].tolist()}")

    if len(conf_drops) > 0:
        insights.append(f"Confidence drops detected around frames {conf_drops[:5].tolist()}")

    # ---------------- CRITICAL MOMENTS ----------------
    critical = ((stress > 0.7) & (align > 0.7)).nonzero()[0]

    if len(critical) > 0:
        insights.append(
            f"Critical moments detected where stress and incongruence are both high "
            f"(frames {critical[:5].tolist()})"
        )

    return insights


def generate_report(session):

    if not session or len(session["data"]) == 0:
        print("No data to generate report")
        return

    data = session["data"]

    # ---------------- Extract Data ----------------
    stress = np.array([d["stress_score"] for d in data])
    align = np.array([d["align_score"] for d in data])
    confidence = np.array([d["confidence"] for d in data])
    timeline_insights = generate_timeline_insights(stress, align, confidence)

    # ---------------- Graphs ----------------
    stress_img = plot_metric(stress, "Stress Trend")
    align_img = plot_metric(align, "Behavioral Incongruence")
    conf_img = plot_metric(confidence, "Confidence")

    # Close buffers after PDF is built — done via finally block below

    # ---------------- Stats ----------------
    avg_stress = np.mean(stress)
    avg_align = np.mean(align)
    avg_conf = np.mean(confidence)

    max_stress = np.max(stress)
    max_align = np.max(align)

    stress_peaks = np.sum(stress > 0.7)
    align_peaks = np.sum(align > 0.7)

    # ---------------- Risk Score ----------------
    risk_score = (0.5 * avg_stress) + (0.5 * avg_align)

    if risk_score > 0.7:
        risk_level = "HIGH RISK"
    elif risk_score > 0.4:
        risk_level = "MODERATE RISK"
    else:
        risk_level = "LOW RISK"

    # ---------------- Insights ----------------
    insights = []

    if avg_stress > 0.6:
        insights.append("Consistently high stress detected throughout the session.")
    elif avg_stress > 0.3:
        insights.append("Moderate stress levels observed.")
    else:
        insights.append("Low stress levels detected.")

    if stress_peaks > len(stress) * 0.2:
        insights.append("Frequent stress spikes detected, indicating unstable emotional state.")

    if avg_align > 0.6:
        insights.append("High behavioral incongruence detected, possible inconsistency between modalities.")
    elif avg_align > 0.3:
        insights.append("Moderate incongruence observed.")
    else:
        insights.append("Behavior appears consistent.")

    if align_peaks > len(align) * 0.2:
        insights.append("Multiple peaks in incongruence suggest possible deceptive or conflicted responses.")

    if avg_conf < 0.4:
        insights.append("Low confidence levels detected across session.")

    # ---------------- PDF ----------------
    os.makedirs("reports", exist_ok=True)
    filename = f"reports/session_{session['id'][:8]}.pdf"

    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()

    elements = []

    # Title
    elements.append(Paragraph("Behavioral Analysis Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    # Metadata
    elements.append(Paragraph(f"<b>Session ID:</b> {session['id']}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Start Time:</b> {session['start_time']}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Generated At:</b> {datetime.now()}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    # Graphs
    elements.append(Paragraph("Stress Analysis", styles["Heading2"]))
    elements.append(Image(stress_img, width=420, height=220))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("Behavioral Incongruence Analysis", styles["Heading2"]))
    elements.append(Image(align_img, width=420, height=220))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("Confidence Analysis", styles["Heading2"]))
    elements.append(Image(conf_img, width=420, height=220))
    elements.append(Spacer(1, 20))

    # Summary Stats
    elements.append(Paragraph("Summary Statistics", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"Average Stress: {round(avg_stress, 3)}", styles["Normal"]))
    elements.append(Paragraph(f"Peak Stress: {round(max_stress, 3)}", styles["Normal"]))
    elements.append(Paragraph(f"Stress Spikes: {stress_peaks}", styles["Normal"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"Average Incongruence: {round(avg_align, 3)}", styles["Normal"]))
    elements.append(Paragraph(f"Peak Incongruence: {round(max_align, 3)}", styles["Normal"]))
    elements.append(Paragraph(f"Incongruence Spikes: {align_peaks}", styles["Normal"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"Average Confidence: {round(avg_conf, 3)}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    # ✅ Behavioral Insights — now correctly rendered under its own heading
    elements.append(Paragraph("Behavioral Insights", styles["Heading2"]))
    elements.append(Spacer(1, 10))
    for insight in insights:  # ✅ insights loop is now here
        elements.append(Paragraph(f"- {insight}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    # ✅ Timeline Analysis — now has only timeline insights under it
    elements.append(Paragraph("Timeline Analysis", styles["Heading2"]))
    elements.append(Spacer(1, 10))
    for insight in timeline_insights:  # ✅ timeline loop is now here
        elements.append(Paragraph(f"- {insight}", styles["Normal"]))
    elements.append(Spacer(1, 20))

    # AI Summary
    elements.append(Paragraph("AI Behavioral Summary", styles["Heading2"]))
    elements.append(Spacer(1, 10))

    summary = []

    if np.mean(stress) > 0.6:
        summary.append("The subject exhibited consistently high stress levels.")
    elif np.mean(stress) > 0.3:
        summary.append("Moderate stress patterns were observed.")
    else:
        summary.append("Stress levels remained low.")

    if np.mean(align) > 0.6:
        summary.append("Frequent behavioral incongruence suggests possible inconsistency.")
    else:
        summary.append("Behavior appeared mostly consistent.")

    if np.mean(confidence) < 0.4:
        summary.append("Confidence remained low throughout the session.")

    elements.append(Paragraph(" ".join(summary), styles["Normal"]))
    elements.append(Spacer(1, 20))

    # Final Risk
    elements.append(Paragraph("Final Risk Assessment", styles["Heading2"]))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"Risk Score: {round(risk_score, 3)}", styles["Normal"]))
    elements.append(Paragraph(f"Risk Level: <b>{risk_level}</b>", styles["Normal"]))

    # Build PDF
    try:
        doc.build(elements)
        print(f"[REPORT GENERATED] {filename}")
    finally:
        stress_img.close()
        align_img.close()
        conf_img.close()