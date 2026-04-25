"""
ProcessGap — Business Process Bottleneck Intelligence
=======================================================
Built by Rutwik Satish | MS Engineering Management, Northeastern University
IE Background | Six Sigma DMAIC | Value Stream Mapping

WHY THIS EXISTS:
  Every company has broken processes. Most of them know it.
  What they don't know is which specific step is costing the most money,
  what it would take to fix it, and whether the fix is worth the investment.

  Process mining tools (Celonis, UiPath) cost $80K–$500K to implement.
  Management consultants charge $200K to produce the same analysis.
  Companies with 50–5,000 employees have nothing affordable.

  ProcessGap applies Value Stream Mapping and Six Sigma DMAIC logic to any
  business process — in 10 minutes, for free — and answers the question
  every operations leader actually needs answered:

    "Which step in this process is costing us the most money,
     and what happens to our bottom line if we fix it?"

THE METHODOLOGY:
  Value Stream Mapping (VSM) — classifies each step as VA / NNVA / NVA
  Bottleneck Detection — utilization analysis (demand vs. capacity per step)
  Waste Quantification — NVA cost + rework cost + wait cost = total annual waste
  Process Efficiency = Value-Added Time / Total Cycle Time × 100
  ROI Simulation — models cost recovery from targeted improvements

  This is standard Industrial Engineering methodology applied as a product.

STACK: Python · Streamlit · Plotly · Pandas · NumPy · Groq (Llama 3, free tier)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ProcessGap | Business Process Intelligence",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DESIGN SYSTEM ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="block-container"] {
    background-color: #f8f9fb !important;
    color: #1a1f2e !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid #1e293b !important;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }
[data-testid="stSidebarNav"] { display: none !important; }
h1,h2,h3,h4 { font-family: 'DM Sans', sans-serif !important; color: #0f172a !important; }

[data-testid="metric-container"] {
    background: #fff !important; border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important; padding: 16px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
[data-testid="stMetricValue"] {
    color: #0f172a !important; font-family: 'DM Mono', monospace !important;
    font-weight: 600 !important; font-size: 1.5rem !important;
}
[data-testid="stMetricLabel"] {
    color: #64748b !important; font-size: 0.72rem !important;
    text-transform: uppercase; letter-spacing: 0.08em;
}
[data-testid="stTabs"] button {
    color: #64748b !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important; background: transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #7c3aed !important; border-bottom: 2px solid #7c3aed !important;
}
[data-testid="stButton"] button {
    background: #7c3aed !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 500 !important;
}
[data-testid="stButton"] button:hover { background: #6d28d9 !important; }
.section-label {
    font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.16em;
    color: #7c3aed; font-weight: 700; margin-bottom: 6px;
}
.stat-card {
    background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 18px 20px; text-align: center;
}
.stat-value {
    font-family: 'DM Mono', monospace; font-size: 1.7rem;
    font-weight: 600; color: #0f172a; line-height: 1;
}
.stat-label {
    font-size: 0.72rem; color: #64748b; text-transform: uppercase;
    letter-spacing: 0.08em; margin-top: 6px;
}
.waste-card {
    background: #fdf4ff; border: 1px solid #e9d5ff;
    border-left: 4px solid #7c3aed; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 10px;
}
.bottleneck-card {
    background: #fef2f2; border: 1px solid #fecaca;
    border-left: 4px solid #dc2626; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 10px;
}
.ok-card {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-left: 4px solid #16a34a; border-radius: 10px;
    padding: 14px 18px; margin-bottom: 10px;
}
.ai-block {
    background: #fff; border: 1px solid #e2e8f0;
    border-left: 4px solid #7c3aed; border-radius: 10px;
    padding: 20px 24px; font-size: 0.9rem; line-height: 1.85;
    color: #1a1f2e; white-space: pre-wrap; font-family: 'DM Sans', sans-serif;
}
.method-tag {
    display: inline-block; background: #f5f3ff; color: #6d28d9;
    font-size: 0.7rem; font-family: 'DM Mono', monospace;
    padding: 2px 8px; border-radius: 4px; margin: 2px;
}
hr { border-color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

def apply_light(fig, height=None, title=None, xaxis_title=None, yaxis_title=None,
               extra_layout=None):
    """Single helper that applies all chart styling — eliminates **LIGHT conflicts."""
    layout_args = dict(
        template="plotly_white",
        paper_bgcolor="#f8f9fb",
        plot_bgcolor="#fff",
        font=dict(color="#1a1f2e", family="DM Sans"),
        margin=dict(t=40, b=44, l=12, r=12),
    )
    if height:        layout_args["height"] = height
    if title:         layout_args["title"] = title
    if xaxis_title:   layout_args["xaxis_title"] = xaxis_title
    if yaxis_title:   layout_args["yaxis_title"] = yaxis_title
    if extra_layout:  layout_args.update(extra_layout)
    fig.update_layout(**layout_args)
    fig.update_xaxes(gridcolor="#f1f5f9", linecolor="#e2e8f0", tickfont=dict(color="#64748b"))
    fig.update_yaxes(gridcolor="#f1f5f9", linecolor="#e2e8f0", tickfont=dict(color="#64748b"))
    return fig

# Keep for any remaining direct references
LIGHT = dict(template="plotly_white", paper_bgcolor="#f8f9fb", plot_bgcolor="#fff",
             font=dict(color="#1a1f2e", family="DM Sans"), margin=dict(t=40, b=44, l=12, r=12))
AXIS_STYLE = dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", tickfont=dict(color="#64748b"))

STEP_TYPES = ["Value-Added (VA)", "Necessary Non-Value-Added (NNVA)", "Non-Value-Added (NVA)"]
TYPE_COLORS = {
    "Value-Added (VA)": "#16a34a",
    "Necessary Non-Value-Added (NNVA)": "#ca8a04",
    "Non-Value-Added (NVA)": "#dc2626",
}
TYPE_SHORT = {
    "Value-Added (VA)": "VA",
    "Necessary Non-Value-Added (NNVA)": "NNVA",
    "Non-Value-Added (NVA)": "NVA",
}

# ── CALCULATIONS ──────────────────────────────────────────────────────────────
def run_analysis(df: pd.DataFrame, daily_volume: int, working_days: int,
                 hours_per_day: float) -> dict:
    """
    Core VSM + bottleneck analysis engine.
    Applies Industrial Engineering methodology:
    - Process Efficiency = VA Time / Total Cycle Time
    - Bottleneck = step with highest utilization (demand / capacity)
    - Annual waste = NVA cost + rework cost + wait cost
    - Improvement ROI = projected cost recovery from fixing bottleneck
    """
    results = []
    annual_volume = daily_volume * working_days

    for _, row in df.iterrows():
        cycle_min    = float(row["Cycle Time (min)"])
        wait_min     = float(row["Wait Time Before (min)"])
        cost_hr      = float(row["Cost per Hour ($)"])
        rework_pct   = float(row["Rework Rate (%)"])
        resources    = max(1, int(row["Resources"]))
        step_type    = row["Step Type"]

        # Capacity analysis (units per day this step can handle)
        capacity_per_resource = (hours_per_day * 60) / cycle_min if cycle_min > 0 else 9999
        total_capacity = capacity_per_resource * resources
        utilization = (daily_volume / total_capacity * 100) if total_capacity > 0 else 0

        # Annual costs
        cost_per_min = cost_hr / 60
        cycle_cost_annual = cycle_min * cost_per_min * resources * annual_volume
        wait_cost_annual  = wait_min  * cost_per_min * resources * annual_volume
        rework_cost_annual = cycle_cost_annual * (rework_pct / 100)
        total_step_cost   = cycle_cost_annual + wait_cost_annual + rework_cost_annual

        # Waste flag
        is_nva  = step_type == "Non-Value-Added (NVA)"
        is_nnva = step_type == "Necessary Non-Value-Added (NNVA)"
        waste_cost = (cycle_cost_annual if is_nva else 0) + wait_cost_annual + rework_cost_annual

        results.append({
            "Step":                row["Process Step"],
            "Type":                step_type,
            "Type_Short":          TYPE_SHORT[step_type],
            "Cycle_min":           cycle_min,
            "Wait_min":            wait_min,
            "Resources":           resources,
            "Cost_hr":             cost_hr,
            "Rework_pct":          rework_pct,
            "Utilization":         round(utilization, 1),
            "Capacity_per_day":    round(total_capacity, 1),
            "Cycle_Cost_Annual":   round(cycle_cost_annual, 0),
            "Wait_Cost_Annual":    round(wait_cost_annual, 0),
            "Rework_Cost_Annual":  round(rework_cost_annual, 0),
            "Total_Step_Cost":     round(total_step_cost, 0),
            "Waste_Cost":          round(waste_cost, 0),
            "Is_NVA":              is_nva,
            "Is_NNVA":             is_nnva,
        })

    rdf = pd.DataFrame(results)

    total_cycle_time   = rdf["Cycle_min"].sum() + rdf["Wait_min"].sum()
    va_cycle_time      = rdf.loc[rdf["Type"] == "Value-Added (VA)", "Cycle_min"].sum()
    process_efficiency = (va_cycle_time / total_cycle_time * 100) if total_cycle_time > 0 else 0

    bottleneck_idx     = rdf["Utilization"].idxmax()
    bottleneck_step    = rdf.loc[bottleneck_idx, "Step"]
    bottleneck_util    = rdf.loc[bottleneck_idx, "Utilization"]

    total_annual_cost  = rdf["Total_Step_Cost"].sum()
    total_waste_cost   = rdf["Waste_Cost"].sum()
    nva_cost           = rdf.loc[rdf["Is_NVA"], "Cycle_Cost_Annual"].sum()
    wait_cost_total    = rdf["Wait_Cost_Annual"].sum()
    rework_cost_total  = rdf["Rework_Cost_Annual"].sum()

    # Recovery potential (conservative: 70% of NVA, 60% of wait, 80% of rework)
    recovery_potential = nva_cost * 0.70 + wait_cost_total * 0.60 + rework_cost_total * 0.80

    throughput = min(rdf["Capacity_per_day"].min(), daily_volume)

    return {
        "steps":               rdf,
        "total_cycle_time":    round(total_cycle_time, 1),
        "va_cycle_time":       round(va_cycle_time, 1),
        "process_efficiency":  round(process_efficiency, 1),
        "bottleneck_step":     bottleneck_step,
        "bottleneck_util":     round(bottleneck_util, 1),
        "total_annual_cost":   round(total_annual_cost, 0),
        "total_waste_cost":    round(total_waste_cost, 0),
        "nva_cost":            round(nva_cost, 0),
        "wait_cost_total":     round(wait_cost_total, 0),
        "rework_cost_total":   round(rework_cost_total, 0),
        "recovery_potential":  round(recovery_potential, 0),
        "annual_volume":       annual_volume,
        "throughput":          round(throughput, 1),
    }

# ── DEFAULT PROCESS: Invoice Approval ────────────────────────────────────────
DEFAULT_STEPS = pd.DataFrame({
    "Process Step": [
        "Receive Invoice", "Manual Data Entry into ERP", "Dept. Manager Review",
        "Finance Validation", "Exception Handling / Query", "CFO Approval (>$10K)",
        "Payment Scheduling", "Vendor Confirmation"
    ],
    "Step Type": [
        "Necessary Non-Value-Added (NNVA)", "Non-Value-Added (NVA)", "Value-Added (VA)",
        "Value-Added (VA)", "Non-Value-Added (NVA)", "Necessary Non-Value-Added (NNVA)",
        "Value-Added (VA)", "Necessary Non-Value-Added (NNVA)"
    ],
    "Cycle Time (min)": [5, 12, 30, 20, 45, 60, 15, 8],
    "Wait Time Before (min)": [0, 60, 480, 120, 1440, 2880, 60, 120],
    "Resources": [1, 1, 1, 1, 2, 1, 1, 1],
    "Cost per Hour ($)": [35, 35, 85, 75, 55, 180, 55, 35],
    "Rework Rate (%)": [0, 8, 2, 3, 15, 1, 0.5, 1],
})

# ── VSM DIAGRAM GENERATOR (AIAG/Lean Standard) ───────────────────────────────
def generate_vsm_svg(df: pd.DataFrame, process_name: str = "Business Process") -> str:
    """
    Generates a standards-compliant Value Stream Map as SVG.

    Follows AIAG/Lean VSM conventions (Rother & Shook, Learning to See):
    - Supplier icon (top left) with factory battlements
    - Customer icon (top right) with factory battlements
    - Production Control box (center top) with MRP/scheduling info
    - Information flow arrows (straight = manual, zigzag = electronic)
    - Process boxes with attached data boxes (C/T, Wait, Resources, Rework)
    - Inventory triangles between steps with queue time
    - Striped push arrows between process steps
    - Operator icons below each step
    - Kaizen burst on the primary bottleneck step
    - Timeline zigzag at bottom (VA time up, wait/NVA time down)
    - Summary box: Production Lead Time + Value-Added Time + Efficiency
    """
    steps = df.to_dict("records")
    n = len(steps)
    if n == 0:
        return "<svg width='400' height='100'><text x='20' y='50' font-size='14' fill='#64748b'>No steps defined.</text></svg>"

    def esc(s):
        return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

    # ── LAYOUT ────────────────────────────────────────────────────────────────
    BOX_W      = 130    # process box width
    BOX_H      = 60     # process box height
    DATA_H     = 68     # data box height below process box
    ARROW_W    = 55     # push arrow width
    MARGIN_L   = 30
    MARGIN_T   = 20

    TOP_ROW_Y  = MARGIN_T + 10       # supplier/customer/production control row
    PROC_ROW_Y = TOP_ROW_Y + 170     # process boxes row
    DATA_ROW_Y = PROC_ROW_Y + BOX_H  # data boxes
    OPS_ROW_Y  = DATA_ROW_Y + DATA_H + 8  # operator icons
    TL_Y       = OPS_ROW_Y + 40     # timeline row
    TL_H       = 56
    FOOT_Y     = TL_Y + TL_H + 30

    total_width  = MARGIN_L * 2 + n * (BOX_W + ARROW_W) + 180
    total_height = FOOT_Y + 80

    # Color palette
    TYPE_BG     = {"Value-Added (VA)": "#dcfce7", "Necessary Non-Value-Added (NNVA)": "#fef9c3", "Non-Value-Added (NVA)": "#fee2e2"}
    TYPE_BORDER = {"Value-Added (VA)": "#16a34a", "Necessary Non-Value-Added (NNVA)": "#ca8a04", "Non-Value-Added (NVA)": "#dc2626"}
    TYPE_SHORT  = {"Value-Added (VA)": "VA", "Necessary Non-Value-Added (NNVA)": "NNVA", "Non-Value-Added (NVA)": "NVA"}

    # Find bottleneck (highest utilization)
    total_ct = sum(s.get("Cycle Time (min)", 0) for s in steps)
    total_wt = sum(s.get("Wait Time Before (min)", 0) for s in steps)
    va_ct    = sum(s.get("Cycle Time (min)", 0) for s in steps if s.get("Step Type","") == "Value-Added (VA)")
    lead_time = total_ct + total_wt
    efficiency = round(va_ct / lead_time * 100, 1) if lead_time > 0 else 0

    # Identify bottleneck: step with highest utilization
    max_util_idx = 0
    max_util = 0
    for i, s in enumerate(steps):
        ct = s.get("Cycle Time (min)", 1)
        res = max(1, s.get("Resources", 1))
        util = 1 / (ct * res) if ct > 0 else 0  # higher = more constrained
        if util > max_util:
            max_util = util
            max_util_idx = i

    L = []  # SVG lines collector

    L.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="{total_height}" '
             f'style="background:#ffffff;font-family:Arial,sans-serif">')

    # ── DEFS: patterns and markers ────────────────────────────────────────────
    L.append("""<defs>
  <pattern id="push_stripe" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">
    <rect width="3" height="6" fill="#374151" opacity="0.7"/>
  </pattern>
  <marker id="arr" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
    <polygon points="0 0, 8 3, 0 6" fill="#374151"/>
  </marker>
  <marker id="arr_blue" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
    <polygon points="0 0, 8 3, 0 6" fill="#2563eb"/>
  </marker>
  <marker id="arr_info" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
    <polygon points="0 0, 8 3, 0 6" fill="#475569"/>
  </marker>
</defs>""")

    # ── TITLE ─────────────────────────────────────────────────────────────────
    L.append(f'<text x="{total_width//2}" y="18" font-size="13" font-weight="700" '
             f'fill="#0f172a" text-anchor="middle">VALUE STREAM MAP — CURRENT STATE</text>')
    L.append(f'<text x="{total_width//2}" y="32" font-size="9" fill="#64748b" text-anchor="middle">'
             f'{esc(process_name)} · {n} Steps · ProcessGap</text>')

    # ── HELPER: factory icon (supplier / customer) ────────────────────────────
    def factory_icon(x, y, w, h, label, sublabel=""):
        """Standard VSM factory/building icon with battlements on top."""
        # Battlements (zigzag top) — 5 teeth
        teeth = 5
        tw = w // teeth
        batt = []
        for t in range(teeth):
            batt_x = x + t * tw
            batt.append(f"{batt_x},{y+12}")
            batt.append(f"{batt_x},{y}")
            batt.append(f"{batt_x+tw//2},{y}")
            batt.append(f"{batt_x+tw//2},{y+12}")
        batt_str = " ".join(batt)
        L.append(f'<polygon points="{x},{y+h} {x},{y+12} {batt_str} {x+w},{y+12} {x+w},{y+h}" '
                 f'fill="#f1f5f9" stroke="#334155" stroke-width="1.5" stroke-linejoin="round"/>')
        L.append(f'<text x="{x+w//2}" y="{y+h-28}" font-size="9" font-weight="700" '
                 f'fill="#0f172a" text-anchor="middle">{esc(label)}</text>')
        if sublabel:
            L.append(f'<text x="{x+w//2}" y="{y+h-16}" font-size="8" '
                     f'fill="#475569" text-anchor="middle">{esc(sublabel)}</text>')

    # ── HELPER: production control box ────────────────────────────────────────
    def prod_control_box(x, y, w, h, label="Production\nControl"):
        L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                 f'fill="#eff6ff" stroke="#2563eb" stroke-width="1.5" rx="4"/>')
        for i, line in enumerate(label.split("\n")):
            L.append(f'<text x="{x+w//2}" y="{y+22+i*14}" font-size="10" font-weight="700" '
                     f'fill="#2563eb" text-anchor="middle">{esc(line)}</text>')
        # MRP label
        L.append(f'<text x="{x+w//2}" y="{y+h-8}" font-size="8" '
                 f'fill="#64748b" text-anchor="middle">MRP / Scheduling</text>')

    # ── HELPER: process box + data box ────────────────────────────────────────
    def process_box(x, y, step, is_bottleneck=False):
        stype  = step.get("Step Type", "Value-Added (VA)")
        bg     = TYPE_BG.get(stype, "#f1f5f9")
        border = TYPE_BORDER.get(stype, "#64748b")
        short  = TYPE_SHORT.get(stype, "VA")
        name   = esc(step.get("Process Step", "Step"))
        ct     = step.get("Cycle Time (min)", 0)
        wt     = step.get("Wait Time Before (min)", 0)
        res    = step.get("Resources", 1)
        rework = step.get("Rework Rate (%)", 0)

        # Shadow
        L.append(f'<rect x="{x+2}" y="{y+2}" width="{BOX_W}" height="{BOX_H}" rx="3" fill="#e2e8f0"/>')
        # Main process box
        L.append(f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" '
                 f'rx="3" fill="{bg}" stroke="{border}" stroke-width="2"/>')

        # Type badge
        L.append(f'<rect x="{x+BOX_W-30}" y="{y+3}" width="27" height="13" '
                 f'rx="2" fill="{border}"/>')
        L.append(f'<text x="{x+BOX_W-17}" y="{y+13}" font-size="8" '
                 f'fill="white" text-anchor="middle" font-weight="700">{esc(short)}</text>')

        # Step name (two lines if needed)
        if len(name) > 15:
            words = name.split(" ")
            mid = len(words) // 2
            line1 = " ".join(words[:mid])
            line2 = " ".join(words[mid:])
            L.append(f'<text x="{x+BOX_W//2}" y="{y+22}" font-size="9" font-weight="700" '
                     f'fill="#0f172a" text-anchor="middle">{esc(line1[:18])}</text>')
            L.append(f'<text x="{x+BOX_W//2}" y="{y+34}" font-size="9" font-weight="700" '
                     f'fill="#0f172a" text-anchor="middle">{esc(line2[:18])}</text>')
        else:
            L.append(f'<text x="{x+BOX_W//2}" y="{y+28}" font-size="9" font-weight="700" '
                     f'fill="#0f172a" text-anchor="middle">{esc(name)}</text>')

        # Uptime / availability indicator
        uptime = max(0, 100 - rework * 2)
        L.append(f'<text x="{x+6}" y="{y+BOX_H-6}" font-size="8" fill="#64748b">'
                 f'Uptime: {uptime:.0f}%</text>')

        # Data box (attached below)
        dy = y + BOX_H
        L.append(f'<rect x="{x}" y="{dy}" width="{BOX_W}" height="{DATA_H}" '
                 f'fill="#f8fafc" stroke="{border}" stroke-width="1.2" stroke-dasharray="3,2"/>')
        # Data lines — standard VSM data box format
        data_lines = [
            ("C/T =", f"{ct} min"),
            ("Wait =", f"{int(wt)} min"),
            ("Res =", f"{int(res)} people"),
            ("Rework =", f"{rework}%"),
        ]
        for j, (key, val) in enumerate(data_lines):
            L.append(f'<text x="{x+5}" y="{dy+13+j*14}" font-size="8" fill="#374151">'
                     f'<tspan font-weight="700">{esc(key)}</tspan> {esc(val)}</text>')

        # Kaizen burst on bottleneck
        if is_bottleneck:
            kx, ky = x + BOX_W - 22, y - 22
            burst_pts = []
            import math
            for a in range(0, 360, 20):
                r = 18 if a % 40 == 0 else 12
                bx = kx + r * math.cos(math.radians(a))
                by = ky + r * math.sin(math.radians(a))
                burst_pts.append(f"{bx:.1f},{by:.1f}")
            L.append(f'<polygon points="{" ".join(burst_pts)}" '
                     f'fill="#fef08a" stroke="#ca8a04" stroke-width="1.5"/>')
            L.append(f'<text x="{kx}" y="{ky-2}" font-size="7" fill="#92400e" '
                     f'text-anchor="middle" font-weight="700">KAIZEN!</text>')
            L.append(f'<text x="{kx}" y="{ky+8}" font-size="7" fill="#92400e" '
                     f'text-anchor="middle">Bottleneck</text>')

    # ── HELPER: operator icon ─────────────────────────────────────────────────
    def operator_icon(cx, y, count=1):
        """Standard VSM operator icon: circle head + stick body."""
        L.append(f'<circle cx="{cx}" cy="{y+6}" r="6" fill="none" stroke="#7c3aed" stroke-width="1.5"/>')
        L.append(f'<line x1="{cx}" y1="{y+12}" x2="{cx}" y2="{y+26}" stroke="#7c3aed" stroke-width="1.5"/>')
        L.append(f'<line x1="{cx-7}" y1="{y+18}" x2="{cx+7}" y2="{y+18}" stroke="#7c3aed" stroke-width="1.5"/>')
        L.append(f'<line x1="{cx}" y1="{y+26}" x2="{cx-5}" y2="{y+36}" stroke="#7c3aed" stroke-width="1.5"/>')
        L.append(f'<line x1="{cx}" y1="{y+26}" x2="{cx+5}" y2="{y+36}" stroke="#7c3aed" stroke-width="1.5"/>')
        if count > 1:
            L.append(f'<text x="{cx+10}" y="{y+20}" font-size="9" fill="#7c3aed" font-weight="700">×{count}</text>')

    # ── HELPER: push arrow (striped, standard VSM) ────────────────────────────
    def push_arrow(x, y, w, h=18):
        cy = y + h // 2
        L.append(f'<rect x="{x}" y="{y}" width="{w-10}" height="{h}" '
                 f'fill="url(#push_stripe)" stroke="#374151" stroke-width="1"/>')
        L.append(f'<polygon points="{x+w-10},{y} {x+w},{cy} {x+w-10},{y+h}" fill="#374151"/>')

    # ── HELPER: inventory triangle ────────────────────────────────────────────
    def inv_triangle(cx, y, qty_label):
        h = 20
        w = 24
        L.append(f'<polygon points="{cx},{y} {cx-w//2},{y+h} {cx+w//2},{y+h}" '
                 f'fill="#fef9c3" stroke="#ca8a04" stroke-width="1.5"/>')
        L.append(f'<text x="{cx}" y="{y+h+11}" font-size="8" fill="#92400e" '
                 f'text-anchor="middle" font-weight="600">{esc(qty_label)}</text>')
        # I label inside
        L.append(f'<text x="{cx}" y="{y+h-4}" font-size="9" fill="#92400e" '
                 f'text-anchor="middle" font-style="italic" font-weight="700">I</text>')

    # ── HELPER: straight info flow arrow ─────────────────────────────────────
    def info_arrow_straight(x1, y1, x2, y2, label=""):
        L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                 f'stroke="#475569" stroke-width="1.5" marker-end="url(#arr_info)"/>')
        if label:
            mx, my = (x1+x2)//2, (y1+y2)//2
            L.append(f'<rect x="{mx-20}" y="{my-9}" width="40" height="12" '
                     f'rx="2" fill="white" opacity="0.8"/>')
            L.append(f'<text x="{mx}" y="{my}" font-size="8" fill="#374151" '
                     f'text-anchor="middle">{esc(label)}</text>')

    # ── HELPER: electronic info flow (zigzag arrow) ───────────────────────────
    def info_arrow_zigzag(x1, y1, x2, y2, label=""):
        mx = (x1 + x2) // 2
        my = (y1 + y2) // 2
        amp = 12
        path = f"M {x1} {y1} "
        segs = 6
        for s in range(segs):
            t = s / segs
            t2 = (s+1) / segs
            px1 = x1 + t * (x2 - x1)
            py1 = y1 + t * (y2 - y1) + (amp if s % 2 == 0 else -amp)
            px2 = x1 + t2 * (x2 - x1)
            py2 = y1 + t2 * (y2 - y1) + (-amp if s % 2 == 0 else amp)
            path += f"L {px1:.0f} {py1:.0f} L {px2:.0f} {py2:.0f} "
        path += f"L {x2} {y2}"
        L.append(f'<path d="{path}" fill="none" stroke="#2563eb" stroke-width="1.5" '
                 f'marker-end="url(#arr_blue)"/>')
        if label:
            L.append(f'<rect x="{mx-22}" y="{my-9}" width="44" height="12" '
                     f'rx="2" fill="white" opacity="0.9"/>')
            L.append(f'<text x="{mx}" y="{my}" font-size="8" fill="#2563eb" '
                     f'text-anchor="middle">{esc(label)}</text>')

    # ══════════════════════════════════════════════════════════════════════════
    # DRAW TOP ROW: Supplier → Production Control → Customer
    # ══════════════════════════════════════════════════════════════════════════
    FACTORY_W = 110
    FACTORY_H = 70
    PC_W      = 130
    PC_H      = 65

    # Supplier (top left)
    sup_x = MARGIN_L
    sup_y = TOP_ROW_Y + 20
    factory_icon(sup_x, sup_y, FACTORY_W, FACTORY_H, "SUPPLIER / TRIGGER", "Process Input")

    # Production Control (center)
    pc_x = (total_width - PC_W) // 2
    pc_y = TOP_ROW_Y + 20
    prod_control_box(pc_x, pc_y, PC_W, PC_H)

    # Customer (top right)
    cust_x = total_width - FACTORY_W - MARGIN_L
    cust_y = TOP_ROW_Y + 20
    factory_icon(cust_x, cust_y, FACTORY_W, FACTORY_H, "CUSTOMER / END USER", "Process Output")

    # Information flows: Supplier ↔ Production Control (electronic)
    info_arrow_zigzag(sup_x + FACTORY_W, sup_y + FACTORY_H//2,
                      pc_x, pc_y + PC_H//2, "Demand / Forecast")
    # Production Control → Customer (manual)
    info_arrow_straight(pc_x + PC_W, pc_y + PC_H//2,
                        cust_x, cust_y + FACTORY_H//2, "Daily Order")

    # Production Control → Process flow (scheduling arrows down)
    pc_mid_x = pc_x + PC_W // 2
    sched_y2 = PROC_ROW_Y - 10
    L.append(f'<line x1="{pc_mid_x}" y1="{pc_y+PC_H}" x2="{pc_mid_x}" y2="{sched_y2}" '
             f'stroke="#2563eb" stroke-width="1.5" stroke-dasharray="5,3" marker-end="url(#arr_blue)"/>')
    L.append(f'<text x="{pc_mid_x+5}" y="{(pc_y+PC_H+sched_y2)//2}" font-size="8" fill="#2563eb">'
             f'Work Orders</text>')

    # ══════════════════════════════════════════════════════════════════════════
    # DRAW PROCESS ROW: Steps + Inventory + Push arrows
    # ══════════════════════════════════════════════════════════════════════════
    proc_xs = []  # x position of each process box
    for i, step in enumerate(steps):
        px = MARGIN_L + i * (BOX_W + ARROW_W)
        proc_xs.append(px)

        # Inventory triangle BEFORE this step (wait time)
        if i > 0:
            wt = step.get("Wait Time Before (min)", 0)
            tri_cx = px - ARROW_W // 2
            if wt >= 60:
                wt_label = f"{wt/60:.1f}h"
            else:
                wt_label = f"{int(wt)}m"
            inv_triangle(tri_cx, PROC_ROW_Y + BOX_H//2 - 10, wt_label)

            # Push arrow
            push_x = px - ARROW_W + 4
            push_y = PROC_ROW_Y + BOX_H//2 - 9
            push_arrow(push_x, push_y, ARROW_W - 8)

        # Process box + data box
        is_bn = (i == max_util_idx)
        process_box(px, PROC_ROW_Y, step, is_bottleneck=is_bn)

        # Operator icon
        op_cx = px + BOX_W // 2
        res = max(1, int(step.get("Resources", 1)))
        operator_icon(op_cx, OPS_ROW_Y, count=res)

    # Arrow from last step to Customer
    last_x = proc_xs[-1] + BOX_W + 8 if proc_xs else MARGIN_L
    cust_arr_y = PROC_ROW_Y + BOX_H // 2
    if last_x < cust_x:
        push_arrow(last_x, cust_arr_y - 9, cust_x - last_x - 4)
    L.append(f'<line x1="{cust_x}" y1="{cust_y+FACTORY_H}" x2="{cust_x+FACTORY_W//2}" '
             f'y2="{PROC_ROW_Y+BOX_H//2}" stroke="#334155" stroke-width="1.5" '
             f'stroke-dasharray="4,3" marker-end="url(#arr_info)"/>')

    # Supplier delivery arrow to first step
    del_y = PROC_ROW_Y + BOX_H//2
    L.append(f'<line x1="{sup_x+FACTORY_W}" y1="{sup_y+FACTORY_H//2}" '
             f'x2="{proc_xs[0]}" y2="{del_y}" stroke="#334155" stroke-width="1.5" '
             f'marker-end="url(#arr_info)"/>')

    # ══════════════════════════════════════════════════════════════════════════
    # TIMELINE ZIGZAG (standard VSM bottom timeline)
    # VA time = peaks UP, Wait/NVA time = peaks DOWN
    # ══════════════════════════════════════════════════════════════════════════
    tl_baseline = TL_Y + TL_H // 2
    tl_up       = TL_Y + 6
    tl_down     = TL_Y + TL_H - 6
    tl_x        = MARGIN_L

    # Timeline label
    L.append(f'<text x="{MARGIN_L}" y="{TL_Y - 6}" font-size="10" font-weight="700" '
             f'fill="#0f172a">Lead Time Timeline</text>')
    L.append(f'<text x="{MARGIN_L + 140}" y="{TL_Y - 6}" font-size="8" fill="#64748b">'
             f'▲ Cycle Time (VA=green, NVA=red)   ▼ Wait / Queue Time</text>')

    # Baseline
    L.append(f'<line x1="{MARGIN_L}" y1="{tl_baseline}" x2="{total_width - MARGIN_L}" '
             f'y2="{tl_baseline}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="3,3"/>')

    # Build zigzag points list + time labels
    max_time = max((s.get("Cycle Time (min)", 0) + s.get("Wait Time Before (min)", 0)) for s in steps) or 1
    total_tl_width = total_width - MARGIN_L * 2 - 200
    scale = total_tl_width / (lead_time if lead_time > 0 else 1)

    pts = [(tl_x, tl_baseline)]
    time_labels = []

    for i, step in enumerate(steps):
        ct   = step.get("Cycle Time (min)", 0)
        wt   = step.get("Wait Time Before (min)", 0) if i > 0 else 0
        stype = step.get("Step Type", "Value-Added (VA)")
        ct_color = "#16a34a" if stype == "Value-Added (VA)" else ("#dc2626" if "NVA" in stype else "#ca8a04")

        # Wait segment — peak DOWN
        if wt > 0:
            ww = max(20, int(wt * scale))
            pts.append((tl_x + ww // 2, tl_down))
            pts.append((tl_x + ww, tl_baseline))
            wt_str = f"{int(wt)}m" if wt < 60 else f"{wt/60:.1f}h"
            time_labels.append({"x": tl_x + ww//2, "y": tl_down + 14, "text": wt_str, "color": "#dc2626"})
            tl_x += ww

        # Cycle time segment — peak UP
        if ct > 0:
            cw = max(24, int(ct * scale))
            pts.append((tl_x + cw // 2, tl_up))
            pts.append((tl_x + cw, tl_baseline))
            ct_str = f"{ct}m"
            time_labels.append({"x": tl_x + cw//2, "y": tl_up - 4, "text": ct_str, "color": ct_color})
            tl_x += cw

    # Draw zigzag polyline
    if len(pts) > 1:
        pts_str = " ".join(f"{p[0]:.0f},{p[1]:.0f}" for p in pts)
        L.append(f'<polyline points="{pts_str}" fill="none" stroke="#7c3aed" '
                 f'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>')

    # Time labels
    for t in time_labels:
        tx = t['x']
        ty = t['y']
        tc = t['color']
        tt = t['text']
        L.append(f'<text x="{tx:.0f}" y="{ty}" font-size="8" fill="{tc}" '
                 f'text-anchor="middle" font-weight="700">{esc(tt)}</text>')

    # ══════════════════════════════════════════════════════════════════════════
    # SUMMARY BOX (bottom right) — Production Lead Time + VA Time + Efficiency
    # ══════════════════════════════════════════════════════════════════════════
    sb_x = total_width - 185 - MARGIN_L
    sb_y = TL_Y
    sb_w = 185
    sb_h = TL_H + 20

    L.append(f'<rect x="{sb_x}" y="{sb_y}" width="{sb_w}" height="{sb_h}" '
             f'rx="5" fill="#f5f3ff" stroke="#7c3aed" stroke-width="2"/>')
    L.append(f'<rect x="{sb_x}" y="{sb_y}" width="{sb_w}" height="20" '
             f'rx="5" fill="#7c3aed"/>')
    L.append(f'<text x="{sb_x+sb_w//2}" y="{sb_y+14}" font-size="10" font-weight="700" '
             f'fill="white" text-anchor="middle">PROCESS SUMMARY</text>')

    lt_str = f"{lead_time:.0f} min" if lead_time < 60 else f"{lead_time/60:.1f} hrs"
    va_str = f"{va_ct:.0f} min" if va_ct < 60 else f"{va_ct/60:.1f} hrs"
    waste_str = f"{lead_time-va_ct:.0f} min" if (lead_time-va_ct) < 60 else f"{(lead_time-va_ct)/60:.1f} hrs"

    eff_color = "#16a34a" if efficiency >= 70 else ("#ca8a04" if efficiency >= 40 else "#dc2626")
    summary_rows = [
        ("Production Lead Time:", lt_str, "#374151"),
        ("Value-Added Time:", va_str, "#16a34a"),
        ("Waste Time:", waste_str, "#dc2626"),
        ("Process Efficiency:", f"{efficiency}%", eff_color),
    ]
    for j, (k, v, c) in enumerate(summary_rows):
        L.append(f'<text x="{sb_x+8}" y="{sb_y+36+j*17}" font-size="9" fill="#374151">'
                 f'<tspan font-weight="700">{esc(k)}</tspan></text>')
        L.append(f'<text x="{sb_x+sb_w-8}" y="{sb_y+36+j*17}" font-size="9" '
                 f'fill="{c}" text-anchor="end" font-weight="700">{esc(v)}</text>')

    # ══════════════════════════════════════════════════════════════════════════
    # LEGEND
    # ══════════════════════════════════════════════════════════════════════════
    leg_y = FOOT_Y + 10
    L.append(f'<text x="{MARGIN_L}" y="{leg_y}" font-size="9" font-weight="700" fill="#374151">LEGEND:</text>')

    legend_items = [
        ("#dcfce7","#16a34a","Value-Added (VA)"),
        ("#fef9c3","#ca8a04","Necessary NVA"),
        ("#fee2e2","#dc2626","Non-Value-Added (NVA)"),
        ("#fef08a","#ca8a04","Kaizen Burst (Bottleneck)"),
        ("#f1f5f9","#334155","Supplier / Customer"),
        ("#eff6ff","#2563eb","Production Control"),
    ]
    lx = MARGIN_L + 60
    for bg, border, label in legend_items:
        L.append(f'<rect x="{lx}" y="{leg_y-9}" width="14" height="10" '
                 f'rx="2" fill="{bg}" stroke="{border}" stroke-width="1.2"/>')
        L.append(f'<text x="{lx+17}" y="{leg_y}" font-size="8.5" fill="#374151">{esc(label)}</text>')
        lx += len(label) * 6.5 + 32

    L.append("</svg>")
    return "\n".join(L)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding:8px 0 16px">
  <div style="font-size:0.62rem;text-transform:uppercase;letter-spacing:0.18em;
       color:#a78bfa;font-weight:700;margin-bottom:4px">PROCESS INTELLIGENCE</div>
  <div style="font-size:1.35rem;font-weight:600;color:#f1f5f9">ProcessGap</div>
  <div style="font-size:0.78rem;color:#64748b;margin-top:2px">
       Bottleneck Detection & ROI Analysis</div>
</div>
""", unsafe_allow_html=True)

    module = st.radio(
        "View",
        ["🏠  Overview", "🗺️  Process Builder", "🗂️  VSM Diagram",
         "📊  VSM Analysis", "🔴  Bottleneck Breakdown",
         "💰  ROI Simulator", "🤖  AI Process Brief"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown('<div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.1em;color:#475569;margin-bottom:8px">PROCESS SETTINGS</div>', unsafe_allow_html=True)
    daily_volume  = st.number_input("Daily Process Volume (units/transactions)", min_value=1, value=50)
    working_days  = st.number_input("Working Days per Year", min_value=1, value=250)
    hours_per_day = st.number_input("Working Hours per Day", min_value=1.0, value=8.0, step=0.5)

    st.markdown("---")
    st.markdown("""
<div style="font-size:0.72rem;color:#475569;line-height:1.7">
<div style="color:#94a3b8;font-size:0.62rem;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px">METHODOLOGY</div>
<span style="color:#a78bfa">VSM</span> Value Stream Mapping<br>
<span style="color:#a78bfa">DMAIC</span> Six Sigma process analysis<br>
<span style="color:#a78bfa">Bottleneck</span> Utilization-based detection<br>
<span style="color:#a78bfa">Waste</span> NVA + Wait + Rework cost<br>
<span style="color:#a78bfa">ROI</span> Conservative recovery model
</div>
""", unsafe_allow_html=True)

# ── PROCESS DATA ──────────────────────────────────────────────────────────────
if "process_df" not in st.session_state:
    st.session_state.process_df = DEFAULT_STEPS.copy()

# ══════════════════════════════════════════════════════════════════════════════
# MODULE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in module:

    st.markdown("""
<div style="padding:20px 0 8px">
  <div class="section-label">BUSINESS PROCESS INTELLIGENCE</div>
  <h1 style="font-size:2rem;font-weight:600;margin:0;letter-spacing:-0.02em">ProcessGap</h1>
  <p style="color:#64748b;font-size:0.9rem;margin-top:6px;max-width:680px">
    Find the step in your business process that's costing you the most money.
    Quantify it. Fix it. Measure the ROI — without a $200K consulting engagement.
  </p>
</div>
""", unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
<div class="bottleneck-card">
  <div class="section-label" style="color:#dc2626">THE PROBLEM</div>
  <p style="font-size:0.87rem;line-height:1.75;color:#374151;margin:0">
    Every company has broken processes. Most know which ones feel slow.
    What nobody knows is <strong>which specific step costs the most money</strong>
    and what fixing it is actually worth. Process mining tools (Celonis, UiPath)
    cost $80K–$500K to implement. Management consultants charge $200K for the
    same analysis. Companies between $5M and $500M in revenue have nothing.
    <br><br>
    The result: the same bottlenecks persist for years because nobody
    has quantified them well enough to justify the fix.
  </p>
</div>
""", unsafe_allow_html=True)

    with col_r:
        st.markdown("""
<div class="ok-card">
  <div class="section-label" style="color:#16a34a">THE SOLUTION</div>
  <p style="font-size:0.87rem;line-height:1.75;color:#374151;margin:0">
    ProcessGap applies Value Stream Mapping and Six Sigma DMAIC logic to any
    business process — in 10 minutes. Map your steps, classify waste,
    and get back a ranked list of bottlenecks with their annual dollar cost.
    <br><br>
    • <strong>VSM Classification</strong> — VA / NNVA / NVA per step<br>
    • <strong>Bottleneck Detection</strong> — utilization-based, not guesswork<br>
    • <strong>Waste Quantification</strong> — NVA + Wait + Rework in dollars<br>
    • <strong>ROI Simulation</strong> — what fixing each bottleneck is worth<br>
    • <strong>AI Brief</strong> — plain English for leadership in 30 seconds
  </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # Run analysis on default data
    res = run_analysis(st.session_state.process_df, daily_volume, working_days, hours_per_day)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Process Steps",        len(st.session_state.process_df))
    c2.metric("Process Efficiency",   f"{res['process_efficiency']}%",
              delta="Target: >80%" if res['process_efficiency'] < 80 else "Healthy",
              delta_color="inverse" if res['process_efficiency'] < 80 else "normal")
    c3.metric("Annual Process Cost",  f"${res['total_annual_cost']:,.0f}")
    c4.metric("Annual Waste Cost",    f"${res['total_waste_cost']:,.0f}",
              delta="Recoverable", delta_color="inverse")
    c5.metric("Recovery Potential",   f"${res['recovery_potential']:,.0f}",
              delta="Conservative est.", delta_color="normal")

    st.markdown("---")

    st.markdown('<div class="section-label">HOW IT WORKS</div>', unsafe_allow_html=True)
    m1,m2,m3,m4,m5 = st.columns(5)
    steps_info = [
        ("01", "Map Your Process", "Enter each step with its cycle time, wait time, resources, and cost. Classify each as VA, NNVA, or NVA."),
        ("02", "VSM Analysis", "ProcessGap calculates total cycle time, value-added ratio, and process efficiency using VSM methodology."),
        ("03", "Bottleneck Detection", "Utilization analysis identifies the step where demand exceeds capacity — your real constraint."),
        ("04", "ROI Simulation", "Model the impact of fixing each bottleneck. See exactly what the improvement is worth annually."),
        ("05", "AI Brief", "Groq AI generates a plain-English recommendation structured for a VP or COO — actionable in 60 seconds."),
    ]
    for col,(num,title,desc) in zip([m1,m2,m3,m4,m5], steps_info):
        col.markdown(f"""
<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px 14px">
  <div style="font-family:'DM Mono',monospace;font-size:1.6rem;font-weight:500;
       color:#7c3aed;line-height:1">{num}</div>
  <div style="font-weight:600;font-size:0.88rem;color:#0f172a;margin-top:6px">{title}</div>
  <div style="font-size:0.79rem;color:#64748b;line-height:1.6;margin-top:4px">{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;
     padding:12px 18px;font-size:0.81rem;color:#64748b">
<strong style="color:#1a1f2e">Demo loaded:</strong> Invoice Approval Process — 8 steps,
a classic high-waste back-office workflow. Edit in the Process Builder tab or load your own process.
<br>
<span style="color:#7c3aed">Methodology tags:</span>
<span class="method-tag">Value Stream Mapping</span>
<span class="method-tag">Six Sigma DMAIC</span>
<span class="method-tag">Utilization Analysis</span>
<span class="method-tag">Little's Law</span>
<span class="method-tag">Lean NVA Classification</span>
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: PROCESS BUILDER
# ══════════════════════════════════════════════════════════════════════════════
elif "Builder" in module:
    st.markdown('<div class="section-label">STEP 1</div>', unsafe_allow_html=True)
    st.markdown("## Process Builder")

    st.markdown("""
<div style="background:#fff;border:1px solid #e2e8f0;border-left:3px solid #7c3aed;
     border-radius:10px;padding:14px 20px;font-size:0.84rem;line-height:1.7;
     color:#374151;margin-bottom:16px">
<strong>How to fill this in:</strong> Add one row per process step in execution order.
<strong>Cycle Time</strong> = active working time per unit (not waiting).
<strong>Wait Time Before</strong> = queue time before this step starts (this is where hidden waste lives).
<strong>Step Type:</strong> VA = directly adds customer value | NNVA = required but non-value-adding (compliance, approvals) |
NVA = pure waste (re-entry, rework, redundant checks).
<strong>Rework Rate</strong> = % of units that must be reprocessed at this step.
</div>
""", unsafe_allow_html=True)

    edited_df = st.data_editor(
        st.session_state.process_df,
        num_rows="dynamic",
        width='stretch',
        column_config={
            "Process Step":         st.column_config.TextColumn("Process Step", width=200),
            "Step Type":            st.column_config.SelectboxColumn("Step Type", options=STEP_TYPES, width=260),
            "Cycle Time (min)":     st.column_config.NumberColumn("Cycle Time (min)", min_value=0.1, step=0.5),
            "Wait Time Before (min)": st.column_config.NumberColumn("Wait Before (min)", min_value=0, step=1),
            "Resources":            st.column_config.NumberColumn("Resources (people)", min_value=1, step=1),
            "Cost per Hour ($)":    st.column_config.NumberColumn("Cost/Hour ($)", min_value=1, step=5),
            "Rework Rate (%)":      st.column_config.NumberColumn("Rework Rate (%)", min_value=0.0, max_value=100.0, step=0.5),
        }
    )
    st.session_state.process_df = edited_df

    col1, col2 = st.columns([1,4])
    with col1:
        if st.button("Run Analysis →", type="primary"):
            st.success("✅ Process saved. Navigate to VSM Analysis to see results.")

    st.markdown("---")
    st.markdown("### Load a Different Example Process")
    example = st.selectbox("Load Example", [
        "Invoice Approval (default)",
        "New Employee Onboarding",
        "Customer Support Ticket Resolution",
        "Purchase Order Approval",
        "Product Return Processing",
    ])

    examples = {
        "New Employee Onboarding": pd.DataFrame({
            "Process Step": ["Job Offer Sent","Background Check","IT Setup Request",
                             "Equipment Provisioning","Systems Access Grant",
                             "Orientation Session","Manager Intro Meeting","30-Day Check-In"],
            "Step Type": ["Value-Added (VA)","Necessary Non-Value-Added (NNVA)",
                          "Non-Value-Added (NVA)","Necessary Non-Value-Added (NNVA)",
                          "Non-Value-Added (NVA)","Value-Added (VA)","Value-Added (VA)",
                          "Value-Added (VA)"],
            "Cycle Time (min)": [30,5,15,45,20,240,60,30],
            "Wait Time Before (min)": [0,1440,2880,4320,1440,0,60,21600],
            "Resources": [1,1,2,2,2,1,1,1],
            "Cost per Hour ($)": [80,45,65,45,65,60,80,80],
            "Rework Rate (%)": [1,0,5,3,8,0,0,0],
        }),
        "Customer Support Ticket Resolution": pd.DataFrame({
            "Process Step": ["Ticket Received","Auto-Classification","Tier 1 Review",
                             "Escalation to Tier 2","Investigation","Resolution Draft",
                             "Customer Approval","Ticket Close & Log"],
            "Step Type": ["Necessary Non-Value-Added (NNVA)","Non-Value-Added (NVA)",
                          "Value-Added (VA)","Non-Value-Added (NVA)","Value-Added (VA)",
                          "Value-Added (VA)","Necessary Non-Value-Added (NNVA)",
                          "Non-Value-Added (NVA)"],
            "Cycle Time (min)": [2,1,20,10,45,30,5,5],
            "Wait Time Before (min)": [0,5,60,240,120,30,1440,60],
            "Resources": [1,1,2,1,2,1,1,1],
            "Cost per Hour ($)": [40,0,45,55,65,55,45,40],
            "Rework Rate (%)": [0,15,5,3,10,8,2,1],
        }),
        "Purchase Order Approval": pd.DataFrame({
            "Process Step": ["PR Submitted","Budget Check","Manager Approval",
                             "Procurement Review","Vendor Quote Collection",
                             "Finance Sign-off","PO Generation","Vendor Notification"],
            "Step Type": ["Necessary Non-Value-Added (NNVA)","Non-Value-Added (NVA)",
                          "Value-Added (VA)","Necessary Non-Value-Added (NNVA)",
                          "Value-Added (VA)","Necessary Non-Value-Added (NNVA)",
                          "Non-Value-Added (NVA)","Value-Added (VA)"],
            "Cycle Time (min)": [5,10,25,20,60,30,10,5],
            "Wait Time Before (min)": [0,240,480,120,1440,2880,60,30],
            "Resources": [1,1,1,1,2,1,1,1],
            "Cost per Hour ($)": [40,55,85,65,55,90,55,40],
            "Rework Rate (%)": [2,5,3,4,5,2,3,0],
        }),
        "Product Return Processing": pd.DataFrame({
            "Process Step": ["Return Request Received","Customer Verification",
                             "Return Auth Generation","Physical Receipt",
                             "Quality Inspection","Disposition Decision",
                             "Refund Processing","Inventory Update"],
            "Step Type": ["Necessary Non-Value-Added (NNVA)","Non-Value-Added (NVA)",
                          "Necessary Non-Value-Added (NNVA)","Necessary Non-Value-Added (NNVA)",
                          "Value-Added (VA)","Value-Added (VA)",
                          "Value-Added (VA)","Non-Value-Added (NVA)"],
            "Cycle Time (min)": [5,10,5,15,25,10,20,8],
            "Wait Time Before (min)": [0,30,120,4320,60,30,120,60],
            "Resources": [1,1,1,2,2,1,1,1],
            "Cost per Hour ($)": [35,35,35,30,45,55,45,35],
            "Rework Rate (%)": [0,5,2,1,10,3,2,5],
        }),
    }

    if example != "Invoice Approval (default)" and example in examples:
        if st.button(f"Load: {example}"):
            st.session_state.process_df = examples[example].copy()
            st.rerun()



# ══════════════════════════════════════════════════════════════════════════════
# MODULE: VSM DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
elif "VSM Diagram" in module:
    st.markdown('<div class="section-label">VALUE STREAM MAP — CURRENT STATE</div>', unsafe_allow_html=True)
    st.markdown("## VSM Diagram")

    st.markdown("""
<div style="background:#fff;border:1px solid #e2e8f0;border-left:3px solid #7c3aed;
     border-radius:10px;padding:12px 18px;font-size:0.83rem;line-height:1.7;
     color:#374151;margin-bottom:16px">
<strong>Current State Value Stream Map</strong> — generated from your process data.
Color coding: <span style="color:#16a34a;font-weight:600">green = Value-Added</span>,
<span style="color:#ca8a04;font-weight:600">yellow = Necessary NVA</span>,
<span style="color:#dc2626;font-weight:600">red = Non-Value-Added (pure waste)</span>.
The zigzag timeline at the bottom is the signature VSM element — peaks up = cycle time,
peaks down = wait time. The wider and lower the downward peaks, the more waste.
</div>""", unsafe_allow_html=True)

    if len(st.session_state.process_df) == 0:
        st.warning("No process steps defined. Go to Process Builder to add steps.")
    else:
        proc_title = st.text_input("Process Name (for diagram title)", value="Business Process")
        svg_content = generate_vsm_svg(st.session_state.process_df, proc_title)

        st.markdown(
            f'''<div style="overflow-x:auto;background:#fff;border:1px solid #e2e8f0;
            border-radius:10px;padding:16px">{svg_content}</div>''',
            unsafe_allow_html=True,
        )

        st.markdown("---")

        res = run_analysis(st.session_state.process_df, daily_volume, working_days, hours_per_day)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Lead Time",    f"{res['total_cycle_time']:.0f} min")
        c2.metric("Value-Added Time",   f"{res['va_cycle_time']:.0f} min")
        c3.metric("Process Efficiency", f"{res['process_efficiency']}%",
                  delta="Below 50% — high waste" if res['process_efficiency'] < 50
                  else ("Room to improve" if res['process_efficiency'] < 80 else "World-class"),
                  delta_color="inverse" if res['process_efficiency'] < 50 else "normal")
        c4.metric("Annual Waste Cost",  f"${res['total_waste_cost']:,.0f}")

        st.markdown("---")
        st.markdown("### How to Read This VSM")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**Process Boxes** — each step in your process. Color = waste classification.
CT is the active work time. Wait time is the queue before the step starts.

**Push Arrows** — material/information flows step to step.

**Inventory Triangles** — yellow triangles represent work waiting in queue.
""")
        with col2:
            st.markdown("""
**Zigzag Timeline** — the signature VSM element at the bottom.
Peaks UP = cycle time. Peaks DOWN = wait/queue waste.

**Process Efficiency** — VA time / total lead time.
World-class back-office processes run >80%.
""")

        st.download_button(
            label="⬇️ Download VSM as SVG",
            data=svg_content,
            file_name="vsm_current_state.svg",
            mime="image/svg+xml",
        )

# ══════════════════════════════════════════════════════════════════════════════
# MODULE: VSM ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "VSM" in module:
    st.markdown('<div class="section-label">STEP 2 — VALUE STREAM MAPPING</div>', unsafe_allow_html=True)
    st.markdown("## VSM Analysis")

    res = run_analysis(st.session_state.process_df, daily_volume, working_days, hours_per_day)
    sdf = res["steps"]

    # KPIs
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Cycle Time", f"{res['total_cycle_time']:.0f} min")
    c2.metric("Value-Added Time", f"{res['va_cycle_time']:.0f} min",
              delta=f"{res['process_efficiency']:.1f}% of total")
    c3.metric("Process Efficiency", f"{res['process_efficiency']}%",
              delta="Below target" if res['process_efficiency'] < 50 else ("Room to improve" if res['process_efficiency'] < 80 else "Healthy"),
              delta_color="inverse" if res['process_efficiency'] < 50 else "normal")
    c4.metric("Annual Waste Cost", f"${res['total_waste_cost']:,.0f}",
              delta="Recoverable", delta_color="inverse")

    st.markdown("---")

    # Process efficiency gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=res["process_efficiency"],
        title={"text": "Process Efficiency %", "font": {"size": 14, "color": "#1a1f2e"}},
        delta={"reference": 80, "valueformat": ".1f",
               "increasing": {"color": "#16a34a"}, "decreasing": {"color": "#dc2626"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"size": 10}},
            "bar": {"color": "#7c3aed"},
            "steps": [
                {"range": [0, 40], "color": "#fef2f2"},
                {"range": [40, 70], "color": "#fefce8"},
                {"range": [70, 100], "color": "#f0fdf4"},
            ],
            "threshold": {"line": {"color": "#16a34a", "width": 3}, "value": 80},
        },
        number={"suffix": "%", "font": {"family": "DM Mono", "size": 36}},
    ))
    fig_gauge.update_layout(paper_bgcolor="#f8f9fb", height=260, margin=dict(t=30,b=10,l=30,r=30))

    col_g, col_vsm = st.columns([1, 2])
    with col_g:
        st.plotly_chart(fig_gauge, width='stretch')
        st.markdown(f"""
<div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:12px 14px;font-size:0.8rem;color:#374151">
<strong>Interpretation:</strong><br>
&lt;40% — High waste, urgent action<br>
40–70% — Significant improvement opportunity<br>
70–80% — Good, targeted improvements<br>
&gt;80% — World-class process efficiency
</div>""", unsafe_allow_html=True)

    with col_vsm:
        # Stacked bar: VA / NNVA / NVA time per step
        fig_vsm = go.Figure()
        colors = {"Value-Added (VA)": "#16a34a",
                  "Necessary Non-Value-Added (NNVA)": "#ca8a04",
                  "Non-Value-Added (NVA)": "#dc2626"}

        for stype, color in colors.items():
            mask = sdf["Type"] == stype
            fig_vsm.add_trace(go.Bar(
                name=TYPE_SHORT[stype],
                x=sdf["Step"],
                y=sdf["Cycle_min"].where(mask, 0),
                marker_color=color,
                opacity=0.85,
            ))

        # Wait time as a separate series
        fig_vsm.add_trace(go.Bar(
            name="Wait / Queue",
            x=sdf["Step"],
            y=sdf["Wait_min"],
            marker_color="#94a3b8",
            opacity=0.6,
        ))

        apply_light(fig_vsm, height=300,
                    title="Cycle Time + Wait Time Breakdown by Step",
                    yaxis_title="Minutes",
                    extra_layout=dict(barmode="stack",
                                     legend=dict(orientation="h", y=1.12, x=0)))
        fig_vsm.update_xaxes(tickangle=-30)
        st.plotly_chart(fig_vsm, width='stretch')

    st.markdown("---")

    # Step-by-step table
    display_cols = ["Step","Type_Short","Cycle_min","Wait_min","Utilization",
                    "Total_Step_Cost","Waste_Cost"]
    display_names = {"Step":"Process Step","Type_Short":"Type","Cycle_min":"Cycle (min)",
                     "Wait_min":"Wait (min)","Utilization":"Utilization %",
                     "Total_Step_Cost":"Annual Cost ($)","Waste_Cost":"Waste Cost ($)"}

    def color_type(val):
        c = {"VA":"#dcfce7","NNVA":"#fef9c3","NVA":"#fee2e2"}
        return f"background-color:{c.get(val,'#fff')}"

    def color_util(val):
        if val >= 90: return "background-color:#fee2e2;color:#dc2626;font-weight:600"
        if val >= 75: return "background-color:#fef9c3"
        return ""

    styled = (sdf[display_cols]
              .rename(columns=display_names)
              .style
              .map(color_type, subset=["Type"])
              .map(color_util, subset=["Utilization %"])
              .format({"Annual Cost ($)": "${:,.0f}", "Waste Cost ($)": "${:,.0f}",
                       "Cycle (min)": "{:.1f}", "Wait (min)": "{:.0f}",
                       "Utilization %": "{:.1f}%"}))

    st.dataframe(styled, width='stretch')

    # Waste breakdown donut
    if res["total_waste_cost"] > 0:
        st.markdown("---")
        st.markdown("### Annual Waste Breakdown")
        waste_labels = ["NVA Step Cost", "Wait/Queue Cost", "Rework Cost"]
        waste_vals   = [res["nva_cost"], res["wait_cost_total"], res["rework_cost_total"]]
        waste_vals   = [v for v in waste_vals if v > 0]
        waste_labels = [l for l,v in zip(waste_labels,[res["nva_cost"],res["wait_cost_total"],res["rework_cost_total"]]) if v > 0]

        fig_donut = go.Figure(go.Pie(
            labels=waste_labels, values=waste_vals,
            hole=0.55, marker_colors=["#dc2626","#94a3b8","#f59e0b"],
            textfont=dict(size=11),
        ))
        fig_donut.update_layout(
            paper_bgcolor="#f8f9fb", showlegend=True, height=320,
            annotations=[dict(text=f"${res['total_waste_cost']:,.0f}<br>Total Waste",
                              x=0.5, y=0.5, font_size=13, showarrow=False,
                              font_color="#0f172a")],
            margin=dict(t=20,b=20,l=20,r=20),
        )
        st.plotly_chart(fig_donut, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: BOTTLENECK BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
elif "Bottleneck" in module:
    st.markdown('<div class="section-label">STEP 3 — BOTTLENECK DETECTION</div>', unsafe_allow_html=True)
    st.markdown("## Bottleneck Breakdown")

    st.markdown("""
<div style="background:#fff;border:1px solid #e2e8f0;border-left:3px solid #7c3aed;
     border-radius:10px;padding:12px 18px;font-size:0.83rem;line-height:1.7;
     color:#374151;margin-bottom:16px">
<strong>Bottleneck detection methodology:</strong> Utilization = daily demand ÷ step capacity.
A step with utilization >100% cannot keep up — it is your constraint. Steps at 80–100% are at risk
under demand spikes. This is the Theory of Constraints applied to business processes: you cannot
improve total throughput without first addressing the bottleneck. Every other improvement is wasted.
</div>""", unsafe_allow_html=True)

    res = run_analysis(st.session_state.process_df, daily_volume, working_days, hours_per_day)
    sdf = res["steps"].sort_values("Utilization", ascending=False)

    # Top bottleneck highlight
    top = sdf.iloc[0]
    status_class = "bottleneck-card" if top["Utilization"] >= 80 else "ok-card"
    st.markdown(f"""
<div class="{status_class}">
  <div style="font-size:0.65rem;text-transform:uppercase;letter-spacing:0.14em;
       color:#dc2626;font-weight:700;margin-bottom:4px">PRIMARY BOTTLENECK</div>
  <div style="font-size:1.1rem;font-weight:600;color:#0f172a">{top["Step"]}</div>
  <div style="font-size:0.84rem;color:#374151;margin-top:6px">
    Utilization: <strong>{top['Utilization']:.1f}%</strong> &nbsp;|&nbsp;
    Capacity: <strong>{top['Capacity_per_day']:.0f} units/day</strong> &nbsp;|&nbsp;
    Demand: <strong>{daily_volume} units/day</strong> &nbsp;|&nbsp;
    Annual Waste Cost: <strong>${top['Waste_Cost']:,.0f}</strong>
  </div>
  <div style="font-size:0.81rem;color:#64748b;margin-top:4px">
    {'⚠️ This step CANNOT keep up with demand. It is actively constraining your throughput.' if top['Utilization'] >= 100 
     else ('🟠 At risk — this step will break under demand spikes.' if top['Utilization'] >= 80 
           else '✅ No critical bottleneck detected. Focus on waste reduction.')}
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Horizontal bar chart — utilization ranked
    colors_util = ["#dc2626" if u >= 100 else ("#ea580c" if u >= 80 else ("#ca8a04" if u >= 60 else "#16a34a"))
                   for u in sdf["Utilization"]]

    fig_util = go.Figure(go.Bar(
        x=sdf["Utilization"], y=sdf["Step"],
        orientation="h",
        marker_color=colors_util,
        text=[f"{u:.1f}%" for u in sdf["Utilization"]],
        textposition="outside",
    ))
    fig_util.add_vline(x=100, line_dash="dash", line_color="#dc2626",
                       annotation_text="100% = Bottleneck threshold",
                       annotation_position="top right",
                       annotation_font_color="#dc2626")
    fig_util.add_vline(x=80, line_dash="dot", line_color="#ca8a04",
                       annotation_text="80% = Risk zone",
                       annotation_position="bottom right",
                       annotation_font_color="#ca8a04")
    apply_light(fig_util, height=380,
                title="Step Utilization — Ranked (Demand ÷ Capacity)")
    fig_util.update_xaxes(title_text="Utilization (%)",
                          range=[0, max(sdf["Utilization"].max() * 1.15, 110)])
    fig_util.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_util, width='stretch')

    st.markdown("---")

    # Cost ranking
    cost_ranked = res["steps"].sort_values("Total_Step_Cost", ascending=False)
    fig_cost = go.Figure(go.Bar(
        x=cost_ranked["Step"],
        y=cost_ranked["Total_Step_Cost"],
        marker_color=[TYPE_COLORS[t] for t in cost_ranked["Type"]],
        text=[f"${v:,.0f}" for v in cost_ranked["Total_Step_Cost"]],
        textposition="outside",
    ))
    apply_light(fig_cost, height=320,
                title="Annual Cost per Step (Cycle + Wait + Rework)",
                yaxis_title="Annual Cost ($)")
    fig_cost.update_xaxes(tickangle=-30)
    st.plotly_chart(fig_cost, width='stretch')

    # Ranked table
    st.markdown("### All Steps — Ranked by Utilization")
    rank_df = sdf[["Step","Type_Short","Utilization","Capacity_per_day",
                   "Total_Step_Cost","Waste_Cost","Rework_pct"]].copy()
    rank_df.columns = ["Step","Type","Utilization %","Capacity/Day","Annual Cost ($)","Waste Cost ($)","Rework %"]

    def highlight_util(val):
        if isinstance(val, float) or isinstance(val, int):
            if val >= 100: return "background:#fee2e2;color:#dc2626;font-weight:700"
            if val >= 80:  return "background:#fff7ed;color:#ea580c;font-weight:600"
        return ""

    st.dataframe(
        rank_df.style
        .map(highlight_util, subset=["Utilization %"])
        .format({"Annual Cost ($)": "${:,.0f}", "Waste Cost ($)": "${:,.0f}",
                 "Utilization %": "{:.1f}%", "Rework %": "{:.1f}%",
                 "Capacity/Day": "{:.1f}"}),
        width='stretch',
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: ROI SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif "ROI" in module:
    st.markdown('<div class="section-label">STEP 4 — IMPROVEMENT ROI SIMULATOR</div>', unsafe_allow_html=True)
    st.markdown("## ROI Simulator")

    st.markdown("""
<div style="background:#fff;border:1px solid #e2e8f0;border-left:3px solid #7c3aed;
     border-radius:10px;padding:12px 18px;font-size:0.83rem;line-height:1.7;
     color:#374151;margin-bottom:16px">
<strong>How to use:</strong> Select a process step to improve, choose your improvement type,
and set the improvement percentage. The simulator recalculates annual cost, waste reduction,
and net ROI — so you can answer the question every manager needs answered before approving a project:
<em>"Is this fix worth the investment?"</em>
</div>""", unsafe_allow_html=True)

    res = run_analysis(st.session_state.process_df, daily_volume, working_days, hours_per_day)
    sdf = res["steps"]

    step_names = sdf["Step"].tolist()
    col1, col2 = st.columns(2)
    selected_step = col1.selectbox("Select Step to Improve", step_names,
                                   index=int(sdf["Utilization"].idxmax()))
    improvement_type = col2.selectbox("Improvement Type", [
        "Reduce Cycle Time (automation / process redesign)",
        "Eliminate Wait Time (workflow restructuring)",
        "Reduce Rework Rate (quality improvement / training)",
        "Add Resources (headcount / parallel processing)",
    ])

    pct_improvement = st.slider(
        "Improvement Magnitude", 10, 90, 50,
        help="% reduction in the selected metric"
    )
    investment_cost = st.number_input(
        "One-Time Investment Cost ($) — training, automation, process redesign",
        min_value=0, value=15000, step=1000
    )

    step_row = sdf[sdf["Step"] == selected_step].iloc[0]

    # Calculate improvement
    original_cycle   = step_row["Cycle_min"]
    original_wait    = step_row["Wait_min"]
    original_rework  = step_row["Rework_pct"]
    original_res     = step_row["Resources"]
    cost_hr          = step_row["Cost_hr"]
    annual_vol       = res["annual_volume"]

    if "Cycle Time" in improvement_type:
        new_cycle = original_cycle * (1 - pct_improvement/100)
        new_wait, new_rework, new_res = original_wait, original_rework, original_res
    elif "Wait Time" in improvement_type:
        new_wait = original_wait * (1 - pct_improvement/100)
        new_cycle, new_rework, new_res = original_cycle, original_rework, original_res
    elif "Rework" in improvement_type:
        new_rework = original_rework * (1 - pct_improvement/100)
        new_cycle, new_wait, new_res = original_cycle, original_wait, original_res
    else:
        new_res = max(1, original_res + int(np.ceil(original_res * pct_improvement/100)))
        new_cycle, new_wait, new_rework = original_cycle, original_wait, original_rework

    def step_annual_cost(cycle, wait, rework, res, cph, vol):
        cpm = cph / 60
        cc  = cycle * cpm * res * vol
        wc  = wait  * cpm * res * vol
        rc  = cc * (rework / 100)
        return cc + wc + rc

    orig_cost    = step_annual_cost(original_cycle, original_wait, original_rework, original_res, cost_hr, annual_vol)
    new_cost     = step_annual_cost(new_cycle, new_wait, new_rework, new_res, cost_hr, annual_vol)
    annual_saving = orig_cost - new_cost
    net_roi      = annual_saving - investment_cost
    payback_mo   = (investment_cost / (annual_saving/12)) if annual_saving > 0 else 999
    roi_pct      = (net_roi / investment_cost * 100) if investment_cost > 0 else float("inf")

    # New utilization
    cap_before = (8 * 60 / original_cycle) * original_res if original_cycle > 0 else 9999
    cap_after  = (8 * 60 / new_cycle) * new_res if new_cycle > 0 else 9999
    util_before = daily_volume / cap_before * 100
    util_after  = daily_volume / cap_after  * 100

    st.markdown("---")
    st.markdown(f"### Results: Improving **{selected_step}**")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Annual Saving", f"${annual_saving:,.0f}",
              delta="Per year", delta_color="normal")
    c2.metric("Net ROI (Year 1)", f"${net_roi:,.0f}",
              delta=f"{roi_pct:.0f}% ROI" if investment_cost > 0 else "No investment required",
              delta_color="normal" if net_roi > 0 else "inverse")
    c3.metric("Payback Period",
              f"{payback_mo:.1f} months" if payback_mo < 900 else "N/A",
              delta="Break-even")
    c4.metric("Utilization Change",
              f"{util_before:.1f}% → {util_after:.1f}%",
              delta="Bottleneck relieved" if util_after < 80 else "Still constrained",
              delta_color="normal" if util_after < 80 else "inverse")

    # Before / After waterfall
    categories = ["Original Cost", "Saving", "Investment", "Net Benefit"]
    values     = [orig_cost, -annual_saving, investment_cost, net_roi]
    bar_colors = ["#94a3b8", "#16a34a", "#dc2626", "#7c3aed" if net_roi > 0 else "#ef4444"]

    fig_wf = go.Figure(go.Bar(
        x=categories, y=[orig_cost, annual_saving, investment_cost, abs(net_roi)],
        marker_color=bar_colors,
        text=[f"${abs(v):,.0f}" for v in [orig_cost, annual_saving, investment_cost, net_roi]],
        textposition="outside",
    ))
    apply_light(fig_wf, height=300,
                title="Improvement Economics — Before vs. After",
                yaxis_title="Annual $ Impact")
    st.plotly_chart(fig_wf, width='stretch')

    # Multi-year projection
    years = list(range(1, 6))
    cumulative = [annual_saving * y - investment_cost for y in years]
    fig_proj = go.Figure()
    fig_proj.add_trace(go.Scatter(
        x=years, y=cumulative,
        mode="lines+markers",
        line=dict(color="#7c3aed", width=2.5),
        marker=dict(size=8, color="#7c3aed"),
        fill="tozeroy",
        fillcolor="rgba(124,58,237,0.08)",
        name="Cumulative Net Benefit",
    ))
    fig_proj.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    apply_light(fig_proj, height=260,
                title="5-Year Cumulative Net Benefit",
                yaxis_title="Cumulative $ Benefit")
    fig_proj.update_xaxes(title_text="Year", tickmode="array", tickvals=years)
    st.plotly_chart(fig_proj, width='stretch')

    if net_roi > 0:
        st.success(f"✅ This improvement pays back in {payback_mo:.1f} months and delivers ${net_roi:,.0f} net benefit in Year 1. Recommend proceeding.")
    else:
        st.warning(f"⚠️ At this investment level, the improvement does not recover its cost in Year 1. Consider reducing investment scope or targeting a higher-impact step.")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE: AI PROCESS BRIEF
# ══════════════════════════════════════════════════════════════════════════════
elif "AI" in module:
    st.markdown('<div class="section-label">STEP 5 — AI PROCESS BRIEF</div>', unsafe_allow_html=True)
    st.markdown("## AI Process Brief")

    st.markdown("""
<div style="background:#fff;border:1px solid #e2e8f0;border-left:3px solid #7c3aed;
     border-radius:10px;padding:12px 18px;font-size:0.83rem;line-height:1.7;
     color:#374151;margin-bottom:16px">
Generates a plain-English process improvement brief structured for a VP or COO.
Covers: current state diagnosis, top 3 bottlenecks by financial impact, recommended
improvement actions with ROI estimates, and a prioritized 30-day action plan.
Powered by <strong>Groq Llama 3</strong>.
</div>""", unsafe_allow_html=True)

    res = run_analysis(st.session_state.process_df, daily_volume, working_days, hours_per_day)
    sdf = res["steps"]

    # Build structured context for AI
    top3_bottlenecks = sdf.nlargest(3, "Utilization")[
        ["Step","Type_Short","Utilization","Total_Step_Cost","Waste_Cost","Rework_pct"]
    ].to_dict("records")

    top3_waste = sdf.nlargest(3, "Waste_Cost")[
        ["Step","Type_Short","Total_Step_Cost","Waste_Cost","Rework_pct"]
    ].to_dict("records")

    nva_steps = sdf[sdf["Is_NVA"]]["Step"].tolist()

    process_context = {
        "process_name":         f"{len(sdf)}-step business process",
        "total_steps":          len(sdf),
        "daily_volume":         daily_volume,
        "annual_volume":        res["annual_volume"],
        "process_efficiency":   res["process_efficiency"],
        "total_cycle_time_min": res["total_cycle_time"],
        "va_time_min":          res["va_cycle_time"],
        "total_annual_cost":    res["total_annual_cost"],
        "total_waste_cost":     res["total_waste_cost"],
        "nva_cost":             res["nva_cost"],
        "wait_cost":            res["wait_cost_total"],
        "rework_cost":          res["rework_cost_total"],
        "recovery_potential":   res["recovery_potential"],
        "primary_bottleneck":   res["bottleneck_step"],
        "bottleneck_util":      res["bottleneck_util"],
        "top3_bottlenecks":     top3_bottlenecks,
        "top3_waste_steps":     top3_waste,
        "nva_steps":            nva_steps,
        "throughput_per_day":   res["throughput"],
    }

    GROQ_PROMPT = f"""You are an expert Industrial Engineer and Business Process Analyst with deep expertise in Value Stream Mapping and Six Sigma DMAIC.

Analyze this business process data and write a structured executive brief for a VP of Operations or COO.

PROCESS DATA:
{json.dumps(process_context, indent=2)}

Write a brief with these sections:
1. CURRENT STATE DIAGNOSIS (2-3 sentences: what the data shows about overall process health)
2. TOP 3 ISSUES BY FINANCIAL IMPACT (specific steps, specific dollar costs, specific causes)
3. RECOMMENDED ACTIONS (3 specific actions, each with expected ROI or cost recovery)
4. 30-DAY ACTION PLAN (3 concrete steps, assigned to roles, with deadlines)
5. EXPECTED OUTCOME (what the process looks like after improvements)

Rules:
- Be specific with numbers from the data
- Avoid generic advice
- Write for a senior leader who needs to make a decision, not read a textbook
- Total length: 350-450 words
- Use clear section headers"""

    if st.button("⚡ Generate AI Process Brief", type="primary"):
        with st.spinner("Analyzing process data and generating brief..."):
            try:
                import requests
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [{"role": "user", "content": GROQ_PROMPT}],
                        "max_tokens": 700,
                        "temperature": 0.3,
                    },
                    timeout=30,
                )
                result = response.json()
                # Show actual API error if request failed
                if response.status_code != 200:
                    err_msg = result.get("error", {}).get("message", str(result))
                    st.error(f"Groq API error ({response.status_code}): {err_msg}")
                elif "choices" not in result:
                    st.error(f"Unexpected response from Groq: {result}")
                else:
                    brief_text = result["choices"][0]["message"]["content"]
                    st.session_state["ai_brief"] = brief_text
            except KeyError as e:
                st.error('GROQ_API_KEY not found in Streamlit secrets. Go to Settings → Secrets and add: GROQ_API_KEY = your_key_here')
            except Exception as e:
                st.error(f"API error: {e}")

    if "ai_brief" in st.session_state:
        st.markdown(f'<div class="ai-block">{st.session_state["ai_brief"]}</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("### Process Summary Snapshot")
        c1,c2,c3 = st.columns(3)
        c1.metric("Process Efficiency", f"{res['process_efficiency']}%")
        c2.metric("Annual Waste",       f"${res['total_waste_cost']:,.0f}")
        c3.metric("Recovery Potential", f"${res['recovery_potential']:,.0f}")
    else:
        st.markdown("""
<div style="background:#fff;border:1px dashed #e2e8f0;border-radius:10px;
     padding:40px 20px;text-align:center;color:#94a3b8;font-size:0.88rem">
Click "Generate AI Process Brief" to get a plain-English executive summary of your process analysis.
<br><br>
<strong style="color:#7c3aed">Add your Groq API key</strong> to Streamlit secrets as GROQ_API_KEY.
<br>Free tier available at console.groq.com
</div>""", unsafe_allow_html=True)

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='font-size:0.74rem;color:#94a3b8;text-align:center'>
ProcessGap · Business Process Bottleneck Intelligence ·
Value Stream Mapping · Six Sigma DMAIC · Utilization-Based Bottleneck Detection ·
Built by Rutwik Satish · MS Engineering Management, Northeastern University
</p>""", unsafe_allow_html=True)
