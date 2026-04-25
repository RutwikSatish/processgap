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

LIGHT = dict(
    template="plotly_white",
    paper_bgcolor="#f8f9fb", plot_bgcolor="#fff",
    font=dict(color="#1a1f2e", family="DM Sans"),
    xaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", tickfont=dict(color="#64748b")),
    yaxis=dict(gridcolor="#f1f5f9", linecolor="#e2e8f0", tickfont=dict(color="#64748b")),
    margin=dict(t=40, b=44, l=12, r=12),
)

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
        ["🏠  Overview", "🗺️  Process Builder", "📊  VSM Analysis",
         "🔴  Bottleneck Breakdown", "💰  ROI Simulator", "🤖  AI Process Brief"],
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
        use_container_width=True,
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
    example = st.selectbox("", [
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
        st.plotly_chart(fig_gauge, use_container_width=True)
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

        fig_vsm.update_layout(
            **LIGHT, barmode="stack", height=300,
            title="Cycle Time + Wait Time Breakdown by Step",
            yaxis_title="Minutes",
            legend=dict(orientation="h", y=1.12, x=0),
        )
        fig_vsm.update_xaxes(tickangle=-30, tickfont=dict(size=10))
        st.plotly_chart(fig_vsm, use_container_width=True)

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

    st.dataframe(styled, use_container_width=True)

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
        st.plotly_chart(fig_donut, use_container_width=True)


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
    fig_util.update_layout(
        **LIGHT, height=380, title="Step Utilization — Ranked (Demand ÷ Capacity)",
        xaxis_title="Utilization (%)", xaxis=dict(range=[0, max(sdf["Utilization"].max() * 1.15, 110)]),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_util, use_container_width=True)

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
    fig_cost.update_layout(
        **LIGHT, height=320, title="Annual Cost per Step (Cycle + Wait + Rework)",
        yaxis_title="Annual Cost ($)",
    )
    fig_cost.update_xaxes(tickangle=-30)
    st.plotly_chart(fig_cost, use_container_width=True)

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
        use_container_width=True,
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
    fig_wf.update_layout(**LIGHT, height=300,
                         title="Improvement Economics — Before vs. After",
                         yaxis_title="Annual $ Impact")
    st.plotly_chart(fig_wf, use_container_width=True)

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
    fig_proj.update_layout(
        **LIGHT, height=260, title="5-Year Cumulative Net Benefit",
        xaxis_title="Year", yaxis_title="Cumulative $ Benefit",
        xaxis=dict(tickmode="array", tickvals=years),
    )
    st.plotly_chart(fig_proj, use_container_width=True)

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
                        "model": "llama3-8b-8192",
                        "messages": [{"role": "user", "content": GROQ_PROMPT}],
                        "max_tokens": 700,
                        "temperature": 0.3,
                    },
                    timeout=30,
                )
                result = response.json()
                brief_text = result["choices"][0]["message"]["content"]
                st.session_state["ai_brief"] = brief_text
            except Exception as e:
                st.error(f"API error: {e}. Check your GROQ_API_KEY in Streamlit secrets.")

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
