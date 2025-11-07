# ===============================================================
# 📊 Insurance Risk Modeling — Streamlit Dashboard
# ===============================================================
# Run with:
#   uv run streamlit run src/app_dashboard.py
# ===============================================================

import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

# ---------------------------------------------------------------
# 📁 PATHS
# ---------------------------------------------------------------
# Adjusted for being inside src/ins
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_RAW = BASE_DIR / "data" / "raw"
OUTPUTS = BASE_DIR / "outputs"

# ---------------------------------------------------------------
# 🧭 PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Insurance Risk Modeling Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Insurance Risk Modeling Dashboard")
st.markdown("An interactive summary of the simulated auto insurance portfolio.")

# ---------------------------------------------------------------
# 📦 LOAD DATA
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    data_path = DATA_RAW / "synthetic_insurance_portfolio.csv"
    if not data_path.exists():
        st.error("❌ Data file not found. Please run `simulate_data.py` first.")
        st.stop()
    df = pd.read_csv(data_path)
    df["pure_premium_actual"] = (df["num_claims"] * df["avg_claim_amount"]) / df["exposure"]
    return df

df = load_data()

# ---------------------------------------------------------------
# 🧮 KPI CALCULATIONS
# ---------------------------------------------------------------
claim_freq = df["num_claims"].sum() / df["exposure"].sum()
mean_severity = df.loc[df["num_claims"] > 0, "avg_claim_amount"].mean()
mean_pure_premium = df["pure_premium_actual"].mean()
total_claims = df["num_claims"].sum()
n_policies = len(df)

# ---------------------------------------------------------------
# 🧱 SIDEBAR FILTERS
# ---------------------------------------------------------------
st.sidebar.header("🔍 Filters")
region = st.sidebar.multiselect("Region", sorted(df["region"].unique()), default=None)
vehicle_type = st.sidebar.multiselect("Vehicle Type", sorted(df["vehicle_type"].unique()), default=None)
urban_density = st.sidebar.multiselect("Urban Density", sorted(df["urban_density"].unique()), default=None)

filtered_df = df.copy()
if region:
    filtered_df = filtered_df[filtered_df["region"].isin(region)]
if vehicle_type:
    filtered_df = filtered_df[filtered_df["vehicle_type"].isin(vehicle_type)]
if urban_density:
    filtered_df = filtered_df[filtered_df["urban_density"].isin(urban_density)]

# ---------------------------------------------------------------
# 📈 KPI DISPLAY
# ---------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Policies", f"{n_policies:,}")
col2.metric("Total Claims", f"{total_claims:,}")
col3.metric("Claim Frequency", f"{claim_freq:.3f}")
col4.metric("Mean Severity (€)", f"{mean_severity:,.0f}")
col5.metric("Mean Pure Premium (€)", f"{mean_pure_premium:,.0f}")

st.markdown("---")

# ---------------------------------------------------------------
# 📊 DISTRIBUTION PLOTS
# ---------------------------------------------------------------
st.subheader("Distribution of Pure Premiums (€)")
fig = px.histogram(
    filtered_df,
    x="pure_premium_actual",
    nbins=50,
    color="vehicle_type",
    barmode="overlay",
    opacity=0.7,
    labels={"pure_premium_actual": "Pure Premium (€)"},
    title="Pure Premium Distribution by Vehicle Type"
)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------
# 📊 SEGMENTED ANALYSIS
# ---------------------------------------------------------------
st.subheader("Segment Summary")

segment_col = st.selectbox("Select Segment:", ["vehicle_type", "urban_density", "region", "garage", "commercial_use"])

seg_summary = (
    filtered_df.groupby(segment_col)
    .agg(
        n_policies=("policy_id", "count"),
        claim_freq=("num_claims", lambda x: x.sum() / len(x)),
        mean_severity=("avg_claim_amount", lambda x: x[filtered_df['num_claims'] > 0].mean()),
        mean_pure_premium=("pure_premium_actual", "mean"),
    )
    .sort_values("mean_pure_premium", ascending=False)
    .reset_index()
)

st.dataframe(seg_summary.style.format({
    "claim_freq": "{:.3f}",
    "mean_severity": "{:,.0f}",
    "mean_pure_premium": "{:,.0f}"
}))

fig2 = px.bar(
    seg_summary,
    x=segment_col,
    y="mean_pure_premium",
    color="mean_pure_premium",
    text=seg_summary["mean_pure_premium"].round(0),
    title=f"Average Pure Premium by {segment_col.replace('_', ' ').title()}",
    color_continuous_scale="viridis"
)
st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------
# 🧾 PORTFOLIO SUMMARY TABLE
# ---------------------------------------------------------------
st.markdown("### 📋 Portfolio Snapshot")
st.dataframe(
    filtered_df[
        ["policy_id", "region", "vehicle_type", "driver_age", "num_claims", "avg_claim_amount", "pure_premium_actual"]
    ].head(50)
)

# ---------------------------------------------------------------
# 📤 EXPORT OPTION
# ---------------------------------------------------------------
st.markdown("### 💾 Export Filtered Data")
csv_export = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Filtered Portfolio (CSV)",
    data=csv_export,
    file_name="filtered_insurance_portfolio.csv",
    mime="text/csv"
)

st.markdown("---")
st.caption("© 2025 Insurance Risk Modeling — Synthetic Auto Insurance Analytics Project")
