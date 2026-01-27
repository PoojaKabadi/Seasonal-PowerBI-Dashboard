import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
import os
import warnings
import traceback
import plotly.express as px
import plotly.graph_objects as go

# Suppress warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None

# Streamlit configuration
st.set_page_config(
    page_title="D&B Reporting for Highmark",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.set_option("client.showErrorDetails", False)
SHOW_UI_ERRORS = False

def log_exception(context: str, exc: Exception) -> None:
    print(f"\n[{datetime.now().isoformat()}] {context}\n{traceback.format_exc()}\n")

def ui_fail_message(msg: str, context: str, exc: Exception) -> None:
    log_exception(context, exc)
    if SHOW_UI_ERRORS:
        st.error(msg)
        st.code(traceback.format_exc())
    else:
        st.info(msg)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #31006f;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="stException"] {display: none;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "step" not in st.session_state:
    st.session_state.step = 1
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {"opp": None, "benchmark": None}

# Header with logo
col1, col2 = st.columns([2, 5])
with col1:
    if os.path.exists("cotiviti_logo.png"):
        st.image("cotiviti_logo.png", width=300)
    else:
        st.markdown("### 📊")

with col2:
    st.markdown('<div class="main-header">D&B Reporting for Highmark</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Data Analysis & Reporting Tool</div>', unsafe_allow_html=True)

st.divider()

# Progress indicator
progress_cols = st.columns(3)
with progress_cols[0]:
    if st.session_state.step >= 1:
        st.success("✅ Step 1: Upload Files")
    else:
        st.info("⏳ Step 1: Upload Files")
with progress_cols[1]:
    if st.session_state.step >= 2:
        st.success("✅ Step 2: Apply Filters")
    else:
        st.info("⏳ Step 2: Apply Filters")
with progress_cols[2]:
    if st.session_state.step >= 3:
        st.success("✅ Step 3: View Results")
    else:
        st.info("⏳ Step 3: View Results")

st.divider()

# Helper function for date bucketing
def bucket_decision_date(date_str):
    """Bucket decision dates into categories"""
    if pd.isna(date_str) or str(date_str).strip().lower() in ['never presented', 'nan', '']:
        return "Never Presented"
    
    try:
        date_obj = pd.to_datetime(str(date_str).strip(), format="%b-%Y", errors='coerce')
        if pd.isna(date_obj):
            return "Never Presented"
        
        today = pd.Timestamp.today()
        months_diff = (today.year - date_obj.year) * 12 + (today.month - date_obj.month)
        
        if months_diff <= 12:
            return "Last 12 Months"
        elif months_diff <= 24:
            return "Last 24 Months"
        else:
            return str(date_obj.year)
    except:
        return "Never Presented"

# ==================== STEP 1: FILE UPLOAD ====================
if st.session_state.step == 1:
    st.markdown("### 📁 Step 1: Upload Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_opp = st.file_uploader("Upload Opportunity Data", type=["xlsx", "xls"], key="opp_upload")
        if uploaded_opp:
            st.session_state.uploaded_files["opp"] = uploaded_opp
            st.success(f"✅ {uploaded_opp.name}")
    
    with col2:
        uploaded_benchmark = st.file_uploader("Upload Benchmark", type=["xlsx", "xls"], key="benchmark_upload")
        if uploaded_benchmark:
            st.session_state.uploaded_files["benchmark"] = uploaded_benchmark
            st.success(f"✅ {uploaded_benchmark.name}")
    
    if st.session_state.uploaded_files["opp"] and st.session_state.uploaded_files["benchmark"]:
        st.divider()
        if st.button("➡️ Next: Apply Filters", type="primary", use_container_width=True):
            st.session_state.step = 2
            st.rerun()
    else:
        st.info("👆 Please upload both files to continue")

# ==================== STEP 2: APPLY FILTERS ====================
elif st.session_state.step == 2:
    st.markdown("### 🔍 Step 2: Apply Filters")
    
    try:
        # Read files ONLY once
        if 'df_cleaned' not in st.session_state:
            with st.spinner("Loading data..."):
                df = pd.read_excel(st.session_state.uploaded_files["opp"], engine="openpyxl")
                df_benchmark_raw = pd.read_excel(st.session_state.uploaded_files["benchmark"], engine="openpyxl")
                
                df = df.iloc[1:].reset_index(drop=True)
                df = df.drop(columns=["Index", "-SUM([Annl Agg Savings - treated])"], errors="ignore")
                
                required_files = ["Policy Collection Mapping.xlsx", "Topic_Dp_Count.xlsx"]
                missing = [f for f in required_files if not os.path.exists(f)]
                if missing:
                    st.info("Reference files are missing: " + ", ".join(missing))
                    st.stop()
                
                df_map = pd.read_excel("Policy Collection Mapping.xlsx", engine="openpyxl",
                                       usecols=["DPKey", "PolicyCollection", "PolicyGroup"])
                df_map["DPKey"] = pd.to_numeric(df_map["DPKey"], errors="coerce")
                df["Dp Key"] = pd.to_numeric(df["Dp Key"], errors="coerce")
                
                df = (df.merge(df_map.drop_duplicates("DPKey"), how="left", left_on="Dp Key", right_on="DPKey")
                       .drop(columns=["DPKey"])
                       .rename(columns={"PolicyCollection": "Policy Collection", "PolicyGroup": "Policy Group"}))
                
                if "Annl Edits" in df.columns:
                    df["Annl Edits"] = pd.to_numeric(df["Annl Edits"], errors="coerce").round(0).astype("Int64")
                if "Annl Agg Savings" in df.columns:
                    df["Annl Agg Savings"] = pd.to_numeric(df["Annl Agg Savings"], errors="coerce").round(0).astype("Int64")
                
                st.session_state.df_cleaned = df
                st.session_state.df_benchmark = df_benchmark_raw
        
        df = st.session_state.df_cleaned
        df_benchmark_raw = st.session_state.df_benchmark
        
        # Create bucketed decision dates
        df["Decision Date Bucket"] = df["Decision Date"].apply(bucket_decision_date)
        
        # Filter options
        lob_options = ["All"] + sorted(df["LOB"].dropna().astype(str).str.strip().unique().tolist())
        decision_status_options = ["All"] + sorted(df["Decision Status"].dropna().astype(str).str.strip().unique().tolist())
        
        # Decision date buckets - Never Presented first, then chronological
        date_buckets = df["Decision Date Bucket"].dropna().unique().tolist()
        date_bucket_order = []
        if "Never Presented" in date_buckets:
            date_bucket_order.append("Never Presented")
        if "Last 12 Months" in date_buckets:
            date_bucket_order.append("Last 12 Months")
        if "Last 24 Months" in date_buckets:
            date_bucket_order.append("Last 24 Months")
        years = sorted([b for b in date_buckets if b not in date_bucket_order], reverse=True)
        date_bucket_order.extend(years)
        
        decision_date_options = ["All"] + date_bucket_order
        
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_lob = st.multiselect("Line of Business (LOB)", options=lob_options, default=["All"])
        with col2:
            selected_status = st.multiselect("Decision Status", options=decision_status_options, default=["All"])
        with col3:
            selected_date = st.multiselect("Decision Date", options=decision_date_options, default=["All"])
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("⬅️ Back to Upload", use_container_width=True):
                st.session_state.step = 1
                st.rerun()
        
        with col2:
            if st.button("➡️ Next: Process Data", type="primary", use_container_width=True):
                with st.spinner("Processing data... Please wait."):
                    try:
                        df_f = df.copy()
                        if "All" not in selected_lob:
                            df_f = df_f[df_f["LOB"].astype(str).str.strip().isin(selected_lob)]
                        if "All" not in selected_status:
                            df_f = df_f[df_f["Decision Status"].astype(str).str.strip().isin(selected_status)]
                        if "All" not in selected_date:
                            df_f = df_f[df_f["Decision Date Bucket"].isin(selected_date)]
                        
                        df_pivot = (df_f.groupby(["Topic", "Policy Collection", "Dp Key", "Payer Short"], as_index=False)
                                       [["Annl Edits", "Annl Agg Savings"]].sum()
                                       .rename(columns={"Annl Edits": "Sum of Annl Edits",
                                                       "Annl Agg Savings": "Sum of Annl Agg Savings"}))
                        
                        df_piv2 = (df_pivot.groupby(["Topic", "Dp Key"], as_index=False)
                                          [["Sum of Annl Edits", "Sum of Annl Agg Savings"]].sum()
                                          .rename(columns={"Sum of Annl Edits": "Annl Edits",
                                                          "Sum of Annl Agg Savings": "Annl Agg Savings"}))
                        df_piv2["Annl Edits"] = pd.to_numeric(df_piv2["Annl Edits"], errors="coerce").fillna(0)
                        df_piv2["Annl Agg Savings"] = pd.to_numeric(df_piv2["Annl Agg Savings"], errors="coerce").fillna(0)
                        
                        tdc = pd.read_excel("Topic_Dp_Count.xlsx", engine="openpyxl")
                        tdc.columns = [c.strip() for c in tdc.columns]
                        tdc = tdc.rename(columns={"Dp key": "Dp Key"})
                        tdc["Dp Key"] = pd.to_numeric(tdc["Dp Key"], errors="coerce")
                        tdc["Topic_Dp_Count"] = pd.to_numeric(tdc["Topic_Dp_Count"], errors="coerce")
                        
                        topic_cnt = tdc.groupby("Topic", as_index=False)["Topic_Dp_Count"].max()
                        topic_cnt_map = dict(zip(topic_cnt["Topic"], topic_cnt["Topic_Dp_Count"]))
                        
                        active_topic = set(df_piv2.loc[(df_piv2["Annl Edits"] > 0) | 
                                                       (df_piv2["Annl Agg Savings"] > 0), "Topic"].unique())
                        small_topics_ok = set(topic_cnt.loc[(topic_cnt["Topic_Dp_Count"] <= 10) & 
                                                           (topic_cnt["Topic"].isin(active_topic)), "Topic"])
                        large_topics = set(topic_cnt.loc[topic_cnt["Topic_Dp_Count"] > 10, "Topic"])
                        
                        df_f2 = df_f.copy()
                        df_f2["Dp Key"] = pd.to_numeric(df_f2["Dp Key"], errors="coerce")
                        df_f2["_dd_dt"] = pd.to_datetime(df_f2["Decision Date"].astype(str).str.strip(),
                                                        format="%b-%Y", errors="coerce")
                        
                        latest_idx = (df_f2.sort_values(["Topic", "Dp Key", "_dd_dt"])
                                          .groupby(["Topic", "Dp Key"], as_index=False).tail(1).index)
                        
                        dec_latest = df_f2.loc[latest_idx, ["Topic", "Dp Key", "Decision Status", "_dd_dt"]].copy()
                        dec_latest = dec_latest.rename(columns={"_dd_dt": "DecisionDate_dt"})
                        dec_latest["DecisionYear"] = dec_latest["DecisionDate_dt"].dt.year
                        
                        def _comment_from_status(status, year, dt):
                            s = "" if pd.isna(status) else str(status).strip()
                            if pd.isna(dt):
                                return "Never presented"
                            if s.lower() == "no decision":
                                return f"No Decision in ({int(year)})" if pd.notna(year) else "No Decision"
                            if s.lower() == "reject":
                                return f"Previously Rejected in ({int(year)})" if pd.notna(year) else "Previously Rejected"
                            if s.lower() == "suppress":
                                return f"Previously Suppressed in ({int(year)})" if pd.notna(year) else "Previously Suppressed"
                            return dt.strftime("%b-%Y")
                        
                        dec_latest["Comments"] = dec_latest.apply(
                            lambda r: _comment_from_status(r["Decision Status"], r["DecisionYear"], r["DecisionDate_dt"]), axis=1)
                        
                        df_small = (tdc.loc[tdc["Topic"].isin(small_topics_ok), ["Topic", "Dp Key"]]
                                      .dropna(subset=["Dp Key"]).drop_duplicates()
                                      .merge(df_piv2, on=["Topic", "Dp Key"], how="left"))
                        df_small["Annl Edits"] = pd.to_numeric(df_small["Annl Edits"], errors="coerce").fillna(0).astype("int64")
                        df_small["Annl Agg Savings"] = pd.to_numeric(df_small["Annl Agg Savings"], errors="coerce").fillna(0).astype("int64")
                        df_small["Comments"] = "To complete topic"
                        df_small["DecisionDate_dt"] = pd.NaT
                        
                        df_large = (df_piv2.loc[df_piv2["Topic"].isin(large_topics) & 
                                               ((df_piv2["Annl Edits"] > 0) | (df_piv2["Annl Agg Savings"] > 0))]
                                          .merge(dec_latest[["Topic", "Dp Key", "Comments", "DecisionDate_dt"]],
                                                on=["Topic", "Dp Key"], how="left"))
                        df_large["Annl Edits"] = df_large["Annl Edits"].astype("int64")
                        df_large["Annl Agg Savings"] = df_large["Annl Agg Savings"].astype("int64")
                        df_large["Comments"] = df_large["Comments"].fillna("Never presented")
                        
                        df_to_present = pd.concat([df_small, df_large], ignore_index=True)
                        
                        pc_map = pd.read_excel("Policy Collection Mapping.xlsx", engine="openpyxl",
                                              usecols=["DPKey", "PolicyCollection"]).drop_duplicates("DPKey")
                        pc_map["DPKey"] = pd.to_numeric(pc_map["DPKey"], errors="coerce")
                        
                        df_to_present = (df_to_present.merge(pc_map, how="left", left_on="Dp Key", right_on="DPKey")
                                                      .drop(columns=["DPKey"])
                                                      .rename(columns={"PolicyCollection": "Policy Collection"}))
                        
                        def completing_topic_label(topic, comment):
                            if str(comment).strip().lower() == "to complete topic":
                                return "YES"
                            n = topic_cnt_map.get(topic, None)
                            if n is None or n <= 10:
                                return "NO"
                            # Create ranges: 11-15, 16-20, 21-25, etc.
                            lower = ((int(n) - 1) // 5) * 5 + 1
                            upper = lower + 4
                            return f"NO- Too many DPs ({lower}-{upper} DPs)"
                        
                        df_to_present["Completing Topic"] = df_to_present.apply(
                            lambda r: completing_topic_label(r["Topic"], r["Comments"]), axis=1)
                        
                        today = pd.Timestamp.today().normalize()
                        cutoff = today - pd.DateOffset(months=24)
                        df_to_present["Present"] = "NO"
                        c_lower = df_to_present["Comments"].astype(str).str.strip().str.lower()
                        df_to_present.loc[c_lower.isin(["never presented", "to complete topic"]), "Present"] = "YES"
                        df_to_present.loc[df_to_present["DecisionDate_dt"].notna() & 
                                         (df_to_present["DecisionDate_dt"] <= cutoff), "Present"] = "YES"
                        
                        df_benchmark = df_benchmark_raw.copy()
                        df_benchmark.columns = df_benchmark.columns.astype(str).str.strip()
                        
                        bench_cols = {"DP Key": "Dp Key", "Payer Adoption Rate": "Payer Adoption Rate_raw",
                                     "GPV %": "GPV_raw", "APV%": "APV_raw", "NPV %": "NPV_raw"}
                        
                        bench_lookup = df_benchmark[list(bench_cols.keys())].copy()
                        bench_lookup["DP Key"] = pd.to_numeric(bench_lookup["DP Key"], errors="coerce")
                        bench_lookup = bench_lookup.dropna(subset=["DP Key"]).drop_duplicates("DP Key")
                        
                        for col in ["Payer Adoption Rate", "GPV %", "APV%", "NPV %"]:
                            bench_lookup[col] = pd.to_numeric(
                                bench_lookup[col].astype(str).str.replace("%", "").str.replace(",", "").str.strip(),
                                errors="coerce")
                        
                        bench_lookup = bench_lookup.rename(columns=bench_cols)
                        
                        df_to_present["Dp Key"] = pd.to_numeric(df_to_present["Dp Key"], errors="coerce")
                        bench_dp_set = set(bench_lookup["Dp Key"].dropna().astype(int).unique())
                        mask_in_prod = df_to_present["Dp Key"].fillna(-1).astype(int).isin(bench_dp_set)
                        df_to_present.loc[mask_in_prod, "Comments"] = "Already in Prod"
                        
                        df_to_present = df_to_present.merge(bench_lookup, on="Dp Key", how="left")
                        mask_prod = df_to_present["Comments"].astype(str).str.strip().eq("Already in Prod")
                        
                        def fmt_percent(series, decimals):
                            s = pd.to_numeric(series, errors="coerce")
                            if s.dropna().empty:
                                return pd.Series([""] * len(series), index=series.index, dtype=object)
                            scale = 100 if s.dropna().max() <= 1 else 1
                            s = s * scale
                            fmt = "{:." + str(decimals) + "f}%"
                            return s.map(lambda v: fmt.format(v) if pd.notna(v) else "")
                        
                        df_to_present["Payer Adoption Rate"] = ""
                        df_to_present["GPV %"] = ""
                        df_to_present["APV %"] = ""
                        df_to_present["NPV %"] = ""
                        
                        df_to_present.loc[mask_prod, "Payer Adoption Rate"] = fmt_percent(
                            df_to_present.loc[mask_prod, "Payer Adoption Rate_raw"], 2)
                        df_to_present.loc[mask_prod, "GPV %"] = fmt_percent(
                            df_to_present.loc[mask_prod, "GPV_raw"], 3)
                        df_to_present.loc[mask_prod, "APV %"] = fmt_percent(
                            df_to_present.loc[mask_prod, "APV_raw"], 3)
                        df_to_present.loc[mask_prod, "NPV %"] = fmt_percent(
                            df_to_present.loc[mask_prod, "NPV_raw"], 3)
                        
                        df_to_present = df_to_present.drop(
                            columns=["Payer Adoption Rate_raw", "GPV_raw", "APV_raw", "NPV_raw", "DecisionDate_dt"],
                            errors="ignore")
                        
                        final_cols = ["Topic", "Policy Collection", "Dp Key", "Annl Edits", "Annl Agg Savings",
                                     "Present", "Comments", "Completing Topic",
                                     "Payer Adoption Rate", "GPV %", "APV %", "NPV %"]
                        df_to_present = df_to_present[final_cols]
                        
                        st.session_state.processed_data = {
                            "df_filtered": df_f,
                            "df_pivot": df_pivot,
                            "df_to_present": df_to_present,
                            "df_benchmark": df_benchmark_raw,
                            "filters": {"LOB": selected_lob, "Decision Status": selected_status,
                                       "Decision Date": selected_date}
                        }
                        
                        st.session_state.step = 3
                        st.success("✅ Data processed successfully!")
                        st.rerun()
                        
                    except Exception as e:
                        ui_fail_message("We couldn't process the data. Please verify and try again.",
                                       "Processing Step 2 failed", e)
                        st.stop()
    
    except Exception as e:
        ui_fail_message("We couldn't read the uploaded files. Please re-upload and try again.",
                       "Reading files failed", e)
        st.stop()

# ==================== STEP 3: VIEW RESULTS ====================
elif st.session_state.step == 3:
    st.markdown("### 📊 Step 3: Results & Export")
    
    if st.session_state.processed_data:
        data = st.session_state.processed_data
        df_to_present = data["df_to_present"]
        
        with st.expander("📋 View Applied Filters", expanded=False):
            for filter_type, filter_values in data["filters"].items():
                st.write(f"**{filter_type}:** {', '.join(filter_values)}")
        
        st.divider()
        st.markdown("#### Complete Results")
        st.dataframe(df_to_present, use_container_width=True, height=600)
        st.info(f"📊 Showing all **{len(df_to_present)}** rows")
        
        st.divider()
        st.markdown("### 📈 Summary Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df_to_present):,}")
        with col2:
            total_edits = df_to_present["Annl Edits"].sum()
            st.metric("Total Annl Edits", f"{total_edits:,}")
        with col3:
            total_savings = df_to_present["Annl Agg Savings"].sum()
            st.metric("Total Annl Agg Savings", f"${total_savings:,}")
        with col4:
            unique_topics = df_to_present["Topic"].nunique()
            st.metric("Unique Topics", unique_topics)
        
        st.divider()
        
        # Filter for Present = YES
        df_present_yes = df_to_present[df_to_present["Present"] == "YES"].copy()
        
        if len(df_present_yes) > 0:
            st.markdown("### 📊 Distribution Charts of to be Presented DP Key")
            
            # Helper function to parse percentages
            def parse_percent(val):
                if pd.isna(val) or val == '':
                    return 0
                return float(str(val).replace('%', ''))
            
            # Chart 1: Top 10 Topics by Annl Agg Savings
            st.markdown("#### Top 10 Opportunity by Topic")
            top_topics = (df_present_yes.groupby("Topic")["Annl Agg Savings"].sum()
                         .sort_values(ascending=True).tail(10))
            
            fig1 = go.Figure(go.Bar(
                x=top_topics.values,
                y=top_topics.index,
                orientation='h',
                marker_color='#31006f',
                text=[f'${v:,.0f}' for v in top_topics.values],
                textposition='outside'
            ))
            fig1.update_layout(
                xaxis_title="Annual Aggregate Savings ($)",
                yaxis_title="Topic",
                height=500,
                showlegend=False,
                margin=dict(l=20, r=120, t=20, b=40)
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            # Chart 2: Top 15 DP Keys by Annl Agg Savings
            st.markdown("#### Opportunity by DP Keys")
            top_dp_keys = (df_present_yes.groupby("Dp Key")["Annl Agg Savings"].sum()
                          .sort_values(ascending=True).tail(15))
            
            fig2 = go.Figure(go.Bar(
                x=top_dp_keys.values,
                y=[f'DP {int(k)}' for k in top_dp_keys.index],
                orientation='h',
                marker_color='#31006f',
                text=[f'${v:,.0f}' for v in top_dp_keys.values],
                textposition='outside'
            ))
            fig2.update_layout(
                xaxis_title="Annual Aggregate Savings ($)",
                yaxis_title="DP Key",
                height=600,
                showlegend=False,
                margin=dict(l=20, r=120, t=20, b=40)
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Chart 3: Opportunity by Decision Status
            st.markdown("#### Opportunity by Decision Status")
            
            # Attach Decision Status from original Opportunity Data for chart only
            df_f_decision = data["df_filtered"][["Dp Key", "Decision Status"]].copy()
            df_f_decision["Dp Key"] = pd.to_numeric(df_f_decision["Dp Key"], errors="coerce")
            df_f_decision = df_f_decision.drop_duplicates("Dp Key")
            
            # Create temporary dataframe for chart with Decision Status
            df_chart = df_present_yes.merge(df_f_decision, on="Dp Key", how="left")
            
            # Group by Decision Status and sum Annl Agg Savings
            status_dist = (df_chart.groupby("Decision Status")["Annl Agg Savings"].sum()
                          .sort_values(ascending=True))
            
            # Remove NaN if exists
            status_dist = status_dist[status_dist.index.notna()]
            
            fig3 = go.Figure(go.Bar(
                x=status_dist.values,
                y=status_dist.index,
                orientation='h',
                marker_color='#31006f',
                text=[f'${v:,.0f}' for v in status_dist.values],
                textposition='outside'
            ))
            fig3.update_layout(
                xaxis_title="Annual Aggregate Savings ($)",
                yaxis_title="Decision Status",
                height=400,
                showlegend=False,
                margin=dict(l=20, r=120, t=20, b=40)
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            # Chart 4: Distribution of DP Keys by Comments with hover metrics
            st.markdown("#### Distribution of DP Keys by Comments")
            
            comments_agg = df_present_yes.groupby("Comments").agg({
                "Dp Key": "count",
                "Annl Agg Savings": "sum",
                "Payer Adoption Rate": lambda x: parse_percent(x.iloc[0]) if len(x) > 0 else 0,
                "GPV %": lambda x: parse_percent(x.iloc[0]) if len(x) > 0 else 0,
                "APV %": lambda x: parse_percent(x.iloc[0]) if len(x) > 0 else 0,
                "NPV %": lambda x: parse_percent(x.iloc[0]) if len(x) > 0 else 0
            }).reset_index()
            
            comments_agg.columns = ["Comments", "DP_Count", "Total_Savings", "Avg_Payer_Rate", 
                                   "Avg_GPV", "Avg_APV", "Avg_NPV"]
            comments_agg["Percentage"] = (comments_agg["DP_Count"] / comments_agg["DP_Count"].sum() * 100)
            comments_agg = comments_agg.sort_values("Percentage", ascending=True)
            
            fig4 = go.Figure(go.Bar(
                x=comments_agg["Percentage"],
                y=comments_agg["Comments"],
                orientation='h',
                marker_color='#31006f',
                text=[f'{v:.1f}%' for v in comments_agg["Percentage"]],
                textposition='outside',
                customdata=comments_agg[["Avg_Payer_Rate", "Avg_GPV", "Avg_APV", "Avg_NPV"]],
                hovertemplate='<b>%{y}</b><br>' +
                             'Percentage: %{x:.1f}%<br>' +
                             'Payer Adoption Rate: %{customdata[0]:.2f}%<br>' +
                             'GPV: %{customdata[1]:.3f}%<br>' +
                             'APV: %{customdata[2]:.3f}%<br>' +
                             'NPV: %{customdata[3]:.3f}%<extra></extra>'
            ))
            fig4.update_layout(
                xaxis_title="Percentage of DP Keys (%)",
                yaxis_title="Comments",
                height=500,
                showlegend=False,
                margin=dict(l=20, r=100, t=20, b=40)
            )
            st.plotly_chart(fig4, use_container_width=True)
        
        else:
            st.warning("No data with Present = YES to display charts")
        
        st.divider()
        
        def create_excel_download(data_dict):
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                filters_df = pd.DataFrame([
                    {"Filter Type": k, "Selected Values": ", ".join(v) if isinstance(v, list) else v}
                    for k, v in data_dict["filters"].items()
                ])
                filters_df.to_excel(writer, sheet_name="Filters Applied", index=False)
                data_dict["df_filtered"].to_excel(writer, sheet_name="Opp Data", index=False)
                data_dict["df_pivot"].to_excel(writer, sheet_name="Pivot", index=False)
                data_dict["df_to_present"].to_excel(writer, sheet_name="To Present", index=False)
                data_dict["df_benchmark"].to_excel(writer, sheet_name="Benchmark", index=False)
            output.seek(0)
            return output
        
        excel_file = create_excel_download(data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"DNB_Report_Highmark_{timestamp}.xlsx"
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.download_button("📥 Download Excel Report", data=excel_file, file_name=filename,
                             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             type="primary", use_container_width=True)
        with col2:
            if st.button("🔄 Start Over", use_container_width=True):
                st.session_state.step = 1
                st.session_state.processed_data = None
                st.session_state.uploaded_files = {"opp": None, "benchmark": None}
                if 'df_cleaned' in st.session_state:
                    del st.session_state.df_cleaned
                if 'df_benchmark' in st.session_state:
                    del st.session_state.df_benchmark
                st.rerun()

st.divider()
st.markdown("""
    <div style='text-align: center; color: #6b7280; padding: 1rem;'>
        <small>D&B Reporting Tool | Powered by Streamlit | © 2025</small>
    </div>
""", unsafe_allow_html=True)