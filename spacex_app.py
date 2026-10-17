import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress sklearn and plotting runtime warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="OrbitIQ | Aerospace Mission Risk & Booster Reuse Analytics",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech Aerospace CSS Theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 15px;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .roi-card {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        border: 1px solid #4338ca;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Data & Model Artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_ml_assets():
    model = joblib.load('spacex_model.pkl')
    features = joblib.load('model_features.pkl')
    return model, features

@st.cache_data
def load_flight_data():
    try:
        df = pd.read_csv('cleaned_data_spacex_h.csv')
        return df
    except Exception:
        return None

model, model_features = load_ml_assets()
flight_df = load_flight_data()

# ---------------------------------------------------------
# Header & Brand Identity
# ---------------------------------------------------------
col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.markdown('<div class="main-header">🛰️ OrbitIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Autonomous Aerospace Mission Risk Assessment & Booster Lifecycle Intelligence Platform</div>', unsafe_allow_html=True)
with col_h2:
    st.caption("SYSTEM STATUS: **OPERATIONAL**")
    st.caption("AI ENGINE: **Logistic V4 Classifier**")

# ---------------------------------------------------------
# Sidebar: Mission Telemetry Configuration
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ Flight Parameters")

# Preset Mission Scenarios
preset = st.sidebar.selectbox(
    "Mission Profile Preset",
    ["Custom Profile", "Starlink Mega-Constellation", "Commercial GTO Satellite", "Deep Space / Scientific Probe"]
)

if preset == "Starlink Mega-Constellation":
    def_flight = 140
    def_payload = 16000
    def_rocket = "Falcon 9"
    def_site = "CCAFS SLC 40"
    def_landing = "ASDS"
    def_core_reused = True
    def_landing_attempt = True
    def_fairings = True
elif preset == "Commercial GTO Satellite":
    def_flight = 90
    def_payload = 6500
    def_rocket = "Falcon 9"
    def_site = "KSC LC 39A"
    def_landing = "ASDS"
    def_core_reused = True
    def_landing_attempt = True
    def_fairings = False
elif preset == "Deep Space / Scientific Probe":
    def_flight = 55
    def_payload = 14000
    def_rocket = "Falcon Heavy"
    def_site = "KSC LC 39A"
    def_landing = "RTLS"
    def_core_reused = False
    def_landing_attempt = True
    def_fairings = False
else:
    def_flight = 75
    def_payload = 4500
    def_rocket = "Falcon 9"
    def_site = "CCAFS SLC 40"
    def_landing = "RTLS"
    def_core_reused = True
    def_landing_attempt = True
    def_fairings = True

flight_number = st.sidebar.number_input("Flight Cadence Sequence #", min_value=1, max_value=350, value=def_flight, step=1)
payload_mass = st.sidebar.slider("Payload Mass to Orbit (kg)", min_value=500, max_value=25000, value=min(def_payload, 25000), step=250)

rocket_choice = st.sidebar.selectbox("Booster Architecture", ["Falcon 9", "Falcon Heavy", "Falcon 1"], index=["Falcon 9", "Falcon Heavy", "Falcon 1"].index(def_rocket))
launch_site_choice = st.sidebar.selectbox("Launch Complex", ["CCAFS SLC 40", "KSC LC 39A", "VAFB SLC 4E", "Kwajalein Atoll"], index=["CCAFS SLC 40", "KSC LC 39A", "VAFB SLC 4E", "Kwajalein Atoll"].index(def_site))
landing_type = st.sidebar.selectbox("Target Recovery Zone", ["ASDS", "RTLS", "Ocean", "unknown"], index=["ASDS", "RTLS", "Ocean", "unknown"].index(def_landing))

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 Component Reusability")
core_reused = st.sidebar.checkbox("Combat-Proven Booster (Flight-Reused Core)", value=def_core_reused)
landing_attempt = st.sidebar.checkbox("First-Stage Recovery Attempt", value=def_landing_attempt)
fairings_reused = st.sidebar.checkbox("Recovered Payload Fairings Reused", value=def_fairings)

# ---------------------------------------------------------
# Model Mapping Logic
# ---------------------------------------------------------
rocket_mapping = {
    'Falcon 9': 'rocket_5e9d0d95eda69973a809d1ec',
    'Falcon Heavy': 'rocket_5e9d0d95eda69974db09d1ed',
    'Falcon 1': None
}
launch_site_mapping = {
    'CCAFS SLC 40': 'launchpad_5e9e4502f509092b78566f87',
    'KSC LC 39A': 'launchpad_5e9e4502f509094188566f88',
    'VAFB SLC 4E': 'launchpad_5e9e4502f5090995de566f86',
    'Kwajalein Atoll': None
}
landing_type_mapping = {
    'RTLS': 'landing_type_RTLS',
    'Ocean': 'landing_type_Ocean',
    'unknown': 'landing_type_unknown',
    'ASDS': None
}

def generate_prediction():
    input_dict = {
        'flight_number': flight_number,
        'core_reused_True': int(core_reused),
        'landing_attempt_True': int(landing_attempt),
        'fairings.reused_True': int(fairings_reused)
    }
    if rocket_mapping.get(rocket_choice):
        input_dict[rocket_mapping[rocket_choice]] = 1
    if launch_site_mapping.get(launch_site_choice):
        input_dict[launch_site_mapping[launch_site_choice]] = 1
    if landing_type_mapping.get(landing_type):
        input_dict[landing_type_mapping[landing_type]] = 1

    input_df = pd.DataFrame([input_dict]).reindex(columns=model_features, fill_value=0)
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    return pred, proba

pred_outcome, pred_confidence = generate_prediction()

# ---------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------
tabs = st.tabs([
    "🚀 Mission Simulation & Financial ROI",
    "📈 Fleet Telemetry & Reliability",
    "🗂️ Mission Archive & Telemetry Explorer",
    "🧠 AI Model Governance & Architecture"
])

# =========================================================
# TAB 1: Mission Simulation & Financial ROI
# =========================================================
with tabs[0]:
    st.markdown("#### 🎯 Real-Time Mission Simulation Results")
    
    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
    
    with col_res1:
        if pred_outcome == 1:
            st.success("✅ **MISSION NOMINAL**")
            st.metric("Predicted Outcome", "SUCCESS")
        else:
            st.error("⚠️ **HIGH ANOMALY RISK**")
            st.metric("Predicted Outcome", "FAILURE RISK")

    with col_res2:
        conf_pct = round(pred_confidence * 100, 1)
        st.metric("Success Probability", f"{conf_pct}%", delta=f"{'+' if conf_pct >= 85 else '-'}{abs(round(conf_pct - 85, 1))}% vs Fleet Avg")

    with col_res3:
        risk_level = "LOW" if pred_confidence > 0.85 else ("MODERATE" if pred_confidence > 0.60 else "CRITICAL")
        st.metric("Flight Risk Classification", risk_level)

    with col_res4:
        recovery_status = "RECOVERABLE" if landing_attempt else "EXPENDABLE"
        st.metric("Stage-1 Recovery Profile", recovery_status)

    # Economic ROI & Cost Savings Breakdown
    st.markdown("---")
    st.markdown("#### 💰 Aerospace Economic Impact & Cost Savings Engine")
    
    # Financial constants (approximate aerospace cost benchmark)
    baseline_expendable_cost = 67.0  # Million USD for standard new expendable Falcon 9
    booster_refurb_saving = 32.5 if core_reused else 0.0  # Refurbished core savings
    fairing_reuse_saving = 5.2 if fairings_reused else 0.0  # Fairing recovery savings
    total_savings = booster_refurb_saving + fairing_reuse_saving
    estimated_mission_cost = baseline_expendable_cost - total_savings

    col_roi1, col_roi2, col_roi3 = st.columns(3)
    with col_roi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Baseline Expendable Cost</div>
            <div class="metric-val">${baseline_expendable_cost:.1f}M</div>
            <p style="font-size:0.8rem; color:#64748b; margin-top:5px;">Traditional non-reusable single flight benchmark</p>
        </div>
        """, unsafe_allow_html=True)
    with col_roi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Projected Mission Cost</div>
            <div class="metric-val" style="color:#10b981;">${estimated_mission_cost:.1f}M</div>
            <p style="font-size:0.8rem; color:#64748b; margin-top:5px;">Optimized cost factoring core & fairing reuse</p>
        </div>
        """, unsafe_allow_html=True)
    with col_roi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Total Net Reusability Savings</div>
            <div class="metric-val" style="color:#f59e0b;">+${total_savings:.1f}M</div>
            <p style="font-size:0.8rem; color:#64748b; margin-top:5px;">Estimated client & operator cost retention</p>
        </div>
        """, unsafe_allow_html=True)

    # Dynamic Aerospace Advisory
    st.markdown("#### 🛰️ Automated Mission Control Advisory")
    advisory_notes = []
    if flight_number > 80:
        advisory_notes.append("✅ **Flight Cadence Factor:** Booster is operating in mature production era with demonstrated high reliability.")
    else:
        advisory_notes.append("ℹ️ **Early Cadence Phase:** Early flight sequence numbers in dataset historical training display higher variance.")
    
    if landing_type == "RTLS":
        advisory_notes.append("🎯 **Recovery Dynamics (RTLS):** Return To Launch Site provides highest telemetry precision and immediate ground recovery.")
    elif landing_type == "ASDS":
        advisory_notes.append("🌊 **Recovery Dynamics (ASDS):** Autonomous Drone Ship landing accommodates high-energy orbital trajectories (e.g., GTO).")

    if core_reused:
        advisory_notes.append("♻️ **Booster Status:** Core reusability validated; flight-proven hardware drastically reduces launch asset amortization.")

    for note in advisory_notes:
        st.info(note)

# =========================================================
# TAB 2: Fleet Telemetry & Reliability Analytics
# =========================================================
with tabs[1]:
    st.markdown("#### 📊 Global Fleet Telemetry & Operational Benchmarking")
    
    if flight_df is not None:
        valid_df = flight_df[flight_df['success'] != 'unknown'].copy()
        valid_df['success_int'] = valid_df['success'].map({'True': 1, 'False': 0, True: 1, False: 0})
        
        # Fleet KPIs
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Total Launches Analyzed", len(flight_df))
        with kpi2:
            fleet_success_rate = round(valid_df['success_int'].mean() * 100, 1)
            st.metric("Historical Success Rate", f"{fleet_success_rate}%")
        with kpi3:
            reused_cores = len(flight_df[flight_df['core_reused'].astype(str) == 'True'])
            st.metric("Booster Reuse Missions", f"{reused_cores}")
        with kpi4:
            st.metric("Active Launch Complexes", "3 Active Pads")

        st.markdown("---")
        
        col_g1, col_g2 = st.columns(2)
        
        # Chart 1: Cumulative Success Rate Progression
        with col_g1:
            st.markdown("##### 📈 Reliability Growth Over Flight Progression")
            sorted_flights = valid_df.sort_values('flight_number').copy()
            sorted_flights['cumulative_rate'] = (sorted_flights['success_int'].cumsum() / (np.arange(len(sorted_flights)) + 1)) * 100
            
            fig, ax = plt.subplots(figsize=(7, 4.2))
            ax.plot(sorted_flights['flight_number'], sorted_flights['cumulative_rate'], color='#0284c7', linewidth=2.5, label='Cumulative Success Rate (%)')
            ax.axhline(fleet_success_rate, color='#f59e0b', linestyle='--', label=f'Fleet Average ({fleet_success_rate}%)')
            ax.set_xlabel('Flight Number Sequence', fontsize=9)
            ax.set_ylabel('Success Rate (%)', fontsize=9)
            ax.set_title('Cumulative Mission Reliability Growth', fontsize=11, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='lower right', fontsize=8)
            st.pyplot(fig)
            plt.close(fig)

        # Chart 2: Launch Complex Performance
        with col_g2:
            st.markdown("##### 📍 Launch Complex Landing Success Comparison")
            pad_names = {
                '5e9e4502f509092b78566f87': 'CCAFS SLC 40',
                '5e9e4502f509094188566f88': 'KSC LC 39A',
                '5e9e4502f5090995de566f86': 'VAFB SLC 4E',
                '5e9e4501f509094ba4566f84': 'Kwajalein Atoll'
            }
            valid_df['pad_name'] = valid_df['launchpad'].map(pad_names).fillna(valid_df['launchpad'])
            pad_stats = valid_df.groupby('pad_name')['success_int'].agg(['count', 'mean']).reset_index()
            pad_stats['mean'] = pad_stats['mean'] * 100
            
            fig2, ax2 = plt.subplots(figsize=(7, 4.2))
            bars = ax2.bar(pad_stats['pad_name'], pad_stats['mean'], color=['#3b82f6', '#10b981', '#6366f1', '#94a3b8'], width=0.55)
            ax2.set_ylabel('Success Rate (%)', fontsize=9)
            ax2.set_title('Mission Success Rate by Launch Complex', fontsize=11, fontweight='bold')
            ax2.set_ylim(0, 110)
            ax2.grid(axis='y', linestyle=':', alpha=0.6)
            
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
            st.pyplot(fig2)
            plt.close(fig2)

        st.markdown("---")
        col_g3, col_g4 = st.columns(2)
        
        # Chart 3: Recovery Strategy Breakdown
        with col_g3:
            st.markdown("##### 🛬 Target Landing Type Distribution")
            landing_counts = flight_df['landing_type'].value_counts()
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            ax3.pie(landing_counts, labels=landing_counts.index, autopct='%1.1f%%', startangle=140, 
                    colors=['#38bdf8', '#34d399', '#fbbf24', '#f87171'])
            ax3.axis('equal')
            ax3.set_title('Landing Method Allocations', fontsize=11, fontweight='bold')
            st.pyplot(fig3)
            plt.close(fig3)

        # Chart 4: Correlation Matrix
        with col_g4:
            st.markdown("##### 🔗 Key Determinants Correlation Heatmap")
            valid_df['core_reused_int'] = valid_df['core_reused'].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(0).astype(int)
            valid_df['landing_attempt_int'] = valid_df['landing_attempt'].map({'True': 1, 'False': 0, True: 1, False: 0}).fillna(0).astype(int)
            
            corr = valid_df[['flight_number', 'success_int', 'core_reused_int', 'landing_attempt_int']].corr()
            corr.rename(columns={
                'flight_number': 'Cadence #',
                'success_int': 'Mission Success',
                'core_reused_int': 'Core Reused',
                'landing_attempt_int': 'Recovery Attempt'
            }, index={
                'flight_number': 'Cadence #',
                'success_int': 'Mission Success',
                'core_reused_int': 'Core Reused',
                'landing_attempt_int': 'Recovery Attempt'
            }, inplace=True)
            
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            sns.heatmap(corr, annot=True, cmap='Blues', fmt='.2f', ax=ax4, cbar=False)
            ax4.set_title('Feature Multicollinearity Matrix', fontsize=11, fontweight='bold')
            st.pyplot(fig4)
            plt.close(fig4)
    else:
        st.warning("Historical dataset 'cleaned_data_spacex_h.csv' not found. Analytics dashboard inactive.")

# =========================================================
# TAB 3: Mission Archive & Telemetry Explorer
# =========================================================
with tabs[2]:
    st.markdown("#### 🗂️ SpaceX Historical Telemetry Archive")
    
    if flight_df is not None:
        search_query = st.text_input("🔍 Search mission name or payload keyword:", "")
        
        display_df = flight_df.copy()
        if search_query:
            display_df = display_df[display_df['name'].str.contains(search_query, case=False, na=False)]

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            status_filter = st.multiselect("Filter by Mission Outcome:", options=display_df['success'].unique(), default=display_df['success'].unique())
        with col_f2:
            reused_filter = st.multiselect("Filter by Core Reused Status:", options=display_df['core_reused'].unique(), default=display_df['core_reused'].unique())

        display_df = display_df[display_df['success'].isin(status_filter) & display_df['core_reused'].isin(reused_filter)]
        
        cols_to_show = ['flight_number', 'name', 'date_utc', 'landing_type', 'core_reused', 'landing_attempt', 'success']
        st.dataframe(
            display_df[cols_to_show].rename(columns={
                'flight_number': 'Flight #',
                'name': 'Mission Designation',
                'date_utc': 'Launch Date (UTC)',
                'landing_type': 'Recovery Zone',
                'core_reused': 'Reused Core',
                'landing_attempt': 'Landing Attempt',
                'success': 'Mission Outcome'
            }),
            use_container_width=True,
            height=380
        )
        st.caption(f"Displaying **{len(display_df)}** matching flight telemetry records.")
    else:
        st.info("Archive data unavailable.")

# =========================================================
# TAB 4: AI Model Governance & Architecture
# =========================================================
with tabs[3]:
    st.markdown("#### 🔬 Technical Architecture & Model Explainability (Recruiter Overview)")
    
    col_tech1, col_tech2 = st.columns([1, 1])
    
    with col_tech1:
        st.markdown("""
        ##### 📐 Model Specifications & Pipeline
        - **Algorithm:** Regularized Logistic Regression Classifier (`L2 penalty`)
        - **Encoding Architecture:** One-Hot Encoded Categorical Matrix (`pd.get_dummies` with baseline drop)
        - **Target Variable ($y$):** Binary Mission Success (`1 = Success`, `0 = Failure / Abort`)
        - **Cadence Calibration:** Normalized Flight Sequence Index factoring vehicle evolutionary upgrades (Block 1 → Block 5).
        - **Convergence:** Hyperparameter tuned with `max_iter=1000` for robust boundary convergence.
        """)
        
        st.markdown("""
        ##### 🏢 Production Relevance & Business Value
        - **Launch Risk Underwriting:** Real-time risk probability modeling for satellite insurers & space underwriters.
        - **Refurbishment Scheduling:** Quantifying mission success probabilities to determine optimum booster retirement cadence.
        - **Payload Manifest Optimization:** Selecting optimum launch complex and recovery profiles based on risk-adjusted margins.
        """)

    with col_tech2:
        st.markdown("##### ⚖️ Key Parameter Weights (Logistic Coefficients)")
        
        # Display key domain features and their weights
        coef_series = pd.Series(model.coef_[0], index=model_features)
        domain_features = {
            'core_reused_True': 'Booster Combat-Reused (+)',
            'landing_type_RTLS': 'Ground Pad RTLS (+)',
            'landing_type_Ocean': 'Ocean Splashdown (+)',
            'launchpad_5e9e4502f509094188566f88': 'KSC LC 39A Complex (+)',
            'launchpad_5e9e4502f509092b78566f87': 'CCAFS SLC 40 Complex (+)',
            'launchpad_5e9e4502f5090995de566f86': 'VAFB SLC 4E Complex (+)',
            'landing_attempt_True': 'Recovery Attempt (-)',
            'rocket_5e9d0d95eda69973a809d1ec': 'Falcon 9 Baseline (-)'
        }
        
        weights_data = []
        for feat, label in domain_features.items():
            if feat in coef_series.index:
                weights_data.append({'Feature Factor': label, 'Coefficient Weight': round(coef_series[feat], 3)})
        
        w_df = pd.DataFrame(weights_data).sort_values('Coefficient Weight', ascending=False)
        st.dataframe(w_df, use_container_width=True, hide_index=True)
        st.caption("Positive weights denote positive contribution towards predicted mission success probability.")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.85rem;'>OrbitIQ System • Developed for Aerospace Mission Assurance & Autonomous Fleet Analytics • SMIT AI Hackathon</p>", unsafe_allow_html=True)
