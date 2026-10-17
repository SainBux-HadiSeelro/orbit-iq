<div align="center">

# 🛰️ OrbitIQ
### Autonomous Aerospace Mission Risk & Booster Recovery Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Live App](https://img.shields.io/badge/Live%20Demo-orbit--iq.streamlit.app-38bdf8?style=for-the-badge&logo=streamlit&logoColor=white)](https://orbit-iq.streamlit.app/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Status](https://img.shields.io/badge/System-Production%20Ready-00C853?style=for-the-badge&logo=statuspage&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)

<br/>

**A mission-critical predictive intelligence engine combining orbital flight telemetry, regularized machine learning, and multi-million dollar aerospace financial modeling.**

[🚀 **Launch Live Demo (Streamlit)**](https://orbit-iq.streamlit.app/) • [📑 System Specifications](SPECIFICATIONS.md) • [📊 Historical Telemetry Dataset](cleaned_data_spacex_h.csv) • [🛠️ Report Issue](https://github.com/SainBux-HadiSeelro/orbit-iq/issues)

</div>

---

## 📌 Table of Contents

- [🌟 Executive Summary](#-executive-summary)
- [🎯 Why OrbitIQ? (The Portfolio Differentiator)](#-why-orbitiq-the-portfolio-differentiator)
- [✨ Core Capabilities & Platform Features](#-core-capabilities--platform-features)
- [📐 Mathematical & Algorithmic Formulation](#-mathematical--algorithmic-formulation)
- [🏗️ System Architecture & ML Pipeline](#️-system-architecture--ml-pipeline)
- [🕹️ Mission Profile Presets](#️-mission-profile-presets)
- [⚡ Technical Specifications & Benchmarks](#-technical-specifications--benchmarks)
- [🚀 Quickstart & Installation](#-quickstart--installation)
- [📁 Repository Structure](#-repository-structure)
- [🌐 Deployment Guide](#-deployment-guide)
- [👨‍💻 Author & Connect](#-author--connect)

---

## 🌟 Executive Summary

In orbital spaceflight, first-stage booster recovery is the primary catalyst that reduces orbital payload insertion costs from **~$67M** down to under **~$30M**. However, determining launch viability, atmospheric recovery dynamics, and pad risk involves multi-variable non-linear dynamics.

**OrbitIQ** bridges raw SpaceX flight telemetry with **decision-support Machine Learning** and **aerospace cost retention analysis**. It provides mission directors, space insurance underwriters, and launch providers with immediate, confidence-calibrated flight risk assessments, stage-1 recovery likelihoods, and financial ROI projections.

---

## 🎯 Why OrbitIQ? (The Portfolio Differentiator)

Most machine learning portfolio projects are simple prediction scripts or generic notebook exports. **OrbitIQ** is built as an end-to-end, domain-specific decision platform:

| Standard ML Projects | 🛰️ OrbitIQ Platform |
| :--- | :--- |
| Single prediction output (0 or 1) | **Calibrated probability scores + Tiered Flight Risk Classifications** |
| Ignores financial and commercial context | **Aerospace Economic ROI Engine ($35M+ savings breakdown)** |
| Static analysis without business relevance | **Automated Mission Control Advisory based on flight cadence** |
| Black-box model outputs | **Full Explainable AI (XAI) with transparent logistic weights** |
| Basic tabular display | **Interactive Multi-Tab Telemetry Suite & Historical Archive** |

---

## ✨ Core Capabilities & Platform Features

### 1. 🎯 Real-Time Mission Simulation & Outcome Forecasting
- Multi-parametric input matrix: Flight sequence cadence, payload mass ($500\text{ kg} - 25,000\text{ kg}$), booster architecture (Falcon 9 / Heavy / 1), launch complex, and target recovery zone (`ASDS`, `RTLS`, `Ocean`).
- Dual-indicator prediction: Binary landing classification combined with a confidence-calibrated probability score.
- Dynamic flight risk tiering:
  - 🟢 **LOW RISK** ($P \ge 85\%$)
  - 🟡 **MODERATE RISK** ($60\% \le P < 85\%$)
  - 🔴 **CRITICAL RISK** ($P < 60\%$)

### 2. 💰 Aerospace Economic Impact & ROI Engine
- Models financial cost-efficiency achieved through first-stage recovery and fairing reuse.
- Compares against the commercial expendable baseline benchmark ($\$67.0\text{M}$):
  - **Refurbished Core Savings:** $+\$32.5\text{M}$
  - **Fairing Halves Recovery Savings:** $+\$5.2\text{M}$
  - **Net Projected Mission Cost:** Computed dynamically in real-time.

### 3. 📈 Global Fleet Telemetry & Reliability Analytics
- **Cumulative Learning Curve:** Tracks SpaceX's historical reliability progression across flight sequences, showcasing the maturation from experimental hops to industrial cadence.
- **Launch Complex Comparison:** Empirical landing success benchmark across Cape Canaveral (SLC-40), Kennedy Space Center (LC-39A), Vandenberg Space Force Base (SLC-4E), and Kwajalein Atoll.
- **Landing Mode Distribution:** Visual breakdown of Autonomous Drone Ship (`ASDS`) vs Return-To-Launch-Site (`RTLS`).
- **Feature Multicollinearity Matrix:** Seaborn correlation heatmap correlating cadence, landing attempts, and booster reuse.

### 4. 🗂️ Mission Archive & Telemetry Explorer
- Full-text search across historical SpaceX missions and payloads.
- Multi-criteria filtering by mission outcome (`True`, `False`, `unknown`) and booster reuse status.
- Comprehensive telemetry log displaying launch date, landing zone, and vehicle designation.

### 5. 🔬 Machine Learning Governance & Explainability (XAI)
- Live parameter weight inspection extracted directly from model coefficients (`model.coef_`).
- Quantifies positive vs negative drivers (e.g., flight-proven core experience, ground-pad RTLS guidance).
- Compliant with mission assurance auditing and transparent AI governance.

---

## 📐 Mathematical & Algorithmic Formulation

### Logistic Probability Model
The landing outcome probability is governed by the regularized logistic sigmoid function:

$$P(Y = 1 \mid X = x) = \sigma(w^T x + b) = \frac{1}{1 + e^{-(w_0 + \sum_{j=1}^{d} w_j x_j)}}$$

Where:
- $Y=1$ represents successful first-stage booster recovery.
- $x \in \mathbb{R}^d$ is the aligned telemetry feature vector.
- $w$ is the optimized weight vector penalized under $\ell_2$ regularization:

$$\min_{w, b} \left[ -\frac{1}{N} \sum_{i=1}^{N} \left( y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right) + \frac{1}{2C} \|w\|_2^2 \right]$$

### Economic Savings Formulation
Total flight savings $S_{\text{total}}$ and client net cost $C_{\text{net}}$ are computed as:

$$S_{\text{total}} = (32.5 \cdot \mathbb{I}_{\text{core\_reused}}) + (5.2 \cdot \mathbb{I}_{\text{fairings\_reused}}) \quad [\text{in Million USD}]$$

$$C_{\text{net}} = 67.0 - S_{\text{total}}$$

---

## 🏗️ System Architecture & ML Pipeline

```
  ┌────────────────────────────────────────────────────────┐
  │                 SpaceX API Telemetry                   │
  │     (Launches, Rockets, Launchpads, Recovery Data)     │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │         Feature Engineering & Transformation           │
  │     - One-Hot Categorical Encoding                     │
  │     - Continuous Sequence Normalization                │
  │     - Imputation & Drop-First Matrix Formulation       │
  └───────────────────────────┬────────────────────────────┘
                              │
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │       Regularized Logistic Regression Engine (L2)      │
  │     - Hyperparameters: C=1.0, max_iter=1000            │
  │     - Target: Binary Booster Recovery State (0 / 1)    │
  └─────────────┬───────────────────────────┬──────────────┘
                │                           │
                ▼                           ▼
  ┌───────────────────────────┐ ┌──────────────────────────┐
  │ Confidence & Risk Tiers   │ │  Aerospace ROI Engine    │
  │ (Low / Moderate / High)   │ │  (Net Savings & Cost)    │
  └─────────────┬─────────────┘ └───────────┬──────────────┘
                │                           │
                └─────────────┬─────────────┘
                              ▼
  ┌────────────────────────────────────────────────────────┐
  │        Streamlit Production Intelligence Suite         │
  │   - Real-Time Simulation    - Fleet Telemetry Analytics│
  │   - Historical Archive      - Model XAI Governance     │
  └────────────────────────────────────────────────────────┘
```

---

## 🕹️ Mission Profile Presets

To enable immediate evaluation without manual input configuration, OrbitIQ includes four pre-calibrated operational scenarios:

| Preset Scenario | Target Vehicle | Primary Site | Recovery Mode | Flight Cadence |
| :--- | :--- | :--- | :--- | :--- |
| **🛰️ Starlink Mega-Constellation** | Falcon 9 | CCAFS SLC-40 | Autonomous Drone Ship (ASDS) | Mature (#140) |
| **📡 Commercial GTO Satellite** | Falcon 9 | KSC LC-39A | Autonomous Drone Ship (ASDS) | Standard (#90) |
| **🌌 Deep Space / Scientific Probe** | Falcon Heavy | KSC LC-39A | Return-To-Launch-Site (RTLS) | Mid (#55) |
| **⚙️ Custom Profile** | User Config | User Config | User Config | Flexible ($1 - 350$) |

---

## ⚡ Technical Specifications & Benchmarks

For comprehensive software requirements and engineering specifications, view the [SPECIFICATIONS.md](SPECIFICATIONS.md) document.

- **Inference Latency:** $\approx 3.2\text{ ms}$ (Sub-second real-time responsiveness)
- **Model Size:** $46.3\text{ KB}$ (Ultra-lightweight serialized artifact)
- **Memory Footprint:** $\approx 140\text{ MB}$ RAM under concurrent load
- **Data Engine:** Vectorized Pandas with Streamlit memory caching (`@st.cache_resource`, `@st.cache_data`)

---

## 🚀 Quickstart & Installation

### Option 1: Local Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/SainBux-HadiSeelro/orbit-iq.git
cd orbit-iq
```

#### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Launch the OrbitIQ Platform
```bash
streamlit run spacex_app.py
```

The application will launch automatically in your browser at `http://localhost:8501`.

---

### Option 2: Docker Setup (Containerized)

```bash
# Build Docker image
docker build -t orbitiq:latest .

# Run container
docker run -p 8501:8501 orbitiq:latest
```

---

## 📁 Repository Structure

```
├── spacex_app.py              # Main OrbitIQ Streamlit interactive dashboard
├── spacex_model.pkl           # Trained regularized Logistic Regression classifier
├── model_features.pkl         # Serialized feature list for exact vector alignment
├── cleaned_data_spacex_h.csv  # Cleaned SpaceX flight telemetry dataset
├── requirements.txt           # Project dependencies and version pinning
├── SPECIFICATIONS.md          # Complete IEEE-style technical specifications
├── Spacex.ipynb               # Telemetry extraction & data cleaning notebook
├── spacex_model.ipynb         # Model training, validation & hyperparameter tuning
└── README.md                  # System overview and portfolio documentation
```

---

## 🌐 Deployment & Live Access

### 🔴 Official Production URL
OrbitIQ is live and deployed in production on Streamlit Community Cloud:

👉 **[https://orbit-iq.streamlit.app/](https://orbit-iq.streamlit.app/)**

### Self-Hosting on Streamlit Community Cloud
1. Fork or push this repository to your GitHub account (`SainBux-HadiSeelro/orbit-iq`).
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **"New App"** and select:
   - **Repository:** `SainBux-HadiSeelro/orbit-iq`
   - **Branch:** `main`
   - **Main file path:** `spacex_app.py`
4. Click **Deploy**. The app will build and initialize using `.streamlit/config.toml` automatically.

---

## 👨‍💻 Author & Connect

**Sain Bux (Hadi Seelro)**  
*Data Scientist & Machine Learning Engineer*

- 🌐 **Portfolio Website:** [sainbux.github.io](https://sainbux.github.io/)
- 🐙 **GitHub:** [@SainBux-HadiSeelro](https://github.com/SainBux-HadiSeelro)
- 💼 **LinkedIn:** [Connect on LinkedIn](https://www.linkedin.com/in/sain-bux/)
- 📧 **Email:** Available via portfolio

---

<div align="center">
  <sub>OrbitIQ System • Aerospace Mission Assurance & Autonomous Fleet Analytics</sub>
</div>
