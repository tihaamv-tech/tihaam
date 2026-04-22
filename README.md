# FairCheck — AI Fairness Auditing System

**Hackathon Edition** | EU AI Act Compliance | Production-Ready

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
streamlit run faircheck.py
```

**Within 60 seconds**, you'll see a dark professional dashboard detecting bias in a real dataset, mapping it to EU law, explaining predictions with SHAP, comparing four models, mitigating bias, and exporting a PDF compliance report.

---

## Overview

FairCheck is a comprehensive AI fairness auditing system designed for:
- **Data scientists** evaluating ML model fairness
- **AI compliance officers** auditing under EU AI Act (Regulation EU 2024/1689)
- **Regulators** assessing automated decision-making systems

### Core Features

| Feature | Description |
|---------|-------------|
| **Bias Detection** | Computes demographic parity, equalized odds, and group accuracy differences |
| **Risk Classification** | 3-tier color-coded risk assessment (LOW/MEDIUM/HIGH) |
| **EU AI Act Compliance** | 8-dimension rule-based compliance checker mapped to Articles 9-15 |
| **Gap Detection** | Priority-sorted gap analysis with implementation effort estimates |
| **Recommendations** | Technical, actionable recommendations tied to specific compliance gaps |
| **Bias Mitigation** | Fairlearn ExponentiatedGradient with DemographicParity constraint |
| **SHAP Explainability** | Global and group-level feature importance visualization |
| **Multi-Model Comparison** | Compare 4 sklearn models side-by-side |
| **PDF Report Generation** | 5-page A4 compliance report with ReportLab |
| **Real-Time Monitor** | Track metrics across analysis runs |

---

## Architecture

```
faircheck.py
├── Section 1:  Dataset & Input Layer (UCI Adult Census)
├── Section 2:  Multi-Model Training (LR, DT, RF, GBM)
├── Section 3:  Fairness Audit Engine (fairlearn.metrics)
├── Section 4:  Bias Risk Classification (3-tier)
├── Section 5:  EU AI Act Compliance Engine (8 checks)
├── Section 6:  Gap Detection Engine
├── Section 7:  Recommendation Engine
├── Section 8:  Bias Mitigation Engine (ExponentiatedGradient)
├── Section 9:  SHAP Explainability Engine
├── Section 10: Plotly Visualizations (6 charts)
├── Section 11: PDF Report Generation
├── Section 12: Audit Logging System
├── Section 13: Streamlit UI (dark dashboard)
└── Section 14: Session State Management
```

---

## Demo Script for Judges

```python
# Demo workflow shows all capabilities
import streamlit as st

# 1. Load dataset (UCI Adult Census, 10K sample)
# 2. Train 4 models (LR, DT, RF, GBM)
# 3. Analyze fairness → get DP diff, EO diff, group accuracy
# 4. Classify risk → LOW/MEDIUM/HIGH
# 5. Check EU AI Act compliance → 8 dimensions
# 6. Detect gaps → priority-sorted list
# 7. Generate recommendations → technical actions
# 8. Mitigate bias → ExponentiatedGradient
# 9. Compute SHAP → feature importance
# 10. Compare models → leaderboard
# 11. Generate PDF → A4 compliance report
```

---

## Technologies

| Layer | Technology |
|-------|------------|
| UI | Streamlit 1.38 |
| Data Science | scikit-learn 1.5 |
| Fairness | fairlearn 0.10 |
| Explainability | SHAP 0.52 |
| Visualization | Plotly 6.0 |
| PDF | ReportLab 4.2 |

---

## EU AI Act Mapping

| Article | Requirement | FairCheck Check |
|---------|------------|----------------|
| Art. 9 | Risk documentation | Risk Documentation |
| Art. 10 | Bias testing | Bias Testing, Equalized Odds |
| Art. 12 | Audit logging | Audit Logging |
| Art. 13 | Transparency | Transparency, Explainability |
| Art. 14 | Human oversight | Human Oversight |
| Art. 15 | Performance | Model Performance |

---

## Running Locally — No Cloud Required

```bash
# Clone the repo
git clone <repo-url>
cd <repo-dir>

# Install
pip install -r requirements.txt

# Run
streamlit run faircheck.py
```

**Zero configuration** — no API keys, no cloud credentials, no external calls.

---

## Output Examples

### Fairness Gauge
- Animated gauge with risk zones (green/orange/red)
- Delta indicator showing threshold distance

### Compliance Radar
- 8-axis spider chart
- Actual vs ideal compliance overlay

### PDF Report
- Cover page with metadata
- Executive summary with PASS/FAIL metrics
- Compliance checklist by article
- Gap analysis with priority coloring
- Mitigation results with improvement %

---

## License

This is a hackathon submission. All rights reserved.

---

**FairCheck** — Making AI Fairness Auditing Simple