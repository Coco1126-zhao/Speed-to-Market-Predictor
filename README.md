# Project Viability & Speed-to-Market Predictor

> **Modo Energy Open Tech Challenge · March 2026**

**83.8% of renewable energy projects that enter a US interconnection queue never get built.**

This tool predicts, at the moment of application, which projects will make it, and how long they will take. It is trained on 24,690 historical projects across all major US ISOs and produces a single viability score that any developer, investor, or analyst can act on before spending a dollar on interconnection studies.

---

## The Problem

The US grid is in the middle of an unprecedented buildout. Texas alone entered 2026 with **14 GW of commercially operational grid-scale battery capacity**, up from near zero in 2020. SPP's interconnection queue now holds over **160 GW of announced projects** through 2031, with 32% including battery storage.

<p align="center">
  <img src="outputs/modo_ercot_bess.png" width="47%"/>
  <img src="outputs/modo_spp_queue.png" width="47%"/>
</p>

*Source: Modo Energy · ERCOT Annual Buildout Report (2026) · SPP Battery Buildout Outlook (2025)*

But the growth headline masks a structural failure: the interconnection queue itself has become the primary bottleneck to the energy transition. Modo Energy's own ERCOT research documents the consequence directly: new battery applications in Texas are falling, driven by falling merchant revenues (from $192/kW in 2023 to $43/kW in 2024) and rising costs from tariff and ITC uncertainty. The question every developer and investor is asking is not *"can we build this?"* but *"will this actually get built, and when?"*

**That question currently has no data-driven answer at the point of application.** Developers rely on gut instinct, ISO-level averages, or expensive consultants. No tool exists to score a project's viability the moment it enters the queue, before study costs are committed.

This is exactly the kind of market opacity Modo is built to resolve. Their platform already benchmarks operational revenue and forecasts buildout trajectories. A viability predictor extends that lens upstream into the development pipeline, before interconnection costs are incurred. Their ERCOT subscriber research explicitly asks: *"How much of the 200+ GW queue is likely to achieve commercial operation, and when?"* This project builds the model that answers it.

---

## What Was Built

A **two-model ML system** trained on the LBNL Interconnection Queue dataset (1970–2024):

| Model | Task | Validation Performance |
|---|---|---|
| **XGBoost Classifier** | P(project reaches commercial operation) | AUC-ROC = 0.851 |
| **XGBoost Regressor** | Expected queue duration in months | RMSE = 14.3 mo (vs 21.9 mo ISO baseline) |

The two models are combined into a single **viability score**: high completion probability + short predicted duration = fast mover worth prioritising. The regressor beats the ISO-mean baseline by **35%**, meaning it adds real signal beyond simply knowing which region a project is in.

---

## Key Findings

### 1. Two signals at submission time dominate both models

The feature importance charts below show what drives completion probability (left) and queue duration (right). Reading them side by side reveals the two most commercially actionable features.

![Classifier Feature Importance](outputs/clf_feature_importance.png)
![Regressor Feature Importance](outputs/reg_feature_importance.png)

**Service type (NRIS/ERIS) is the strongest completion signal.** Projects requesting both Network Resource and Energy Resource Interconnection Service simultaneously are the highest-completion-rate applicants in the dataset. NRIS/ERIS applicants require firm network access rights, not energy-only delivery, signalling that they have committed capital to a more complex study because they need the transmission to underpin a real project. It is visible on day one and costs nothing to observe.

**Filing a proposed online date ranks #6 for completion and #3 for duration** and the only feature to appear prominently in both models. Developers who submit with a target commercial operation date are significantly more likely to complete and, among those that do complete, move through the queue faster. This is a commitment signal: it indicates an active PPA negotiation, a financing timeline, or a regulatory deadline. A platform that tracks whether this field was populated at submission gains an immediate enrichment layer on its viability scores.

Other top classifier features worth noting: **`iso_region_SPP`** (SPP applicants complete at higher rates than most ISOs), **`tech_bucket_Solar`** and **`tech_bucket_Gas`** (technology type interacts with vintage era), **`capacity_bucket_small`** (smaller projects face lower study complexity), and **`is_slow_iso`** (PJM and MISO applicants face a structurally harder path due to serial study processes).

---

### 2. Hybrid projects take measurably longer, even when they complete

`tech_bucket_Hybrid` is the dominant duration predictor at 15.6% of regressor feature importance, nearly double the next feature. Solar+Battery and Wind+Battery projects that successfully complete interconnection take significantly longer than single-technology projects.

This matters directly for Modo's customers. As SPP's queue tilts toward hybrids (32% of queued capacity includes BESS) and ERCOT's battery pipeline continues to scale, developers and investors building financial models for these projects need quantified planning buffers, not ISO-level averages. On a median completed project duration of 42.5 months, the regressor's 14.3-month RMSE represents a genuine narrowing of uncertainty: 35% better than the ISO-mean benchmark.

`proposed_lead_years` is the second strongest duration predictor. The relationship is near-linear: every additional year of planned lead time adds roughly 10–15 months of actual queue time. Developers who plan longer runways do so because they know they are facing a complex multi-technology study environment.

![Regressor SHAP Dependence](outputs/shap_reg_dependence.png)

---

### 3. Where a project is matters as much as what it is

ISO region is among the top predictors in both models.

![Fast Mover Profile](outputs/fast_mover_profile.png)

**SPP (25%) and ISO-NE (20%)** show the highest fast-mover rates. SPP's central plains geography hosts fewer competing projects per transmission node and has historically maintained a less congested, more predictable queue process. ISO-NE's compact geography and strong historical completion discipline produce a similar pattern.

**CAISO, NYISO, and ERCOT all show near-zero fast-mover rates.** Each case has a different explanation:

For **CAISO and NYISO**, this is structurally expected and well-documented. Both markets have seen thousands of solar and battery projects compete for congested coastal transmission capacity. Queue backlogs are severe, study timelines have extended significantly, and withdrawal rates are among the highest in the country. The model correctly identifies this pattern.

**ERCOT is the most interesting case.** Despite being the most successful battery market in the US, with 14 GW of commercially operational BESS entering 2026, ERCOT's fast-mover rate is near zero. The explanation is not that ERCOT is a bad market; it is that ERCOT's success has attracted so much new capital that the queue ratio has collapsed. The pipeline now holds 200+ GW of new applications against a grid that can realistically absorb a fraction of that. Modo's own ERCOT research documents this exactly: the fall in new applications reflects tempering investor expectations as merchant revenues drop and queue timelines lengthen. Our model is a product of its training data (pre-2019 ERCOT, which was far more manageable), so ERCOT predictions should be read as lower bounds, not current estimates. This is a known limitation.

**On the technology fast-mover chart:** Nuclear and Hydro dominating the top is a historical artefact. Those are small samples of pre-2000 legacy plants that completed in a completely different regulatory environment. They are not investable signals. For Modo's core markets (batteries, solar, and wind), the commercially relevant question is not whether these technologies rank below gas in aggregate, but how they rank *against each other* by region and vintage. A battery project in SPP filed in 2019 scores very differently from the same project in CAISO in 2023. The viability score captures that difference; the technology fast-mover chart alone cannot.

The completion rate by ISO below shows the model's predicted rates against actuals. Predicted rates are systematically higher across all ISOs because the model was trained on pre-2019 cohorts where 21.8% of projects completed, and is then applied to the full dataset including recent projects that are still active but not yet resolved. The ranking across ISOs is preserved, which is what matters for relative viability scoring.

![Completion Rate by ISO](outputs/clf_completion_rate_by_iso.png)

---

### 4. What actually drives completion probability

The SHAP beeswarm below is the most information-dense chart in this analysis. Here is how to read it: each dot is one project, its horizontal position shows how much that feature pushed completion probability up (right) or down (left), and its colour shows whether the project had a high (red) or low (blue) value for that feature.

![SHAP Classifier Beeswarm](outputs/shap_clf_beeswarm.png)

Three patterns matter most for a practitioner:

**Large projects complete less often than small ones.** `log_capacity_mw` is the top feature, and red dots (large projects) cluster on the left, meaning large capacity *reduces* predicted completion probability. Large projects trigger more extensive network upgrade requirements, face higher capital hurdles, and attract more regulatory scrutiny. Small projects (blue, right side) are quicker to study and cheaper to develop, so developers who commit to them tend to follow through.

**Entering the queue before 2019 was a genuine advantage.** `queue_year` shows that recent entrants (red = high year value, left side) have lower predicted completion rates. This is not a modelling artefact. It reflects a real structural shift. Before 2019, the average ISO queue had far fewer competing projects per transmission node. Today's queues are overwhelmed. The same project in the same ISO has worse odds now than it would have had a decade ago, and the model has learned this from two decades of outcome data.

**Queue congestion adds a real but secondary penalty.** `log_queue_backlog` nudges completion probability down when congestion is high. But the effect is smaller than regional identity or service type. Which ISO you are in matters more than how busy it happens to be at the moment of submission.

---

### 5. Case Studies: the model at the project level

**Fast-mover (high completion probability)**

A large Fossil project in ISO-NE where raw scale is the dominant positive driver. ISO-NE's strong completion discipline supports the prediction further. The `is_hybrid` flag introduces a small headwind, but the overall viability score is high.

![SHAP Waterfall Fast Mover](outputs/shap_waterfall_fast_mover.png)

**High-risk project (low completion probability)**

A project where very large capacity and high queue congestion combine to push the prediction far below the model baseline. Every feature in this waterfall pulls in the same direction: this project entered a congested ISO at large scale with a profile that historically precedes withdrawal.

![SHAP Waterfall Hard Case](outputs/shap_waterfall_hard_case.png)

---

## Model Performance

### Classifier

| Split | AUC-ROC | AUC-PR | n | Pos. Rate |
|---|---|---|---|---|
| Train (pre-2019) | 0.888 | 0.702 | 16,270 | 21.8% |
| Val (2019–2021) | 0.851 | 0.384 | 5,213 | 7.1% |
| Test (2022–2024) | 0.771 | 0.114 | 3,207 | 2.3% |

AUC-ROC of 0.851 means the model correctly ranks 85% of completed vs withdrawn project pairs by probability. The AUC-PR drop on the test set (0.114) reflects data immaturity: 2022–2024 projects have not had time to complete, making the observed positive rate (2.3%) lower than the true long-run rate. Validation metrics are the appropriate benchmark.

### Regressor (completed projects only)

| Split | RMSE | MAE | R² | ISO-Baseline RMSE |
|---|---|---|---|---|
| Train | 13.3 mo | 8.9 mo | 0.813 | 28.8 mo |
| Val | 14.3 mo | 11.2 mo | 0.344 | 21.9 mo |

35% improvement over ISO-mean baseline on the validation set. The scatter below shows strong calibration in the 20–60 month range where most projects fall.

![Regressor Actual vs Predicted](outputs/reg_actual_vs_predicted.png)

---

## Data

**Source:** [LBNL Interconnection Queue ("Queued Up"), 2024 release](https://emp.lbl.gov/publications/queued-2025-edition-characteristics)

See `data/downloading from LBNL` for instructions. The raw Excel file is not committed to this repo due to size.

**Coverage:** 36,441 projects across all major US ISOs, 1970–2024
**Modelling subset:** 24,690 completed or withdrawn projects (active and suspended excluded from supervised targets)

---

## Features

All features use only information observable at or before the queue entry date to prevent data leakage.

| Feature | Type | Description |
|---|---|---|
| `log_capacity_mw` | Numeric | Log-transformed MW capacity |
| `log_queue_backlog` | Numeric | Log of 3-year rolling project count in same ISO |
| `proposed_lead_years` | Numeric | Proposed online year minus queue entry year |
| `queue_year`, `queue_month`, `queue_quarter` | Numeric | Temporal entry features |
| `is_hybrid` | Binary | Whether project combines multiple technologies |
| `is_slow_iso` | Binary | PJM or MISO flag (historically longer queues) |
| `is_nris` | Binary | Network Resource Interconnection Service flag |
| `has_proposed_date` | Binary | Developer filed a target online date |
| `post_ferc_2003`, `post_ferc_2023` | Binary | Regulatory reform era indicators |
| `itc_active`, `ptc_bonus_period`, `ira_era` | Binary | Federal subsidy policy cycle flags |
| `iso_region` | Categorical | ISO/RTO region (9 categories) |
| `tech_bucket` | Categorical | Technology type (Solar, Wind, Battery, Hybrid, etc.) |
| `capacity_bucket` | Categorical | Small / Mid / Large / Utility size tier |
| `service_type` | Categorical | NRIS / ERIS / NRIS+ERIS / Other |
| `queue_decade` | Categorical | Decade of queue entry |

---

## Limitations

**ERCOT recent dynamics are underrepresented.** The model was trained on pre-2019 data. ERCOT's queue has since exploded from a manageable pipeline to 200+ GW of applications. ERCOT viability scores should be read as lower bounds, not current estimates, and would benefit from retraining on post-2020 completed outcomes.

**Survivorship bias in legacy technology types.** Nuclear and Hydro rank highest in the fast-mover technology profile due to pre-2000 legacy plants that completed in a fundamentally different environment. Forward-looking signals for modern clean energy portfolios come from Solar, Wind, Battery, and Hybrid categories.

**Correlation, not causation.** SHAP importance is associative. High importance of ISO region reflects correlated structural differences: regulatory environment, grid topology, market design, rather than a single causal mechanism.

**Test set maturity.** AUC-PR of 0.114 on the test set reflects data immaturity, not model failure. AUC-ROC of 0.851 on the validation set is the appropriate benchmark.

**Missing duration data.** Approximately 50% of projects lack a recorded end date, limiting the regression training set to 1,722 rows. A survival model (Cox PH or accelerated failure time) would allow active projects to be included as censored observations and is a natural next step.

**FERC Order 2023.** Cluster study reforms took effect in late 2023. Their full impact on queue dynamics will only be measurable with several more years of outcome data.

---

## Repo Structure

```
Speed-to-Market-Predictor/
├── README.md
├── how_to_run_colab_version.py    <- full pipeline as Colab-ready cells
├── data/
│   └── downloading from LBNL     <- data source and download instructions
├── src/
│   ├── data_loader.py             <- loads and cleans LBNL xlsx, builds targets
│   ├── features.py                <- feature engineering and sklearn preprocessor
│   ├── models.py                  <- trains and evaluates both XGBoost models
│   └── shap_analysis.py           <- SHAP attribution and all visualisations
├── outputs/                       <- all generated plots
└── Xueru Zhao-Resume-20260220.pdf
```

---

## How to Run

**Option A: Google Colab (recommended)**

Open `how_to_run_colab_version.py` and run cells sequentially. The file is structured as annotated Colab cells and handles Google Drive mounting automatically.

**Option B: Local**

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib openpyxl

python src/data_loader.py  data/raw/lbnl_ix_queue_data_file_thru2024.xlsx
python src/models.py       data/raw/lbnl_ix_queue_data_file_thru2024.xlsx
python src/shap_analysis.py
```

`features.py` is imported automatically by both `models.py` and `shap_analysis.py` and does not need to be run directly.

---

## AI Workflow

Claude (Anthropic) was used as a coding collaborator to generate the Python code across `data_loader.py`, `features.py`, `models.py`, and `shap_analysis.py`, and to debug issues during execution. The problem framing, data source selection, feature design decisions, and all findings interpretations are my own work.
