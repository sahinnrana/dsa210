# Residential Energy Demand and Weather Patterns

> **Tagline:** Quantifying how weather and calendar patterns drive residential electricity demand — and predicting peak-load days before they happen.

This repository implements an **end-to-end data science pipeline** to analyze how **weather conditions** and **calendar effects** (weekends, holidays, seasons) shape **residential electricity consumption** across regions in the United States.

The project is designed to fully align with DSA210 expectations: raw data ingestion, cleaning, feature engineering, **EDA**, **hypothesis testing**, and **supervised machine learning** (both **classification** and **regression**) with proper evaluation and documentation.

---

## 1. Business Question

> **How do weather and calendar patterns impact residential electricity demand, and can we proactively predict high-load days?**

From this guiding question, the project delivers two workstreams:

1. **Descriptive & Statistical Analysis**
   - Do temperature extremes (very hot/cold days) significantly increase daily residential electricity consumption?
   - Do weekends and holidays exhibit systematically different demand profiles compared to weekdays?

2. **Predictive Modeling**
   - Can we **classify peak-demand days** (top x% of demand) using exogenous drivers like weather and calendar features?
   - Can we **forecast daily energy consumption** using a regression model based on weather, lagged demand, and basic sociodemographic indicators?

---

## 2. Project Scope & Objectives

**Strategic objectives:**

- Quantify the impact of **temperature**, **degree-days**, and **calendar effects** on residential demand.
- Build interpretable, robust ML models for:
  - **Peak-day classification**
  - **Daily demand regression**
- Produce a structured, reproducible artifact set:
  - Cleaned and documented datasets
  - EDA and hypothesis testing notebooks
  - ML modeling notebook
  - Final written report compliant with DSA210 criteria

**Key deliverables:**

- Integrated dataset: **electricity load + weather + holidays + (optional) demographics**
- EDA and visualization package
- Hypothesis testing outputs with clearly defined H₀ / H₁ and interpretations
- Classification and regression models with metrics and diagnostics
- Final report (PDF/notebook section) summarizing methodology, results, limitations, and future directions

---

## 3. Data Assets

All raw assets live under `data/raw/`. Processed outputs go under `data/processed/`.

### 3.1. Dataset 1 – Residential Electricity Load

- **Name:** Daily Residential Electricity Consumption (Region-Level)
- **Source:** Public open data (e.g., U.S. EIA / ISO / Kaggle-aggregated load dataset)
- **Typical schema:**
  - `date`  
  - `region` (state/utility/ISO area)  
  - `residential_demand` or `daily_consumption` (kWh / MWh)  
- **Usage:** Core dependent variable for both EDA and modeling.

---

### 3.2. Dataset 2 – Weather / Climate

- **Name:** Daily Weather by Region
- **Source:** Public weather repositories or Kaggle
- **Typical schema:**
  - `date`, `region` (or mapping from station to region)  
  - `avg_temp`, `min_temp`, `max_temp`  
  - `humidity`, `wind_speed`, `precipitation`, `snowfall`  
  - `heating_degree_days` (HDD), `cooling_degree_days` (CDD)
- **Usage:** Primary driver features for both statistical tests and ML.

---

### 3.3. Dataset 3 – Calendar & Holidays

- **Name:** US Holidays and Calendar Features
- **Source:** Kaggle / official holiday datasets
- **Typical schema:**
  - `date`  
  - `holiday_name`  
  - `is_holiday` (0/1)
- **Usage:** Generate `is_weekend`, `is_holiday`, `season`, and special periods (e.g., Christmas week).

---

### 3.4. Dataset 4 – Demographic Enrichment (Optional)

- **Name:** Regional Demographics (Context)
- **Source:** US Census / Kaggle
- **Typical schema:**
  - `region`  
  - `population`  
  - `median_income`  
  - `urbanization_index`
- **Usage:** Normalization (e.g., per capita demand) and additional explanatory signals for the regression model.

---

## 4. Data Pipeline

### 4.1. Raw Ingestion

Implemented in `src/data_prep.py` and/or notebooks:

1. Load:
   - `electric_load.csv`
   - `weather_daily.csv`
   - `us_holidays.csv`
   - `demographics.csv` (optional)
2. Convert `date` columns to `datetime`.

---

### 4.2. Data Cleaning

**Electric load:**

- Drop duplicates and obviously invalid rows (negative or impossible consumption).
- Handle missing `residential_demand` lines (drop if sparse, impute if systematic).
- If data is hourly:
  - Aggregate to daily:
    - `daily_consumption` = sum of hourly loads
    - `peak_hourly_load` = max hourly load per day

**Weather:**

- Normalize region identifiers to match load data.
- Impute missing weather features:
  - Median per region or time-based interpolation.
- Compute derived fields (if not already provided):
  - HDD, CDD from `avg_temp` and base temperature threshold.

**Holidays:**

- Ensure unique row per date.
- Map to `is_holiday`.

**Demographics:**

- Clean region names and ensure one row per region.

---

### 4.3. Aggregation & Integration

**Unit of analysis:** `region × date` (daily grain)

1. Aggregate load to daily level per region.
2. Aggregate / map weather to the same region-date grain.
3. Join:
   - `load_daily` ⨝ `weather_daily` on (`region`, `date`)
   - ⨝ `holidays` on `date`
   - ⨝ `demographics` on `region`

Persist the merged dataset into `data/processed/merged_region_daily.parquet` (or `.csv`).

---

### 4.4. Feature Engineering

Key engineered fields:

- **Temporal:**
  - `year`, `month`, `day_of_week`
  - `is_weekend` (Saturday/Sunday)
  - `is_holiday`
  - `season` (Winter/Spring/Summer/Fall)

- **Weather-Intensity:**
  - `HDD`, `CDD`
  - `is_extreme_cold` (HDD above threshold)
  - `is_extreme_heat` (CDD above threshold)
  - `is_precip_day` (rain/snow present)

- **Demand metrics:**
  - `DemandPerCapita = daily_consumption / population`
  - `PeakDayFlag = 1` if `daily_consumption` > region-specific 90th percentile; else `0`.

- **Lagged/rolling features:**
  - `lag1_consumption` (yesterday’s demand)
  - `lag7_consumption`
  - `rolling_7d_avg` (7-day moving average)

These are implemented in `src/features.py` and/or directly inside `notebooks/`.

---

## 5. Exploratory Data Analysis (EDA)

Documented in **`notebooks/01_eda.ipynb`**.

Scope:

- **Descriptive stats:**
  - `.describe()` and `.skew()` for `daily_consumption`, `DemandPerCapita`, HDD, CDD, etc.
- **Time-series views:**
  - Demand over time by region.
  - Seasonal breakdowns by `month`/`season`.
- **Segmented boxplots:**
  - `daily_consumption` by `season`
  - `daily_consumption` by `is_weekend`, `is_holiday`
  - `daily_consumption` by `is_extreme_heat/cold`
- **Correlation maps:**
  - Heatmaps for correlations between demand and weather/calendar/demographic features.
- **Scatter analyses:**
  - `daily_consumption` vs `HDD` / `CDD` (expected U-shaped temperature-demand relationship).

Outcome: quantified view of how demand moves with temperature and calendar signals.

---

## 6. Hypothesis Testing

Implemented in **`notebooks/02_hypothesis_testing.ipynb`** with explicit H₀/H₁ and α-levels.

### Hypothesis 1 – Temperature Extremes

- **H₀:** Mean daily consumption on mild days equals mean daily consumption on extreme temperature days.  
- **H₁:** Mean daily consumption differs between mild and extreme days.

**Approach:**

- Segment days into `mild` (low HDD/CDD) vs `extreme` (high HDD/CDD).
- Use:
  - Two-sample **t-test** if normality / variance assumptions are reasonably satisfied.
  - Otherwise **Mann–Whitney U** test.
- Report test statistic, p-value, and decision at α = 0.05.

---

### Hypothesis 2 – Weekdays vs Weekends/Holidays

- **H₀:** Residential daily consumption is the same on weekdays as on weekends/holidays.  
- **H₁:** Weekends/holidays exhibit different mean residential daily consumption.

**Approach:**

- Group by `is_weekend` and `is_holiday`.
- Use t-tests or single-factor **ANOVA** across categories (weekday vs weekend vs holiday).
- Interpret sign and magnitude of differences.

---

### Hypothesis 3 – Correlation with Degree Days

- **H₀:** There is no linear correlation between residential electricity consumption and HDD/CDD.  
- **H₁:** There is a significant correlation between consumption and HDD/CDD.

**Approach:**

- Compute **Pearson** and **Spearman**:
  - `daily_consumption` vs `HDD`
  - `daily_consumption` vs `CDD`
- Report correlation coefficients, p-values, and classify effect sizes (weak/moderate/strong).

---

## 7. Machine Learning Models

All ML experiments are encapsulated in **`notebooks/03_ml_models.ipynb`**.

### 7.1. Classification – Peak-Demand Flag

**Objective:** Predict if a region-date will be a **peak-demand day**.

- **Target:** `PeakDayFlag` (0/1)
- **Features:**
  - Time: `month`, `day_of_week`, `is_weekend`, `is_holiday`, `season`
  - Weather: `HDD`, `CDD`, `avg_temp`, `precipitation`, `is_extreme_heat`, `is_extreme_cold`
  - Lagged demand: `lag1_consumption`, `rolling_7d_avg`
  - (Optional) Demographics: `population`, `urbanization_index`

- **Preprocessing:**
  - Missing-value imputation.
  - One-hot encoding of categorical variables (e.g., `season`).
  - Train–test split (preferably stratified by target, with temporal awareness).
  - Handle class imbalance with **SMOTE** or class weights.

- **Model:**
  - `RandomForestClassifier` with tuned hyperparameters (`n_estimators`, `max_depth`, etc.)

- **Evaluation:**
  - **Accuracy**
  - **Precision/Recall/F1** for the positive (peak) class
  - **ROC–AUC**
  - Confusion matrix
  - Feature importance plots for interpretability.

---

### 7.2. Regression – Daily Demand Forecast

**Objective:** Predict numeric daily residential demand (or per-capita demand).

- **Target:** `daily_consumption` (or `DemandPerCapita`)
- **Features:**
  - Same exogenous and lagged variables as classification, without target leakage.

- **Models:**
  - Baseline: `LinearRegression`
  - Optional: `RandomForestRegressor` or regularized models (Ridge/Lasso) for nonlinear effects and overfitting control.

- **Evaluation:**
  - **R²**
  - **MAE**
  - **RMSE**
- **Diagnostics:**
  - Actual vs predicted scatter plot.
  - Residual distribution and residual vs fitted plots.

---

## 8. Limitations & Future Work

**Known limitations:**

- Daily regional aggregation obscures intra-day patterns and local grid constraints.
- Residential vs total load separation may be imperfect depending on data source.
- Behavioral and policy drivers (e.g., tariff changes, demand response programs) are not explicitly modeled.
- Causal inference is outside scope; we describe associations, not guaranteed causality.

**Future work:**

- Extend to **hourly** resolution to capture intraday peaks.
- Incorporate **tariff and pricing** data for demand elasticity analysis.
- Integrate **weather forecasts** to generate forward-looking risk scores.
- Experiment with time-series models (e.g., ARIMA, LSTM) for sequential demand forecasting.

---

## 9. Tech Stack

- **Language:** Python
- **Core libraries:**
  - `pandas`, `numpy` – data handling
  - `matplotlib`, `seaborn` – visualization
  - `scikit-learn` – ML and preprocessing
  - `imbalanced-learn` – imbalance handling (SMOTE)
  - `scipy` – statistical tests
- **Tooling:**
  - Jupyter Notebook
  - VSCode (optional)
  - Git & GitHub for version control and submission

---

## 10. Repository Structure

Proposed repo layout:

```text
.
├── data/
│   ├── raw/
│   │   ├── electric_load.csv
│   │   ├── weather_daily.csv
│   │   ├── us_holidays.csv
│   │   └── demographics.csv         # optional
│   └── processed/
│       └── merged_region_daily.parquet
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_hypothesis_testing.ipynb
│   └── 03_ml_models.ipynb
├── src/
│   ├── __init__.py
│   ├── data_prep.py
│   ├── features.py
│   └── models.py
├── reports/
│   └── final_report.pdf
├── requirements.txt
└── README.md
