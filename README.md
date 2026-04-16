# 🏠 Connecticut Housing Intelligence System

A machine learning system for predicting residential sale prices and ranking 
investment opportunities across 170 Connecticut towns, built on 1.1M+ real 
estate transactions from 2001–2023.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [How to Run](#how-to-run)
- [Streamlit App](#streamlit-app)
- [Future Improvements](#future-improvements)
- [Technologies Used](#technologies-used)

---

## Overview

Connecticut housing sale records contain inconsistent free-text remarks, 
invalid transactions, and incomplete property data — making price prediction 
unreliable without careful preprocessing.

This project:
1. Cleans and categorizes 1.1M+ messy transactions using an OOP regex-based 
   remarks classifier
2. Engineers features capturing market dynamics, property characteristics, 
   and flip patterns
3. Trains an XGBoost model with Bayesian hyperparameter optimization
4. Builds a composite investment score ranking all 170 CT towns
5. Delivers findings through an interactive Streamlit application

---

## Key Results

| Metric | Value |
|--------|-------|
| Final Model RMSE | $115,231 |
| RMSE with complete property data | $101,829 |
| Predictions within $50K | 60.9% |
| Predictions within $100K | 82.3% |
| A/B Test p-value (XGBoost vs Ridge) | 0.0263 |
| Training data size | 1,119,333 rows |
| Towns ranked | 170 |

---

## Project Structure

```
ct_housing/
│
├── Connecticut_House_Data.ipynb   # Main analysis notebook
│
├── ct_housing_app/                # Streamlit application
│   ├── app.py                     # Main app file
│   ├── utils.py                   # Shared utility functions
│   ├── requirements.txt           # Python dependencies
│   └── models/                    # Saved model artifacts
│       ├── final_model.pkl        # Trained XGBoost model (4.8 MB)
│       ├── preprocessor.pkl       # Sklearn preprocessing pipeline
│       ├── town_means.pkl         # Target encoding lookup table
│       ├── global_mean.pkl        # Fallback encoding for unknown towns
│       ├── town_metrics.csv       # Investment scores for 170 towns
│       └── comps_data.csv         # 487K comparable sales (2015–2024)
│
└── README.md
```

---

## Dataset

| Source | Description |
|--------|-------------|
| [CT Office of Policy and Management](https://catalog.data.gov/dataset/real-estate-sales-2001-2018) | 1.1M real estate transactions 2001–2023 |
| HomeHarvest (scraped) | Property details (beds, sqft, garage) for select towns 2015–2024 |

**Key columns:**

| Column | Description | Missing % |
|--------|-------------|-----------|
| Sale Amount | Actual sale price (target) | 0% |
| Assessed Value | Town-assessed property value | 0% |
| Town | CT town name (170 unique) | 0% |
| Property Type | Single Family, Condo, etc. | 33% |
| Sqft | Square footage | 76% |
| Beds | Bedrooms | 76% |
| Year Built | Year constructed | 76% |
| OPM Remarks | Free-text sale notes (79K unique) | 99% |
| Assessor Remarks | Free-text assessor notes | 84% |

---

## Methodology

### 1. Data Cleaning
- Fixed malformed date formats and Sales Ratio strings
- Removed invalid sales (duplicates, vacant lots, non-market transfers)
- Capped sale prices at $2M to remove outliers skewing training
- Calculated `Days_To_Sell` from list year to recorded date

### 2. Remarks Categorization
Built an OOP `RemarksProcessor` class using regex pattern matching to 
normalize 79,000+ unique assessor remarks and 7,200+ OPM remarks into 
10 meaningful categories:

| Category | Example Remarks |
|----------|----------------|
| RENOVATED/REMODELED | "TOTAL RENOVATION PER MLS", "COMPLETELY UPDATED" |
| GOOD SALE | "GOOD SALE PER MLS", "GOOD SALE PER TOWN SITE" |
| FORECLOSURE / BANK SALE | "SHORT SALE PER MLS", "BANK OWNED" |
| ESTATE / FAMILY / RELOCATION | "ESTATE SALE", "FAMILY SALE", "RELOCATION" |
| NEW / NON-RESIDENTIAL | "NEW CONSTRUCTION", "MOBILE HOME" |

### 3. Feature Engineering
| Feature | Description |
|---------|-------------|
| `town_encoded` | Target-encoded town median log price (CV-safe) |
| `town_yoy_growth` | Year-over-year price growth per town at time of sale |
| `Flip_Candidate` | 1 if property resold within 3 years at 10%+ profit |
| `Home_Age` | Years since construction at time of listing |
| `Is_Older_Home` | 1 if built before 1980 |
| `Luxury_Home` | 1 if 5+ beds and 4000+ sqft |

### 4. Modeling
- **Algorithm:** XGBoost with `reg:squarederror` objective
- **Target transformation:** `np.log1p(Sale Amount)` for normalization
- **Hyperparameter tuning:** Bayesian Optimization (30 iterations)
- **Preprocessing:** Median imputation + StandardScaler for numeric, 
  OHE for categorical

### 5. Evaluation
- **A/B Test:** Paired t-test comparing Ridge vs XGBoost across 5 CV folds
- **Stratified Modeling:** Separate models for rows with/without property details
- **Residual Analysis:** Error breakdown by town and year
- **SHAP:** Feature importance and prediction explainability

---

## Key Findings

### Model Performance
- Assessed Value and town identity are the two strongest predictors
- Market timing (year) is the 3rd strongest signal — capturing COVID-era price surge
- Flip candidates predict 15% higher sale prices on average
- Data completeness is the primary bottleneck — 76% of records missing Sqft/Beds

### Investment Score
Towns are ranked using a composite score:
- **35%** Long-term YoY price growth
- **30%** Recent momentum (2021–2023)
- **20%** Flip rate (investor activity)
- **15%** Market liquidity (days to sell)

**Top investment markets:** Hartford, Waterbury, Meriden, Putnam  
**Lowest investment returns:** Greenwich, Darien, New Canaan, Wilton

> Counterintuitive finding: CT's Gold Coast luxury towns rank at the 
> bottom — high absolute prices don't correlate with high investment returns

### Sale Type Analysis
- Foreclosure sales average **27% below** Good Sale baseline
- Renovated properties sell **9% above** baseline
- New construction sells **34% above** baseline

### A/B Test Results
- XGBoost significantly outperforms Ridge (p=0.0263)
- XGBoost RMSE std: $1,330 across 5 folds (highly consistent)
- Ridge RMSE std: $1,344,481 (collapsed on fold 2 — $3.8M RMSE)
- Confirms non-linear relationships require tree-based models

### Residual Analysis
- Model has a +$24K positive bias (slight overprediction)
- Luxury markets (Westport, Greenwich) show highest error — 
  assessed value doesn't capture location premium
- Consistent suburban markets (Wolcott, Plymouth) show lowest error

---

## How to Run

### Prerequisites
```bash
conda create -n housing_project python=3.11
conda activate housing_project
pip install -r requirements.txt
```

### Run the Streamlit App
```bash
cd ct_housing_app
streamlit run app.py
```

### Run the Notebook
```bash
cd ct_housing
jupyter notebook
```
Open `Connecticut_House_Data.ipynb` and run all cells.

> **Note:** First run takes ~30 minutes due to preprocessing 1.1M rows. 
> Subsequent runs load cached arrays from disk.

---

## Streamlit App

https://ct-housing-intelligence-9pn7hqumhhv4qxn99aifcs.streamlit.app/

The app has three tabs:

**💰 Price Predictor**
- Input town, assessed value, property details
- Returns predicted price with confidence interval
- Shows town investment context (score, YoY growth, days to sell)

**📈 Investment Score**
- Interactive bar chart of top N towns by investment score
- Filter by minimum transaction count
- Full sortable rankings table

**🔍 Comparable Sales**
- Search 487K sales by town, property type, year range, price range
- Price distribution histogram
- Median price over time chart
- Full comparable sales table

---

## Future Improvements

| Improvement | Expected Impact |
|-------------|----------------|
| Add school district ratings | High — major pricing signal |
| Add Walk Score / transit proximity | Medium — especially for urban towns |
| Collect Sqft/Beds for missing 76% | High — RMSE likely drops to ~$80K |
| Add lot size data | Medium |
| Implement cluster-based models | Medium — separate luxury vs affordable |
| Time series forecasting with Prophet | Medium — town-level price predictions |

---

## Technologies Used

| Category | Tools |
|----------|-------|
| Data manipulation | pandas, numpy |
| Machine learning | scikit-learn, XGBoost |
| Hyperparameter tuning | bayesian-optimization |
| Explainability | SHAP |
| Visualization | matplotlib, seaborn, plotly |
| Statistical testing | scipy |
| App framework | Streamlit |
| Model persistence | joblib |

---

## Author
**Mario Perez**  
MariosGIT3526 | www.linkedin.com/in/
mario-perez-34b23b3b6
  

*Built as a portfolio project demonstrating end-to-end data science:
data cleaning, feature engineering, ML modeling, statistical testing,
and interactive deployment.*
