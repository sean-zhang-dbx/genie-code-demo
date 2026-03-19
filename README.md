# Genie Code Demo: End-to-End Data Lifecycle on Databricks

An end-to-end demonstration of the Databricks data lifecycle — from raw data exploration, through a data engineering pipeline, to ML model training, and finally serving insights via an AI/BI Genie space. All driven entirely by code.

## Dataset

**`samples.nyctaxi.trips`** — ~22K NYC yellow taxi trip records from January–February 2016.

| Column | Type | Description |
|---|---|---|
| `tpep_pickup_datetime` | timestamp | Trip pickup time |
| `tpep_dropoff_datetime` | timestamp | Trip dropoff time |
| `trip_distance` | double | Trip distance in miles (0.0–30.6) |
| `fare_amount` | double | Fare in USD (-8.0–275.0) |
| `pickup_zip` | int | Pickup ZIP code (~120 unique) |
| `dropoff_zip` | int | Dropoff ZIP code (~193 unique) |

---

## Demo Outline

### Phase 1: Data Exploration

Profile the raw `samples.nyctaxi.trips` data to understand its shape, quality, and characteristics.

- [ ] **Schema & volume** — Inspect column types, row count, and table metadata
- [ ] **Summary statistics** — Compute min/max/avg/stddev for numeric columns (`trip_distance`, `fare_amount`)
- [ ] **Temporal coverage** — Determine date range and trip volume by day/hour
- [ ] **Categorical distributions** — Analyze top pickup/dropoff ZIP codes and trip patterns
- [ ] **Data quality assessment** — Identify nulls, anomalies (negative fares, zero-distance trips), and outliers
- [ ] **Visualizations** — Fare distribution histogram, trip distance vs. fare scatter plot, hourly trip volume

### Phase 2: Data Engineering Pipeline

Clean and transform the raw data into an analysis-ready Delta table with engineered features.

- [ ] **Create target schema** — `CREATE SCHEMA IF NOT EXISTS` for the demo catalog/schema
- [ ] **Data cleaning** — Remove invalid records:
  - Negative or zero fares
  - Zero-distance trips
  - Unreasonably long trips (>100 miles) or high fares (>$500)
  - Null values in critical columns
- [ ] **Feature engineering** — Derive new columns:
  - `trip_duration_minutes` — computed from pickup/dropoff timestamps
  - `pickup_hour` — hour of day (0–23)
  - `pickup_day_of_week` — day name (Monday–Sunday)
  - `is_weekend` — boolean weekend flag
  - `speed_mph` — average speed (distance / duration)
  - `fare_per_mile` — fare efficiency metric
- [ ] **Outlier filtering** — Remove trips with impossible speeds (>100 mph) or durations (<1 min or >3 hrs)
- [ ] **Write to Delta** — Save the cleaned, feature-enriched table as a managed Delta table
- [ ] **Validation** — Compare row counts before/after, verify no nulls in engineered columns

### Phase 3: ML Model — Fare Prediction

Train a regression model to predict `fare_amount` from trip features, logged with MLflow.

- [ ] **Feature selection** — Select predictive features:
  - `trip_distance`, `trip_duration_minutes`, `pickup_hour`, `pickup_day_of_week`, `is_weekend`, `speed_mph`
- [ ] **Train/test split** — 80/20 split with reproducible seed
- [ ] **Model training** — Train a LightGBM regressor (or scikit-learn GradientBoosting)
- [ ] **MLflow experiment tracking** — Log:
  - Model parameters
  - Metrics: RMSE, MAE, R²
  - Feature importance plot
  - Model artifact with signature and input example
- [ ] **Evaluation** — Visualize predictions vs. actuals, residuals distribution
- [ ] **Register model** — Register the best model to Unity Catalog for serving

### Phase 4: AI/BI Genie Space

Create a Genie space on the cleaned data so business users can ask natural-language questions about taxi trips.

- [ ] **Discover resources** — Find an eligible SQL warehouse (pro or serverless)
- [ ] **Define the space** — Configure:
  - Title, description, and sample questions
  - SQL expressions for key metrics (avg fare, total trips, avg speed)
  - Text instructions for business context
  - Column configs with prompt matching enabled
- [ ] **Validate configuration** — Run the config validator before creating
- [ ] **Create via API** — Use the Databricks SDK to create the Genie space programmatically
- [ ] **Verify** — Open the space and test with sample questions

---

## Project Structure

```
genie-code-demo/
├── README.md                      # This file — demo plan and documentation
├── 01_data_exploration.py         # Phase 1: Raw data profiling and visualization
├── 02_data_engineering.py         # Phase 2: Cleaning, feature engineering, Delta write
├── 03_ml_model.py                 # Phase 3: Model training, MLflow logging, registration
└── 04_genie_space.py              # Phase 4: Genie space creation via API
```

## Prerequisites

- Databricks workspace with Unity Catalog enabled
- Access to the `samples` catalog (built-in)
- A catalog/schema with write permissions for the cleaned table and ML model
- A pro or serverless SQL warehouse (for Genie space)
- Python packages: `lightgbm`, `scikit-learn`, `mlflow` (pre-installed on Databricks)

## Getting Started

1. Clone this repo into your Databricks workspace
2. Run the notebooks in order: `01` → `02` → `03` → `04`
3. Each notebook is self-contained — just attach to a cluster and run all cells

---

*Built with Databricks Assistant — demonstrating the full power of code-driven data intelligence.*
