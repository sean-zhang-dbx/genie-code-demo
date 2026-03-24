# Databricks notebook source

# COMMAND ----------

# MAGIC # Serious Adverse Event Prediction (Binary Classification)
# MAGIC 
# MAGIC **Objective**: Predict whether an adverse event is **serious** (`serious` flag) based on drug, treatment, and event characteristics.
# MAGIC 
# MAGIC **Primary Optimization Metric**: **ROC-AUC** — chosen because:
# MAGIC - It is **threshold-independent**, allowing flexible threshold tuning for clinical deployment
# MAGIC - It is **robust to class imbalance** (the `serious` flag is expected to be moderately imbalanced)
# MAGIC - In clinical safety, missing a serious event (FN) is far costlier than a false alarm (FP); ROC-AUC captures overall discrimination, and the operating threshold can later be set to favor high recall
# MAGIC 
# MAGIC **Secondary Metrics**: Accuracy, F1-score, PR-AUC, Confusion Matrix
# MAGIC 
# MAGIC **Workflow**: EDA → Preprocessing → Baseline LightGBM → Hyperparameter Tuning (Hyperopt + SparkTrials) → Final Evaluation

# COMMAND ----------

dbutils.widgets.text("catalog", "", "UC Catalog")
dbutils.widgets.text("schema", "genie_code_assets", "UC Schema")
dbutils.widgets.text("model_name", "serious_adverse_event_classifier", "Model Name")

# COMMAND ----------

%pip install category_encoders lightgbm hyperopt --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, classification_report
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import category_encoders as ce
import lightgbm as lgb

import mlflow
import mlflow.lightgbm
from mlflow.models import infer_signature

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, SparkTrials
from hyperopt.pyll import scope

sns.set_theme(style="whitegrid")
print("All imports loaded successfully.")

# COMMAND ----------

# MAGIC ## 1. Exploratory Data Analysis
# MAGIC ### 1.1 Data Loading & Profiling
# MAGIC Load the adverse events table and inspect shape, types, missing values, and descriptive statistics.

# COMMAND ----------

# Load data from Unity Catalog
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_NAME = dbutils.widgets.get("model_name")
current_user = spark.sql("SELECT current_user()").first()[0]

TABLE_NAME = f"{CATALOG}.{SCHEMA}.silver_adverse_events"
spark_df = spark.table(TABLE_NAME)
df = spark_df.toPandas()

print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# COMMAND ----------

# Descriptive statistics for numeric columns
df.describe()

# COMMAND ----------

# Feature type summary for modeling reference
feature_summary = pd.DataFrame({
    "Feature": ["drug_name", "ae_type", "severity", "reported_by", "trial_id",
                "concomitant_drug", "dosage_mg", "days_on_treatment", "risk_score",
                "drug_interaction_flag", "serious (TARGET)"],
    "Type": ["Categorical / Nominal", "Categorical / Nominal", "Categorical / Ordinal (Mild < Moderate < Severe)",
             "Categorical / Nominal", "Categorical / Nominal", "Categorical / Nominal",
             "Numeric / Integer", "Numeric / Integer", "Numeric / Float",
             "Boolean / Binary", "Boolean / Binary (TARGET)"]
})
print("Feature types for modeling:")
display(feature_summary)

print(f"\nDropped columns (IDs/metadata): event_id, patient_id, onset_date, outcome, _ingestion_timestamp")
print("Note: 'outcome' and 'severity' excluded to prevent data leakage — they are determined after the event.")

# COMMAND ----------

# MAGIC ### 1.2 Target Distribution & Class Balance
# MAGIC Understanding the class balance of `serious` is critical since it directly impacts our choice of ROC-AUC as the optimization metric.

# COMMAND ----------

# Target variable distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Count plot
serious_counts = df["serious"].value_counts()
serious_counts.plot(kind="bar", ax=axes[0], color=["steelblue", "coral"])
axes[0].set_title("Class Distribution of 'serious'")
axes[0].set_xlabel("Serious")
axes[0].set_ylabel("Count")
axes[0].set_xticklabels(["False", "True"], rotation=0)
for i, v in enumerate(serious_counts.values):
    axes[0].text(i, v + 30, f"{v} ({v/len(df)*100:.1f}%)", ha="center", fontweight="bold")

# Proportion pie chart
serious_counts.plot(kind="pie", ax=axes[1], autopct="%1.1f%%", colors=["steelblue", "coral"],
                     labels=["Not Serious", "Serious"], startangle=90)
axes[1].set_ylabel("")
axes[1].set_title("Class Proportions")

plt.tight_layout()
plt.show()

ratio = serious_counts[False] / serious_counts[True]
print(f"Class ratio (Not Serious : Serious) = {ratio:.2f} : 1")
print(f"scale_pos_weight for LightGBM = {ratio:.2f}")

# COMMAND ----------

# MAGIC ### 1.3 Feature Correlations
# MAGIC Correlogram of numeric features and boolean target to identify linear relationships.

# COMMAND ----------

# Correlogram of numeric features + boolean target
numeric_cols = ["dosage_mg", "days_on_treatment", "risk_score", "drug_interaction_flag", "serious"]
corr_df = df[numeric_cols].copy()
corr_df["serious"] = corr_df["serious"].astype(int)
corr_df["drug_interaction_flag"] = corr_df["drug_interaction_flag"].astype(int)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_df.corr(), annot=True, cmap="RdBu_r", center=0, fmt=".2f",
            square=True, linewidths=0.5, ax=ax)
ax.set_title("Correlation Matrix: Numeric Features & Target")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC ### 1.4 Feature–Target Relationships
# MAGIC Visualizing how key features relate to the `serious` outcome to guide feature engineering.

# COMMAND ----------

# Adverse event type vs. serious outcome
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# ae_type vs serious rate
ae_serious = df.groupby("ae_type")["serious"].mean().sort_values(ascending=False)
ae_serious.plot(kind="barh", ax=axes[0], color="coral")
axes[0].set_title("Serious Event Rate by Adverse Event Type")
axes[0].set_xlabel("Proportion Serious")
axes[0].axvline(x=df["serious"].mean(), color="black", linestyle="--", label=f"Overall: {df['serious'].mean():.2f}")
axes[0].legend()

# drug_name vs serious rate
drug_serious = df.groupby("drug_name")["serious"].mean().sort_values(ascending=False)
drug_serious.plot(kind="barh", ax=axes[1], color="steelblue")
axes[1].set_title("Serious Event Rate by Drug Name")
axes[1].set_xlabel("Proportion Serious")
axes[1].axvline(x=df["serious"].mean(), color="black", linestyle="--", label=f"Overall: {df['serious'].mean():.2f}")
axes[1].legend()

plt.tight_layout()
plt.show()

# COMMAND ----------

# Numeric feature distributions split by serious flag
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for i, col in enumerate(["risk_score", "dosage_mg", "days_on_treatment"]):
    for label, color in [(True, "coral"), (False, "steelblue")]:
        subset = df[df["serious"] == label][col]
        axes[i].hist(subset, bins=30, alpha=0.6, label=f"Serious={label}", color=color, density=True)
    axes[i].set_title(f"{col} by Serious Flag")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Density")
    axes[i].legend()

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC ### 1.5 EDA Summary
# MAGIC 
# MAGIC | Finding | Detail | Modeling Implication |
# MAGIC | --- | --- | --- |
# MAGIC | **Class balance** | 51% Not Serious / 49% Serious (ratio 0.96:1) | Nearly balanced — no resampling or `scale_pos_weight` needed |
# MAGIC | **Strongest predictor** | `risk_score` (r = 0.65 with target) | Expect high feature importance; watch for potential data leakage if risk_score is derived from the target |
# MAGIC | **Second predictor** | `dosage_mg` (r = 0.36) — higher doses shift toward serious | Useful numeric signal; correlated with risk_score (r = 0.52) |
# MAGIC | **AE type signal** | Anaphylaxis, Cardiac Arrhythmia, Liver Injury are ~100% serious; Injection Site Reaction, Headache are ~15% | `ae_type` will be a very strong categorical feature |
# MAGIC | **Drug name** | All drugs cluster tightly around the 51% mean | Weak standalone predictor — seriousness depends more on event type than specific drug |
# MAGIC | **Days on treatment** | Nearly uniform distribution, r = −0.03 | Uninformative for prediction; model may assign low importance |
# MAGIC | **Drug interaction flag** | Weak correlation (r = 0.10) | Minor signal; include but don't expect high importance |
# MAGIC | **Missing values** | Zero nulls across all 16 columns | No imputation required |
# MAGIC | **Data leakage** | `outcome` and `severity` are post-event determinations | Excluded from features to prevent leakage |

# COMMAND ----------

# MAGIC ## 2. Data Preprocessing
# MAGIC ### 2.1 Feature Selection & Encoding
# MAGIC Drop ID/metadata columns and leakage-prone features (`outcome`, `severity`). Encode categoricals with ordinal encoding (LightGBM handles categoricals natively).

# COMMAND ----------

# --- Feature selection ---
# Drop IDs, metadata, and leakage-prone columns
drop_cols = ["event_id", "patient_id", "onset_date", "outcome", "severity", "_ingestion_timestamp"]
df_model = df.drop(columns=drop_cols)

# Separate target and features
target_col = "serious"
y = df_model[target_col].astype(int)
X = df_model.drop(columns=[target_col])

# Identify column types
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
boolean_cols = X.select_dtypes(include=["bool"]).columns.tolist()

print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Boolean features ({len(boolean_cols)}): {boolean_cols}")
print(f"Target: {target_col}")
print(f"\nTotal features: {X.shape[1]}")

# COMMAND ----------

# --- Encode categoricals with OrdinalEncoder ---
# LightGBM handles ordinal-encoded categoricals natively
encoder = ce.OrdinalEncoder(cols=categorical_cols, handle_unknown="value", handle_missing="value")
X_encoded = encoder.fit_transform(X)

# Convert boolean columns to int
for col in boolean_cols:
    X_encoded[col] = X_encoded[col].astype(int)

print(f"Encoded feature matrix shape: {X_encoded.shape}")
print(f"\nEncoded dtypes:\n{X_encoded.dtypes}")

# COMMAND ----------

# MAGIC ### 2.2 Train / Validation / Test Split
# MAGIC 70% train, 15% validation, 15% test. Stratified on the target to preserve class proportions.

# COMMAND ----------

# --- Stratified train/val/test split (70/15/15) ---
X_train, X_temp, y_train, y_temp = train_test_split(
    X_encoded, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("Split sizes (stratified):")
print(f"  Train: {X_train.shape[0]:,} ({X_train.shape[0]/len(y)*100:.0f}%)")
print(f"  Val:   {X_val.shape[0]:,} ({X_val.shape[0]/len(y)*100:.0f}%)")
print(f"  Test:  {X_test.shape[0]:,} ({X_test.shape[0]/len(y)*100:.0f}%)")
print(f"\nClass proportions (train): {y_train.value_counts(normalize=True).to_dict()}")
print(f"Class proportions (val):   {y_val.value_counts(normalize=True).to_dict()}")
print(f"Class proportions (test):  {y_test.value_counts(normalize=True).to_dict()}")

# COMMAND ----------

# MAGIC ## 3. Baseline Model Training
# MAGIC ### LightGBM with MLflow Tracking
# MAGIC Train a LightGBM binary classifier with default hyperparameters as a baseline. All parameters, metrics, and the model artifact are logged to MLflow. **Optimizing for ROC-AUC.**

# COMMAND ----------

# --- Configure MLflow experiment ---
experiment_name = f"/Users/{current_user}/adverse_events_serious_prediction"
mlflow.set_experiment(experiment_name)

# --- Specify categorical features for LightGBM ---
categorical_feature_names = categorical_cols  # ['trial_id', 'drug_name', 'ae_type', 'reported_by', 'concomitant_drug']

# --- Baseline LightGBM parameters ---
baseline_params = {
    "objective": "binary",
    "metric": "auc",         # Primary metric: ROC-AUC
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}

print("Baseline LightGBM parameters:")
for k, v in baseline_params.items():
    print(f"  {k}: {v}")

# COMMAND ----------

# --- Train baseline model with MLflow logging ---
with mlflow.start_run(run_name="baseline_lightgbm") as run:
    baseline_run_id = run.info.run_id
    
    # Log parameters
    mlflow.log_params(baseline_params)
    mlflow.log_param("features", categorical_cols + numeric_cols + boolean_cols)
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_param("n_train_samples", X_train.shape[0])
    mlflow.log_param("optimization_metric", "ROC-AUC")
    
    # Train model
    baseline_model = lgb.LGBMClassifier(**baseline_params)
    baseline_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.log_evaluation(period=50)],
        categorical_feature=categorical_feature_names,
    )
    
    # Predict on validation set
    y_val_pred = baseline_model.predict(X_val)
    y_val_proba = baseline_model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_roc_auc = roc_auc_score(y_val, y_val_proba)
    val_pr_auc = average_precision_score(y_val, y_val_proba)
    
    # Log metrics
    mlflow.log_metric("val_accuracy", val_accuracy)
    mlflow.log_metric("val_f1", val_f1)
    mlflow.log_metric("val_roc_auc", val_roc_auc)
    mlflow.log_metric("val_pr_auc", val_pr_auc)
    
    # Log model with signature
    signature = infer_signature(X_train, baseline_model.predict(X_train))
    mlflow.lightgbm.log_model(
        baseline_model, artifact_path="model",
        signature=signature, input_example=X_train.head(5),
    )
    
    # Log confusion matrix as artifact
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_val, y_val_pred, ax=ax_cm, cmap="Blues")
    ax_cm.set_title("Baseline Confusion Matrix (Validation)")
    mlflow.log_figure(fig_cm, "confusion_matrix_val.png")
    plt.show()
    
    print("\n" + "="*50)
    print("BASELINE MODEL RESULTS (Validation Set)")
    print("="*50)
    print(f"  ROC-AUC (primary):  {val_roc_auc:.4f}")
    print(f"  PR-AUC:             {val_pr_auc:.4f}")
    print(f"  F1 Score:           {val_f1:.4f}")
    print(f"  Accuracy:           {val_accuracy:.4f}")
    print(f"\n  MLflow Run ID: {baseline_run_id}")
    print("="*50)

# COMMAND ----------

# MAGIC ## 4. Hyperparameter Tuning (Hyperopt + SparkTrials)
# MAGIC 
# MAGIC Using **Hyperopt** with **SparkTrials** for distributed hyperparameter search as per project conventions. Optimizing for **ROC-AUC** on the validation set.
# MAGIC 
# MAGIC > **Note:** The baseline already achieves near-perfect ROC-AUC (1.0). This is driven by highly discriminative features (`ae_type`, `risk_score`). Tuning will aim to find a more parsimonious model with fewer boosting rounds while maintaining performance.

# COMMAND ----------

# --- Define Hyperopt search space ---
search_space = {
    "num_leaves":        scope.int(hp.quniform("num_leaves", 15, 63, 1)),
    "max_depth":         scope.int(hp.quniform("max_depth", 3, 12, 1)),
    "learning_rate":     hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
    "n_estimators":      scope.int(hp.quniform("n_estimators", 50, 500, 25)),
    "min_child_samples": scope.int(hp.quniform("min_child_samples", 5, 50, 1)),
    "subsample":         hp.uniform("subsample", 0.6, 1.0),
    "colsample_bytree":  hp.uniform("colsample_bytree", 0.6, 1.0),
    "reg_alpha":         hp.loguniform("reg_alpha", np.log(1e-8), np.log(10.0)),
    "reg_lambda":        hp.loguniform("reg_lambda", np.log(1e-8), np.log(10.0)),
}

def objective(params):
    """Hyperopt objective: train LightGBM and return negative ROC-AUC."""
    with mlflow.start_run(nested=True):
        # Ensure integer params
        params["num_leaves"] = int(params["num_leaves"])
        params["max_depth"] = int(params["max_depth"])
        params["n_estimators"] = int(params["n_estimators"])
        params["min_child_samples"] = int(params["min_child_samples"])
        
        model = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            boosting_type="gbdt",
            random_state=42,
            verbose=-1,
            **params,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
            categorical_feature=categorical_feature_names,
        )
        
        y_proba = model.predict_proba(X_val)[:, 1]
        roc_auc = roc_auc_score(y_val, y_proba)
        
        mlflow.log_params(params)
        mlflow.log_metric("val_roc_auc", roc_auc)
        
        # Hyperopt minimizes, so return negative AUC
        return {"loss": -roc_auc, "status": STATUS_OK}

print("Search space defined. Ready for Hyperopt tuning.")

# COMMAND ----------

# --- Run Hyperopt with Trials ---
# Note: Using Trials (local) instead of SparkTrials due to serverless compute
from functools import partial

with mlflow.start_run(run_name="hyperopt_lightgbm_tuning") as parent_run:
    hyperopt_run_id = parent_run.info.run_id
    mlflow.log_param("optimization_metric", "ROC-AUC")
    mlflow.log_param("tuning_method", "Hyperopt (Trials)")
    
    trials = Trials()
    rng = np.random.default_rng(42)
    
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=partial(tpe.suggest, n_startup_jobs=5),
        max_evals=32,
        trials=trials,
        rstate=rng,
    )
    
    # Convert best params to proper types
    best_params["num_leaves"] = int(best_params["num_leaves"])
    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["n_estimators"] = int(best_params["n_estimators"])
    best_params["min_child_samples"] = int(best_params["min_child_samples"])
    
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    
    print("\nBest hyperparameters found:")
    for k, v in sorted(best_params.items()):
        print(f"  {k}: {v}")

# COMMAND ----------

# MAGIC ## 5. Final Evaluation on Test Set
# MAGIC Retrain the best model from hyperparameter tuning and evaluate on the held-out test set. Log all final metrics, confusion matrix, ROC curve, and feature importances to MLflow.

# COMMAND ----------

# --- Retrain best model and evaluate on test set ---
with mlflow.start_run(run_name="best_lightgbm_final") as final_run:
    final_run_id = final_run.info.run_id
    
    # Build final model with best hyperparameters
    final_model = lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        random_state=42,
        verbose=-1,
        **best_params,
    )
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        categorical_feature=categorical_feature_names,
    )
    
    # Predictions on test set
    y_test_pred = final_model.predict(X_test)
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    
    # --- Calculate all metrics ---
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    test_pr_auc = average_precision_score(y_test, y_test_proba)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    # Log params and metrics
    mlflow.log_params(best_params)
    mlflow.log_param("evaluation_set", "test")
    mlflow.log_param("optimization_metric", "ROC-AUC")
    mlflow.log_metric("test_roc_auc", test_roc_auc)
    mlflow.log_metric("test_pr_auc", test_pr_auc)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1", test_f1)
    
    # Log model with signature
    signature = infer_signature(X_train, final_model.predict(X_train))
    mlflow.lightgbm.log_model(
        final_model, artifact_path="model",
        signature=signature, input_example=X_train.head(5),
    )
    
    # --- Visualizations ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, ax=axes[0], cmap="Blues")
    axes[0].set_title("Confusion Matrix (Test Set)")
    
    # 2. ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_test_proba, ax=axes[1], name="LightGBM")
    axes[1].set_title(f"ROC Curve (AUC = {test_roc_auc:.4f})")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)
    
    # 3. Feature Importances
    importances = pd.Series(
        final_model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=True)
    importances.plot(kind="barh", ax=axes[2], color="steelblue")
    axes[2].set_title("Feature Importances (split count)")
    axes[2].set_xlabel("Importance")
    
    plt.tight_layout()
    mlflow.log_figure(fig, "final_evaluation_plots.png")
    plt.show()
    
    # Print final summary
    print("\n" + "="*55)
    print("FINAL MODEL RESULTS (Test Set) — Optimized for ROC-AUC")
    print("="*55)
    print(f"  ROC-AUC (primary):  {test_roc_auc:.4f}")
    print(f"  PR-AUC:             {test_pr_auc:.4f}")
    print(f"  F1 Score:           {test_f1:.4f}")
    print(f"  Accuracy:           {test_accuracy:.4f}")
    print(f"\n  MLflow Run ID: {final_run_id}")
    print("="*55)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=["Not Serious", "Serious"]))

# COMMAND ----------

# MAGIC ## 5b. Ablation: Retrain Without `risk_score`
# MAGIC 
# MAGIC The near-perfect ROC-AUC (1.0) is suspicious. Feature importances confirm `risk_score` dominates all other features. It may be **derived from the target** (data leakage). This section retrains the full pipeline — baseline, Hyperopt tuning, and final evaluation — with `risk_score` **removed** to measure realistic model performance.

# COMMAND ----------

# --- Prepare data WITHOUT risk_score ---
drop_feature = "risk_score"

X_train_nr = X_train.drop(columns=[drop_feature])
X_val_nr = X_val.drop(columns=[drop_feature])
X_test_nr = X_test.drop(columns=[drop_feature])

cat_features_nr = categorical_feature_names  # unchanged, risk_score was numeric

print(f"Features after dropping '{drop_feature}': {list(X_train_nr.columns)}")
print(f"Shape: {X_train_nr.shape}")

# COMMAND ----------

# --- Baseline model without risk_score ---
with mlflow.start_run(run_name="baseline_no_risk_score") as run:
    nr_baseline_run_id = run.info.run_id
    mlflow.log_param("ablation", "without_risk_score")
    mlflow.log_param("optimization_metric", "ROC-AUC")
    mlflow.log_params(baseline_params)

    nr_baseline = lgb.LGBMClassifier(**baseline_params)
    nr_baseline.fit(
        X_train_nr, y_train,
        eval_set=[(X_val_nr, y_val)],
        eval_metric="auc",
        callbacks=[lgb.log_evaluation(period=50)],
        categorical_feature=cat_features_nr,
    )

    y_val_proba_nr = nr_baseline.predict_proba(X_val_nr)[:, 1]
    y_val_pred_nr = nr_baseline.predict(X_val_nr)

    nr_val_roc_auc = roc_auc_score(y_val, y_val_proba_nr)
    nr_val_f1 = f1_score(y_val, y_val_pred_nr)
    nr_val_acc = accuracy_score(y_val, y_val_pred_nr)
    nr_val_pr_auc = average_precision_score(y_val, y_val_proba_nr)

    mlflow.log_metric("val_roc_auc", nr_val_roc_auc)
    mlflow.log_metric("val_f1", nr_val_f1)
    mlflow.log_metric("val_accuracy", nr_val_acc)
    mlflow.log_metric("val_pr_auc", nr_val_pr_auc)

    print("="*55)
    print("BASELINE (No risk_score) — Validation Set")
    print("="*55)
    print(f"  ROC-AUC (primary):  {nr_val_roc_auc:.4f}")
    print(f"  PR-AUC:             {nr_val_pr_auc:.4f}")
    print(f"  F1 Score:           {nr_val_f1:.4f}")
    print(f"  Accuracy:           {nr_val_acc:.4f}")
    print("="*55)

# COMMAND ----------

# --- Hyperopt tuning WITHOUT risk_score ---
from functools import partial

def objective_nr(params):
    """Hyperopt objective without risk_score: minimize negative ROC-AUC."""
    with mlflow.start_run(nested=True):
        params["num_leaves"] = int(params["num_leaves"])
        params["max_depth"] = int(params["max_depth"])
        params["n_estimators"] = int(params["n_estimators"])
        params["min_child_samples"] = int(params["min_child_samples"])

        model = lgb.LGBMClassifier(
            objective="binary", metric="auc", boosting_type="gbdt",
            random_state=42, verbose=-1, **params,
        )
        model.fit(
            X_train_nr, y_train,
            eval_set=[(X_val_nr, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
            categorical_feature=cat_features_nr,
        )
        y_proba = model.predict_proba(X_val_nr)[:, 1]
        roc_auc = roc_auc_score(y_val, y_proba)
        mlflow.log_params(params)
        mlflow.log_metric("val_roc_auc", roc_auc)
        return {"loss": -roc_auc, "status": STATUS_OK}

with mlflow.start_run(run_name="hyperopt_no_risk_score") as parent_run:
    nr_hyperopt_run_id = parent_run.info.run_id
    mlflow.log_param("ablation", "without_risk_score")
    mlflow.log_param("optimization_metric", "ROC-AUC")

    trials_nr = Trials()
    rng = np.random.default_rng(42)

    best_params_nr = fmin(
        fn=objective_nr,
        space=search_space,
        algo=partial(tpe.suggest, n_startup_jobs=5),
        max_evals=32,
        trials=trials_nr,
        rstate=rng,
    )

    best_params_nr["num_leaves"] = int(best_params_nr["num_leaves"])
    best_params_nr["max_depth"] = int(best_params_nr["max_depth"])
    best_params_nr["n_estimators"] = int(best_params_nr["n_estimators"])
    best_params_nr["min_child_samples"] = int(best_params_nr["min_child_samples"])

    mlflow.log_params({f"best_{k}": v for k, v in best_params_nr.items()})

    print("\nBest hyperparameters (no risk_score):")
    for k, v in sorted(best_params_nr.items()):
        print(f"  {k}: {v}")

# COMMAND ----------

# --- Final evaluation WITHOUT risk_score on test set ---
with mlflow.start_run(run_name="best_no_risk_score_final") as final_nr_run:
    final_nr_run_id = final_nr_run.info.run_id
    mlflow.log_param("ablation", "without_risk_score")
    mlflow.log_param("optimization_metric", "ROC-AUC")

    final_model_nr = lgb.LGBMClassifier(
        objective="binary", metric="auc", boosting_type="gbdt",
        random_state=42, verbose=-1, **best_params_nr,
    )
    final_model_nr.fit(
        X_train_nr, y_train,
        eval_set=[(X_val_nr, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
        categorical_feature=cat_features_nr,
    )

    y_test_pred_nr = final_model_nr.predict(X_test_nr)
    y_test_proba_nr = final_model_nr.predict_proba(X_test_nr)[:, 1]

    nr_test_roc_auc = roc_auc_score(y_test, y_test_proba_nr)
    nr_test_pr_auc = average_precision_score(y_test, y_test_proba_nr)
    nr_test_acc = accuracy_score(y_test, y_test_pred_nr)
    nr_test_f1 = f1_score(y_test, y_test_pred_nr)

    mlflow.log_params(best_params_nr)
    mlflow.log_metric("test_roc_auc", nr_test_roc_auc)
    mlflow.log_metric("test_pr_auc", nr_test_pr_auc)
    mlflow.log_metric("test_accuracy", nr_test_acc)
    mlflow.log_metric("test_f1", nr_test_f1)

    signature_nr = infer_signature(X_train_nr, final_model_nr.predict(X_train_nr))
    mlflow.lightgbm.log_model(
        final_model_nr, artifact_path="model",
        signature=signature_nr, input_example=X_train_nr.head(5),
    )

    # --- Visualizations ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred_nr, ax=axes[0], cmap="Oranges")
    axes[0].set_title("Confusion Matrix — No risk_score (Test)")

    RocCurveDisplay.from_predictions(y_test, y_test_proba_nr, ax=axes[1], name="LightGBM (no risk_score)")
    axes[1].set_title(f"ROC Curve (AUC = {nr_test_roc_auc:.4f})")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)

    importances_nr = pd.Series(
        final_model_nr.feature_importances_, index=X_train_nr.columns
    ).sort_values(ascending=True)
    importances_nr.plot(kind="barh", ax=axes[2], color="darkorange")
    axes[2].set_title("Feature Importances — No risk_score")
    axes[2].set_xlabel("Importance")

    plt.tight_layout()
    mlflow.log_figure(fig, "final_eval_no_risk_score.png")
    plt.show()

    print("\n" + "="*60)
    print("FINAL RESULTS WITHOUT risk_score (Test) — ROC-AUC Optimized")
    print("="*60)
    print(f"  ROC-AUC (primary):  {nr_test_roc_auc:.4f}")
    print(f"  PR-AUC:             {nr_test_pr_auc:.4f}")
    print(f"  F1 Score:           {nr_test_f1:.4f}")
    print(f"  Accuracy:           {nr_test_acc:.4f}")
    print(f"\n  MLflow Run ID: {final_nr_run_id}")
    print("="*60)
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred_nr, target_names=["Not Serious", "Serious"]))

    # --- Side-by-side comparison ---
    print("\n" + "="*60)
    print("COMPARISON: With vs Without risk_score (Test Set)")
    print("="*60)
    print(f"  {'Metric':<20} {'With risk_score':>17} {'Without risk_score':>20}")
    print(f"  {'-'*57}")
    print(f"  {'ROC-AUC':<20} {test_roc_auc:>17.4f} {nr_test_roc_auc:>20.4f}")
    print(f"  {'PR-AUC':<20} {test_pr_auc:>17.4f} {nr_test_pr_auc:>20.4f}")
    print(f"  {'F1 Score':<20} {test_f1:>17.4f} {nr_test_f1:>20.4f}")
    print(f"  {'Accuracy':<20} {test_accuracy:>17.4f} {nr_test_acc:>20.4f}")
    print("="*60)

# COMMAND ----------

# MAGIC ## 7. Register Model to Unity Catalog
# MAGIC Register the leakage-free model (without `risk_score`) to Unity Catalog and set the **Champion** alias for production use.

# COMMAND ----------

# --- Register the no-risk-score model to Unity Catalog ---
import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")

uc_model_name = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"

# Register model from the best no-risk-score run
model_uri = f"runs:/{final_nr_run_id}/model"
registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=uc_model_name,
)

print(f"Model registered: {uc_model_name}")
print(f"  Version: {registered_model.version}")
print(f"  Source run: {final_nr_run_id}")
print(f"  ROC-AUC (test): {nr_test_roc_auc:.4f}")
print(f"  Features: {list(X_train_nr.columns)} (risk_score excluded)")

# COMMAND ----------

# --- Set Champion alias for production ---
client = MlflowClient()

client.set_registered_model_alias(
    name=uc_model_name,
    alias="Champion",
    version=registered_model.version,
)

print(f"Alias 'Champion' set on {uc_model_name} v{registered_model.version}")
print(f"\nLoad in production with:")
print(f"  mlflow.lightgbm.load_model('models:/{uc_model_name}@Champion')")

# COMMAND ----------

# MAGIC ## 6. Summary
# MAGIC 
# MAGIC ### Approach
# MAGIC - **Task**: Binary classification — predicting `serious` adverse events
# MAGIC - **Model**: LightGBM with Hyperopt + SparkTrials tuning
# MAGIC - **Primary metric**: ROC-AUC (threshold-independent, robust to imbalance, suitable for clinical safety)
# MAGIC 
# MAGIC ### Key EDA Insights
# MAGIC - Dataset is well-balanced (51/49% split) with zero missing values
# MAGIC - `risk_score` (r=0.65) and `ae_type` (near-perfect separation for some types) are the strongest predictors
# MAGIC - `days_on_treatment` contributes minimal signal
# MAGIC - `outcome` and `severity` were excluded to prevent data leakage
# MAGIC 
# MAGIC ### Modeling Notes
# MAGIC - The feature space is highly discriminative — the baseline model already achieves near-perfect ROC-AUC
# MAGIC - If `risk_score` is derived from the target variable, it should be investigated and potentially removed for a more realistic evaluation
# MAGIC - All experiments are tracked in MLflow under the configured experiment path
