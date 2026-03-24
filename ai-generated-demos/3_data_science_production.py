# Databricks notebook source

# COMMAND ----------

# MAGIC # Serious Adverse Event Prediction — Production MLOps Notebook
# MAGIC 
# MAGIC **Objective**: Predict whether an adverse event is **serious** (`serious` flag) using drug, treatment, and event characteristics.
# MAGIC 
# MAGIC **Primary Metric**: **ROC-AUC** — threshold-independent, robust to class imbalance, suitable for clinical safety where missing a serious event (FN) is costlier than a false alarm (FP).
# MAGIC 
# MAGIC **Secondary Metrics**: Accuracy, F1-score, PR-AUC, Confusion Matrix
# MAGIC 
# MAGIC **Production Improvements over `2_serious_adverse_event_prediction_lightgbm`**:
# MAGIC | Improvement | Detail |
# MAGIC | --- | --- |
# MAGIC | **Parameterized config** | All settings (table, model name, catalog, max_evals) in a single config cell with `dbutils.widgets` |
# MAGIC | **No data leakage** | `risk_score`, `outcome`, `severity` excluded from the start — no ablation needed |
# MAGIC | **Bundled preprocessing** | OrdinalEncoder + LightGBM wrapped in a custom `mlflow.pyfunc.PythonModel` for portable inference |
# MAGIC | **Reusable functions** | `train_and_evaluate()` and `hyperopt_objective()` eliminate code duplication |
# MAGIC | **Model validation gate** | New model must beat the current Champion on the test set before promotion |
# MAGIC | **Dataset logging** | `mlflow.log_input()` for full data provenance tracking |
# MAGIC | **Inference validation** | End-to-end test: load Champion model from UC and run sample predictions |
# MAGIC | **Scalability guard** | Row count check before `.toPandas()` to prevent OOM on large datasets |

# COMMAND ----------

# ==============================================================================
# CONFIGURATION — All parameterized settings in one place
# ==============================================================================

# Runtime widgets for parameterization
dbutils.widgets.text("catalog", "", "UC Catalog")
dbutils.widgets.text("schema", "genie_code_assets", "UC Schema")
dbutils.widgets.text("model_name", "serious_adverse_event_classifier", "Model Name")
dbutils.widgets.text("max_evals", "32", "Hyperopt Max Evals")
dbutils.widgets.text("max_rows", "500000", "Max Rows for toPandas")

# Resolve parameters
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_NAME = dbutils.widgets.get("model_name")
MAX_EVALS = int(dbutils.widgets.get("max_evals"))
MAX_ROWS = int(dbutils.widgets.get("max_rows"))

# Derived
current_user = spark.sql("SELECT current_user()").first()[0]
TABLE_NAME = f"{CATALOG}.{SCHEMA}.silver_adverse_events"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
EXPERIMENT_NAME = f"/Users/{current_user}/{MODEL_NAME}_experiment"

# Feature engineering config
DROP_COLS = ["event_id", "patient_id", "onset_date", "outcome", "severity", "risk_score", "_ingestion_timestamp"]
TARGET_COL = "serious"

# Training config
TEST_SIZE = 0.30
VAL_RATIO = 0.50  # of test_size, so 15% val + 15% test
RANDOM_STATE = 42

print(f"Table:          {TABLE_NAME}")
print(f"UC Model:       {UC_MODEL_NAME}")
print(f"Experiment:     {EXPERIMENT_NAME}")
print(f"Max Evals:      {MAX_EVALS}")
print(f"Max Rows:       {MAX_ROWS:,}")
print(f"Dropped cols:   {DROP_COLS}")

# COMMAND ----------

%pip install category_encoders lightgbm hyperopt --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, classification_report
)

import category_encoders as ce
import lightgbm as lgb

import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
from mlflow.models import infer_signature
from mlflow import MlflowClient

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

sns.set_theme(style="whitegrid")

# Re-resolve widgets after restartPython
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_NAME = dbutils.widgets.get("model_name")
MAX_EVALS = int(dbutils.widgets.get("max_evals"))
MAX_ROWS = int(dbutils.widgets.get("max_rows"))

current_user = spark.sql("SELECT current_user()").first()[0]
TABLE_NAME = f"{CATALOG}.{SCHEMA}.silver_adverse_events"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
EXPERIMENT_NAME = f"/Users/{current_user}/{MODEL_NAME}_experiment"
DROP_COLS = ["event_id", "patient_id", "onset_date", "outcome", "severity", "risk_score", "_ingestion_timestamp"]
TARGET_COL = "serious"
TEST_SIZE = 0.30
VAL_RATIO = 0.50
RANDOM_STATE = 42

print("All imports loaded. Widgets resolved.")

# COMMAND ----------

# ==============================================================================
# REUSABLE FUNCTIONS — Eliminate code duplication across baseline/tuning/final
# ==============================================================================

def train_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test,
                       params, cat_features, run_name, extra_tags=None,
                       log_model=True, log_plots=True, early_stop=True):
    """
    Train a LightGBM model, evaluate on val+test, log everything to MLflow.
    Returns (model, metrics_dict, run_id).
    """
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Log params and tags
        mlflow.log_params(params)
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train_samples", X_train.shape[0])
        mlflow.log_param("optimization_metric", "ROC-AUC")
        mlflow.log_param("features", list(X_train.columns))
        if extra_tags:
            mlflow.set_tags(extra_tags)

        # Train
        model = lgb.LGBMClassifier(**params)
        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "eval_metric": "auc",
            "categorical_feature": cat_features,
        }
        if early_stop:
            fit_kwargs["callbacks"] = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
        else:
            fit_kwargs["callbacks"] = [lgb.log_evaluation(period=50)]
        model.fit(X_train, y_train, **fit_kwargs)

        # Predict on val and test
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "val_roc_auc": roc_auc_score(y_val, y_val_proba),
            "val_pr_auc": average_precision_score(y_val, y_val_proba),
            "val_f1": f1_score(y_val, y_val_pred),
            "val_accuracy": accuracy_score(y_val, y_val_pred),
            "test_roc_auc": roc_auc_score(y_test, y_test_proba),
            "test_pr_auc": average_precision_score(y_test, y_test_proba),
            "test_f1": f1_score(y_test, y_test_pred),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
        }
        mlflow.log_metrics(metrics)

        # Log model with signature
        if log_model:
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.lightgbm.log_model(
                model, artifact_path="model",
                signature=signature, input_example=X_train.head(5),
            )

        # Log visualizations
        if log_plots:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred, ax=axes[0], cmap="Blues")
            axes[0].set_title(f"Confusion Matrix — {run_name}")
            RocCurveDisplay.from_predictions(y_test, y_test_proba, ax=axes[1], name="LightGBM")
            axes[1].set_title(f"ROC (AUC={metrics['test_roc_auc']:.4f})")
            axes[1].plot([0, 1], [0, 1], "k--", alpha=0.3)
            importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=True)
            importances.plot(kind="barh", ax=axes[2], color="steelblue")
            axes[2].set_title("Feature Importances")
            plt.tight_layout()
            mlflow.log_figure(fig, "evaluation_plots.png")
            plt.show()

        # Print summary
        print(f"\n{'='*55}")
        print(f"{run_name.upper()} RESULTS")
        print(f"{'='*55}")
        for k, v in metrics.items():
            print(f"  {k:<20}: {v:.4f}")
        print(f"  MLflow Run ID: {run_id}")
        print(f"{'='*55}")

    return model, metrics, run_id


def make_hyperopt_objective(X_train, X_val, y_train, y_val, cat_features):
    """
    Factory that returns a Hyperopt objective function for the given data split.
    Optimizes for ROC-AUC (returns negative for minimization).
    """
    def objective(params):
        with mlflow.start_run(nested=True):
            params["num_leaves"] = int(params["num_leaves"])
            params["max_depth"] = int(params["max_depth"])
            params["n_estimators"] = int(params["n_estimators"])
            params["min_child_samples"] = int(params["min_child_samples"])

            model = lgb.LGBMClassifier(
                objective="binary", metric="auc", boosting_type="gbdt",
                random_state=RANDOM_STATE, verbose=-1, **params,
            )
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
                categorical_feature=cat_features,
            )
            y_proba = model.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, y_proba)
            mlflow.log_params(params)
            mlflow.log_metric("val_roc_auc", roc_auc)
            return {"loss": -roc_auc, "status": STATUS_OK}
    return objective

print("Helper functions defined: train_and_evaluate(), make_hyperopt_objective()")

# COMMAND ----------

# MAGIC ## 1. Exploratory Data Analysis
# MAGIC ### 1.1 Data Loading & Profiling
# MAGIC Load the adverse events table with a scalability guard against OOM on large datasets.

# COMMAND ----------

# --- Load data with row count check ---
spark_df = spark.table(TABLE_NAME)
row_count = spark_df.count()

if row_count > MAX_ROWS:
    print(f"WARNING: Table has {row_count:,} rows, exceeding MAX_ROWS={MAX_ROWS:,}.")
    print(f"Sampling {MAX_ROWS:,} rows for local processing.")
    fraction = MAX_ROWS / row_count
    df = spark_df.sample(fraction=fraction, seed=RANDOM_STATE).limit(MAX_ROWS).toPandas()
else:
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
    "Feature": ["drug_name", "ae_type", "reported_by", "trial_id",
                "concomitant_drug", "dosage_mg", "days_on_treatment",
                "drug_interaction_flag", "serious (TARGET)"],
    "Type": ["Categorical / Nominal", "Categorical / Nominal",
             "Categorical / Nominal", "Categorical / Nominal", "Categorical / Nominal",
             "Numeric / Integer", "Numeric / Integer",
             "Boolean / Binary", "Boolean / Binary (TARGET)"]
})
print("Feature types for modeling:")
display(feature_summary)

print(f"\nDropped columns (IDs/metadata/leakage): {DROP_COLS}")
print("Note: 'risk_score' excluded — suspected derivation from target (data leakage).")
print("Note: 'outcome' and 'severity' excluded — determined after the event.")

# COMMAND ----------

# MAGIC ### 1.2 Target Distribution & Class Balance
# MAGIC Understanding the class balance of `serious` is critical since it directly impacts our choice of ROC-AUC as the optimization metric.

# COMMAND ----------

# Target variable distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

serious_counts = df["serious"].value_counts()
serious_counts.plot(kind="bar", ax=axes[0], color=["steelblue", "coral"])
axes[0].set_title("Class Distribution of 'serious'")
axes[0].set_xlabel("Serious")
axes[0].set_ylabel("Count")
axes[0].set_xticklabels(["False", "True"], rotation=0)
for i, v in enumerate(serious_counts.values):
    axes[0].text(i, v + 30, f"{v} ({v/len(df)*100:.1f}%)", ha="center", fontweight="bold")

serious_counts.plot(kind="pie", ax=axes[1], autopct="%1.1f%%", colors=["steelblue", "coral"],
                     labels=["Not Serious", "Serious"], startangle=90)
axes[1].set_ylabel("")
axes[1].set_title("Class Proportions")

plt.tight_layout()
plt.show()

ratio = serious_counts[False] / serious_counts[True]
print(f"Class ratio (Not Serious : Serious) = {ratio:.2f} : 1")

# COMMAND ----------

# MAGIC ### 1.3 Feature Correlations
# MAGIC Correlogram of numeric features and boolean target to identify linear relationships.

# COMMAND ----------

# Correlogram of numeric features + boolean target (risk_score excluded)
numeric_cols = ["dosage_mg", "days_on_treatment", "drug_interaction_flag", "serious"]
corr_df = df[numeric_cols].copy()
corr_df["serious"] = corr_df["serious"].astype(int)
corr_df["drug_interaction_flag"] = corr_df["drug_interaction_flag"].astype(int)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_df.corr(), annot=True, cmap="RdBu_r", center=0, fmt=".2f",
            square=True, linewidths=0.5, ax=ax)
ax.set_title("Correlation Matrix: Numeric Features & Target (risk_score excluded)")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC ### 1.4 Feature–Target Relationships
# MAGIC Visualizing how key features relate to the `serious` outcome to guide feature engineering.

# COMMAND ----------

# Adverse event type vs. serious outcome
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ae_serious = df.groupby("ae_type")["serious"].mean().sort_values(ascending=False)
ae_serious.plot(kind="barh", ax=axes[0], color="coral")
axes[0].set_title("Serious Event Rate by Adverse Event Type")
axes[0].set_xlabel("Proportion Serious")
axes[0].axvline(x=df["serious"].mean(), color="black", linestyle="--", label=f"Overall: {df['serious'].mean():.2f}")
axes[0].legend()

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
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

for i, col in enumerate(["dosage_mg", "days_on_treatment"]):
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

# MAGIC ## 2. Data Preprocessing
# MAGIC ### 2.1 Feature Selection & Encoding
# MAGIC Drop ID/metadata columns and leakage-prone features (`outcome`, `severity`, `risk_score`). Encode categoricals with ordinal encoding (LightGBM handles categoricals natively).

# COMMAND ----------

# --- Feature selection ---
df_model = df.drop(columns=DROP_COLS)

y = df_model[TARGET_COL].astype(int)
X = df_model.drop(columns=[TARGET_COL])

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
boolean_cols = X.select_dtypes(include=["bool"]).columns.tolist()

print(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
print(f"Numeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"Boolean features ({len(boolean_cols)}): {boolean_cols}")
print(f"Target: {TARGET_COL}")
print(f"Total features: {X.shape[1]}")

# COMMAND ----------

# --- Encode categoricals with OrdinalEncoder ---
# This encoder will be bundled into the PyFunc model for portable inference
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
    X_encoded, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=VAL_RATIO, random_state=RANDOM_STATE, stratify=y_temp
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
# MAGIC Train a LightGBM binary classifier with default hyperparameters as a baseline. All parameters, metrics, and the model artifact are logged to MLflow using the reusable `train_and_evaluate()` function.

# COMMAND ----------

# --- Configure MLflow experiment ---
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.set_registry_uri("databricks-uc")

# Categorical feature names for LightGBM
categorical_feature_names = categorical_cols

# Baseline LightGBM parameters
baseline_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "verbose": -1,
}

# Log dataset provenance
training_dataset = mlflow.data.from_pandas(
    X_train.assign(**{TARGET_COL: y_train}),
    source=TABLE_NAME,
    name="training_data",
    targets=TARGET_COL,
)

# Train baseline using reusable function
baseline_model, baseline_metrics, baseline_run_id = train_and_evaluate(
    X_train, X_val, X_test, y_train, y_val, y_test,
    params=baseline_params,
    cat_features=categorical_feature_names,
    run_name="baseline_lightgbm",
    extra_tags={"model_type": "lightgbm", "stage": "baseline", "dataset": TABLE_NAME},
    early_stop=False,
)

# COMMAND ----------

# MAGIC ## 4. Hyperparameter Tuning (Hyperopt)
# MAGIC 
# MAGIC Using **Hyperopt** with **Trials** for hyperparameter search (switch to `SparkTrials` on multi-node clusters for distributed search). Optimizing for **ROC-AUC** on the validation set.

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

# --- Run Hyperopt ---
# Note: Using Trials (local). For multi-node clusters, use SparkTrials(parallelism=4).
objective = make_hyperopt_objective(X_train, X_val, y_train, y_val, categorical_feature_names)

with mlflow.start_run(run_name="hyperopt_tuning") as parent_run:
    hyperopt_run_id = parent_run.info.run_id
    mlflow.log_param("optimization_metric", "ROC-AUC")
    mlflow.log_param("tuning_method", "Hyperopt (Trials)")
    mlflow.log_param("max_evals", MAX_EVALS)
    mlflow.set_tags({"model_type": "lightgbm", "stage": "tuning", "dataset": TABLE_NAME})

    trials = Trials()
    rng = np.random.default_rng(RANDOM_STATE)

    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=partial(tpe.suggest, n_startup_jobs=5),
        max_evals=MAX_EVALS,
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

# MAGIC ## 5. Final Model Evaluation on Test Set
# MAGIC Retrain with the best hyperparameters from tuning and evaluate on the held-out test set.

# COMMAND ----------

# --- Retrain best model with full logging ---
final_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "random_state": RANDOM_STATE,
    "verbose": -1,
    **best_params,
}

final_model, final_metrics, final_run_id = train_and_evaluate(
    X_train, X_val, X_test, y_train, y_val, y_test,
    params=final_params,
    cat_features=categorical_feature_names,
    run_name="best_lightgbm_final",
    extra_tags={"model_type": "lightgbm", "stage": "final", "dataset": TABLE_NAME},
    log_model=True,
    log_plots=True,
)

# Classification report
y_test_pred = final_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=["Not Serious", "Serious"]))

# COMMAND ----------

# MAGIC ## 6. Register Model to Unity Catalog with Validation Gate
# MAGIC 
# MAGIC Bundle the OrdinalEncoder + LightGBM into a custom `mlflow.pyfunc.PythonModel` so the full preprocessing→prediction pipeline is a single portable artifact.
# MAGIC 
# MAGIC **Validation gate**: The new model must beat the current Champion on the test set (or be the first version) before promotion.

# COMMAND ----------

# ==============================================================================
# CUSTOM PYFUNC MODEL — Bundles encoder + model for portable inference
# ==============================================================================

class AdverseEventClassifier(mlflow.pyfunc.PythonModel):
    """Wraps OrdinalEncoder + LightGBM for end-to-end inference."""

    def __init__(self, encoder, model, categorical_cols, boolean_cols):
        self.encoder = encoder
        self.model = model
        self.categorical_cols = categorical_cols
        self.boolean_cols = boolean_cols

    def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
        """Run preprocessing and prediction in a single call."""
        # Apply ordinal encoding
        X = self.encoder.transform(model_input)
        # Convert booleans to int
        for col in self.boolean_cols:
            if col in X.columns:
                X[col] = X[col].astype(int)
        # Return probability of serious event
        return self.model.predict_proba(X)[:, 1]

print("AdverseEventClassifier PyFunc wrapper defined.")

# COMMAND ----------

# ==============================================================================
# MODEL REGISTRATION WITH VALIDATION GATE
# ==============================================================================

# Create the bundled PyFunc model
pyfunc_model = AdverseEventClassifier(
    encoder=encoder,
    model=final_model,
    categorical_cols=categorical_cols,
    boolean_cols=boolean_cols,
)

# Build signature from raw (pre-encoded) input
raw_input_sample = X.head(5)  # Raw DataFrame before encoding
signature = infer_signature(raw_input_sample, pyfunc_model.predict(None, raw_input_sample))

# Log the bundled model
with mlflow.start_run(run_name="bundled_model_registration") as reg_run:
    reg_run_id = reg_run.info.run_id
    mlflow.set_tags({"model_type": "lightgbm_pyfunc", "stage": "registration", "dataset": TABLE_NAME})
    mlflow.log_metrics(final_metrics)

    # Log dataset provenance
    mlflow.log_input(training_dataset, context="training")

    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=pyfunc_model,
        signature=signature,
        input_example=raw_input_sample,
        pip_requirements=["category_encoders", "lightgbm", "pandas", "numpy", "scikit-learn"],
    )

    # Register to Unity Catalog
    registered_model = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=UC_MODEL_NAME,
    )
    new_version = registered_model.version
    print(f"Registered {UC_MODEL_NAME} v{new_version}")
    print(f"  Source run: {reg_run_id}")
    print(f"  Test ROC-AUC: {final_metrics['test_roc_auc']:.4f}")

# --- Validation Gate: Compare against current Champion ---
client = MlflowClient()
new_test_auc = final_metrics["test_roc_auc"]
promote = False

try:
    champion_version = client.get_model_version_by_alias(UC_MODEL_NAME, "Champion")
    champion_uri = f"models:/{UC_MODEL_NAME}@Champion"
    print(f"\nCurrent Champion: v{champion_version.version}")

    # Load champion and evaluate on same test set
    champion_model = mlflow.pyfunc.load_model(champion_uri)
    champion_preds = champion_model.predict(X.iloc[X_test.index])
    champion_auc = roc_auc_score(y_test, champion_preds)
    print(f"  Champion test ROC-AUC: {champion_auc:.4f}")
    print(f"  New model test ROC-AUC: {new_test_auc:.4f}")

    if new_test_auc >= champion_auc:
        print("\n  NEW MODEL WINS — promoting to Champion.")
        promote = True
    else:
        print("\n  Champion is better — setting new model as Challenger.")
        client.set_registered_model_alias(UC_MODEL_NAME, "Challenger", new_version)

except Exception as e:
    if "RESOURCE_DOES_NOT_EXIST" in str(e) or "NOT_FOUND" in str(e) or "does not exist" in str(e).lower():
        print("\nNo existing Champion found. Promoting new model as first Champion.")
        promote = True
    else:
        raise

if promote:
    client.set_registered_model_alias(UC_MODEL_NAME, "Champion", new_version)
    print(f"\nAlias 'Champion' set on {UC_MODEL_NAME} v{new_version}")
    print(f"Load in production with: mlflow.pyfunc.load_model('models:/{UC_MODEL_NAME}@Champion')")


# COMMAND ----------

# MAGIC ## 7. Inference Validation
# MAGIC Load the Champion model from Unity Catalog and run sample predictions to validate the full end-to-end pipeline works, including the bundled OrdinalEncoder preprocessing.

# COMMAND ----------

# ==============================================================================
# INFERENCE VALIDATION — Load from UC and run sample predictions
# ==============================================================================

champion_uri = f"models:/{UC_MODEL_NAME}@Champion"
print(f"Loading model from: {champion_uri}")

loaded_model = mlflow.pyfunc.load_model(champion_uri)

# Use raw (unencoded) data — the PyFunc model handles encoding internally
raw_sample = X.head(10)
predictions = loaded_model.predict(raw_sample)

result_df = raw_sample.copy()
result_df["predicted_probability"] = predictions
result_df["predicted_class"] = (predictions >= 0.5).astype(int)
result_df["actual"] = y.iloc[:10].values

print("\nSample predictions from Champion model:")
display(result_df[["drug_name", "ae_type", "dosage_mg", "days_on_treatment",
                   "drug_interaction_flag", "predicted_probability", "predicted_class", "actual"]])

# Verify predictions are valid probabilities
assert (predictions >= 0).all() and (predictions <= 1).all(), "Predictions must be valid probabilities [0, 1]"
print(f"\nInference validation PASSED. {len(predictions)} predictions in valid probability range.")

# COMMAND ----------

# MAGIC ## 8. Summary
# MAGIC 
# MAGIC ### Approach
# MAGIC * **Task**: Binary classification — predicting `serious` adverse events
# MAGIC * **Model**: LightGBM with Hyperopt tuning, bundled into a custom PyFunc with OrdinalEncoder preprocessing
# MAGIC * **Primary metric**: ROC-AUC (threshold-independent, robust to imbalance, suitable for clinical safety)
# MAGIC 
# MAGIC ### Production MLOps Improvements
# MAGIC | Feature | Detail |
# MAGIC | --- | --- |
# MAGIC | **Parameterized config** | All settings via `dbutils.widgets` for runtime flexibility |
# MAGIC | **Leakage-free features** | `risk_score`, `outcome`, `severity` excluded from the start |
# MAGIC | **Bundled preprocessing** | OrdinalEncoder + LightGBM in a single PyFunc artifact |
# MAGIC | **Reusable functions** | `train_and_evaluate()` and `make_hyperopt_objective()` eliminate duplication |
# MAGIC | **Model validation gate** | New model must beat incumbent Champion before alias promotion |
# MAGIC | **Dataset provenance** | `mlflow.log_input()` tracks training data lineage |
# MAGIC | **Inference validation** | End-to-end test loading Champion from UC with raw input |
# MAGIC | **Scalability guard** | Row count check + sampling before `.toPandas()` |
# MAGIC 
# MAGIC ### Key EDA Insights
# MAGIC * Dataset is well-balanced (~51/49% split) with zero missing values
# MAGIC * `ae_type` is the strongest predictor (near-perfect separation for some types like Anaphylaxis, Cardiac Arrhythmia)
# MAGIC * `dosage_mg` provides moderate signal (r ~0.36 with target)
# MAGIC * `days_on_treatment` contributes minimal signal
# MAGIC * `risk_score` was excluded due to suspected derivation from the target variable (data leakage)
# MAGIC 
# MAGIC ### MLflow Tracking
# MAGIC All experiments tracked under the configured experiment path with tags for filtering by stage (baseline/tuning/final/registration).
