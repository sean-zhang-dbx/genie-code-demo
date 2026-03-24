# Databricks notebook source

# COMMAND ----------

# MAGIC ## Pharma Clinical Data Pipeline
# MAGIC 
# MAGIC **Medallion Architecture** — Spark Declarative Pipelines (SDP)
# MAGIC 
# MAGIC | Layer | Tables | Pattern |
# MAGIC |-------|--------|---------|
# MAGIC | Bronze | `adverse_events`, `clinical_trials`, `patients`, `trial_results` | Streaming tables via Auto Loader |
# MAGIC | Silver | Cleansed versions of each bronze table | Materialized views with type casting, expectations, CDF |
# MAGIC | Gold | `trial_safety_summary`, `patient_outcomes`, `drug_efficacy` | Materialized views for BI aggregates |
# MAGIC 
# MAGIC **Source:** Configured via `pipeline.source_path` pipeline configuration.

# COMMAND ----------

from pyspark import pipelines as dp
from pyspark.sql.functions import col, to_date, current_timestamp, count, avg, sum as _sum, when, lit, round as _round

CATALOG = spark.conf.get("pipeline.catalog")
SCHEMA = spark.conf.get("pipeline.schema", "genie_code_assets")
VOLUME = spark.conf.get("pipeline.volume", "raw_pharma_data")

SOURCE_PATH = spark.conf.get(
    "pipeline.source_path",
    f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"
)

# COMMAND ----------

# MAGIC ### Bronze Layer — Raw Ingestion via Auto Loader

# COMMAND ----------

@dp.table(
    comment="Raw adverse events ingested from JSON via Auto Loader",
    cluster_by=["trial_id", "patient_id"]
)
def bronze_adverse_events():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("multiLine", "true")
        .option("pathGlobFilter", "adverse_events.json")
        .load(SOURCE_PATH)
        .withColumn("_ingestion_timestamp", current_timestamp())
    )

# COMMAND ----------

@dp.table(
    comment="Raw clinical trials ingested from JSON via Auto Loader",
    cluster_by=["trial_id"]
)
def bronze_clinical_trials():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("multiLine", "true")
        .option("pathGlobFilter", "clinical_trials.json")
        .load(SOURCE_PATH)
        .withColumn("_ingestion_timestamp", current_timestamp())
    )

# COMMAND ----------

@dp.table(
    comment="Raw patient demographics ingested from JSON via Auto Loader",
    cluster_by=["patient_id"]
)
def bronze_patients():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("multiLine", "true")
        .option("pathGlobFilter", "patients.json")
        .load(SOURCE_PATH)
        .withColumn("_ingestion_timestamp", current_timestamp())
    )

# COMMAND ----------

@dp.table(
    comment="Raw trial results ingested from JSON via Auto Loader",
    cluster_by=["trial_id", "patient_id"]
)
def bronze_trial_results():
    return (
        spark.readStream
        .format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.inferColumnTypes", "true")
        .option("multiLine", "true")
        .option("pathGlobFilter", "trial_results.json")
        .load(SOURCE_PATH)
        .withColumn("_ingestion_timestamp", current_timestamp())
    )

# COMMAND ----------

# MAGIC ### Silver Layer — Cleansed & Conformed (Materialized Views, CDF Enabled)

# COMMAND ----------

@dp.materialized_view(
    comment="Cleansed adverse events with typed dates and quality checks",
    table_properties={"delta.enableChangeDataFeed": "true"},
    cluster_by=["trial_id", "patient_id"]
)
@dp.expect_or_drop("valid_event_id", "event_id IS NOT NULL")
@dp.expect_or_drop("valid_patient_id", "patient_id IS NOT NULL")
@dp.expect("valid_onset_date", "onset_date IS NOT NULL")
def silver_adverse_events():
    return (
        spark.read.table("bronze_adverse_events")
        .select(
            col("event_id"),
            col("patient_id"),
            col("trial_id"),
            col("drug_name"),
            col("ae_type"),
            col("severity"),
            col("serious"),
            to_date(col("onset_date")).alias("onset_date"),
            col("outcome"),
            col("reported_by"),
            col("dosage_mg"),
            col("days_on_treatment"),
            col("risk_score"),
            col("drug_interaction_flag"),
            col("concomitant_drug"),
            col("_ingestion_timestamp")
        )
    )

# COMMAND ----------

@dp.materialized_view(
    comment="Cleansed clinical trials with typed dates",
    table_properties={"delta.enableChangeDataFeed": "true"},
    cluster_by=["trial_id"]
)
@dp.expect_or_drop("valid_trial_id", "trial_id IS NOT NULL")
@dp.expect("valid_drug_name", "drug_name IS NOT NULL")
def silver_clinical_trials():
    return (
        spark.read.table("bronze_clinical_trials")
        .select(
            col("trial_id"),
            col("trial_name"),
            col("drug_name"),
            col("indication"),
            col("phase"),
            col("sponsor"),
            col("status"),
            col("sites"),
            col("target_enrollment"),
            to_date(col("start_date")).alias("start_date"),
            to_date(col("end_date")).alias("end_date"),
            col("_ingestion_timestamp")
        )
    )

# COMMAND ----------

@dp.materialized_view(
    comment="Cleansed patient demographics with typed dates",
    table_properties={"delta.enableChangeDataFeed": "true"},
    cluster_by=["patient_id"]
)
@dp.expect_or_drop("valid_patient_id", "patient_id IS NOT NULL")
@dp.expect("valid_age", "age > 0 AND age < 120")
def silver_patients():
    return (
        spark.read.table("bronze_patients")
        .select(
            col("patient_id"),
            col("age"),
            col("gender"),
            col("blood_type"),
            col("region"),
            col("bmi"),
            col("height_cm"),
            col("weight_kg"),
            col("num_pre_existing"),
            col("pre_existing_conditions"),
            col("baseline_creatinine_mg_dl"),
            col("baseline_alt_iu_l"),
            to_date(col("enrollment_date")).alias("enrollment_date"),
            col("_ingestion_timestamp")
        )
    )

# COMMAND ----------

@dp.materialized_view(
    comment="Cleansed trial results with typed dates and quality checks",
    table_properties={"delta.enableChangeDataFeed": "true"},
    cluster_by=["trial_id", "patient_id"]
)
@dp.expect_or_drop("valid_result_id", "result_id IS NOT NULL")
@dp.expect_or_drop("valid_patient_id", "patient_id IS NOT NULL")
@dp.expect("valid_vitals", "blood_pressure_systolic > 0 AND heart_rate_bpm > 0")
def silver_trial_results():
    return (
        spark.read.table("bronze_trial_results")
        .select(
            col("result_id"),
            col("patient_id"),
            col("trial_id"),
            col("drug_name"),
            col("treatment_arm"),
            col("response"),
            col("visit_number"),
            to_date(col("visit_date")).alias("visit_date"),
            col("dosage_mg"),
            col("blood_pressure_systolic"),
            col("blood_pressure_diastolic"),
            col("heart_rate_bpm"),
            col("alt_iu_l"),
            col("creatinine_mg_dl"),
            col("tumor_size_mm"),
            col("_ingestion_timestamp")
        )
    )

# COMMAND ----------

# MAGIC ### Gold Layer — Curated Aggregates for BI (Materialized Views, CDF Enabled)

# COMMAND ----------

@dp.materialized_view(
    comment="Safety summary per trial and drug: adverse event counts, severity breakdown, and risk metrics",
    table_properties={"delta.enableChangeDataFeed": "true"},
    cluster_by=["trial_id", "drug_name"]
)
def gold_trial_safety_summary():
    ae = spark.read.table("silver_adverse_events")
    ct = spark.read.table("silver_clinical_trials")

    return (
        ae.groupBy("trial_id", "drug_name")
        .agg(
            count("*").alias("total_adverse_events"),
            _sum(when(col("serious") == True, 1).otherwise(0)).alias("serious_event_count"),
            _sum(when(col("severity") == "Severe", 1).otherwise(0)).alias("severe_count"),
            _sum(when(col("severity") == "Moderate", 1).otherwise(0)).alias("moderate_count"),
            _sum(when(col("severity") == "Mild", 1).otherwise(0)).alias("mild_count"),
            _round(avg("risk_score"), 2).alias("avg_risk_score"),
            _sum(when(col("drug_interaction_flag") == True, 1).otherwise(0)).alias("drug_interaction_count"),
            _sum(when(col("outcome") == "Fatal", 1).otherwise(0)).alias("fatal_outcome_count")
        )
        .join(
            ct.select("trial_id", "trial_name", "phase", "sponsor", "indication"),
            on="trial_id",
            how="left"
        )
    )

# COMMAND ----------

@dp.materialized_view(
    comment="Patient-level outcomes joining demographics with latest trial results",
    table_properties={"delta.enableChangeDataFeed": "true"},
    cluster_by=["patient_id", "trial_id"]
)
def gold_patient_outcomes():
    from pyspark.sql.window import Window
    from pyspark.sql.functions import row_number

    patients = spark.read.table("silver_patients")
    results = spark.read.table("silver_trial_results")
    ae = spark.read.table("silver_adverse_events")

    # Get latest visit per patient-trial
    latest_window = Window.partitionBy("patient_id", "trial_id").orderBy(col("visit_number").desc())
    latest_results = (
        results
        .withColumn("_rank", row_number().over(latest_window))
        .filter(col("_rank") == 1)
        .drop("_rank")
    )

    # Count adverse events per patient-trial
    ae_counts = (
        ae.groupBy("patient_id", "trial_id")
        .agg(
            count("*").alias("adverse_event_count"),
            _sum(when(col("serious") == True, 1).otherwise(0)).alias("serious_ae_count")
        )
    )

    return (
        latest_results
        .join(patients, on="patient_id", how="left")
        .join(ae_counts, on=["patient_id", "trial_id"], how="left")
        .select(
            latest_results["patient_id"],
            latest_results["trial_id"],
            col("drug_name"),
            col("treatment_arm"),
            col("response"),
            col("visit_number"),
            col("visit_date"),
            col("blood_pressure_systolic"),
            col("blood_pressure_diastolic"),
            col("heart_rate_bpm"),
            col("alt_iu_l"),
            col("creatinine_mg_dl"),
            col("tumor_size_mm"),
            col("age"),
            col("gender"),
            col("region"),
            col("bmi"),
            col("num_pre_existing"),
            col("adverse_event_count"),
            col("serious_ae_count")
        )
    )

# COMMAND ----------

@dp.materialized_view(
    comment="Drug efficacy metrics aggregated by drug and treatment arm",
    table_properties={"delta.enableChangeDataFeed": "true"},
    cluster_by=["drug_name", "treatment_arm"]
)
def gold_drug_efficacy():
    results = spark.read.table("silver_trial_results")
    trials = spark.read.table("silver_clinical_trials")

    return (
        results.groupBy("drug_name", "treatment_arm", "trial_id")
        .agg(
            count("*").alias("total_observations"),
            _sum(when(col("response") == "Complete Response", 1).otherwise(0)).alias("complete_response_count"),
            _sum(when(col("response") == "Partial Response", 1).otherwise(0)).alias("partial_response_count"),
            _sum(when(col("response") == "Stable Disease", 1).otherwise(0)).alias("stable_disease_count"),
            _sum(when(col("response") == "Progressive Disease", 1).otherwise(0)).alias("progressive_disease_count"),
            _round(avg("blood_pressure_systolic"), 1).alias("avg_systolic_bp"),
            _round(avg("heart_rate_bpm"), 1).alias("avg_heart_rate"),
            _round(avg("alt_iu_l"), 1).alias("avg_alt"),
            _round(avg("creatinine_mg_dl"), 2).alias("avg_creatinine")
        )
        .join(
            trials.select("trial_id", "trial_name", "phase", "indication", "sponsor"),
            on="trial_id",
            how="left"
        )
    )
