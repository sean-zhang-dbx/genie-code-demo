## Data Engineering
* Default to **Spark Declarative Pipelines (SDP)** for all data pipeline development
* Follow the **medallion architecture**: `bronze` = raw ingestion, `silver` = cleansed/conformed, `gold` = curated aggregates for BI
* Use **Auto Loader** (`cloudFiles`) for incremental file ingestion into bronze tables
* use @dp instead of @dlt (dlt is legacy syntax)
* Use **streaming tables** for append-only ingestion and **materialized views** for transformation layers
* Parameterize pipelines using **pipeline configurations** rather than hardcoded values
* Apply **Liquid Clustering** over traditional partitioning for Delta tables
* Enable **Change Data Feed (CDF)** on silver/gold tables when downstream consumers need change tracking

## SQL & BI Workflows
* Always use **Unity Catalog three-level namespace** (`catalog.schema.table`) in all SQL queries
* Prefer **Databricks SQL Warehouse** (serverless) for ad-hoc queries and dashboard workloads
* Use **parameterized queries** (`:param_name` syntax) for reusable, filterable reports
* Create **gold-layer materialized views** or **aggregation tables** to back dashboards — never query raw bronze tables directly from BI
* Use `DESCRIBE EXTENDED`, `SHOW TABLES`, and `INFORMATION_SCHEMA` for catalog exploration
* Optimize query performance with **ANALYZE TABLE** to compute statistics and leverage **predictive I/O**

## Data Science
* Always perform EDA - look into descriptive statistics such as distributions, null counts, and correlation between features and target
* Always track experiments using **MLflow** - log parameters, metrics, and artifacts to an MLflow experiment
* Perform **hyperparameter tuning** using `Hyperopt` with `SparkTrials` for distributed search
* Register final models to the **Unity Catalog Model Registry** (`models:/catalog.schema.model_name`)
* Use **Feature Engineering in Unity Catalog** for feature storage and lineage tracking
* Develop and iterate in Databricks **notebooks**, then promote to production via **MLflow Models**
* Set model lifecycle stages (Champion/Challenger) via **model aliases** in Unity Catalog
* Use `mlflow.autolog()` to automatically capture framework-specific metrics (sklearn, pytorch, etc.)
* For large-scale training, leverage **pandas UDFs** or **Spark ML pipelines** to distribute workloads

## Custom Skills

### Genie Space Management
When working with Databricks AI/BI Genie spaces — creating, managing, auditing, diagnosing issues, or optimizing:
- **Always load first**: `/Users/{username}/.assistant/skills/prompt-to-genie/SKILL.md`
- This contains the most up-to-date API documentation, error codes, best practices, and troubleshooting guidance
- Use the **Create a New Space** workflow when the user wants to build a new Genie space
- Use the **Diagnose and Optimize an Existing Space** workflow when the user wants to review, audit, fix, or optimize an existing space
