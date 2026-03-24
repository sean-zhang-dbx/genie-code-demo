# Databricks notebook source

# MAGIC %md
# MAGIC # Genie Space — Drug Efficacy Analytics
# MAGIC
# MAGIC Creates a Genie AI/BI space for drug efficacy comparison across clinical trials, including ORR calculations, safety profiles, and response breakdowns.

# COMMAND ----------

dbutils.widgets.text("catalog", "", "UC Catalog")
dbutils.widgets.text("schema", "genie_code_assets", "UC Schema")
dbutils.widgets.text("warehouse_id", "", "SQL Warehouse ID")

# COMMAND ----------

# DBTITLE 1,Build Genie space configuration
import secrets
import json

def gen_id():
    return secrets.token_hex(16)

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

serialized_space = {
    "version": 2,
    "config": {
        "sample_questions": sorted([
            {"id": gen_id(), "question": ["What is the ORR for Nexivora?"]},
            {"id": gen_id(), "question": ["Compare efficacy of Drug vs Placebo arms across all trials"]},
            {"id": gen_id(), "question": ["Which drugs have the highest serious adverse event rate?"]},
            {"id": gen_id(), "question": ["Show the top drugs by overall response rate in oncology indications"]},
            {"id": gen_id(), "question": ["What is the safety profile for Oncozumab?"]},
        ], key=lambda x: x["id"])
    },
    "data_sources": {
        "tables": sorted([
            {
                "identifier": f"{catalog}.{schema}.gold_patient_outcomes",
                "description": ["Patient-level outcomes joining demographics with latest trial results, including vitals, lab values, tumor measurements, and adverse event counts."],
                "column_configs": sorted([
                    {"column_name": "drug_name", "description": ["Name of the drug being tested"], "enable_format_assistance": True, "enable_entity_matching": True},
                    {"column_name": "treatment_arm", "description": ["Treatment group: Drug, Drug + Standard of Care, Placebo, or Standard of Care"], "enable_format_assistance": True, "enable_entity_matching": True},
                    {"column_name": "response", "description": ["Patient treatment response: Complete Response, Partial Response, Stable Disease, or Progressive Disease"], "enable_format_assistance": True, "enable_entity_matching": True},
                    {"column_name": "gender", "description": ["Patient gender: Male or Female"], "enable_format_assistance": True, "enable_entity_matching": True},
                    {"column_name": "region", "description": ["Geographic region: Africa, Asia, Europe, Middle East, North America, Oceania, South America"], "enable_format_assistance": True, "enable_entity_matching": True},
                    {"column_name": "trial_id", "description": ["Unique trial identifier (e.g., TRIAL-001)"], "enable_format_assistance": True, "enable_entity_matching": True},
                ], key=lambda x: x["column_name"])
            },
            {
                "identifier": f"{catalog}.{schema}.gold_drug_efficacy",
                "description": ["Drug efficacy metrics aggregated by trial, drug, and treatment arm. Includes response counts (complete, partial, stable, progressive), average vitals/labs, and trial metadata."],
                "column_configs": sorted([
                    {"column_name": "drug_name", "description": ["Name of the drug being tested"], "enable_format_assistance": True, "enable_entity_matching": True},
                    {"column_name": "treatment_arm", "description": ["Treatment group: Drug, Drug + Standard of Care, Placebo, or Standard of Care"], "enable_format_assistance": True, "enable_entity_matching": True},
                    {"column_name": "trial_id", "description": ["Unique trial identifier"], "enable_format_assistance": True, "enable_entity_matching": True},
                    {"column_name": "trial_name", "description": ["Human-readable trial name (e.g., Study A-962)"], "enable_format_assistance": True, "enable_entity_matching": True},
                    {"column_name": "phase", "description": ["Clinical trial phase: Phase I, Phase II, Phase III, Phase IV"], "enable_format_assistance": True, "enable_entity_matching": True},
                    {"column_name": "indication", "description": ["Disease or condition being treated"], "enable_format_assistance": True, "enable_entity_matching": True},
                    {"column_name": "sponsor", "description": ["Pharmaceutical company sponsoring the trial"], "enable_format_assistance": True, "enable_entity_matching": True},
                ], key=lambda x: x["column_name"])
            },
            {
                "identifier": f"{catalog}.{schema}.gold_trial_safety_summary",
                "description": ["Safety summary per trial and drug including adverse event counts by severity (mild, moderate, severe, serious), average risk score, drug interactions, and fatal outcomes."],
                "column_configs": sorted([
                    {"column_name": "drug_name", "description": ["Name of the drug"], "enable_format_assistance": True, "enable_entity_matching": True},
                    {"column_name": "trial_id", "description": ["Unique trial identifier"], "enable_format_assistance": True, "enable_entity_matching": True},
                    {"column_name": "trial_name", "description": ["Human-readable trial name"], "enable_format_assistance": True, "enable_entity_matching": True},
                    {"column_name": "phase", "description": ["Clinical trial phase"], "enable_format_assistance": True, "enable_entity_matching": True},
                    {"column_name": "sponsor", "description": ["Sponsoring company"], "enable_format_assistance": True, "enable_entity_matching": True},
                    {"column_name": "indication", "description": ["Disease being treated"], "enable_format_assistance": True, "enable_entity_matching": True},
                ], key=lambda x: x["column_name"])
            },
        ], key=lambda x: x["identifier"])
    },
    "instructions": {
        "text_instructions": [
            {
                "id": gen_id(),
                "content": [
                    "This space focuses on drug efficacy comparison across clinical trials.\n",
                    "ORR (Overall Response Rate) is calculated as (Complete Response + Partial Response) / Total Observations * 100, expressed as a percentage.\n",
                    "Response categories are: Complete Response, Partial Response, Stable Disease, and Progressive Disease.\n",
                    "Treatment arms are: Drug (experimental), Drug + Standard of Care (combination), Placebo, and Standard of Care.\n",
                    "When comparing drug efficacy, always include the treatment arm for context.\n",
                    "Oncology indications include: Non-Small Cell Lung Cancer, Breast Cancer, Colorectal Cancer, and Hepatocellular Carcinoma.\n",
                    "All percentage values should be rounded to one decimal place.\n",
                ]
            }
        ],
        "example_question_sqls": sorted([
            {
                "id": gen_id(),
                "question": ["What is the ORR by drug and treatment arm?"],
                "sql": [
                    "SELECT\n",
                    "  drug_name,\n",
                    "  treatment_arm,\n",
                    "  total_observations,\n",
                    "  complete_response_count + partial_response_count AS responders,\n",
                    "  ROUND((complete_response_count + partial_response_count) * 100.0 / total_observations, 1) AS orr_pct\n",
                    f"FROM {catalog}.{schema}.gold_drug_efficacy\n",
                    "ORDER BY orr_pct DESC"
                ],
                "usage_guidance": ["Use this pattern for ORR calculations across drugs and treatment arms"]
            },
            {
                "id": gen_id(),
                "question": ["Compare drug vs. placebo response rates for a specific trial"],
                "sql": [
                    "SELECT\n",
                    "  drug_name,\n",
                    "  treatment_arm,\n",
                    "  total_observations,\n",
                    "  complete_response_count,\n",
                    "  partial_response_count,\n",
                    "  ROUND((complete_response_count + partial_response_count) * 100.0 / total_observations, 1) AS orr_pct\n",
                    f"FROM {catalog}.{schema}.gold_drug_efficacy\n",
                    "WHERE trial_id = :trial_id\n",
                    "ORDER BY orr_pct DESC"
                ],
                "parameters": [
                    {"name": "trial_id", "description": ["Trial ID to filter on (e.g., TRIAL-001)"], "type_hint": "STRING", "default_value": {"values": ["TRIAL-001"]}}
                ],
                "usage_guidance": ["Use for head-to-head drug vs. placebo comparisons within a single trial"]
            },
            {
                "id": gen_id(),
                "question": ["What is the safety profile for each drug?"],
                "sql": [
                    "SELECT\n",
                    "  drug_name,\n",
                    "  total_adverse_events,\n",
                    "  serious_event_count,\n",
                    "  ROUND(serious_event_count * 100.0 / total_adverse_events, 1) AS serious_ae_rate_pct,\n",
                    "  severe_count,\n",
                    "  moderate_count,\n",
                    "  mild_count,\n",
                    "  avg_risk_score,\n",
                    "  fatal_outcome_count\n",
                    f"FROM {catalog}.{schema}.gold_trial_safety_summary\n",
                    "ORDER BY serious_ae_rate_pct DESC"
                ],
                "usage_guidance": ["Use for safety profile and adverse event analysis by drug"]
            },
            {
                "id": gen_id(),
                "question": ["Which drugs have the highest ORR in oncology indications?"],
                "sql": [
                    "SELECT\n",
                    "  drug_name,\n",
                    "  indication,\n",
                    "  phase,\n",
                    "  treatment_arm,\n",
                    "  ROUND((complete_response_count + partial_response_count) * 100.0 / total_observations, 1) AS orr_pct\n",
                    f"FROM {catalog}.{schema}.gold_drug_efficacy\n",
                    "WHERE indication IN ('Non-Small Cell Lung Cancer', 'Breast Cancer', 'Colorectal Cancer', 'Hepatocellular Carcinoma')\n",
                    "  AND treatment_arm IN ('Drug', 'Drug + Standard of Care')\n",
                    "ORDER BY orr_pct DESC"
                ],
                "usage_guidance": ["Use for oncology-specific efficacy ranking"]
            },
        ], key=lambda x: x["id"]),
        "join_specs": sorted([
            {
                "id": gen_id(),
                "left": {"identifier": f"{catalog}.{schema}.gold_patient_outcomes", "alias": "gold_patient_outcomes"},
                "right": {"identifier": f"{catalog}.{schema}.gold_drug_efficacy", "alias": "gold_drug_efficacy"},
                "sql": [
                    "`gold_patient_outcomes`.`trial_id` = `gold_drug_efficacy`.`trial_id` AND `gold_patient_outcomes`.`drug_name` = `gold_drug_efficacy`.`drug_name` AND `gold_patient_outcomes`.`treatment_arm` = `gold_drug_efficacy`.`treatment_arm`",
                    "--rt=FROM_RELATIONSHIP_TYPE_MANY_TO_ONE--"
                ],
                "comment": ["Join patient outcomes to drug efficacy aggregates on trial, drug, and treatment arm"],
                "instruction": ["Use this join to relate individual patient data with aggregated efficacy metrics"]
            },
            {
                "id": gen_id(),
                "left": {"identifier": f"{catalog}.{schema}.gold_drug_efficacy", "alias": "gold_drug_efficacy"},
                "right": {"identifier": f"{catalog}.{schema}.gold_trial_safety_summary", "alias": "gold_trial_safety_summary"},
                "sql": [
                    "`gold_drug_efficacy`.`trial_id` = `gold_trial_safety_summary`.`trial_id` AND `gold_drug_efficacy`.`drug_name` = `gold_trial_safety_summary`.`drug_name`",
                    "--rt=FROM_RELATIONSHIP_TYPE_MANY_TO_ONE--"
                ],
                "comment": ["Join drug efficacy to trial safety summary on trial and drug"],
                "instruction": ["Use this join to combine efficacy and safety data for a drug"]
            },
        ], key=lambda x: x["id"]),
        "sql_snippets": {
            "measures": sorted([
                {
                    "id": gen_id(),
                    "alias": "orr_pct",
                    "display_name": "Overall Response Rate (ORR)",
                    "sql": ["ROUND((gold_drug_efficacy.complete_response_count + gold_drug_efficacy.partial_response_count) * 100.0 / gold_drug_efficacy.total_observations, 1)"],
                    "synonyms": ["ORR", "overall response rate", "response rate", "efficacy rate"],
                    "instruction": ["Use for any question about drug response rate or ORR"],
                    "comment": ["ORR = (CR + PR) / total * 100, rounded to 1 decimal"]
                },
                {
                    "id": gen_id(),
                    "alias": "serious_ae_rate_pct",
                    "display_name": "Serious Adverse Event Rate",
                    "sql": ["ROUND(gold_trial_safety_summary.serious_event_count * 100.0 / gold_trial_safety_summary.total_adverse_events, 1)"],
                    "synonyms": ["serious AE rate", "SAE rate", "serious adverse event percentage"],
                    "instruction": ["Use for questions about safety and serious adverse event rates"],
                    "comment": ["Serious AE rate = serious events / total AEs * 100"]
                },
            ], key=lambda x: x["id"]),
            "filters": sorted([
                {
                    "id": gen_id(),
                    "display_name": "oncology trials",
                    "sql": ["gold_drug_efficacy.indication IN ('Non-Small Cell Lung Cancer', 'Breast Cancer', 'Colorectal Cancer', 'Hepatocellular Carcinoma')"],
                    "synonyms": ["oncology", "cancer trials", "cancer indications"],
                    "instruction": ["Apply when users ask about oncology or cancer trials"]
                },
                {
                    "id": gen_id(),
                    "display_name": "high risk",
                    "sql": ["gold_trial_safety_summary.avg_risk_score > 3.0"],
                    "synonyms": ["high risk score", "risky", "dangerous"],
                    "instruction": ["Apply when users ask about high-risk drugs or trials"]
                },
            ], key=lambda x: x["id"]),
            "expressions": sorted([
                {
                    "id": gen_id(),
                    "alias": "trial_phase",
                    "display_name": "Trial Phase",
                    "sql": ["gold_drug_efficacy.phase"],
                    "synonyms": ["phase", "clinical phase", "study phase"],
                    "instruction": ["Use for grouping or filtering by trial phase"]
                },
            ], key=lambda x: x["id"])
        }
    },
    "benchmarks": {
        "questions": sorted([
            # Core benchmarks: rephrased example SQL queries (reuse exact ground truth SQL)
            {"id": gen_id(), "question": ["Show me the overall response rate for each drug across all treatment arms"], "answer": [{"format": "SQL", "content": ["SELECT\n", "  drug_name,\n", "  treatment_arm,\n", "  total_observations,\n", "  complete_response_count + partial_response_count AS responders,\n", "  ROUND((complete_response_count + partial_response_count) * 100.0 / total_observations, 1) AS orr_pct\n", f"FROM {catalog}.{schema}.gold_drug_efficacy\n", "ORDER BY orr_pct DESC"]}]},
            {"id": gen_id(), "question": ["Rank all drugs by their ORR percentage"], "answer": [{"format": "SQL", "content": ["SELECT\n", "  drug_name,\n", "  treatment_arm,\n", "  total_observations,\n", "  complete_response_count + partial_response_count AS responders,\n", "  ROUND((complete_response_count + partial_response_count) * 100.0 / total_observations, 1) AS orr_pct\n", f"FROM {catalog}.{schema}.gold_drug_efficacy\n", "ORDER BY orr_pct DESC"]}]},
            {"id": gen_id(), "question": ["Show the adverse event breakdown by severity for all drugs"], "answer": [{"format": "SQL", "content": ["SELECT\n", "  drug_name,\n", "  total_adverse_events,\n", "  serious_event_count,\n", "  ROUND(serious_event_count * 100.0 / total_adverse_events, 1) AS serious_ae_rate_pct,\n", "  severe_count,\n", "  moderate_count,\n", "  mild_count,\n", "  avg_risk_score,\n", "  fatal_outcome_count\n", f"FROM {catalog}.{schema}.gold_trial_safety_summary\n", "ORDER BY serious_ae_rate_pct DESC"]}]},
            {"id": gen_id(), "question": ["Which drugs have the highest rate of serious adverse events?"], "answer": [{"format": "SQL", "content": ["SELECT\n", "  drug_name,\n", "  total_adverse_events,\n", "  serious_event_count,\n", "  ROUND(serious_event_count * 100.0 / total_adverse_events, 1) AS serious_ae_rate_pct,\n", "  severe_count,\n", "  moderate_count,\n", "  mild_count,\n", "  avg_risk_score,\n", "  fatal_outcome_count\n", f"FROM {catalog}.{schema}.gold_trial_safety_summary\n", "ORDER BY serious_ae_rate_pct DESC"]}]},
            {"id": gen_id(), "question": ["What are the best-performing drugs in cancer indications by response rate?"], "answer": [{"format": "SQL", "content": ["SELECT\n", "  drug_name,\n", "  indication,\n", "  phase,\n", "  treatment_arm,\n", "  ROUND((complete_response_count + partial_response_count) * 100.0 / total_observations, 1) AS orr_pct\n", f"FROM {catalog}.{schema}.gold_drug_efficacy\n", "WHERE indication IN ('Non-Small Cell Lung Cancer', 'Breast Cancer', 'Colorectal Cancer', 'Hepatocellular Carcinoma')\n", "  AND treatment_arm IN ('Drug', 'Drug + Standard of Care')\n", "ORDER BY orr_pct DESC"]}]},
            {"id": gen_id(), "question": ["Rank oncology drugs by ORR for active treatment arms only"], "answer": [{"format": "SQL", "content": ["SELECT\n", "  drug_name,\n", "  indication,\n", "  phase,\n", "  treatment_arm,\n", "  ROUND((complete_response_count + partial_response_count) * 100.0 / total_observations, 1) AS orr_pct\n", f"FROM {catalog}.{schema}.gold_drug_efficacy\n", "WHERE indication IN ('Non-Small Cell Lung Cancer', 'Breast Cancer', 'Colorectal Cancer', 'Hepatocellular Carcinoma')\n", "  AND treatment_arm IN ('Drug', 'Drug + Standard of Care')\n", "ORDER BY orr_pct DESC"]}]},
            # Stretch benchmarks: new questions without example SQL
            {"id": gen_id(), "question": ["What is the average risk score by sponsor?"], "answer": [{"format": "SQL", "content": ["SELECT\n", "  sponsor,\n", "  ROUND(AVG(avg_risk_score), 2) AS mean_risk_score\n", f"FROM {catalog}.{schema}.gold_trial_safety_summary\n", "GROUP BY sponsor\n", "ORDER BY mean_risk_score DESC"]}]},
            {"id": gen_id(), "question": ["Which Phase III trials have the most total adverse events?"], "answer": [{"format": "SQL", "content": ["SELECT\n", "  trial_name,\n", "  drug_name,\n", "  total_adverse_events,\n", "  serious_event_count,\n", "  fatal_outcome_count\n", f"FROM {catalog}.{schema}.gold_trial_safety_summary\n", "WHERE phase = 'Phase III'\n", "ORDER BY total_adverse_events DESC"]}]},
            {"id": gen_id(), "question": ["How many patients had a complete response by drug and region?"], "answer": [{"format": "SQL", "content": ["SELECT\n", "  drug_name,\n", "  region,\n", "  COUNT(*) AS complete_responders\n", f"FROM {catalog}.{schema}.gold_patient_outcomes\n", "WHERE response = 'Complete Response'\n", "GROUP BY drug_name, region\n", "ORDER BY complete_responders DESC"]}]},
            {"id": gen_id(), "question": ["What is the ORR for drugs in Phase II trials?"], "answer": [{"format": "SQL", "content": ["SELECT\n", "  drug_name,\n", "  treatment_arm,\n", "  indication,\n", "  ROUND((complete_response_count + partial_response_count) * 100.0 / total_observations, 1) AS orr_pct\n", f"FROM {catalog}.{schema}.gold_drug_efficacy\n", "WHERE phase = 'Phase II'\n", "ORDER BY orr_pct DESC"]}]},
        ], key=lambda x: x["id"])
    }
}

serialized_space_json = json.dumps(serialized_space, indent=2)
print(f"Config built successfully:")
print(f"  - {len(serialized_space['config']['sample_questions'])} sample questions")
print(f"  - {len(serialized_space['data_sources']['tables'])} tables")
print(f"  - {len(serialized_space['instructions']['example_question_sqls'])} example SQL queries")
print(f"  - {len(serialized_space['instructions']['sql_snippets']['measures'])} measures, {len(serialized_space['instructions']['sql_snippets']['filters'])} filters, {len(serialized_space['instructions']['sql_snippets']['expressions'])} dimensions")
print(f"  - {len(serialized_space['instructions']['join_specs'])} join specs")
print(f"  - {len(serialized_space['benchmarks']['questions'])} benchmarks (6 core + 4 stretch)")

# COMMAND ----------

# DBTITLE 1,Validate configuration
import re

current_user = spark.sql("SELECT current_user()").first()[0]
script_path = f"/Workspace/Users/{current_user}/.assistant/skills/prompt-to-genie/scripts/validate_config.py"
with open(script_path, "r") as f:
    script_content = f.read()

# Override the script's default config=None with our actual config
script_content = script_content.replace(
    "config = None",
    "config = __injected_config__",
    1  # Only replace the first occurrence
)

script_globals = {
    "__injected_config__": serialized_space,
    "__builtins__": __builtins__
}
exec(compile(script_content, script_path, "exec"), script_globals)

# COMMAND ----------

# DBTITLE 1,Test example SQL queries
passed = 0
failed = 0

for i, eq in enumerate(serialized_space["instructions"]["example_question_sqls"]):
    question = eq["question"][0]
    sql_parts = eq["sql"]
    query = "".join(sql_parts)

    # Replace parameters with default values
    for param in eq.get("parameters", []):
        default_val = param.get("default_value", {}).get("values", [""])[0]
        query = query.replace(f":{param['name']}", f"'{default_val}'")

    try:
        result = spark.sql(query)
        row_count = result.count()
        if row_count == 0:
            print(f"  ⚠️  Q{i+1}: '{question}' — 0 rows returned")
        else:
            print(f"  ✅ Q{i+1}: '{question}' — {row_count} rows")
            passed += 1
    except Exception as e:
        print(f"  ❌ Q{i+1}: '{question}' — ERROR: {e}")
        failed += 1

print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed out of {len(serialized_space['instructions']['example_question_sqls'])} queries")

# COMMAND ----------

# DBTITLE 1,Create Genie space
from databricks.sdk import WorkspaceClient
import json

w = WorkspaceClient()

warehouse_id = dbutils.widgets.get("warehouse_id")
if not warehouse_id:
    raise ValueError("warehouse_id widget is required. Set it to your SQL warehouse ID.")
current_user = spark.sql("SELECT current_user()").first()[0]
parent_path = f"/Users/{current_user}"
title = "Drug Efficacy Analytics"
description = "Compare drug efficacy across clinical trials — ORR, response breakdowns, and safety profiles for principal investigators and scientists."

response = w.api_client.do(
    method="POST",
    path="/api/2.0/genie/spaces",
    body={
        "title": title,
        "description": description,
        "warehouse_id": warehouse_id,
        "parent_path": parent_path,
        "serialized_space": json.dumps(serialized_space),
    },
)

space_id = response["space_id"]
workspace_url = w.config.host
print(f"✅ Genie space created successfully!")
print(f"   Title: {response.get('title')}")
print(f"   Space ID: {space_id}")
print(f"   URL: {workspace_url}/genie/rooms/{space_id}")
