# Databricks notebook source

# COMMAND ----------

# MAGIC ## Pharma Data Generation
# MAGIC Generates fictional clinical trial data in **JSON format** and writes to a Unity Catalog **Volume**.
# MAGIC 
# MAGIC **Schema:** `sean_zhang_catalog.genie_code_assets`
# MAGIC 
# MAGIC **Files produced:**
# MAGIC - `patients.json` — demographics, BMI, pre-existing conditions, baseline labs
# MAGIC - `clinical_trials.json` — trial metadata, drug info, phases
# MAGIC - `trial_results.json` — per-visit treatment outcomes, lab values
# MAGIC - `adverse_events.json` — severity (Mild/Moderate/Severe) with **baked-in predictive signal**
# MAGIC 
# MAGIC **Classification target:** `severity` in adverse_events — correlated with age, BMI, dosage, liver enzymes, pre-existing conditions, and drug interactions.

# COMMAND ----------

catalog = "sean_zhang_catalog"
schema = "genie_code_assets"
volume = "raw_pharma_data"

spark.sql(f"DROP SCHEMA IF EXISTS {catalog}.{schema} CASCADE")
spark.sql(f"CREATE SCHEMA {catalog}.{schema}")
spark.sql(f"DROP VOLUME IF EXISTS {catalog}.{schema}.{volume}")
spark.sql(f"CREATE VOLUME {catalog}.{schema}.{volume}")

volume_path = f"/Volumes/{catalog}/{schema}/{volume}"
print(f"Volume path: {volume_path}")

# COMMAND ----------

import json
import random
import uuid
from datetime import datetime, timedelta

random.seed(42)

# --- Configuration ---
num_patients = 2000
num_trials = 15
num_results = 8000
num_adverse_events = 5000

regions = [
    "North America", "South America", "Europe", "Asia", "Africa", "Oceania", "Middle East"
]
genders = ["Male", "Female"]
blood_types = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
pre_existing = ["Diabetes", "Hypertension", "Asthma", "Hepatitis B", "CKD", "Obesity", "Anemia", "None"]

def random_date(start_year=2018, end_year=2024):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).strftime("%Y-%m-%d")

# --- Generate patients ---
patients = []
for i in range(num_patients):
    age = random.randint(18, 85)
    gender = random.choice(genders)
    height_cm = random.gauss(165 if gender == "Male" else 155, 8)
    weight_kg = random.gauss(72 if gender == "Male" else 62, 15)
    bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
    conditions = random.choices(pre_existing, k=random.randint(0, 3))
    conditions = list(set(c for c in conditions if c != "None")) or ["None"]

    # Baseline labs — higher liver enzymes for older/heavier patients (signal)
    base_alt = random.gauss(25, 10) + (age - 40) * 0.3 + max(0, (bmi - 28)) * 1.5
    base_creatinine = random.gauss(1.0, 0.2) + (age - 40) * 0.005

    patients.append({
        "patient_id": f"PAT-{i+1:05d}",
        "age": age,
        "gender": gender,
        "height_cm": round(height_cm, 1),
        "weight_kg": round(weight_kg, 1),
        "bmi": bmi,
        "blood_type": random.choice(blood_types),
        "region": random.choice(regions),
        "pre_existing_conditions": conditions,
        "num_pre_existing": len([c for c in conditions if c != "None"]),
        "baseline_alt_iu_l": round(max(5, base_alt), 1),
        "baseline_creatinine_mg_dl": round(max(0.4, base_creatinine), 2),
        "enrollment_date": random_date()
    })

print(f"Generated {len(patients)} patients")
print(f"Sample: {json.dumps(patients[0], indent=2)}")

# COMMAND ----------

drug_names = [
    "Nexivora", "Cardiflex", "Oncozumab", "Renapril", "Hepacure",
    "Pulmovent", "Neurostil", "Dermavax", "Glucomine", "Immunorel",
    "Thrombaxin", "Gastroval", "Osteonex", "Retivir", "Anxiolam"
]
indications = [
    "Non-Small Cell Lung Cancer", "Type 2 Diabetes", "Chronic Kidney Disease",
    "Hepatocellular Carcinoma", "Rheumatoid Arthritis", "Major Depressive Disorder",
    "Atrial Fibrillation", "COPD", "Psoriasis", "Breast Cancer",
    "Colorectal Cancer", "Hypertension", "Osteoporosis", "HIV", "Generalized Anxiety"
]
phases = ["Phase I", "Phase II", "Phase III", "Phase IV"]
trial_statuses = ["Recruiting", "Active", "Completed", "Suspended"]

# Drug interaction pairs (known to cause more severe AEs)
drug_interaction_pairs = {
    "Nexivora": ["Hepacure", "Renapril"],
    "Oncozumab": ["Immunorel", "Thrombaxin"],
    "Cardiflex": ["Thrombaxin"],
    "Neurostil": ["Anxiolam"]
}

clinical_trials = []
for i in range(num_trials):
    clinical_trials.append({
        "trial_id": f"TRIAL-{i+1:03d}",
        "trial_name": f"Study {chr(65+i)}-{random.randint(100,999)}",
        "drug_name": drug_names[i],
        "indication": indications[i],
        "phase": random.choice(phases),
        "start_date": random_date(2019, 2022),
        "end_date": random_date(2023, 2025),
        "status": random.choice(trial_statuses),
        "target_enrollment": random.randint(100, 500),
        "sponsor": random.choice(["GSK India", "Biocon", "Sun Pharma", "Dr. Reddy's", "Cipla"]),
        "sites": random.randint(3, 20)
    })

print(f"Generated {len(clinical_trials)} clinical trials")
print(f"Sample: {json.dumps(clinical_trials[0], indent=2)}")

# COMMAND ----------

treatment_arms = ["Drug", "Placebo", "Drug + Standard of Care", "Standard of Care"]
response_types = ["Complete Response", "Partial Response", "Stable Disease", "Progressive Disease"]

# Build patient lookup for correlation
patient_lookup = {p["patient_id"]: p for p in patients}

trial_results = []
for i in range(num_results):
    patient = random.choice(patients)
    trial = random.choice(clinical_trials)
    arm = random.choice(treatment_arms)
    dosage_mg = random.choice([50, 100, 150, 200, 250, 300, 400, 500])
    visit_num = random.randint(1, 12)

    # Response correlates with drug arm and dosage
    if arm in ["Drug", "Drug + Standard of Care"] and dosage_mg >= 200:
        response_weights = [0.25, 0.35, 0.25, 0.15]
    elif arm == "Placebo":
        response_weights = [0.05, 0.15, 0.40, 0.40]
    else:
        response_weights = [0.15, 0.25, 0.35, 0.25]

    trial_results.append({
        "result_id": f"RES-{i+1:06d}",
        "patient_id": patient["patient_id"],
        "trial_id": trial["trial_id"],
        "drug_name": trial["drug_name"],
        "treatment_arm": arm,
        "dosage_mg": dosage_mg,
        "visit_number": visit_num,
        "visit_date": random_date(2020, 2024),
        "response": random.choices(response_types, weights=response_weights, k=1)[0],
        "tumor_size_mm": round(random.gauss(35, 15), 1) if "Cancer" in trial["indication"] else None,
        "blood_pressure_systolic": random.randint(100, 180),
        "blood_pressure_diastolic": random.randint(60, 110),
        "heart_rate_bpm": random.randint(55, 110),
        "alt_iu_l": round(max(5, patient["baseline_alt_iu_l"] + random.gauss(0, 8) + dosage_mg * 0.02), 1),
        "creatinine_mg_dl": round(max(0.4, patient["baseline_creatinine_mg_dl"] + random.gauss(0, 0.1)), 2)
    })

print(f"Generated {len(trial_results)} trial results")
print(f"Sample: {json.dumps(trial_results[0], indent=2)}")

# COMMAND ----------

ae_types = [
    "Nausea", "Headache", "Fatigue", "Rash", "Diarrhea", "Liver Injury",
    "Neutropenia", "Anaphylaxis", "Cardiac Arrhythmia", "Renal Impairment",
    "Peripheral Neuropathy", "Thrombocytopenia", "Injection Site Reaction",
    "Insomnia", "Dizziness", "Elevated Liver Enzymes", "Hypertension"
]
outcomes = ["Recovered", "Recovering", "Not Recovered", "Fatal", "Unknown"]
severities = ["Mild", "Moderate", "Severe"]

def compute_severity(patient, dosage_mg, drug_name, concomitant_drug):
    """Compute severity with baked-in predictive signal."""
    # Start with a base risk score
    risk_score = 0.0

    # Age: older patients = higher risk
    risk_score += (patient["age"] - 40) * 0.03

    # BMI extremes: underweight or obese = higher risk
    bmi = patient["bmi"]
    if bmi < 18.5 or bmi > 32:
        risk_score += 1.5
    elif bmi > 28:
        risk_score += 0.7

    # Dosage: higher dosage = higher risk
    risk_score += (dosage_mg - 150) * 0.008

    # Pre-existing conditions: more = higher risk
    risk_score += patient["num_pre_existing"] * 0.8

    # Baseline liver enzymes: elevated ALT = higher risk
    if patient["baseline_alt_iu_l"] > 40:
        risk_score += 1.2
    if patient["baseline_alt_iu_l"] > 60:
        risk_score += 1.0

    # Baseline creatinine: elevated = higher risk
    if patient["baseline_creatinine_mg_dl"] > 1.3:
        risk_score += 0.9

    # Drug interactions: known pairs = significant bump
    has_interaction = False
    if drug_name in drug_interaction_pairs:
        if concomitant_drug in drug_interaction_pairs[drug_name]:
            risk_score += 2.5
            has_interaction = True

    # Add noise
    risk_score += random.gauss(0, 1.0)

    # Map score to severity
    if risk_score > 3.5:
        severity = "Severe"
    elif risk_score > 1.5:
        severity = "Moderate"
    else:
        severity = "Mild"

    return severity, has_interaction, round(risk_score, 2)

# --- Generate adverse events ---
adverse_events = []
for i in range(num_adverse_events):
    patient = random.choice(patients)
    trial = random.choice(clinical_trials)
    dosage_mg = random.choice([50, 100, 150, 200, 250, 300, 400, 500])
    concomitant = random.choice(drug_names)  # could trigger interaction

    severity, has_interaction, risk_score = compute_severity(
        patient, dosage_mg, trial["drug_name"], concomitant
    )

    # Outcome correlates with severity
    if severity == "Severe":
        outcome_weights = [0.15, 0.20, 0.35, 0.15, 0.15]
    elif severity == "Moderate":
        outcome_weights = [0.35, 0.30, 0.20, 0.02, 0.13]
    else:
        outcome_weights = [0.60, 0.25, 0.08, 0.0, 0.07]

    ae_type = random.choice(ae_types)
    # Severe AEs more likely to be serious types
    if severity == "Severe" and random.random() > 0.4:
        ae_type = random.choice(["Liver Injury", "Neutropenia", "Anaphylaxis",
                                  "Cardiac Arrhythmia", "Renal Impairment", "Thrombocytopenia"])

    onset_date = random_date(2020, 2024)
    adverse_events.append({
        "event_id": f"AE-{i+1:06d}",
        "patient_id": patient["patient_id"],
        "trial_id": trial["trial_id"],
        "drug_name": trial["drug_name"],
        "dosage_mg": dosage_mg,
        "concomitant_drug": concomitant,
        "drug_interaction_flag": has_interaction,
        "ae_type": ae_type,
        "severity": severity,
        "risk_score": risk_score,
        "onset_date": onset_date,
        "days_on_treatment": random.randint(1, 365),
        "outcome": random.choices(outcomes, weights=outcome_weights, k=1)[0],
        "serious": severity == "Severe" or ae_type in ["Anaphylaxis", "Cardiac Arrhythmia", "Liver Injury"],
        "reported_by": random.choice(["Physician", "Patient", "Nurse", "Pharmacist"])
    })

# Print distribution
from collections import Counter
sev_dist = Counter(ae["severity"] for ae in adverse_events)
print(f"Generated {len(adverse_events)} adverse events")
print(f"Severity distribution: {dict(sev_dist)}")
print(f"\nSample: {json.dumps(adverse_events[0], indent=2)}")

# COMMAND ----------

import os

def write_json_to_volume(data, filename):
    path = f"{volume_path}/{filename}"
    json_str = json.dumps(data, indent=2)
    dbutils.fs.put(path, json_str, overwrite=True)
    size_mb = len(json_str) / (1024 * 1024)
    print(f"Wrote {path} ({len(data):,} records, {size_mb:.2f} MB)")

write_json_to_volume(patients, "patients.json")
write_json_to_volume(clinical_trials, "clinical_trials.json")
write_json_to_volume(trial_results, "trial_results.json")
write_json_to_volume(adverse_events, "adverse_events.json")

print("\n--- Volume contents ---")
display(dbutils.fs.ls(volume_path))

# COMMAND ----------

import pandas as pd

# Quick verification of predictive signal in adverse events
ae_df = pd.DataFrame(adverse_events)
patient_df = pd.DataFrame(patients)

# Merge to check correlations
merged = ae_df.merge(patient_df, on="patient_id", how="left")

print("=== Severity Distribution ===")
print(merged["severity"].value_counts())
print(f"\n=== Mean Age by Severity ===")
print(merged.groupby("severity")["age"].mean().round(1))
print(f"\n=== Mean BMI by Severity ===")
print(merged.groupby("severity")["bmi"].mean().round(1))
print(f"\n=== Mean Dosage by Severity ===")
print(merged.groupby("severity")["dosage_mg"].mean().round(1))
print(f"\n=== Mean Baseline ALT by Severity ===")
print(merged.groupby("severity")["baseline_alt_iu_l"].mean().round(1))
print(f"\n=== Mean Pre-existing Conditions by Severity ===")
print(merged.groupby("severity")["num_pre_existing"].mean().round(2))
print(f"\n=== Drug Interaction Rate by Severity ===")
print(merged.groupby("severity")["drug_interaction_flag"].mean().round(3))
print(f"\n=== Mean Risk Score by Severity ===")
print(merged.groupby("severity")["risk_score"].mean().round(2))
