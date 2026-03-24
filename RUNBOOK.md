# Genie Code Foundations: Demo Runbook

Step-by-step presenter guide for the Genie Code Foundations walkthrough.

**Workspace**: _(your Databricks workspace URL)_

**Catalog**: _(configured via `dbutils.widgets` at runtime; set your catalog and schema before running)_

---

## Agenda

| # | Section | What It Shows | Prompt |
|---|---------|---------------|--------|
| 1 | [Context Setting](#part-1-context-setting) | Unity Catalog metadata and the Genie Code interface | - |
| 2 | [Instructions](#part-2-instructions) | `.assistant_instructions.md`: persistent rules that guide every conversation | - |
| 3.1 | [Data Engineering Agent](#31-data-engineering-agent-instructions--no-plan) | SDP medallion pipeline generated from a single sentence, because instructions were already set | `Create a data pipeline for raw data in <catalog>.<schema>.<volume>` |
| 3.2 | [Dashboard Agent](#32-dashboard-agent-no-instructions--plan-public-preview) | Agent explores data, writes a PRD, then builds the dashboard | `Help me make a plan to create a clinical trials monitoring dashboard for: @gold_patient_outcomes @gold_drug_efficacy @gold_trial_safety_summary...` |
| 3.3 | [Data Science Agent](#33-data-science-agent-instructions--plan--interrupt) | EDA, model training, data leakage detection via human interrupt, model registration | `I want to create a machine learning model based on @silver_adverse_events. Could you come up with a plan to explore the data, summarize the findings from EDA, train initial model, then develop hyperparameter tuning?` |
| 4.1 | [Agent Skills](#41-agent-skills) | Custom skill triggers Genie Space creation (no built-in agent for this) | `I want to create a Genie space on the gold data in genie_code_assets.` |
| 4.2 | [MCP Servers](#42-mcp-servers-optional) _(optional)_ | GitHub MCP connects Genie Code to external systems | `Push this notebook to the genie-code-demo repository on GitHub.` |
| 5 | [Wrap-Up](#part-5-wrap-up) | Takeaways, call to action, roadmap | - |

---

## Pre-Demo Setup

Run these steps **before** the session starts.

### 1. Generate synthetic data

Run [`setup/0_pharma_data_generation.py`](setup/0_pharma_data_generation.py) in the workspace. Set the `catalog`, `schema`, and `volume` widgets before running. This creates the catalog, schema, volume, and writes JSON source files (patients, clinical trials, trial results, adverse events).

### 2. Deploy `.assistant_instructions.md`

Copy [.assistant_instructions.md](.assistant_instructions.md) to `/Users/<your-email>/.assistant_instructions.md` in the workspace.

### 3. (Optional) Set up GitHub MCP (for Part 4.2)

Only needed if you plan to demo the MCP section. Follow the official guide: [Enterprise code search via GitHub MCP](https://learn.microsoft.com/en-us/azure/databricks/genie-code/github-mcp)

Summary of steps:
1. Create a GitHub App (may require admin access) with callback URL `https://<workspace-url>/login/oauth/http.html`
2. Generate a client secret for the app
3. Create a Unity Catalog HTTP connection (OAuth User to Machine, host: `https://api.githubcopilot.com`)
4. Log in to the connection and add it as an MCP server in Genie Code settings

Have a **GitHub repository** ready to push code to during the MCP demo.

### 4. Verify Unity Catalog assets

Confirm these tables exist after the pipeline has been run at least once:

- **Bronze**: `bronze_adverse_events`, `bronze_clinical_trials`, `bronze_patients`, `bronze_trial_results`
- **Silver**: `silver_adverse_events`, `silver_clinical_trials`, `silver_patients`, `silver_trial_results`
- **Gold**: `gold_trial_safety_summary`, `gold_patient_outcomes`, `gold_drug_efficacy`
- **Model**: `serious_adverse_event_classifier` (registered after data science demo)

---

## Part 1: Context Setting

Before diving into demos, establish why Genie Code is different from other AI coding assistants: it has direct access to Unity Catalog metadata. This context is what makes every subsequent demo possible.

### 1.1 Show Unity Catalog Metadata

**Open**: UC Explorer, then navigate to your catalog > schema > `bronze_adverse_events`

**Talking points**:
- Genie Code has access to Unity Catalog, which other AI coding assistants do not
- It understands your data: table names, column descriptions, lineage, and usage patterns
- It also understands the Databricks platform: SDP, MLflow, UC model registry, dashboards

> **Screenshot**: _UC Explorer showing bronze_adverse_events table with columns and metadata_

### 1.2 Show the Genie Code Interface

**Open**: A new notebook with the Genie Code panel visible.

**Talking points**:
- Chat mode (single-turn) vs Agent mode (multi-step, autonomous)
- `@` references let you pull in UC tables, notebooks, and files as explicit context
- The agent can create multiple cells, run them, recover from errors, and iterate

---

## Part 2: Instructions

With Unity Catalog providing the data context, the next question is: how do you teach Genie Code your team's conventions? That is where instructions come in.

### 2.1 Show `.assistant_instructions.md`

**Open**: The `.assistant_instructions.md` file in the workspace editor.

**Source**: [.assistant_instructions.md](.assistant_instructions.md)

**Talking points**:
- Instructions are the foundation of working effectively with Genie Code
- If you have coding standards or established ways of working, encode them here
- I have used this file to guide the agent so that every conversation produces consistent, standards-compliant output without retyping detailed instructions each time
- Walk through the sections:
  - **Data Engineering**: Default to SDP, medallion architecture, Auto Loader, `@dp` not `@dlt`, Liquid Clustering, CDF
  - **SQL & BI**: Three-level UC namespace, serverless warehouse, parameterized queries, gold-layer only for BI
  - **Data Science**: EDA first, MLflow tracking, Hyperopt, UC Model Registry, Champion/Challenger aliases
  - **Custom Skills**: Genie Space management skill reference

**Key message**: Set rules once, and every AI interaction follows them automatically. Instructions can be set at user level or workspace level (admin).

---

## Part 3: Platform Walkthrough (Three Prompting Patterns)

Now that the audience understands the two types of context Genie Code works with (Unity Catalog metadata and custom instructions), demonstrate what happens when you combine them in practice.

The scenario: a pharma clinical research project collecting data on adverse events, patients, and trial outcomes. We need data engineering, data science, dashboarding, and a Genie space.

**Important**: All notebooks in `ai-generated-demos/` were **generated by Genie Code**, not hand-written. The only manual step was `setup/0_pharma_data_generation.py` (the synthetic data). Everything else was produced through conversations with the AI agent, each demonstrating a different prompting pattern.

---

### 3.1 Data Engineering Agent: Instructions + No Plan

> **Pattern**: Instructions were already set. No explicit plan in the prompt. The agent follows the instructions autonomously.

**Prompt used**:

```
Create a data pipeline for raw data in <catalog>.<schema>.<volume>
```

_(Replace `<catalog>.<schema>.<volume>` with your actual UC path, e.g. `my_catalog.genie_code_assets.raw_pharma_data`)_

**Open**:
- Notebook: `ai-generated-demos/1_data_engineering.py` (workspace copy)
- Genie Code conversation history for this notebook

**What to show**:
1. Open the conversation history. Note that there was **no plan requested** in the prompt, just a single sentence.
2. Because `.assistant_instructions.md` was already configured, the agent knew to use SDP, medallion architecture, Auto Loader, `@dp` syntax, Liquid Clustering, and CDF.
3. Briefly scroll through the notebook output:
   - **Bronze layer**: Streaming tables via Auto Loader with `cloudFiles`
   - **Silver layer**: Materialized views with type casting, data quality expectations (`@dp.expect_or_drop`), CDF enabled
   - **Gold layer**: Aggregation materialized views for BI (`gold_trial_safety_summary`, `gold_patient_outcomes`, `gold_drug_efficacy`)

**Talking points**:
- No detailed plan was needed because the instructions file already encoded all the conventions
- The agent picked up the standards from `.assistant_instructions.md` and applied them autonomously
- This is the "Instructions + No Plan" pattern, best when your standards are well-defined and the task is straightforward

---

### 3.2 Dashboard Agent: No Instructions + Plan (Public Preview)

> **Pattern**: No specific dashboard instructions were set. Asked Genie Code to create a plan first, reviewed it, then proceeded.
>
> **Note**: The Dashboard Agent is still in Public Preview. Your audience may not have access yet.

**Prompt used**:

```
Help me make a plan to create a clinical trials monitoring dashboard for:
@gold_patient_outcomes @gold_drug_efficacy @gold_trial_safety_summary.
First explore the data, then write a PRD outlining the basic structure of
the dashboard. The audience is mainly for principal investigators and
research scientists.
```

**Open**:
- The dashboard notebook + conversation history

**What to show**:
1. Show the prompt. Note the `@` references to three gold tables, and the explicit ask for a plan (PRD) before building.
2. Show the plan Genie Code generated. It should include data exploration findings and a dashboard structure.
3. Show that you reviewed the plan and asked questions before proceeding.
4. Show the final dashboard output.

**Talking points**:
- In this case, there were no dashboard-specific instructions, so the plan in the prompt served as a lightweight contract between you and the agent
- The `@` references give Genie Code direct access to Unity Catalog table schemas, which it uses to understand the available data
- This is the "Plan in Prompt + No Instructions" pattern, best for one-off tasks or new domains where you have not yet written instructions

> **Screenshot**: _Genie Code's generated plan / PRD for the dashboard_

---

### 3.3 Data Science Agent: Instructions + Plan + Interrupt

> **Pattern**: Instructions were set, a plan was requested, and the human interrupted mid-execution when something looked wrong.

**Prompt used**:

```
I want to create a machine learning model based on @silver_adverse_events.
Could you come up with a plan to explore the data, summarize the findings
from EDA, train initial model, then develop hyperparameter tuning?
```

**Open**:
- Notebook: `ai-generated-demos/2_data_science_exploratory.py` (workspace copy)
- Notebook: `ai-generated-demos/3_data_science_production.py` (workspace copy)
- Genie Code conversation history

**What to show**:

#### Step 1: Plan

1. Show the prompt and the plan Genie Code generated
2. The plan covers: EDA, preprocessing, baseline model, hyperparameter tuning, evaluation, registration

#### Step 2: Execute

3. Let the agent start executing: EDA, correlation matrix, feature-target visualizations
4. Baseline LightGBM model achieves **near-perfect ROC-AUC (~1.0)**

#### Step 3: Interrupt and Review Results

5. **Stop the agent here.** This is the key moment to demonstrate human-in-the-loop review.
6. Show the confusion matrix. It shows almost 100% accuracy. A perfect model is suspicious, and you should never fully trust this.
7. Show the feature importance chart. `risk_score` dominates everything.
8. Interrupt: "This looks like data leakage. `risk_score` might be derived from the target."
9. The agent helped identify and **drop the leaky feature** (`risk_score`), plus `outcome` and `severity` (post-event fields)
10. Retrained without leakage. Realistic ROC-AUC drops to a believable range.
11. Side-by-side comparison: with vs without `risk_score`

#### Step 4: Register

12. Model registered to MLflow and Unity Catalog Model Registry
13. Champion/Challenger alias pattern: validation gate compares new model against incumbent
14. Inference validation: loading Champion from UC and running sample predictions

**Talking points**:
- This is the "Instructions + Plan + Interrupt" pattern, the most effective approach for complex tasks
- The interrupt is critical: **never outsource all your thinking to AI**. This maps directly to the "Refine" step in the Plan, Build, Refine, Deploy workflow.
- The agent is a collaborator, not a replacement. You bring domain expertise (clinical safety leading to leakage suspicion), the agent brings implementation speed.
- Production notebook (`3_data_science_production.py`) adds: parameterized config, bundled PyFunc preprocessing, reusable functions, model validation gate, inference validation, scalability guard

> **Screenshot**: _Confusion matrix showing near-100% accuracy (the red flag)_

> **Screenshot**: _Feature importance chart showing risk_score dominating (the leakage signal)_

> **Screenshot**: _Conversation where the human interrupts about data leakage_

---

### 3.4 Slash Commands

**Demo** (optional, time permitting): Select a code block and run `/explain`, or demonstrate `/fix` on a cell with an error.

| Command | What It Does |
|---------|-------------|
| `/explain` | Explains code in plain English |
| `/fix` | Proposes a fix for code errors (diff view) |
| `/optimize` | Improves SQL, Python, and PySpark code |
| `/findTables` | Searches for tables via UC metadata |
| `/prettify` | Formats code for readability |
| `/doc` | Adds code comments (diff view) |
| `/repairEnvironment` | Diagnoses pip and environment issues |
| `/settings` | Opens Genie Code settings |

---

## Part 4: Customization

So far we have covered two layers of context: Unity Catalog metadata (automatic) and instructions (manual, always-on). This section introduces two more advanced customization mechanisms: Skills and MCP.

---

### 4.1 Agent Skills

**Talking points**:
- Skills are specialized, on-demand expertise packages for Genie Code. Think of them as domain-specific playbooks that are loaded only when the task requires them.
- Unlike instructions (which are always-on and limited to 20K characters), skills are **loaded on demand** when the task is relevant
- Location: `/Users/<you>/.assistant/skills/<name>/SKILL.md`

**How skills work**:
1. Only the **header** (title + description) of `SKILL.md` is exposed to Genie Code at all times, loaded alongside assistant instructions
2. A **full read** is triggered when the user mentions something matching the skill's description
3. Once triggered (show model thinking), the **entire skill** is loaded into context
4. Skills can include **references and scripts** that the agent can refer to and execute

**Example: `prompt-to-genie` custom skill**:

Use the `prompt-to-genie` skill as a concrete example. This is an open-source custom skill that teaches Genie Code how to create Genie AI/BI spaces, a workflow that has no built-in agent.

- **What it does**: Given a natural language description of the analytics use case and the relevant UC tables, the skill guides Genie Code to generate the full Genie Space configuration (sample questions, table descriptions, column configs, example SQL, join specs, measures, filters, benchmarks), validate it, test the SQL, and deploy via the SDK.
- **Why it matters**: There is no built-in "Genie Space agent". This capability was created entirely as a custom skill. Teams can build their own skills for any repeatable workflow.
- **Repo**: [prompt-to-genie](https://github.com/sean-zhang-dbx/prompt-to-genie/)

**Prompt used** (with the skill installed):

```
I want to create a Genie space on the gold data in genie_code_assets.
```

**What to show**:
1. Show the prompt. It is intentionally simple. The skill provides all the domain expertise.
2. Show the resulting Genie Space that was created (`ai-generated-demos/4_genie_space_custom_skill.py`)
3. If time permits, show the `SKILL.md` header and the model thinking panel where the skill gets loaded

**Key message**: Skills are how you extend Genie Code beyond its built-in agents. If your team has a repeatable workflow (code reviews, data quality audits, migration patterns, etc.), you can encode it as a skill.

> **Screenshot**: _The created Genie Space in the Databricks UI_

---

### 4.2 MCP Servers _(Optional)_

> Skip this section if you haven't set up the GitHub MCP server. You can still explain the concept without a live demo.

**Caveat**: Warn the audience that MCP setup requires some admin effort (private link configuration in some environments). See setup guide: [Enterprise code search via GitHub MCP](https://learn.microsoft.com/en-us/azure/databricks/genie-code/github-mcp)

**Talking points**:
- MCP (Model Context Protocol) connects Genie Code to **external systems** such as GitHub, JIRA, Slack, and email
- An MCP server is a set of functions/tools that the agent can call to interact with other systems
- Types: UC Functions, Vector Search, Genie Spaces, External (UC Connections), Custom (DB Apps)
- **20 tool limit** at the moment

**Demo: GitHub MCP**

**What to show**:
1. Show the MCP configuration in Genie Code settings
2. Ask Genie Code to push the notebook code to your GitHub repository. For example:

```
Push this notebook to the genie-code-demo repository on GitHub.
```

3. Show the conversation where the agent uses the GitHub MCP tools to interact with the repo (creating files, committing, etc.)

**Key message**: MCP is how you bring the rest of your organization's knowledge and tools into the coding assistant.

---

## Part 5: Wrap-Up

Wrap up with takeaways, call to action, and roadmap.

---

## Appendix: Repository Structure

```
genie-code-demo/
├── README.md                              # Project overview
├── RUNBOOK.md                             # This file (presenter guide)
├── .assistant_instructions.md             # Genie Code instructions (deployed to workspace)
├── setup/
│   └── 0_pharma_data_generation.py        # Pre-requisite: generates synthetic clinical trial data
└── ai-generated-demos/
    ├── 1_data_engineering.py              # Data Engineering Agent demo (SDP pipeline)
    ├── 2_data_science_exploratory.py      # Data Science Agent demo (EDA + leakage detection)
    ├── 3_data_science_production.py       # Data Science Agent demo (production MLOps)
    └── 4_genie_space_custom_skill.py      # Custom Skill demo (Genie Space creation)
```
