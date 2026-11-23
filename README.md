# Vantage Automotive: Business Operations Predictive Support System

**A Complete Predictive Support System that Helps Reduce Customer Churn, Stops Payment Fraud and Boosts Operational Value**

**SQL | Python | Power BI | Streamlit** | Five predictive models for a fictional global carmaker

**Author:** Adewale Oshinukan | Data Scientist | [GitHub](https://github.com/waleoshin75-sketch) | [Live App](https://vantage-automotive-business-operations-predictive-support-syst.streamlit.app/)

[Live Streamlit App](https://vantage-automotive-business-operations-predictive-support-syst.streamlit.app/) | [GitHub Repository](https://github.com/waleoshin75-sketch/Vantage-Automotive-Business-Operations-Predictive-Support-System)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vantage-automotive-business-operations-predictive-support-syst.streamlit.app/)

---

## At a glance

I took seven messy tables from a fictional global carmaker (customers, vehicles, payments, service jobs, support contacts, connected-car trips and branches), audited and cleaned them in **SQL Server**, modelled them in **Python**, wrote the predictions back into SQL Server results tables, and built **Power BI** pages on top of those tables. The saved models also run in a Streamlit app: enter a customer, payment or service job and get a score and a suggested action.

| Tool | What it did in this project |
|---|---|
| **SQL** (SQL Server, T-SQL) | Audit, cleaning, feature engineering, relationships, analytical base tables, results tables |
| **Python** (pandas, scikit-learn, matplotlib, joblib) | EDA, encoding, splitting, scaling, training, prediction, evaluation, business analysis, saving models |
| **Power BI** | Business analysis, findings, recommendations, impact and conclusion pages, read from the `bi_*` results tables in SQL Server |
| Streamlit | Live prediction app on top of the saved models |

**Predictions go back from Python into SQL Server results tables, and Power BI reads those.**

```
SQL Server (raw -> clean -> features -> ABT)  ->  Python (models)  ->  SQL Server (bi_* results tables)  ->  Power BI
```

| Business objective | ML task | Model that led | Result on unseen test data | Verdict |
|---|---|---|---|---|
| Reduce customer churn | Classification | Logistic Regression | Contacting the riskiest **20%** of customers reaches **36%** of the customers who leave (1.8x random) | USEFUL |
| Detect fraudulent transactions | Classification | Logistic Regression | Reviewing the riskiest **5%** of payments catches **96%** of fraud value | STRONG |
| Segment customers and understand behaviour | Clustering | K-Means (k = 3) | One segment is **31.5%** of customers and **58.4%** of next-year spend, with the lowest churn | USEFUL (as tiers) |
| Forecast customer lifetime value | Regression | Ridge Regression | Beats the median baseline with confidence, R² of 0.5 or more | STRONG |
| Predict service and repair cost | Regression | Gradient Boosting | Whole-workshop cost forecast within **3.6%** of actual | STRONG |

**Scorecard:** all 5 objectives passed their pre-set success bars: 3 STRONG and 2 USEFUL (section 19).

**What it is worth:** the impact is modelled in three scenarios (conservative, expected, optimistic) with every assumption stated (section 24). One evidence-backed example: at a 25% save rate, the churn campaign retains about **$54,000** of next-year spend for **$4,800** of contact cost.

The data is synthetic and was built with **82 planted quality problems**. Churn probabilities are inflated by class weights, so I use them for ranking only, never as literal percentages. Segments are tiers on a continuum, not sharply separated groups. I report these limits as they are.

---
## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Overview](#2-dataset-overview)
3. [Data Dictionary](#3-data-dictionary)
4. [Data Quality Audit](#4-data-quality-audit)
5. [Data Cleaning and Error Checking](#5-data-cleaning-and-error-checking)
6. [Feature Engineering](#6-feature-engineering)
7. [Create Relationship Table](#7-create-relationship-table)
8. [Exploratory Data Analysis](#8-exploratory-data-analysis)
9. [Analytical Base Table (SQL build + Python load)](#9-analytical-base-table-sql-build--python-load)
10. [Encode Categorical Columns](#10-encode-categorical-columns)
11. [Separate X and Y](#11-separate-x-and-y)
12. [Train Test Split](#12-train-test-split)
13. [Feature Scaling of Numerical Columns](#13-feature-scaling-of-numerical-columns)
14. [Train Model](#14-train-model)
15. [Predict X_test](#15-predict-x_test)
16. [Compare X and Y](#16-compare-x-and-y)
17. [Evaluate Model](#17-evaluate-model)
18. [Business Analysis](#18-business-analysis)
19. [Machine Learning Objectives](#19-machine-learning-objectives)
20. [Save Model, Scaler and Features](#20-save-model-scaler-and-features)
21. [Download Model, Scaler and Features](#21-download-model-scaler-and-features)
22. [Key Findings and Insights](#22-key-findings-and-insights)
23. [Recommendations](#23-recommendations)
24. [Impact (Power BI)](#24-impact-power-bi)
25. [Conclusion (Power BI)](#25-conclusion-power-bi)

---
## 1. Project Overview

Vantage Motors Global is a fictional multinational carmaker with customers, vehicles, payments, workshops, support desks and connected-car trips across eight countries. Its data sits in seven untidy tables, and five business goals depend on it:

| # | Business objective | ML task | Target | One row is a | Owner |
|---|---|---|---|---|---|
| 1 | Reduce customer churn | Classification | `churned` (30.7% of customers) | customer | Retention and CRM |
| 2 | Detect fraudulent transactions | Classification | `is_fraud` (2.4% of transactions) | transaction | Fraud operations |
| 3 | Segment customers and understand behaviour | Clustering (K-Means) | none | customer | Marketing and Sales |
| 4 | Forecast customer lifetime value | Regression | `next_12m_spend_usd` (average $4,529) | customer | Marketing and Finance |
| 5 | Predict service and repair cost | Regression | `total_cost_usd` (average $686) | service job | Service operations |

**How I built it, in order:** audit, cleaning, feature engineering and relationships in T-SQL; EDA in Python; one analytical base table per objective built in SQL and loaded into Python; encoding, X/y split, a leakage-safe train/test split, scaling; five candidate models per objective; prediction, comparison, evaluation; business analysis; saving model, scaler and features with joblib; a Streamlit app; findings, recommendations, impact and conclusion written back to SQL Server for Power BI.

**Rules I held to throughout**
- **No peeking.** Everything is fitted on training rows only. The test set is touched once, at prediction.
- **No leakage.** Columns that contain or reveal the answer are left out on purpose (section 9).
- **Honest verdicts.** A model is only called useful if its 95% bootstrap range beats a no-skill benchmark. The rules were fixed before looking at results (section 17).

### Repository structure

```
├── README.md
├── sql/                 01 audit · 02 cleaning · 03 features · 04 relationships · 06 analytical base tables
├── python/              05 EDA · 06 load ABT · 07 encode · 08 X/y · 09 split · 10 scale
│                        11 train (+ 11b training charts) · 12 predict · 13 compare · 14 evaluate (one set per objective)
│                        15 business analysis · 16 scorecard · 17 save · 18 package · 19 findings
│                        20 recommendations · 21 impact · 22 conclusion
├── streamlit/           app.py · requirements.txt
├── models/              26 joblib files (model, scaler, features, prep, meta per objective + encoding recipe)
├── powerbi/             Power BI report (reads the bi_* tables in SQL Server)
├── vantage_motors_data/ the seven raw tables
└── images/              every chart shown in this README
```

---

## 2. Dataset Overview

Seven tables in a star schema, **147,324 raw rows**. After cleaning, every fact row finds its parent.

| Table | Role | Rows (raw) | Rows (clean) | What it holds |
|---|---|---|---|---|
| `dim_branches` | Dimension | 60 | 60 | Branches: country, region, type, opening date |
| `dim_customers` | Dimension | 6,126 | 6,000 | Customer profile, loyalty tier, churn flag, next-year spend |
| `dim_vehicles` | Dimension | 9,189 | cleaned | Vehicles: model, body and fuel type, price, odometer, warranty |
| `fact_transactions` | Fact | 61,500 | 59,387 | Payments: amount, method, channel, countries, device |
| `fact_service_records` | Fact | 14,294 | 13,836 | Workshop jobs: service type, labour hours, warranty claim, cost |
| `fact_support_interactions` | Fact | 15,315 | cleaned | Support contacts: channel, issue, CSAT, sentiment |
| `fact_trips` | Fact | 40,840 | cleaned | Connected-car trips: distance, speed, braking, idling |

**After feature engineering, three analytical base tables**

| Analytical base table | Rows | Target balance |
|---|---|---|
| `abt_customer` | 6,000 customers | 30.7% churned; average next-12-month spend $4,529 |
| `abt_fraud` | 58,656 transactions | 2.4% fraud |
| `abt_service` | 13,836 service jobs | average cost $686 |

Observation cutoff for all features: **31 December 2025**.

### Code: row counts in SQL Server (T-SQL)

```sql
SELECT 'dim_branches' AS table_name, COUNT(*) AS n_rows FROM dbo.dim_branches
UNION ALL SELECT 'dim_customers',             COUNT(*) FROM dbo.dim_customers
UNION ALL SELECT 'dim_vehicles',              COUNT(*) FROM dbo.dim_vehicles
UNION ALL SELECT 'fact_transactions',         COUNT(*) FROM dbo.fact_transactions
UNION ALL SELECT 'fact_service_records',      COUNT(*) FROM dbo.fact_service_records
UNION ALL SELECT 'fact_support_interactions', COUNT(*) FROM dbo.fact_support_interactions
UNION ALL SELECT 'fact_trips',                COUNT(*) FROM dbo.fact_trips;
```

### Code: load the seven raw tables (Python)

```python
import pandas as pd

folder = "vantage_motors_data"
tables = ["dim_branches", "dim_customers", "dim_vehicles", "fact_transactions",
          "fact_service_records", "fact_support_interactions", "fact_trips"]

raw = {t: pd.read_csv(f"{folder}/{t}.csv", dtype=str) for t in tables}

for t, df in raw.items():
    print(t, df.shape)
    display(df.head(3))
```

---

## 3. Data Dictionary

Key columns by table. Columns marked **target** are what the models predict. Columns marked **leakage** hold information from after the outcome, so they are never used as features.

| Table | Key columns | Notes |
|---|---|---|
| `dim_branches` | `branch_id`, `country`, `region`, `branch_type`, `opened_date` | Where customers buy and vehicles are serviced |
| `dim_customers` | `customer_id`, `gender`, `birth_date`, `country`, `city`, `customer_type`, `signup_date`, `acquisition_channel`, `loyalty_tier`, `home_branch_id`, `email_opt_in`, `annual_income_usd` | Customer profile |
| | `churned` | **target** (churn) |
| | `next_12m_spend_usd` | **target** (lifetime value) |
| | `churn_date`, `last_activity_date` | **leakage**, excluded |
| `dim_vehicles` | `vehicle_id`, `customer_id`, `model_name`, `body_type`, `fuel_type`, `purchase_date`, `list_price_usd`, `odometer_km`, `odometer_unit`, `warranty_months`, `branch_id` | Odometer arrives in km or miles |
| `fact_transactions` | `txn_id`, `customer_id`, `vehicle_id`, `txn_date`, `txn_hour`, `txn_type`, `amount_usd`, `payment_method`, `channel`, `billing_country`, `txn_country`, `is_international`, `device_type` | Payment details |
| | `is_fraud` | **target** (fraud) |
| | `fraud_review_note`, `chargeback_flag` | **leakage**, excluded |
| `fact_service_records` | `service_id`, `vehicle_id`, `branch_id`, `service_date`, `service_type`, `odometer_at_service_km`, `vehicle_age_months`, `fuel_type`, `labor_hours`, `warranty_claim`, `customer_satisfaction` | Workshop jobs |
| | `total_cost_usd` | **target** (service cost) |
| | `parts_cost_usd`, `labor_cost_usd` | **leakage** (they add up to the target), excluded |
| `fact_support_interactions` | `interaction_id`, `customer_id`, `interaction_date`, `contact_channel`, `issue_category`, `resolution_hours`, `csat_score`, `escalated`, `sentiment_score` | Support history |
| `fact_trips` | `trip_id`, `vehicle_id`, `trip_date`, `distance_km`, `duration_min`, `avg_speed_kmh`, `harsh_braking_events`, `idle_minutes`, `night_trip` | Driving behaviour |

The full column-by-column dictionary of the analytical base tables is stored in SQL Server as `abt_data_dictionary`, and the objective-to-table mapping as `ml_model_map`.

---

## 4. Data Quality Audit

Script: `sql/01_data_quality_audit.sql`. It is read-only on the raw tables and writes every finding into `dbo.audit_results`.

**What it checks, on all seven tables**
- Exact duplicate rows and duplicate IDs
- Missing or blank values in every column
- Dates in mixed formats and unreadable junk dates
- Money stored as text (`$`, `USD`, commas)
- Orphan keys (a child row pointing to a parent that does not exist)
- Impossible values (hour 25, speed 400 km/h, zero or negative amounts) and sentinel dates (1900-01-01, 2099-12-31)
- Odometer in miles mixed with km
- Spelling, case and spacing variants in category columns
- Churn flag and churn date that contradict each other

**What it found: 82 issues**

| Issue type | Count |
|---|---|
| Columns with missing values | 22 |
| Category columns with spelling or case variants | 16 |
| Date columns in mixed formats | 10 |
| Impossible-value checks that fired | 8 |
| Orphan foreign keys | 7 |
| Money columns stored as text | 6 |
| Tables with conflicting duplicate IDs | 6 |
| Tables with exact duplicate rows | 6 |
| Odometer in miles | 1 |

### Code: data quality audit (T-SQL)

```sql
/* =====================================================================
   VANTAGE MOTORS GLOBAL - STEP 1: DATA QUALITY AUDIT (SQL Server)
   READ-ONLY on the raw tables. Writes findings to dbo.audit_results only.
   Run in SSMS in one query window. Change the database name below if needed.
   ===================================================================== */
USE [Vantage_Automotive];
GO
SET NOCOUNT ON;

DROP TABLE IF EXISTS dbo.audit_results;
CREATE TABLE dbo.audit_results (
    table_name  VARCHAR(40),
    column_name VARCHAR(60),
    check_name  VARCHAR(40),
    issue_rows  INT,
    total_rows  INT,
    pct AS CAST(100.0 * issue_rows / NULLIF(total_rows, 0) AS DECIMAL(6,2)));

DECLARE @tables TABLE (t SYSNAME, id_col SYSNAME);
INSERT @tables VALUES ('dim_branches','branch_id'),('dim_customers','customer_id'),('dim_vehicles','vehicle_id'),
  ('fact_transactions','txn_id'),('fact_service_records','service_id'),('fact_support_interactions','interaction_id'),('fact_trips','trip_id');

DECLARE @t SYSNAME, @c SYSNAME, @sql NVARCHAR(MAX);

/* ---- A. EXACT DUPLICATE ROWS and DUPLICATE IDs ---- */
DECLARE cur CURSOR FAST_FORWARD FOR SELECT t, id_col FROM @tables;
OPEN cur; FETCH NEXT FROM cur INTO @t, @c;
WHILE @@FETCH_STATUS = 0
BEGIN
    SET @sql = N'INSERT INTO dbo.audit_results (table_name,column_name,check_name,issue_rows,total_rows)
        SELECT ''' + @t + N''', ''(all columns)'', ''exact_duplicate_rows'', COUNT(*) - (SELECT COUNT(*) FROM (SELECT DISTINCT * FROM dbo.' + QUOTENAME(@t) + N') d), COUNT(*) FROM dbo.' + QUOTENAME(@t) + N';
        INSERT INTO dbo.audit_results (table_name,column_name,check_name,issue_rows,total_rows)
        SELECT ''' + @t + N''', ''' + @c + N''', ''duplicate_id_rows'', COUNT(*) - COUNT(DISTINCT ' + QUOTENAME(@c) + N'), COUNT(*) FROM dbo.' + QUOTENAME(@t) + N';';
    EXEC sp_executesql @sql;
    FETCH NEXT FROM cur INTO @t, @c;
END
CLOSE cur; DEALLOCATE cur;

/* ---- B. MISSING / BLANK VALUES in every column of every table ---- */
DECLARE cur CURSOR FAST_FORWARD FOR
    SELECT TABLE_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME IN (SELECT t FROM @tables) ORDER BY TABLE_NAME, ORDINAL_POSITION;
OPEN cur; FETCH NEXT FROM cur INTO @t, @c;
WHILE @@FETCH_STATUS = 0
BEGIN
    SET @sql = N'INSERT INTO dbo.audit_results (table_name,column_name,check_name,issue_rows,total_rows)
        SELECT ''' + @t + N''', ''' + @c + N''', ''missing_or_blank'',
        SUM(CASE WHEN ' + QUOTENAME(@c) + N' IS NULL OR LTRIM(RTRIM(CAST(' + QUOTENAME(@c) + N' AS NVARCHAR(4000)))) = '''' THEN 1 ELSE 0 END), COUNT(*)
        FROM dbo.' + QUOTENAME(@t) + N';';
    EXEC sp_executesql @sql;
    FETCH NEXT FROM cur INTO @t, @c;
END
CLOSE cur; DEALLOCATE cur;

/* ---- C. DATE COLUMNS: mixed formats and unreadable junk ---- */
DECLARE @dates TABLE (t SYSNAME, c SYSNAME);
INSERT @dates VALUES ('dim_branches','opened_date'),('dim_customers','birth_date'),('dim_customers','signup_date'),
  ('dim_customers','churn_date'),('dim_customers','last_activity_date'),('dim_vehicles','purchase_date'),
  ('fact_transactions','txn_date'),('fact_service_records','service_date'),('fact_support_interactions','interaction_date'),('fact_trips','trip_date');
DECLARE cur CURSOR FAST_FORWARD FOR SELECT t, c FROM @dates;
OPEN cur; FETCH NEXT FROM cur INTO @t, @c;
WHILE @@FETCH_STATUS = 0
BEGIN
    SET @sql = N'INSERT INTO dbo.audit_results (table_name,column_name,check_name,issue_rows,total_rows)
        SELECT ''{t}'', ''{c}'', ''mixed_date_format'',
               SUM(CASE WHEN q.v <> '''' AND q.v NOT LIKE ''[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]%'' THEN 1 ELSE 0 END), COUNT(*)
        FROM dbo.{t} CROSS APPLY (SELECT LTRIM(RTRIM(COALESCE(CAST({c} AS NVARCHAR(100)), ''''))) AS v) q;
        INSERT INTO dbo.audit_results (table_name,column_name,check_name,issue_rows,total_rows)
        SELECT ''{t}'', ''{c}'', ''unreadable_date_junk'',
               SUM(CASE WHEN q.v <> '''' AND TRY_CONVERT(DATE,q.v,23) IS NULL AND TRY_CONVERT(DATE,q.v,103) IS NULL AND TRY_CONVERT(DATE,q.v,110) IS NULL
                         AND TRY_CONVERT(DATE,q.v,106) IS NULL AND TRY_CONVERT(DATE,q.v) IS NULL THEN 1 ELSE 0 END), COUNT(*)
        FROM dbo.{t} CROSS APPLY (SELECT LTRIM(RTRIM(COALESCE(CAST({c} AS NVARCHAR(100)), ''''))) AS v) q;';
    SET @sql = REPLACE(REPLACE(@sql, '{t}', @t), '{c}', @c);
    EXEC sp_executesql @sql;
    FETCH NEXT FROM cur INTO @t, @c;
END
CLOSE cur; DEALLOCATE cur;

/* ---- D. MONEY STORED AS TEXT ($, USD, commas) ---- */
DECLARE @money TABLE (t SYSNAME, c SYSNAME);
INSERT @money VALUES ('dim_customers','annual_income_usd'),('dim_vehicles','list_price_usd'),('fact_transactions','amount_usd'),
  ('fact_service_records','parts_cost_usd'),('fact_service_records','labor_cost_usd'),('fact_service_records','total_cost_usd');
DECLARE cur CURSOR FAST_FORWARD FOR SELECT t, c FROM @money;
OPEN cur; FETCH NEXT FROM cur INTO @t, @c;
WHILE @@FETCH_STATUS = 0
BEGIN
    SET @sql = N'INSERT INTO dbo.audit_results (table_name,column_name,check_name,issue_rows,total_rows)
        SELECT ''{t}'', ''{c}'', ''currency_text'',
               SUM(CASE WHEN CAST({c} AS NVARCHAR(100)) LIKE ''%[$,]%'' OR UPPER(CAST({c} AS NVARCHAR(100))) LIKE ''%USD%'' THEN 1 ELSE 0 END), COUNT(*)
        FROM dbo.{t};';
    SET @sql = REPLACE(REPLACE(@sql, '{t}', @t), '{c}', @c);
    EXEC sp_executesql @sql;
    FETCH NEXT FROM cur INTO @t, @c;
END
CLOSE cur; DEALLOCATE cur;
GO

/* ---- E. ORPHAN FOREIGN KEYS (child key not found in parent table) ---- */
INSERT INTO dbo.audit_results (table_name,column_name,check_name,issue_rows,total_rows)
SELECT 'dim_customers','home_branch_id','orphan_fk', SUM(CASE WHEN LTRIM(RTRIM(COALESCE(f.home_branch_id,''))) <> '' AND p.branch_id IS NULL THEN 1 ELSE 0 END), COUNT(*)
FROM dbo.dim_customers f LEFT JOIN (SELECT DISTINCT LTRIM(RTRIM(branch_id)) AS branch_id FROM dbo.dim_branches) p ON p.branch_id = LTRIM(RTRIM(f.home_branch_id))
UNION ALL
SELECT 'dim_vehicles','customer_id','orphan_fk', SUM(CASE WHEN LTRIM(RTRIM(COALESCE(f.customer_id,''))) <> '' AND p.customer_id IS NULL THEN 1 ELSE 0 END), COUNT(*)
FROM dbo.dim_vehicles f LEFT JOIN (SELECT DISTINCT LTRIM(RTRIM(customer_id)) AS customer_id FROM dbo.dim_customers) p ON p.customer_id = LTRIM(RTRIM(f.customer_id))
UNION ALL
SELECT 'dim_vehicles','branch_id','orphan_fk', SUM(CASE WHEN LTRIM(RTRIM(COALESCE(f.branch_id,''))) <> '' AND p.branch_id IS NULL THEN 1 ELSE 0 END), COUNT(*)
FROM dbo.dim_vehicles f LEFT JOIN (SELECT DISTINCT LTRIM(RTRIM(branch_id)) AS branch_id FROM dbo.dim_branches) p ON p.branch_id = LTRIM(RTRIM(f.branch_id))
UNION ALL
SELECT 'fact_transactions','customer_id','orphan_fk', SUM(CASE WHEN LTRIM(RTRIM(COALESCE(f.customer_id,''))) <> '' AND p.customer_id IS NULL THEN 1 ELSE 0 END), COUNT(*)
FROM dbo.fact_transactions f LEFT JOIN (SELECT DISTINCT LTRIM(RTRIM(customer_id)) AS customer_id FROM dbo.dim_customers) p ON p.customer_id = LTRIM(RTRIM(f.customer_id))
UNION ALL
SELECT 'fact_transactions','vehicle_id','orphan_fk', SUM(CASE WHEN LTRIM(RTRIM(COALESCE(f.vehicle_id,''))) <> '' AND p.vehicle_id IS NULL THEN 1 ELSE 0 END), COUNT(*)
FROM dbo.fact_transactions f LEFT JOIN (SELECT DISTINCT LTRIM(RTRIM(vehicle_id)) AS vehicle_id FROM dbo.dim_vehicles) p ON p.vehicle_id = LTRIM(RTRIM(f.vehicle_id))
UNION ALL
SELECT 'fact_service_records','vehicle_id','orphan_fk', SUM(CASE WHEN LTRIM(RTRIM(COALESCE(f.vehicle_id,''))) <> '' AND p.vehicle_id IS NULL THEN 1 ELSE 0 END), COUNT(*)
FROM dbo.fact_service_records f LEFT JOIN (SELECT DISTINCT LTRIM(RTRIM(vehicle_id)) AS vehicle_id FROM dbo.dim_vehicles) p ON p.vehicle_id = LTRIM(RTRIM(f.vehicle_id))
UNION ALL
SELECT 'fact_service_records','branch_id','orphan_fk', SUM(CASE WHEN LTRIM(RTRIM(COALESCE(f.branch_id,''))) <> '' AND p.branch_id IS NULL THEN 1 ELSE 0 END), COUNT(*)
FROM dbo.fact_service_records f LEFT JOIN (SELECT DISTINCT LTRIM(RTRIM(branch_id)) AS branch_id FROM dbo.dim_branches) p ON p.branch_id = LTRIM(RTRIM(f.branch_id))
UNION ALL
SELECT 'fact_support_interactions','customer_id','orphan_fk', SUM(CASE WHEN LTRIM(RTRIM(COALESCE(f.customer_id,''))) <> '' AND p.customer_id IS NULL THEN 1 ELSE 0 END), COUNT(*)
FROM dbo.fact_support_interactions f LEFT JOIN (SELECT DISTINCT LTRIM(RTRIM(customer_id)) AS customer_id FROM dbo.dim_customers) p ON p.customer_id = LTRIM(RTRIM(f.customer_id))
UNION ALL
SELECT 'fact_trips','vehicle_id','orphan_fk', SUM(CASE WHEN LTRIM(RTRIM(COALESCE(f.vehicle_id,''))) <> '' AND p.vehicle_id IS NULL THEN 1 ELSE 0 END), COUNT(*)
FROM dbo.fact_trips f LEFT JOIN (SELECT DISTINCT LTRIM(RTRIM(vehicle_id)) AS vehicle_id FROM dbo.dim_vehicles) p ON p.vehicle_id = LTRIM(RTRIM(f.vehicle_id));
GO

/* ---- F. IMPOSSIBLE VALUES, MIXED UNITS and LABEL CONSISTENCY ---- */
INSERT INTO dbo.audit_results (table_name,column_name,check_name,issue_rows,total_rows)
SELECT 'dim_customers','birth_date','sentinel_date',
       SUM(CASE WHEN LEFT(LTRIM(RTRIM(CAST(birth_date AS NVARCHAR(100)))),10) IN ('1900-01-01','1970-01-01','2099-12-31') THEN 1 ELSE 0 END), COUNT(*) FROM dbo.dim_customers
UNION ALL
SELECT 'dim_customers','churn_date','churned1_but_no_churn_date',
       SUM(CASE WHEN TRY_CONVERT(FLOAT,churned) = 1 AND LTRIM(RTRIM(COALESCE(CAST(churn_date AS NVARCHAR(100)),''))) = '' THEN 1 ELSE 0 END), COUNT(*) FROM dbo.dim_customers
UNION ALL
SELECT 'dim_customers','churn_date','churned0_but_has_churn_date',
       SUM(CASE WHEN TRY_CONVERT(FLOAT,churned) = 0 AND LTRIM(RTRIM(COALESCE(CAST(churn_date AS NVARCHAR(100)),''))) <> '' THEN 1 ELSE 0 END), COUNT(*) FROM dbo.dim_customers
UNION ALL
SELECT 'dim_vehicles','odometer_unit','odometer_in_miles',
       SUM(CASE WHEN LOWER(LTRIM(RTRIM(odometer_unit))) = 'mi' THEN 1 ELSE 0 END), COUNT(*) FROM dbo.dim_vehicles
UNION ALL
SELECT 'dim_vehicles','odometer_km','impossible_value',
       SUM(CASE WHEN TRY_CONVERT(FLOAT,odometer_km) <= 0 OR TRY_CONVERT(FLOAT,odometer_km) >= 1000000 THEN 1 ELSE 0 END), COUNT(*) FROM dbo.dim_vehicles
UNION ALL
SELECT 'fact_transactions','txn_hour','impossible_value',
       SUM(CASE WHEN TRY_CONVERT(FLOAT,txn_hour) < 0 OR TRY_CONVERT(FLOAT,txn_hour) > 23 THEN 1 ELSE 0 END), COUNT(*) FROM dbo.fact_transactions
UNION ALL
SELECT 'fact_transactions','amount_usd','impossible_value',
       SUM(CASE WHEN TRY_CONVERT(DECIMAL(18,2), NULLIF(REPLACE(REPLACE(REPLACE(REPLACE(UPPER(CAST(amount_usd AS NVARCHAR(100))),'USD',''),'$',''),',',''),' ',''),'')) <= 0
                 OR TRY_CONVERT(DECIMAL(18,2), NULLIF(REPLACE(REPLACE(REPLACE(REPLACE(UPPER(CAST(amount_usd AS NVARCHAR(100))),'USD',''),'$',''),',',''),' ',''),'')) >= 10000000 THEN 1 ELSE 0 END), COUNT(*) FROM dbo.fact_transactions
UNION ALL
SELECT 'fact_service_records','labor_hours','impossible_value',
       SUM(CASE WHEN TRY_CONVERT(FLOAT,labor_hours) <= 0 OR TRY_CONVERT(FLOAT,labor_hours) > 100 THEN 1 ELSE 0 END), COUNT(*) FROM dbo.fact_service_records
UNION ALL
SELECT 'fact_support_interactions','resolution_hours','impossible_value',
       SUM(CASE WHEN TRY_CONVERT(FLOAT,resolution_hours) <= 0 OR TRY_CONVERT(FLOAT,resolution_hours) > 1000 THEN 1 ELSE 0 END), COUNT(*) FROM dbo.fact_support_interactions
UNION ALL
SELECT 'fact_trips','avg_speed_kmh','impossible_value',
       SUM(CASE WHEN TRY_CONVERT(FLOAT,avg_speed_kmh) <= 0 OR TRY_CONVERT(FLOAT,avg_speed_kmh) > 250 THEN 1 ELSE 0 END), COUNT(*) FROM dbo.fact_trips
UNION ALL
SELECT 'fact_trips','distance_km','impossible_value',
       SUM(CASE WHEN TRY_CONVERT(FLOAT,distance_km) <= 0 OR TRY_CONVERT(FLOAT,distance_km) > 2000 THEN 1 ELSE 0 END), COUNT(*) FROM dbo.fact_trips;
GO

/* =================== REPORTS (these are what you read) =================== */
-- R1. Row counts
SELECT 'dim_branches' AS table_name, COUNT(*) AS row_count FROM dbo.dim_branches
UNION ALL SELECT 'dim_customers', COUNT(*) FROM dbo.dim_customers
UNION ALL SELECT 'dim_vehicles', COUNT(*) FROM dbo.dim_vehicles
UNION ALL SELECT 'fact_transactions', COUNT(*) FROM dbo.fact_transactions
UNION ALL SELECT 'fact_service_records', COUNT(*) FROM dbo.fact_service_records
UNION ALL SELECT 'fact_support_interactions', COUNT(*) FROM dbo.fact_support_interactions
UNION ALL SELECT 'fact_trips', COUNT(*) FROM dbo.fact_trips;

-- R2. Every issue found, worst first
SELECT table_name, column_name, check_name, issue_rows, total_rows, pct
FROM dbo.audit_results WHERE issue_rows > 0 ORDER BY pct DESC, table_name, column_name;

-- R3. Issues per table
SELECT table_name, COUNT(*) AS issue_types, SUM(issue_rows) AS issue_rows
FROM dbo.audit_results WHERE issue_rows > 0 GROUP BY table_name ORDER BY issue_rows DESC;

-- R4. Category spelling variants (case-sensitive so ' GOLD ' and 'Gold' show separately)
SELECT 'dim_customers.country' AS col, country COLLATE Latin1_General_BIN AS value, COUNT(*) AS n FROM dbo.dim_customers GROUP BY country COLLATE Latin1_General_BIN
UNION ALL SELECT 'dim_customers.gender', gender COLLATE Latin1_General_BIN, COUNT(*) FROM dbo.dim_customers GROUP BY gender COLLATE Latin1_General_BIN
UNION ALL SELECT 'dim_customers.customer_type', customer_type COLLATE Latin1_General_BIN, COUNT(*) FROM dbo.dim_customers GROUP BY customer_type COLLATE Latin1_General_BIN
UNION ALL SELECT 'dim_customers.loyalty_tier', loyalty_tier COLLATE Latin1_General_BIN, COUNT(*) FROM dbo.dim_customers GROUP BY loyalty_tier COLLATE Latin1_General_BIN
UNION ALL SELECT 'dim_vehicles.fuel_type', fuel_type COLLATE Latin1_General_BIN, COUNT(*) FROM dbo.dim_vehicles GROUP BY fuel_type COLLATE Latin1_General_BIN
UNION ALL SELECT 'dim_vehicles.model_name', model_name COLLATE Latin1_General_BIN, COUNT(*) FROM dbo.dim_vehicles GROUP BY model_name COLLATE Latin1_General_BIN
UNION ALL SELECT 'fact_transactions.payment_method', payment_method COLLATE Latin1_General_BIN, COUNT(*) FROM dbo.fact_transactions GROUP BY payment_method COLLATE Latin1_General_BIN
UNION ALL SELECT 'fact_transactions.txn_type', txn_type COLLATE Latin1_General_BIN, COUNT(*) FROM dbo.fact_transactions GROUP BY txn_type COLLATE Latin1_General_BIN
UNION ALL SELECT 'fact_service_records.service_type', service_type COLLATE Latin1_General_BIN, COUNT(*) FROM dbo.fact_service_records GROUP BY service_type COLLATE Latin1_General_BIN
UNION ALL SELECT 'fact_support_interactions.contact_channel', contact_channel COLLATE Latin1_General_BIN, COUNT(*) FROM dbo.fact_support_interactions GROUP BY contact_channel COLLATE Latin1_General_BIN
ORDER BY col, n DESC;

-- R5. Target balance (churn and fraud). Duplicates are not removed yet, so rates are approximate.
SELECT 'churned' AS target, TRY_CONVERT(INT, TRY_CONVERT(FLOAT, churned)) AS class, COUNT(*) AS n FROM dbo.dim_customers GROUP BY TRY_CONVERT(INT, TRY_CONVERT(FLOAT, churned))
UNION ALL SELECT 'is_fraud', TRY_CONVERT(INT, TRY_CONVERT(FLOAT, is_fraud)), COUNT(*) FROM dbo.fact_transactions GROUP BY TRY_CONVERT(INT, TRY_CONVERT(FLOAT, is_fraud))
ORDER BY target, class;
GO
```

---

## 5. Data Cleaning and Error Checking

Script: `sql/02_data_cleaning.sql`. It never edits the raw tables. It builds seven `clean_*` tables plus two support tables:
- `quarantine_rows` keeps every removed row with the reason, so nothing disappears silently.
- `map_values` is the lookup that maps every spelling variant to one standard value.

**Decisions**

| Problem | Fix |
|---|---|
| Spelling and case variants (` GOLD `, `gold`) | Mapped to one standard label through `map_values` |
| Dates in five formats, junk like `N/A` | Parsed by one function; unreadable values become NULL |
| Money as text | `$`, `USD` and commas stripped, converted to decimal |
| Duplicate IDs with conflicting copies | Kept the copy that is valid (for example, a real gender or an existing owner) |
| Exact duplicates | Removed |
| Orphan keys | Row moved to `quarantine_rows` |
| Impossible values and sentinel dates | Set to NULL, not guessed |
| Odometer in miles | Converted to km |
| Missing values | Left as NULL on purpose and imputed later in Python on train only |

**Result:** 6,126 raw customers became 6,000 clean customers, 61,500 transactions became 59,387, and 14,294 service jobs became 13,836. The script finishes with validation checks: no duplicate IDs, no category outside the standard list, and sane ranges.

### Code: data cleaning and error checking (T-SQL)

```sql
/* =====================================================================
   VANTAGE MOTORS GLOBAL - STEP 2: DATA CLEANING (raw -> clean_*)
   Run in SSMS, top to bottom, in ONE query window (temp tables are used).
   Assumes: raw tables imported as dbo.dim_branches, dbo.dim_customers,
   dbo.dim_vehicles, dbo.fact_transactions, dbo.fact_service_records,
   dbo.fact_support_interactions, dbo.fact_trips (all columns as text is fine).
   Database: Vantage_Automotive.
   Output: dbo.clean_<table> x7, dbo.quarantine_rows, dbo.map_values.
   Missing values are left as NULL on purpose (impute in Python, train only).
   ===================================================================== */
USE [Vantage_Automotive];
GO

/* ---------- 1. HELPER FUNCTIONS ---------- */
IF OBJECT_ID('dbo.fn_parse_date') IS NOT NULL DROP FUNCTION dbo.fn_parse_date;
IF OBJECT_ID('dbo.fn_parse_money') IS NOT NULL DROP FUNCTION dbo.fn_parse_money;
IF OBJECT_ID('dbo.fn_num') IS NOT NULL DROP FUNCTION dbo.fn_num;
IF OBJECT_ID('dbo.fn_map') IS NOT NULL DROP FUNCTION dbo.fn_map;
GO
CREATE FUNCTION dbo.fn_parse_date (@s NVARCHAR(100))
RETURNS DATE
AS
BEGIN
    SET @s = NULLIF(LTRIM(RTRIM(@s)), '');
    IF @s IS NULL RETURN NULL;
    RETURN CASE
        WHEN @s LIKE '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]%' THEN TRY_CONVERT(DATE, LEFT(@s, 10), 23)
        WHEN @s LIKE '[0-9][0-9][0-9][0-9]/[0-9][0-9]/[0-9][0-9]'  THEN TRY_CONVERT(DATE, REPLACE(@s, '/', '-'), 23)
        WHEN @s LIKE '[0-9][0-9]/[0-9][0-9]/[0-9][0-9][0-9][0-9]'  THEN TRY_CONVERT(DATE, @s, 103)  -- dd/mm/yyyy
        WHEN @s LIKE '[0-9][0-9]-[0-9][0-9]-[0-9][0-9][0-9][0-9]'  THEN TRY_CONVERT(DATE, @s, 110)  -- mm-dd-yyyy
        WHEN @s LIKE '[0-9][0-9] [A-Za-z][A-Za-z][A-Za-z] [0-9][0-9][0-9][0-9]' THEN TRY_CONVERT(DATE, @s, 106) -- 05 Mar 2024
        WHEN @s LIKE '[A-Za-z]%' THEN TRY_CONVERT(DATE, @s)   -- March 05, 2024 (junk like N/A, TBD -> NULL)
        ELSE NULL END;
END
GO
CREATE FUNCTION dbo.fn_parse_money (@s NVARCHAR(100))
RETURNS DECIMAL(18,2)
AS
BEGIN
    DECLARE @t NVARCHAR(100) = NULLIF(LTRIM(RTRIM(@s)), '');
    IF @t IS NULL RETURN NULL;
    SET @t = REPLACE(REPLACE(REPLACE(REPLACE(UPPER(@t), 'USD', ''), '$', ''), ',', ''), ' ', '');
    RETURN TRY_CONVERT(DECIMAL(18,2), NULLIF(@t, ''));
END
GO
CREATE FUNCTION dbo.fn_num (@s NVARCHAR(100))
RETURNS FLOAT
AS
BEGIN
    RETURN TRY_CONVERT(FLOAT, NULLIF(LTRIM(RTRIM(@s)), ''));
END
GO

/* ---------- 2. CATEGORY MAPPING TABLE (variants -> standard value) ---------- */
DROP TABLE IF EXISTS dbo.map_values;
CREATE TABLE dbo.map_values (
    domain VARCHAR(30) NOT NULL, raw_value NVARCHAR(100) NOT NULL, clean_value NVARCHAR(100) NOT NULL,
    PRIMARY KEY (domain, raw_value));
-- raw_value is stored UPPER-case; fn_map upper-cases and trims the input first
INSERT INTO dbo.map_values (domain, raw_value, clean_value) VALUES
('country','USA','USA'),('country','UNITED STATES','USA'),('country','U.S.A','USA'),('country','US','USA'),
('country','GERMANY','Germany'),('country','DEUTSCHLAND','Germany'),('country','GER','Germany'),
('country','UK','UK'),('country','UNITED KINGDOM','UK'),('country','U.K.','UK'),('country','GREAT BRITAIN','UK'),
('country','JAPAN','Japan'),('country','JPN','Japan'),
('country','INDIA','India'),('country','IND','India'),('country','BHARAT','India'),
('country','NIGERIA','Nigeria'),('country','NGA','Nigeria'),('country','NIGERIA.','Nigeria'),
('country','BRAZIL','Brazil'),('country','BRASIL','Brazil'),('country','BRA','Brazil'),
('country','UAE','UAE'),('country','UNITED ARAB EMIRATES','UAE'),('country','U.A.E','UAE'),('country','U.A.E.','UAE'),
('gender','MALE','Male'),('gender','M','Male'),('gender','FEMALE','Female'),('gender','F','Female'),
('gender','OTHER','Other'),('gender','NON-BINARY','Other'),
('customer_type','INDIVIDUAL','Individual'),('customer_type','CORPORATE','Corporate'),('customer_type','CORP','Corporate'),
('customer_type','COMPANY','Corporate'),('customer_type','FLEET OPERATOR','Fleet Operator'),('customer_type','FLEET','Fleet Operator'),
('customer_type','FLEET OP.','Fleet Operator'),
('loyalty','BRONZE','Bronze'),('loyalty','SILVER','Silver'),('loyalty','GOLD','Gold'),('loyalty','PLATINUM','Platinum'),('loyalty','PLAT','Platinum'),
('fuel','PETROL','Petrol'),('fuel','GASOLINE','Petrol'),('fuel','GAS','Petrol'),('fuel','DIESEL','Diesel'),
('fuel','HYBRID','Hybrid'),('fuel','HEV','Hybrid'),('fuel','ELECTRIC','Electric'),('fuel','EV','Electric'),('fuel','BEV','Electric'),
('model','AURORA SEDAN','Aurora Sedan'),('model','VECTOR SUV','Vector SUV'),('model','VECTOR','Vector SUV'),
('model','NOVA EV','Nova EV'),('model','NOVA-EV','Nova EV'),('model','PULSE HATCH','Pulse Hatch'),
('model','TITAN PICKUP','Titan Pickup'),('model','VOLT CARGO VAN','Volt Cargo Van'),('model','APEX COUPE','Apex Coupe'),
('payment','CREDIT CARD','Credit Card'),('payment','CC','Credit Card'),('payment','DEBIT CARD','Debit Card'),
('payment','BANK TRANSFER','Bank Transfer'),('payment','EFT','Bank Transfer'),('payment','WIRE','Bank Transfer'),
('payment','MOBILE WALLET','Mobile Wallet'),('payment','WALLET','Mobile Wallet'),('payment','APPLEPAY/GPAY','Mobile Wallet'),
('payment','FINANCE PLAN','Finance Plan'),
('txn_type','VEHICLE PURCHASE','Vehicle Purchase'),('txn_type','PURCHASE','Vehicle Purchase'),
('txn_type','SERVICE PAYMENT','Service Payment'),('txn_type','SERVICE PMT','Service Payment'),
('txn_type','PARTS ORDER','Parts Order'),('txn_type','SUBSCRIPTION','Subscription'),('txn_type','RENTAL','Rental'),
('service_type','ROUTINE MAINTENANCE','Routine Maintenance'),('service_type','ROUTINE SERVICE','Routine Maintenance'),('service_type','MAINT.','Routine Maintenance'),
('service_type','BRAKE REPAIR','Brake Repair'),('service_type','ENGINE REPAIR','Engine Repair'),
('service_type','BATTERY/EV SYSTEM','Battery/EV System'),('service_type','EV BATTERY','Battery/EV System'),
('service_type','TRANSMISSION','Transmission'),('service_type','TYRES','Tyres'),
('service_type','BODY WORK','Body Work'),('service_type','BODYWORK','Body Work'),
('contact','PHONE','Phone'),('contact','CALL','Phone'),('contact','TELEPHONE','Phone'),('contact','EMAIL','Email'),
('contact','CHAT','Chat'),('contact','LIVE CHAT','Chat'),('contact','BRANCH VISIT','Branch Visit'),
('contact','SOCIAL MEDIA','Social Media'),('contact','TWITTER/X','Social Media'),('contact','FACEBOOK','Social Media'),
('issue','BILLING','Billing'),('issue','INVOICE QUERY','Billing'),('issue','SERVICE DELAY','Service Delay'),
('issue','VEHICLE DEFECT','Vehicle Defect'),('issue','DEFECT','Vehicle Defect'),('issue','WARRANTY','Warranty'),
('issue','DELIVERY','Delivery'),('issue','APP/CONNECTIVITY','App/Connectivity'),
('branch_type','SHOWROOM','Showroom'),('branch_type','SERVICE CENTRE','Service Centre'),
('branch_type','SHOWROOM+SERVICE','Showroom+Service'),('branch_type','FLEET HUB','Fleet Hub'),
('channel','ONLINE','Online'),('channel','DEALER POS','Dealer POS'),('channel','MOBILE APP','Mobile App'),('channel','CALL CENTRE','Call Centre');
GO
CREATE FUNCTION dbo.fn_map (@domain VARCHAR(30), @v NVARCHAR(100))
RETURNS NVARCHAR(100)
AS
BEGIN
    DECLARE @k NVARCHAR(100) = NULLIF(LTRIM(RTRIM(@v)), '');
    IF @k IS NULL RETURN NULL;
    DECLARE @c NVARCHAR(100);
    SELECT @c = clean_value FROM dbo.map_values WHERE domain = @domain AND raw_value = UPPER(@k);
    RETURN COALESCE(@c, @k);   -- unmapped values stay visible; section 6 lists them
END
GO

/* ---------- 3. QUARANTINE TABLE (rows removed from clean tables) ---------- */
DROP TABLE IF EXISTS dbo.quarantine_rows;
CREATE TABLE dbo.quarantine_rows (source_table VARCHAR(40), record_id NVARCHAR(50), reason VARCHAR(100));
GO

/* ---------- 4. DIMENSION TABLES ---------- */
-- 4.1 branches
DROP TABLE IF EXISTS dbo.clean_dim_branches;
SELECT LTRIM(RTRIM(branch_id))                AS branch_id,
       LTRIM(RTRIM(branch_name))              AS branch_name,
       dbo.fn_map('country', country)         AS country,
       LTRIM(RTRIM(region))                   AS region,
       dbo.fn_map('branch_type', branch_type) AS branch_type,
       dbo.fn_parse_date(opened_date)         AS opened_date
INTO dbo.clean_dim_branches
FROM (SELECT DISTINCT * FROM dbo.dim_branches) b;
GO

-- 4.2 customers: duplicate IDs -> keep the row with a valid gender (the conflicting copy has them swapped)
DROP TABLE IF EXISTS dbo.clean_dim_customers;
WITH r AS (
    SELECT c.*,
           ROW_NUMBER() OVER (PARTITION BY LTRIM(RTRIM(customer_id))
               ORDER BY CASE WHEN gender IS NULL OR UPPER(LTRIM(RTRIM(gender))) IN
                    ('MALE','M','FEMALE','F','OTHER','NON-BINARY') THEN 0 ELSE 1 END) AS rn
    FROM dbo.dim_customers c)
SELECT LTRIM(RTRIM(r.customer_id))                AS customer_id,
       LTRIM(RTRIM(r.customer_name))              AS customer_name,
       dbo.fn_map('gender', r.gender)             AS gender,
       CASE WHEN YEAR(x.bd) BETWEEN 1930 AND 2007 THEN x.bd END AS birth_date,   -- 1900/1970/2099 sentinels -> NULL
       dbo.fn_map('country', r.country)           AS country,
       LTRIM(RTRIM(r.city))                       AS city,
       dbo.fn_map('customer_type', r.customer_type) AS customer_type,
       dbo.fn_parse_date(r.signup_date)           AS signup_date,
       LTRIM(RTRIM(r.acquisition_channel))        AS acquisition_channel,
       dbo.fn_map('loyalty', r.loyalty_tier)      AS loyalty_tier,
       b.branch_id                                AS home_branch_id,            -- orphan branch -> NULL
       LTRIM(RTRIM(r.email_opt_in))               AS email_opt_in,
       CASE WHEN m.inc > 0 THEN m.inc END         AS annual_income_usd,
       CAST(dbo.fn_num(r.churned) AS INT)         AS churned,
       dbo.fn_parse_date(r.churn_date)            AS churn_date,
       dbo.fn_parse_date(r.last_activity_date)    AS last_activity_date,
       dbo.fn_parse_money(r.next_12m_spend_usd)   AS next_12m_spend_usd
INTO dbo.clean_dim_customers
FROM r
CROSS APPLY (SELECT dbo.fn_parse_date(r.birth_date) AS bd) x
CROSS APPLY (SELECT dbo.fn_parse_money(r.annual_income_usd) AS inc) m
LEFT JOIN dbo.clean_dim_branches b ON b.branch_id = LTRIM(RTRIM(r.home_branch_id))
WHERE r.rn = 1;
GO

-- 4.3 vehicles: keep the duplicate whose owner exists; miles -> km; sentinels -> NULL
DROP TABLE IF EXISTS dbo.clean_dim_vehicles;
WITH s AS (
    SELECT v.*, LTRIM(RTRIM(v.vehicle_id)) AS vid,
           CASE WHEN c.customer_id IS NULL THEN 1 ELSE 0 END AS bad_fk
    FROM dbo.dim_vehicles v
    LEFT JOIN dbo.clean_dim_customers c ON c.customer_id = LTRIM(RTRIM(v.customer_id))),
r AS (SELECT s.*, ROW_NUMBER() OVER (PARTITION BY vid ORDER BY bad_fk) AS rn FROM s)
SELECT r.vid                                       AS vehicle_id,
       c.customer_id                               AS customer_id,                -- orphan owner -> NULL
       dbo.fn_map('model', r.model_name)           AS model_name,
       LTRIM(RTRIM(r.body_type))                   AS body_type,
       dbo.fn_map('fuel', r.fuel_type)             AS fuel_type,
       dbo.fn_parse_date(r.purchase_date)          AS purchase_date,
       CASE WHEN p.price > 0 THEN p.price END      AS list_price_usd,
       CASE WHEN o.odo > 0 AND o.odo < 1000000
            THEN ROUND(CASE WHEN LOWER(LTRIM(RTRIM(r.odometer_unit))) = 'mi' THEN o.odo * 1.609344 ELSE o.odo END, 0) END AS odometer_km,
       CAST(dbo.fn_num(r.warranty_months) AS INT)  AS warranty_months,
       LTRIM(RTRIM(r.branch_id))                   AS branch_id
INTO dbo.clean_dim_vehicles
FROM r
CROSS APPLY (SELECT dbo.fn_num(r.odometer_km) AS odo) o
CROSS APPLY (SELECT dbo.fn_parse_money(r.list_price_usd) AS price) p
LEFT JOIN dbo.clean_dim_customers c ON c.customer_id = LTRIM(RTRIM(r.customer_id))
WHERE r.rn = 1;
GO

/* ---------- 5. FACT TABLES (dedupe -> quarantine orphans -> clean) ---------- */
-- 5.1 transactions
DROP TABLE IF EXISTS #t;
WITH s AS (
    SELECT f.*, LTRIM(RTRIM(f.txn_id)) AS tid, c.customer_id AS cid_ok
    FROM dbo.fact_transactions f
    LEFT JOIN dbo.clean_dim_customers c ON c.customer_id = LTRIM(RTRIM(f.customer_id)))
SELECT s.*, ROW_NUMBER() OVER (PARTITION BY tid ORDER BY CASE WHEN cid_ok IS NULL THEN 1 ELSE 0 END) AS rn
INTO #t FROM s;

INSERT INTO dbo.quarantine_rows
SELECT 'fact_transactions', tid, 'orphan customer_id' FROM #t WHERE rn = 1 AND cid_ok IS NULL;

DROP TABLE IF EXISTS dbo.clean_fact_transactions;
SELECT t.tid                                        AS txn_id,
       t.cid_ok                                     AS customer_id,
       v.vehicle_id                                 AS vehicle_id,
       dbo.fn_parse_date(t.txn_date)                AS txn_date,
       CASE WHEN h.hr BETWEEN 0 AND 23 THEN CAST(h.hr AS INT) END AS txn_hour,
       dbo.fn_map('txn_type', t.txn_type)           AS txn_type,
       CASE WHEN a.amt > 0 AND a.amt < 10000000 THEN a.amt END AS amount_usd,
       dbo.fn_map('payment', t.payment_method)      AS payment_method,
       dbo.fn_map('channel', t.channel)             AS channel,
       dbo.fn_map('country', t.billing_country)     AS billing_country,
       dbo.fn_map('country', t.txn_country)         AS txn_country,
       CAST(dbo.fn_num(t.is_international) AS INT)  AS is_international,
       LTRIM(RTRIM(t.device_type))                  AS device_type,
       CAST(dbo.fn_num(t.is_fraud) AS INT)          AS is_fraud,
       LTRIM(RTRIM(t.fraud_review_note))            AS fraud_review_note,
       CAST(dbo.fn_num(t.chargeback_flag) AS INT)   AS chargeback_flag
INTO dbo.clean_fact_transactions
FROM #t t
CROSS APPLY (SELECT dbo.fn_num(t.txn_hour) AS hr) h
CROSS APPLY (SELECT dbo.fn_parse_money(t.amount_usd) AS amt) a
LEFT JOIN dbo.clean_dim_vehicles v ON v.vehicle_id = LTRIM(RTRIM(t.vehicle_id))
WHERE t.rn = 1 AND t.cid_ok IS NOT NULL;
GO

-- 5.2 service records
DROP TABLE IF EXISTS #s;
WITH s AS (
    SELECT f.*, LTRIM(RTRIM(f.service_id)) AS sid, v.vehicle_id AS vid_ok
    FROM dbo.fact_service_records f
    LEFT JOIN dbo.clean_dim_vehicles v ON v.vehicle_id = LTRIM(RTRIM(f.vehicle_id)))
SELECT s.*, ROW_NUMBER() OVER (PARTITION BY sid ORDER BY CASE WHEN vid_ok IS NULL THEN 1 ELSE 0 END) AS rn
INTO #s FROM s;

INSERT INTO dbo.quarantine_rows
SELECT 'fact_service_records', sid, 'orphan vehicle_id' FROM #s WHERE rn = 1 AND vid_ok IS NULL;

DROP TABLE IF EXISTS dbo.clean_fact_service_records;
SELECT s.sid                                         AS service_id,
       s.vid_ok                                      AS vehicle_id,
       b.branch_id                                   AS branch_id,                -- orphan branch -> NULL
       dbo.fn_parse_date(s.service_date)             AS service_date,
       dbo.fn_map('service_type', s.service_type)    AS service_type,
       CASE WHEN o.odo >= 0 THEN ROUND(o.odo, 0) END AS odometer_at_service_km,
       ROUND(dbo.fn_num(s.vehicle_age_months), 1)    AS vehicle_age_months,
       dbo.fn_map('fuel', s.fuel_type)               AS fuel_type,
       CASE WHEN l.lh > 0 AND l.lh <= 100 THEN l.lh END AS labor_hours,
       CASE WHEN pc.v >= 0 THEN pc.v END             AS parts_cost_usd,
       CASE WHEN lc.v >= 0 THEN lc.v END             AS labor_cost_usd,
       CAST(dbo.fn_num(s.warranty_claim) AS INT)     AS warranty_claim,
       CASE WHEN cs.v BETWEEN 1 AND 5 THEN cs.v END  AS customer_satisfaction,
       CASE WHEN tc.v > 0 THEN tc.v END              AS total_cost_usd
INTO dbo.clean_fact_service_records
FROM #s s
CROSS APPLY (SELECT dbo.fn_num(s.odometer_at_service_km) AS odo) o
CROSS APPLY (SELECT dbo.fn_num(s.labor_hours) AS lh) l
CROSS APPLY (SELECT dbo.fn_parse_money(s.parts_cost_usd) AS v) pc
CROSS APPLY (SELECT dbo.fn_parse_money(s.labor_cost_usd) AS v) lc
CROSS APPLY (SELECT dbo.fn_parse_money(s.total_cost_usd) AS v) tc
CROSS APPLY (SELECT dbo.fn_num(s.customer_satisfaction) AS v) cs
LEFT JOIN dbo.clean_dim_branches b ON b.branch_id = LTRIM(RTRIM(s.branch_id))
WHERE s.rn = 1 AND s.vid_ok IS NOT NULL;
GO

-- 5.3 support interactions
DROP TABLE IF EXISTS #p;
WITH s AS (
    SELECT f.*, LTRIM(RTRIM(f.interaction_id)) AS iid, c.customer_id AS cid_ok
    FROM dbo.fact_support_interactions f
    LEFT JOIN dbo.clean_dim_customers c ON c.customer_id = LTRIM(RTRIM(f.customer_id)))
SELECT s.*, ROW_NUMBER() OVER (PARTITION BY iid ORDER BY CASE WHEN cid_ok IS NULL THEN 1 ELSE 0 END) AS rn
INTO #p FROM s;

INSERT INTO dbo.quarantine_rows
SELECT 'fact_support_interactions', iid, 'orphan customer_id' FROM #p WHERE rn = 1 AND cid_ok IS NULL;

DROP TABLE IF EXISTS dbo.clean_fact_support_interactions;
SELECT p.iid                                         AS interaction_id,
       p.cid_ok                                      AS customer_id,
       dbo.fn_parse_date(p.interaction_date)         AS interaction_date,
       dbo.fn_map('contact', p.contact_channel)      AS contact_channel,
       dbo.fn_map('issue', p.issue_category)         AS issue_category,
       CASE WHEN r.v > 0 AND r.v <= 1000 THEN r.v END AS resolution_hours,
       CASE WHEN c.v BETWEEN 1 AND 5 THEN c.v END    AS csat_score,
       CAST(dbo.fn_num(p.escalated) AS INT)          AS escalated,
       CASE WHEN s.v BETWEEN -1 AND 1 THEN s.v END   AS sentiment_score
INTO dbo.clean_fact_support_interactions
FROM #p p
CROSS APPLY (SELECT dbo.fn_num(p.resolution_hours) AS v) r
CROSS APPLY (SELECT dbo.fn_num(p.csat_score) AS v) c
CROSS APPLY (SELECT dbo.fn_num(p.sentiment_score) AS v) s
WHERE p.rn = 1 AND p.cid_ok IS NOT NULL;
GO

-- 5.4 trips
DROP TABLE IF EXISTS #r;
WITH s AS (
    SELECT f.*, LTRIM(RTRIM(f.trip_id)) AS rid, v.vehicle_id AS vid_ok
    FROM dbo.fact_trips f
    LEFT JOIN dbo.clean_dim_vehicles v ON v.vehicle_id = LTRIM(RTRIM(f.vehicle_id)))
SELECT s.*, ROW_NUMBER() OVER (PARTITION BY rid ORDER BY CASE WHEN vid_ok IS NULL THEN 1 ELSE 0 END) AS rn
INTO #r FROM s;

INSERT INTO dbo.quarantine_rows
SELECT 'fact_trips', rid, 'orphan vehicle_id' FROM #r WHERE rn = 1 AND vid_ok IS NULL;

DROP TABLE IF EXISTS dbo.clean_fact_trips;
SELECT r.rid                                         AS trip_id,
       r.vid_ok                                      AS vehicle_id,
       dbo.fn_parse_date(r.trip_date)                AS trip_date,
       CASE WHEN d.v > 0 AND d.v <= 2000 THEN d.v END AS distance_km,
       dbo.fn_num(r.duration_min)                    AS duration_min,
       CASE WHEN sp.v > 0 AND sp.v <= 250 THEN sp.v END AS avg_speed_kmh,
       CAST(dbo.fn_num(r.harsh_braking_events) AS INT) AS harsh_braking_events,
       dbo.fn_num(r.idle_minutes)                    AS idle_minutes,
       CAST(dbo.fn_num(r.night_trip) AS INT)         AS night_trip
INTO dbo.clean_fact_trips
FROM #r r
CROSS APPLY (SELECT dbo.fn_num(r.distance_km) AS v) d
CROSS APPLY (SELECT dbo.fn_num(r.avg_speed_kmh) AS v) sp
WHERE r.rn = 1 AND r.vid_ok IS NOT NULL;
GO

/* ---------- 6. VALIDATION CHECKS (all should look clean) ---------- */
-- 6.1 row counts: raw vs clean vs quarantined
SELECT 'dim_branches' AS tbl, (SELECT COUNT(*) FROM dbo.dim_branches) AS raw_rows, (SELECT COUNT(*) FROM dbo.clean_dim_branches) AS clean_rows, 0 AS quarantined
UNION ALL SELECT 'dim_customers', (SELECT COUNT(*) FROM dbo.dim_customers), (SELECT COUNT(*) FROM dbo.clean_dim_customers), 0
UNION ALL SELECT 'dim_vehicles', (SELECT COUNT(*) FROM dbo.dim_vehicles), (SELECT COUNT(*) FROM dbo.clean_dim_vehicles), 0
UNION ALL SELECT 'fact_transactions', (SELECT COUNT(*) FROM dbo.fact_transactions), (SELECT COUNT(*) FROM dbo.clean_fact_transactions), (SELECT COUNT(*) FROM dbo.quarantine_rows WHERE source_table='fact_transactions')
UNION ALL SELECT 'fact_service_records', (SELECT COUNT(*) FROM dbo.fact_service_records), (SELECT COUNT(*) FROM dbo.clean_fact_service_records), (SELECT COUNT(*) FROM dbo.quarantine_rows WHERE source_table='fact_service_records')
UNION ALL SELECT 'fact_support_interactions', (SELECT COUNT(*) FROM dbo.fact_support_interactions), (SELECT COUNT(*) FROM dbo.clean_fact_support_interactions), (SELECT COUNT(*) FROM dbo.quarantine_rows WHERE source_table='fact_support_interactions')
UNION ALL SELECT 'fact_trips', (SELECT COUNT(*) FROM dbo.fact_trips), (SELECT COUNT(*) FROM dbo.clean_fact_trips), (SELECT COUNT(*) FROM dbo.quarantine_rows WHERE source_table='fact_trips');

-- 6.2 duplicate IDs left (expect 0 rows)
SELECT 'customers' AS tbl, customer_id AS id, COUNT(*) AS n FROM dbo.clean_dim_customers GROUP BY customer_id HAVING COUNT(*) > 1
UNION ALL SELECT 'vehicles', vehicle_id, COUNT(*) FROM dbo.clean_dim_vehicles GROUP BY vehicle_id HAVING COUNT(*) > 1
UNION ALL SELECT 'transactions', txn_id, COUNT(*) FROM dbo.clean_fact_transactions GROUP BY txn_id HAVING COUNT(*) > 1
UNION ALL SELECT 'service', service_id, COUNT(*) FROM dbo.clean_fact_service_records GROUP BY service_id HAVING COUNT(*) > 1
UNION ALL SELECT 'support', interaction_id, COUNT(*) FROM dbo.clean_fact_support_interactions GROUP BY interaction_id HAVING COUNT(*) > 1
UNION ALL SELECT 'trips', trip_id, COUNT(*) FROM dbo.clean_fact_trips GROUP BY trip_id HAVING COUNT(*) > 1;

-- 6.3 category values not in the standard list (expect 0 rows)
SELECT 'customers.country' AS col, country AS value, COUNT(*) AS n FROM dbo.clean_dim_customers WHERE country NOT IN (SELECT clean_value FROM dbo.map_values WHERE domain='country') GROUP BY country
UNION ALL SELECT 'customers.gender', gender, COUNT(*) FROM dbo.clean_dim_customers WHERE gender NOT IN (SELECT clean_value FROM dbo.map_values WHERE domain='gender') GROUP BY gender
UNION ALL SELECT 'customers.customer_type', customer_type, COUNT(*) FROM dbo.clean_dim_customers WHERE customer_type NOT IN (SELECT clean_value FROM dbo.map_values WHERE domain='customer_type') GROUP BY customer_type
UNION ALL SELECT 'customers.loyalty_tier', loyalty_tier, COUNT(*) FROM dbo.clean_dim_customers WHERE loyalty_tier NOT IN (SELECT clean_value FROM dbo.map_values WHERE domain='loyalty') GROUP BY loyalty_tier
UNION ALL SELECT 'vehicles.fuel_type', fuel_type, COUNT(*) FROM dbo.clean_dim_vehicles WHERE fuel_type NOT IN (SELECT clean_value FROM dbo.map_values WHERE domain='fuel') GROUP BY fuel_type
UNION ALL SELECT 'vehicles.model_name', model_name, COUNT(*) FROM dbo.clean_dim_vehicles WHERE model_name NOT IN (SELECT clean_value FROM dbo.map_values WHERE domain='model') GROUP BY model_name
UNION ALL SELECT 'txn.payment_method', payment_method, COUNT(*) FROM dbo.clean_fact_transactions WHERE payment_method NOT IN (SELECT clean_value FROM dbo.map_values WHERE domain='payment') GROUP BY payment_method
UNION ALL SELECT 'txn.txn_type', txn_type, COUNT(*) FROM dbo.clean_fact_transactions WHERE txn_type NOT IN (SELECT clean_value FROM dbo.map_values WHERE domain='txn_type') GROUP BY txn_type
UNION ALL SELECT 'txn.billing_country', billing_country, COUNT(*) FROM dbo.clean_fact_transactions WHERE billing_country NOT IN (SELECT clean_value FROM dbo.map_values WHERE domain='country') GROUP BY billing_country
UNION ALL SELECT 'service.service_type', service_type, COUNT(*) FROM dbo.clean_fact_service_records WHERE service_type NOT IN (SELECT clean_value FROM dbo.map_values WHERE domain='service_type') GROUP BY service_type
UNION ALL SELECT 'support.contact_channel', contact_channel, COUNT(*) FROM dbo.clean_fact_support_interactions WHERE contact_channel NOT IN (SELECT clean_value FROM dbo.map_values WHERE domain='contact') GROUP BY contact_channel
UNION ALL SELECT 'support.issue_category', issue_category, COUNT(*) FROM dbo.clean_fact_support_interactions WHERE issue_category NOT IN (SELECT clean_value FROM dbo.map_values WHERE domain='issue') GROUP BY issue_category;

-- 6.4 missing values after cleaning (what you will impute in Python)
SELECT 'customers' AS tbl, COUNT(*) AS n,
       SUM(CASE WHEN birth_date IS NULL THEN 1 ELSE 0 END) AS birth_date_null,
       SUM(CASE WHEN annual_income_usd IS NULL THEN 1 ELSE 0 END) AS income_null,
       SUM(CASE WHEN gender IS NULL THEN 1 ELSE 0 END) AS gender_null,
       SUM(CASE WHEN signup_date IS NULL THEN 1 ELSE 0 END) AS signup_null
FROM dbo.clean_dim_customers;
SELECT 'transactions' AS tbl, COUNT(*) AS n,
       SUM(CASE WHEN txn_date IS NULL THEN 1 ELSE 0 END) AS date_null,
       SUM(CASE WHEN amount_usd IS NULL THEN 1 ELSE 0 END) AS amount_null,
       SUM(CASE WHEN txn_hour IS NULL THEN 1 ELSE 0 END) AS hour_null
FROM dbo.clean_fact_transactions;

-- 6.5 target and range sanity
SELECT 'churn rate' AS metric, CAST(AVG(CAST(churned AS FLOAT)) AS DECIMAL(6,4)) AS value FROM dbo.clean_dim_customers
UNION ALL SELECT 'fraud rate', CAST(AVG(CAST(is_fraud AS FLOAT)) AS DECIMAL(6,4)) FROM dbo.clean_fact_transactions
UNION ALL SELECT 'min txn amount', MIN(amount_usd) FROM dbo.clean_fact_transactions
UNION ALL SELECT 'max txn amount', MAX(amount_usd) FROM dbo.clean_fact_transactions
UNION ALL SELECT 'min odometer km', MIN(odometer_km) FROM dbo.clean_dim_vehicles
UNION ALL SELECT 'max odometer km', MAX(odometer_km) FROM dbo.clean_dim_vehicles
UNION ALL SELECT 'avg service cost', AVG(total_cost_usd) FROM dbo.clean_fact_service_records;
GO
```

---

## 6. Feature Engineering

Script: `sql/03_feature_engineering.sql`. It builds three feature tables, one for each grain the models need.

| Table | One row per | Feeds |
|---|---|---|
| `feat_customer` | customer | churn, lifetime value, segmentation |
| `feat_transaction` | transaction | fraud |
| `feat_service` | service job | service cost |

**Customer features** summarise each customer up to the cutoff date:
- Vehicles: number owned, total value, average age, electrified share, average odometer, warranty
- Spending: transaction count, total, average and maximum spend, spend in the last 180 days, days since last purchase, international and online share
- Service: visits, total and average cost, days since last service, warranty claim rate, satisfaction, major repairs
- Support: contacts (total and last 180 days), escalations, CSAT, resolution time, sentiment, complaints
- Driving: trips, distance, average speed, harsh braking per 100 km, night-trip share, idle ratio
- Missing flags: `income_missing`, `age_missing`, `has_txn`, `has_support_contact`, `has_service`, `has_trips`

**Transaction features** describe the payment and the customer at that moment: hour, night flag, weekday, month, international flag, **country mismatch** (billing country differs from transaction country), tenure, and history features built from **earlier** transactions only: prior count, prior average amount, amount versus prior average, days since previous transaction.

**Service features:** service type, fuel type, vehicle age, odometer, kilometres per month, labour hours, warranty claim, out-of-warranty flag, list price, prior service count and prior average cost, branch country, region and type.

**Leakage rules**
- Not carried over: `churn_date`, `last_activity_date`, `fraud_review_note`, `chargeback_flag`, `parts_cost_usd`, `labor_cost_usd`.
- Transaction history uses only rows that happened before the current one.

**Checks run:** one row per ID, row counts match the clean tables, and the signal shows up in the EDA (section 8).

### Code: feature engineering (T-SQL)

```sql
/* =====================================================================
   VANTAGE MOTORS GLOBAL - STEP 3: FEATURE ENGINEERING (SQL Server)
   Reads the clean_* tables from step 2 and builds three feature tables:
     dbo.feat_customer     one row per customer     -> churn, CLV, segmentation
     dbo.feat_transaction  one row per transaction  -> fraud
     dbo.feat_service      one row per service job  -> service cost
   Observation cutoff = 2025-12-31. Leakage columns are NOT carried over:
     churn_date, last_activity_date, fraud_review_note, chargeback_flag,
     parts_cost_usd, labor_cost_usd.
   Missing values stay NULL (impute in Python, fitted on train only).
   ===================================================================== */
USE [Vantage_Automotive];
GO

/* ---------- 3.1 CUSTOMER FEATURES ---------- */
DECLARE @cutoff DATE = '2025-12-31';
DROP TABLE IF EXISTS dbo.feat_customer;

WITH veh AS (
    SELECT customer_id,
           COUNT(*)                                             AS n_vehicles,
           SUM(list_price_usd)                                  AS total_vehicle_value_usd,
           AVG(DATEDIFF(MONTH, purchase_date, @cutoff) * 1.0)   AS avg_vehicle_age_months,
           AVG(CASE WHEN fuel_type IN ('Electric','Hybrid') THEN 1.0 ELSE 0.0 END) AS electrified_share,
           AVG(odometer_km)                                     AS avg_odometer_km,
           AVG(warranty_months * 1.0)                           AS avg_warranty_months
    FROM dbo.clean_dim_vehicles WHERE customer_id IS NOT NULL GROUP BY customer_id),
txn AS (
    SELECT customer_id,
           COUNT(*)                                             AS txn_count,
           SUM(amount_usd)                                      AS total_spend_usd,
           AVG(CAST(amount_usd AS FLOAT))                       AS avg_txn_usd,
           MAX(amount_usd)                                      AS max_txn_usd,
           DATEDIFF(DAY, MAX(txn_date), @cutoff)                AS days_since_last_txn,
           SUM(CASE WHEN txn_date > DATEADD(DAY, -180, @cutoff) THEN 1 ELSE 0 END)           AS txn_count_180d,
           SUM(CASE WHEN txn_date > DATEADD(DAY, -180, @cutoff) THEN amount_usd ELSE 0 END)  AS spend_180d_usd,
           COUNT(DISTINCT txn_type)                             AS n_txn_types,
           AVG(CAST(is_international AS FLOAT))                 AS intl_txn_share,
           AVG(CASE WHEN channel = 'Online' THEN 1.0 ELSE 0.0 END) AS online_txn_share
    FROM dbo.clean_fact_transactions GROUP BY customer_id),
svc AS (
    SELECT v.customer_id,
           COUNT(*)                                             AS service_count,
           SUM(s.total_cost_usd)                                AS total_service_cost_usd,
           AVG(CAST(s.total_cost_usd AS FLOAT))                 AS avg_service_cost_usd,
           DATEDIFF(DAY, MAX(s.service_date), @cutoff)          AS days_since_last_service,
           AVG(CAST(s.warranty_claim AS FLOAT))                 AS warranty_claim_rate,
           AVG(s.customer_satisfaction)                         AS avg_service_csat,
           SUM(CASE WHEN s.service_type IN ('Engine Repair','Transmission','Battery/EV System') THEN 1 ELSE 0 END) AS major_repair_count
    FROM dbo.clean_fact_service_records s
    JOIN dbo.clean_dim_vehicles v ON v.vehicle_id = s.vehicle_id
    WHERE v.customer_id IS NOT NULL GROUP BY v.customer_id),
sup AS (
    SELECT customer_id,
           COUNT(*)                                             AS support_contacts,
           SUM(CASE WHEN interaction_date > DATEADD(DAY, -180, @cutoff) THEN 1 ELSE 0 END) AS support_contacts_180d,
           SUM(escalated)                                       AS escalations,
           AVG(CAST(escalated AS FLOAT))                        AS escalation_rate,
           AVG(csat_score)                                      AS avg_support_csat,
           AVG(resolution_hours)                                AS avg_resolution_hours,
           AVG(sentiment_score)                                 AS avg_sentiment,
           SUM(CASE WHEN issue_category IN ('Vehicle Defect','Service Delay','Warranty','Billing') THEN 1 ELSE 0 END) AS complaint_contacts
    FROM dbo.clean_fact_support_interactions GROUP BY customer_id),
trp AS (
    SELECT v.customer_id,
           COUNT(*)                                             AS trip_count,
           SUM(t.distance_km)                                   AS total_trip_km,
           AVG(t.avg_speed_kmh)                                 AS avg_speed_kmh,
           SUM(t.harsh_braking_events) * 100.0 / NULLIF(SUM(t.distance_km), 0) AS harsh_braking_per_100km,
           AVG(CAST(t.night_trip AS FLOAT))                     AS night_trip_share,
           SUM(t.idle_minutes) / NULLIF(SUM(t.duration_min), 0) AS idle_time_ratio
    FROM dbo.clean_fact_trips t
    JOIN dbo.clean_dim_vehicles v ON v.vehicle_id = t.vehicle_id
    WHERE v.customer_id IS NOT NULL GROUP BY v.customer_id)
SELECT c.customer_id,
       -- profile
       c.gender, c.country, b.region AS home_region, c.customer_type, c.loyalty_tier,
       c.acquisition_channel, c.email_opt_in, c.annual_income_usd,
       DATEDIFF(YEAR, c.birth_date, @cutoff)   AS age_years,
       DATEDIFF(MONTH, c.signup_date, @cutoff) AS tenure_months,
       -- vehicles
       COALESCE(veh.n_vehicles, 0) AS n_vehicles, COALESCE(veh.total_vehicle_value_usd, 0) AS total_vehicle_value_usd,
       veh.avg_vehicle_age_months, veh.electrified_share, veh.avg_odometer_km, veh.avg_warranty_months,
       -- transactions (RFM = days_since_last_txn, txn_count, total_spend_usd)
       COALESCE(txn.txn_count, 0) AS txn_count, COALESCE(txn.total_spend_usd, 0) AS total_spend_usd,
       txn.avg_txn_usd, txn.max_txn_usd, txn.days_since_last_txn,
       COALESCE(txn.txn_count_180d, 0) AS txn_count_180d, COALESCE(txn.spend_180d_usd, 0) AS spend_180d_usd,
       COALESCE(txn.n_txn_types, 0) AS n_txn_types, txn.intl_txn_share, txn.online_txn_share,
       -- service
       COALESCE(svc.service_count, 0) AS service_count, COALESCE(svc.total_service_cost_usd, 0) AS total_service_cost_usd,
       svc.avg_service_cost_usd, svc.days_since_last_service, svc.warranty_claim_rate, svc.avg_service_csat,
       COALESCE(svc.major_repair_count, 0) AS major_repair_count,
       -- support
       COALESCE(sup.support_contacts, 0) AS support_contacts, COALESCE(sup.support_contacts_180d, 0) AS support_contacts_180d,
       COALESCE(sup.escalations, 0) AS escalations, sup.escalation_rate, sup.avg_support_csat,
       sup.avg_resolution_hours, sup.avg_sentiment, COALESCE(sup.complaint_contacts, 0) AS complaint_contacts,
       -- telematics
       COALESCE(trp.trip_count, 0) AS trip_count, COALESCE(trp.total_trip_km, 0) AS total_trip_km,
       trp.avg_speed_kmh, trp.harsh_braking_per_100km, trp.night_trip_share, trp.idle_time_ratio,
       -- targets
       c.churned, c.next_12m_spend_usd
INTO dbo.feat_customer
FROM dbo.clean_dim_customers c
LEFT JOIN dbo.clean_dim_branches b ON b.branch_id = c.home_branch_id
LEFT JOIN veh ON veh.customer_id = c.customer_id
LEFT JOIN txn ON txn.customer_id = c.customer_id
LEFT JOIN svc ON svc.customer_id = c.customer_id
LEFT JOIN sup ON sup.customer_id = c.customer_id
LEFT JOIN trp ON trp.customer_id = c.customer_id;
GO

/* ---------- 3.2 TRANSACTION FEATURES (fraud). History uses PRIOR transactions only ---------- */
DROP TABLE IF EXISTS dbo.feat_transaction;
WITH h AS (
    SELECT t.*,
           COUNT(*) OVER (PARTITION BY t.customer_id ORDER BY t.txn_date, t.txn_id
                          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)                 AS prior_txn_count,
           AVG(CAST(t.amount_usd AS FLOAT)) OVER (PARTITION BY t.customer_id ORDER BY t.txn_date, t.txn_id
                          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)                 AS prior_avg_amount,
           LAG(t.txn_date) OVER (PARTITION BY t.customer_id ORDER BY t.txn_date, t.txn_id)  AS prev_txn_date
    FROM dbo.clean_fact_transactions t)
SELECT h.txn_id, h.customer_id, h.txn_date,
       h.txn_hour,
       CASE WHEN h.txn_hour IS NULL THEN NULL WHEN h.txn_hour BETWEEN 0 AND 4 THEN 1 ELSE 0 END AS is_night,
       DATEDIFF(DAY, '19000101', h.txn_date) % 7                          AS day_of_week,      -- 0 = Monday
       CASE WHEN DATEDIFF(DAY, '19000101', h.txn_date) % 7 >= 5 THEN 1 ELSE 0 END AS is_weekend,
       MONTH(h.txn_date)                                                  AS txn_month,
       h.txn_type, h.amount_usd, h.payment_method, h.channel, h.device_type,
       h.billing_country, h.txn_country, h.is_international,
       CASE WHEN h.billing_country IS NULL OR h.txn_country IS NULL THEN NULL
            WHEN h.billing_country <> h.txn_country THEN 1 ELSE 0 END     AS country_mismatch,
       -- customer context at the time of the transaction
       c.customer_type, c.loyalty_tier, c.country AS customer_country,
       DATEDIFF(MONTH, c.signup_date, h.txn_date)                         AS customer_tenure_months,
       h.prior_txn_count,
       h.prior_avg_amount,
       h.amount_usd / NULLIF(h.prior_avg_amount, 0)                       AS amount_vs_prior_avg,
       DATEDIFF(DAY, h.prev_txn_date, h.txn_date)                         AS days_since_prev_txn,
       -- target
       h.is_fraud
INTO dbo.feat_transaction
FROM h
LEFT JOIN dbo.clean_dim_customers c ON c.customer_id = h.customer_id;
GO

/* ---------- 3.3 SERVICE FEATURES (service cost). parts/labour costs are excluded (leakage) ---------- */
DROP TABLE IF EXISTS dbo.feat_service;
WITH h AS (
    SELECT s.*,
           COUNT(*) OVER (PARTITION BY s.vehicle_id ORDER BY s.service_date, s.service_id
                          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)                 AS prior_service_count,
           AVG(CAST(s.total_cost_usd AS FLOAT)) OVER (PARTITION BY s.vehicle_id ORDER BY s.service_date, s.service_id
                          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)                 AS prior_avg_cost_usd
    FROM dbo.clean_fact_service_records s)
SELECT h.service_id, h.vehicle_id, h.service_date,
       h.service_type, h.fuel_type, h.vehicle_age_months, h.odometer_at_service_km, h.labor_hours,
       h.warranty_claim,                                           -- known when the job is quoted (claim filed or not)
       v.model_name, v.body_type, v.list_price_usd, v.warranty_months,
       CASE WHEN h.vehicle_age_months > v.warranty_months THEN 1 ELSE 0 END AS out_of_warranty,
       h.odometer_at_service_km / NULLIF(h.vehicle_age_months, 0)  AS km_per_month,
       b.country AS branch_country, b.region AS branch_region, b.branch_type,
       MONTH(h.service_date)                                       AS service_month,
       h.prior_service_count, h.prior_avg_cost_usd,
       -- target
       h.total_cost_usd
INTO dbo.feat_service
FROM h
LEFT JOIN dbo.clean_dim_vehicles v ON v.vehicle_id = h.vehicle_id
LEFT JOIN dbo.clean_dim_branches b ON b.branch_id = h.branch_id;
GO

/* ---------- 3.4 CHECKS (send me these results) ---------- */
-- C1. Row counts must match the clean tables, one row per ID
SELECT 'feat_customer' AS tbl, (SELECT COUNT(*) FROM dbo.clean_dim_customers) AS source_rows, COUNT(*) AS feature_rows, COUNT(DISTINCT customer_id) AS distinct_ids FROM dbo.feat_customer
UNION ALL SELECT 'feat_transaction', (SELECT COUNT(*) FROM dbo.clean_fact_transactions), COUNT(*), COUNT(DISTINCT txn_id) FROM dbo.feat_transaction
UNION ALL SELECT 'feat_service', (SELECT COUNT(*) FROM dbo.clean_fact_service_records), COUNT(*), COUNT(DISTINCT service_id) FROM dbo.feat_service;

-- C2. Targets
SELECT 'churn rate' AS metric, CAST(AVG(CAST(churned AS FLOAT)) AS DECIMAL(6,4)) AS value FROM dbo.feat_customer
UNION ALL SELECT 'fraud rate', CAST(AVG(CAST(is_fraud AS FLOAT)) AS DECIMAL(6,4)) FROM dbo.feat_transaction
UNION ALL SELECT 'avg next_12m_spend_usd', CAST(AVG(next_12m_spend_usd) AS DECIMAL(18,2)) FROM dbo.feat_customer
UNION ALL SELECT 'avg service total_cost_usd', CAST(AVG(total_cost_usd) AS DECIMAL(18,2)) FROM dbo.feat_service;

-- C3. Does the signal show up? churn and fraud rate by key features
SELECT CASE WHEN support_contacts >= 5 THEN '5+' WHEN support_contacts >= 3 THEN '3-4' WHEN support_contacts >= 1 THEN '1-2' ELSE '0' END AS support_contacts_band,
       COUNT(*) AS customers, CAST(AVG(CAST(churned AS FLOAT)) AS DECIMAL(6,3)) AS churn_rate
FROM dbo.feat_customer GROUP BY CASE WHEN support_contacts >= 5 THEN '5+' WHEN support_contacts >= 3 THEN '3-4' WHEN support_contacts >= 1 THEN '1-2' ELSE '0' END ORDER BY 1;

SELECT is_night, is_international, COUNT(*) AS txns, CAST(AVG(CAST(is_fraud AS FLOAT)) AS DECIMAL(6,4)) AS fraud_rate
FROM dbo.feat_transaction GROUP BY is_night, is_international ORDER BY is_night, is_international;

-- C4. Share of missing values in the main numeric features
SELECT 'feat_customer' AS tbl,
       CAST(100.0 * SUM(CASE WHEN age_years IS NULL THEN 1 ELSE 0 END) / COUNT(*) AS DECIMAL(5,1)) AS age_null_pct,
       CAST(100.0 * SUM(CASE WHEN annual_income_usd IS NULL THEN 1 ELSE 0 END) / COUNT(*) AS DECIMAL(5,1)) AS income_null_pct,
       CAST(100.0 * SUM(CASE WHEN days_since_last_txn IS NULL THEN 1 ELSE 0 END) / COUNT(*) AS DECIMAL(5,1)) AS no_txn_pct,
       CAST(100.0 * SUM(CASE WHEN avg_support_csat IS NULL THEN 1 ELSE 0 END) / COUNT(*) AS DECIMAL(5,1)) AS csat_null_pct
FROM dbo.feat_customer;
GO
```

---

## 7. Create Relationship Table

Script: `sql/04_relationship_table.sql`.

- Adds **primary keys** to all seven clean tables and **foreign keys** between them (a star schema).
- Builds `dbo.table_relationships`, one row per parent-child link, with child rows, matched rows, **orphan rows** and **match %**.
- Power BI reads the keys and this table to set up its model.
- Safe to re-run: it drops old constraints first.

### Charts

<p align="center">
  <img width="1235" height="666" alt="star_schema" src="https://github.com/user-attachments/assets/27b26527-4854-44d6-808f-a9d7338147b5" />
</p>
<p align="center"><em>Star schema: dimensions (blue) and fact tables (orange)</em></p>

### Code: relationship table (T-SQL)

```sql
/* =====================================================================
   VANTAGE MOTORS GLOBAL - STEP 4: RELATIONSHIP TABLE (SQL Server)
   1. Adds primary keys and foreign keys to the clean_* tables (star schema).
   2. Builds dbo.table_relationships: every parent-child link with row counts,
      orphan rows and match %. Power BI reads the keys and this table.
   Run in ONE query window. Safe to re-run.
   NOTE: if you ever re-run step 2 (cleaning), run section 0 below first,
   otherwise DROP TABLE fails because foreign keys still point at the tables.
   ===================================================================== */
USE [Vantage_Automotive];
GO

/* ---------- 0. DROP OLD CONSTRAINTS (makes the script re-runnable) ---------- */
DECLARE @sql NVARCHAR(MAX) = N'';
SELECT @sql += N'ALTER TABLE dbo.' + QUOTENAME(OBJECT_NAME(parent_object_id)) + N' DROP CONSTRAINT ' + QUOTENAME(name) + N';'
FROM sys.foreign_keys WHERE name LIKE 'FK[_]clean[_]%';
EXEC sp_executesql @sql;
SET @sql = N'';
SELECT @sql += N'ALTER TABLE dbo.' + QUOTENAME(OBJECT_NAME(parent_object_id)) + N' DROP CONSTRAINT ' + QUOTENAME(name) + N';'
FROM sys.key_constraints WHERE type = 'PK' AND name LIKE 'PK[_]clean[_]%';
EXEC sp_executesql @sql;
GO

/* ---------- 1. PRIMARY KEYS (key columns need a fixed type and NOT NULL) ---------- */
ALTER TABLE dbo.clean_dim_branches ALTER COLUMN branch_id NVARCHAR(20) NOT NULL;
ALTER TABLE dbo.clean_dim_customers ALTER COLUMN customer_id NVARCHAR(20) NOT NULL;
ALTER TABLE dbo.clean_dim_vehicles ALTER COLUMN vehicle_id NVARCHAR(20) NOT NULL;
ALTER TABLE dbo.clean_fact_transactions ALTER COLUMN txn_id NVARCHAR(20) NOT NULL;
ALTER TABLE dbo.clean_fact_service_records ALTER COLUMN service_id NVARCHAR(20) NOT NULL;
ALTER TABLE dbo.clean_fact_support_interactions ALTER COLUMN interaction_id NVARCHAR(20) NOT NULL;
ALTER TABLE dbo.clean_fact_trips ALTER COLUMN trip_id NVARCHAR(20) NOT NULL;
GO
ALTER TABLE dbo.clean_dim_branches ADD CONSTRAINT PK_clean_dim_branches PRIMARY KEY (branch_id);
ALTER TABLE dbo.clean_dim_customers ADD CONSTRAINT PK_clean_dim_customers PRIMARY KEY (customer_id);
ALTER TABLE dbo.clean_dim_vehicles ADD CONSTRAINT PK_clean_dim_vehicles PRIMARY KEY (vehicle_id);
ALTER TABLE dbo.clean_fact_transactions ADD CONSTRAINT PK_clean_fact_transactions PRIMARY KEY (txn_id);
ALTER TABLE dbo.clean_fact_service_records ADD CONSTRAINT PK_clean_fact_service_records PRIMARY KEY (service_id);
ALTER TABLE dbo.clean_fact_support_interactions ADD CONSTRAINT PK_clean_fact_support_interactions PRIMARY KEY (interaction_id);
ALTER TABLE dbo.clean_fact_trips ADD CONSTRAINT PK_clean_fact_trips PRIMARY KEY (trip_id);
GO

/* ---------- 2. FOREIGN KEYS ---------- */
ALTER TABLE dbo.clean_dim_customers ALTER COLUMN home_branch_id NVARCHAR(20) NULL;
ALTER TABLE dbo.clean_dim_vehicles ALTER COLUMN customer_id NVARCHAR(20) NULL;
ALTER TABLE dbo.clean_dim_vehicles ALTER COLUMN branch_id NVARCHAR(20) NULL;
ALTER TABLE dbo.clean_fact_transactions ALTER COLUMN customer_id NVARCHAR(20) NULL;
ALTER TABLE dbo.clean_fact_transactions ALTER COLUMN vehicle_id NVARCHAR(20) NULL;
ALTER TABLE dbo.clean_fact_service_records ALTER COLUMN vehicle_id NVARCHAR(20) NULL;
ALTER TABLE dbo.clean_fact_service_records ALTER COLUMN branch_id NVARCHAR(20) NULL;
ALTER TABLE dbo.clean_fact_support_interactions ALTER COLUMN customer_id NVARCHAR(20) NULL;
ALTER TABLE dbo.clean_fact_trips ALTER COLUMN vehicle_id NVARCHAR(20) NULL;
GO
ALTER TABLE dbo.clean_dim_customers ADD CONSTRAINT FK_clean_customers_branch FOREIGN KEY (home_branch_id) REFERENCES dbo.clean_dim_branches (branch_id);
GO
ALTER TABLE dbo.clean_dim_vehicles ADD CONSTRAINT FK_clean_vehicles_customer FOREIGN KEY (customer_id) REFERENCES dbo.clean_dim_customers (customer_id);
GO
ALTER TABLE dbo.clean_dim_vehicles ADD CONSTRAINT FK_clean_vehicles_branch FOREIGN KEY (branch_id) REFERENCES dbo.clean_dim_branches (branch_id);
GO
ALTER TABLE dbo.clean_fact_transactions ADD CONSTRAINT FK_clean_txn_customer FOREIGN KEY (customer_id) REFERENCES dbo.clean_dim_customers (customer_id);
GO
ALTER TABLE dbo.clean_fact_transactions ADD CONSTRAINT FK_clean_txn_vehicle FOREIGN KEY (vehicle_id) REFERENCES dbo.clean_dim_vehicles (vehicle_id);
GO
ALTER TABLE dbo.clean_fact_service_records ADD CONSTRAINT FK_clean_service_vehicle FOREIGN KEY (vehicle_id) REFERENCES dbo.clean_dim_vehicles (vehicle_id);
GO
ALTER TABLE dbo.clean_fact_service_records ADD CONSTRAINT FK_clean_service_branch FOREIGN KEY (branch_id) REFERENCES dbo.clean_dim_branches (branch_id);
GO
ALTER TABLE dbo.clean_fact_support_interactions ADD CONSTRAINT FK_clean_support_customer FOREIGN KEY (customer_id) REFERENCES dbo.clean_dim_customers (customer_id);
GO
ALTER TABLE dbo.clean_fact_trips ADD CONSTRAINT FK_clean_trips_vehicle FOREIGN KEY (vehicle_id) REFERENCES dbo.clean_dim_vehicles (vehicle_id);
GO

/* ---------- 3. RELATIONSHIP TABLE ---------- */
DROP TABLE IF EXISTS dbo.table_relationships;
CREATE TABLE dbo.table_relationships (
    rel_id INT PRIMARY KEY, parent_table VARCHAR(40), parent_key VARCHAR(30),
    child_table VARCHAR(40), child_key VARCHAR(30), cardinality VARCHAR(10), business_meaning VARCHAR(160),
    child_rows INT NULL, child_rows_with_key INT NULL, orphan_rows INT NULL,
    match_pct AS CAST(100.0 * (child_rows_with_key - orphan_rows) / NULLIF(child_rows_with_key, 0) AS DECIMAL(6,2)));

INSERT INTO dbo.table_relationships (rel_id, parent_table, parent_key, child_table, child_key, cardinality, business_meaning) VALUES
(1,'clean_dim_branches','branch_id','clean_dim_customers','home_branch_id','1:M','A branch is the home branch of many customers'),
(2,'clean_dim_customers','customer_id','clean_dim_vehicles','customer_id','1:M','A customer owns one or more vehicles'),
(3,'clean_dim_branches','branch_id','clean_dim_vehicles','branch_id','1:M','A branch sells many vehicles'),
(4,'clean_dim_customers','customer_id','clean_fact_transactions','customer_id','1:M','A customer makes many payments'),
(5,'clean_dim_vehicles','vehicle_id','clean_fact_transactions','vehicle_id','1:M','A payment can relate to a vehicle (often empty)'),
(6,'clean_dim_vehicles','vehicle_id','clean_fact_service_records','vehicle_id','1:M','A vehicle has many service jobs'),
(7,'clean_dim_branches','branch_id','clean_fact_service_records','branch_id','1:M','A branch performs many service jobs'),
(8,'clean_dim_customers','customer_id','clean_fact_support_interactions','customer_id','1:M','A customer contacts support many times'),
(9,'clean_dim_vehicles','vehicle_id','clean_fact_trips','vehicle_id','1:M','A vehicle records many trips'),
(10,'clean_dim_customers','customer_id','feat_customer','customer_id','1:1','Customer feature row (churn, CLV, segmentation)'),
(11,'clean_fact_transactions','txn_id','feat_transaction','txn_id','1:1','Transaction feature row (fraud)'),
(12,'clean_fact_service_records','service_id','feat_service','service_id','1:1','Service-job feature row (service cost)');
GO

/* fill the counts: child rows, rows that carry a key, rows whose key has no parent */
DECLARE @id INT, @pt VARCHAR(40), @pk VARCHAR(30), @ct VARCHAR(40), @ck VARCHAR(30), @sql NVARCHAR(MAX);
DECLARE cur CURSOR FAST_FORWARD FOR SELECT rel_id, parent_table, parent_key, child_table, child_key FROM dbo.table_relationships;
OPEN cur; FETCH NEXT FROM cur INTO @id, @pt, @pk, @ct, @ck;
WHILE @@FETCH_STATUS = 0
BEGIN
    SET @sql = N'UPDATE r SET child_rows = x.n, child_rows_with_key = x.k, orphan_rows = x.o
        FROM dbo.table_relationships r
        CROSS JOIN (SELECT COUNT(*) AS n, COUNT(c.' + QUOTENAME(@ck) + N') AS k,
                           SUM(CASE WHEN c.' + QUOTENAME(@ck) + N' IS NOT NULL AND p.' + QUOTENAME(@pk) + N' IS NULL THEN 1 ELSE 0 END) AS o
                    FROM dbo.' + QUOTENAME(@ct) + N' c
                    LEFT JOIN dbo.' + QUOTENAME(@pt) + N' p ON p.' + QUOTENAME(@pk) + N' = c.' + QUOTENAME(@ck) + N') x
        WHERE r.rel_id = ' + CAST(@id AS NVARCHAR(10)) + N';';
    EXEC sp_executesql @sql;
    FETCH NEXT FROM cur INTO @id, @pt, @pk, @ct, @ck;
END
CLOSE cur; DEALLOCATE cur;
GO

/* ---------- 4. CHECKS (send me these results) ---------- */
-- R1. The relationship table: orphan_rows must be 0 and match_pct 100 on every row
SELECT rel_id, parent_table, parent_key, child_table, child_key, cardinality,
       child_rows, child_rows_with_key, orphan_rows, match_pct, business_meaning
FROM dbo.table_relationships ORDER BY rel_id;

-- R2. Constraints that were created: expect 7 primary keys and 9 foreign keys
SELECT 'primary keys' AS constraint_type, COUNT(*) AS n FROM sys.key_constraints WHERE type = 'PK' AND name LIKE 'PK[_]clean[_]%'
UNION ALL SELECT 'foreign keys', COUNT(*) FROM sys.foreign_keys WHERE name LIKE 'FK[_]clean[_]%';

-- R3. End-to-end join test: customers -> vehicles -> service jobs, with no row lost or doubled
SELECT (SELECT COUNT(*) FROM dbo.clean_fact_service_records) AS service_rows,
       COUNT(*) AS joined_rows
FROM dbo.clean_fact_service_records s
JOIN dbo.clean_dim_vehicles v ON v.vehicle_id = s.vehicle_id
LEFT JOIN dbo.clean_dim_customers c ON c.customer_id = v.customer_id
LEFT JOIN dbo.clean_dim_branches b ON b.branch_id = s.branch_id;
GO
```

---

## 8. Exploratory Data Analysis

Script: `python/05_eda.py`. It reads the three feature tables from SQL Server, changes nothing, and saves every chart. Five findings stood out:

- **Churn climbs with support contacts.** 0 contacts: **17.6%** churn; 1-2: 24.3%; 3-4: 36.1%; 5 or more: **53.6%**.
- **Loyalty tier and customer type matter.** Bronze churns at 37.3% against 23.6% for Platinum. Individuals churn at 32.3%, fleet operators at 23.6%.
- **Fraud is a combination.** Domestic daytime: 1.7%; international daytime: 7.8%; domestic night: 6.3%; **international at night: 25.0%**.
- **Spend is concentrated.** Fleet operators spend about **$12,060** a year, corporate customers $7,100 and individuals $3,361.
- **Missing values** (customers): support CSAT 29.9%, income 11.8%, age 3.8%, no transactions 1.8%.

**What I did about it**
- Fraud is rare (2.4%), so every fraud model is judged on **PR AUC**, not accuracy.
- Missing values are imputed on train only, and missing flags are kept as features.
- Skewed money and count columns get a log transform before scaling (section 13).

### Charts

#### Data overview and target balance

<p align="center">
  <img src="images/eda_01_missing_customers.png" alt="Share of missing values by customer column (filled after the split, on train only)" width="100%">
</p>
<p align="center"><em>Share of missing values by customer column (filled after the split, on train only)</em></p>

<p align="center">
  <img src="images/eda_02_target_balance.png" alt="Target balance: churn is balanced enough, fraud is rare" width="100%">
</p>
<p align="center"><em>Target balance: churn is balanced enough, fraud is rare</em></p>

#### Objective 1: churn

<p align="center">
  <img src="images/eda_03_churn_by_category.png" alt="Churn rate by loyalty tier, customer type, country and acquisition channel" width="100%">
</p>
<p align="center"><em>Churn rate by loyalty tier, customer type, country and acquisition channel</em></p>

<p align="center">
  <img src="images/eda_04_churn_by_behaviour.png" alt="Churn rate by support contacts, tenure and days since last purchase" width="100%">
</p>
<p align="center"><em>Churn rate by support contacts, tenure and days since last purchase</em></p>

<p align="center">
  <img src="images/eda_05_churn_numeric_boxplots.png" alt="Numeric features for churned and retained customers" width="100%">
</p>
<p align="center"><em>Numeric features for churned and retained customers</em></p>

<p align="center">
  <img src="images/eda_06_churn_correlations.png" alt="Top correlations with churn" width="100%">
</p>
<p align="center"><em>Top correlations with churn</em></p>

#### Objective 2: fraud

<p align="center">
  <img src="images/eda_07_fraud_by_category.png" alt="Fraud rate by hour, payment method, channel, transaction type, international flag and country mismatch" width="100%">
</p>
<p align="center"><em>Fraud rate by hour, payment method, channel, transaction type, international flag and country mismatch</em></p>

<p align="center">
  <img src="images/eda_08_fraud_behaviour.png" alt="Fraud rate by amount against the customer's usual and by number of prior transactions" width="100%">
</p>
<p align="center"><em>Fraud rate by amount against the customer's usual and by number of prior transactions</em></p>

<p align="center">
  <img src="images/eda_09_fraud_amount_distribution.png" alt="Transaction amount: fraud against legitimate (log scale)" width="100%">
</p>
<p align="center"><em>Transaction amount: fraud against legitimate (log scale)</em></p>

<p align="center">
  <img src="images/eda_10_fraud_correlations.png" alt="Top correlations with fraud" width="100%">
</p>
<p align="center"><em>Top correlations with fraud</em></p>

#### Objective 3: customer segmentation (behaviour)

<p align="center">
  <img src="images/eda_18_behaviour_distributions.png" alt="Distribution of the behaviour features used for segmentation (log scale)" width="100%">
</p>
<p align="center"><em>Distribution of the behaviour features used for segmentation (log scale)</em></p>

<p align="center">
  <img src="images/eda_19_behaviour_correlation.png" alt="Correlation between behaviour features" width="100%">
</p>
<p align="center"><em>Correlation between behaviour features</em></p>

<p align="center">
  <img src="images/eda_20_value_concentration.png" alt="Customer share against spend share" width="100%">
</p>
<p align="center"><em>Customer share against spend share</em></p>

#### Objective 4: customer lifetime value

<p align="center">
  <img src="images/eda_11_clv_distribution.png" alt="Distribution of next-12-month spend" width="100%">
</p>
<p align="center"><em>Distribution of next-12-month spend</em></p>

<p align="center">
  <img src="images/eda_12_clv_drivers.png" alt="Past spend against future spend, and spend by customer type and loyalty tier" width="100%">
</p>
<p align="center"><em>Past spend against future spend, and spend by customer type and loyalty tier</em></p>

<p align="center">
  <img src="images/eda_13_clv_correlations.png" alt="Top correlations with next-12-month spend" width="100%">
</p>
<p align="center"><em>Top correlations with next-12-month spend</em></p>

#### Objective 5: service cost

<p align="center">
  <img src="images/eda_14_service_cost_distribution.png" alt="Distribution of service cost" width="100%">
</p>
<p align="center"><em>Distribution of service cost</em></p>

<p align="center">
  <img src="images/eda_15_service_cost_by_category.png" alt="Service cost by service type, fuel type and warranty claim" width="100%">
</p>
<p align="center"><em>Service cost by service type, fuel type and warranty claim</em></p>

<p align="center">
  <img src="images/eda_16_service_cost_numeric.png" alt="Service cost against numeric features" width="100%">
</p>
<p align="center"><em>Service cost against numeric features</em></p>

<p align="center">
  <img src="images/eda_17_service_correlations.png" alt="Top correlations with service cost" width="100%">
</p>
<p align="center"><em>Top correlations with service cost</em></p>

### Code: exploratory data analysis (Python)

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 5: EXPLORATORY DATA ANALYSIS (Python)
# Reads the three feature tables from SQL Server (built in step 3).
# Makes NO changes to the data. Saves every chart to ./eda_outputs.
# Needs: pandas numpy matplotlib seaborn sqlalchemy pyodbc
# Run cell by cell (# %% markers) in Kilo/VS Code, or as one script.
# If you run it as a plain script, set SHOW = False so charts do not block.
# =====================================================================
# %% 0. SETTINGS AND CONNECTION
import os, warnings, urllib.parse
from decimal import Decimal
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import pyodbc
from sqlalchemy import create_engine

warnings.filterwarnings("ignore")
SERVER   = r"localhost\SQLEXPRESS"      # same server name you use in SSMS
DATABASE = "Vantage_Automotive"
SHOW     = True                         # False = save charts only
OUT      = "eda_outputs"
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 100

def get_engine():
    drivers = [d for d in pyodbc.drivers() if "SQL Server" in d]
    pref = [d for d in drivers if "ODBC Driver 18" in d] + [d for d in drivers if "ODBC Driver 17" in d] + drivers
    conn = (f"DRIVER={{{pref[0]}}};SERVER={SERVER};DATABASE={DATABASE};"
            "Trusted_Connection=yes;TrustServerCertificate=yes;")
    return create_engine("mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(conn))

def fix_types(df):
    """SQL DECIMAL columns arrive as Decimal objects; dates as text-like. Convert them."""
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].dropna()
            if len(s) and isinstance(s.iloc[0], Decimal):
                df[c] = df[c].astype(float)
    for c in [c for c in df.columns if c.endswith("_date")]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

# %% 1. LOAD THE FEATURE TABLES
engine = get_engine()
cust = fix_types(pd.read_sql("SELECT * FROM dbo.feat_customer", engine))
txn  = fix_types(pd.read_sql("SELECT * FROM dbo.feat_transaction", engine))
svc  = fix_types(pd.read_sql("SELECT * FROM dbo.feat_service", engine))

# %% helpers used by every section below
def save(name):
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight")
    if SHOW: plt.show()
    plt.close()

def rate_bar(ax, df, col, target, sort_index=False, title=None, min_n=30):
    k = df[col]
    if k.dtype == object or k.dtype.name in ("string", "str", "category"):
        k = k.astype("string").fillna("Missing") if k.dtype.name != "category" else k
    d = df.groupby(k, observed=True)[target].agg(["mean", "count"])
    d = d[d["count"] >= min_n]
    d = d.sort_index() if sort_index else d.sort_values("mean", ascending=False)
    ax.bar(d.index.astype(str), d["mean"] * 100, color="#2F6DB5")
    ax.axhline(df[target].mean() * 100, color="#C0392B", ls="--", lw=1)
    for i, v in enumerate(d["mean"]):
        ax.text(i, v * 100, f"{v*100:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_title(title or f"{target} rate by {col} (%)"); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=45)

def top_corr(df, target, drop=(), n=15):
    num = df.select_dtypes("number").drop(columns=[target, *drop], errors="ignore")
    c = num.corrwith(df[target]).dropna()
    return c.reindex(c.abs().sort_values(ascending=False).index).head(n)

def corr_bar(c, title, fname):
    plt.figure(figsize=(8, 5))
    plt.barh(c.index[::-1], c.values[::-1], color=["#2F6DB5" if v >= 0 else "#C0392B" for v in c.values[::-1]])
    plt.title(title); plt.xlabel("Pearson correlation")
    save(fname)

def sample(df, n=3000):
    return df.sample(min(len(df), n), random_state=42)

# %% 2. DATA OVERVIEW
for name, df in [("feat_customer", cust), ("feat_transaction", txn), ("feat_service", svc)]:
    print(f"\n=== {name}: {df.shape[0]:,} rows x {df.shape[1]} columns ===")
    print(df.dtypes.value_counts().to_string())
    miss = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
    print("Top missing %:\n", miss[miss > 0].head(10).to_string())

print("\nNumeric summary - customers (first 12 numeric columns)")
print(cust.select_dtypes("number").iloc[:, :12].describe().T.round(2).to_string())

miss_c = (cust.isna().mean() * 100).sort_values(ascending=False)
miss_c = miss_c[miss_c > 0].head(15)
plt.figure(figsize=(8, 5)); plt.barh(miss_c.index[::-1], miss_c.values[::-1], color="#2F6DB5")
plt.title("Missing values by column - feat_customer (%)"); save("01_missing_customers")

# %% 3. TARGET BALANCE
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for a, (d, t, nm) in zip(ax, [(cust, "churned", "Churn"), (txn, "is_fraud", "Fraud")]):
    v = d[t].value_counts(normalize=True).sort_index() * 100
    a.bar(["Not " + nm.lower(), nm.lower()], v.values, color=["#2F6DB5", "#C0392B"])
    for i, x in enumerate(v.values): a.text(i, x, f"{x:.1f}%", ha="center", va="bottom")
    a.set_title(f"{nm} balance"); a.set_ylabel("%")
save("02_target_balance")
print(f"Churn rate {cust.churned.mean():.3%} | Fraud rate {txn.is_fraud.mean():.3%} (fraud is rare: use class weights and PR AUC)")

# %% 4. CHURN EDA (objective 1)
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
for a, col in zip(ax.ravel(), ["loyalty_tier", "customer_type", "country", "acquisition_channel"]):
    rate_bar(a, cust, col, "churned")
save("03_churn_by_category")

c = cust.copy()
c["support_band"] = pd.cut(c.support_contacts, [-1, 0, 2, 4, np.inf], labels=["0", "1-2", "3-4", "5+"])
c["tenure_band"]  = pd.cut(c.tenure_months, [-1, 12, 24, 48, 72, np.inf], labels=["<1y", "1-2y", "2-4y", "4-6y", "6y+"])
c["idle_band"]    = pd.cut(c.days_since_last_txn, [-1, 90, 180, 365, np.inf], labels=["<90d", "90-180d", "180-365d", "365d+"])
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, col in zip(ax, ["support_band", "tenure_band", "idle_band"]):
    rate_bar(a, c, col, "churned", sort_index=True)
save("04_churn_by_behaviour")

fig, ax = plt.subplots(2, 3, figsize=(13, 7))
for a, col in zip(ax.ravel(), ["support_contacts", "tenure_months", "total_spend_usd", "txn_count", "avg_support_csat", "n_vehicles"]):
    d = cust[[col, "churned"]].copy()
    if col == "total_spend_usd": d[col] = np.log1p(d[col]); a.set_title("log(1 + total_spend_usd)")
    else: a.set_title(col)
    sns.boxplot(data=d, x="churned", y=col, ax=a, showfliers=False, palette=["#2F6DB5", "#C0392B"]); a.set_xlabel("churned"); a.set_ylabel("")
save("05_churn_numeric_boxplots")

cc = top_corr(cust, "churned", drop=["next_12m_spend_usd"])
corr_bar(cc, "Top correlations with churn", "06_churn_correlations")
print("\nChurn: strongest numeric correlations\n", cc.round(3).to_string())
print("\nChurn rate by support band\n", (c.groupby("support_band", observed=True).churned.mean() * 100).round(1).to_string())

# %% 5. FRAUD EDA (objective 2)
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
rate_bar(ax[0, 0], txn, "txn_hour", "is_fraud", sort_index=True, title="Fraud rate by hour (%)")
for a, col in zip(ax.ravel()[1:], ["payment_method", "channel", "txn_type", "is_international", "country_mismatch"]):
    rate_bar(a, txn, col, "is_fraud")
save("07_fraud_by_category")

t = txn.copy()
t["amount_ratio_band"] = pd.cut(t.amount_vs_prior_avg, [0, .5, 1, 2, 4, np.inf], labels=["<0.5x", "0.5-1x", "1-2x", "2-4x", "4x+"])
t["prior_txn_band"]    = pd.cut(t.prior_txn_count, [-1, 0, 2, 5, 10, np.inf], labels=["0", "1-2", "3-5", "6-10", "10+"])
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
rate_bar(ax[0], t, "amount_ratio_band", "is_fraud", sort_index=True, title="Fraud rate by amount vs customer's usual (%)")
rate_bar(ax[1], t, "prior_txn_band", "is_fraud", sort_index=True, title="Fraud rate by prior transactions (%)")
save("08_fraud_behaviour")

plt.figure(figsize=(8, 4.5))
for flag, colr in [(0, "#2F6DB5"), (1, "#C0392B")]:
    v = txn.loc[(txn.is_fraud == flag) & (txn.amount_usd > 0), "amount_usd"]
    plt.hist(np.log10(v), bins=40, alpha=.6, density=True, color=colr, label="fraud" if flag else "legit")
plt.xlabel("log10(amount_usd)"); plt.legend(); plt.title("Transaction amount: fraud vs legitimate")
save("09_fraud_amount_distribution")

fc = top_corr(txn, "is_fraud")
corr_bar(fc, "Top correlations with fraud", "10_fraud_correlations")
print("\nFraud: strongest numeric correlations\n", fc.round(3).to_string())
print("\nMedian amount (legit vs fraud):\n", txn.groupby("is_fraud").amount_usd.median().round(2).to_string())

# %% 6. CUSTOMER VALUE EDA (objective 4: next-12-month spend)
y = cust["next_12m_spend_usd"]
print(f"\nnext_12m_spend_usd: mean {y.mean():,.0f} | median {y.median():,.0f} | skew {y.skew():.2f} | skew after log {np.log1p(y).skew():.2f}")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(y, bins=50, ax=ax[0], color="#2F6DB5"); ax[0].set_title("next_12m_spend_usd")
sns.histplot(np.log1p(y), bins=50, ax=ax[1], color="#2F6DB5"); ax[1].set_title("log(1 + next_12m_spend_usd)")
save("11_clv_distribution")

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
s = sample(cust)
ax[0].scatter(np.log1p(s.total_spend_usd), np.log1p(s.next_12m_spend_usd), s=6, alpha=.4, color="#2F6DB5")
ax[0].set_xlabel("log(1 + total_spend_usd)"); ax[0].set_ylabel("log(1 + next_12m_spend_usd)"); ax[0].set_title("Past spend vs future spend")
sns.boxplot(data=cust, x="customer_type", y="next_12m_spend_usd", ax=ax[1], showfliers=False, color="#9DB9DD"); ax[1].set_title("By customer type")
sns.boxplot(data=cust, x="loyalty_tier", y="next_12m_spend_usd", ax=ax[2], showfliers=False, color="#9DB9DD",
            order=["Bronze", "Silver", "Gold", "Platinum"]); ax[2].set_title("By loyalty tier")
save("12_clv_drivers")
vc = top_corr(cust, "next_12m_spend_usd", drop=["churned"])
corr_bar(vc, "Top correlations with next-12-month spend", "13_clv_correlations")
print("\nCLV: strongest numeric correlations\n", vc.round(3).to_string())

# %% 7. SERVICE COST EDA (objective 5)
y = svc["total_cost_usd"]
print(f"\ntotal_cost_usd: mean {y.mean():,.0f} | median {y.median():,.0f} | skew {y.skew():.2f} | skew after log {np.log1p(y).skew():.2f}")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(y, bins=50, ax=ax[0], color="#2F6DB5"); ax[0].set_title("total_cost_usd")
sns.histplot(np.log1p(y.dropna()), bins=50, ax=ax[1], color="#2F6DB5"); ax[1].set_title("log(1 + total_cost_usd)")
save("14_service_cost_distribution")

order = svc.groupby("service_type").total_cost_usd.median().sort_values(ascending=False).index
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
sns.boxplot(data=svc, x="total_cost_usd", y="service_type", order=order, ax=ax[0], showfliers=False, color="#9DB9DD"); ax[0].set_title("Cost by service type")
sns.boxplot(data=svc, x="fuel_type", y="total_cost_usd", ax=ax[1], showfliers=False, color="#9DB9DD"); ax[1].set_title("Cost by fuel type")
sns.boxplot(data=svc, x="warranty_claim", y="total_cost_usd", ax=ax[2], showfliers=False, color="#9DB9DD"); ax[2].set_title("Cost: warranty claim (0/1)")
save("15_service_cost_by_category")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
s = sample(svc)
ax[0].scatter(s.labor_hours, s.total_cost_usd, s=6, alpha=.4, color="#2F6DB5"); ax[0].set_xlabel("labor_hours"); ax[0].set_ylabel("total_cost_usd")
ax[1].scatter(s.vehicle_age_months, s.total_cost_usd, s=6, alpha=.4, color="#2F6DB5"); ax[1].set_xlabel("vehicle_age_months")
ax[2].scatter(s.odometer_at_service_km, s.total_cost_usd, s=6, alpha=.4, color="#2F6DB5"); ax[2].set_xlabel("odometer_at_service_km")
for a in ax: a.set_yscale("log")
save("16_service_cost_numeric")
sc = top_corr(svc, "total_cost_usd")
corr_bar(sc, "Top correlations with service cost", "17_service_correlations")
print("\nService cost: strongest numeric correlations\n", sc.round(3).to_string())
print("\nMedian cost by warranty claim / out of warranty\n", svc.groupby(["warranty_claim", "out_of_warranty"]).total_cost_usd.median().round(0).to_string())

# %% 8. CUSTOMER BEHAVIOUR EDA (objective 3: segmentation)
beh = ["days_since_last_txn", "txn_count", "total_spend_usd", "avg_txn_usd", "support_contacts", "trip_count",
       "total_trip_km", "service_count", "n_vehicles", "tenure_months"]
sk = pd.DataFrame({"skew": cust[beh].skew(), "skew_after_log1p": np.log1p(cust[beh].clip(lower=0)).skew()}).round(2)
print("\nSkewness of behaviour features (K-Means needs log + scaling when skew is high)\n", sk.to_string())

fig, ax = plt.subplots(2, 3, figsize=(14, 7))
for a, col in zip(ax.ravel(), ["days_since_last_txn", "txn_count", "total_spend_usd", "support_contacts", "trip_count", "total_trip_km"]):
    sns.histplot(np.log1p(cust[col].dropna().clip(lower=0)), bins=40, ax=a, color="#2F6DB5"); a.set_title(f"log(1 + {col})")
save("18_behaviour_distributions")

plt.figure(figsize=(9, 7))
sns.heatmap(cust[beh].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot_kws={"size": 8})
plt.title("Correlation between behaviour features"); save("19_behaviour_correlation")

share = cust.groupby("customer_type").agg(customers=("customer_id", "count"), spend=("total_spend_usd", "sum"))
share["customer_share_%"] = (share.customers / share.customers.sum() * 100).round(1)
share["spend_share_%"] = (share.spend / share.spend.sum() * 100).round(1)
print("\nCustomer vs spend share by type\n", share[["customer_share_%", "spend_share_%"]].to_string())
share[["customer_share_%", "spend_share_%"]].plot(kind="bar", figsize=(7, 4), color=["#9DB9DD", "#2F6DB5"])
plt.title("Customer share vs spend share (%)"); plt.xticks(rotation=0); save("20_value_concentration")

# %% 9. OUTLIERS AND SUMMARY FILE
def iqr_outliers(s):
    s = s.dropna(); q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return int(((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).sum())
rows = []
for nm, df, cols in [("feat_customer", cust, ["total_spend_usd", "next_12m_spend_usd", "txn_count", "support_contacts"]),
                     ("feat_transaction", txn, ["amount_usd", "amount_vs_prior_avg"]),
                     ("feat_service", svc, ["total_cost_usd", "labor_hours", "odometer_at_service_km"])]:
    for col in cols:
        rows.append({"table": nm, "column": col, "iqr_outliers": iqr_outliers(df[col]),
                     "pct": round(100 * iqr_outliers(df[col]) / df[col].notna().sum(), 1), "skew": round(df[col].skew(), 2)})
summary = pd.DataFrame(rows)
print("\nOutliers (IQR rule) - keep them, they are real; use log transforms or tree models\n", summary.to_string(index=False))
summary.to_csv(f"{OUT}/eda_outlier_summary.csv", index=False)
print(f"\nDONE. {len(os.listdir(OUT))} files saved in: {os.path.abspath(OUT)}")
```

---

## 9. Analytical Base Table (SQL build + Python load)

**SQL build:** `sql/06_analytical_base_table.sql` creates the three modelling tables from the feature tables:
- `abt_customer`: 6,000 rows
- `abt_fraud`: 58,656 rows (transactions without a fraud label or date are dropped)
- `abt_service`: 13,836 rows

Rows are removed **only** when the target or the split date is missing. Everything else stays NULL for Python to fill on the training data. The script also creates `abt_data_dictionary` (every column, its type and role: key, target or feature) and `ml_model_map` (objective, ML task, ABT, target, split rule).

**Leakage rules applied**

| Objective | Left out of the features |
|---|---|
| Churn | `churn_date`, `last_activity_date`, `next_12m_spend_usd` |
| Fraud | `fraud_review_note`, `chargeback_flag` |
| Lifetime value | `churned`, `churn_date`, `last_activity_date` |
| Service cost | `parts_cost_usd`, `labor_cost_usd` |

**Python load:** `python/06_load_abt.py` reads the three ABTs from SQL Server, fixes data types (SQL decimals to floats, text dates to dates), runs sanity checks, and saves local copies in `abt_data/` so later steps do not need the database.

### Code: analytical base tables, SQL build (T-SQL)

```sql
/* =====================================================================
   VANTAGE MOTORS GLOBAL - STEP 6: ANALYTICAL BASE TABLES (SQL Server)
   Builds the final modelling tables from the feature tables (step 3):
     dbo.abt_customer  one row per customer     -> churn, CLV, segmentation
     dbo.abt_fraud     one row per transaction  -> fraud
     dbo.abt_service   one row per service job  -> service cost
   Also builds dbo.abt_data_dictionary and dbo.ml_model_map.
   Rules applied: leakage columns are absent; rows are removed ONLY when the
   target (or the split date) is missing; everything else stays NULL for
   Python to impute on the training data. No train/test split here (step 9).
   Run in ONE query window.
   ===================================================================== */
USE [Vantage_Automotive];
GO

/* ---------- 6.1 CUSTOMER ABT ---------- */
DROP TABLE IF EXISTS dbo.abt_customer;
SELECT f.customer_id,
       f.gender, f.country, f.home_region, f.customer_type, f.loyalty_tier, f.acquisition_channel, f.email_opt_in,
       f.annual_income_usd, f.age_years, f.tenure_months,
       f.n_vehicles, f.total_vehicle_value_usd, f.avg_vehicle_age_months, f.electrified_share, f.avg_odometer_km, f.avg_warranty_months,
       f.txn_count, f.total_spend_usd, f.avg_txn_usd, f.max_txn_usd, f.days_since_last_txn, f.txn_count_180d, f.spend_180d_usd,
       f.n_txn_types, f.intl_txn_share, f.online_txn_share,
       f.service_count, f.total_service_cost_usd, f.avg_service_cost_usd, f.days_since_last_service, f.warranty_claim_rate,
       f.avg_service_csat, f.major_repair_count,
       f.support_contacts, f.support_contacts_180d, f.escalations, f.escalation_rate, f.avg_support_csat,
       f.avg_resolution_hours, f.avg_sentiment, f.complaint_contacts,
       f.trip_count, f.total_trip_km, f.avg_speed_kmh, f.harsh_braking_per_100km, f.night_trip_share, f.idle_time_ratio,
       -- activity flags (tell the model "no history" apart from "low value")
       CASE WHEN f.txn_count > 0 THEN 1 ELSE 0 END             AS has_txn,
       CASE WHEN f.support_contacts > 0 THEN 1 ELSE 0 END      AS has_support_contact,
       CASE WHEN f.service_count > 0 THEN 1 ELSE 0 END         AS has_service,
       CASE WHEN f.trip_count > 0 THEN 1 ELSE 0 END            AS has_trips,
       CASE WHEN f.annual_income_usd IS NULL THEN 1 ELSE 0 END AS income_missing,
       CASE WHEN f.age_years IS NULL THEN 1 ELSE 0 END         AS age_missing,
       -- targets
       f.churned, f.next_12m_spend_usd
INTO dbo.abt_customer
FROM dbo.feat_customer f
WHERE f.churned IS NOT NULL;
GO

/* ---------- 6.2 FRAUD ABT (needs a transaction date for the time-based split) ---------- */
DROP TABLE IF EXISTS dbo.abt_fraud;
SELECT f.txn_id, f.customer_id, f.txn_date,
       f.txn_hour, f.is_night, f.day_of_week, f.is_weekend, f.txn_month,
       f.txn_type, f.amount_usd, f.payment_method, f.channel, f.device_type,
       f.billing_country, f.txn_country, f.is_international, f.country_mismatch,
       f.customer_type, f.loyalty_tier, f.customer_tenure_months,
       f.prior_txn_count, f.prior_avg_amount, f.amount_vs_prior_avg, f.days_since_prev_txn,
       CASE WHEN f.prior_txn_count = 0 THEN 1 ELSE 0 END AS is_first_txn,
       -- target
       f.is_fraud
INTO dbo.abt_fraud
FROM dbo.feat_transaction f
WHERE f.is_fraud IS NOT NULL AND f.txn_date IS NOT NULL;
GO

/* ---------- 6.3 SERVICE ABT (rows without a cost cannot be used to train) ---------- */
DROP TABLE IF EXISTS dbo.abt_service;
SELECT f.service_id, f.vehicle_id, f.service_date,
       f.service_type, f.fuel_type, f.vehicle_age_months, f.odometer_at_service_km, f.labor_hours, f.warranty_claim,
       f.model_name, f.body_type, f.list_price_usd, f.warranty_months, f.out_of_warranty, f.km_per_month,
       f.branch_country, f.branch_region, f.branch_type, f.service_month,
       f.prior_service_count, f.prior_avg_cost_usd,
       CASE WHEN f.prior_service_count = 0 THEN 1 ELSE 0 END AS is_first_service,
       -- target
       f.total_cost_usd
INTO dbo.abt_service
FROM dbo.feat_service f
WHERE f.total_cost_usd IS NOT NULL;
GO

/* ---------- 6.4 ABT DATA DICTIONARY and MODEL MAP ---------- */
DROP TABLE IF EXISTS dbo.abt_data_dictionary;
SELECT TABLE_NAME AS abt_name, COLUMN_NAME AS column_name, ORDINAL_POSITION AS position, DATA_TYPE AS sql_type,
       CASE WHEN COLUMN_NAME IN ('customer_id','txn_id','service_id','vehicle_id') THEN 'key (not a feature)'
            WHEN COLUMN_NAME IN ('churned','next_12m_spend_usd','is_fraud','total_cost_usd') THEN 'target'
            WHEN COLUMN_NAME IN ('txn_date','service_date') THEN 'date (used for the split only)'
            ELSE 'feature' END AS role
INTO dbo.abt_data_dictionary
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME IN ('abt_customer','abt_fraud','abt_service');

DROP TABLE IF EXISTS dbo.ml_model_map;
CREATE TABLE dbo.ml_model_map (
    objective_no INT, business_objective VARCHAR(60), ml_task VARCHAR(30), abt_name VARCHAR(30),
    target VARCHAR(30), exclude_from_features VARCHAR(200), split_strategy VARCHAR(120));
INSERT INTO dbo.ml_model_map VALUES
(1,'Customer churn','Classification','abt_customer','churned','next_12m_spend_usd, customer_id','Stratified 80/20 on churned'),
(2,'Fraud detection','Classification','abt_fraud','is_fraud','txn_id, customer_id, txn_date','Time-based: train to 2025-06-30, test after'),
(3,'Customer segmentation','Clustering (K-Means)','abt_customer','none','churned, next_12m_spend_usd, customer_id','No target; fit on 80%, assign the other 20%'),
(4,'Customer lifetime value','Regression','abt_customer','next_12m_spend_usd','churned, customer_id','Random 80/20'),
(5,'Service cost','Regression','abt_service','total_cost_usd','service_id, vehicle_id, service_date','Random 80/20 grouped by vehicle_id');
GO

/* ---------- 6.5 CHECKS (send me these results) ---------- */
-- C1. Rows kept vs rows available
SELECT 'abt_customer' AS abt, (SELECT COUNT(*) FROM dbo.feat_customer) AS source_rows, COUNT(*) AS abt_rows, (SELECT COUNT(*) FROM dbo.feat_customer) - COUNT(*) AS rows_removed FROM dbo.abt_customer
UNION ALL SELECT 'abt_fraud', (SELECT COUNT(*) FROM dbo.feat_transaction), COUNT(*), (SELECT COUNT(*) FROM dbo.feat_transaction) - COUNT(*) FROM dbo.abt_fraud
UNION ALL SELECT 'abt_service', (SELECT COUNT(*) FROM dbo.feat_service), COUNT(*), (SELECT COUNT(*) FROM dbo.feat_service) - COUNT(*) FROM dbo.abt_service;

-- C2. Targets
SELECT 'churn rate' AS metric, CAST(AVG(CAST(churned AS FLOAT)) AS DECIMAL(8,4)) AS value FROM dbo.abt_customer
UNION ALL SELECT 'customers with next_12m_spend', COUNT(next_12m_spend_usd) FROM dbo.abt_customer
UNION ALL SELECT 'fraud rate', CAST(AVG(CAST(is_fraud AS FLOAT)) AS DECIMAL(8,4)) FROM dbo.abt_fraud
UNION ALL SELECT 'fraud cases', SUM(is_fraud) FROM dbo.abt_fraud
UNION ALL SELECT 'fraud cases before 2025-07-01 (train)', SUM(CASE WHEN txn_date < '2025-07-01' THEN is_fraud ELSE 0 END) FROM dbo.abt_fraud
UNION ALL SELECT 'fraud cases from 2025-07-01 (test)', SUM(CASE WHEN txn_date >= '2025-07-01' THEN is_fraud ELSE 0 END) FROM dbo.abt_fraud
UNION ALL SELECT 'avg service cost', CAST(AVG(total_cost_usd) AS DECIMAL(18,2)) FROM dbo.abt_service;

-- C3. Columns by role in each ABT
SELECT abt_name, role, COUNT(*) AS columns FROM dbo.abt_data_dictionary GROUP BY abt_name, role ORDER BY abt_name, role;

-- C4. Model map
SELECT * FROM dbo.ml_model_map ORDER BY objective_no;
GO
```

### Code: analytical base tables, Python load

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 6 (Python part): LOAD AND CHECK THE ABTs
# Loads abt_customer, abt_fraud, abt_service from SQL Server, checks them,
# and saves local copies in ./abt_data so the later steps do not need SQL.
# Needs: pandas numpy sqlalchemy pyodbc. Changes nothing in the database.
# =====================================================================
# %% 0. SETTINGS AND CONNECTION
import os, warnings, urllib.parse
from decimal import Decimal
import numpy as np, pandas as pd
import pyodbc
from sqlalchemy import create_engine

warnings.filterwarnings("ignore")
SERVER   = r"localhost\SQLEXPRESS"
DATABASE = "Vantage_Automotive"
OUT      = "abt_data"
os.makedirs(OUT, exist_ok=True)

def get_engine():
    drivers = [d for d in pyodbc.drivers() if "SQL Server" in d]
    pref = [d for d in drivers if "ODBC Driver 18" in d] + [d for d in drivers if "ODBC Driver 17" in d] + drivers
    conn = (f"DRIVER={{{pref[0]}}};SERVER={SERVER};DATABASE={DATABASE};"
            "Trusted_Connection=yes;TrustServerCertificate=yes;")
    return create_engine("mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(conn))

def fix_types(df):
    for c in df.columns:
        if df[c].dtype == object:
            s = df[c].dropna()
            if len(s) and isinstance(s.iloc[0], Decimal):
                df[c] = df[c].astype(float)
    for c in [c for c in df.columns if c.endswith("_date")]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

# %% 1. LOAD
engine = get_engine()
abt_customer = fix_types(pd.read_sql("SELECT * FROM dbo.abt_customer", engine))
abt_fraud    = fix_types(pd.read_sql("SELECT * FROM dbo.abt_fraud", engine))
abt_service  = fix_types(pd.read_sql("SELECT * FROM dbo.abt_service", engine))
dictionary   = pd.read_sql("SELECT * FROM dbo.abt_data_dictionary ORDER BY abt_name, position", engine)
model_map    = pd.read_sql("SELECT * FROM dbo.ml_model_map ORDER BY objective_no", engine)

# %% 2. CHECKS
ABT = {"abt_customer": (abt_customer, "customer_id", ["churned", "next_12m_spend_usd"]),
       "abt_fraud":    (abt_fraud, "txn_id", ["is_fraud"]),
       "abt_service":  (abt_service, "service_id", ["total_cost_usd"])}
for name, (df, key, targets) in ABT.items():
    print(f"\n=== {name}: {df.shape[0]:,} rows x {df.shape[1]} columns ===")
    print("duplicate keys:", int(df[key].duplicated().sum()), "| duplicate rows:", int(df.duplicated().sum()))
    for t in targets:
        print(f"target {t}: missing {df[t].isna().sum()} | mean {df[t].mean():,.4f} | min {df[t].min():,.2f} | max {df[t].max():,.2f}")
    miss = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
    print("columns with missing values (%):\n", miss[miss > 0].head(8).to_string())
    print("column types:", df.dtypes.astype(str).value_counts().to_dict())

print("\nModel map\n", model_map[["objective_no", "business_objective", "ml_task", "abt_name", "target"]].to_string(index=False))
print("\nColumns by role\n", dictionary.groupby(["abt_name", "role"]).size().to_string())

# leakage guard: none of these may exist in any ABT
LEAK = ["churn_date", "last_activity_date", "fraud_review_note", "chargeback_flag", "parts_cost_usd", "labor_cost_usd"]
found = [(n, c) for n, (d, _, _) in ABT.items() for c in LEAK if c in d.columns]
print("\nLeakage columns found in ABTs:", found if found else "none")

# %% 3. SAVE LOCAL COPIES FOR THE NEXT STEPS
abt_customer.to_pickle(f"{OUT}/abt_customer.pkl")
abt_fraud.to_pickle(f"{OUT}/abt_fraud.pkl")
abt_service.to_pickle(f"{OUT}/abt_service.pkl")
dictionary.to_csv(f"{OUT}/abt_data_dictionary.csv", index=False)
print("\nSaved to", os.path.abspath(OUT), "->", sorted(os.listdir(OUT)))
```

---

## 10. Encode Categorical Columns

Script: `python/07_encoding.py`.

- **Ordered columns** (loyalty tier: Bronze, Silver, Gold, Platinum) become 0 to 3.
- **Unordered columns** (country, customer type, channel, payment method, service type, and so on) are one-hot encoded. The most common level is the baseline and is dropped.
- Missing text becomes its own level, `Missing`.
- Unseen values at prediction time fall back to the baseline.
- No target is used, so there is no leakage.
- The recipe is saved as `encoding_spec.json`. The Streamlit app reuses the exact same recipe, so a new customer is encoded the same way as the training data.

### Code: encode categorical columns (Python)

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 7: ENCODE CATEGORICAL COLUMNS (Python)
# Reads the ABTs saved in step 6 (./abt_data), encodes the text columns,
# and saves encoded copies plus an encoding recipe (encoding_spec.json).
# The recipe is what the Streamlit app will use later, so a new customer or
# transaction is encoded exactly like the training data.
# Method: ordinal for ordered columns (loyalty tier), one-hot for the rest.
# Missing text values become their own level "Missing". No target is used,
# so there is no leakage. Needs: pandas numpy.
# =====================================================================
# %% 0. SETTINGS
import os, re, json
import numpy as np, pandas as pd

DATA = "abt_data"                      # same folder step 6 saved into
LOYALTY = {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3}

# per ABT: key columns, target columns, ordered columns, one-hot columns
CONFIG = {
    "customer": {"file": "abt_customer.pkl", "keys": ["customer_id"], "targets": ["churned", "next_12m_spend_usd"],
                 "ordinal": {"loyalty_tier": LOYALTY},
                 "onehot": ["gender", "country", "home_region", "customer_type", "acquisition_channel", "email_opt_in"]},
    "fraud":    {"file": "abt_fraud.pkl", "keys": ["txn_id", "customer_id", "txn_date"], "targets": ["is_fraud"],
                 "ordinal": {"loyalty_tier": LOYALTY},
                 "onehot": ["txn_type", "payment_method", "channel", "device_type", "billing_country", "txn_country", "customer_type"]},
    "service":  {"file": "abt_service.pkl", "keys": ["service_id", "vehicle_id", "service_date"], "targets": ["total_cost_usd"],
                 "ordinal": {},
                 "onehot": ["service_type", "fuel_type", "model_name", "body_type", "branch_country", "branch_region", "branch_type"]},
}

# %% 1. ENCODING FUNCTIONS
def clean_name(text):
    return re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_")

def fit_spec(df, cfg):
    """Learn the recipe: categories kept per column. Most frequent level is the baseline (dropped)."""
    onehot = {}
    for col in cfg["onehot"]:
        s = df[col].astype("string").fillna("Missing")
        order = s.value_counts().index.tolist()
        onehot[col] = {"baseline": order[0], "kept": order[1:]}
    return {"keys": cfg["keys"], "targets": cfg["targets"], "ordinal": cfg["ordinal"], "onehot": onehot}

def apply_spec(df, spec):
    """Encode any dataframe with a recipe. Unseen levels become all zeros (= baseline)."""
    out = df.copy()
    for col, mapping in spec["ordinal"].items():
        out[col] = out[col].astype("string").map(mapping).astype("float")
    new_cols = []
    for col, info in spec["onehot"].items():
        s = out[col].astype("string").fillna("Missing")
        for cat in info["kept"]:
            name = f"{col}_{clean_name(cat)}"
            out[name] = (s == cat).astype(int)
            new_cols.append(name)
        out = out.drop(columns=[col])
    fixed = spec["keys"] + spec["targets"]
    features = [c for c in out.columns if c not in fixed]
    return out[spec["keys"] + features + spec["targets"]], new_cols

# %% 2. ENCODE EACH ABT
spec_all, encoded = {}, {}
for name, cfg in CONFIG.items():
    df = pd.read_pickle(f"{DATA}/{cfg['file']}")
    spec = fit_spec(df, cfg)
    enc, new_cols = apply_spec(df, spec)
    spec_all[name], encoded[name] = spec, enc

    print(f"\n=== {name}: {df.shape[1]} columns -> {enc.shape[1]} columns ({len(new_cols)} new one-hot columns) ===")
    for col, info in spec["onehot"].items():
        print(f"  {col:22s} {len(info['kept']) + 1:2d} levels | baseline (dropped): {info['baseline']}")
    for col in spec["ordinal"]:
        print(f"  {col:22s} ordinal 0-3 | unmapped values: {int(enc[col].isna().sum())}")

# %% 3. CHECKS
for name, enc in encoded.items():
    spec = spec_all[name]
    non_numeric = [c for c in enc.columns if c not in spec["keys"] and not pd.api.types.is_numeric_dtype(enc[c])]
    dup_cols = enc.columns[enc.columns.duplicated()].tolist()
    print(f"\n{name}: rows {len(enc):,} | text columns left: {non_numeric if non_numeric else 'none'} | duplicate column names: {dup_cols if dup_cols else 'none'}")
    # re-applying the recipe to the original data must give exactly the same table
    again, _ = apply_spec(pd.read_pickle(f"{DATA}/{CONFIG[name]['file']}"), spec)
    print("  recipe reproduces the encoded table:", bool(again.equals(enc)))

# %% 4. SAVE
for name, enc in encoded.items():
    enc.to_pickle(f"{DATA}/enc_{name}.pkl")
with open(f"{DATA}/encoding_spec.json", "w") as f:
    json.dump(spec_all, f, indent=2)
cols = pd.DataFrame([(n, c, "key" if c in spec_all[n]["keys"] else "target" if c in spec_all[n]["targets"] else "feature")
                     for n, e in encoded.items() for c in e.columns], columns=["abt", "column", "role"])
cols.to_csv(f"{DATA}/encoded_columns.csv", index=False)
print("\nSaved:", sorted(f for f in os.listdir(DATA) if f.startswith("enc") or f.endswith(".json")))
print(cols.groupby(["abt", "role"]).size().to_string())
```

---

## 11. Separate X and Y

Script: `python/08_separate_xy.py`. Each objective gets its own features (X) and target (y). Keys and dates are kept aside as metadata.

| Objective | y (target) | Excluded from X |
|---|---|---|
| Churn | `churned` | `next_12m_spend_usd`, IDs, dates |
| Fraud | `is_fraud` | IDs, dates |
| Segmentation | none | both targets |
| Lifetime value | `next_12m_spend_usd` | `churned` (it would hand over the answer), IDs |
| Service cost | `total_cost_usd` | IDs, dates |

### Code: separate X and Y (Python)

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 8: SEPARATE X AND y (Python)
# Reads the encoded tables from step 7 (./abt_data) and, for each of the
# 5 models, separates the features (X) from the target (y).
#   churn, fraud            -> classification
#   clv, service            -> regression
#   segmentation            -> clustering (X only, no y)
# Keys, dates and the other model's target are kept aside in "meta" so they
# can never leak into X, but are still available for the split (step 9) and
# for writing predictions back to SQL Server later. Needs: pandas numpy.
# =====================================================================
# %% 0. SETTINGS
import os, json, pickle
import numpy as np, pandas as pd

DATA = "abt_data"
LEAKAGE = ["churn_date", "last_activity_date", "fraud_review_note", "chargeback_flag", "parts_cost_usd", "labor_cost_usd"]

# Segmentation uses BEHAVIOUR only. Demographics (country, type, gender) are left out
# on purpose, so the segments are built from what customers DO and can then be profiled by who they ARE.
SEG_FEATURES = ["days_since_last_txn", "txn_count", "total_spend_usd", "avg_txn_usd", "txn_count_180d", "spend_180d_usd",
                "n_txn_types", "online_txn_share", "intl_txn_share", "service_count", "total_service_cost_usd",
                "major_repair_count", "support_contacts", "escalation_rate", "avg_support_csat", "complaint_contacts",
                "trip_count", "total_trip_km", "harsh_braking_per_100km", "night_trip_share",
                "n_vehicles", "total_vehicle_value_usd", "tenure_months", "loyalty_tier"]

MODELS = {
    "churn":        {"source": "customer", "task": "classification", "target": "churned",
                     "meta": ["customer_id"]},
    "fraud":        {"source": "fraud",    "task": "classification", "target": "is_fraud",
                     "meta": ["txn_id", "customer_id", "txn_date"]},
    "clv":          {"source": "customer", "task": "regression",     "target": "next_12m_spend_usd",
                     "meta": ["customer_id"]},
    "service":      {"source": "service",  "task": "regression",     "target": "total_cost_usd",
                     "meta": ["service_id", "vehicle_id", "service_date"]},
    "segmentation": {"source": "customer", "task": "clustering",     "target": None,
                     "meta": ["customer_id"], "features": SEG_FEATURES},
}
ALL_TARGETS = {"customer": ["churned", "next_12m_spend_usd"], "fraud": ["is_fraud"], "service": ["total_cost_usd"]}
ALL_KEYS    = {"customer": ["customer_id"], "fraud": ["txn_id", "customer_id", "txn_date"],
               "service": ["service_id", "vehicle_id", "service_date"]}

# %% 1. LOAD THE ENCODED TABLES
enc = {k: pd.read_pickle(f"{DATA}/enc_{k}.pkl") for k in ["customer", "fraud", "service"]}
for k, d in enc.items():
    print(f"enc_{k}: {d.shape[0]:,} rows x {d.shape[1]} columns")

# %% 2. SEPARATE X AND y FOR EACH MODEL
def split_xy(cfg):
    df = enc[cfg["source"]]
    meta = df[cfg["meta"]].copy()
    y = df[cfg["target"]].copy() if cfg["target"] else None
    X = df.drop(columns=ALL_KEYS[cfg["source"]] + ALL_TARGETS[cfg["source"]])   # no key, date or target can stay in X
    if "features" in cfg:
        missing = [c for c in cfg["features"] if c not in X.columns]
        assert not missing, f"segmentation features not found: {missing}"
        X = X[cfg["features"]]
    return X, y, meta

result = {}
for name, cfg in MODELS.items():
    X, y, meta = split_xy(cfg)
    result[name] = {"X": X, "y": y, "meta": meta, "features": X.columns.tolist(), "task": cfg["task"], "target": cfg["target"]}

# %% 3. CHECKS
rows = []
for name, r in result.items():
    X, y = r["X"], r["y"]
    bad = [c for c in X.columns if c in LEAKAGE or c in sum(ALL_TARGETS.values(), []) or c in sum(ALL_KEYS.values(), [])]
    non_num = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    const = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    assert not bad, f"{name}: forbidden columns in X: {bad}"
    assert not non_num, f"{name}: non-numeric columns in X: {non_num}"
    assert X.index.equals(r["meta"].index) and (y is None or X.index.equals(y.index)), f"{name}: index mismatch"
    print(f"\n=== {name} ({r['task']}) ===")
    print(f"X: {X.shape[0]:,} rows x {X.shape[1]} features | forbidden columns in X: none | text columns in X: none")
    print(f"missing cells in X: {X.isna().mean().mean():.1%} | constant columns: {const if const else 'none'}")
    if r["task"] == "classification":
        vc = y.value_counts().sort_index()
        print(f"y = {r['target']} | missing {int(y.isna().sum())} | class 0: {vc.get(0, 0):,} | class 1: {vc.get(1, 0):,} | positive rate {y.mean():.2%}")
        info = f"positive rate {y.mean():.2%}"
    elif r["task"] == "regression":
        print(f"y = {r['target']} | missing {int(y.isna().sum())} | mean {y.mean():,.1f} | median {y.median():,.1f} | min {y.min():,.1f} | max {y.max():,.1f} | skew {y.skew():.2f}")
        info = f"mean {y.mean():,.0f}, skew {y.skew():.2f}"
    else:
        print("y = none (unsupervised): clusters are found from X alone")
        info = "no target"
    num = X.select_dtypes("number")
    if num.shape[1] > 1:
        corr = num.sample(min(len(num), 10000), random_state=42).corr().abs()
        pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), 1)).stack()
        pairs = pairs[pairs > 0.95].sort_values(ascending=False).head(5)
        print("feature pairs with |correlation| > 0.95 (informational):", "none" if pairs.empty else "")
        for (a, b), v in pairs.items(): print(f"   {a}  <->  {b}: {v:.3f}")
    rows.append({"model": name, "task": r["task"], "rows": len(X), "features": X.shape[1], "target": r["target"] or "none", "y_info": info})

print("\nSUMMARY\n", pd.DataFrame(rows).to_string(index=False))

# %% 4. SAVE
for name, r in result.items():
    with open(f"{DATA}/xy_{name}.pkl", "wb") as f:
        pickle.dump({k: r[k] for k in ["X", "y", "meta", "features", "task", "target"]}, f)
    with open(f"{DATA}/features_{name}.json", "w") as f:
        json.dump(r["features"], f, indent=2)
print("\nSaved:", sorted(x for x in os.listdir(DATA) if x.startswith("xy_") or x.startswith("features_")))
```

---

## 12. Train Test Split

Script: `python/09_train_test_split.py`. The split matches how each model will be used.

| Objective | Split | Why |
|---|---|---|
| Churn, lifetime value, segmentation | 80/20 on customers, stratified on churn, **one shared split** | The same customer can never be train for one model and test for another |
| Fraud | **Time-based**: train up to 2025-06-30, test after | Fraud models predict the future. A random split would let them peek |
| Service cost | 80/20 **grouped by vehicle** | A vehicle never appears in both sets |

Test sizes: 1,200 customers, 9,821 transactions (249 fraud cases), 2,810 service jobs. The script prints the overlap check for vehicles and customers (zero).

### Code: train test split (Python)

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 9: TRAIN / TEST SPLIT (Python)
# Reads the X/y files from step 8 (./abt_data) and splits each model.
#   churn, clv, segmentation : ONE shared customer split, 80/20, stratified on churn
#                              (the same customers are in train for all three models)
#   fraud                    : time-based. Train = up to 2025-06-30, test = 2025-07-01 onward
#   service                  : 80/20 grouped by vehicle_id (a vehicle is never in both sets)
# Nothing is imputed or scaled here. That happens in step 10 using the TRAIN set only.
# Needs: pandas numpy scikit-learn.
# =====================================================================
# %% 0. SETTINGS
import pickle
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

DATA   = "abt_data"
SEED   = 42
TEST   = 0.20
CUTOFF = pd.Timestamp("2025-06-30")        # last day of the fraud training period

# %% 1. LOAD
MODELS = ["churn", "clv", "segmentation", "fraud", "service"]
xy = {m: pickle.load(open(f"{DATA}/xy_{m}.pkl", "rb")) for m in MODELS}
for m in MODELS:
    print(f"{m:13s} X {xy[m]['X'].shape}")

# %% 2. WORK OUT THE TRAIN / TEST ROWS FOR EACH MODEL
idx = {}

# 2.1 customers: one split shared by churn, clv and segmentation
assert xy["churn"]["X"].index.equals(xy["clv"]["X"].index) and xy["churn"]["X"].index.equals(xy["segmentation"]["X"].index)
y_churn = xy["churn"]["y"]
cust_train, cust_test = train_test_split(y_churn.index, test_size=TEST, stratify=y_churn, random_state=SEED)
for m in ["churn", "clv", "segmentation"]:
    idx[m] = (cust_train, cust_test)

# 2.2 fraud: time-based (the model must predict FUTURE transactions)
dates = xy["fraud"]["meta"]["txn_date"]
idx["fraud"] = (dates.index[dates <= CUTOFF], dates.index[dates > CUTOFF])

# 2.3 service: grouped by vehicle
svc_meta = xy["service"]["meta"]
gss = GroupShuffleSplit(n_splits=1, test_size=TEST, random_state=SEED)
tr, te = next(gss.split(xy["service"]["X"], groups=svc_meta["vehicle_id"]))
idx["service"] = (svc_meta.index[tr], svc_meta.index[te])

# %% 3. BUILD THE SPLITS
split = {}
for m in MODELS:
    tr_i, te_i = idx[m]
    d = xy[m]
    split[m] = {"X_train": d["X"].loc[tr_i], "X_test": d["X"].loc[te_i],
                "y_train": d["y"].loc[tr_i] if d["y"] is not None else None,
                "y_test":  d["y"].loc[te_i] if d["y"] is not None else None,
                "meta_train": d["meta"].loc[tr_i], "meta_test": d["meta"].loc[te_i],
                "features": d["features"], "task": d["task"], "target": d["target"]}

# %% 4. CHECKS
rows = []
for m in MODELS:
    s = split[m]
    n_tr, n_te = len(s["X_train"]), len(s["X_test"])
    assert not (set(s["X_train"].index) & set(s["X_test"].index)), f"{m}: rows appear in both sets"
    assert n_tr + n_te == len(xy[m]["X"]), f"{m}: rows lost in the split"
    print(f"\n=== {m} ({s['task']}) === train {n_tr:,} ({n_tr/(n_tr+n_te):.1%}) | test {n_te:,} ({n_te/(n_tr+n_te):.1%})")
    note = ""
    if s["task"] == "classification":
        p_tr, p_te = s["y_train"].mean(), s["y_test"].mean()
        print(f"positive rate: train {p_tr:.2%} ({int(s['y_train'].sum()):,} cases) | test {p_te:.2%} ({int(s['y_test'].sum()):,} cases)")
        note = f"positives train {int(s['y_train'].sum())} / test {int(s['y_test'].sum())}"
        if int(s["y_test"].sum()) < 100: print("WARNING: fewer than 100 positive cases in test - metrics will be unstable")
    elif s["task"] == "regression":
        print(f"y mean: train {s['y_train'].mean():,.1f} | test {s['y_test'].mean():,.1f}   median: train {s['y_train'].median():,.1f} | test {s['y_test'].median():,.1f}")
        note = f"mean train {s['y_train'].mean():,.0f} / test {s['y_test'].mean():,.0f}"
    if m in ("churn", "clv", "segmentation"):
        print("customers in both train and test:", len(set(s["meta_train"]["customer_id"]) & set(s["meta_test"]["customer_id"])))
    if m == "clv":
        print("(same customers as churn:", s["X_train"].index.equals(split["churn"]["X_train"].index), ")")
    if m == "fraud":
        print(f"train dates {s['meta_train']['txn_date'].min().date()} to {s['meta_train']['txn_date'].max().date()} | "
              f"test dates {s['meta_test']['txn_date'].min().date()} to {s['meta_test']['txn_date'].max().date()}")
    if m == "service":
        overlap = len(set(s["meta_train"]["vehicle_id"]) & set(s["meta_test"]["vehicle_id"]))
        print("vehicles in both train and test:", overlap)
        assert overlap == 0
    rows.append({"model": m, "task": s["task"], "train_rows": n_tr, "test_rows": n_te, "split": {"churn": "stratified random", "clv": "shared customer split",
                 "segmentation": "shared customer split", "fraud": "time-based", "service": "grouped by vehicle"}[m], "note": note})
summary = pd.DataFrame(rows)
print("\nSUMMARY\n", summary.to_string(index=False))

# %% 5. SAVE
for m in MODELS:
    with open(f"{DATA}/split_{m}.pkl", "wb") as f:
        pickle.dump(split[m], f)
summary.to_csv(f"{DATA}/split_summary.csv", index=False)
print("\nSaved:", [f"split_{m}.pkl" for m in MODELS], "+ split_summary.csv")
```

---

## 13. Feature Scaling of Numerical Columns

Script: `python/10_feature_scaling.py`. **Everything is fitted on train only** and then applied to test.

Order of operations:
1. Keep only the model columns. Drop constants, duplicates and redundant dummy columns.
2. Fill missing values: median for numbers, most frequent value for 0/1 columns.
3. `log1p` on non-negative columns with skew above 1.5 (26 of the 41 continuous customer columns).
4. `StandardScaler` on continuous columns only. 0/1 columns stay as they are.

Saved per objective: `ready_<model>.pkl`, `scaler_<model>.joblib` and `prep_<model>.json` (the full recipe).

### Code: feature scaling (Python)

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 10: FEATURE SCALING OF NUMERICAL COLUMNS (Python)
# Reads the splits from step 9 (./abt_data). For each model, ALL fitting is done
# on the TRAIN set only, then applied to the test set:
#   1. drop useless columns   (service: branch_region_* copies country; constant and duplicate columns)
#   2. fill missing values    (median for numbers, most frequent for 0/1 columns)
#   3. log1p the very skewed non-negative columns (skew > 1.5)
#   4. StandardScaler on the continuous columns (0/1 columns are NOT scaled)
# The targets (y) are not touched. Needs: pandas numpy scikit-learn joblib.
# =====================================================================
# %% 0. SETTINGS
import os, json, pickle
import numpy as np, pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

DATA = "abt_data"
SKEW_LIMIT = 1.5
MODELS = ["churn", "clv", "segmentation", "fraud", "service"]
split = {m: pickle.load(open(f"{DATA}/split_{m}.pkl", "rb")) for m in MODELS}

# %% 1. FUNCTIONS (the same functions will be reused by the Streamlit app)
def is_binary(s):
    return set(s.dropna().unique()) <= {0, 1, 0.0, 1.0}

def fit_prep(X_train, model):
    X = X_train.copy()
    drop = [c for c in X.columns if model == "service" and c.startswith("branch_region_")]
    X = X.drop(columns=drop)
    const = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=const)
    dupes = X.columns[X.T.duplicated()].tolist()
    X = X.drop(columns=dupes)
    binary = [c for c in X.columns if is_binary(X[c])]
    cont = [c for c in X.columns if c not in binary]
    fill = {c: float(X[c].mode().iloc[0]) if c in binary else float(X[c].median()) for c in X.columns}
    Xf = X.fillna(pd.Series(fill))
    log_cols = [c for c in cont if Xf[c].min() >= 0 and Xf[c].skew() > SKEW_LIMIT]
    Xf[log_cols] = np.log1p(Xf[log_cols])
    scaler = StandardScaler().fit(Xf[cont])
    prep = {"dropped_redundant": drop, "dropped_constant": const, "dropped_duplicate": dupes,
            "features": X.columns.tolist(), "binary_cols": binary, "scale_cols": cont,
            "log_cols": log_cols, "fill_values": fill}
    return prep, scaler

def apply_prep(X, prep, scaler):
    X = X[prep["features"]].copy().astype(float)
    X = X.fillna(pd.Series(prep["fill_values"]))
    if prep["log_cols"]:
        X[prep["log_cols"]] = np.log1p(X[prep["log_cols"]].clip(lower=0))
    X[prep["scale_cols"]] = scaler.transform(X[prep["scale_cols"]])
    return X

# %% 2. FIT ON TRAIN, APPLY TO TRAIN AND TEST
ready, preps, scalers = {}, {}, {}
for m in MODELS:
    s = split[m]
    prep, scaler = fit_prep(s["X_train"], m)
    Xtr, Xte = apply_prep(s["X_train"], prep, scaler), apply_prep(s["X_test"], prep, scaler)
    preps[m], scalers[m] = prep, scaler
    ready[m] = {**{k: s[k] for k in ["y_train", "y_test", "meta_train", "meta_test", "task", "target"]},
                "X_train": Xtr, "X_test": Xte, "features": prep["features"]}

# %% 3. CHECKS
rows = []
for m in MODELS:
    s, p, r = split[m], preps[m], ready[m]
    Xtr, Xte = r["X_train"], r["X_test"]
    sc = p["scale_cols"]
    assert not Xtr.isna().any().any() and not Xte.isna().any().any(), f"{m}: missing values left"
    assert np.isfinite(Xtr.values).all() and np.isfinite(Xte.values).all(), f"{m}: infinite values"
    assert Xtr.index.equals(s["X_train"].index) and Xte.index.equals(s["X_test"].index), f"{m}: row order changed"
    again = apply_prep(s["X_train"], p, scalers[m])
    print(f"\n=== {m} ({s['task']}) ===")
    print(f"columns: {s['X_train'].shape[1]} -> {Xtr.shape[1]} | train {Xtr.shape[0]:,} rows | test {Xte.shape[0]:,} rows")
    print(f"dropped: redundant {p['dropped_redundant'][:6]}{'...' if len(p['dropped_redundant']) > 6 else ''} | constant {p['dropped_constant']} | duplicate {p['dropped_duplicate']}")
    print(f"missing cells filled (train): {s['X_train'].isna().sum().sum():,} | test: {s['X_test'].isna().sum().sum():,} -> now 0")
    print(f"0/1 columns left unscaled: {len(p['binary_cols'])} | scaled continuous columns: {len(sc)} | log1p applied to {len(p['log_cols'])}: {p['log_cols'][:8]}")
    print(f"scaled train: mean range [{Xtr[sc].mean().min():.3f}, {Xtr[sc].mean().max():.3f}] std range [{Xtr[sc].std().min():.3f}, {Xtr[sc].std().max():.3f}]")
    print(f"scaled test : mean range [{Xte[sc].mean().min():.3f}, {Xte[sc].mean().max():.3f}] std range [{Xte[sc].std().min():.3f}, {Xte[sc].std().max():.3f}]  (close to train, not exactly 0/1: that is correct)")
    print("recipe reproduces the scaled train set:", bool(np.allclose(again.values, Xtr.values)))
    rows.append({"model": m, "features_in": s["X_train"].shape[1], "features_out": Xtr.shape[1], "scaled_cols": len(sc),
                 "binary_cols": len(p["binary_cols"]), "log_cols": len(p["log_cols"]), "train_rows": len(Xtr), "test_rows": len(Xte)})
summary = pd.DataFrame(rows)
print("\nSUMMARY\n", summary.to_string(index=False))

# %% 4. SAVE
for m in MODELS:
    with open(f"{DATA}/ready_{m}.pkl", "wb") as f:
        pickle.dump(ready[m], f)
    joblib.dump(scalers[m], f"{DATA}/scaler_{m}.joblib")
    with open(f"{DATA}/prep_{m}.json", "w") as f:
        json.dump(preps[m], f, indent=2)
summary.to_csv(f"{DATA}/scaling_summary.csv", index=False)
print("\nSaved: ready_<model>.pkl, scaler_<model>.joblib, prep_<model>.json for", MODELS)
```

---

## 14. Train Model

Scripts: `python/11_train_<objective>.py`, plus `python/11b_train_charts.py` for the training charts. Five candidate models for each classification and regression objective:

| Classification (churn, fraud) | Regression (lifetime value, service cost) |
|---|---|
| Logistic Regression | Ridge Regression |
| KNN | KNN |
| Decision Tree | Decision Tree |
| Random Forest | Random Forest |
| Gradient Boosting | Gradient Boosting |

**How the champion is chosen**
- **Cross-validation inside the train set only**, 5 folds. Churn is stratified, fraud uses time-ordered folds, service cost is grouped by vehicle.
- Classification: **class weights** for imbalance, **PR AUC** as the deciding metric. The decision threshold is also picked on train cross-validation.
- Regression: the target is highly skewed, so every model learns `log1p(target)` and predictions are converted back to dollars. The deciding metric is **MAE**.
- Every model is compared with a **no-skill baseline** (the base rate, or the median).
- **Segmentation:** K-Means with k from 3 to 7, chosen on silhouette, Davies-Bouldin, Calinski-Harabasz and stability across reruns. Every segment must hold at least 5% of customers.

**Champions:** Logistic Regression (churn, 73 features), Logistic Regression (fraud, 48), K-Means k = 3 (24), Ridge Regression (lifetime value, 73), Gradient Boosting (service cost, 43).

### Charts

#### Objective 1: customer churn

<p align="center">
  <img src="images/churn_11_1_model_comparison.png" alt="Churn: five candidate models in cross-validation, train against cross-validated score" width="100%">
</p>
<p align="center"><em>Churn: five candidate models in cross-validation, train against cross-validated score</em></p>

#### Objective 2: fraud detection

<p align="center">
  <img src="images/fraud_11_1_model_comparison.png" alt="Fraud: five candidate models in cross-validation, train against cross-validated score" width="100%">
</p>
<p align="center"><em>Fraud: five candidate models in cross-validation, train against cross-validated score</em></p>

#### Objective 3: customer segmentation

<p align="center">
  <img src="images/segmentation_11_1_choosing_k.png" alt="Segmentation: choosing the number of segments (k) on train data" width="100%">
</p>
<p align="center"><em>Segmentation: choosing the number of segments (k) on train data</em></p>

<p align="center">
  <img src="images/segmentation_11_2_segments_overview.png" alt="Segmentation: the three segments at a glance" width="100%">
</p>
<p align="center"><em>Segmentation: the three segments at a glance</em></p>

#### Objective 4: customer lifetime value

<p align="center">
  <img src="images/clv_11_1_model_comparison.png" alt="Lifetime value: five candidate models in cross-validation, train against cross-validated score" width="100%">
</p>
<p align="center"><em>Lifetime value: five candidate models in cross-validation, train against cross-validated score</em></p>

#### Objective 5: service cost

<p align="center">
  <img src="images/service_11_1_model_comparison.png" alt="Service cost: five candidate models in cross-validation, train against cross-validated score" width="100%">
</p>
<p align="center"><em>Service cost: five candidate models in cross-validation, train against cross-validated score</em></p>

### Code: train model (Python)

#### Objective 1: customer churn

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 11: TRAIN MODEL - OBJECTIVE 1: CUSTOMER CHURN
# Task: classification. Target: churned (1 = customer left, 0 = retained).
# Reads ready_churn.pkl from step 10 (./abt_data) and trains 5 classifiers on the
# TRAIN set only. The test set is NOT touched here: predicting x_test and the
# evaluation are the next steps.
# To choose between algorithms fairly, each one is also cross-validated INSIDE the
# training set (5 stratified folds). The best one by PR AUC is marked as the champion.
# Needs: pandas numpy scikit-learn joblib.
# =====================================================================
# %% 0. SETTINGS
import time, pickle, warnings
import numpy as np, pandas as pd
import joblib
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
DATA, SEED = "abt_data", 42

# %% 1. LOAD THE TRAIN DATA (scaled and imputed in step 10)
d = pickle.load(open(f"{DATA}/ready_churn.pkl", "rb"))
X_train, y_train = d["X_train"], d["y_train"]
print(f"train: {X_train.shape[0]:,} customers x {X_train.shape[1]} features | churn rate {y_train.mean():.2%}")

# %% 2. THE CANDIDATE MODELS (churn is 31% of customers, so classes are weighted to count the minority fairly)
MODELS = {
    "Logistic Regression": LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, random_state=SEED),
    "KNN":                 KNeighborsClassifier(n_neighbors=35, weights="distance", n_jobs=-1),
    "Decision Tree":       DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, class_weight="balanced", random_state=SEED),
    "Random Forest":       RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features="sqrt",
                                                  class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=SEED),
}

def fit_one(model, X, y):
    """Gradient Boosting has no class_weight option, so it gets balanced sample weights instead."""
    if isinstance(model, GradientBoostingClassifier):
        return model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return model.fit(X, y)

# %% 3. CROSS-VALIDATE INSIDE THE TRAIN SET, THEN FIT EACH MODEL ON ALL OF TRAIN
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows, fitted = [], {}
for name, model in MODELS.items():
    t0 = time.time()
    oof = np.zeros(len(y_train))
    for tr, va in cv.split(X_train, y_train):
        m = fit_one(clone(model), X_train.iloc[tr], y_train.iloc[tr])
        oof[va] = m.predict_proba(X_train.iloc[va])[:, 1]
    fold_pr = [average_precision_score(y_train.iloc[va], oof[va]) for _, va in cv.split(X_train, y_train)]
    final = fit_one(clone(model), X_train, y_train)
    p_fit = final.predict_proba(X_train)[:, 1]
    fitted[name] = final
    rows.append({"model": name,
                 "cv_roc_auc": roc_auc_score(y_train, oof), "cv_pr_auc": average_precision_score(y_train, oof),
                 "cv_pr_auc_std": np.std(fold_pr), "train_pr_auc": average_precision_score(y_train, p_fit),
                 "seconds": round(time.time() - t0, 1)})
    print(f"trained {name:20s} cv PR AUC {rows[-1]['cv_pr_auc']:.3f} | cv ROC AUC {rows[-1]['cv_roc_auc']:.3f} | {rows[-1]['seconds']}s")

# %% 4. COMPARE AND PICK THE CHAMPION
res = pd.DataFrame(rows).sort_values("cv_pr_auc", ascending=False).reset_index(drop=True)
res["overfit_gap"] = res["train_pr_auc"] - res["cv_pr_auc"]
champion = res.loc[0, "model"]
print(f"\nbaseline PR AUC for a model with no skill = churn rate = {y_train.mean():.3f}")
print(res.round(3).to_string(index=False))
print(f"\nChampion (best cross-validated PR AUC): {champion}")

# %% 5. SAVE
joblib.dump({"models": fitted, "champion": champion, "features": X_train.columns.tolist()}, f"{DATA}/churn_models.joblib")
res.to_csv(f"{DATA}/churn_training_summary.csv", index=False)
print("Saved: churn_models.joblib (all 5 trained models + champion + feature list), churn_training_summary.csv")
```

#### Objective 2: fraud detection

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 11: TRAIN MODEL - OBJECTIVE 2: FRAUD DETECTION
# Task: classification. Target: is_fraud (1 = fraudulent transaction).
# Reads ready_fraud.pkl from step 10 (./abt_data) and trains 5 classifiers on the
# TRAIN period only (transactions up to 2025-06-30). The test period is NOT touched.
# Fraud is rare (about 2.4%), so every model is class-weighted, and the algorithms
# are compared inside the training period with TIME-ORDERED cross-validation:
# each fold trains on the earlier transactions and is scored on the next block,
# which is how the model will really be used (predict tomorrow from yesterday).
# Needs: pandas numpy scikit-learn joblib. Expect several minutes to run.
# =====================================================================
# %% 0. SETTINGS
import time, pickle, warnings
import numpy as np, pandas as pd
import joblib
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
DATA, SEED, N_BLOCKS = "abt_data", 42, 6

# %% 1. LOAD THE TRAIN DATA (scaled and imputed in step 10)
d = pickle.load(open(f"{DATA}/ready_fraud.pkl", "rb"))
X_train, y_train, dates = d["X_train"], d["y_train"], d["meta_train"]["txn_date"]
print(f"train: {X_train.shape[0]:,} transactions x {X_train.shape[1]} features | "
      f"fraud cases {int(y_train.sum()):,} ({y_train.mean():.2%}) | {dates.min().date()} to {dates.max().date()}")

# %% 2. THE CANDIDATE MODELS (all weighted so the rare fraud class counts)
MODELS = {
    "Logistic Regression": LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, random_state=SEED),
    "KNN":                 KNeighborsClassifier(n_neighbors=50, weights="distance", n_jobs=-1),
    "Decision Tree":       DecisionTreeClassifier(max_depth=6, min_samples_leaf=100, class_weight="balanced", random_state=SEED),
    "Random Forest":       RandomForestClassifier(n_estimators=250, min_samples_leaf=30, max_features="sqrt",
                                                  class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=3, subsample=0.8, random_state=SEED),
}

def fit_one(model, X, y):
    """Gradient Boosting has no class_weight option, so it gets balanced sample weights instead."""
    if isinstance(model, GradientBoostingClassifier):
        return model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return model.fit(X, y)

def recall_in_top(y, p, frac):
    """Share of all fraud caught when only the top `frac` highest-scored transactions are reviewed."""
    k = max(1, int(len(y) * frac))
    return float(np.asarray(y)[np.argsort(-p)[:k]].sum() / max(1, np.sum(y)))

# %% 3. TIME-ORDERED FOLDS INSIDE THE TRAIN PERIOD
order = np.argsort(dates.values, kind="stable")
blocks = np.array_split(order, N_BLOCKS)
folds = [(np.concatenate(blocks[:k]), blocks[k]) for k in range(1, N_BLOCKS)]    # train on blocks 0..k-1, score block k
print(f"{len(folds)} time-ordered folds; each scores a later block than it trains on")

# %% 4. CROSS-VALIDATE, THEN FIT EACH MODEL ON ALL OF TRAIN
rows, fitted = [], {}
for name, model in MODELS.items():
    t0 = time.time()
    oof_idx, oof_p, fold_pr = [], [], []
    for tr, va in folds:
        m = fit_one(clone(model), X_train.iloc[tr], y_train.iloc[tr])
        p = m.predict_proba(X_train.iloc[va])[:, 1]
        oof_idx.append(va); oof_p.append(p)
        if y_train.iloc[va].sum() > 0: fold_pr.append(average_precision_score(y_train.iloc[va], p))
    idx, p = np.concatenate(oof_idx), np.concatenate(oof_p)
    yv = y_train.iloc[idx].values
    final = fit_one(clone(model), X_train, y_train)
    fitted[name] = final
    rows.append({"model": name, "cv_pr_auc": average_precision_score(yv, p), "cv_pr_auc_std": float(np.std(fold_pr)),
                 "cv_roc_auc": roc_auc_score(yv, p), "cv_recall_top2pct": recall_in_top(yv, p, 0.02),
                 "train_pr_auc": average_precision_score(y_train, final.predict_proba(X_train)[:, 1]),
                 "seconds": round(time.time() - t0, 1)})
    r = rows[-1]
    print(f"trained {name:20s} cv PR AUC {r['cv_pr_auc']:.3f} | cv ROC AUC {r['cv_roc_auc']:.3f} | fraud caught in top 2%: {r['cv_recall_top2pct']:.1%} | {r['seconds']}s")

# %% 5. COMPARE AND PICK THE CHAMPION
res = pd.DataFrame(rows).sort_values("cv_pr_auc", ascending=False).reset_index(drop=True)
res["overfit_gap"] = res["train_pr_auc"] - res["cv_pr_auc"]
champion = res.loc[0, "model"]
print(f"\nbaseline PR AUC for a model with no skill = fraud rate = {y_train.mean():.3f}  (random review of 2% of transactions catches about 2% of fraud)")
print(res.round(3).to_string(index=False))
print(f"\nChampion (best time-ordered cross-validated PR AUC): {champion}")

# %% 6. SAVE
joblib.dump({"models": fitted, "champion": champion, "features": X_train.columns.tolist()}, f"{DATA}/fraud_models.joblib")
res.to_csv(f"{DATA}/fraud_training_summary.csv", index=False)
print("Saved: fraud_models.joblib (all 5 trained models + champion + feature list), fraud_training_summary.csv")
```

#### Objective 3: customer segmentation

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 11: TRAIN MODEL - OBJECTIVE 3: CUSTOMER SEGMENTATION
# Task: clustering (K-Means). There is no target: the model groups customers by behaviour.
# Reads ready_segmentation.pkl from step 10 (./abt_data) and uses the TRAIN customers only
# (the test customers are assigned to a segment in the next step).
# How the number of segments (k) is chosen, for k = 2 to 8, all on the train customers:
#   elbow (inertia), silhouette, Davies-Bouldin, Calinski-Harabasz, and STABILITY
#   (re-cluster 10 random 80% samples and check the same customers stay together).
# Rule: k must be 3 to 7 (fewer is not a segmentation, more is hard to act on), and every
# segment must hold at least 5% of customers. Within that, the best average rank wins.
# Set K_OVERRIDE to force a number. Needs: pandas numpy scikit-learn matplotlib joblib.
# =====================================================================
# %% 0. SETTINGS
import pickle, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

warnings.filterwarnings("ignore")
DATA, OUT, SEED, SHOW = "abt_data", "segmentation_outputs", 42, True
K_RANGE, K_MIN, K_MAX, MIN_SHARE, N_STAB, K_OVERRIDE = range(2, 9), 3, 7, 0.05, 10, None
import os; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
BLUE, RED, GREY = "#2F6DB5", "#C0392B", "#9AA5B1"

def save(name):
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight")
    if SHOW: plt.show()
    plt.close()

# %% 1. LOAD THE TRAIN CUSTOMERS (log-transformed, imputed and scaled in step 10)
d = pickle.load(open(f"{DATA}/ready_segmentation.pkl", "rb"))
X = d["X_train"]
print(f"train: {X.shape[0]:,} customers x {X.shape[1]} behaviour features")

# %% 2. TRY EVERY k
rng = np.random.default_rng(SEED)
rows = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
    labels = km.labels_
    # stability: cluster random 80% samples, assign ALL customers to those centres, compare with the main labelling
    aris = []
    for s in range(N_STAB):
        sub = rng.choice(len(X), int(0.8 * len(X)), replace=False)
        kms = KMeans(n_clusters=k, n_init=5, random_state=SEED + s).fit(X.iloc[sub])
        aris.append(adjusted_rand_score(labels, kms.predict(X)))
    rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, labels),
                 "davies_bouldin": davies_bouldin_score(X, labels), "calinski_harabasz": calinski_harabasz_score(X, labels),
                 "stability_ari": float(np.mean(aris)), "smallest_cluster_%": 100 * np.bincount(labels).min() / len(labels)})
    r = rows[-1]
    print(f"k={k}: silhouette {r['silhouette']:.3f} | Davies-Bouldin {r['davies_bouldin']:.2f} | stability {r['stability_ari']:.2f} | smallest segment {r['smallest_cluster_%']:.1f}%")
res = pd.DataFrame(rows)

# %% 3. CHOOSE k
ok = res[(res.k >= K_MIN) & (res.k <= K_MAX) & (res["smallest_cluster_%"] >= MIN_SHARE * 100)].copy()
if ok.empty: ok = res[res.k >= K_MIN].copy()
ok["rank_score"] = (ok.silhouette.rank() + ok.calinski_harabasz.rank() + (-ok.davies_bouldin).rank() + ok.stability_ari.rank())
best_k = int(ok.sort_values(["rank_score", "silhouette"], ascending=False).iloc[0]["k"])
k_final = K_OVERRIDE or best_k
print("\n", res.round(3).to_string(index=False))
print(f"\nChosen k = {k_final}" + (" (manual override)" if K_OVERRIDE else " (best average rank among k = 3 to 7 with every segment at least 5%)"))
print("Reading guide: silhouette above 0.25 is usable on behaviour data, above 0.5 is strong. Stability above 0.8 means the segments are reliable.")

# %% 4. TRAIN THE FINAL MODEL
km = KMeans(n_clusters=k_final, n_init=50, random_state=SEED).fit(X)
labels = pd.Series(km.labels_, index=X.index, name="segment")
sizes = labels.value_counts().sort_index()
print("\nSegment sizes (train customers)")
print(pd.DataFrame({"customers": sizes, "share_%": (100 * sizes / len(labels)).round(1)}).to_string())
print(f"final silhouette {silhouette_score(X, labels):.3f} | Davies-Bouldin {davies_bouldin_score(X, labels):.2f}")

# %% 5. CHARTS
fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
for a, col, title in zip(ax, ["inertia", "silhouette", "davies_bouldin", "stability_ari"],
                         ["Elbow: inertia (lower)", "Silhouette (higher)", "Davies-Bouldin (lower)", "Stability ARI (higher)"]):
    a.plot(res.k, res[col], "o-", color=BLUE); a.axvline(k_final, color=RED, ls="--", lw=1); a.set_title(title); a.set_xlabel("number of segments k")
save("11_1_choosing_k")

pca = PCA(n_components=2, random_state=SEED).fit(X)
P = pca.transform(X)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
sc = ax[0].scatter(P[:, 0], P[:, 1], c=labels, cmap="tab10", s=8, alpha=.6)
ax[0].set_title(f"Customers in 2D (PCA, {100 * pca.explained_variance_ratio_.sum():.0f}% of variance shown)"); ax[0].set_xlabel("component 1"); ax[0].set_ylabel("component 2")
ax[0].legend(*sc.legend_elements(), title="segment", fontsize=8)
ax[1].bar(sizes.index.astype(str), 100 * sizes.values / sizes.sum(), color=BLUE); ax[1].set_title("Segment size (% of customers)"); ax[1].set_xlabel("segment")
save("11_2_segments_overview")

# %% 6. SAVE
joblib.dump({"model": km, "k": k_final, "features": X.columns.tolist()}, f"{DATA}/segmentation_model.joblib")
labels.to_frame().join(d["meta_train"]).to_pickle(f"{DATA}/segmentation_train_labels.pkl")
res.to_csv(f"{DATA}/segmentation_k_selection.csv", index=False)
print("\nSaved: segmentation_model.joblib (K-Means + k + feature list), segmentation_train_labels.pkl, segmentation_k_selection.csv, 2 charts in", OUT)
```

#### Objective 4: customer lifetime value

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 11: TRAIN MODEL - OBJECTIVE 4: CUSTOMER LIFETIME VALUE
# Task: regression. Target: next_12m_spend_usd (what a customer will spend in the next 12 months).
# Reads ready_clv.pkl from step 10 (./abt_data) and trains 5 regressors on the TRAIN customers only.
# The test customers are NOT touched here: predicting x_test and the evaluation are next steps.
# The target is very skewed (a few big spenders), so every model learns log(1 + spend) and its
# predictions are converted back to dollars. Errors below are therefore in real dollars.
# Models are compared with 5-fold cross-validation INSIDE the train set. The best one by MAE
# (average dollar error) is marked as the champion.
# Needs: pandas numpy scikit-learn joblib.
# =====================================================================
# %% 0. SETTINGS
import time, pickle, warnings
import numpy as np, pandas as pd
import joblib
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

warnings.filterwarnings("ignore")
DATA, SEED = "abt_data", 42

# %% 1. LOAD THE TRAIN DATA (scaled and imputed in step 10)
with open(f"{DATA}/ready_clv.pkl", "rb") as f:
    d = pickle.load(f)
X_train, y_train = d["X_train"], d["y_train"]
print(f"train: {X_train.shape[0]:,} customers x {X_train.shape[1]} features")
print(f"next_12m_spend_usd: mean {y_train.mean():,.0f} | median {y_train.median():,.0f} | max {y_train.max():,.0f} | skew {y_train.skew():.2f} -> the models learn log(1 + spend)")

# %% 2. THE CANDIDATE MODELS (each wrapped so it learns log(1 + spend) and predicts dollars)
def log_target(model):
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)

MODELS = {
    "Ridge Regression":  log_target(Ridge(alpha=10.0)),
    "KNN":               log_target(KNeighborsRegressor(n_neighbors=25, weights="distance", n_jobs=-1)),
    "Decision Tree":     log_target(DecisionTreeRegressor(max_depth=6, min_samples_leaf=30, random_state=SEED)),
    "Random Forest":     log_target(RandomForestRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.5, n_jobs=-1, random_state=SEED)),
    "Gradient Boosting": log_target(GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=SEED)),
}
BASELINE = DummyRegressor(strategy="median")                  # "predict the typical customer for everyone"

def metrics(y, p):
    p = np.clip(p, 0, None)
    return {"mae": mean_absolute_error(y, p), "rmse": float(np.sqrt(mean_squared_error(y, p))), "median_ae": median_absolute_error(y, p), "r2": r2_score(y, p)}

# %% 3. CROSS-VALIDATE INSIDE THE TRAIN SET, THEN FIT EACH MODEL ON ALL OF TRAIN
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
folds = list(cv.split(X_train))

def cross_validate(model):
    oof = np.zeros(len(y_train)); fold_mae = []
    for tr, va in folds:
        m = clone(model).fit(X_train.iloc[tr], y_train.iloc[tr])
        oof[va] = m.predict(X_train.iloc[va]); fold_mae.append(mean_absolute_error(y_train.iloc[va], np.clip(oof[va], 0, None)))
    return oof, fold_mae

base_oof, _ = cross_validate(BASELINE)
base = metrics(y_train, base_oof)
print(f"\nbaseline (predict the median spend for everyone): MAE {base['mae']:,.0f} | RMSE {base['rmse']:,.0f} | R2 {base['r2']:.3f}")

rows, fitted = [], {}
for name, model in MODELS.items():
    t0 = time.time()
    oof, fold_mae = cross_validate(model)
    final = clone(model).fit(X_train, y_train)
    fitted[name] = final
    cvm, trm = metrics(y_train, oof), metrics(y_train, final.predict(X_train))
    rows.append({"model": name, "cv_mae": cvm["mae"], "cv_mae_std": float(np.std(fold_mae)), "cv_rmse": cvm["rmse"], "cv_median_ae": cvm["median_ae"],
                 "cv_r2": cvm["r2"], "train_mae": trm["mae"], "seconds": round(time.time() - t0, 1)})
    print(f"trained {name:18s} cv MAE {cvm['mae']:,.0f} | cv RMSE {cvm['rmse']:,.0f} | cv R2 {cvm['r2']:.3f} | {rows[-1]['seconds']}s")

# %% 4. COMPARE AND PICK THE CHAMPION
res = pd.DataFrame(rows).sort_values("cv_mae").reset_index(drop=True)
res["overfit_gap_mae"] = res["cv_mae"] - res["train_mae"]
res["mae_vs_baseline_%"] = 100 * (res["cv_mae"] / base["mae"] - 1)
champion = res.loc[0, "model"]
print("\n", res.round(3).to_string(index=False))
print(f"\nChampion (lowest cross-validated MAE): {champion}")
print("Reading guide: MAE = the average dollar error per customer (lower is better). R2 = share of the variation explained (1 is perfect, 0 is no better than the average).")

# %% 5. SAVE
joblib.dump({"models": fitted, "champion": champion, "features": X_train.columns.tolist(), "target_transform": "log1p"}, f"{DATA}/clv_models.joblib")
res.to_csv(f"{DATA}/clv_training_summary.csv", index=False)
print("Saved: clv_models.joblib (all 5 trained models + champion + feature list), clv_training_summary.csv")
```

#### Objective 5: service cost

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 11: TRAIN MODEL - OBJECTIVE 5: SERVICE COST
# Task: regression. Target: total_cost_usd (what a service job will cost the customer).
# Reads ready_service.pkl from step 10 (./abt_data) and trains 5 regressors on the TRAIN service jobs only.
# The test jobs are NOT touched here: predicting x_test and the evaluation are next steps.
# The target is right-skewed (a few very expensive repairs), so every model learns log(1 + cost) and its
# predictions are converted back to dollars. Errors below are therefore in real dollars.
# Models are compared with 5-fold cross-validation INSIDE the train set, GROUPED BY VEHICLE: all jobs of a
# vehicle stay in the same fold, matching how train and test were split (a vehicle is never in both).
# The best model by MAE (average dollar error per job) is marked as the champion.
# Needs: pandas numpy scikit-learn joblib.
# =====================================================================
# %% 0. SETTINGS
import time, pickle, warnings
import numpy as np, pandas as pd
import joblib
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

warnings.filterwarnings("ignore")
DATA, SEED = "abt_data", 42

# %% 1. LOAD THE TRAIN DATA (scaled and imputed in step 10)
with open(f"{DATA}/ready_service.pkl", "rb") as f:
    d = pickle.load(f)
X_train, y_train, groups = d["X_train"], d["y_train"], d["meta_train"]["vehicle_id"].values
print(f"train: {X_train.shape[0]:,} service jobs ({len(set(groups)):,} vehicles) x {X_train.shape[1]} features")
print(f"total_cost_usd: mean {y_train.mean():,.0f} | median {y_train.median():,.0f} | max {y_train.max():,.0f} | skew {y_train.skew():.2f} -> the models learn log(1 + cost)")

# %% 2. THE CANDIDATE MODELS (each wrapped so it learns log(1 + cost) and predicts dollars)
def log_target(model):
    return TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)

MODELS = {
    "Ridge Regression":  log_target(Ridge(alpha=10.0)),
    "KNN":               log_target(KNeighborsRegressor(n_neighbors=25, weights="distance", n_jobs=-1)),
    "Decision Tree":     log_target(DecisionTreeRegressor(max_depth=6, min_samples_leaf=30, random_state=SEED)),
    "Random Forest":     log_target(RandomForestRegressor(n_estimators=300, min_samples_leaf=10, max_features=0.5, n_jobs=-1, random_state=SEED)),
    "Gradient Boosting": log_target(GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=SEED)),
}
BASELINE = DummyRegressor(strategy="median")                  # "predict the typical job for everyone"

def metrics(y, p):
    p = np.clip(p, 0, None)
    return {"mae": mean_absolute_error(y, p), "rmse": float(np.sqrt(mean_squared_error(y, p))), "median_ae": median_absolute_error(y, p), "r2": r2_score(y, p)}

# %% 3. CROSS-VALIDATE INSIDE THE TRAIN SET, THEN FIT EACH MODEL ON ALL OF TRAIN
folds = list(GroupKFold(n_splits=5).split(X_train, y_train, groups))   # a vehicle never sits in two folds

def cross_validate(model):
    oof = np.zeros(len(y_train)); fold_mae = []
    for tr, va in folds:
        m = clone(model).fit(X_train.iloc[tr], y_train.iloc[tr])
        oof[va] = m.predict(X_train.iloc[va]); fold_mae.append(mean_absolute_error(y_train.iloc[va], np.clip(oof[va], 0, None)))
    return oof, fold_mae

base_oof, _ = cross_validate(BASELINE)
base = metrics(y_train, base_oof)
print(f"\nbaseline (predict the median job cost for everyone): MAE {base['mae']:,.0f} | RMSE {base['rmse']:,.0f} | R2 {base['r2']:.3f}")

rows, fitted = [], {}
for name, model in MODELS.items():
    t0 = time.time()
    oof, fold_mae = cross_validate(model)
    final = clone(model).fit(X_train, y_train)
    fitted[name] = final
    cvm, trm = metrics(y_train, oof), metrics(y_train, final.predict(X_train))
    rows.append({"model": name, "cv_mae": cvm["mae"], "cv_mae_std": float(np.std(fold_mae)), "cv_rmse": cvm["rmse"], "cv_median_ae": cvm["median_ae"],
                 "cv_r2": cvm["r2"], "train_mae": trm["mae"], "seconds": round(time.time() - t0, 1)})
    print(f"trained {name:18s} cv MAE {cvm['mae']:,.0f} | cv RMSE {cvm['rmse']:,.0f} | cv R2 {cvm['r2']:.3f} | {rows[-1]['seconds']}s")

# %% 4. COMPARE AND PICK THE CHAMPION
res = pd.DataFrame(rows).sort_values("cv_mae").reset_index(drop=True)
res["overfit_gap_mae"] = res["cv_mae"] - res["train_mae"]
res["mae_vs_baseline_%"] = 100 * (res["cv_mae"] / base["mae"] - 1)
champion = res.loc[0, "model"]
print("\n", res.round(3).to_string(index=False))
print(f"\nChampion (lowest cross-validated MAE): {champion}")
print("Reading guide: MAE = the average dollar error per job (lower is better). R2 = share of the variation explained (1 is perfect, 0 is no better than the average).")

# %% 5. SAVE
joblib.dump({"models": fitted, "champion": champion, "features": X_train.columns.tolist(), "target_transform": "log1p"}, f"{DATA}/service_models.joblib")
res.to_csv(f"{DATA}/service_training_summary.csv", index=False)
print("Saved: service_models.joblib (all 5 trained models + champion + feature list), service_training_summary.csv")
```

#### Training charts for objectives 1, 2, 4 and 5

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 11b: TRAINING CHARTS (objectives 1, 2, 4, 5)
# Draws one chart per objective from the training summaries saved by step 11:
#   left   = cross-validated score of the five candidate models (champion highlighted)
#   middle = train score vs cross-validated score (the gap shows over-fitting)
#   right  = a second view of the same models (ROC AUC for classification, R2 for regression)
# Reads abt_data/<objective>_training_summary.csv and <objective>_models.joblib. Changes nothing.
# Saves <objective>_outputs/11_1_model_comparison.png (segmentation already saves its own training charts in step 11).
# Needs: pandas matplotlib joblib scikit-learn. Set SHOW = False to save charts only.
# =====================================================================
import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

DATA, SHOW = "abt_data", True
BLUE, RED, GREY = "#2F6DB5", "#C0392B", "#9AA5B1"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})

# objective -> (title, task, main score, its spread, train score, second view, second label)
SPEC = {
    "churn":   ("Objective 1: customer churn",           "clf", "cv_pr_auc", "cv_pr_auc_std", "train_pr_auc", "cv_roc_auc", "cross-validated ROC AUC"),
    "fraud":   ("Objective 2: fraud detection",          "clf", "cv_pr_auc", "cv_pr_auc_std", "train_pr_auc", "cv_roc_auc", "cross-validated ROC AUC"),
    "clv":     ("Objective 4: customer lifetime value",  "reg", "cv_mae",    "cv_mae_std",    "train_mae",    "cv_r2",      "cross-validated R2"),
    "service": ("Objective 5: service cost",             "reg", "cv_mae",    "cv_mae_std",    "train_mae",    "cv_r2",      "cross-validated R2"),
}

for obj, (title, task, main, spread, train, second, second_label) in SPEC.items():
    f = f"{DATA}/{obj}_training_summary.csv"
    if not os.path.exists(f):
        print(f"skipped {obj}: {f} not found (run step 11 for this objective first)"); continue
    res = pd.read_csv(f)
    champion = joblib.load(f"{DATA}/{obj}_models.joblib")["champion"]
    res = res.sort_values(main, ascending=(task == "reg")).reset_index(drop=True)   # best model first
    colours = [RED if m == champion else BLUE for m in res["model"]]
    better = "higher is better" if task == "clf" else "lower is better"
    metric = "PR AUC" if task == "clf" else "MAE (USD)"

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].barh(res["model"], res[main], xerr=res[spread], color=colours, ecolor="#444", capsize=3)
    ax[0].invert_yaxis(); ax[0].set_xlabel(f"cross-validated {metric} ({better})")
    ax[0].set_title("Cross-validation on the train set\n(red = champion)")
    for i, (v, e) in enumerate(zip(res[main], res[spread])):
        ax[0].text(v + e, i, f"  {v:.3f}" if task == "clf" else f"  {v:,.0f}", va="center", fontsize=8)
    ax[0].margins(x=0.15)

    y = range(len(res)); h = 0.38
    ax[1].barh([i - h / 2 for i in y], res[train], height=h, color=GREY, label="train")
    ax[1].barh([i + h / 2 for i in y], res[main], height=h, color=colours, label="cross-validated")
    ax[1].set_yticks(list(y)); ax[1].set_yticklabels(res["model"]); ax[1].invert_yaxis()
    ax[1].set_xlabel(metric); ax[1].set_title("Train vs cross-validated\n(a big gap means over-fitting)"); ax[1].legend(fontsize=8, loc="lower right")

    ax[2].barh(res["model"], res[second], color=colours)
    ax[2].invert_yaxis(); ax[2].set_xlabel(second_label); ax[2].set_title("Second view of the same models")
    if task == "reg": ax[2].axvline(0, color="black", lw=.8)
    for i, v in enumerate(res[second]):
        ax[2].text(v, i, f" {v:.3f}", va="center", fontsize=8)

    plt.suptitle(f"{title}: five candidate models, champion = {champion}", y=1.03, fontsize=12)
    plt.tight_layout()
    os.makedirs(f"{obj}_outputs", exist_ok=True)
    plt.savefig(f"{obj}_outputs/11_1_model_comparison.png", dpi=150, bbox_inches="tight")
    if SHOW: plt.show()
    plt.close()
    print(f"saved {obj}_outputs/11_1_model_comparison.png (champion: {champion})")
```

---

## 15. Predict X_test

Scripts: `python/12_predict_<objective>.py`. Each script loads the saved models and predicts the test set only.

- Churn and fraud: a probability for every customer or transaction, plus a flag at the chosen threshold.
- Lifetime value and service cost: a dollar prediction, never below zero.
- Segmentation: a segment for each customer, plus how close each customer is to the next segment.
- No scores are calculated here. Predictions are saved to `.pkl` and `.csv` for the next steps.

### Code: predict x_test (Python)

#### Objective 1: customer churn

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 12: PREDICT x_test - OBJECTIVE 1: CUSTOMER CHURN
# Loads the 5 trained churn models (step 11) and predicts the test customers.
# For every model it stores the churn probability and the 0/1 prediction
# (threshold 0.5). It does NOT calculate any score: the evaluation is a later step.
# Needs: pandas numpy joblib.
# =====================================================================
# %% 0. SETTINGS
import pickle
import numpy as np, pandas as pd
import joblib

DATA, THRESHOLD = "abt_data", 0.5

# %% 1. LOAD THE TEST DATA AND THE TRAINED MODELS
d = pickle.load(open(f"{DATA}/ready_churn.pkl", "rb"))
X_test, y_test, meta_test = d["X_test"], d["y_test"], d["meta_test"]
bundle = joblib.load(f"{DATA}/churn_models.joblib")
models, champion, features = bundle["models"], bundle["champion"], bundle["features"]

assert list(X_test.columns) == features, "test columns differ from the training features"
assert not X_test.isna().any().any(), "missing values in X_test"
print(f"test set: {X_test.shape[0]:,} customers x {X_test.shape[1]} features | champion from step 11: {champion}")

# %% 2. PREDICT
pred = meta_test[["customer_id"]].copy()
pred["actual_churned"] = y_test.values
for name, model in models.items():
    proba = model.predict_proba(X_test)[:, 1]
    pred[f"{name}_proba"] = proba
    pred[f"{name}_pred"] = (proba >= THRESHOLD).astype(int)
pred["champion_proba"] = pred[f"{champion}_proba"]
pred["champion_pred"] = pred[f"{champion}_pred"]

# %% 3. LOOK AT THE PREDICTIONS (no scores yet)
assert len(pred) == len(X_test) and not pred.isna().any().any()
print(f"\nactual churn rate in test: {y_test.mean():.2%} ({int(y_test.sum()):,} of {len(y_test):,} customers)")
summary = pd.DataFrame([{"model": n,
                         "predicted_churners": int(pred[f"{n}_pred"].sum()),
                         "predicted_rate_%": round(100 * pred[f"{n}_pred"].mean(), 1),
                         "avg_probability": round(pred[f"{n}_proba"].mean(), 3),
                         "min_proba": round(pred[f"{n}_proba"].min(), 3),
                         "max_proba": round(pred[f"{n}_proba"].max(), 3)} for n in models])
print(summary.to_string(index=False))

print(f"\nFirst 10 test customers ({champion}):")
print(pred[["customer_id", "actual_churned", "champion_proba", "champion_pred"]].head(10).round(3).to_string(index=False))

bands = pd.cut(pred["champion_proba"], [0, .2, .4, .6, .8, 1.0], include_lowest=True)
print("\nCustomers by predicted churn probability band (champion):")
print(pred.groupby(bands, observed=True).agg(customers=("customer_id", "count"), actual_churn_rate=("actual_churned", "mean")).round(3).to_string())

# %% 4. SAVE
pred.to_pickle(f"{DATA}/churn_test_predictions.pkl")
pred.to_csv(f"{DATA}/churn_test_predictions.csv", index=False)
print("\nSaved: churn_test_predictions.pkl and churn_test_predictions.csv in", DATA)
```

#### Objective 2: fraud detection

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 12: PREDICT x_test - OBJECTIVE 2: FRAUD DETECTION
# Loads the 5 trained fraud models (step 11) and predicts the TEST PERIOD
# (transactions from 2025-07-01 onward, never seen in training).
# For every model it stores the fraud probability and the 0/1 prediction (threshold 0.5).
# For the champion it also flags the top 2% highest-risk transactions, because a
# fraud team has limited review capacity. It does NOT calculate any score: the
# evaluation is a later step. Needs: pandas numpy joblib.
# =====================================================================
# %% 0. SETTINGS
import pickle
import numpy as np, pandas as pd
import joblib

DATA, THRESHOLD, REVIEW_SHARE = "abt_data", 0.5, 0.02

# %% 1. LOAD THE TEST DATA AND THE TRAINED MODELS
d = pickle.load(open(f"{DATA}/ready_fraud.pkl", "rb"))
X_test, y_test, meta_test = d["X_test"], d["y_test"], d["meta_test"]
bundle = joblib.load(f"{DATA}/fraud_models.joblib")
models, champion, features = bundle["models"], bundle["champion"], bundle["features"]

assert list(X_test.columns) == features, "test columns differ from the training features"
assert not X_test.isna().any().any(), "missing values in X_test"
print(f"test set: {X_test.shape[0]:,} transactions x {X_test.shape[1]} features | "
      f"{meta_test['txn_date'].min().date()} to {meta_test['txn_date'].max().date()} | champion from step 11: {champion}")

# %% 2. PREDICT
pred = meta_test[["txn_id", "customer_id", "txn_date"]].copy()
pred["actual_fraud"] = y_test.values
for name, model in models.items():
    proba = model.predict_proba(X_test)[:, 1]
    pred[f"{name}_proba"] = proba
    pred[f"{name}_pred"] = (proba >= THRESHOLD).astype(int)
pred["champion_proba"] = pred[f"{champion}_proba"]
pred["champion_pred"] = pred[f"{champion}_pred"]
n_review = max(1, int(len(pred) * REVIEW_SHARE))
cut = pred["champion_proba"].sort_values(ascending=False).iloc[n_review - 1]
pred["champion_review_flag"] = (pred["champion_proba"] >= cut).astype(int)     # top 2% by risk
pred["champion_risk_rank_%"] = (pred["champion_proba"].rank(ascending=False, pct=True) * 100).round(2)

# %% 3. LOOK AT THE PREDICTIONS (no scores yet)
assert len(pred) == len(X_test) and not pred.isna().any().any()
print(f"\nactual fraud in test period: {y_test.mean():.2%} ({int(y_test.sum()):,} of {len(y_test):,} transactions)")
summary = pd.DataFrame([{"model": n,
                         "flagged_as_fraud": int(pred[f"{n}_pred"].sum()),
                         "flagged_rate_%": round(100 * pred[f"{n}_pred"].mean(), 2),
                         "avg_probability": round(pred[f"{n}_proba"].mean(), 3),
                         "max_proba": round(pred[f"{n}_proba"].max(), 3)} for n in models])
print(summary.to_string(index=False))
print(f"\nChampion review list (top {REVIEW_SHARE:.0%} by risk): {int(pred['champion_review_flag'].sum()):,} transactions")

print(f"\nFirst 10 highest-risk transactions ({champion}):")
print(pred.sort_values("champion_proba", ascending=False)[["txn_id", "txn_date", "actual_fraud", "champion_proba", "champion_pred"]]
      .head(10).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

bands = pd.cut(pred["champion_proba"], [0, .1, .3, .5, .7, 1.0], include_lowest=True)
print("\nTransactions by predicted fraud probability band (champion):")
print(pred.groupby(bands, observed=True).agg(transactions=("txn_id", "count"), actual_fraud_rate=("actual_fraud", "mean")).round(4).to_string())

# %% 4. SAVE
pred.to_pickle(f"{DATA}/fraud_test_predictions.pkl")
pred.to_csv(f"{DATA}/fraud_test_predictions.csv", index=False)
print("\nSaved: fraud_test_predictions.pkl and fraud_test_predictions.csv in", DATA)
```

#### Objective 3: customer segmentation

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 12: PREDICT x_test - OBJECTIVE 3: CUSTOMER SEGMENTATION
# Assigns every TEST customer to the nearest segment of the K-Means model trained in step 11.
# This is the "predict x_test" step for clustering: there is no right answer to compare with,
# so instead of a score it shows (a) how many test customers land in each segment, compared
# with train, (b) how close each customer sits to the centre of their segment, and (c) which
# customers are borderline between two segments. The quality checks come in step 14.
# Also writes ONE table with the segment of every customer (train + test), which is the table
# we will write back to SQL Server for Power BI. Needs: pandas numpy joblib.
# =====================================================================
# %% 0. SETTINGS AND LOAD
import pickle
import numpy as np, pandas as pd
import joblib

DATA = "abt_data"
with open(f"{DATA}/ready_segmentation.pkl", "rb") as f:
    d = pickle.load(f)
X_train, X_test, meta_test = d["X_train"], d["X_test"], d["meta_test"]
bundle = joblib.load(f"{DATA}/segmentation_model.joblib")
km, k, features = bundle["model"], bundle["k"], bundle["features"]
train_labels = pd.read_pickle(f"{DATA}/segmentation_train_labels.pkl")

assert list(X_test.columns) == features, "test columns differ from the training features"
assert not X_test.isna().any().any(), "missing values in X_test"
print(f"test set: {X_test.shape[0]:,} customers x {X_test.shape[1]} features | model: K-Means with k = {k}")

# %% 1. ASSIGN EACH TEST CUSTOMER TO THE NEAREST SEGMENT
def assign(X):
    dist = km.transform(X)                              # distance from each customer to every segment centre
    srt = np.sort(dist, axis=1)
    return pd.DataFrame({"segment": km.predict(X), "distance_to_centre": srt[:, 0],
                         "distance_to_next_segment": srt[:, 1], "margin": srt[:, 1] - srt[:, 0]}, index=X.index)

tr, te = assign(X_train), assign(X_test)
assert (tr["segment"].values == train_labels["segment"].values).all(), "train assignments differ from step 11"

# thresholds learned from TRAIN only
far_limit = tr["distance_to_centre"].quantile(0.95)       # beyond this = unusual customer
border_limit = tr["margin"].quantile(0.10)                # below this = borderline between two segments
te["unusual_customer"] = (te["distance_to_centre"] > far_limit).astype(int)
te["borderline"] = (te["margin"] < border_limit).astype(int)
tr["unusual_customer"] = (tr["distance_to_centre"] > far_limit).astype(int)
tr["borderline"] = (tr["margin"] < border_limit).astype(int)

# %% 2. LOOK AT THE ASSIGNMENTS (no scores)
share = pd.DataFrame({"train_customers": tr["segment"].value_counts().reindex(range(k), fill_value=0),
                      "test_customers": te["segment"].value_counts().reindex(range(k), fill_value=0)})
share["train_share_%"] = (100 * share.train_customers / share.train_customers.sum()).round(1)
share["test_share_%"] = (100 * share.test_customers / share.test_customers.sum()).round(1)
share["difference_pts"] = (share["test_share_%"] - share["train_share_%"]).round(1)
print("\nSegment sizes: train vs test (the shares should be close if the segments generalise)\n", share.to_string())
print(f"\nAverage distance to own segment centre: train {tr['distance_to_centre'].mean():.2f} | test {te['distance_to_centre'].mean():.2f}")
print(f"Unusual customers (further than the train 95th percentile): train {tr['unusual_customer'].mean():.1%} | test {te['unusual_customer'].mean():.1%}  (expect about 5%)")
print(f"Borderline customers (closest 10% margin in train):          train {tr['borderline'].mean():.1%} | test {te['borderline'].mean():.1%}  (expect about 10%)")

print("\nFirst 10 test customers")
out_te = meta_test[["customer_id"]].join(te)
print(out_te.head(10).to_string(index=False, float_format=lambda v: f"{v:.2f}"))

# %% 3. SAVE: TEST ASSIGNMENTS AND THE FULL CUSTOMER TABLE (train + test)
out_tr = d["meta_train"][["customer_id"]].join(tr)
out_te.assign(dataset="test").to_pickle(f"{DATA}/segmentation_test_assignments.pkl")
out_te.assign(dataset="test").to_csv(f"{DATA}/segmentation_test_assignments.csv", index=False)
full = pd.concat([out_tr.assign(dataset="train"), out_te.assign(dataset="test")]).sort_values("customer_id").reset_index(drop=True)
full.to_pickle(f"{DATA}/segmentation_all_customers.pkl")
full.to_csv(f"{DATA}/segmentation_all_customers.csv", index=False)
print(f"\nSaved: segmentation_test_assignments (.pkl/.csv) and segmentation_all_customers (.pkl/.csv, {len(full):,} customers) in {DATA}")
```

#### Objective 4: customer lifetime value

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 12: PREDICT x_test - OBJECTIVE 4: CUSTOMER LIFETIME VALUE
# Loads the 5 trained CLV models (step 11) and predicts next-12-month spend (in dollars)
# for the test customers. Predictions are never negative (clipped at 0).
# It does NOT calculate any score: the evaluation is a later step.
# Needs: pandas numpy joblib.
# =====================================================================
# %% 0. SETTINGS
import pickle
import numpy as np, pandas as pd
import joblib

DATA = "abt_data"

# %% 1. LOAD THE TEST DATA AND THE TRAINED MODELS
with open(f"{DATA}/ready_clv.pkl", "rb") as f:
    d = pickle.load(f)
X_test, y_test, meta_test = d["X_test"], d["y_test"], d["meta_test"]
bundle = joblib.load(f"{DATA}/clv_models.joblib")
models, champion, features = bundle["models"], bundle["champion"], bundle["features"]

assert list(X_test.columns) == features, "test columns differ from the training features"
assert not X_test.isna().any().any(), "missing values in X_test"
print(f"test set: {X_test.shape[0]:,} customers x {X_test.shape[1]} features | champion from step 11: {champion}")

# %% 2. PREDICT (each model returns dollars)
pred = meta_test[["customer_id"]].copy()
pred["actual_spend_usd"] = y_test.values
for name, model in models.items():
    pred[f"{name}_pred"] = np.clip(model.predict(X_test), 0, None)
pred["champion_pred"] = pred[f"{champion}_pred"]

# %% 3. LOOK AT THE PREDICTIONS (no scores yet)
assert len(pred) == len(X_test) and not pred.isna().any().any()
print(f"\nactual next-12-month spend in test: mean {y_test.mean():,.0f} | median {y_test.median():,.0f} | max {y_test.max():,.0f}")
summary = pd.DataFrame([{"model": n,
                         "avg_predicted": round(pred[f"{n}_pred"].mean(), 0),
                         "median_predicted": round(pred[f"{n}_pred"].median(), 0),
                         "min": round(pred[f"{n}_pred"].min(), 0),
                         "max": round(pred[f"{n}_pred"].max(), 0)} for n in models])
print(summary.to_string(index=False))

print(f"\nFirst 10 test customers ({champion}):")
show = pred[["customer_id", "actual_spend_usd", "champion_pred"]].head(10).copy()
show["difference"] = show["champion_pred"] - show["actual_spend_usd"]
print(show.round(0).to_string(index=False))

pred["value_band"] = pd.qcut(pred["champion_pred"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"], duplicates="drop")
print("\nCustomers by predicted value quartile (champion):")
print(pred.groupby("value_band", observed=True).agg(customers=("customer_id", "count"), avg_predicted=("champion_pred", "mean"), avg_actual=("actual_spend_usd", "mean")).round(0).to_string())

# %% 4. SAVE
pred = pred.drop(columns="value_band")
pred.to_pickle(f"{DATA}/clv_test_predictions.pkl")
pred.to_csv(f"{DATA}/clv_test_predictions.csv", index=False, float_format="%.2f")
print("\nSaved: clv_test_predictions.pkl and clv_test_predictions.csv in", DATA)
```

#### Objective 5: service cost

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 12: PREDICT x_test - OBJECTIVE 5: SERVICE COST
# Loads the 5 trained service cost models (step 11) and predicts the cost of each test service job (in dollars).
# Predictions are never negative (clipped at 0).
# It does NOT calculate any score: the evaluation is a later step.
# Needs: pandas numpy joblib.
# =====================================================================
# %% 0. SETTINGS
import pickle
import numpy as np, pandas as pd
import joblib

DATA = "abt_data"

# %% 1. LOAD THE TEST DATA AND THE TRAINED MODELS
with open(f"{DATA}/ready_service.pkl", "rb") as f:
    d = pickle.load(f)
X_test, y_test, meta_test = d["X_test"], d["y_test"], d["meta_test"]
bundle = joblib.load(f"{DATA}/service_models.joblib")
models, champion, features = bundle["models"], bundle["champion"], bundle["features"]

assert list(X_test.columns) == features, "test columns differ from the training features"
assert not X_test.isna().any().any(), "missing values in X_test"
print(f"test set: {X_test.shape[0]:,} service jobs x {X_test.shape[1]} features | champion from step 11: {champion}")

# %% 2. PREDICT (each model returns dollars)
pred = meta_test[["service_id", "vehicle_id"]].copy()
pred["actual_cost_usd"] = y_test.values
for name, model in models.items():
    pred[f"{name}_pred"] = np.clip(model.predict(X_test), 0, None)
pred["champion_pred"] = pred[f"{champion}_pred"]

# %% 3. LOOK AT THE PREDICTIONS (no scores yet)
assert len(pred) == len(X_test) and not pred.isna().any().any()
print(f"\nactual job cost in test: mean {y_test.mean():,.0f} | median {y_test.median():,.0f} | max {y_test.max():,.0f}")
summary = pd.DataFrame([{"model": n,
                         "avg_predicted": round(pred[f"{n}_pred"].mean(), 0),
                         "median_predicted": round(pred[f"{n}_pred"].median(), 0),
                         "min": round(pred[f"{n}_pred"].min(), 0),
                         "max": round(pred[f"{n}_pred"].max(), 0)} for n in models])
print(summary.to_string(index=False))

print(f"\nFirst 10 test service jobs ({champion}):")
show = pred[["service_id", "actual_cost_usd", "champion_pred"]].head(10).copy()
show["difference"] = show["champion_pred"] - show["actual_cost_usd"]
print(show.round(0).to_string(index=False))

pred["cost_band"] = pd.qcut(pred["champion_pred"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"], duplicates="drop")
print("\nService jobs by predicted cost quartile (champion):")
print(pred.groupby("cost_band", observed=True).agg(jobs=("service_id", "count"), avg_predicted=("champion_pred", "mean"), avg_actual=("actual_cost_usd", "mean")).round(0).to_string())

# %% 4. SAVE
pred = pred.drop(columns="cost_band")
pred.to_pickle(f"{DATA}/service_test_predictions.pkl")
pred.to_csv(f"{DATA}/service_test_predictions.csv", index=False, float_format="%.2f")
print("\nSaved: service_test_predictions.pkl and service_test_predictions.csv in", DATA)
```

---

## 16. Compare X and Y

Scripts: `python/13_compare_<objective>.py`.

- **Target:** actual against predicted, as plain counts for classification and as dollar distributions and quartiles for regression.
- **Features:** which features drive each model (tree importances and standardised coefficients side by side). A driver is only trusted when both methods agree.
- Charts show the outcome rate or median across the values of the top features, in original units.

### Charts

#### Objective 1: customer churn

<p align="center">
  <img src="images/churn_13_1_actual_vs_predicted.png" alt="Actual against predicted churn" width="100%">
</p>
<p align="center"><em>Actual against predicted churn</em></p>

<p align="center">
  <img src="images/churn_13_2_probability_by_outcome.png" alt="Predicted probability for churned and retained customers" width="100%">
</p>
<p align="center"><em>Predicted probability for churned and retained customers</em></p>

<p align="center">
  <img src="images/churn_13_3_feature_importance.png" alt="What drives churn: tree importance and standardised coefficients side by side" width="100%">
</p>
<p align="center"><em>What drives churn: tree importance and standardised coefficients side by side</em></p>

<p align="center">
  <img src="images/churn_13_4_top_features_vs_churn.png" alt="Churn rate across the values of the top drivers" width="100%">
</p>
<p align="center"><em>Churn rate across the values of the top drivers</em></p>

#### Objective 2: fraud detection

<p align="center">
  <img src="images/fraud_13_1_actual_vs_predicted.png" alt="Actual against predicted fraud" width="100%">
</p>
<p align="center"><em>Actual against predicted fraud</em></p>

<p align="center">
  <img src="images/fraud_13_2_probability_by_outcome.png" alt="Predicted probability for fraud and genuine payments" width="100%">
</p>
<p align="center"><em>Predicted probability for fraud and genuine payments</em></p>

<p align="center">
  <img src="images/fraud_13_3_feature_importance.png" alt="What drives fraud: tree importance and standardised coefficients side by side" width="100%">
</p>
<p align="center"><em>What drives fraud: tree importance and standardised coefficients side by side</em></p>

<p align="center">
  <img src="images/fraud_13_4_top_features_vs_fraud.png" alt="Fraud rate across the values of the top drivers" width="100%">
</p>
<p align="center"><em>Fraud rate across the values of the top drivers</em></p>

#### Objective 3: customer segmentation

<p align="center">
  <img src="images/segmentation_13_1_segment_profiles.png" alt="Segment profiles on the behaviour features" width="100%">
</p>
<p align="center"><em>Segment profiles on the behaviour features</em></p>

<p align="center">
  <img src="images/segmentation_13_2_segment_outcomes.png" alt="Spend and churn by segment" width="100%">
</p>
<p align="center"><em>Spend and churn by segment</em></p>

#### Objective 4: customer lifetime value

<p align="center">
  <img src="images/clv_13_1_actual_vs_predicted.png" alt="Actual against predicted next-12-month spend" width="100%">
</p>
<p align="center"><em>Actual against predicted next-12-month spend</em></p>

<p align="center">
  <img src="images/clv_13_2_quartiles.png" alt="Actual and predicted spend by quartile" width="100%">
</p>
<p align="center"><em>Actual and predicted spend by quartile</em></p>

<p align="center">
  <img src="images/clv_13_3_feature_importance.png" alt="What drives spend: tree importance and standardised coefficients side by side" width="100%">
</p>
<p align="center"><em>What drives spend: tree importance and standardised coefficients side by side</em></p>

<p align="center">
  <img src="images/clv_13_4_top_features_vs_spend.png" alt="Spend across the values of the top drivers" width="100%">
</p>
<p align="center"><em>Spend across the values of the top drivers</em></p>

#### Objective 5: service cost

<p align="center">
  <img src="images/service_13_1_actual_vs_predicted.png" alt="Actual against predicted service cost" width="100%">
</p>
<p align="center"><em>Actual against predicted service cost</em></p>

<p align="center">
  <img src="images/service_13_2_quartiles.png" alt="Actual and predicted cost by quartile" width="100%">
</p>
<p align="center"><em>Actual and predicted cost by quartile</em></p>

<p align="center">
  <img src="images/service_13_3_feature_importance.png" alt="What drives cost: tree importance and standardised coefficients side by side" width="100%">
</p>
<p align="center"><em>What drives cost: tree importance and standardised coefficients side by side</em></p>

<p align="center">
  <img src="images/service_13_4_top_features_vs_cost.png" alt="Cost across the values of the top drivers" width="100%">
</p>
<p align="center"><em>Cost across the values of the top drivers</em></p>

### Code: compare X and Y (Python)

#### Objective 1: customer churn

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 13: COMPARE FEATURES AND TARGET - OBJECTIVE 1: CHURN
# Two comparisons, no scores yet (scores come in step 14):
#   A. TARGET: actual churn (y_test) against predicted churn, as plain counts.
#   B. FEATURES: which features the models use to separate churners from stayers,
#      and how the churn rate changes across the values of the top features.
# Feature importance comes from the trained models and the TRAIN data only.
# Saves charts to ./churn_outputs. Needs: pandas numpy matplotlib scikit-learn joblib.
# Run cell by cell, or set SHOW = False to save the charts without displaying them.
# =====================================================================
# %% 0. SETTINGS AND LOAD
import os, pickle, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
DATA, OUT, SHOW = "abt_data", "churn_outputs", True
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
BLUE, RED, GREY = "#2F6DB5", "#C0392B", "#9AA5B1"

ready  = pickle.load(open(f"{DATA}/ready_churn.pkl", "rb"))     # scaled data
raw    = pickle.load(open(f"{DATA}/split_churn.pkl", "rb"))     # same rows, original units (for readable bands)
bundle = joblib.load(f"{DATA}/churn_models.joblib")
pred   = pd.read_pickle(f"{DATA}/churn_test_predictions.pkl")
models, champion, features = bundle["models"], bundle["champion"], bundle["features"]
y_train, y_test = ready["y_train"], ready["y_test"]
assert (pred["actual_churned"].values == y_test.values).all(), "prediction file does not match y_test"

def save(name):
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight")
    if SHOW: plt.show()
    plt.close()

# %% A1. ACTUAL VS PREDICTED: COUNTS FOR EVERY MODEL
rows = []
for n in models:
    tn, fp, fn, tp = confusion_matrix(pred["actual_churned"], pred[f"{n}_pred"], labels=[0, 1]).ravel()
    rows.append({"model": n, "actual_churners": int(tp + fn), "predicted_churners": int(tp + fp),
                 "churners_caught": int(tp), "churners_missed": int(fn), "false_alarms": int(fp), "stayers_correct": int(tn)})
counts = pd.DataFrame(rows)
print("ACTUAL vs PREDICTED churn on the test set (counts only)\n", counts.to_string(index=False))

fig, ax = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.4))
for a, n in zip(ax, models):
    cm = confusion_matrix(pred["actual_churned"], pred[f"{n}_pred"], labels=[0, 1])
    a.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm): a.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    a.set_xticks([0, 1]); a.set_xticklabels(["stays", "churns"]); a.set_yticks([0, 1]); a.set_yticklabels(["stays", "churns"])
    a.set_xlabel("predicted"); a.set_title(n + (" *" if n == champion else ""), fontsize=9)
ax[0].set_ylabel("actual")
plt.suptitle("Actual vs predicted churn (* = champion)", y=1.02); save("13_1_actual_vs_predicted")

# %% A2. PREDICTED PROBABILITY BY ACTUAL OUTCOME (champion)
plt.figure(figsize=(8, 4))
plt.hist(pred.loc[pred.actual_churned == 0, "champion_proba"], bins=30, alpha=.65, color=BLUE, label="actually stayed", density=True)
plt.hist(pred.loc[pred.actual_churned == 1, "champion_proba"], bins=30, alpha=.65, color=RED, label="actually churned", density=True)
plt.axvline(0.5, color="black", ls="--", lw=1); plt.xlabel(f"predicted churn probability ({champion})"); plt.ylabel("density")
plt.title("Do churners get higher probabilities than stayers?"); plt.legend(); save("13_2_probability_by_outcome")
print(f"\nMedian predicted probability - actual stayers: {pred.loc[pred.actual_churned == 0, 'champion_proba'].median():.3f} | "
      f"actual churners: {pred.loc[pred.actual_churned == 1, 'champion_proba'].median():.3f}")

# %% B1. FEATURE IMPORTANCE FROM THE TRAINED MODELS
imp = {}
for n, m in models.items():
    if hasattr(m, "feature_importances_"): imp[n] = pd.Series(m.feature_importances_, index=features)
    elif hasattr(m, "coef_"):              imp[n] = pd.Series(m.coef_[0], index=features)          # features are standardised, so comparable
tree_name = champion if champion in imp and hasattr(models[champion], "feature_importances_") else "Random Forest"
top_tree = imp[tree_name].sort_values(ascending=False).head(15)
lr = imp["Logistic Regression"]
top_lr = lr.reindex(lr.abs().sort_values(ascending=False).head(15).index)

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
ax[0].barh(top_tree.index[::-1], top_tree.values[::-1], color=BLUE); ax[0].set_title(f"{tree_name}: top 15 importances")
ax[1].barh(top_lr.index[::-1], top_lr.values[::-1], color=[BLUE if v >= 0 else RED for v in top_lr.values[::-1]])
ax[1].set_title("Logistic Regression: top 15 coefficients\n(blue = more churn, red = less churn)"); ax[1].axvline(0, color="black", lw=.8)
save("13_3_feature_importance")
print(f"\n{tree_name} - top 10 features\n", top_tree.head(10).round(4).to_string())
print("\nLogistic Regression - top 10 coefficients (positive = raises churn risk)\n", top_lr.head(10).round(3).to_string())
both = sorted(set(top_tree.index) & set(top_lr.index))
print("\nFeatures in BOTH top-15 lists (the most trustworthy drivers):", both if both else "none")

# %% B2. FEATURES VS TARGET ON THE TRAIN DATA (no test labels used)
corr = ready["X_train"].corrwith(y_train).dropna()
corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(12)
print("\nStrongest correlations with churn (train, scaled features)\n", corr.round(3).to_string())

# %% B3. CHURN RATE ACROSS THE VALUES OF THE TOP FEATURES (original units)
top4 = top_tree.index[:4].tolist()
fig, ax = plt.subplots(1, 4, figsize=(17, 4))
for a, col in zip(ax, top4):
    s = raw["X_train"][col]
    if s.nunique() <= 6: bands = s.astype("string").fillna("missing")
    else: bands = pd.qcut(s, 4, duplicates="drop").astype("string").fillna("missing")
    rate = y_train.groupby(bands).agg(["mean", "count"]); rate = rate[rate["count"] >= 30]
    if s.nunique() > 6:   # order the bands low -> high, with "missing" last
        rate = rate.reindex(sorted(rate.index, key=lambda x: (x == "missing", 0.0 if x == "missing" else float(x.split(",")[0].strip("([")))))
    a.bar(range(len(rate)), rate["mean"] * 100, color=BLUE); a.axhline(y_train.mean() * 100, color=RED, ls="--", lw=1)
    a.set_xticks(range(len(rate))); a.set_xticklabels(rate.index, rotation=30, ha="right", fontsize=8)
    for i, v in enumerate(rate["mean"]): a.text(i, v * 100, f"{v*100:.0f}%", ha="center", va="bottom", fontsize=8)
    a.set_title(col, fontsize=10); a.set_ylabel("churn rate (%)")
    print(f"\nChurn rate by {col} (train)\n", (rate["mean"] * 100).round(1).to_string())
plt.suptitle("Churn rate across the values of the top features (red line = average)", y=1.03); save("13_4_top_features_vs_churn")

# %% C. SAVE A SMALL SUMMARY FILE
counts.to_csv(f"{OUT}/actual_vs_predicted_counts.csv", index=False)
pd.DataFrame({"tree_importance": imp[tree_name], "logistic_coef": lr}).sort_values("tree_importance", ascending=False).to_csv(f"{OUT}/feature_importance.csv")
print(f"\nDONE. Charts and tables saved in: {os.path.abspath(OUT)}")
```

#### Objective 2: fraud detection

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 13: COMPARE FEATURES AND TARGET - OBJECTIVE 2: FRAUD
# Two comparisons, no scores yet (scores come in step 14):
#   A. TARGET: actual fraud (y_test) against predicted fraud, as plain counts.
#   B. FEATURES: which features the models use to separate fraud from legitimate
#      transactions, and how the fraud rate changes across the values of the top features.
# Feature importance comes from the trained models and the TRAIN period only.
# Saves charts to ./fraud_outputs. Needs: pandas numpy matplotlib scikit-learn joblib.
# Run cell by cell, or set SHOW = False to save the charts without displaying them.
# =====================================================================
# %% 0. SETTINGS AND LOAD
import os, pickle, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
DATA, OUT, SHOW = "abt_data", "fraud_outputs", True
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
BLUE, RED, GREY = "#2F6DB5", "#C0392B", "#9AA5B1"

ready  = pickle.load(open(f"{DATA}/ready_fraud.pkl", "rb"))     # scaled data
raw    = pickle.load(open(f"{DATA}/split_fraud.pkl", "rb"))     # same rows, original units (for readable bands)
bundle = joblib.load(f"{DATA}/fraud_models.joblib")
pred   = pd.read_pickle(f"{DATA}/fraud_test_predictions.pkl")
models, champion, features = bundle["models"], bundle["champion"], bundle["features"]
y_train, y_test = ready["y_train"], ready["y_test"]
assert (pred["actual_fraud"].values == y_test.values).all(), "prediction file does not match y_test"

def save(name):
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight")
    if SHOW: plt.show()
    plt.close()

# %% A1. ACTUAL VS PREDICTED: COUNTS FOR EVERY MODEL (threshold 0.5) AND FOR THE TOP-2% REVIEW LIST
def count_row(label, actual, flagged):
    tn, fp, fn, tp = confusion_matrix(actual, flagged, labels=[0, 1]).ravel()
    return {"model": label, "actual_fraud": int(tp + fn), "flagged": int(tp + fp), "fraud_caught": int(tp),
            "fraud_missed": int(fn), "false_alarms": int(fp), "legit_correct": int(tn)}
rows = [count_row(n, pred["actual_fraud"], pred[f"{n}_pred"]) for n in models]
rows.append(count_row(f"{champion} - top 2% review list", pred["actual_fraud"], pred["champion_review_flag"]))
counts = pd.DataFrame(rows)
print("ACTUAL vs PREDICTED fraud on the test period (counts only)\n", counts.to_string(index=False))

fig, ax = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.4))
for a, n in zip(ax, models):
    cm = confusion_matrix(pred["actual_fraud"], pred[f"{n}_pred"], labels=[0, 1])
    a.imshow(np.log1p(cm), cmap="Blues")                                   # log colours: legit rows would hide the fraud cells
    for (i, j), v in np.ndenumerate(cm): a.text(j, i, f"{v:,}", ha="center", va="center", color="white" if np.log1p(v) > np.log1p(cm).max() / 2 else "black")
    a.set_xticks([0, 1]); a.set_xticklabels(["legit", "fraud"]); a.set_yticks([0, 1]); a.set_yticklabels(["legit", "fraud"])
    a.set_xlabel("predicted"); a.set_title(n + (" *" if n == champion else ""), fontsize=9)
ax[0].set_ylabel("actual")
plt.suptitle("Actual vs predicted fraud (* = champion; colours on a log scale)", y=1.02); save("13_1_actual_vs_predicted")

# %% A2. PREDICTED PROBABILITY BY ACTUAL OUTCOME (champion)
plt.figure(figsize=(8, 4))
plt.hist(pred.loc[pred.actual_fraud == 0, "champion_proba"], bins=30, alpha=.65, color=BLUE, label="legitimate")
plt.hist(pred.loc[pred.actual_fraud == 1, "champion_proba"], bins=30, alpha=.75, color=RED, label="fraud")
plt.yscale("log"); plt.xlabel(f"predicted fraud probability ({champion})"); plt.ylabel("transactions (log scale)")
plt.title("Do fraudulent transactions get higher probabilities?"); plt.legend(); save("13_2_probability_by_outcome")
print(f"\nMedian predicted probability - legitimate: {pred.loc[pred.actual_fraud == 0, 'champion_proba'].median():.3f} | "
      f"fraud: {pred.loc[pred.actual_fraud == 1, 'champion_proba'].median():.3f}")

# %% B1. FEATURE IMPORTANCE FROM THE TRAINED MODELS
imp = {}
for n, m in models.items():
    if hasattr(m, "feature_importances_"): imp[n] = pd.Series(m.feature_importances_, index=features)
    elif hasattr(m, "coef_"):              imp[n] = pd.Series(m.coef_[0], index=features)          # features are standardised, so comparable
tree_name = champion if hasattr(models[champion], "feature_importances_") else "Random Forest"
top_tree = imp[tree_name].sort_values(ascending=False).head(15)
lr = imp["Logistic Regression"]
top_lr = lr.reindex(lr.abs().sort_values(ascending=False).head(15).index)

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
ax[0].barh(top_tree.index[::-1], top_tree.values[::-1], color=BLUE); ax[0].set_title(f"{tree_name}: top 15 importances")
ax[1].barh(top_lr.index[::-1], top_lr.values[::-1], color=[BLUE if v >= 0 else RED for v in top_lr.values[::-1]])
ax[1].set_title("Logistic Regression: top 15 coefficients\n(blue = more fraud risk, red = less fraud risk)"); ax[1].axvline(0, color="black", lw=.8)
save("13_3_feature_importance")
print(f"\n{tree_name} - top 10 features\n", top_tree.head(10).round(4).to_string())
print("\nLogistic Regression - top 10 coefficients (positive = raises fraud risk)\n", top_lr.head(10).round(3).to_string())
both = sorted(set(top_tree.index) & set(top_lr.index))
print("\nFeatures in BOTH top-15 lists (the most trustworthy drivers):", both if both else "none")

# %% B2. FEATURES VS TARGET ON THE TRAIN DATA (no test labels used)
corr = ready["X_train"].corrwith(y_train).dropna()
corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(12)
print("\nStrongest correlations with fraud (train, scaled features)\n", corr.round(3).to_string())

# %% B3. FRAUD RATE ACROSS THE VALUES OF THE TOP FEATURES (original units)
top4 = top_tree.index[:4].tolist()
fig, ax = plt.subplots(1, 4, figsize=(17, 4))
for a, col in zip(ax, top4):
    s = raw["X_train"][col]
    if s.nunique() <= 6: bands = s.astype("string").fillna("missing")
    else: bands = pd.qcut(s, 5, duplicates="drop").astype("string").fillna("missing")
    rate = y_train.groupby(bands).agg(["mean", "count"]); rate = rate[rate["count"] >= 30]
    if s.nunique() > 6:   # order the bands low -> high, with "missing" last
        rate = rate.reindex(sorted(rate.index, key=lambda x: (x == "missing", 0.0 if x == "missing" else float(x.split(",")[0].strip("([")))))
    a.bar(range(len(rate)), rate["mean"] * 100, color=BLUE); a.axhline(y_train.mean() * 100, color=RED, ls="--", lw=1)
    a.set_xticks(range(len(rate))); a.set_xticklabels(rate.index, rotation=30, ha="right", fontsize=8)
    for i, v in enumerate(rate["mean"]): a.text(i, v * 100, f"{v*100:.1f}%", ha="center", va="bottom", fontsize=8)
    a.set_title(col, fontsize=10); a.set_ylabel("fraud rate (%)")
    print(f"\nFraud rate by {col} (train)\n", (rate["mean"] * 100).round(2).to_string())
plt.suptitle("Fraud rate across the values of the top features (red line = average)", y=1.03); save("13_4_top_features_vs_fraud")

# %% C. SAVE A SMALL SUMMARY FILE
counts.to_csv(f"{OUT}/actual_vs_predicted_counts.csv", index=False)
pd.DataFrame({"tree_importance": imp[tree_name], "logistic_coef": lr}).sort_values("tree_importance", ascending=False).to_csv(f"{OUT}/feature_importance.csv")
print(f"\nDONE. Charts and tables saved in: {os.path.abspath(OUT)}")
```

#### Objective 3: customer segmentation

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 13: COMPARE X AND y - OBJECTIVE 3: CUSTOMER SEGMENTATION
# Clustering has no y, so the comparison is done in two parts:
#   A. X: how the segments differ on the behaviour features the model used (original units).
#   B. OUTCOMES NOT USED TO BUILD THE SEGMENTS: churn, next-12-month spend, customer type,
#      loyalty tier and country. If the segments differ clearly on these, they carry real
#      business meaning. If they do not, the segments are only statistical groupings.
# Also checks that the segment profiles are the same on train and test customers.
# No scores yet (silhouette etc. come in step 14). Saves charts to ./segmentation_outputs.
# Needs: pandas numpy matplotlib scipy joblib. Set SHOW = False to save charts only.
# =====================================================================
# %% 0. SETTINGS AND LOAD
import os, pickle, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy import stats

warnings.filterwarnings("ignore")
DATA, OUT, SHOW = "abt_data", "segmentation_outputs", True
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
BLUE, RED, GREY = "#2F6DB5", "#C0392B", "#9AA5B1"

def load(path):
    with open(path, "rb") as f: return pickle.load(f)
ready  = load(f"{DATA}/ready_segmentation.pkl")                 # scaled features
raw    = load(f"{DATA}/split_segmentation.pkl")                 # same customers, original units
bundle = joblib.load(f"{DATA}/segmentation_model.joblib")
km, k, features = bundle["model"], bundle["k"], bundle["features"]
seg    = pd.read_pickle(f"{DATA}/segmentation_all_customers.pkl")[["customer_id", "segment", "dataset", "borderline"]]
abt    = pd.read_pickle(f"{DATA}/abt_customer.pkl")             # outcomes and descriptors, NOT used by the model

def save(name):
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight")
    if SHOW: plt.show()
    plt.close()

# one table: every customer with raw behaviour features, segment, and the outcomes
raw_x = pd.concat([raw["X_train"].join(raw["meta_train"][["customer_id"]]), raw["X_test"].join(raw["meta_test"][["customer_id"]])])
df = raw_x.merge(seg, on="customer_id").merge(
    abt[["customer_id", "churned", "next_12m_spend_usd", "customer_type", "loyalty_tier", "country"]].rename(columns={"loyalty_tier": "loyalty_name"}),
    on="customer_id")                                   # "loyalty_tier" is also a model feature (numeric), so the text label gets its own name
assert len(df) == len(seg) == len(abt), "customers lost when joining"
print(f"{len(df):,} customers in {k} segments (train {int((df.dataset == 'train').sum()):,} | test {int((df.dataset == 'test').sum()):,})")

# %% A1. HOW MUCH EACH FEATURE SEPARATES THE SEGMENTS (share of variance explained, scaled train features)
Xs, labels = ready["X_train"], km.predict(ready["X_train"])
tot = ((Xs - Xs.mean()) ** 2).sum()
between = sum(len(g) * (g.mean() - Xs.mean()) ** 2 for _, g in Xs.groupby(labels))
eta = (between / tot).sort_values(ascending=False)
print("\nFeatures that separate the segments most (share of the feature's variance explained by the segment)\n", (eta.head(10) * 100).round(1).astype(str).add("%").to_string())

# %% A2. SEGMENT CENTRES (scaled units: 0 = average customer, +1 = one standard deviation above)
centres = pd.DataFrame(km.cluster_centers_, columns=features).T
centres.columns = [f"Segment {i}" for i in range(k)]
fig, ax = plt.subplots(figsize=(1.6 * k + 4, 8))
im = ax.imshow(centres.values, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
ax.set_xticks(range(k)); ax.set_xticklabels(centres.columns); ax.set_yticks(range(len(features))); ax.set_yticklabels(features, fontsize=8)
for (i, j), v in np.ndenumerate(centres.values): ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7)
plt.colorbar(im, label="standard deviations from the average customer"); ax.set_title("Segment profiles (blue = above average, red = below)")
save("13_1_segment_profiles")
for s in range(k):
    c = centres[f"Segment {s}"]; hi, lo = c.sort_values(ascending=False).head(3), c.sort_values().head(3)
    print(f"\nSegment {s}: HIGH on {', '.join(f'{a} ({b:+.1f})' for a, b in hi.items())}")
    print(f"           LOW on  {', '.join(f'{a} ({b:+.1f})' for a, b in lo.items())}")

# %% A3. THE SAME PROFILE IN ORIGINAL UNITS (medians)
show = [c for c in ["days_since_last_txn", "txn_count", "total_spend_usd", "avg_txn_usd", "txn_count_180d", "spend_180d_usd", "service_count",
                    "total_service_cost_usd", "support_contacts", "complaint_contacts", "trip_count", "total_trip_km", "n_vehicles",
                    "total_vehicle_value_usd", "tenure_months", "online_txn_share"] if c in df.columns]
med = df.groupby("segment")[show].median().T.round(1)
med.columns = [f"Segment {c}" for c in med.columns]
med["all customers"] = df[show].median().round(1)
print("\nMEDIAN BEHAVIOUR BY SEGMENT (original units)\n", med.to_string())

# %% B1. OUTCOMES NOT USED TO BUILD THE SEGMENTS
g = df.groupby("segment")
out = pd.DataFrame({"customers": g.size(), "customer_share_%": (100 * g.size() / len(df)).round(1),
                    "churn_rate_%": (100 * g.churned.mean()).round(1),
                    "avg_next_12m_spend": g.next_12m_spend_usd.mean().round(0),
                    "median_next_12m_spend": g.next_12m_spend_usd.median().round(0)})
out["share_of_future_spend_%"] = (100 * g.next_12m_spend_usd.sum() / df.next_12m_spend_usd.sum()).round(1)
out["spend_vs_customer_share"] = (out["share_of_future_spend_%"] / out["customer_share_%"]).round(2)
print(f"\nOUTCOMES BY SEGMENT (overall churn {100 * df.churned.mean():.1f}%, overall average next-12-month spend {df.next_12m_spend_usd.mean():,.0f})\n", out.to_string())

# do the segments really differ? (tests on all customers)
chi_p = stats.chi2_contingency(pd.crosstab(df.segment, df.churned))[1]
kw_p = stats.kruskal(*[x.next_12m_spend_usd.dropna() for _, x in df.groupby("segment")])[1]
print(f"\nDo the segments differ on churn?  chi-square p = {chi_p:.2g}  (churn ranges {out['churn_rate_%'].min():.1f}% to {out['churn_rate_%'].max():.1f}%)")
print(f"Do the segments differ on spend?  Kruskal-Wallis p = {kw_p:.2g}  (average spend ranges {out.avg_next_12m_spend.min():,.0f} to {out.avg_next_12m_spend.max():,.0f})")
print("(a p-value below 0.05 means the difference is not chance; the size of the gap is what matters for the business)")

# %% B2. WHO IS IN EACH SEGMENT
mix_type = pd.crosstab(df.segment, df.customer_type, normalize="index") * 100
mix_tier = pd.crosstab(df.segment, df.loyalty_name, normalize="index")[[c for c in ["Bronze", "Silver", "Gold", "Platinum"] if c in df.loyalty_name.unique()]] * 100
print("\nCUSTOMER TYPE MIX BY SEGMENT (%)\n", mix_type.round(1).to_string())
print("\nLOYALTY TIER MIX BY SEGMENT (%)\n", mix_tier.round(1).to_string())
top_c = df.groupby("segment").country.agg(lambda s: ", ".join(f"{a} {100*b:.0f}%" for a, b in s.value_counts(normalize=True).head(3).items()))
print("\nTOP 3 COUNTRIES BY SEGMENT\n", top_c.to_string())

fig, ax = plt.subplots(1, 4, figsize=(19, 4.2)); ids = out.index.astype(str)
ax[0].bar(ids, out["churn_rate_%"], color=BLUE); ax[0].axhline(100 * df.churned.mean(), color=RED, ls="--", lw=1); ax[0].set_title("Churn rate (%)")
ax[1].bar(ids, out.avg_next_12m_spend, color=BLUE); ax[1].axhline(df.next_12m_spend_usd.mean(), color=RED, ls="--", lw=1); ax[1].set_title("Average next-12-month spend (USD)")
w = 0.38; xs = np.arange(k)
ax[2].bar(xs - w / 2, out["customer_share_%"], w, color=GREY, label="% of customers"); ax[2].bar(xs + w / 2, out["share_of_future_spend_%"], w, color=BLUE, label="% of future spend")
ax[2].set_xticks(xs); ax[2].set_xticklabels(ids); ax[2].legend(fontsize=8); ax[2].set_title("Customers vs spend share")
mix_type.plot(kind="bar", stacked=True, ax=ax[3], width=.75, colormap="Blues"); ax[3].set_title("Customer type mix (%)"); ax[3].legend(fontsize=7); ax[3].tick_params(axis="x", rotation=0)
for a in ax[:3]: a.set_xlabel("segment")
save("13_2_segment_outcomes")

# %% C. DO TRAIN AND TEST CUSTOMERS LOOK THE SAME IN EACH SEGMENT?
Xt = ready["X_test"]; lt = km.predict(Xt)
print("\nTRAIN vs TEST segment profile (correlation of the two centre vectors; close to 1.00 means the segment looks the same)")
for s in range(k):
    a = Xs[labels == s].mean(); b = Xt[lt == s].mean() if (lt == s).sum() > 1 else None
    print(f"  Segment {s}: " + (f"{np.corrcoef(a, b)[0, 1]:.2f} (test customers: {(lt == s).sum():,})" if b is not None else "too few test customers"))
print(f"Borderline customers overall: {100 * df.borderline.mean():.1f}%")

# %% D. SAVE
out.to_csv(f"{OUT}/segment_outcomes.csv"); med.to_csv(f"{OUT}/segment_median_profile.csv"); centres.round(3).to_csv(f"{OUT}/segment_centres.csv")
print(f"\nDONE. Charts and tables saved in: {os.path.abspath(OUT)}")
```

#### Objective 4: customer lifetime value

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 13: COMPARE FEATURES AND TARGET - OBJECTIVE 4: CUSTOMER LIFETIME VALUE
# Two comparisons, no scores yet (scores come in step 14):
#   A. TARGET: actual next-12-month spend (y_test) against predicted spend, in dollars.
#   B. FEATURES: which features the models use to separate high spenders from low spenders,
#      and how average spend changes across the values of the top features.
# Feature importance comes from the trained models and the TRAIN data only.
# Saves charts to ./clv_outputs. Needs: pandas numpy matplotlib scipy scikit-learn joblib.
# Run cell by cell, or set SHOW = False to save the charts without displaying them.
# =====================================================================
# %% 0. SETTINGS AND LOAD
import os, pickle, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")
DATA, OUT, SHOW = "abt_data", "clv_outputs", True
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
BLUE, RED, GREY = "#2F6DB5", "#C0392B", "#9AA5B1"

with open(f"{DATA}/ready_clv.pkl", "rb") as f: ready = pickle.load(f)     # scaled data
with open(f"{DATA}/split_clv.pkl", "rb") as f: raw = pickle.load(f)       # same rows, original units (for readable bands)
bundle = joblib.load(f"{DATA}/clv_models.joblib")
pred = pd.read_pickle(f"{DATA}/clv_test_predictions.pkl")
models, champion, features = bundle["models"], bundle["champion"], bundle["features"]
y_train, y_test = ready["y_train"], ready["y_test"]
assert np.allclose(pred["actual_spend_usd"].values, y_test.values), "prediction file does not match y_test"

def save(name):
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight")
    if SHOW: plt.show()
    plt.close()

# %% A1. ACTUAL VS PREDICTED SPEND: SUMMARY FOR EVERY MODEL
rows = [{"model": "ACTUAL", "mean": y_test.mean(), "median": y_test.median(), "p90": y_test.quantile(.9), "max": y_test.max()}]
for n in models:
    p = pred[f"{n}_pred"]
    rows.append({"model": n, "mean": p.mean(), "median": p.median(), "p90": p.quantile(.9), "max": p.max()})
summ = pd.DataFrame(rows)
print("ACTUAL vs PREDICTED next-12-month spend on the test set (USD, no scores yet)\n", summ.round(0).to_string(index=False))

# %% A2. SCATTER: PREDICTED VS ACTUAL (every model, log axes so the big spenders do not hide everyone else)
lim = [max(1, min(y_test.min(), 1)), max(y_test.max(), pred[[f"{n}_pred" for n in models]].values.max()) * 1.1]
fig, ax = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.6), sharex=True, sharey=True)
for a, n in zip(ax, models):
    a.scatter(pred["actual_spend_usd"] + 1, pred[f"{n}_pred"] + 1, s=10, alpha=.45, color=RED if n == champion else BLUE)
    a.plot(lim, lim, color="black", lw=1, ls="--"); a.set_xscale("log"); a.set_yscale("log")
    a.set_title(n + (" *" if n == champion else ""), fontsize=9); a.set_xlabel("actual spend (USD)")
ax[0].set_ylabel("predicted spend (USD)")
plt.suptitle("Predicted vs actual spend (dashed line = perfect prediction, * = champion)", y=1.02); save("13_1_actual_vs_predicted")

# %% A3. AVERAGE ACTUAL VS PREDICTED BY VALUE QUARTILE (champion)
pred["value_band"] = pd.qcut(pred["champion_pred"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"], duplicates="drop")
q = pred.groupby("value_band", observed=True).agg(customers=("customer_id", "count"), avg_predicted=("champion_pred", "mean"), avg_actual=("actual_spend_usd", "mean"))
print(f"\nCustomers ranked by predicted spend ({champion})\n", q.round(0).to_string())
x = np.arange(len(q)); plt.figure(figsize=(7, 4))
plt.bar(x - .2, q["avg_predicted"], .4, color=BLUE, label="average predicted"); plt.bar(x + .2, q["avg_actual"], .4, color=RED, label="average actual")
plt.xticks(x, q.index); plt.ylabel("USD per customer"); plt.title("Does a higher prediction mean higher real spend?"); plt.legend(); save("13_2_quartiles")

# %% B1. FEATURE IMPORTANCE FROM THE TRAINED MODELS (models are wrapped, so open the inner regressor)
imp = {}
for n, m in models.items():
    inner = m.regressor_
    if hasattr(inner, "feature_importances_"): imp[n] = pd.Series(inner.feature_importances_, index=features)
    elif hasattr(inner, "coef_"):              imp[n] = pd.Series(np.ravel(inner.coef_), index=features)   # features are standardised, so comparable
tree_name = champion if champion in imp and hasattr(models[champion].regressor_, "feature_importances_") else "Gradient Boosting"
top_tree = imp[tree_name].sort_values(ascending=False).head(15)
rg = imp["Ridge Regression"]
top_rg = rg.reindex(rg.abs().sort_values(ascending=False).head(15).index)

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
ax[0].barh(top_tree.index[::-1], top_tree.values[::-1], color=BLUE); ax[0].set_title(f"{tree_name}: top 15 importances")
ax[1].barh(top_rg.index[::-1], top_rg.values[::-1], color=[BLUE if v >= 0 else RED for v in top_rg.values[::-1]])
ax[1].set_title("Ridge Regression: top 15 coefficients\n(blue = higher spend, red = lower spend; on the log scale)"); ax[1].axvline(0, color="black", lw=.8)
save("13_3_feature_importance")
print(f"\n{tree_name} - top 10 features\n", top_tree.head(10).round(4).to_string())
print("\nRidge Regression - top 10 coefficients (positive = raises spend)\n", top_rg.head(10).round(3).to_string())
both = sorted(set(top_tree.index) & set(top_rg.index))
print("\nFeatures in BOTH top-15 lists (the most trustworthy drivers):", both if both else "none")

# %% B2. FEATURES VS TARGET ON THE TRAIN DATA (Spearman rank correlation, no test labels used)
corr = ready["X_train"].apply(lambda c: c.corr(y_train, method="spearman")).dropna()
corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(12)
print("\nStrongest rank correlations with next-12-month spend (train)\n", corr.round(3).to_string())

# %% B3. AVERAGE SPEND ACROSS THE VALUES OF THE TOP FEATURES (original units)
top4 = top_tree.index[:4].tolist()
fig, ax = plt.subplots(1, 4, figsize=(17, 4))
for a, col in zip(ax, top4):
    s = raw["X_train"][col]
    if s.nunique() <= 6: bands = s.astype("string").fillna("missing")
    else: bands = pd.qcut(s, 4, duplicates="drop").astype("string").fillna("missing")
    g = y_train.groupby(bands).agg(["median", "count"]); g = g[g["count"] >= 30]
    if s.nunique() > 6:   # order the bands low -> high, with "missing" last
        g = g.reindex(sorted(g.index, key=lambda x: (x == "missing", 0.0 if x == "missing" else float(x.split(",")[0].strip("([")))))
    a.bar(range(len(g)), g["median"], color=BLUE); a.axhline(y_train.median(), color=RED, ls="--", lw=1)
    a.set_xticks(range(len(g))); a.set_xticklabels(g.index, rotation=30, ha="right", fontsize=8)
    for i, v in enumerate(g["median"]): a.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    a.set_title(col, fontsize=10); a.set_ylabel("median spend (USD)")
    print(f"\nMedian next-12-month spend by {col} (train)\n", g["median"].round(0).to_string())
plt.suptitle("Median spend across the values of the top features (red line = overall median)", y=1.03); save("13_4_top_features_vs_spend")

# %% C. SAVE A SMALL SUMMARY FILE
summ.to_csv(f"{OUT}/actual_vs_predicted_summary.csv", index=False)
q.to_csv(f"{OUT}/predicted_quartiles.csv")
pd.DataFrame({"tree_importance": imp[tree_name], "ridge_coef": rg}).sort_values("tree_importance", ascending=False).to_csv(f"{OUT}/feature_importance.csv")
print(f"\nDONE. Charts and tables saved in: {os.path.abspath(OUT)}")
```

#### Objective 5: service cost

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 13: COMPARE FEATURES AND TARGET - OBJECTIVE 5: SERVICE COST
# Two comparisons, no scores yet (scores come in step 14):
#   A. TARGET: actual service job cost (y_test) against predicted cost, in dollars.
#   B. FEATURES: which features the models use to separate expensive jobs from cheap jobs,
#      and how typical cost changes across the values of the top features.
# Feature importance comes from the trained models and the TRAIN data only.
# Saves charts to ./service_outputs. Needs: pandas numpy matplotlib scipy scikit-learn joblib.
# Run cell by cell, or set SHOW = False to save the charts without displaying them.
# =====================================================================
# %% 0. SETTINGS AND LOAD
import os, pickle, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")
DATA, OUT, SHOW = "abt_data", "service_outputs", True
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
BLUE, RED, GREY = "#2F6DB5", "#C0392B", "#9AA5B1"

with open(f"{DATA}/ready_service.pkl", "rb") as f: ready = pickle.load(f)     # scaled data
with open(f"{DATA}/split_service.pkl", "rb") as f: raw = pickle.load(f)       # same rows, original units (for readable bands)
bundle = joblib.load(f"{DATA}/service_models.joblib")
pred = pd.read_pickle(f"{DATA}/service_test_predictions.pkl")
models, champion, features = bundle["models"], bundle["champion"], bundle["features"]
y_train, y_test = ready["y_train"], ready["y_test"]
assert np.allclose(pred["actual_cost_usd"].values, y_test.values), "prediction file does not match y_test"

def save(name):
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight")
    if SHOW: plt.show()
    plt.close()

# %% A1. ACTUAL VS PREDICTED COST: SUMMARY FOR EVERY MODEL
rows = [{"model": "ACTUAL", "mean": y_test.mean(), "median": y_test.median(), "p90": y_test.quantile(.9), "max": y_test.max()}]
for n in models:
    p = pred[f"{n}_pred"]
    rows.append({"model": n, "mean": p.mean(), "median": p.median(), "p90": p.quantile(.9), "max": p.max()})
summ = pd.DataFrame(rows)
print("ACTUAL vs PREDICTED service job cost on the test set (USD, no scores yet)\n", summ.round(0).to_string(index=False))

# %% A2. SCATTER: PREDICTED VS ACTUAL (every model, log axes so the expensive jobs do not hide everyone else)
lim = [max(1, min(y_test.min(), 1)), max(y_test.max(), pred[[f"{n}_pred" for n in models]].values.max()) * 1.1]
fig, ax = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.6), sharex=True, sharey=True)
for a, n in zip(ax, models):
    a.scatter(pred["actual_cost_usd"] + 1, pred[f"{n}_pred"] + 1, s=10, alpha=.45, color=RED if n == champion else BLUE)
    a.plot(lim, lim, color="black", lw=1, ls="--"); a.set_xscale("log"); a.set_yscale("log")
    a.set_title(n + (" *" if n == champion else ""), fontsize=9); a.set_xlabel("actual cost (USD)")
ax[0].set_ylabel("predicted cost (USD)")
plt.suptitle("Predicted vs actual cost (dashed line = perfect prediction, * = champion)", y=1.02); save("13_1_actual_vs_predicted")

# %% A3. AVERAGE ACTUAL VS PREDICTED BY COST QUARTILE (champion)
pred["cost_band"] = pd.qcut(pred["champion_pred"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"], duplicates="drop")
q = pred.groupby("cost_band", observed=True).agg(jobs=("service_id", "count"), avg_predicted=("champion_pred", "mean"), avg_actual=("actual_cost_usd", "mean"))
print(f"\nService jobs ranked by predicted cost ({champion})\n", q.round(0).to_string())
x = np.arange(len(q)); plt.figure(figsize=(7, 4))
plt.bar(x - .2, q["avg_predicted"], .4, color=BLUE, label="average predicted"); plt.bar(x + .2, q["avg_actual"], .4, color=RED, label="average actual")
plt.xticks(x, q.index); plt.ylabel("USD per job"); plt.title("Does a higher prediction mean a higher real cost?"); plt.legend(); save("13_2_quartiles")

# %% B1. FEATURE IMPORTANCE FROM THE TRAINED MODELS (models are wrapped, so open the inner regressor)
imp = {}
for n, m in models.items():
    inner = m.regressor_
    if hasattr(inner, "feature_importances_"): imp[n] = pd.Series(inner.feature_importances_, index=features)
    elif hasattr(inner, "coef_"):              imp[n] = pd.Series(np.ravel(inner.coef_), index=features)   # features are standardised, so comparable
tree_name = champion if champion in imp and hasattr(models[champion].regressor_, "feature_importances_") else "Random Forest"
top_tree = imp[tree_name].sort_values(ascending=False).head(15)
rg = imp["Ridge Regression"]
top_rg = rg.reindex(rg.abs().sort_values(ascending=False).head(15).index)

fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
ax[0].barh(top_tree.index[::-1], top_tree.values[::-1], color=BLUE); ax[0].set_title(f"{tree_name}: top 15 importances")
ax[1].barh(top_rg.index[::-1], top_rg.values[::-1], color=[BLUE if v >= 0 else RED for v in top_rg.values[::-1]])
ax[1].set_title("Ridge Regression: top 15 coefficients\n(blue = higher cost, red = lower cost; on the log scale)"); ax[1].axvline(0, color="black", lw=.8)
save("13_3_feature_importance")
print(f"\n{tree_name} - top 10 features\n", top_tree.head(10).round(4).to_string())
print("\nRidge Regression - top 10 coefficients (positive = raises cost)\n", top_rg.head(10).round(3).to_string())
both = sorted(set(top_tree.index) & set(top_rg.index))
print("\nFeatures in BOTH top-15 lists (the most trustworthy drivers):", both if both else "none")

# %% B2. FEATURES VS TARGET ON THE TRAIN DATA (Spearman rank correlation, no test labels used)
corr = ready["X_train"].apply(lambda c: c.corr(y_train, method="spearman")).dropna()
corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(12)
print("\nStrongest rank correlations with service cost (train)\n", corr.round(3).to_string())

# %% B3. MEDIAN COST ACROSS THE VALUES OF THE TOP FEATURES (original units)
top4 = top_tree.index[:4].tolist()
fig, ax = plt.subplots(1, 4, figsize=(17, 4))
for a, col in zip(ax, top4):
    s = raw["X_train"][col]
    if s.nunique() <= 6: bands = s.astype("string").fillna("missing")
    else: bands = pd.qcut(s, 4, duplicates="drop").astype("string").fillna("missing")
    g = y_train.groupby(bands).agg(["median", "count"]); g = g[g["count"] >= 30]
    if s.nunique() > 6:   # order the bands low -> high, with "missing" last
        g = g.reindex(sorted(g.index, key=lambda x: (x == "missing", 0.0 if x == "missing" else float(x.split(",")[0].strip("([")))))
    a.bar(range(len(g)), g["median"], color=BLUE); a.axhline(y_train.median(), color=RED, ls="--", lw=1)
    a.set_xticks(range(len(g))); a.set_xticklabels(g.index, rotation=30, ha="right", fontsize=8)
    for i, v in enumerate(g["median"]): a.text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    a.set_title(col, fontsize=10); a.set_ylabel("median cost (USD)")
    print(f"\nMedian service cost by {col} (train)\n", g["median"].round(0).to_string())
plt.suptitle("Median cost across the values of the top features (red line = overall median)", y=1.03); save("13_4_top_features_vs_cost")

# %% C. SAVE A SMALL SUMMARY FILE
summ.to_csv(f"{OUT}/actual_vs_predicted_summary.csv", index=False)
q.to_csv(f"{OUT}/predicted_quartiles.csv")
pd.DataFrame({"tree_importance": imp[tree_name], "ridge_coef": rg}).sort_values("tree_importance", ascending=False).to_csv(f"{OUT}/feature_importance.csv")
print(f"\nDONE. Charts and tables saved in: {os.path.abspath(OUT)}")
```

---

## 17. Evaluate Model

Scripts: `python/14_evaluate_<objective>.py`. Every model is scored on the test data and compared with its benchmark. **Bootstrap** resampling gives a 95% range, and for service cost the bootstrap resamples whole vehicles.

**Fixed verdict rules, written before looking at results**

| Verdict | Classification (churn, fraud) | Regression (lifetime value, service cost) |
|---|---|---|
| NOT PROVEN | 95% range does not beat no-skill | does not beat the median baseline with confidence |
| WEAK | beats no-skill | beats baseline, R² below 0.25 |
| USEFUL | also ROC AUC of 0.65 or more | beats baseline, R² of 0.25 or more |
| STRONG | also ROC AUC of 0.80 or more | beats baseline, R² of 0.5 or more and ranking correlation of 0.7 or more |

Segmentation is USEFUL (as tiers) when the adjusted Rand index between reruns is 0.8 or more, the test silhouette is at least the larger of 0.10 and twice the silhouette of noise, and the segments differ clearly on spend and churn.

**Results**

| Objective | Champion | Verdict | What it means |
|---|---|---|---|
| Churn | Logistic Regression | USEFUL | Beats chance across its whole 95% range with ROC AUC of at least 0.65 |
| Fraud | Logistic Regression | STRONG | Beats chance across its whole 95% range with ROC AUC of at least 0.80 |
| Segmentation | K-Means, k = 3 | USEFUL (as tiers) | Silhouette 0.16 on train and test against 0.04 for noise. Lowest stability score 0.94. Segments differ clearly on spend (3.6x) and churn (9.5 points) for customers the model never saw |
| Lifetime value | Ridge Regression | STRONG | Beats the median baseline with confidence, R² of at least 0.5, ranking correlation of at least 0.7 |
| Service cost | Gradient Boosting | STRONG | Same bar as lifetime value |

The champion is the model picked on train cross-validation. If another model scores slightly higher on test, the champion is kept anyway, because choosing by test score would leak the test set.

### Charts

#### Objective 1: customer churn

<p align="center">
  <img src="images/churn_14_1_roc_pr_gains.png" alt="ROC curve, precision-recall curve and cumulative gains (churn)" width="100%">
</p>
<p align="center"><em>ROC curve, precision-recall curve and cumulative gains (churn)</em></p>

<p align="center">
  <img src="images/churn_14_2_metrics_and_calibration.png" alt="Metrics for all five models and calibration (churn)" width="100%">
</p>
<p align="center"><em>Metrics for all five models and calibration (churn)</em></p>

#### Objective 2: fraud detection

<p align="center">
  <img src="images/fraud_14_1_roc_pr_gains.png" alt="ROC curve, precision-recall curve and cumulative gains (fraud)" width="100%">
</p>
<p align="center"><em>ROC curve, precision-recall curve and cumulative gains (fraud)</em></p>

<p align="center">
  <img src="images/fraud_14_2_metrics_and_deciles.png" alt="Metrics for all five models and fraud by risk decile" width="100%">
</p>
<p align="center"><em>Metrics for all five models and fraud by risk decile</em></p>

#### Objective 3: customer segmentation

<p align="center">
  <img src="images/segmentation_14_1_segmentation_quality.png" alt="Segmentation quality on unseen customers: silhouette, stability and outcome differences" width="100%">
</p>
<p align="center"><em>Segmentation quality on unseen customers: silhouette, stability and outcome differences</em></p>

#### Objective 4: customer lifetime value

<p align="center">
  <img src="images/clv_14_1_scatter_residuals_gains.png" alt="Actual against predicted spend, residuals and cumulative gains (lifetime value)" width="100%">
</p>
<p align="center"><em>Actual against predicted spend, residuals and cumulative gains (lifetime value)</em></p>

<p align="center">
  <img src="images/clv_14_2_metrics_and_calibration.png" alt="Metrics for all five models and calibration (lifetime value)" width="100%">
</p>
<p align="center"><em>Metrics for all five models and calibration (lifetime value)</em></p>

#### Objective 5: service cost

<p align="center">
  <img src="images/service_14_1_scatter_residuals_gains.png" alt="Actual against predicted cost, residuals and cumulative gains (service cost)" width="100%">
</p>
<p align="center"><em>Actual against predicted cost, residuals and cumulative gains (service cost)</em></p>

<p align="center">
  <img src="images/service_14_2_metrics_and_calibration.png" alt="Metrics for all five models and calibration (service cost)" width="100%">
</p>
<p align="center"><em>Metrics for all five models and calibration (service cost)</em></p>

### Code: evaluate model (Python)

#### Objective 1: customer churn

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 14: EVALUATE MODEL - OBJECTIVE 1: CUSTOMER CHURN
# Scores the 5 churn models on the TEST set (predictions from step 12), then
# judges the champion chosen in step 11:
#   1. metrics for all models (ROC AUC, PR AUC, precision, recall, F1, accuracy, Brier)
#   2. the decision threshold, chosen on TRAIN cross-validation (never on the test set)
#   3. bootstrap ranges (how much the scores could move on new customers)
#   4. business view: how many churners the retention team catches by contacting the top 10/20/30% riskiest customers
# Saves charts and churn_evaluation.json to ./churn_outputs.
# Needs: pandas numpy matplotlib scikit-learn joblib. Set SHOW = False to save charts only.
# =====================================================================
# %% 0. SETTINGS AND LOAD
import os, json, pickle, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score, recall_score, f1_score,
                             accuracy_score, brier_score_loss, roc_curve, precision_recall_curve)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
DATA, OUT, SHOW, SEED = "abt_data", "churn_outputs", True, 42
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
BLUE, RED, GREY = "#2F6DB5", "#C0392B", "#9AA5B1"

ready  = pickle.load(open(f"{DATA}/ready_churn.pkl", "rb"))
bundle = joblib.load(f"{DATA}/churn_models.joblib")
pred   = pd.read_pickle(f"{DATA}/churn_test_predictions.pkl")
models, champion = bundle["models"], bundle["champion"]
X_train, y_train, y_test = ready["X_train"], ready["y_train"], ready["y_test"]
y = pred["actual_churned"].values
base = y.mean()
assert (y == y_test.values).all()

def save(name):
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight")
    if SHOW: plt.show()
    plt.close()

def scores(y_true, proba, t=0.5):
    p = (proba >= t).astype(int)
    return {"roc_auc": roc_auc_score(y_true, proba), "pr_auc": average_precision_score(y_true, proba),
            "precision": precision_score(y_true, p, zero_division=0), "recall": recall_score(y_true, p),
            "f1": f1_score(y_true, p), "accuracy": accuracy_score(y_true, p), "brier": brier_score_loss(y_true, proba)}

# %% 1. METRICS FOR ALL MODELS (threshold 0.5)
tab = pd.DataFrame({n: scores(y, pred[f"{n}_proba"].values) for n in models}).T.sort_values("pr_auc", ascending=False)
tab.index.name = "model"
print(f"TEST SET: {len(y):,} customers, {int(y.sum()):,} churners ({base:.1%}). A model with no skill scores PR AUC = {base:.3f}, ROC AUC = 0.500\n")
print(tab.round(3).to_string())
print(f"\nChampion from step 11 (chosen on train cross-validation): {champion}")
best_test = tab.index[0]
if best_test != champion:
    print(f"Note: {best_test} scores slightly higher on the test set, but the champion stays {champion} because picking by test score would leak the test set.")

# %% 2. CHOOSE THE THRESHOLD ON TRAIN CROSS-VALIDATION, THEN USE IT ON THE TEST SET
def fit_one(model, X, yy):
    if isinstance(model, GradientBoostingClassifier):
        return model.fit(X, yy, sample_weight=compute_sample_weight("balanced", yy))
    return model.fit(X, yy)

oof = np.zeros(len(y_train))
for tr, va in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_train, y_train):
    oof[va] = fit_one(clone(models[champion]), X_train.iloc[tr], y_train.iloc[tr]).predict_proba(X_train.iloc[va])[:, 1]
grid = np.arange(0.10, 0.91, 0.01)
f1s = [f1_score(y_train, (oof >= t).astype(int)) for t in grid]
THRESH = float(grid[int(np.argmax(f1s))])
champ = pred["champion_proba"].values
at05, atT = scores(y, champ, 0.5), scores(y, champ, THRESH)
print(f"\nThreshold that maximises F1 on train cross-validation: {THRESH:.2f}")
print(pd.DataFrame({"threshold 0.50": at05, f"threshold {THRESH:.2f}": atT}).round(3).to_string())

# %% 3. BOOTSTRAP RANGES FOR THE CHAMPION
rng = np.random.default_rng(SEED)
boot = []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    if y[i].min() == y[i].max(): continue
    boot.append((roc_auc_score(y[i], champ[i]), average_precision_score(y[i], champ[i])))
boot = np.array(boot)
ci = {"roc_auc": np.percentile(boot[:, 0], [2.5, 97.5]).round(3).tolist(), "pr_auc": np.percentile(boot[:, 1], [2.5, 97.5]).round(3).tolist()}
print(f"\n95% bootstrap range - ROC AUC {ci['roc_auc']} | PR AUC {ci['pr_auc']}  (no-skill PR AUC = {base:.3f})")

# %% 4. BUSINESS VIEW: CONTACT THE RISKIEST CUSTOMERS FIRST
order = np.argsort(-champ)
rows = []
for pct in [5, 10, 20, 30, 40, 50]:
    k = int(round(len(y) * pct / 100)); hit = int(y[order[:k]].sum())
    rows.append({"contact_top_%": pct, "customers_contacted": k, "churners_caught": hit,
                 "share_of_all_churners_%": round(100 * hit / y.sum(), 1), "precision_%": round(100 * hit / k, 1), "lift_x": round((hit / k) / base, 2)})
topk = pd.DataFrame(rows)
print("\nRETENTION PLAN: contacting the riskiest customers first\n", topk.to_string(index=False))

dec = pd.qcut(pd.Series(champ).rank(method="first"), 10, labels=False)
calib = pd.DataFrame({"predicted": champ, "actual": y, "decile": dec}).groupby("decile").mean().round(3)
calib.index = [f"D{int(i)+1}" for i in calib.index]
print("\nCalibration by risk decile (D10 = riskiest): average predicted probability vs actual churn rate\n", calib.to_string())

# %% 5. CHARTS
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for n in models:
    p = pred[f"{n}_proba"].values; fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
    lw = 2.6 if n == champion else 1.2
    ax[0].plot(fpr, tpr, lw=lw, label=f"{n} ({tab.loc[n, 'roc_auc']:.3f})"); ax[1].plot(rc, pr, lw=lw, label=f"{n} ({tab.loc[n, 'pr_auc']:.3f})")
ax[0].plot([0, 1], [0, 1], "--", color=GREY); ax[0].set_title("ROC curve (AUC)"); ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
ax[1].axhline(base, ls="--", color=GREY); ax[1].set_title("Precision-recall curve (PR AUC)"); ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
for a in ax[:2]: a.legend(fontsize=8)
ax[2].plot([0, 100], [0, 100], "--", color=GREY, label="random targeting")
share = [0] + [100 * y[order[:int(len(y) * p / 100)]].sum() / y.sum() for p in range(1, 101)]
ax[2].plot(range(101), share, color=BLUE, lw=2.5, label=champion); ax[2].set_xlabel("% of customers contacted (riskiest first)")
ax[2].set_ylabel("% of churners caught"); ax[2].set_title("Cumulative gains"); ax[2].legend()
save("14_1_roc_pr_gains")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
tab[["roc_auc", "pr_auc", "recall", "precision", "f1"]].plot(kind="bar", ax=ax[0], width=.8); ax[0].set_title("Test metrics by model (threshold 0.5)")
ax[0].tick_params(axis="x", rotation=25); ax[0].legend(fontsize=8, ncol=3); ax[0].set_ylim(0, 1)
ax[1].plot([0, 1], [0, 1], "--", color=GREY); ax[1].plot(calib["predicted"], calib["actual"], "o-", color=BLUE)
ax[1].set_xlabel("average predicted probability (per decile)"); ax[1].set_ylabel("actual churn rate"); ax[1].set_title(f"Calibration - {champion}")
save("14_2_metrics_and_calibration")

# %% 6. SAVE THE RESULT
result = {"objective": "Customer churn", "task": "classification", "champion": champion, "test_customers": int(len(y)), "test_churners": int(y.sum()),
          "no_skill_pr_auc": round(float(base), 4), "threshold": round(THRESH, 2),
          "metrics_at_0.5": {k: round(float(v), 4) for k, v in at05.items()}, "metrics_at_threshold": {k: round(float(v), 4) for k, v in atT.items()},
          "bootstrap_95": ci, "contact_top_k": topk.to_dict(orient="records"),
          "all_models": {n: {k: round(float(v), 4) for k, v in r.items()} for n, r in tab.iterrows()}}
with open(f"{OUT}/churn_evaluation.json", "w") as f: json.dump(result, f, indent=2)
tab.round(4).to_csv(f"{OUT}/churn_test_metrics.csv"); topk.to_csv(f"{OUT}/churn_top_k_plan.csv", index=False)
print(f"\nDONE. Saved churn_evaluation.json, churn_test_metrics.csv, churn_top_k_plan.csv and 2 charts in {os.path.abspath(OUT)}")
```

#### Objective 2: fraud detection

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 14: EVALUATE MODEL - OBJECTIVE 2: FRAUD DETECTION
# Scores the 5 fraud models on the TEST PERIOD (predictions from step 12), then
# judges the champion chosen in step 11:
#   1. metrics for all models (ROC AUC, PR AUC, precision, recall, F1, Brier)
#   2. the decision threshold, chosen on TRAIN time-ordered cross-validation (never on the test period)
#   3. bootstrap ranges (how much the scores could move on new transactions)
#   4. business view: how much fraud (cases and dollars) the team catches by reviewing the
#      highest-risk 0.5%, 1%, 2%, 5% or 10% of transactions
# Fraud is rare, so PR AUC and the review-budget table matter more than accuracy.
# Saves charts and fraud_evaluation.json to ./fraud_outputs.
# Needs: pandas numpy matplotlib scikit-learn joblib. Set SHOW = False to save charts only.
# =====================================================================
# %% 0. SETTINGS AND LOAD
import os, json, pickle, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score, recall_score, f1_score,
                             accuracy_score, brier_score_loss, roc_curve, precision_recall_curve)
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
DATA, OUT, SHOW, SEED, N_BLOCKS, N_BOOT = "abt_data", "fraud_outputs", True, 42, 6, 500
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
BLUE, RED, GREY = "#2F6DB5", "#C0392B", "#9AA5B1"

ready  = pickle.load(open(f"{DATA}/ready_fraud.pkl", "rb"))
raw    = pickle.load(open(f"{DATA}/split_fraud.pkl", "rb"))       # original units, to read the dollar amounts
bundle = joblib.load(f"{DATA}/fraud_models.joblib")
pred   = pd.read_pickle(f"{DATA}/fraud_test_predictions.pkl")
models, champion = bundle["models"], bundle["champion"]
X_train, y_train, dates = ready["X_train"], ready["y_train"], ready["meta_train"]["txn_date"]
y = pred["actual_fraud"].values
base = y.mean()
assert (y == ready["y_test"].values).all()
amount = raw["X_test"]["amount_usd"].reindex(ready["X_test"].index).fillna(0).values if "amount_usd" in raw["X_test"].columns else None

def save(name):
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight")
    if SHOW: plt.show()
    plt.close()

def scores(y_true, proba, t=0.5):
    p = (proba >= t).astype(int)
    return {"roc_auc": roc_auc_score(y_true, proba), "pr_auc": average_precision_score(y_true, proba),
            "precision": precision_score(y_true, p, zero_division=0), "recall": recall_score(y_true, p),
            "f1": f1_score(y_true, p, zero_division=0), "accuracy": accuracy_score(y_true, p), "brier": brier_score_loss(y_true, proba)}

# %% 1. METRICS FOR ALL MODELS (threshold 0.5)
tab = pd.DataFrame({n: scores(y, pred[f"{n}_proba"].values) for n in models}).T.sort_values("pr_auc", ascending=False)
tab.index.name = "model"
print(f"TEST PERIOD: {len(y):,} transactions, {int(y.sum()):,} fraud cases ({base:.2%}). A model with no skill scores PR AUC = {base:.3f}, ROC AUC = 0.500")
if y.sum() < 100: print("WARNING: fewer than 100 fraud cases in the test period - scores will be unstable")
print("Accuracy is NOT a useful score here: flagging nothing is already", f"{1 - base:.1%}", "accurate.\n")
print(tab.round(3).to_string())
print(f"\nChampion from step 11 (chosen on train time-ordered cross-validation): {champion}")
if tab.index[0] != champion:
    print(f"Note: {tab.index[0]} scores slightly higher on the test period, but the champion stays {champion} because picking by test score would leak the test set.")

# %% 2. CHOOSE THE THRESHOLD ON TRAIN TIME-ORDERED CROSS-VALIDATION
def fit_one(model, X, yy):
    if isinstance(model, GradientBoostingClassifier):
        return model.fit(X, yy, sample_weight=compute_sample_weight("balanced", yy))
    return model.fit(X, yy)

order_tr = np.argsort(dates.values, kind="stable")
blocks = np.array_split(order_tr, N_BLOCKS)
oof_idx, oof_p = [], []
for k in range(1, N_BLOCKS):
    tr, va = np.concatenate(blocks[:k]), blocks[k]
    m = fit_one(clone(models[champion]), X_train.iloc[tr], y_train.iloc[tr])
    oof_idx.append(va); oof_p.append(m.predict_proba(X_train.iloc[va])[:, 1])
oof_idx, oof_p = np.concatenate(oof_idx), np.concatenate(oof_p)
yv = y_train.iloc[oof_idx].values
grid = np.arange(0.10, 0.96, 0.01)
f1s = [f1_score(yv, (oof_p >= t).astype(int), zero_division=0) for t in grid]
THRESH = float(grid[int(np.argmax(f1s))])
champ = pred["champion_proba"].values
at05, atT = scores(y, champ, 0.5), scores(y, champ, THRESH)
print(f"\nThreshold that maximises F1 on train time-ordered cross-validation: {THRESH:.2f}")
print(pd.DataFrame({"threshold 0.50": at05, f"threshold {THRESH:.2f}": atT}).round(3).to_string())

# %% 3. BOOTSTRAP RANGES FOR THE CHAMPION
rng = np.random.default_rng(SEED)
boot = []
for _ in range(N_BOOT):
    i = rng.integers(0, len(y), len(y))
    if y[i].min() == y[i].max(): continue
    boot.append((roc_auc_score(y[i], champ[i]), average_precision_score(y[i], champ[i])))
boot = np.array(boot)
ci = {"roc_auc": np.percentile(boot[:, 0], [2.5, 97.5]).round(3).tolist(), "pr_auc": np.percentile(boot[:, 1], [2.5, 97.5]).round(3).tolist()}
print(f"\n95% bootstrap range - ROC AUC {ci['roc_auc']} | PR AUC {ci['pr_auc']}  (no-skill PR AUC = {base:.3f})")

# %% 4. BUSINESS VIEW: REVIEW THE HIGHEST-RISK TRANSACTIONS FIRST
order = np.argsort(-champ)
fraud_value = float((amount * y).sum()) if amount is not None else None
rows = []
for pct in [0.5, 1, 2, 5, 10, 20]:
    k = max(1, int(round(len(y) * pct / 100))); hit = int(y[order[:k]].sum())
    r = {"review_top_%": pct, "transactions_reviewed": k, "fraud_caught": hit,
         "share_of_all_fraud_%": round(100 * hit / y.sum(), 1), "precision_%": round(100 * hit / k, 1), "lift_x": round((hit / k) / base, 1)}
    if fraud_value:
        r["fraud_$_caught"] = round(float((amount[order[:k]] * y[order[:k]]).sum()), 0)
        r["share_of_fraud_$_%"] = round(100 * r["fraud_$_caught"] / fraud_value, 1)
    rows.append(r)
topk = pd.DataFrame(rows)
print("\nREVIEW BUDGET: what the fraud team catches by reviewing the riskiest transactions first\n", topk.to_string(index=False))
if fraud_value: print(f"(total fraud value in the test period: ${fraud_value:,.0f})")

dec = pd.qcut(pd.Series(champ).rank(method="first"), 10, labels=False)
calib = pd.DataFrame({"predicted": champ, "actual": y, "decile": dec}).groupby("decile").mean().round(4)
calib.index = [f"D{int(i)+1}" for i in calib.index]
print("\nRisk deciles (D10 = riskiest): average predicted probability vs actual fraud rate\n", calib.to_string())
print("(models trained with class weights over-state probabilities; rank order is what matters for a review list)")

# %% 5. CHARTS
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for n in models:
    p = pred[f"{n}_proba"].values; fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
    lw = 2.6 if n == champion else 1.2
    ax[0].plot(fpr, tpr, lw=lw, label=f"{n} ({tab.loc[n, 'roc_auc']:.3f})"); ax[1].plot(rc, pr, lw=lw, label=f"{n} ({tab.loc[n, 'pr_auc']:.3f})")
ax[0].plot([0, 1], [0, 1], "--", color=GREY); ax[0].set_title("ROC curve (AUC)"); ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
ax[1].axhline(base, ls="--", color=GREY); ax[1].set_title("Precision-recall curve (PR AUC)"); ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
for a in ax[:2]: a.legend(fontsize=8)
ax[2].plot([0, 100], [0, 100], "--", color=GREY, label="random review")
pcts = np.arange(0, 101)
share = [0] + [100 * y[order[:max(1, int(len(y) * p / 100))]].sum() / y.sum() for p in pcts[1:]]
ax[2].plot(pcts, share, color=BLUE, lw=2.5, label="fraud cases caught")
if fraud_value:
    ax[2].plot(pcts, [0] + [100 * (amount[order[:max(1, int(len(y) * p / 100))]] * y[order[:max(1, int(len(y) * p / 100))]]).sum() / fraud_value for p in pcts[1:]],
               color=RED, lw=2, label="fraud $ caught")
ax[2].set_xlim(0, 50); ax[2].set_ylim(0, 100); ax[2].set_xlabel("% of transactions reviewed (riskiest first)"); ax[2].set_ylabel("% of fraud caught")
ax[2].set_title("Cumulative gains"); ax[2].legend()
save("14_1_roc_pr_gains")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
tab[["roc_auc", "pr_auc", "recall", "precision", "f1"]].plot(kind="bar", ax=ax[0], width=.8); ax[0].set_title("Test metrics by model (threshold 0.5)")
ax[0].tick_params(axis="x", rotation=25); ax[0].legend(fontsize=8, ncol=3); ax[0].set_ylim(0, 1)
ax[1].bar(calib.index, calib["actual"] * 100, color=BLUE); ax[1].axhline(base * 100, color=RED, ls="--", lw=1)
ax[1].set_ylabel("actual fraud rate (%)"); ax[1].set_title(f"Actual fraud rate by risk decile - {champion}")
save("14_2_metrics_and_deciles")

# %% 6. SAVE THE RESULT
result = {"objective": "Fraud detection", "task": "classification", "champion": champion, "test_transactions": int(len(y)), "test_fraud_cases": int(y.sum()),
          "no_skill_pr_auc": round(float(base), 4), "threshold": round(THRESH, 2),
          "metrics_at_0.5": {k: round(float(v), 4) for k, v in at05.items()}, "metrics_at_threshold": {k: round(float(v), 4) for k, v in atT.items()},
          "bootstrap_95": ci, "review_budget": topk.to_dict(orient="records"),
          "all_models": {n: {k: round(float(v), 4) for k, v in r.items()} for n, r in tab.iterrows()}}
with open(f"{OUT}/fraud_evaluation.json", "w") as f: json.dump(result, f, indent=2)
tab.round(4).to_csv(f"{OUT}/fraud_test_metrics.csv"); topk.to_csv(f"{OUT}/fraud_review_budget.csv", index=False)
print(f"\nDONE. Saved fraud_evaluation.json, fraud_test_metrics.csv, fraud_review_budget.csv and 2 charts in {os.path.abspath(OUT)}")
```

#### Objective 3: customer segmentation

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 14: EVALUATE MODEL - OBJECTIVE 3: CUSTOMER SEGMENTATION
# There is no right answer to score against, so a clustering model is judged on four questions:
#   1. SEPARATION  - are customers closer to their own segment than to the others?
#                    (silhouette, Davies-Bouldin, Calinski-Harabasz, on train AND test customers;
#                     and compared with the same data with the structure destroyed = pure noise)
#   2. STABILITY   - do the same customers stay together when we refit with other seeds,
#                    other samples, or on the test customers alone?
#   3. MEANING     - do the segments differ on outcomes the model never saw (churn, future spend)?
#                    Measured on the TEST customers, which were never used to build the segments.
#   4. VERDICT     - a plain-language summary built from rules of thumb.
# Saves charts and segmentation_evaluation.json to ./segmentation_outputs.
# Needs: pandas numpy matplotlib scikit-learn scipy joblib. Set SHOW = False to save charts only.
# =====================================================================
# %% 0. SETTINGS AND LOAD
import os, json, pickle, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, silhouette_samples, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score)

warnings.filterwarnings("ignore")
DATA, OUT, SHOW, SEED, N_REP = "abt_data", "segmentation_outputs", True, 42, 20
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
BLUE, RED, GREY = "#2F6DB5", "#C0392B", "#9AA5B1"

def load(path):
    with open(path, "rb") as f: return pickle.load(f)
ready  = load(f"{DATA}/ready_segmentation.pkl")
bundle = joblib.load(f"{DATA}/segmentation_model.joblib")
km, k = bundle["model"], bundle["k"]
Xtr, Xte = ready["X_train"], ready["X_test"]
ltr, lte = km.predict(Xtr), km.predict(Xte)
abt = pd.read_pickle(f"{DATA}/abt_customer.pkl")[["customer_id", "churned", "next_12m_spend_usd"]]
rng = np.random.default_rng(SEED)

def save(name):
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight")
    if SHOW: plt.show()
    plt.close()

# %% 1. SEPARATION
def sep(X, lab):
    return {"silhouette": silhouette_score(X, lab), "davies_bouldin": davies_bouldin_score(X, lab), "calinski_harabasz": calinski_harabasz_score(X, lab)}
sep_tab = pd.DataFrame({"train customers": sep(Xtr, ltr), "test customers": sep(Xte, lte)}).round(3)
print(f"1. SEPARATION (K-Means, k = {k})\n", sep_tab.to_string())
print("   silhouette: above 0.5 strong | 0.25 to 0.5 usable | 0.15 to 0.25 weak | below 0.15 no real structure.  Davies-Bouldin: lower is better.")

# the same data with every column shuffled independently: any structure is destroyed, so this is the "noise" baseline
perm = []
for _ in range(5):
    Xp = pd.DataFrame({c: rng.permutation(Xtr[c].values) for c in Xtr.columns})
    perm.append(silhouette_score(Xp, KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Xp)))
sil_noise = float(np.mean(perm))
print(f"   noise baseline (same data with columns shuffled): silhouette {sil_noise:.3f}  -> real data is {sep_tab.loc['silhouette', 'train customers'] - sil_noise:+.3f} above noise")

sil = silhouette_samples(Xtr, ltr)
per_seg = pd.DataFrame({"customers": np.bincount(ltr), "mean_silhouette": [sil[ltr == s].mean() for s in range(k)],
                        "misplaced_%": [100 * (sil[ltr == s] < 0).mean() for s in range(k)]}).round(3)
per_seg.index.name = "segment"
print("\n   Per segment (train). misplaced = customers closer to another segment than their own\n", per_seg.to_string())

# %% 2. STABILITY
ari_seed, ari_boot = [], []
for s in range(N_REP):
    ari_seed.append(adjusted_rand_score(ltr, KMeans(n_clusters=k, n_init=10, random_state=1000 + s).fit_predict(Xtr)))
    sub = rng.choice(len(Xtr), int(0.8 * len(Xtr)), replace=False)
    ari_boot.append(adjusted_rand_score(ltr, KMeans(n_clusters=k, n_init=5, random_state=2000 + s).fit(Xtr.iloc[sub]).predict(Xtr)))
refit_test = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(Xte)
ari_test = adjusted_rand_score(lte, refit_test)
stab = {"refit with other seeds (full train)": np.mean(ari_seed), "refit on random 80% samples": np.mean(ari_boot), "model built on test customers alone vs train model": ari_test}
print("\n2. STABILITY (adjusted Rand index: 1.00 = identical grouping, 0 = no agreement)")
for a, b in stab.items(): print(f"   {a:55s} {b:.3f}")
print(f"   range over {N_REP} repeats: seeds {min(ari_seed):.2f} to {max(ari_seed):.2f} | samples {min(ari_boot):.2f} to {max(ari_boot):.2f}")

# %% 3. MEANING: DO THE SEGMENTS DIFFER ON OUTCOMES THE MODEL NEVER SAW? (test customers only)
te = ready["meta_test"][["customer_id"]].assign(segment=lte).merge(abt, on="customer_id")
g = te.groupby("segment")
res = pd.DataFrame({"customers": g.size(), "churn_rate_%": (100 * g.churned.mean()).round(1),
                    "avg_next_12m_spend": g.next_12m_spend_usd.mean().round(0)})
print(f"\n3. MEANING on {len(te):,} test customers (overall churn {100 * te.churned.mean():.1f}%, overall average spend {te.next_12m_spend_usd.mean():,.0f})\n", res.to_string())
tab_ct = pd.crosstab(te.segment, te.churned); chi2, p_chi = stats.chi2_contingency(tab_ct)[:2]
cramers_v = float(np.sqrt(chi2 / (len(te) * (min(tab_ct.shape) - 1))))
H, p_kw = stats.kruskal(*[x.next_12m_spend_usd.dropna() for _, x in te.groupby("segment")])
eps2 = float(H * (len(te) + 1) / (len(te) ** 2 - 1))
big = res[res.customers >= 30]                                   # ignore tiny segments: their rates are noise
if len(big) < 2: big = res
churn_spread = float(big["churn_rate_%"].max() - big["churn_rate_%"].min())
spend_ratio = float(big.avg_next_12m_spend.max() / big.avg_next_12m_spend.min())
print(f"   churn: spread {churn_spread:.1f} points between segments | Cramer's V {cramers_v:.3f} | p = {p_chi:.2g}")
print(f"   spend: highest / lowest segment = {spend_ratio:.2f}x | epsilon-squared {eps2:.3f} | p = {p_kw:.2g}")
print("   (effect sizes: V or epsilon-squared below 0.02 negligible, 0.02 to 0.06 small, 0.06 to 0.14 medium, above 0.14 large)")

# %% 4. VERDICT
sil_tr = sep_tab.loc["silhouette", "train customers"]; sil_te = sep_tab.loc["silhouette", "test customers"]
stable = min(stab.values()) >= 0.8
if sil_tr >= 0.5: shape = "strongly separated groups"
elif sil_tr >= 0.25: shape = "usable, clearly distinguishable groups"
elif sil_tr >= 0.15 and sil_tr - sil_noise > 0.05: shape = "weakly separated groups: customers lie on a continuum, so the segments work as tiers, not natural clusters"
else: shape = "no real cluster structure"
holds_test = sil_te >= 0.5 * sil_tr
significant = (p_chi < 0.05) or (p_kw < 0.05)
if not significant: business = "negligible (not statistically significant)"
elif churn_spread >= 8 or spend_ratio >= 1.25 or max(cramers_v, eps2) >= 0.06: business = "clear"
elif max(cramers_v, eps2) >= 0.02 or spend_ratio >= 1.10: business = "small"
else: business = "negligible"
verdict = [f"Separation: {shape} (silhouette train {sil_tr:.2f}, test {sil_te:.2f}; noise baseline {sil_noise:.2f}). "
           + ("It holds on the test customers." if holds_test else "It does NOT hold on the test customers."),
           f"Stability: {'reliable' if stable else 'NOT reliable on every check'} (lowest ARI {min(stab.values()):.2f}).",
           f"Business meaning: {business} differences on outcomes the model never saw (churn spread {churn_spread:.1f} pts, spend ratio {spend_ratio:.2f}x, segments with 30+ test customers only)."]
print("\n4. VERDICT (rules of thumb, not guarantees)"); [print("   -", v) for v in verdict]
if business.startswith("negligible"): print("   - Recommendation: the segments do not separate churn or spend. Revisit the feature set (step 8) before using them for decisions.")
elif not stable or not holds_test: print("   - Recommendation: the grouping moves between refits or does not hold on test customers. Try fewer segments or drop highly correlated features.")
else: print("   - Recommendation: use the segments as reliable customer tiers; name them from the step 13 profiles and attach one action to each.")

# %% 5. CHARTS
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
y0 = 0
for s in range(k):
    v = np.sort(sil[ltr == s]); ax[0].fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, alpha=.85, label=f"segment {s}"); y0 += len(v) + 50
ax[0].axvline(sil.mean(), color=RED, ls="--", lw=1); ax[0].set_title(f"Silhouette by customer (train, average {sil.mean():.2f})"); ax[0].set_xlabel("silhouette"); ax[0].set_yticks([]); ax[0].legend(fontsize=8)
bars = {"train": sil_tr, "test": sil_te, "noise baseline": sil_noise}
ax[1].bar(bars.keys(), bars.values(), color=[BLUE, BLUE, GREY])
for i, v in enumerate(bars.values()): ax[1].text(i, v, f"{v:.2f}", ha="center", va="bottom")
ax[1].axhline(0.25, color=RED, ls="--", lw=1); ax[1].set_title("Separation vs noise (red = 0.25 'usable')")
ax[2].barh(list(stab.keys())[::-1], list(stab.values())[::-1], color=BLUE); ax[2].axvline(0.8, color=RED, ls="--", lw=1); ax[2].set_xlim(0, 1)
ax[2].set_title("Stability (ARI, red = 0.8 'reliable')"); ax[2].tick_params(axis="y", labelsize=8)
save("14_1_segmentation_quality")

# %% 6. SAVE
result = {"objective": "Customer segmentation", "task": "clustering (K-Means)", "k": int(k),
          "separation": {c: {i: float(v) for i, v in sep_tab[c].items()} for c in sep_tab.columns}, "noise_baseline_silhouette": round(sil_noise, 3),
          "per_segment": per_seg.reset_index().to_dict(orient="records"), "stability_ari": {a: round(float(b), 3) for a, b in stab.items()},
          "meaning_on_test": {"churn_spread_points": round(churn_spread, 1), "cramers_v": round(cramers_v, 3), "spend_ratio": round(spend_ratio, 2),
                              "epsilon_squared": round(eps2, 3), "by_segment": res.reset_index().to_dict(orient="records")},
          "verdict": verdict}
with open(f"{OUT}/segmentation_evaluation.json", "w") as f: json.dump(result, f, indent=2, default=float)
sep_tab.to_csv(f"{OUT}/segmentation_separation_metrics.csv"); per_seg.to_csv(f"{OUT}/segmentation_per_segment.csv")
print(f"\nDONE. Saved segmentation_evaluation.json, 2 tables and 1 chart in {os.path.abspath(OUT)}")
```

#### Objective 4: customer lifetime value

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 14: EVALUATE MODEL - OBJECTIVE 4: CUSTOMER LIFETIME VALUE
# Scores the 5 CLV models on the TEST set (predictions from step 12), then judges the champion chosen in step 11:
#   1. metrics for all models in dollars (MAE, RMSE, median error, R2, WAPE, bias) against a "predict the median" baseline
#   2. bootstrap ranges (how much the scores could move on new customers) and the gain over the baseline
#   3. where the model is wrong: error by spend size (big spenders are the hard part)
#   4. business view: how much of the real spend sits in the customers the model ranks highest
#   5. a plain verdict
# Saves charts and clv_evaluation.json to ./clv_outputs.
# Needs: pandas numpy matplotlib scipy scikit-learn joblib. Set SHOW = False to save charts only.
# =====================================================================
# %% 0. SETTINGS AND LOAD
import os, json, pickle, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

warnings.filterwarnings("ignore")
DATA, OUT, SHOW, SEED = "abt_data", "clv_outputs", True, 42
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
BLUE, RED, GREY = "#2F6DB5", "#C0392B", "#9AA5B1"

with open(f"{DATA}/ready_clv.pkl", "rb") as f: ready = pickle.load(f)
bundle = joblib.load(f"{DATA}/clv_models.joblib")
pred = pd.read_pickle(f"{DATA}/clv_test_predictions.pkl")
models, champion = bundle["models"], bundle["champion"]
y_train, y_test = ready["y_train"], ready["y_test"]
y = pred["actual_spend_usd"].values
assert np.allclose(y, y_test.values)
baseline_pred = np.full(len(y), float(y_train.median()))        # the typical TRAIN customer, for everyone
champ = pred["champion_pred"].values

def save(name):
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight")
    if SHOW: plt.show()
    plt.close()

def scores(t, p):
    return {"mae": mean_absolute_error(t, p), "rmse": float(np.sqrt(mean_squared_error(t, p))), "median_ae": median_absolute_error(t, p),
            "r2": r2_score(t, p), "wape_%": 100 * np.abs(t - p).sum() / t.sum(), "bias": float((p - t).mean())}

# %% 1. METRICS FOR ALL MODELS
tab = pd.DataFrame({n: scores(y, pred[f"{n}_pred"].values) for n in models}).T
tab.loc["Baseline (median)"] = scores(y, baseline_pred)
tab = tab.sort_values("mae"); tab.index.name = "model"
tab["spearman"] = [spearmanr(y, pred[f"{n}_pred"])[0] if n in models else np.nan for n in tab.index]
print(f"TEST SET: {len(y):,} customers | average real spend {y.mean():,.0f} USD | median {np.median(y):,.0f} USD\n")
print(tab.round(3).to_string())
print("\nMAE = average dollar error per customer | RMSE punishes big misses | median_ae = the typical customer's error | R2 = share of variation explained")
print("WAPE = total absolute error as % of total spend | bias = average predicted minus actual (negative = under-predicts) | spearman = how well the ranking of customers is right")
print(f"\nChampion from step 11 (chosen on train cross-validation): {champion}")
best_test = tab.drop(index="Baseline (median)").index[0]
if best_test != champion:
    print(f"Note: {best_test} scores slightly better on the test set, but the champion stays {champion} because picking by test score would leak the test set.")

# %% 2. BOOTSTRAP RANGES FOR THE CHAMPION AND ITS GAIN OVER THE BASELINE
rng = np.random.default_rng(SEED)
b = []
for _ in range(1000):
    i = rng.integers(0, len(y), len(y))
    m, bm = mean_absolute_error(y[i], champ[i]), mean_absolute_error(y[i], baseline_pred[i])
    b.append((m, r2_score(y[i], champ[i]), bm - m, float(np.sqrt(mean_squared_error(y[i], champ[i])))))
b = np.array(b)
ci = {"mae": np.percentile(b[:, 0], [2.5, 97.5]).round(1).tolist(), "r2": np.percentile(b[:, 1], [2.5, 97.5]).round(3).tolist(),
      "mae_gain_vs_baseline": np.percentile(b[:, 2], [2.5, 97.5]).round(1).tolist(), "rmse": np.percentile(b[:, 3], [2.5, 97.5]).round(1).tolist()}
print(f"\n95% bootstrap range for {champion}: MAE {ci['mae']} | RMSE {ci['rmse']} | R2 {ci['r2']}")
print(f"Dollars saved per customer versus the baseline (MAE gain): {ci['mae_gain_vs_baseline']}  (if the low end is above 0, the model beats the baseline for sure)")

# %% 3. WHERE IS THE MODEL WRONG? ERROR BY SPEND SIZE
err = pd.DataFrame({"actual": y, "predicted": champ}); err["error"] = err["predicted"] - err["actual"]
err["spend_band"] = pd.qcut(err["actual"], 4, labels=["Q1 lowest spenders", "Q2", "Q3", "Q4 biggest spenders"], duplicates="drop")
band = err.groupby("spend_band", observed=True).agg(customers=("actual", "count"), avg_actual=("actual", "mean"), avg_predicted=("predicted", "mean"),
                                                    mae=("error", lambda e: e.abs().mean()), bias=("error", "mean")).round(0)
print(f"\nERROR BY REAL SPEND SIZE ({champion})\n", band.to_string())
within = {f"within_{p}%": round(100 * float((np.abs(err["error"]) <= p / 100 * err["actual"].clip(lower=1)).mean()), 1) for p in (25, 50)}
within["within_500_usd"] = round(100 * float((err["error"].abs() <= 500).mean()), 1)
print("\nShare of customers predicted within a tolerance:", within)

# %% 4. BUSINESS VIEW: HOW MUCH REAL SPEND SITS IN THE TOP-RANKED CUSTOMERS
order, best = np.argsort(-champ), np.argsort(-y)
rows = []
for pct in [5, 10, 20, 30, 40, 50]:
    k = int(round(len(y) * pct / 100))
    rows.append({"top_%_by_prediction": pct, "customers": k, "share_of_all_spend_%": round(100 * y[order[:k]].sum() / y.sum(), 1),
                 "best_possible_%": round(100 * y[best[:k]].sum() / y.sum(), 1), "random_%": pct})
topk = pd.DataFrame(rows)
print("\nVALUE PLAN: share of next-year spend held by the customers the model ranks highest\n", topk.to_string(index=False))
dec = pd.qcut(pd.Series(champ).rank(method="first"), 10, labels=False)
calib = pd.DataFrame({"predicted": champ, "actual": y, "decile": dec}).groupby("decile").mean().round(0)
calib.index = [f"D{int(i)+1}" for i in calib.index]
print("\nCalibration by predicted-value decile (D10 = highest): average predicted vs average actual spend\n", calib.to_string())

# %% 5. CHARTS
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
ax[0].scatter(y, champ, s=14, alpha=.5, color=BLUE); top = max(y.max(), champ.max()) * 1.05
ax[0].plot([0, top], [0, top], "--", color=GREY); ax[0].set_xlabel("actual spend (USD)"); ax[0].set_ylabel("predicted spend (USD)"); ax[0].set_title(f"Predicted vs actual - {champion}")
ax[1].hist(err["error"], bins=30, color=BLUE, alpha=.8); ax[1].axvline(0, color=RED, lw=1.2)
ax[1].set_xlabel("prediction error in USD (predicted - actual)"); ax[1].set_title("Residuals (left = under-predicts)")
share = [0] + [100 * y[order[:int(len(y) * p / 100)]].sum() / y.sum() for p in range(1, 101)]
ideal = [0] + [100 * y[best[:int(len(y) * p / 100)]].sum() / y.sum() for p in range(1, 101)]
ax[2].plot([0, 100], [0, 100], "--", color=GREY, label="random"); ax[2].plot(range(101), ideal, color=GREY, lw=1.2, label="perfect ranking")
ax[2].plot(range(101), share, color=BLUE, lw=2.5, label=champion); ax[2].set_xlabel("% of customers (highest predicted first)"); ax[2].set_ylabel("% of all spend captured")
ax[2].set_title("Cumulative gains"); ax[2].legend(); save("14_1_scatter_residuals_gains")

fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))
tab[["mae", "rmse"]].plot(kind="bar", ax=ax[0], width=.8, color=[BLUE, RED]); ax[0].set_title("Dollar error on test (lower is better)"); ax[0].tick_params(axis="x", rotation=25); ax[0].set_ylabel("USD")
ax[1].bar(range(len(tab)), tab["r2"], color=[RED if n == champion else BLUE for n in tab.index]); ax[1].axhline(0, color="black", lw=.8)
ax[1].set_xticks(range(len(tab))); ax[1].set_xticklabels(tab.index, rotation=25, ha="right"); ax[1].set_title("R2 on test (0 = no better than average)")
ax[2].plot([0, calib.values.max() * 1.05], [0, calib.values.max() * 1.05], "--", color=GREY); ax[2].plot(calib["predicted"], calib["actual"], "o-", color=BLUE)
ax[2].set_xlabel("average predicted (per decile)"); ax[2].set_ylabel("average actual"); ax[2].set_title(f"Calibration - {champion}"); save("14_2_metrics_and_calibration")

# %% 6. VERDICT
c, bl = tab.loc[champion], tab.loc["Baseline (median)"]
beats = ci["mae_gain_vs_baseline"][0] > 0
if beats and c["r2"] >= 0.5 and c["spearman"] >= 0.7: verdict = "STRONG: clearly beats the baseline and ranks customers well. Safe to use for value-based decisions."
elif beats and c["r2"] >= 0.25: verdict = "USEFUL: clearly beats the baseline, but a large share of spend is still unexplained. Use it to rank and group customers, not to promise exact dollar amounts."
elif beats: verdict = "WEAK: beats the baseline only slightly. Use with caution and revisit the features."
else: verdict = "NOT PROVEN: it does not beat the baseline with confidence. Do not use it for decisions until the features are improved."
print(f"\nVERDICT for {champion}: {verdict}")
print(f"  MAE {c['mae']:,.0f} vs baseline {bl['mae']:,.0f} ({100 * (c['mae'] / bl['mae'] - 1):+.1f}%) | R2 {c['r2']:.3f} | ranking (Spearman) {c['spearman']:.3f}")
if c["bias"] < -0.1 * y.mean(): print("  Note: the model under-predicts on average; expect real spend above its forecast, mostly from the biggest spenders.")

# %% 7. SAVE THE RESULT
result = {"objective": "Customer lifetime value", "task": "regression", "champion": champion, "test_customers": int(len(y)), "target": "next_12m_spend_usd",
          "avg_actual_spend": round(float(y.mean()), 2), "verdict": verdict, "metrics_champion": {k: round(float(v), 4) for k, v in c.items()},
          "metrics_baseline": {k: round(float(v), 4) for k, v in bl.items() if k != "spearman"}, "bootstrap_95": ci, "tolerance": within,
          "gains_table": topk.to_dict(orient="records"), "error_by_spend_band": band.reset_index().astype({"spend_band": str}).to_dict(orient="records"),
          "all_models": {n: {k: round(float(v), 4) for k, v in r.items() if pd.notna(v)} for n, r in tab.iterrows()}}
with open(f"{OUT}/clv_evaluation.json", "w") as f: json.dump(result, f, indent=2)
tab.round(4).to_csv(f"{OUT}/clv_test_metrics.csv"); topk.to_csv(f"{OUT}/clv_gains_table.csv", index=False)
print(f"\nDONE. Saved clv_evaluation.json, clv_test_metrics.csv, clv_gains_table.csv and 2 charts in {os.path.abspath(OUT)}")
```

#### Objective 5: service cost

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 14: EVALUATE MODEL - OBJECTIVE 5: SERVICE COST
# Scores the 5 service cost models on the TEST set (predictions from step 12), then judges the champion chosen in step 11:
#   1. metrics for all models in dollars (MAE, RMSE, median error, R2, WAPE, bias) against a "predict the median" baseline
#   2. bootstrap ranges (how much the scores could move on new jobs, resampling whole vehicles) and the gain over the baseline
#   3. where the model is wrong: error by job cost size (the very expensive repairs are the hard part)
#   4. business view: how much of the real cost sits in the jobs the model ranks as most expensive (quoting and parts planning)
#   5. a plain verdict
# Saves charts and service_evaluation.json to ./service_outputs.
# Needs: pandas numpy matplotlib scipy scikit-learn joblib. Set SHOW = False to save charts only.
# =====================================================================
# %% 0. SETTINGS AND LOAD
import os, json, pickle, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import joblib
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

warnings.filterwarnings("ignore")
DATA, OUT, SHOW, SEED = "abt_data", "service_outputs", True, 42
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
BLUE, RED, GREY = "#2F6DB5", "#C0392B", "#9AA5B1"

with open(f"{DATA}/ready_service.pkl", "rb") as f: ready = pickle.load(f)
bundle = joblib.load(f"{DATA}/service_models.joblib")
pred = pd.read_pickle(f"{DATA}/service_test_predictions.pkl")
models, champion = bundle["models"], bundle["champion"]
y_train, y_test = ready["y_train"], ready["y_test"]
y = pred["actual_cost_usd"].values
assert np.allclose(y, y_test.values)
baseline_pred = np.full(len(y), float(y_train.median()))        # the typical TRAIN job, for everyone
champ = pred["champion_pred"].values

def save(name):
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight")
    if SHOW: plt.show()
    plt.close()

def scores(t, p):
    return {"mae": mean_absolute_error(t, p), "rmse": float(np.sqrt(mean_squared_error(t, p))), "median_ae": median_absolute_error(t, p),
            "r2": r2_score(t, p), "wape_%": 100 * np.abs(t - p).sum() / t.sum(), "bias": float((p - t).mean())}

# %% 1. METRICS FOR ALL MODELS
tab = pd.DataFrame({n: scores(y, pred[f"{n}_pred"].values) for n in models}).T
tab.loc["Baseline (median)"] = scores(y, baseline_pred)
tab = tab.sort_values("mae"); tab.index.name = "model"
tab["spearman"] = [spearmanr(y, pred[f"{n}_pred"])[0] if n in models else np.nan for n in tab.index]
print(f"TEST SET: {len(y):,} service jobs | average real cost {y.mean():,.0f} USD | median {np.median(y):,.0f} USD\n")
print(tab.round(3).to_string())
print("\nMAE = average dollar error per job | RMSE punishes big misses | median_ae = the typical job's error | R2 = share of variation explained")
print("WAPE = total absolute error as % of total cost | bias = average predicted minus actual (negative = under-predicts) | spearman = how well the ranking of jobs by cost is right")
print(f"\nChampion from step 11 (chosen on train cross-validation): {champion}")
best_test = tab.drop(index="Baseline (median)").index[0]
if best_test != champion:
    print(f"Note: {best_test} scores slightly better on the test set, but the champion stays {champion} because picking by test score would leak the test set.")

# %% 2. BOOTSTRAP RANGES FOR THE CHAMPION AND ITS GAIN OVER THE BASELINE
rng = np.random.default_rng(SEED)
veh = pred["vehicle_id"].values
groups = {v: np.where(veh == v)[0] for v in np.unique(veh)}
keys = list(groups)
b = []
for _ in range(1000):
    i = np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])     # resample whole vehicles, not single jobs
    m, bm = mean_absolute_error(y[i], champ[i]), mean_absolute_error(y[i], baseline_pred[i])
    b.append((m, r2_score(y[i], champ[i]), bm - m, float(np.sqrt(mean_squared_error(y[i], champ[i])))))
b = np.array(b)
ci = {"mae": np.percentile(b[:, 0], [2.5, 97.5]).round(1).tolist(), "r2": np.percentile(b[:, 1], [2.5, 97.5]).round(3).tolist(),
      "mae_gain_vs_baseline": np.percentile(b[:, 2], [2.5, 97.5]).round(1).tolist(), "rmse": np.percentile(b[:, 3], [2.5, 97.5]).round(1).tolist()}
print(f"\n95% bootstrap range for {champion}: MAE {ci['mae']} | RMSE {ci['rmse']} | R2 {ci['r2']}")
print(f"Dollars saved per job versus the baseline (MAE gain): {ci['mae_gain_vs_baseline']}  (if the low end is above 0, the model beats the baseline for sure)")

# %% 3. WHERE IS THE MODEL WRONG? ERROR BY COST SIZE
err = pd.DataFrame({"actual": y, "predicted": champ}); err["error"] = err["predicted"] - err["actual"]
err["cost_band"] = pd.qcut(err["actual"], 4, labels=["Q1 cheapest jobs", "Q2", "Q3", "Q4 costliest jobs"], duplicates="drop")
band = err.groupby("cost_band", observed=True).agg(jobs=("actual", "count"), avg_actual=("actual", "mean"), avg_predicted=("predicted", "mean"),
                                                    mae=("error", lambda e: e.abs().mean()), bias=("error", "mean")).round(0)
print(f"\nERROR BY REAL COST SIZE ({champion})\n", band.to_string())
within = {f"within_{p}%": round(100 * float((np.abs(err["error"]) <= p / 100 * err["actual"].clip(lower=1)).mean()), 1) for p in (25, 50)}
within["within_100_usd"] = round(100 * float((err["error"].abs() <= 100).mean()), 1)
print("\nShare of jobs predicted within a tolerance:", within)

# %% 4. BUSINESS VIEW: HOW MUCH REAL COST SITS IN THE JOBS RANKED MOST EXPENSIVE
order, best = np.argsort(-champ), np.argsort(-y)
rows = []
for pct in [5, 10, 20, 30, 40, 50]:
    k = int(round(len(y) * pct / 100))
    rows.append({"top_%_by_prediction": pct, "jobs": k, "share_of_all_cost_%": round(100 * y[order[:k]].sum() / y.sum(), 1),
                 "best_possible_%": round(100 * y[best[:k]].sum() / y.sum(), 1), "random_%": pct})
topk = pd.DataFrame(rows)
print("\nCOST PLAN: share of total service cost held by the jobs the model ranks most expensive\n", topk.to_string(index=False))
dec = pd.qcut(pd.Series(champ).rank(method="first"), 10, labels=False)
calib = pd.DataFrame({"predicted": champ, "actual": y, "decile": dec}).groupby("decile").mean().round(0)
calib.index = [f"D{int(i)+1}" for i in calib.index]
print("\nCalibration by predicted-cost decile (D10 = highest): average predicted vs average actual cost\n", calib.to_string())

# %% 5. CHARTS
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
ax[0].scatter(y, champ, s=14, alpha=.5, color=BLUE); top = max(y.max(), champ.max()) * 1.05
ax[0].plot([0, top], [0, top], "--", color=GREY); ax[0].set_xlabel("actual cost (USD)"); ax[0].set_ylabel("predicted cost (USD)"); ax[0].set_title(f"Predicted vs actual - {champion}")
ax[1].hist(err["error"], bins=30, color=BLUE, alpha=.8); ax[1].axvline(0, color=RED, lw=1.2)
ax[1].set_xlabel("prediction error in USD (predicted - actual)"); ax[1].set_title("Residuals (left = under-predicts)")
share = [0] + [100 * y[order[:int(len(y) * p / 100)]].sum() / y.sum() for p in range(1, 101)]
ideal = [0] + [100 * y[best[:int(len(y) * p / 100)]].sum() / y.sum() for p in range(1, 101)]
ax[2].plot([0, 100], [0, 100], "--", color=GREY, label="random"); ax[2].plot(range(101), ideal, color=GREY, lw=1.2, label="perfect ranking")
ax[2].plot(range(101), share, color=BLUE, lw=2.5, label=champion); ax[2].set_xlabel("% of jobs (highest predicted first)"); ax[2].set_ylabel("% of all cost captured")
ax[2].set_title("Cumulative gains"); ax[2].legend(); save("14_1_scatter_residuals_gains")

fig, ax = plt.subplots(1, 3, figsize=(17, 4.3))
tab[["mae", "rmse"]].plot(kind="bar", ax=ax[0], width=.8, color=[BLUE, RED]); ax[0].set_title("Dollar error on test (lower is better)"); ax[0].tick_params(axis="x", rotation=25); ax[0].set_ylabel("USD")
ax[1].bar(range(len(tab)), tab["r2"], color=[RED if n == champion else BLUE for n in tab.index]); ax[1].axhline(0, color="black", lw=.8)
ax[1].set_xticks(range(len(tab))); ax[1].set_xticklabels(tab.index, rotation=25, ha="right"); ax[1].set_title("R2 on test (0 = no better than average)")
ax[2].plot([0, calib.values.max() * 1.05], [0, calib.values.max() * 1.05], "--", color=GREY); ax[2].plot(calib["predicted"], calib["actual"], "o-", color=BLUE)
ax[2].set_xlabel("average predicted (per decile)"); ax[2].set_ylabel("average actual"); ax[2].set_title(f"Calibration - {champion}"); save("14_2_metrics_and_calibration")

# %% 6. VERDICT
c, bl = tab.loc[champion], tab.loc["Baseline (median)"]
beats = ci["mae_gain_vs_baseline"][0] > 0
if beats and c["r2"] >= 0.5 and c["spearman"] >= 0.7: verdict = "STRONG: clearly beats the baseline and ranks jobs well. Safe to use for quoting and parts and labour planning."
elif beats and c["r2"] >= 0.25: verdict = "USEFUL: clearly beats the baseline, but a large share of cost is still unexplained. Use it for planning and rough quotes, not for exact prices."
elif beats: verdict = "WEAK: beats the baseline only slightly. Use with caution and revisit the features."
else: verdict = "NOT PROVEN: it does not beat the baseline with confidence. Do not use it for decisions until the features are improved."
print(f"\nVERDICT for {champion}: {verdict}")
print(f"  MAE {c['mae']:,.0f} vs baseline {bl['mae']:,.0f} ({100 * (c['mae'] / bl['mae'] - 1):+.1f}%) | R2 {c['r2']:.3f} | ranking (Spearman) {c['spearman']:.3f}")
if c["bias"] < -0.1 * y.mean(): print("  Note: the model under-predicts on average; expect real cost above its forecast, mostly from the most expensive repairs.")

# %% 7. SAVE THE RESULT
result = {"objective": "Service cost", "task": "regression", "champion": champion, "test_jobs": int(len(y)), "target": "total_cost_usd",
          "avg_actual_cost": round(float(y.mean()), 2), "verdict": verdict, "metrics_champion": {k: round(float(v), 4) for k, v in c.items()},
          "metrics_baseline": {k: round(float(v), 4) for k, v in bl.items() if k != "spearman"}, "bootstrap_95": ci, "tolerance": within,
          "gains_table": topk.to_dict(orient="records"), "error_by_cost_band": band.reset_index().astype({"cost_band": str}).to_dict(orient="records"),
          "all_models": {n: {k: round(float(v), 4) for k, v in r.items() if pd.notna(v)} for n, r in tab.iterrows()}}
with open(f"{OUT}/service_evaluation.json", "w") as f: json.dump(result, f, indent=2)
tab.round(4).to_csv(f"{OUT}/service_test_metrics.csv"); topk.to_csv(f"{OUT}/service_gains_table.csv", index=False)
print(f"\nDONE. Saved service_evaluation.json, service_test_metrics.csv, service_gains_table.csv and 2 charts in {os.path.abspath(OUT)}")
```

---

## 18. Business Analysis

Script: `python/15_business_analysis.py`. It turns predictions into decisions and dollars, using test-set predictions only (1,200 customers; 9,821 transactions; 2,810 service jobs). The results are written back to SQL Server as `bi_*` tables for Power BI.

**Churn: who to contact first**

| Contact the riskiest | Customers | Churners reached | Share of all churners |
|---|---|---|---|
| 5% | 60 | 43 | 11.7% |
| 10% | 120 | 79 | 21.5% |
| **20%** | **240** | **134** | **36.4%** |
| 30% | 360 | 173 | 47.0% |
| 50% | 600 | 248 | 67.4% |

Random contact would reach 20% of churners at the 20% mark, so the model nearly doubles the reach.

**Fraud: how big should the review team be**

| Review the riskiest | Transactions | Fraud cases caught | Fraud value caught |
|---|---|---|---|
| 1% | 98 | 28.5% | **72.4%** |
| 2% | 196 | 43.0% | 84.6% |
| **5%** | **491** | **59.8%** | **95.9%** |
| 10% | 982 | 73.5% | 98.4% |

Fraud money is concentrated in a few large transactions, so a small queue covers almost all of it.

**Segments (all 6,000 customers)**

| Segment | Share of customers | Share of next-year spend | Churn rate | Average spend |
|---|---|---|---|---|
| 0 | 33.6% | 22.5% | 34.3% | $3,036 |
| 1 | 35.0% | 19.1% | 34.5% | $2,477 |
| **2** | **31.5%** | **58.4%** | **22.5%** | **$8,404** |

**Service cost: forecast against actual (test jobs).** Whole workshop: $1.83M predicted against $1.90M actual, **-3.6%**.

| Service type | Jobs | Actual cost | Forecast error |
|---|---|---|---|
| Routine Maintenance | 1,159 | $384,225 | -1.0% |
| Battery/EV System | 277 | $351,241 | -6.5% |
| Engine Repair | 220 | $305,510 | -2.9% |
| Tyres | 399 | $235,978 | -4.0% |
| Transmission | 137 | $228,595 | -4.0% |
| Body Work | 231 | $199,864 | -6.0% |
| Brake Repair | 387 | $198,962 | -1.8% |

**Action groups for customers:** combining churn risk with predicted value gives four groups: **Protect now** (high value, likely to leave), **Grow** (high value, loyal), **Low-cost win-back** (likely to leave, lower value) and **Maintain**. Churn probabilities are inflated by class weights, so `revenue_risk_index_usd` is used for ranking only and real churner spend is shown next to it.

**Written back to SQL Server for Power BI:** `bi_customer_actions`, `bi_action_summary`, `bi_segment_profile`, `bi_kpi_by_group`, `bi_churn_retention_roi`, `bi_fraud_review_roi`, `bi_service_forecast_by_type`, `bi_service_forecast_by_country`.

### Charts

<p align="center">
  <img src="images/business_15_1_churn_campaign.png" alt="Retention campaign: net benefit and churners caught by contact depth" width="100%">
</p>
<p align="center"><em>Retention campaign: net benefit and churners caught by contact depth</em></p>

<p align="center">
  <img src="images/business_15_2_action_groups.png" alt="Customers per action group and real next-year spend lost to churn" width="100%">
</p>
<p align="center"><em>Customers per action group and real next-year spend lost to churn</em></p>

<p align="center">
  <img src="images/business_15_3_fraud_review.png" alt="Fraud review: what a review team catches and what it costs" width="100%">
</p>
<p align="center"><em>Fraud review: what a review team catches and what it costs</em></p>

<p align="center">
  <img src="images/business_15_4_segments.png" alt="Segment size against spend share, and churn rate by segment" width="100%">
</p>
<p align="center"><em>Segment size against spend share, and churn rate by segment</em></p>

<p align="center">
  <img src="images/business_15_5_service_forecast.png" alt="Workshop cost: actual against predicted by service type, and forecast error by branch country" width="100%">
</p>
<p align="center"><em>Workshop cost: actual against predicted by service type, and forecast error by branch country</em></p>

<p align="center">
  <img src="images/business_15_6_kpi_churn.png" alt="Churn rate by customer group" width="100%">
</p>
<p align="center"><em>Churn rate by customer group</em></p>

### Code: business analysis (Python)

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 15: BUSINESS ANALYSIS
# Turns the five models into decisions and dollar values, using only TEST-set predictions
# (customers, transactions and jobs the models never saw), so the numbers are honest.
#   A. Churn      : which customers to contact, and what the retention campaign could return
#   B. Value + risk: who to protect, grow or maintain (churn risk x predicted lifetime value)
#   C. Fraud      : how much fraud money a fixed review team catches, and what it costs
#   D. Segments   : size, churn and spend of each customer segment (all customers)
#   E. Service    : how well total workshop cost is forecast by service type and country
#   F. KPIs       : churn and spend by country, loyalty tier and customer type
# The dollar results rest on a few ASSUMPTIONS below (campaign cost, save rate...). They are
# not facts from the data: edit them to your own figures and re-run.
# Saves CSVs AND 6 business charts to ./business_outputs and, if WRITE_TO_SQL = True, the same tables to SQL Server
# (bi_* tables in Vantage_Automotive) for Power BI.
# Needs: pandas numpy matplotlib sqlalchemy pyodbc (SQL write only). Set SHOW = False to save charts without displaying them.
# =====================================================================
# %% 0. SETTINGS
import os, json, pickle, warnings, urllib.parse
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
DATA, OUT = "abt_data", "business_outputs"
os.makedirs(OUT, exist_ok=True)
WRITE_TO_SQL, SHOW = True, True
SERVER, DATABASE = r"localhost\SQLEXPRESS", "Vantage_Automotive"

# ---- ASSUMPTIONS (edit these) ----
SAVE_RATE        = 0.25    # share of contacted would-be churners the retention offer actually keeps
CONTACT_COST_USD = 20      # cost of one retention contact or offer
REVIEW_COST_USD  = 8       # cost of an analyst reviewing one flagged transaction
HIGH_VALUE_SHARE = 0.25    # the top 25% by predicted spend count as "high value"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
BLUE, RED, GREY, GREEN = "#2F6DB5", "#C0392B", "#9AA5B1", "#2E8B57"
def save(name):
    plt.tight_layout(); plt.savefig(f"{OUT}/{name}.png", dpi=150, bbox_inches="tight")
    if SHOW: plt.show()
    plt.close()
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

def pick(name): return pd.read_pickle(f"{DATA}/{name}.pkl")
def js(path):
    with open(path) as f: return json.load(f)
tables = {}
def keep(name, df):
    tables[name] = df; df.to_csv(f"{OUT}/{name}.csv", index=False)

cust    = pick("abt_customer")
churn_p = pick("churn_test_predictions"); clv_p = pick("clv_test_predictions")
fraud_p = pick("fraud_test_predictions"); svc_p = pick("service_test_predictions")
seg_all = pick("segmentation_all_customers")
abt_fraud, abt_svc = pick("abt_fraud"), pick("abt_service")
churn_json, fraud_json, seg_json = js("churn_outputs/churn_evaluation.json"), js("fraud_outputs/fraud_evaluation.json"), js("segmentation_outputs/segmentation_evaluation.json")
print(f"Assumptions: save rate {SAVE_RATE:.0%} | contact cost ${CONTACT_COST_USD} | review cost ${REVIEW_COST_USD} | high value = top {HIGH_VALUE_SHARE:.0%}")

# %% A. CHURN: RETENTION CAMPAIGN RETURN (test customers, riskiest first)
t = churn_p[["customer_id", "actual_churned", "champion_proba"]].merge(cust[["customer_id", "next_12m_spend_usd"]], on="customer_id", how="left")
t = t.sort_values("champion_proba", ascending=False).reset_index(drop=True)
total_churn_spend = t.loc[t.actual_churned == 1, "next_12m_spend_usd"].sum()
rows = []
for pct in [5, 10, 20, 30, 40, 50]:
    k = int(round(len(t) * pct / 100)); top = t.head(k); caught = top[top.actual_churned == 1]
    saved = caught["next_12m_spend_usd"].sum() * SAVE_RATE; cost = k * CONTACT_COST_USD
    rows.append({"contact_top_%": pct, "customers_contacted": k, "churners_caught": len(caught), "share_of_churners_%": round(100 * len(caught) / max(t.actual_churned.sum(), 1), 1),
                 "spend_of_caught_churners_usd": round(caught["next_12m_spend_usd"].sum()), "revenue_retained_usd": round(saved), "campaign_cost_usd": cost, "net_benefit_usd": round(saved - cost)})
roi_churn = pd.DataFrame(rows); keep("bi_churn_retention_roi", roi_churn)
print(f"\nA. CHURN RETENTION CAMPAIGN (test: {len(t):,} customers, {int(t.actual_churned.sum())} churners, ${total_churn_spend:,.0f} of next-year spend belongs to churners)\n", roi_churn.to_string(index=False))
best = roi_churn.loc[roi_churn.net_benefit_usd.idxmax()]
print(f"-> Best contact depth on the test set: top {int(best['contact_top_%'])}% of customers, net benefit ${best.net_benefit_usd:,.0f} (at a {SAVE_RATE:.0%} save rate).")

# %% B. VALUE x RISK: WHO TO PROTECT, GROW OR MAINTAIN
thr = churn_json.get("threshold", 0.5)
b = churn_p[["customer_id", "champion_proba"]].rename(columns={"champion_proba": "churn_probability"}).merge(
    clv_p[["customer_id", "champion_pred", "actual_spend_usd"]].rename(columns={"champion_pred": "predicted_spend_usd"}), on="customer_id")
b = b.merge(cust[["customer_id", "country", "loyalty_tier", "customer_type", "churned"]], on="customer_id", how="left").merge(seg_all[["customer_id", "segment"]], on="customer_id", how="left")
hv_cut = b["predicted_spend_usd"].quantile(1 - HIGH_VALUE_SHARE)
b["high_risk"], b["high_value"] = b.churn_probability >= thr, b.predicted_spend_usd >= hv_cut
# the churn probabilities come from a class-weighted model, so they run higher than real churn rates: use this as a RANKING index, not as a dollar forecast
b["revenue_risk_index_usd"] = (b.churn_probability * b.predicted_spend_usd).round(0)
b["actual_churner_spend_usd"] = np.where(b.churned == 1, b.actual_spend_usd, 0.0)
b["action"] = np.select([b.high_risk & b.high_value, ~b.high_risk & b.high_value, b.high_risk & ~b.high_value],
                        ["PROTECT NOW (high value, likely to leave)", "GROW (high value, loyal)", "LOW-COST WIN-BACK (likely to leave, lower value)"], "MAINTAIN")
actions = b.sort_values("revenue_risk_index_usd", ascending=False); keep("bi_customer_actions", actions)
summ = actions.groupby("action").agg(customers=("customer_id", "count"), avg_churn_probability=("churn_probability", "mean"), avg_predicted_spend=("predicted_spend_usd", "mean"),
                                     actual_churn_rate=("churned", "mean"), actual_avg_spend=("actual_spend_usd", "mean"), actual_spend_lost_to_churn_usd=("actual_churner_spend_usd", "sum"), risk_index_usd=("revenue_risk_index_usd", "sum")).round(2).reset_index()
keep("bi_action_summary", summ)
print(f"\nB. CUSTOMER ACTION GROUPS (churn threshold {thr:.2f}, high value = predicted spend >= {hv_cut:,.0f})\n", summ.to_string(index=False))
print("   Read actual_spend_lost_to_churn_usd as the real dollars. risk_index_usd is only for ranking customers: churn probabilities are inflated by class weighting, so the index overstates dollars.")
print("\nTop 10 customers to protect first (largest revenue risk index = churn probability x predicted spend):\n", actions[["customer_id", "churn_probability", "predicted_spend_usd", "revenue_risk_index_usd", "action"]].head(10).round(2).to_string(index=False))

# %% C. FRAUD: WHAT A FIXED REVIEW TEAM CATCHES, IN DOLLARS
f = fraud_p[["txn_id", "actual_fraud", "champion_proba"]].merge(abt_fraud[["txn_id", "amount_usd"]], on="txn_id", how="left").sort_values("champion_proba", ascending=False).reset_index(drop=True)
fraud_total = f.loc[f.actual_fraud == 1, "amount_usd"].sum(); n_fraud = int(f.actual_fraud.sum())
rows = []
for pct in [1, 2, 5, 10, 20, 30]:
    k = max(int(round(len(f) * pct / 100)), 1); top = f.head(k); hit = top[top.actual_fraud == 1]
    rows.append({"review_top_%": pct, "transactions_reviewed": k, "fraud_cases_caught": len(hit), "share_of_fraud_cases_%": round(100 * len(hit) / max(n_fraud, 1), 1),
                 "fraud_usd_caught": round(hit.amount_usd.sum()), "share_of_fraud_usd_%": round(100 * hit.amount_usd.sum() / max(fraud_total, 1), 1),
                 "review_cost_usd": k * REVIEW_COST_USD, "net_benefit_usd": round(hit.amount_usd.sum() - k * REVIEW_COST_USD)})
roi_fraud = pd.DataFrame(rows); keep("bi_fraud_review_roi", roi_fraud)
print(f"\nC. FRAUD REVIEW (test: {len(f):,} transactions, {n_fraud} fraud cases, ${fraud_total:,.0f} fraud value)\n", roi_fraud.to_string(index=False))
print("   (money counted as caught = value of the fraudulent transactions a reviewer would stop; recoveries and customer impact are not included)")

# %% D. SEGMENTS: SIZE, CHURN AND SPEND (all customers)
s = seg_all[["customer_id", "segment"]].merge(cust[["customer_id", "churned", "next_12m_spend_usd", "country", "loyalty_tier", "customer_type"]], on="customer_id")
prof = s.groupby("segment").agg(customers=("customer_id", "count"), churn_rate=("churned", "mean"), avg_next_12m_spend=("next_12m_spend_usd", "mean"), total_next_12m_spend=("next_12m_spend_usd", "sum")).reset_index()
prof["share_of_customers_%"] = 100 * prof.customers / prof.customers.sum(); prof["share_of_spend_%"] = 100 * prof.total_next_12m_spend / prof.total_next_12m_spend.sum()
prof = prof.round(3); keep("bi_segment_profile", prof)
verdict = seg_json.get("verdict", "n/a"); verdict = "\n   - " + "\n   - ".join(verdict) if isinstance(verdict, list) else verdict
print("\nD. SEGMENT PROFILE (all customers)\n", prof.to_string(index=False))
print("   Segmentation verdict from step 14:", verdict)
print("   Use the segments for targeting only if the verdict says the segments are clear and the churn or spend columns genuinely differ between them.")

# %% E. SERVICE: HOW WELL IS TOTAL WORKSHOP COST FORECAST?
v = svc_p.merge(abt_svc[["service_id", "service_type", "branch_country"]], on="service_id", how="left")
def forecast(group):
    g = v.groupby(group).agg(jobs=("service_id", "count"), actual_total=("actual_cost_usd", "sum"), predicted_total=("champion_pred", "sum")).reset_index()
    g["forecast_error_%"] = 100 * (g.predicted_total - g.actual_total) / g.actual_total
    return g.round(1).sort_values("actual_total", ascending=False)
fc_type, fc_country = forecast("service_type"), forecast("branch_country")
all_row = pd.DataFrame([{"jobs": len(v), "actual_total": v.actual_cost_usd.sum(), "predicted_total": v.champion_pred.sum()}]); all_row["forecast_error_%"] = 100 * (all_row.predicted_total - all_row.actual_total) / all_row.actual_total
keep("bi_service_forecast_by_type", fc_type); keep("bi_service_forecast_by_country", fc_country)
print(f"\nE. SERVICE COST FORECAST (test: {len(v):,} jobs). Whole workshop: actual ${all_row.actual_total[0]:,.0f} vs predicted ${all_row.predicted_total[0]:,.0f} ({all_row['forecast_error_%'][0]:+.1f}%)\n", fc_type.to_string(index=False), "\n\n", fc_country.to_string(index=False))
print("   Totals are usually forecast better than single jobs: errors on cheap and expensive jobs partly cancel out.")

# %% F. KPIs BY COUNTRY, LOYALTY TIER AND CUSTOMER TYPE (all customers)
parts = []
for dim in ["country", "loyalty_tier", "customer_type"]:
    g = cust.groupby(dim, dropna=False).agg(customers=("customer_id", "count"), churn_rate=("churned", "mean"), avg_next_12m_spend=("next_12m_spend_usd", "mean"), total_next_12m_spend=("next_12m_spend_usd", "sum")).reset_index()
    g.insert(0, "dimension", dim); g = g.rename(columns={dim: "value"}); g["value"] = g["value"].astype("string").fillna("Missing"); parts.append(g)
kpi = pd.concat(parts, ignore_index=True).round(3); keep("bi_kpi_by_group", kpi)
print("\nF. KPIs BY GROUP\n", kpi.to_string(index=False))

# %% CHARTS (one per analysis; Power BI will rebuild the same views from the bi_* tables)
fig, ax = plt.subplots(figsize=(8, 4.2)); x = roi_churn["contact_top_%"].astype(str) + "%"
ax.bar(x, roi_churn.net_benefit_usd, color=BLUE); ax.set_ylabel("net benefit (USD)"); ax.set_xlabel("share of customers contacted (riskiest first)")
ax2 = ax.twinx(); ax2.plot(x, roi_churn["share_of_churners_%"], "o-", color=RED); ax2.set_ylabel("% of all churners caught"); ax2.set_ylim(0, 100); ax2.spines["top"].set_visible(False)
ax.set_title(f"Retention campaign: net benefit and churners caught (save rate {SAVE_RATE:.0%}, ${CONTACT_COST_USD}/contact)"); save("15_1_churn_campaign")

order_ = ["PROTECT NOW (high value, likely to leave)", "LOW-COST WIN-BACK (likely to leave, lower value)", "GROW (high value, loyal)", "MAINTAIN"]
sm = summ.set_index("action").reindex([o for o in order_ if o in set(summ.action)]); lab = [i.split(" (")[0] for i in sm.index]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].bar(lab, sm.customers, color=[RED, "#E39C3A", GREEN, GREY][:len(sm)]); ax[0].set_title("Customers per action group"); ax[0].tick_params(axis="x", rotation=15)
ax[1].bar(lab, sm.actual_spend_lost_to_churn_usd, color=[RED, "#E39C3A", GREEN, GREY][:len(sm)]); ax[1].set_title("Real next-year spend lost to churn (USD)"); ax[1].tick_params(axis="x", rotation=15)
save("15_2_action_groups")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot(roi_fraud["review_top_%"], roi_fraud["share_of_fraud_usd_%"], "o-", color=BLUE, label="fraud value caught"); ax[0].plot(roi_fraud["review_top_%"], roi_fraud["share_of_fraud_cases_%"], "o--", color=RED, label="fraud cases caught")
ax[0].set_xlabel("% of transactions reviewed (riskiest first)"); ax[0].set_ylabel("% of all fraud caught"); ax[0].set_ylim(0, 105); ax[0].legend(); ax[0].set_title("What a review team catches")
ax[1].bar(roi_fraud["review_top_%"].astype(str) + "%", roi_fraud.review_cost_usd, color=GREY, label="review cost"); ax[1].set_ylabel("review cost (USD)"); ax[1].set_xlabel("% of transactions reviewed"); ax[1].set_title("What the review costs")
save("15_3_fraud_review")

x = np.arange(len(prof)); fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].bar(x - .2, prof["share_of_customers_%"], .4, color=BLUE, label="% of customers"); ax[0].bar(x + .2, prof["share_of_spend_%"], .4, color=RED, label="% of next-year spend")
ax[0].set_xticks(x); ax[0].set_xticklabels([f"Segment {i}" for i in prof.segment]); ax[0].legend(); ax[0].set_title("Segment size vs spend share")
ax[1].bar([f"Segment {i}" for i in prof.segment], prof.churn_rate * 100, color=BLUE); ax[1].set_ylabel("churn rate (%)"); ax[1].set_title("Churn rate by segment")
for i, v in enumerate(prof.churn_rate * 100): ax[1].text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
save("15_4_segments")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.4)); ft = fc_type.sort_values("actual_total"); yy = np.arange(len(ft))
ax[0].barh(yy - .2, ft.actual_total, .4, color=BLUE, label="actual"); ax[0].barh(yy + .2, ft.predicted_total, .4, color=RED, label="predicted")
ax[0].set_yticks(yy); ax[0].set_yticklabels(ft.service_type); ax[0].set_xlabel("total cost (USD)"); ax[0].legend(); ax[0].set_title("Workshop cost: actual vs predicted (test)")
fe = fc_country.sort_values("forecast_error_%"); ax[1].barh(fe.branch_country, fe["forecast_error_%"], color=[RED if v < 0 else GREEN for v in fe["forecast_error_%"]])
ax[1].axvline(0, color="black", lw=.8); ax[1].set_xlabel("forecast error (%)  negative = under-forecast"); ax[1].set_title("Forecast error by branch country")
save("15_5_service_forecast")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, dim in zip(ax, ["country", "loyalty_tier", "customer_type"]):
    g = kpi[kpi.dimension == dim].sort_values("churn_rate"); a.barh(g.value, g.churn_rate * 100, color=BLUE); a.set_title(f"Churn rate by {dim.replace('_', ' ')} (%)")
    for i, v in enumerate(g.churn_rate * 100): a.text(v, i, f" {v:.1f}", va="center", fontsize=8)
save("15_6_kpi_churn")

# %% G. WRITE THE TABLES TO SQL SERVER FOR POWER BI
if WRITE_TO_SQL:
    try:
        import pyodbc
        from sqlalchemy import create_engine
        drivers = [d for d in pyodbc.drivers() if "SQL Server" in d]
        pref = [d for d in drivers if "ODBC Driver 18" in d] + [d for d in drivers if "ODBC Driver 17" in d] + drivers
        conn = f"DRIVER={{{pref[0]}}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;TrustServerCertificate=yes;"
        engine = create_engine("mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(conn))
        for name, df in tables.items():
            out = df.copy()
            for c in out.columns:
                if str(out[c].dtype) in ("str", "string", "category"): out[c] = out[c].astype(object)
            out.to_sql(name, engine, schema="dbo", if_exists="replace", index=False)
        print(f"\nWrote {len(tables)} tables to {DATABASE}.dbo: " + ", ".join(tables))
    except Exception as e:
        print("\nSQL write skipped:", repr(e)[:300], "\nThe CSV files are still saved, so nothing is lost.")
print(f"\nDONE. {len(tables)} CSV files and 6 charts saved in {os.path.abspath(OUT)}")
```

---

## 19. Machine Learning Objectives

Script: `python/16_ml_objectives.py`. One scorecard that puts each objective next to its success bar and its result on unseen data. It writes `bi_ml_scorecard` and `bi_ml_model_comparison` to SQL Server for Power BI.

| # | Objective | ML task | Success bar (fixed in advance) | Champion | Result |
|---|---|---|---|---|---|
| 1 | Reduce customer churn | Classification | 95% range beats no-skill and ROC AUC of 0.65 or more | Logistic Regression | **USEFUL** |
| 2 | Detect fraudulent transactions | Classification | 95% range beats no-skill and ROC AUC of 0.80 or more | Logistic Regression | **STRONG** |
| 3 | Segment customers | Clustering | Stability 0.8 or more, silhouette above noise, clear outcome differences | K-Means, k = 3 | **USEFUL (as tiers)** |
| 4 | Forecast lifetime value | Regression | R² of 0.5 or more and ranking correlation of 0.7 or more | Ridge Regression | **STRONG** |
| 5 | Predict service cost | Regression | R² of 0.5 or more and ranking correlation of 0.7 or more | Gradient Boosting | **STRONG** |

**Scorecard: 5 of 5 objectives met their bars (3 STRONG, 2 USEFUL).** A model that does not reach USEFUL is marked "do not deploy" and gets no action plan.

### Charts

<p align="center">
  <img src="images/business_16_ml_objectives_scorecard.png" alt="Each model against its benchmark on the test set" width="100%">
</p>
<p align="center"><em>Each model against its benchmark on the test set</em></p>

### Code: machine learning objectives (Python)

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 16: MACHINE LEARNING OBJECTIVES (SCORECARD)
# Puts the five business objectives side by side and answers, for each one:
#   what was asked, which ML task solves it, which model won, how well it did on the TEST set
#   compared with a no-skill benchmark, whether that gain is real (95% bootstrap range),
#   and whether it is ready to deploy.
# It reads the evaluation files from step 14 (nothing is retrained or re-scored) and checks that the
# files step 17 needs (model, scaler, features) exist.
# Saves ml_objectives_scorecard.csv, ml_model_comparison.csv and a chart to ./business_outputs and,
# if WRITE_TO_SQL = True, the same tables to SQL Server (bi_ml_scorecard, bi_ml_model_comparison).
# Needs: pandas numpy matplotlib sqlalchemy pyodbc (SQL write only). Set SHOW = False to save the chart only.
# =====================================================================
# %% 0. SETTINGS AND LOAD
import os, json, warnings, urllib.parse
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
DATA, OUT = "abt_data", "business_outputs"
WRITE_TO_SQL, SHOW = True, True
SERVER, DATABASE = r"localhost\SQLEXPRESS", "Vantage_Automotive"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
BLUE, RED, GREY, GREEN, AMBER = "#2F6DB5", "#C0392B", "#9AA5B1", "#2E8B57", "#E39C3A"
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 70)

def js(p):
    with open(p) as f: return json.load(f)
J = {"churn": js("churn_outputs/churn_evaluation.json"), "fraud": js("fraud_outputs/fraud_evaluation.json"), "seg": js("segmentation_outputs/segmentation_evaluation.json"),
     "clv": js("clv_outputs/clv_evaluation.json"), "svc": js("service_outputs/service_evaluation.json")}

def row_at(table, key, value):
    for r in table:
        if abs(r[key] - value) < 1e-9: return r
    return table[0]

# %% 1. VERDICT RULES (fixed rules, written before looking at the results)
def classify(j):
    """classification: judged on PR AUC against the no-skill rate, because both churn and fraud are imbalanced."""
    m, bs, base = j["metrics_at_threshold"], j["bootstrap_95"], j["no_skill_pr_auc"]
    above = bs["pr_auc"][0] > base and bs["roc_auc"][0] > 0.5            # the whole 95% range beats no-skill
    if above and m["roc_auc"] >= 0.80: return "STRONG"
    if above and m["roc_auc"] >= 0.65: return "USEFUL"
    if above: return "WEAK"
    return "NOT PROVEN"

def regress(j):
    c, ci = j["metrics_champion"], j["bootstrap_95"]
    return j["verdict"].split(":")[0]                                      # the verdict already written in step 14

def cluster(j):
    stable = min(j["stability_ari"].values()) >= 0.8
    holds = j["separation"]["test customers"]["silhouette"] >= max(0.10, 2 * j["noise_baseline_silhouette"])
    clear = "clear differences" in " ".join(j["verdict"])
    return "USEFUL (as tiers)" if stable and holds and clear else ("WEAK" if (stable and holds) or clear else "NOT PROVEN")

READY = {"STRONG": "DEPLOY", "USEFUL": "DEPLOY WITH CARE", "USEFUL (as tiers)": "DEPLOY WITH CARE", "WEAK": "HOLD - improve features first", "NOT PROVEN": "DO NOT DEPLOY"}

# %% 2. BUILD THE SCORECARD
rows = []
c, f, s, v, w = J["churn"], J["fraud"], J["seg"], J["clv"], J["svc"]
t20 = row_at(c["contact_top_k"], "contact_top_%", 20); f5 = row_at(f["review_budget"], "review_top_%", 5.0)
v20 = row_at(v["gains_table"], "top_%_by_prediction", 20); st = s["meaning_on_test"]; s_t = s["separation"]["test customers"]; s_tr = s["separation"]["train customers"]
rows.append({"no": 1, "business_objective": "Which customers will leave?", "ml_task": "Classification", "target": "churned", "grain": "customer", "split": "stratified 80/20",
             "champion": c["champion"], "test_size": f"{c['test_customers']:,} customers ({c['test_churners']:,} churners)", "primary_metric": "PR AUC",
             "result": c["metrics_at_threshold"]["pr_auc"], "ci_95": str(c["bootstrap_95"]["pr_auc"]), "benchmark": c["no_skill_pr_auc"], "benchmark_name": "no-skill (churn rate)",
             "also": f"ROC AUC {c['metrics_at_threshold']['roc_auc']:.3f} | recall {c['metrics_at_threshold']['recall']:.2f} | precision {c['metrics_at_threshold']['precision']:.2f} at threshold {c['threshold']:.2f}",
             "business_result": f"Contacting the riskiest 20% reaches {t20['share_of_all_churners_%']}% of churners ({t20['lift_x']}x random)", "verdict": classify(c)})
rows.append({"no": 2, "business_objective": "Which transactions are fraud?", "ml_task": "Classification", "target": "is_fraud", "grain": "transaction", "split": "time-based (train to 2025-06-30, test after)",
             "champion": f["champion"], "test_size": f"{f['test_transactions']:,} transactions ({f['test_fraud_cases']:,} fraud)", "primary_metric": "PR AUC",
             "result": f["metrics_at_threshold"]["pr_auc"], "ci_95": str(f["bootstrap_95"]["pr_auc"]), "benchmark": f["no_skill_pr_auc"], "benchmark_name": "no-skill (fraud rate)",
             "also": f"ROC AUC {f['metrics_at_threshold']['roc_auc']:.3f} | recall {f['metrics_at_threshold']['recall']:.2f} | precision {f['metrics_at_threshold']['precision']:.2f} at threshold {f['threshold']:.2f}",
             "business_result": f"Reviewing the riskiest 5% catches {f5['share_of_all_fraud_%']}% of fraud cases and {f5['share_of_fraud_$_%']}% of fraud value", "verdict": classify(f)})
rows.append({"no": 3, "business_objective": "What kinds of customers do we have?", "ml_task": "Clustering (K-Means)", "target": "none", "grain": "customer", "split": "shared customer split",
             "champion": f"K-Means, k = {s['k']}", "test_size": "all customers (test silhouette measured on the held-out 20%)", "primary_metric": "silhouette (test)",
             "result": s_t["silhouette"], "ci_95": "n/a", "benchmark": s["noise_baseline_silhouette"], "benchmark_name": "noise baseline",
             "also": f"train silhouette {s_tr['silhouette']:.2f} | lowest stability ARI {min(s['stability_ari'].values()):.2f}",
             "business_result": f"Segments differ on outcomes the model never saw: spend ratio {st['spend_ratio']}x, churn spread {st['churn_spread_points']} points", "verdict": cluster(s)})
rows.append({"no": 4, "business_objective": "How much will each customer spend next year?", "ml_task": "Regression", "target": "next_12m_spend_usd", "grain": "customer", "split": "shared customer split",
             "champion": v["champion"], "test_size": f"{v['test_customers']:,} customers", "primary_metric": "MAE (USD, lower is better)",
             "result": v["metrics_champion"]["mae"], "ci_95": str(v["bootstrap_95"]["mae"]), "benchmark": v["metrics_baseline"]["mae"], "benchmark_name": "predict the median",
             "also": f"R2 {v['metrics_champion']['r2']:.3f} | ranking (Spearman) {v['metrics_champion']['spearman']:.3f} | WAPE {v['metrics_champion']['wape_%']:.1f}%",
             "business_result": f"The top 20% by prediction hold {v20['share_of_all_spend_%']}% of real spend (best possible {v20['best_possible_%']}%, random 20%)", "verdict": regress(v)})
rows.append({"no": 5, "business_objective": "What will a service job cost?", "ml_task": "Regression", "target": "total_cost_usd", "grain": "service job", "split": "grouped 80/20 by vehicle",
             "champion": w["champion"], "test_size": f"{w['test_jobs']:,} jobs", "primary_metric": "MAE (USD, lower is better)",
             "result": w["metrics_champion"]["mae"], "ci_95": str(w["bootstrap_95"]["mae"]), "benchmark": w["metrics_baseline"]["mae"], "benchmark_name": "predict the median",
             "also": f"R2 {w['metrics_champion']['r2']:.3f} | ranking (Spearman) {w['metrics_champion']['spearman']:.3f} | WAPE {w['metrics_champion']['wape_%']:.1f}%",
             "business_result": f"{w['tolerance']['within_50%']}% of jobs predicted within 50% of the real cost; totals are forecast far better than single jobs (see step 15, section E)", "verdict": regress(w)})
score = pd.DataFrame(rows); score["deployment"] = score.verdict.map(READY)
score["gain_vs_benchmark_%"] = np.where(score.ml_task == "Regression", 100 * (score.benchmark - score.result) / score.benchmark, 100 * (score.result - score.benchmark) / score.benchmark)
score.loc[score.ml_task.str.startswith("Clustering"), "gain_vs_benchmark_%"] = np.nan   # not meaningful for a silhouette

print("MACHINE LEARNING OBJECTIVES SCORECARD (all results are on the TEST set)\n")
print(score[["no", "business_objective", "ml_task", "champion", "primary_metric", "result", "benchmark", "gain_vs_benchmark_%", "verdict", "deployment"]].round(3).to_string(index=False))
for _, r in score.iterrows():
    print(f"\n#{r.no} {r.business_objective}  [{r.verdict}]\n   target: {r.target} | grain: {r.grain} | split: {r.split} | test: {r.test_size}"
          f"\n   {r.primary_metric}: {r.result:.3f} (95% range {r.ci_95}) vs {r.benchmark_name} {r.benchmark:.3f}\n   also: {r.also}\n   business: {r.business_result}")

# %% 3. ALL MODELS SIDE BY SIDE (test set)
comp = []
for key, name, metric, better in [("churn", "Churn", "pr_auc", "high"), ("fraud", "Fraud", "pr_auc", "high"), ("clv", "CLV", "mae", "low"), ("svc", "Service cost", "mae", "low")]:
    for m, r in J[key]["all_models"].items():
        comp.append({"objective": name, "model": m, "metric": metric, "value": r[metric], "is_champion": m == J[key]["champion"], "better_when": better})
comp = pd.DataFrame(comp); comp = comp[~comp.model.str.startswith("Baseline")].reset_index(drop=True)   # the baseline is a yardstick, not a candidate
comp["rank_on_test"] = np.nan
for obj, g in comp.groupby("objective"):
    asc = g.better_when.iloc[0] == "low"; comp.loc[g.index, "rank_on_test"] = g["value"].rank(ascending=asc, method="min")
print("\nEVERY MODEL ON THE TEST SET (champion chosen in step 11 on train cross-validation; ranks shown for information only)\n",
      comp.pivot_table(index="model", columns="objective", values="value").round(3).to_string())
print("\nChampions that are NOT first on the test set (kept on purpose, because choosing by test score would leak the test set):")
off = comp[(comp.is_champion) & (comp.rank_on_test > 1)]
print(off[["objective", "model", "rank_on_test"]].to_string(index=False) if len(off) else "none")

# %% 4. ARE THE FILES FOR STEP 17 (SAVE MODEL, SCALER, FEATURES) IN PLACE?
need = {"Churn": ["churn_models.joblib", "scaler_churn.joblib", "features_churn.json", "prep_churn.json"],
        "Fraud": ["fraud_models.joblib", "scaler_fraud.joblib", "features_fraud.json", "prep_fraud.json"],
        "Segmentation": ["segmentation_model.joblib", "scaler_segmentation.joblib", "features_segmentation.json", "prep_segmentation.json"],
        "CLV": ["clv_models.joblib", "scaler_clv.joblib", "features_clv.json", "prep_clv.json"],
        "Service cost": ["service_models.joblib", "scaler_service.joblib", "features_service.json", "prep_service.json"]}
files = pd.DataFrame([{"objective": o, "file": fn, "found": os.path.exists(f"{DATA}/{fn}")} for o, fl in need.items() for fn in fl])
miss = files[~files.found]
print("\nFILES NEEDED FOR STEP 17:", "all 20 found." if miss.empty else f"MISSING {len(miss)}:\n" + miss.to_string(index=False))

# %% 5. CHART
col = {"STRONG": GREEN, "USEFUL": GREEN, "USEFUL (as tiers)": GREEN, "WEAK": AMBER, "NOT PROVEN": RED}
fig, ax = plt.subplots(1, 5, figsize=(18, 4.2))
for a, (_, r) in zip(ax, score.iterrows()):
    a.bar(["benchmark", "model"], [r.benchmark, r.result], color=[GREY, col.get(r.verdict, BLUE)])
    for i, val in enumerate([r.benchmark, r.result]): a.text(i, val, f"{val:,.2f}" if val < 10 else f"{val:,.0f}", ha="center", va="bottom", fontsize=9)
    a.set_title(f"#{r.no} {r.business_objective.split('?')[0]}\n{r.primary_metric}", fontsize=9); a.set_title(a.get_title() + f"\n[{r.verdict}]", fontsize=9, color=col.get(r.verdict, BLUE))
plt.suptitle("Model against its benchmark on the test set (for MAE lower is better; for the others higher is better)", y=1.06)
plt.tight_layout(); plt.savefig(f"{OUT}/16_ml_objectives_scorecard.png", dpi=150, bbox_inches="tight")
if SHOW: plt.show()
plt.close()

# %% 6. SAVE
score.to_csv(f"{OUT}/ml_objectives_scorecard.csv", index=False); comp.to_csv(f"{OUT}/ml_model_comparison.csv", index=False)
if WRITE_TO_SQL:
    try:
        import pyodbc
        from sqlalchemy import create_engine
        drivers = [d for d in pyodbc.drivers() if "SQL Server" in d]
        pref = [d for d in drivers if "ODBC Driver 18" in d] + [d for d in drivers if "ODBC Driver 17" in d] + drivers
        conn = f"DRIVER={{{pref[0]}}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes;TrustServerCertificate=yes;"
        engine = create_engine("mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(conn))
        for name, df in {"bi_ml_scorecard": score, "bi_ml_model_comparison": comp}.items():
            out = df.copy()
            for cc in out.columns:
                if str(out[cc].dtype) in ("str", "string", "category"): out[cc] = out[cc].astype(object)
            out.to_sql(name, engine, schema="dbo", if_exists="replace", index=False)
        print(f"\nWrote bi_ml_scorecard and bi_ml_model_comparison to {DATABASE}.dbo")
    except Exception as e:
        print("\nSQL write skipped:", repr(e)[:300], "\nThe CSV files are still saved.")
print(f"\nDONE. ml_objectives_scorecard.csv, ml_model_comparison.csv and the chart are in {os.path.abspath(OUT)}")
```

---

## 20. Save Model, Scaler and Features

Script: `python/17_save_model_scaler_features.py`. Everything is written with **joblib only**, into `deployment/`.

| File | Contents |
|---|---|
| `<name>_model.joblib` | The champion model |
| `<name>_scaler.joblib` | The StandardScaler fitted on train |
| `<name>_features.joblib` | Feature names in the exact order the model expects |
| `<name>_prep.joblib` | Cleaning recipe: columns kept, fill values, log1p columns, scaled columns |
| `<name>_meta.joblib` | Task, target, threshold, test results, verdict, library versions |
| `encoding_spec.joblib` | Recipe that turns raw columns into model columns |

`<name>` is one of `churn`, `fraud`, `segmentation`, `clv`, `service`, which gives 26 files in total.

**Round-trip check:** the script reloads every saved file from disk, runs the raw test rows through the saved recipe and compares the output with the original predictions. **All five objectives reproduce exactly (difference 0.0)** on 1,200 customers, 9,821 transactions and 2,810 service jobs.

### Code: save model, scaler and features (Python)

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 17: SAVE MODEL, SCALER AND FEATURES (joblib only)
# For each of the 5 objectives it saves, into ./deployment, everything the Streamlit app needs:
#   <name>_model.joblib     the champion model (K-Means for segmentation)
#   <name>_scaler.joblib    the StandardScaler fitted on the TRAIN data
#   <name>_features.joblib  the exact feature names, in the exact order the model expects
#   <name>_prep.joblib      the cleaning recipe: columns used, fill values, log1p columns, scaled columns
#   <name>_meta.joblib      task, target, champion, threshold, test results, verdict, library versions
#   encoding_spec.joblib    the recipe that turns raw columns (text, tiers) into model columns
# Then it LOADS every saved file back from disk and checks that the reloaded model reproduces the step 12
# predictions exactly, so you know the files work before anything is deployed.
# Only joblib is used to write files. Needs: pandas numpy scikit-learn joblib.
# =====================================================================
# %% 0. SETTINGS
import os, sys, json, datetime, warnings
import numpy as np, pandas as pd
import joblib, sklearn

warnings.filterwarnings("ignore")
DATA, OUT = "abt_data", "deployment"
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 200)

# name -> (model file, split file with original units, prediction file from step 12, evaluation json, prediction column, kind)
OBJ = {
    "churn":        ("churn_models.joblib",        "split_churn.pkl",        "churn_test_predictions.pkl",   "churn_outputs/churn_evaluation.json",                 "champion_proba",   "proba"),
    "fraud":        ("fraud_models.joblib",        "split_fraud.pkl",        "fraud_test_predictions.pkl",   "fraud_outputs/fraud_evaluation.json",                 "champion_proba",   "proba"),
    "segmentation": ("segmentation_model.joblib",  "split_segmentation.pkl", "segmentation_test_assignments.pkl", "segmentation_outputs/segmentation_evaluation.json", "segment",         "cluster"),
    "clv":          ("clv_models.joblib",          "split_clv.pkl",          "clv_test_predictions.pkl",     "clv_outputs/clv_evaluation.json",                     "champion_pred",    "value"),
    "service":      ("service_models.joblib",      "split_service.pkl",      "service_test_predictions.pkl", "service_outputs/service_evaluation.json",             "champion_pred",    "value"),
}
SC_FILE = "business_outputs/ml_objectives_scorecard.csv"
scorecard = pd.read_csv(SC_FILE) if os.path.exists(SC_FILE) else None
SC_NO = {"churn": 1, "fraud": 2, "segmentation": 3, "clv": 4, "service": 5}

def read_json(p):
    with open(p) as f: return json.load(f)

def apply_prep(X, prep, scaler):
    """identical to step 10: keep the model columns, fill, log1p, scale"""
    X = X[prep["features"]].copy().astype(float)
    X = X.fillna(pd.Series(prep["fill_values"]))
    if prep["log_cols"]:
        X[prep["log_cols"]] = np.log1p(X[prep["log_cols"]].clip(lower=0))
    X[prep["scale_cols"]] = scaler.transform(X[prep["scale_cols"]])
    return X

# %% 1. SAVE
manifest = []
versions = {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__}
for name, (mfile, split_f, pred_f, ev_f, pcol, kind) in OBJ.items():
    bundle = joblib.load(f"{DATA}/{mfile}")
    if kind == "cluster":
        model, champion = bundle["model"], f"K-Means (k = {bundle['k']})"
    else:
        champion = bundle["champion"]; model = bundle["models"][champion]
    scaler = joblib.load(f"{DATA}/scaler_{name}.joblib")
    prep = read_json(f"{DATA}/prep_{name}.json")
    features = list(prep["features"])
    ev = read_json(ev_f)
    assert features == list(bundle["features"]), f"{name}: feature list differs between prep and model bundle"
    meta = {"objective": name, "task": ev["task"], "champion": champion, "target": None if kind == "cluster" else ev.get("target", {"churn": "churned", "fraud": "is_fraud"}.get(name)),
            "threshold": ev.get("threshold"), "n_features": len(features), "versions": versions, "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "evaluation": {k: ev[k] for k in ev if k in ("metrics_at_threshold", "metrics_champion", "metrics_baseline", "bootstrap_95", "no_skill_pr_auc", "k", "separation", "stability_ari", "avg_actual_spend", "avg_actual_cost")}}
    if kind == "value": meta["prediction_scale"] = "dollars (the model learns log1p of the target and converts back)"
    if kind == "proba": meta["prediction_scale"] = "probability between 0 and 1; flag when probability >= threshold"
    if kind == "cluster": meta["prediction_scale"] = "segment number 0 .. k-1"
    if scorecard is not None:
        r = scorecard[scorecard.no == SC_NO[name]].iloc[0]; meta["verdict"], meta["deployment_status"] = r["verdict"], r["deployment"]
    for part, obj in {"model": model, "scaler": scaler, "features": features, "prep": prep, "meta": meta}.items():
        joblib.dump(obj, f"{OUT}/{name}_{part}.joblib")
    manifest.append({"objective": name, "champion": champion, "n_features": len(features), "scaled_cols": len(prep["scale_cols"]), "log1p_cols": len(prep["log_cols"]),
                     "verdict": meta.get("verdict", "n/a"), "deployment_status": meta.get("deployment_status", "n/a")})

spec_path = f"{DATA}/encoding_spec.json"
if os.path.exists(spec_path):
    joblib.dump(read_json(spec_path), f"{OUT}/encoding_spec.joblib"); spec_note = "saved"
else:
    spec_note = "NOT FOUND in abt_data (run step 7 again): the app cannot turn raw text columns into model columns without it"
manifest = pd.DataFrame(manifest)
print("SAVED to", os.path.abspath(OUT)); print(manifest.to_string(index=False)); print("encoding_spec.joblib:", spec_note)
print("Library versions used for training (the Streamlit environment must match):", versions)

# %% 2. LOAD EVERYTHING BACK FROM DISK AND CHECK IT REPRODUCES THE STEP 12 PREDICTIONS
print("\nROUND-TRIP CHECK (fresh load from ./deployment -> raw test rows -> predictions vs step 12)")
rows = []
for name, (mfile, split_f, pred_f, ev_f, pcol, kind) in OBJ.items():
    model, scaler = joblib.load(f"{OUT}/{name}_model.joblib"), joblib.load(f"{OUT}/{name}_scaler.joblib")
    prep, feats, meta = joblib.load(f"{OUT}/{name}_prep.joblib"), joblib.load(f"{OUT}/{name}_features.joblib"), joblib.load(f"{OUT}/{name}_meta.joblib")
    X_raw = pd.read_pickle(f"{DATA}/{split_f}")["X_test"]
    X = apply_prep(X_raw, prep, scaler); assert list(X.columns) == feats and not X.isna().any().any()
    saved = pd.read_pickle(f"{DATA}/{pred_f}")[pcol].to_numpy()
    new = model.predict(X) if kind in ("cluster", "value") else model.predict_proba(X)[:, 1]
    if kind == "value": new = np.clip(new, 0, None)
    same = len(new) == len(saved) and (np.array_equal(new, saved) if kind == "cluster" else np.allclose(new, saved, rtol=1e-6, atol=1e-6))
    rows.append({"objective": name, "test_rows": len(X), "max_abs_difference": float(np.max(np.abs(new - saved))) if len(new) == len(saved) else np.nan, "reproduces_step_12": same})
chk = pd.DataFrame(rows); print(chk.to_string(index=False))
print("\nALL FILES WORK" if chk.reproduces_step_12.all() else "\nWARNING: at least one objective does not reproduce its predictions. Do not deploy before this is fixed.")
print("\nFiles in ./deployment:")
for f in sorted(os.listdir(OUT)):
    print("  ", f, f"{os.path.getsize(os.path.join(OUT, f)) / 1024:,.0f} KB")
```

---

## 21. Download Model, Scaler and Features

Script: `python/18_download_model_scaler_features.py`. It packs `deployment/` into one file, `vantage_deployment_package.zip`:

- `models/`: the 26 `.joblib` files
- `requirements.txt`: the library versions the models were trained with
- `MANIFEST.txt`: every file with its size and SHA-256 fingerprint
- `README.txt`: what each file is, with a short loading example

The script reopens the zip and loads all 26 files from inside it to prove the package is not damaged. Training versions are pinned (scikit-learn 1.9.0, pandas 3.0.3, numpy 2.4.6, joblib 1.5.3) because models saved with one scikit-learn version can fail to load with another.

### Streamlit app

**[Open the live Streamlit app](https://vantage-automotive-business-operations-predictive-support-syst.streamlit.app/)**

The app loads the saved joblib files from `models/` and runs the same pipeline as training: raw inputs go through the saved encoding recipe, then the saved prep and scaler, then the model. Pages: Home, Customer churn, Fraud detection, Segmentation and behaviour, Lifetime value, Service cost, Customer 360 (churn, value and segment in one form), Batch scoring (upload a CSV, download predictions) and Model performance. Each form shows the most important inputs and fills the rest with typical training values. A banner warns when a model is not marked STRONG.

### Run the app locally

```
my_app/
  app.py
  models/        <- the 26 .joblib files from the zip
```

```
pip install -r requirements.txt
streamlit run app.py
```

### Code: download model, scaler and features (Python)

```python
# =====================================================================
# VANTAGE MOTORS GLOBAL - STEP 18: DOWNLOAD MODEL, SCALER AND FEATURES
# Packages the ./deployment folder from step 17 into ONE zip file you can download, move or upload
# (for example into the Streamlit app repository on GitHub).
#   vantage_deployment_package.zip
#       models/            all <name>_model / scaler / features / prep / meta .joblib files + encoding_spec.joblib
#       requirements.txt   the exact library versions the models were trained with
#       MANIFEST.txt       every file with its size and SHA-256 fingerprint
#       README.txt         what each file is and how to load it
# It then OPENS the zip and loads every .joblib inside it, to prove the package is not damaged.
# Needs: joblib. Nothing is retrained.
# =====================================================================
# %% 0. SETTINGS
import os, io, sys, zipfile, hashlib, datetime, warnings
import joblib

warnings.filterwarnings("ignore")
SRC, ZIP_NAME = "deployment", "vantage_deployment_package.zip"
NAMES = ["churn", "fraud", "segmentation", "clv", "service"]
PARTS = ["model", "scaler", "features", "prep", "meta"]

# %% 1. CHECK THAT EVERYTHING IS THERE
expected = [f"{n}_{p}.joblib" for n in NAMES for p in PARTS] + ["encoding_spec.joblib"]
missing = [f for f in expected if not os.path.exists(f"{SRC}/{f}")]
assert not missing, f"missing files, run step 17 again: {missing}"
print(f"All {len(expected)} files found in ./{SRC}")

# %% 2. READ THE TRAINING VERSIONS AND BUILD THE EXTRA FILES
v = joblib.load(f"{SRC}/churn_meta.joblib")["versions"]
for n in NAMES:
    assert joblib.load(f"{SRC}/{n}_meta.joblib")["versions"] == v, f"{n} was saved with different library versions"
requirements = (f"# libraries the models were trained with (Python {v['python']}).\n# scikit-learn and numpy MUST match to load the models safely.\n"
                f"scikit-learn=={v['scikit-learn']}\nnumpy=={v['numpy']}\npandas=={v['pandas']}\njoblib=={v['joblib']}\nstreamlit\n")

meta = {n: joblib.load(f"{SRC}/{n}_meta.joblib") for n in NAMES}
lines = [f"{n:13s} champion: {meta[n]['champion']:20s} verdict: {meta[n].get('verdict', 'n/a'):18s} status: {meta[n].get('deployment_status', 'n/a')}" for n in NAMES]
readme = f"""VANTAGE MOTORS GLOBAL - DEPLOYMENT PACKAGE
Created: {datetime.datetime.now().isoformat(timespec='seconds')}

OBJECTIVES
{chr(10).join(lines)}

FILES (inside models/)
  <name>_model.joblib     trained champion model (K-Means for segmentation)
  <name>_scaler.joblib    StandardScaler fitted on the training data
  <name>_features.joblib  feature names in the exact order the model expects
  <name>_prep.joblib      cleaning recipe: columns kept, fill values, log1p columns, scaled columns
  <name>_meta.joblib      task, target, threshold, test results, verdict, library versions
  encoding_spec.joblib    recipe that turns raw columns (text, tiers) into model columns
<name> is one of: churn, fraud, segmentation, clv, service

HOW TO LOAD (same order as training)
  import joblib, numpy as np, pandas as pd
  model, scaler = joblib.load("models/churn_model.joblib"), joblib.load("models/churn_scaler.joblib")
  prep = joblib.load("models/churn_prep.joblib")
  X = row_or_table[prep["features"]].astype(float)                  # raw columns already encoded with encoding_spec
  X = X.fillna(pd.Series(prep["fill_values"]))
  X[prep["log_cols"]] = np.log1p(X[prep["log_cols"]].clip(lower=0))
  X[prep["scale_cols"]] = scaler.transform(X[prep["scale_cols"]])
  probability = model.predict_proba(X)[:, 1]      # churn and fraud: flag when >= meta["threshold"]
  # clv and service: model.predict(X) returns dollars | segmentation: model.predict(X) returns the segment number

NOTES
  - Install the versions in requirements.txt before loading the models.
  - Only objectives marked DEPLOY or DEPLOY WITH CARE should be used for decisions.
"""

# %% 3. BUILD THE ZIP
manifest_rows = []
with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as z:
    for f in expected:
        data = open(f"{SRC}/{f}", "rb").read()
        z.writestr(f"models/{f}", data)
        manifest_rows.append(f"{hashlib.sha256(data).hexdigest()}  {len(data):>10,d} bytes  models/{f}")
    z.writestr("requirements.txt", requirements)
    z.writestr("README.txt", readme)
    z.writestr("MANIFEST.txt", "SHA-256 fingerprint, size, file\n" + "\n".join(manifest_rows) + "\n")
size_kb = os.path.getsize(ZIP_NAME) / 1024
print(f"\nCreated {os.path.abspath(ZIP_NAME)} ({size_kb:,.0f} KB, {len(expected) + 3} files)")

# %% 4. OPEN THE ZIP AND LOAD EVERY FILE INSIDE IT
with zipfile.ZipFile(ZIP_NAME) as z:
    assert z.testzip() is None, "the zip is damaged"
    ok = 0
    for f in expected:
        raw = z.read(f"models/{f}")
        assert hashlib.sha256(raw).hexdigest() == hashlib.sha256(open(f"{SRC}/{f}", "rb").read()).hexdigest()
        joblib.load(io.BytesIO(raw)); ok += 1
    print(f"Check passed: {ok} of {len(expected)} .joblib files inside the zip load correctly and match the originals.")
    print("Zip contents:", ", ".join(sorted({n.split('/')[0] if '/' in n else n for n in z.namelist()})))
print("\nRequirements saved in the zip:\n" + requirements)

# %% 5. DOWNLOAD
try:                                                  # inside Google Colab this starts the browser download
    from google.colab import files
    files.download(ZIP_NAME)
except ImportError:                                   # on your own computer the file is already in the folder printed above
    if sys.platform.startswith("win"):
        try: os.startfile(os.path.dirname(os.path.abspath(ZIP_NAME)))      # opens the folder in File Explorer
        except Exception: pass
    print(f"Your download is ready: {os.path.abspath(ZIP_NAME)}")
```

### Code: Streamlit app (`app.py`)

```python
# =====================================================================
# VANTAGE AUTOMOTIVE - BUSINESS OPERATIONS PREDICTIVE SUPPORT SYSTEM (Streamlit app)
# One simple app that uses the five models you trained and saved in steps 17-18:
#   1. Churn           - how likely is a customer to leave?          (classification)
#   2. Fraud           - how likely is a transaction to be fraud?    (classification)
#   3. Segmentation    - which customer segment does a customer belong to? (clustering)
#   4. Lifetime value  - how much will a customer spend next year?   (regression)
#   5. Service cost    - how much will a service job cost?           (regression)
#
# HOW TO RUN (in the VS Code terminal, inside the folder that holds this file):
#       pip install streamlit pandas numpy scikit-learn joblib
#       streamlit run app.py
#
# FOLDER LAYOUT (unzip vantage_deployment_package.zip next to app.py):
#       my_app/
#         app.py
#         models/   <- the 26 .joblib files from the zip  (churn_model.joblib, churn_scaler.joblib, ...)
#
# IMPORTANT: install the SAME library versions the models were trained with (they are in the zip's requirements.txt),
# or the models may refuse to load.
# =====================================================================
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Vantage Automotive Predictive Support System", page_icon="🚗", layout="wide")

# ---------------------------------------------------------------------
# 1. SETTINGS (the only things you may want to change)
# ---------------------------------------------------------------------
MODEL_DIR = "models"                  # folder with the .joblib files
HIGH_VALUE_MULTIPLE = 1.5             # a customer is "high value" if predicted spend >= 1.5 x the average customer
TOP_FIELDS = 6                        # how many of the most important inputs each form shows
NAMES = ["churn", "fraud", "segmentation", "clv", "service"]
TITLES = {"churn": "Customer churn", "fraud": "Fraud detection", "segmentation": "Customer segmentation",
          "clv": "Customer lifetime value", "service": "Service cost"}
# which encoding recipe (from encoding_spec.joblib) each model uses
SPEC_OF = {"churn": "customer", "clv": "customer", "segmentation": "customer", "fraud": "fraud", "service": "service"}
ID_COLUMNS = ["customer_id", "txn_id", "service_id", "vehicle_id"]


# ---------------------------------------------------------------------
# 2. LOAD THE SAVED FILES (cached, so they load only once)
# ---------------------------------------------------------------------
@st.cache_resource
def load_everything():
    bundle = {}
    for n in NAMES:
        bundle[n] = {part: joblib.load(os.path.join(MODEL_DIR, f"{n}_{part}.joblib")) for part in ["model", "scaler", "features", "prep", "meta"]}
    spec = joblib.load(os.path.join(MODEL_DIR, "encoding_spec.joblib"))
    return bundle, spec


try:
    B, SPEC = load_everything()
except Exception as err:
    st.error(f"Could not load the model files from the '{MODEL_DIR}' folder.\n\n{err}")
    st.info("Check that the 26 .joblib files are inside the folder, and that scikit-learn, numpy and pandas have the versions listed in requirements.txt.")
    st.stop()


# ---------------------------------------------------------------------
# 3. SMALL HELPERS
# ---------------------------------------------------------------------
def nice(name):
    """customer_tenure_months -> Customer tenure months"""
    text = name.replace("_usd", " (USD)").replace("_", " ").strip()
    return text[0].upper() + text[1:]


def clean_name(text):
    import re
    return re.sub(r"[^0-9A-Za-z]+", "_", str(text)).strip("_")


def encode(df, spec):
    """Turn raw columns (text, tiers) into model columns. Same recipe as step 7. Missing text columns use the most common level."""
    out = df.copy()
    for col, mapping in spec["ordinal"].items():
        if col in out:
            out[col] = out[col].astype("string").map(mapping).astype("float")
    for col, info in spec["onehot"].items():
        s = out[col].astype("string").fillna("Missing") if col in out else pd.Series([info["baseline"]] * len(out), index=out.index, dtype="string")
        for cat in info["kept"]:
            out[f"{col}_{clean_name(cat)}"] = (s == cat).astype(int)
        out = out.drop(columns=[col], errors="ignore")
    return out


def prepare(name, encoded):
    """Same cleaning as step 10: keep model columns, fill gaps, log1p, scale."""
    prep, scaler = B[name]["prep"], B[name]["scaler"]
    X = encoded.reindex(columns=prep["features"]).apply(pd.to_numeric, errors="coerce").astype(float)
    X = X.fillna(pd.Series(prep["fill_values"]))                       # missing values -> typical training value
    if prep["log_cols"]:
        X[prep["log_cols"]] = np.log1p(X[prep["log_cols"]].clip(lower=0))
    X[prep["scale_cols"]] = scaler.transform(X[prep["scale_cols"]])
    return X


def predict(name, raw_df):
    """raw rows in -> predictions out (probability, dollars or segment number)."""
    X = prepare(name, encode(raw_df, SPEC[SPEC_OF[name]]))
    model = B[name]["model"]
    if name in ("churn", "fraud"):
        return model.predict_proba(X)[:, 1]
    if name == "segmentation":
        return model.predict(X)
    return np.clip(model.predict(X), 0, None)


def importance(name):
    """How much each feature matters to the model (used to decide which inputs to show first)."""
    model, n = B[name]["model"], len(B[name]["features"])
    model = getattr(model, "regressor_", model)
    if hasattr(model, "feature_importances_"):
        return np.abs(model.feature_importances_)
    if hasattr(model, "coef_"):
        return np.abs(np.ravel(model.coef_))
    if hasattr(model, "cluster_centers_"):
        return model.cluster_centers_.std(axis=0)
    return np.ones(n)


def input_fields(name):
    """The 'raw' inputs behind a model: a number, or a category (several dummy columns = one drop-down).
    Returns a list of (field, kind, options) from most to least important."""
    spec, feats, imp = SPEC[SPEC_OF[name]], B[name]["features"], importance(name)
    dummy_of = {f"{col}_{clean_name(c)}": col for col, info in spec["onehot"].items() for c in info["kept"]}
    score, kind = {}, {}
    for f, v in zip(feats, imp):
        field = dummy_of.get(f, f)
        score[field] = score.get(field, 0) + v
        kind[field] = "category" if f in dummy_of else ("tier" if f in spec["ordinal"] else "number")
    fields = []
    for field in sorted(score, key=score.get, reverse=True):
        if kind[field] == "category":
            info = spec["onehot"][field]; options = [info["baseline"]] + list(info["kept"])
        elif kind[field] == "tier":
            options = list(spec["ordinal"][field].keys())
        else:
            options = None
        fields.append((field, kind[field], options))
    return fields


def typical_value(names, field):
    """The typical (training median or most common) value of a number field."""
    for n in names:
        v = B[n]["prep"]["fill_values"].get(field)
        if v is not None:
            return float(v)
    return 0.0


def draw_form(names, key):
    """Draw the input widgets for the most important fields of one or more models. Returns {field: value}."""
    chosen, seen = [], set()
    for n in names:                                                    # top fields of each model, no duplicates
        for field in input_fields(n)[:TOP_FIELDS]:
            if field[0] not in seen:
                seen.add(field[0]); chosen.append(field)
    values, cols = {}, st.columns(3)
    for i, (field, kind, options) in enumerate(chosen):
        with cols[i % 3]:
            if kind == "category":
                values[field] = st.selectbox(nice(field), options, key=f"{key}_{field}")            # first option = most common level
            elif kind == "tier":
                typ = typical_value(names, field)                                                    # start on the typical tier
                values[field] = st.selectbox(nice(field), options, index=min(int(round(typ)), len(options) - 1), key=f"{key}_{field}")
            else:
                binary = any(field in B[n]["prep"]["binary_cols"] for n in names)
                typ = typical_value(names, field)
                if binary:
                    values[field] = st.selectbox(nice(field), [0, 1], index=int(round(typ)), format_func=lambda x: "Yes" if x == 1 else "No", key=f"{key}_{field}")
                else:
                    values[field] = st.number_input(nice(field), value=round(typ, 2), key=f"{key}_{field}", help=f"Typical value in the training data: {typ:,.2f}")
    return values


def status_banner(name):
    status = B[name]["meta"].get("deployment_status", "")
    verdict = B[name]["meta"].get("verdict", "")
    if status.startswith("DO NOT") or status.startswith("HOLD"):
        st.error(f"{TITLES[name]} model: NOT proven on the test data (verdict: {verdict}). Use it for testing only, not for decisions.")
    elif status == "DEPLOY WITH CARE":
        st.warning(f"{TITLES[name]} model: verdict {verdict}. Use it to rank and prioritise, and double-check important decisions.")


def risk_text(p, threshold):
    return ("HIGH RISK", "error") if p >= threshold else ("LOWER RISK", "success")


# ---------------------------------------------------------------------
# 4. PAGES
# ---------------------------------------------------------------------
OBJECTIVE = {"churn": "Reduce customer churn", "fraud": "Detect fraudulent transactions", "segmentation": "Segment customers and understand behaviour",
             "clv": "Forecast customer lifetime value", "service": "Predict service and repair cost"}
PAGE_OF = {"churn": "1. Customer churn", "fraud": "2. Fraud detection", "segmentation": "3. Segmentation and behaviour", "clv": "4. Lifetime value", "service": "5. Service cost"}


def page_home():
    st.title("🚗 Vantage Automotive: Business Operations Predictive Support System")
    st.write("Five machine learning models that help the business decide **who to keep, which payments to check, "
             "which customers matter most, and what a repair will cost.** Pick a page on the left.")
    order = ["churn", "fraud", "segmentation", "clv", "service"]
    rows = []
    for i, n in enumerate(order, 1):
        m = B[n]["meta"]
        rows.append({"#": i, "Business objective": OBJECTIVE[n], "ML task": m["task"], "Champion model": m["champion"], "Verdict": m.get("verdict", "n/a"), "Status": m.get("deployment_status", "n/a"), "Page": PAGE_OF[n]})
    st.subheader("The five models")
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.subheader("Other pages")
    st.markdown("- **Customer 360** : churn risk, expected spend and segment for one customer in one go, plus a recommended action.\n"
                "- **Batch scoring** : upload a CSV with many rows, get predictions back, download them.\n"
                "- **Model performance** : a report (not a prediction) showing how well each model did on the test data.")
    st.caption("Predictions are decision support, not certainty. Fields you leave unchanged use the typical value from the training data.")


def page_customer():
    st.title("Customer 360: churn + lifetime value + segment")
    st.write("One form, three models. Fill in what you know about the customer; anything you leave unchanged uses the typical value.")
    for n in ("churn", "clv", "segmentation"):
        status_banner(n)
    with st.form("customer_form"):
        values = draw_form(["churn", "clv", "segmentation"], "cust")
        go = st.form_submit_button("Predict")
    if not go:
        return
    row = pd.DataFrame([values])
    p_churn, spend, seg = predict("churn", row)[0], predict("clv", row)[0], int(predict("segmentation", row)[0])
    thr, avg = B["churn"]["meta"]["threshold"], B["clv"]["meta"]["evaluation"].get("avg_actual_spend", 0)
    high_risk, high_value = p_churn >= thr, spend >= HIGH_VALUE_MULTIPLE * avg
    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability", f"{p_churn:.0%}", "High risk" if high_risk else "Lower risk", delta_color="inverse" if high_risk else "normal")
    c2.metric("Expected spend, next 12 months", f"${spend:,.0f}", "High value" if high_value else "Standard value", delta_color="off")
    c3.metric("Customer segment", f"Segment {seg}")
    st.progress(float(min(max(p_churn, 0), 1)), text=f"Churn risk (flag level: {thr:.0%})")
    if high_risk and high_value:
        st.error("**PROTECT NOW**: high value and likely to leave. Give this customer personal outreach from an account manager.")
    elif high_value:
        st.success("**GROW**: high value and loyal. Offer priority service or perks.")
    elif high_risk:
        st.warning("**LOW-COST WIN-BACK**: likely to leave but lower value. Send a low-cost automated offer.")
    else:
        st.info("**MAINTAIN**: no action needed beyond normal service.")
    centres = pd.Series(B["segmentation"]["model"].cluster_centers_[seg], index=B["segmentation"]["features"]).sort_values()
    st.caption("Segment %d stands out for: highest on %s; lowest on %s." % (seg, ", ".join(nice(f).lower() for f in centres.index[::-1][:3]), ", ".join(nice(f).lower() for f in centres.index[:2])))
    st.caption("Churn probability is best used to rank customers: the model was trained to find leavers, so it errs on the high side.")


def page_churn():
    st.title("1. Customer churn: who is likely to leave?")
    status_banner("churn")
    with st.form("churn_form"):
        values = draw_form(["churn"], "churn")
        go = st.form_submit_button("Predict churn risk")
    if not go:
        return
    p, thr = predict("churn", pd.DataFrame([values]))[0], B["churn"]["meta"]["threshold"]
    label, kind = risk_text(p, thr)
    st.metric("Churn probability", f"{p:.0%}", label, delta_color="inverse" if kind == "error" else "normal")
    st.progress(float(min(max(p, 0), 1)), text=f"Flag level: {thr:.0%}")
    if p >= thr:
        st.error("**Add to the retention list.** Contact this customer with a retention offer.")
    else:
        st.success("Lower risk. Keep serving as normal.")
    st.caption("Best used to rank customers: the model was trained to find leavers, so it errs on the high side.")


def page_transaction():
    st.title("2. Fraud detection: is this payment suspicious?")
    status_banner("fraud")
    with st.form("txn_form"):
        values = draw_form(["fraud"], "txn")
        go = st.form_submit_button("Check transaction")
    if not go:
        return
    p, thr = predict("fraud", pd.DataFrame([values]))[0], B["fraud"]["meta"]["threshold"]
    label, kind = risk_text(p, thr)
    st.metric("Fraud probability", f"{p:.1%}", label, delta_color="inverse" if kind == "error" else "normal")
    st.progress(float(min(max(p, 0), 1)), text=f"Review level: {thr:.1%}")
    if p >= thr:
        st.error("**SEND TO MANUAL REVIEW.** Check the customer and the payment before approving.")
    else:
        st.success("Below the review level. Approve as normal.")
    st.caption("The review level was chosen on the training data. Risk is a ranking: higher means review sooner.")


def segment_profile(seg):
    """How a segment differs from the average customer, as plain-language behaviour traits.
    The numbers are standard deviations from the average customer: above 0 = higher than average, below 0 = lower."""
    centres = pd.Series(B["segmentation"]["model"].cluster_centers_[seg], index=B["segmentation"]["features"])
    return centres


def page_segmentation():
    st.title("3. Segmentation and behaviour: what kind of customer is this?")
    st.write("Customers are grouped by how they buy, use services, contact support and drive. Enter a customer to see their segment and behaviour profile.")
    status_banner("segmentation")
    with st.form("seg_form"):
        values = draw_form(["segmentation"], "seg")
        go = st.form_submit_button("Find segment")
    k = B["segmentation"]["model"].n_clusters
    if go:
        seg = int(predict("segmentation", pd.DataFrame([values]))[0])
        st.metric("Customer segment", f"Segment {seg} of {k}")
        prof = segment_profile(seg).sort_values()
        high, low = prof.index[::-1][:3], prof.index[:3]
        st.write(f"**Behaviour profile:** this segment is **highest** on {', '.join(nice(f).lower() for f in high)} and **lowest** on {', '.join(nice(f).lower() for f in low)}.")
        st.caption("Bars show how far the segment is from the average customer. Right of zero = above average, left = below average.")
        shown = pd.concat([prof.head(6), prof.tail(6)]).drop_duplicates()
        st.bar_chart(shown.rename(index=nice))
    with st.expander("Compare all segments"):
        table = pd.DataFrame(B["segmentation"]["model"].cluster_centers_, columns=B["segmentation"]["features"]).T
        table.columns = [f"Segment {i}" for i in range(k)]
        top = table.abs().max(axis=1).sort_values(ascending=False).head(12).index
        st.write("Behaviour of each segment compared with the average customer (0 = average, positive = higher, negative = lower). The 12 features that separate the segments most:")
        st.dataframe(table.loc[top].rename(index=nice).round(2))
    st.caption("Segments are tiers on a continuum, not sharply separated groups. Customers near a boundary can move between segments.")


def page_clv():
    st.title("4. Lifetime value: how much will this customer spend?")
    status_banner("clv")
    with st.form("clv_form"):
        values = draw_form(["clv"], "clv")
        go = st.form_submit_button("Forecast spend")
    if not go:
        return
    spend, avg = predict("clv", pd.DataFrame([values]))[0], B["clv"]["meta"]["evaluation"].get("avg_actual_spend", 0)
    mae = B["clv"]["meta"]["evaluation"].get("metrics_champion", {}).get("mae", 0)
    st.metric("Expected spend, next 12 months", f"${spend:,.0f}", "High value" if spend >= HIGH_VALUE_MULTIPLE * avg else "Standard value", delta_color="off")
    st.write(f"The average customer spends about **${avg:,.0f}**. This customer is forecast at **{spend / avg:.1f}x** the average." if avg else "")
    st.caption(f"Typical error per customer is about ${mae:,.0f}, so use it to rank and group customers, not to promise an exact amount.")


def page_service():
    st.title("5. Service cost: what will this job cost?")
    status_banner("service")
    with st.form("svc_form"):
        values = draw_form(["service"], "svc")
        go = st.form_submit_button("Estimate cost")
    if not go:
        return
    cost = predict("service", pd.DataFrame([values]))[0]
    mae = B["service"]["meta"]["evaluation"].get("metrics_champion", {}).get("mae", 0)
    st.metric("Estimated cost of the job", f"${cost:,.0f}")
    st.write(f"Likely range: **${max(cost - mae, 0):,.0f} to ${cost + mae:,.0f}** (typical error of the model is about ${mae:,.0f} per job).")
    st.caption("Use as a rough quote or for planning. Very expensive repairs tend to be under-estimated.")


def page_batch():
    st.title("Batch scoring")
    st.write("Upload a CSV with one row per customer, transaction or service job. The columns should have the **same names as your analytical base tables** "
             "(for example `annual_income_usd`, `loyalty_tier`, `country`). Missing columns use the typical value.")
    kind = st.selectbox("What are you scoring?", ["Customers (churn, lifetime value, segment)", "Transactions (fraud)", "Service jobs (cost)"])
    names = {"Customers": ["churn", "clv", "segmentation"], "Transactions": ["fraud"], "Service": ["service"]}[kind.split(" ")[0]]
    # a one-row template with every raw column the models use
    template, seen = {}, set()
    for n in names:
        for field, k, options in input_fields(n):
            if field not in seen:
                seen.add(field); template[field] = options[0] if options else round(typical_value(names, field), 2)
    id_col = {"churn": "customer_id", "fraud": "txn_id", "service": "service_id"}[names[0]]
    st.download_button("Download an empty template (one example row)", pd.DataFrame([{id_col: "ID_1", **template}]).to_csv(index=False), "template.csv", "text/csv")
    file = st.file_uploader("Upload your CSV", type="csv")
    if file is None:
        return
    raw = pd.read_csv(file)
    st.write(f"Loaded {len(raw):,} rows and {raw.shape[1]} columns.")
    result = raw[[c for c in ID_COLUMNS if c in raw.columns]].copy()
    if "churn" in names:
        result["churn_probability"] = predict("churn", raw).round(4)
        result["predicted_spend_usd"] = predict("clv", raw).round(2)
        result["segment"] = predict("segmentation", raw)
        thr, avg = B["churn"]["meta"]["threshold"], B["clv"]["meta"]["evaluation"].get("avg_actual_spend", 0)
        hr, hv = result.churn_probability >= thr, result.predicted_spend_usd >= HIGH_VALUE_MULTIPLE * avg
        result["action"] = np.select([hr & hv, hv, hr], ["PROTECT NOW", "GROW", "LOW-COST WIN-BACK"], "MAINTAIN")
        result["revenue_risk_index"] = (result.churn_probability * result.predicted_spend_usd).round(0)
        result = result.sort_values("revenue_risk_index", ascending=False)
    elif "fraud" in names:
        result["fraud_probability"] = predict("fraud", raw).round(4)
        result["review_flag"] = np.where(result.fraud_probability >= B["fraud"]["meta"]["threshold"], "REVIEW", "ok")
        result = result.sort_values("fraud_probability", ascending=False)
    else:
        result["predicted_cost_usd"] = predict("service", raw).round(2)
    st.dataframe(result, hide_index=True)
    st.download_button("Download predictions", result.to_csv(index=False), "predictions.csv", "text/csv")


def page_performance():
    st.title("Model performance: how good is each model?")
    st.write("This is a report, not a prediction. It shows how each of the five models scored on customers, transactions and jobs it never saw during training.")
    for n in NAMES:
        m = B[n]["meta"]; ev = m["evaluation"]
        with st.expander(f"{OBJECTIVE[n]}  |  {m['champion']}  |  {m.get('verdict', 'n/a')}", expanded=True):
            st.write(f"**Task:** {m['task']}  |  **Status:** {m.get('deployment_status', 'n/a')}  |  **Features used:** {m['n_features']}")
            if n in ("churn", "fraud"):
                a = ev["metrics_at_threshold"]
                st.write(f"PR AUC **{a['pr_auc']:.3f}** (no-skill {ev['no_skill_pr_auc']:.3f}) | ROC AUC **{a['roc_auc']:.3f}** | recall {a['recall']:.2f} | precision {a['precision']:.2f} | flag level {m['threshold']:.2f}")
            elif n == "segmentation":
                st.write(f"{ev['k']} segments | train silhouette {ev['separation']['train customers']['silhouette']:.2f} | test silhouette {ev['separation']['test customers']['silhouette']:.2f} | lowest stability (ARI) {min(ev['stability_ari'].values()):.2f}")
            else:
                a, b = ev["metrics_champion"], ev["metrics_baseline"]
                st.write(f"MAE **${a['mae']:,.0f}** (predicting the median would give ${b['mae']:,.0f}) | R² **{a['r2']:.2f}** | ranking correlation {a.get('spearman', float('nan')):.2f}")
            st.caption(f"Trained with scikit-learn {m['versions']['scikit-learn']}, saved {m['saved_at']}.")


# ---------------------------------------------------------------------
# 5. NAVIGATION
# ---------------------------------------------------------------------
PAGES = {"Home": page_home, "1. Customer churn": page_churn, "2. Fraud detection": page_transaction, "3. Segmentation and behaviour": page_segmentation,
         "4. Lifetime value": page_clv, "5. Service cost": page_service, "Customer 360 (churn + value + segment)": page_customer,
         "Batch scoring": page_batch, "Model performance": page_performance}
st.sidebar.title("Vantage Automotive")
choice = st.sidebar.radio("Go to", list(PAGES))
st.sidebar.caption("Predictive support system built from five models: churn, fraud, segmentation, lifetime value and service cost.")
PAGES[choice]()
```

### Code: Streamlit requirements (`requirements.txt`)

```
# the versions your models were trained with (from your step 17 output)
scikit-learn==1.9.0
numpy==2.4.6
pandas==3.0.3
joblib==1.5.3
streamlit
```

---

## 22. Key Findings and Insights

Script: `python/19_key_findings.py`. Findings are written from the saved results and change wording when a verdict changes. They are saved as `key_findings.md` and the `bi_key_findings` table for Power BI.

**Churn**
- Support contact volume is the clearest warning sign. Churn climbs from **17.6%** with no contacts to **53.6%** with five or more.
- **Bronze churns at 37.3%**, Silver 26.8%, Gold 23.7%, Platinum 23.6%.
- Individual customers churn at **32.3%** against **23.6%** for fleet operators.

**Fraud**
- Fraud is a combination of conditions, not one. A night-time international payment is fraud **25%** of the time, against 1.7% for a domestic daytime payment.
- Fraud value is concentrated: **5% of transactions hold 96% of fraud value**.

**Customer value and segments**
- Fleet operators spend an average of **$12,060** a year, corporate customers **$7,100** and individuals **$3,361**.
- One segment (31.5% of customers) holds **58.4%** of next-year spend and has the lowest churn. It is the revenue engine.

**Service cost**
- Three heavy repair types (Battery/EV, Engine, Transmission) are **23% of jobs but 46% of cost**. Routine maintenance is 41% of jobs but only 20% of cost.
- Battery/EV repairs average about **$1,268** a job and Transmission about **$1,669**, against about $332 for routine maintenance.
- Totals are forecast far better than single jobs: whole-workshop error is **-3.6%**, and every service type is within 7%.

**Across objectives**
- Churn risk and predicted value together separate customers worth a personal call from customers who deserve a cheap automated message.
- Effort aimed at small, well-chosen groups captures most of the value: the top 20% of customers by risk, the top 5% of transactions, one segment of three.

---

## 23. Recommendations

Script: `python/20_recommendations.py`. Saved as `recommendations.md` and the `bi_recommendations` table for Power BI.

| # | Recommendation | Owner | When |
|---|---|---|---|
| 1 | Give "protect now" customers (high value, likely to leave) personal outreach from an account manager | Retention | Now |
| 2 | Run a retention campaign on the riskiest 20% of customers, with a 10% control group left untouched | Marketing and CRM | Now |
| 3 | Send low-value, likely-to-leave customers a low-cost automated win-back message | Marketing and CRM | Next |
| 4 | Trigger a retention check when a customer reaches 3 or more support contacts; review the Bronze tier | Customer care | Next |
| 5 | Send the riskiest 5% of transactions to manual review; add a shadow-mode hold on the riskiest 1% | Fraud operations | Now |
| 6 | Treat the top-value segment as a priority tier with dedicated service and perks | Sales | Next |
| 7 | Rank customers by predicted spend, and cap acquisition and retention spend by group | Marketing and finance | Next |
| 8 | Use model totals to plan parts stock, technician hours and budgets by service type and country | Service operations | Next |
| 9 | Show advisors a price range at booking, and route top-10% predicted costs to senior technicians | Service operations | Next |
| 10 | Test every action against a control group and replace the assumptions with measured results | Data science | Next |
| 11 | Monitor monthly (score distributions, flag rates, accuracy) and retrain on a schedule | Data science | Ongoing |

Each recommendation in `recommendations.md` also lists its evidence, expected impact, KPI and guardrail. A model that is not marked STRONG or USEFUL gets "do not deploy" instead of an action plan.

---

## 24. Impact (Power BI)

Script: `python/21_impact.py`. The test data proves the models **rank** well. It cannot prove how customers or fraudsters **react**. So the impact is modelled in three scenarios, and every assumption is stated.

| Assumption | Conservative | Expected | Optimistic |
|---|---|---|---|
| Save rate (share of contacted would-be churners kept) | 10% | 25% | 40% |
| Gross margin on retained revenue | 15% | 20% | 25% |
| Fraud value in the review queue actually stopped | 50% | 70% | 90% |
| Spend uplift from priority service for top customers | 0% | 1% | 3% |
| Workshop cost saved by better planning | 0.25% | 0.75% | 1.5% |

Programme costs are included: a build cost, a yearly running cost, a three-year cash flow, payback and NPV at 10%.

**Evidence-backed examples (test set)**
- **Churn campaign:** contacting the riskiest 20% (240 customers) reaches churners worth **$217,337** of next-year spend. At a 25% save rate that is about **$54,000 retained for $4,800 of contact cost**.
- **Break-even:** with a 20% margin, the campaign pays back if the real save rate is above about **11%** (contact cost only).
- **Fraud:** a review queue of 491 transactions covers **$8.24M of the $8.59M** fraud value in the test period, for about $3,900 of review cost at an assumed $8 per review.
- **Service:** budget forecasts within **3.6%** of actual cost.

**Decision guide produced by the script**
- **Approve:** the benefit survives even the conservative case and the model is proven.
- **Pilot with a control group:** it pays in the expected case, but its key assumption is untested.
- **Defer:** it does not pay in the expected case, or the model is not proven.

A churn sensitivity table shows net profit for every contact depth and save rate, so a manager can see exactly what has to be true for the campaign to pay.

**Power BI page:** reads `bi_impact_scenarios`, `bi_impact_programme`, `bi_impact_decisions` and `bi_impact_churn_sensitivity` from SQL Server. Visuals: scenario cards, a three-year cash-flow and payback chart, the approve / pilot / defer table, and a save-rate view over the sensitivity table.

### Charts

<p align="center">
  <img src="images/business_21_1_initiative_scenarios.png" alt="Net annual impact by initiative and scenario" width="100%">
</p>
<p align="center"><em>Net annual impact by initiative and scenario</em></p>

<p align="center">
  <img src="images/business_21_2_total_evidence.png" alt="Total impact: model-backed against assumption-only" width="100%">
</p>
<p align="center"><em>Total impact: model-backed against assumption-only</em></p>

<p align="center">
  <img src="images/business_21_3_churn_sensitivity.png" alt="Churn campaign: net profit by contact depth and save rate" width="100%">
</p>
<p align="center"><em>Churn campaign: net profit by contact depth and save rate</em></p>

<p align="center">
  <img src="images/business_21_4_cumulative_cash.png" alt="Programme cash flow, all initiatives" width="100%">
</p>
<p align="center"><em>Programme cash flow, all initiatives</em></p>

---

## 25. Conclusion (Power BI)

Script: `python/22_conclusion.py`. The closing statement is written from the saved results and stored as `bi_conclusion` in SQL Server.

Five models, one pipeline, one app. The data was audited, cleaned and engineered in SQL Server, modelled in Python with no test-data leakage, written back to SQL Server results tables, reported in Power BI and deployed in Streamlit.

- **All five objectives passed the test-data bar:** three STRONG (fraud, lifetime value, service cost) and two USEFUL (churn, segmentation as tiers).
- **The business value sits in small groups:** the riskiest 20% of customers for retention, the riskiest 5% of transactions for fraud, and one segment that carries 58% of spend.
- **The numbers are honest:** every dollar figure states its assumptions, and the next step is to run controlled pilots and replace those assumptions with measured results.

**Limits**
- Customer reaction to offers is assumed, not measured.
- Drivers shown by the models are associations, not proven causes.
- Churn probabilities are inflated by class weights: rank with them, do not read them as literal percentages.
- All data comes from one source and one period, so models need monitoring and retraining.
- Segments are tiers on a continuum, not sharply separated groups.

**Power BI page:** reads `bi_conclusion` (one row per point) next to `bi_ml_scorecard` (verdict for each objective), so the last page of the report shows the same message as this section.

**Next steps:** pilots with control groups, scheduled refresh of the `bi_*` tables, scheduled retraining and monitoring.

---

**Author:** Adewale Oshinukan | Data Scientist | [GitHub](https://github.com/waleoshin75-sketch) | [Live App](https://vantage-automotive-business-operations-predictive-support-syst.streamlit.app/)
