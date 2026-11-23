# Vantage Automotive: Business Operations Predictive Support System

### A complete predictive support system that helps reduce customer churn, stops payment fraud and boosts operational value.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vantage-automotive-business-operations-predictive-support-syst.streamlit.app/)

**Live app:** https://vantage-automotive-business-operations-predictive-support-syst.streamlit.app/

**Tools:** **SQL** (SQL Server, SSMS) · **Python** (pandas, scikit-learn, matplotlib, joblib) · **Power BI** · Streamlit (for the live app)

```
SQL Server  ->  Python  ->  SQL Server results tables  ->  Power BI
(clean, features)  (models)   (bi_* tables)                (dashboards)
```

Predictions go back from Python into SQL Server results tables, and Power BI reads those.

| Business objective | ML task | Champion model | Verdict on unseen data |
|---|---|---|---|
| Reduce customer churn | Classification | Logistic Regression | USEFUL |
| Detect fraudulent transactions | Classification | Logistic Regression | STRONG |
| Segment customers and understand behaviour | Clustering | K-Means (k = 3) | USEFUL (as tiers) |
| Forecast customer lifetime value | Regression | Ridge Regression | STRONG |
| Predict service and repair cost | Regression | Gradient Boosting | STRONG |

**Headline results (test data the models never saw)**
- Contacting the riskiest **20%** of customers reaches **36%** of the customers who will leave.
- Reviewing only the riskiest **5%** of transactions catches **96%** of fraud value.
- One customer segment is **31%** of customers but **58%** of spend, with the lowest churn.
- Total workshop cost is forecast within **3.6%** of actual.

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
9. [Analytical Base Table (SQL build and Python load)](#9-analytical-base-table-sql-build-and-python-load)
10. [Encode Categorical Columns](#10-encode-categorical-columns)
11. [Separate X and Y](#11-separate-x-and-y)
12. [Train Test Split](#12-train-test-split)
13. [Feature Scaling of Numerical Columns](#13-feature-scaling-of-numerical-columns)
14. [Train Model](#14-train-model)
15. [Predict x_test](#15-predict-x_test)
16. [Compare X and Y](#16-compare-x-and-y)
17. [Evaluate Model](#17-evaluate-model)
18. [Business Analysis](#18-business-analysis)
19. [Save Model, Scaler and Features](#19-save-model-scaler-and-features)
20. [Download Model, Scaler and Features](#20-download-model-scaler-and-features)
21. [Key Findings and Insights](#21-key-findings-and-insights)
22. [Recommendations](#22-recommendations)
23. [Impact (Power BI)](#23-impact-power-bi)
24. [Conclusion (Power BI)](#24-conclusion-power-bi)

---

## 1. Project Overview

Vantage Motors Global is a fictional multinational carmaker with customers, vehicles, payments, workshops, support desks and connected-car trips across eight countries. Its data lives in seven messy tables. This project turns that data into a working decision tool.

**Five business questions, five models**

| # | Business objective | ML task | Target | One row is a | App page |
|---|---|---|---|---|---|
| 1 | Reduce customer churn | Classification | `churned` | customer | Customer churn |
| 2 | Detect fraudulent transactions | Classification | `is_fraud` | transaction | Fraud detection |
| 3 | Segment customers, understand behaviour | Clustering (K-Means) | none | customer | Segmentation and behaviour |
| 4 | Forecast customer lifetime value | Regression | `next_12m_spend_usd` | customer | Lifetime value |
| 5 | Predict service and repair cost | Regression | `total_cost_usd` | service job | Service cost |

**Tools and what each one does**

| Tool | Used for |
|---|---|
| **SQL** (SQL Server, `Vantage_Automotive`) | Audit, clean, engineer features, build relationships, build the analytical base tables, and **store the results tables** Python writes back |
| **Python** | EDA, encoding, splitting, scaling, training, prediction, evaluation, business analysis, saving models, writing `bi_*` tables to SQL Server |
| **Power BI** | Business Analysis, Key Findings, Recommendations, Impact and Conclusion pages. It **reads the `bi_*` results tables** and the relationship table from SQL Server |
| Streamlit | Live prediction app: single forms, batch CSV scoring, model performance |

**How the results travel:** Python scores the test data, then writes the business tables back into SQL Server (`if_exists="replace"`, schema `dbo`). Power BI connects to SQL Server and reads only those tables, so a dashboard refresh never needs Python to be open.

**The workflow (24 steps)**

```
Raw tables -> Audit -> Clean -> Features -> Relationships -> EDA -> ABT
-> Encode -> X/y -> Split -> Scale -> Train -> Predict -> Compare -> Evaluate
-> Business analysis -> Save -> Download -> Findings -> Recommendations -> Impact -> Conclusion
```

**Rules I set for myself**
- No test data touches any training decision. Everything is fitted on train only.
- Columns that leak the answer are removed before modelling.
- A model is only called useful if it beats a no-skill benchmark with a 95% confidence range.
- Models that fail that bar are not deployed.

---

## 2. Dataset Overview

Seven tables in a star schema, **147,324 raw rows**. The data is deliberately dirty: **82 data quality issues** were planted across the tables, so cleaning is real work, not a formality.

| Table | Type | Raw rows | What a row is |
|---|---|---|---|
| `dim_branches` | dimension | 60 | a branch |
| `dim_customers` | dimension | 6,126 | a customer |
| `dim_vehicles` | dimension | 9,189 | a vehicle |
| `fact_transactions` | fact | 61,500 | a payment |
| `fact_service_records` | fact | 14,294 | a service job |
| `fact_support_interactions` | fact | 15,315 | a support contact |
| `fact_trips` | fact | 40,840 | a connected-car trip |

**After cleaning and feature engineering**

| Analytical base table | Rows | Target balance |
|---|---|---|
| `abt_customer` | 6,000 customers | 30.7% churned; average next-12-month spend $4,529 |
| `abt_fraud` | 58,656 transactions | 2.4% fraud |
| `abt_service` | 13,836 service jobs | average cost $686 |

Observation cutoff for all features: **31 December 2025**.

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

### 5.1 Cleaning code (excerpt)

Money parsing helper:

```sql
CREATE FUNCTION dbo.fn_parse_money (@s NVARCHAR(100))
RETURNS DECIMAL(18,2)
AS
BEGIN
    DECLARE @t NVARCHAR(100) = NULLIF(LTRIM(RTRIM(@s)), '');
    IF @t IS NULL RETURN NULL;
    SET @t = REPLACE(REPLACE(REPLACE(REPLACE(UPPER(@t), 'USD', ''), '$', ''), ',', ''), ' ', '');
    RETURN TRY_CONVERT(DECIMAL(18,2), NULLIF(@t, ''));
END
```

Deduplicate, quarantine orphans, clean (transactions):

```sql
WITH s AS (
    SELECT f.*, LTRIM(RTRIM(f.txn_id)) AS tid, c.customer_id AS cid_ok
    FROM dbo.fact_transactions f
    LEFT JOIN dbo.clean_dim_customers c ON c.customer_id = LTRIM(RTRIM(f.customer_id)))
SELECT s.*, ROW_NUMBER() OVER (PARTITION BY tid
       ORDER BY CASE WHEN cid_ok IS NULL THEN 1 ELSE 0 END) AS rn
INTO #t FROM s;

INSERT INTO dbo.quarantine_rows
SELECT 'fact_transactions', tid, 'orphan customer_id' FROM #t WHERE rn = 1 AND cid_ok IS NULL;

SELECT t.tid AS txn_id,
       dbo.fn_parse_date(t.txn_date)                          AS txn_date,
       CASE WHEN h.hr BETWEEN 0 AND 23 THEN CAST(h.hr AS INT) END AS txn_hour,
       CASE WHEN a.amt > 0 AND a.amt < 10000000 THEN a.amt END AS amount_usd,
       dbo.fn_map('payment', t.payment_method)                AS payment_method
INTO dbo.clean_fact_transactions
FROM #t t
CROSS APPLY (SELECT dbo.fn_num(t.txn_hour) AS hr) h
CROSS APPLY (SELECT dbo.fn_parse_money(t.amount_usd) AS amt) a
WHERE t.rn = 1 AND t.cid_ok IS NOT NULL;
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

**Checks run:** one row per ID, row counts match the clean tables, and the signal shows up (see EDA).

### 6.1 Feature engineering code (excerpt)

History features for fraud, using only earlier transactions:

```sql
WITH h AS (
    SELECT t.*,
           COUNT(*) OVER (PARTITION BY t.customer_id ORDER BY t.txn_date, t.txn_id
                          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_txn_count,
           AVG(CAST(t.amount_usd AS FLOAT)) OVER (PARTITION BY t.customer_id ORDER BY t.txn_date, t.txn_id
                          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_avg_amount,
           LAG(t.txn_date) OVER (PARTITION BY t.customer_id ORDER BY t.txn_date, t.txn_id) AS prev_txn_date
    FROM dbo.clean_fact_transactions t)
SELECT h.txn_id, h.customer_id, h.txn_date, h.txn_hour,
       CASE WHEN h.txn_hour BETWEEN 0 AND 4 THEN 1 ELSE 0 END          AS is_night,
       CASE WHEN h.billing_country <> h.txn_country THEN 1 ELSE 0 END  AS country_mismatch,
       h.amount_usd / NULLIF(h.prior_avg_amount, 0)                    AS amount_vs_prior_avg,
       DATEDIFF(DAY, h.prev_txn_date, h.txn_date)                      AS days_since_prev_txn,
       h.is_fraud
INTO dbo.feat_transaction
FROM h;
```

Customer-level counts per vehicle owner:

```sql
SELECT customer_id,
       COUNT(*)                                             AS n_vehicles,
       SUM(list_price_usd)                                  AS total_vehicle_value_usd,
       AVG(DATEDIFF(MONTH, purchase_date, @cutoff) * 1.0)   AS avg_vehicle_age_months,
       AVG(CASE WHEN fuel_type IN ('Electric','Hybrid') THEN 1.0 ELSE 0.0 END) AS electrified_share
FROM dbo.clean_dim_vehicles WHERE customer_id IS NOT NULL GROUP BY customer_id;
```

---

## 7. Create Relationship Table

Script: `sql/04_relationship_table.sql`.

- Adds **primary keys** to all seven clean tables and **foreign keys** between them (a star schema).
- Builds `dbo.table_relationships`, one row per parent-child link, with child rows, matched rows, **orphan rows** and **match %**.
- Power BI reads the keys and this table to set up its model.
- Safe to re-run: it drops old constraints first.

```
dim_branches  <- dim_customers (home_branch_id)
              <- dim_vehicles  (branch_id)  <- fact_service_records
dim_customers <- dim_vehicles (customer_id)
              <- fact_transactions, fact_support_interactions
dim_vehicles  <- fact_transactions, fact_service_records, fact_trips
```

---

## 8. Exploratory Data Analysis

Script: `python/05_eda.py`. It reads the feature tables from SQL Server, changes nothing, and saves every chart to `eda_outputs/`.

**What the data already shows, before any model**

| Signal | Finding |
|---|---|
| Churn by support contacts | 0 contacts: **17.6%**; 1-2: 24.3%; 3-4: 36.1%; 5 or more: **53.6%** |
| Fraud by night and international | Domestic, daytime: 1.7%; international, daytime: 7.8%; domestic, night: 6.3%; **international and night: 25.0%** |
| Class balance | Churn 30.7% (balanced enough); fraud 2.4% (rare, so accuracy would be misleading) |
| Missing values (customers) | Support CSAT 29.9%, income 11.8%, age 3.8%, no transactions 1.8% |

**What I did about it**
- Fraud is rare, so every fraud model is judged on **PR AUC**, not accuracy.
- Missing values are imputed on train only, and missing flags are kept as features.
- Skewed money and count columns get a log transform before scaling (step 13).

---

## 9. Analytical Base Table (SQL build and Python load)

**SQL build:** `sql/06_analytical_base_table.sql` creates the three modelling tables from the feature tables:
- `abt_customer`: 6,000 rows
- `abt_fraud`: 58,656 rows (transactions without a fraud label or date are dropped)
- `abt_service`: 13,836 rows

It also creates `abt_data_dictionary` (every column, its type and role: key, target or feature) and `ml_model_map` (objective, ML task, ABT, target, split rule).

**Python load:** `python/06_load_abt.py` reads the three ABTs from SQL Server, fixes data types (SQL decimals to floats, text dates to dates), runs sanity checks, and saves local copies in `abt_data/` so later steps do not need the database.

---

## 10. Encode Categorical Columns

Script: `python/07_encoding.py`.

- **Ordered columns** (loyalty tier: Bronze, Silver, Gold, Platinum) become 0 to 3.
- **Unordered columns** (country, customer type, channel, payment method, service type, and so on) are one-hot encoded. The most common level is the baseline and is dropped.
- Missing text becomes its own level, `Missing`.
- Unseen values at prediction time fall back to the baseline.
- No target is used, so there is no leakage.
- The recipe is saved as `encoding_spec.json`. The Streamlit app reuses the exact same recipe, so a new customer is encoded the same way as the training data.

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

---

## 12. Train Test Split

Script: `python/09_train_test_split.py`. The split matches how the model will be used.

| Objective | Split | Why |
|---|---|---|
| Churn, lifetime value, segmentation | 80/20 on customers, stratified on churn, **one shared split** | The same customer can never be train for one model and test for another |
| Fraud | **Time-based**: train up to 2025-06-30, test after | Fraud models predict the future. A random split would let them peek |
| Service cost | 80/20 **grouped by vehicle** | A vehicle never appears in both sets |

Test sizes: 1,200 customers, 9,821 transactions (249 fraud cases), 2,810 service jobs. The script prints the overlap check for vehicles and customers (zero).

---

## 13. Feature Scaling of Numerical Columns

Script: `python/10_feature_scaling.py`. **Everything is fitted on train only** and then applied to test.

Order of operations:
1. Keep only the model columns. Drop constants, duplicates and redundant dummy columns.
2. Fill missing values: median for numbers, most frequent value for 0/1 columns.
3. `log1p` on non-negative columns with skew above 1.5 (26 of the 41 continuous customer columns).
4. `StandardScaler` on continuous columns only. 0/1 columns stay as they are.

Saved per objective: `ready_<model>.pkl`, `scaler_<model>.joblib` and `prep_<model>.json` (the full recipe).

---

## 14. Train Model

Scripts: `python/11_train_<objective>.py`. Five candidate models for each classification and regression objective:

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

---

## 15. Predict x_test

Scripts: `python/12_predict_<objective>.py`. Each script loads the saved models and predicts the test set only.

- Churn and fraud: a probability for every customer or transaction, plus a flag at the chosen threshold.
- Lifetime value and service cost: a dollar prediction, never below zero.
- Segmentation: a segment for each customer, plus how close each customer is to the next segment.
- No scores are calculated here. Predictions are saved to `.pkl` and `.csv` for the next steps.

---

## 16. Compare X and Y

Scripts: `python/13_compare_<objective>.py`.

- **Target:** actual against predicted, as plain counts for classification and as dollar distributions and quartiles for regression.
- **Features:** which features drive each model (tree importances and standardised coefficients side by side). A driver is only trusted when both methods agree.
- Charts show the outcome rate or median across the values of the top features, in original units.

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

**Results**

| Objective | Champion | Verdict | What it means |
|---|---|---|---|
| Churn | Logistic Regression | USEFUL | Beats chance across its whole 95% range with ROC AUC of at least 0.65 |
| Fraud | Logistic Regression | STRONG | Beats chance across its whole 95% range with ROC AUC of at least 0.80 |
| Segmentation | K-Means, k = 3 | USEFUL (as tiers) | Silhouette 0.16 on train and test against 0.04 for noise. Lowest stability score 0.94. Segments differ clearly on spend (3.6x) and churn (9.5 points) for customers the model never saw |
| Lifetime value | Ridge Regression | STRONG | Beats the median baseline with confidence, R² of at least 0.5, ranking correlation of at least 0.7 |
| Service cost | Gradient Boosting | STRONG | Same bar as lifetime value |

The champion is the model picked on train cross-validation. If another model scores slightly higher on test, the champion is kept anyway, because choosing by test score would leak the test set.

> Exact metric values (PR AUC, ROC AUC, MAE, R²) for every candidate are printed by each `14_evaluate` script and saved to `churn_outputs/`, `fraud_outputs/`, `clv_outputs/` and `service_outputs/`.

---

## 18. Business Analysis

Script: `python/15_business_analysis.py`. It turns predictions into decisions and dollars, using test-set predictions only. Results below are on the test set (1,200 customers; 9,821 transactions; 2,810 service jobs).

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

**Service cost: forecast against actual (test jobs)**

Whole workshop: $1.83M predicted against $1.90M actual, **-3.6%**.

| Service type | Jobs | Actual cost | Forecast error |
|---|---|---|---|
| Routine Maintenance | 1,159 | $384,225 | -1.0% |
| Battery/EV System | 277 | $351,241 | -6.5% |
| Engine Repair | 220 | $305,510 | -2.9% |
| Tyres | 399 | $235,978 | -4.0% |
| Transmission | 137 | $228,595 | -4.0% |
| Body Work | 231 | $199,864 | -6.0% |
| Brake Repair | 387 | $198,962 | -1.8% |

**Action groups for customers:** combining churn risk with predicted value gives four groups: **Protect now** (high value, likely to leave), **Grow** (high value, loyal), **Low-cost win-back** (likely to leave, lower value) and **Maintain**. **Written back to SQL Server for Power BI:** `bi_customer_actions`, `bi_action_summary`, `bi_segment_profile`, `bi_kpi_by_group`, `bi_churn_retention_roi`, `bi_fraud_review_roi`, `bi_service_forecast_by_type`, `bi_service_forecast_by_country`. Step 16 adds `bi_ml_scorecard` and `bi_ml_model_comparison`, step 19 `bi_key_findings`, and step 20 `bi_recommendations`.

---

## 19. Save Model, Scaler and Features

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

---

## 20. Download Model, Scaler and Features

Script: `python/18_download_model_scaler_features.py`. It packs `deployment/` into one file, `vantage_deployment_package.zip`:

- `models/`: the 26 `.joblib` files
- `requirements.txt`: the library versions the models were trained with
- `MANIFEST.txt`: every file with its size and SHA-256 fingerprint
- `README.txt`: what each file is, with a short loading example

The script reopens the zip and loads all 26 files from inside it to prove the package is not damaged. Training versions are pinned (scikit-learn 1.9.0, pandas 3.0.3, numpy 2.4.6, joblib 1.5.3) because models saved with one scikit-learn version can fail to load with another.

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

**App pages:** Home, Customer churn, Fraud detection, Segmentation and behaviour, Lifetime value, Service cost, Customer 360 (churn, value and segment in one form), Batch scoring (upload a CSV, download predictions) and Model performance. Each form shows the most important inputs and fills the rest with typical training values. A banner warns when a model is not marked STRONG.

---

## 21. Key Findings and Insights

**Churn**
- Support contact volume is the clearest warning sign. Churn climbs from **17.6%** with no contacts to **53.6%** with five or more.
- Loyalty tier matters: **Bronze churns at 37.3%**, Silver 26.8%, Gold 23.7%, Platinum 23.6%.
- Individual customers churn at **32.3%** against **23.6%** for fleet operators.

**Fraud**
- Fraud is a combination of conditions, not one. A night-time international payment is fraud **25%** of the time, against 1.7% for a domestic daytime payment.
- Fraud value is concentrated: **5% of transactions hold 96% of fraud value**.

**Customer value and segments**
- Fleet operators spend an average of **$12,060** a year, corporate customers **$7,100** and individuals **$3,361**, which is 3.6 times more for fleets than individuals.
- One segment (31.5% of customers) holds **58.4%** of next-year spend and has the lowest churn. It is the revenue engine.

**Service cost**
- Three heavy repair types (Battery/EV, Engine, Transmission) are **23% of jobs but 46% of cost**. Routine maintenance is 41% of jobs but only 20% of cost.
- Battery/EV repairs average about **$1,268** a job and Transmission about **$1,669**, against about $332 for routine maintenance.
- Totals are forecast far better than single jobs: whole-workshop error is **-3.6%**, and every service type is within 7%.

**Cross-objective**
- Churn risk and predicted value together separate customers worth a personal call from customers who deserve a cheap automated message.
- Effort aimed at small, well-chosen groups captures most of the value: the top 20% of customers by risk, the top 5% of transactions, one segment of three.

---

## 22. Recommendations

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

## 23. Impact (Power BI)

Script: `python/21_impact.py`. The test data proves the models **rank** well. It cannot prove how customers or fraudsters **react**. So the impact is modelled in three scenarios (conservative, expected, optimistic), and every assumption is stated.

| Assumption | Conservative | Expected | Optimistic |
|---|---|---|---|
| Save rate (share of contacted would-be churners kept) | 10% | 25% | 40% |
| Gross margin on retained revenue | 15% | 20% | 25% |
| Fraud value in the review queue actually stopped | 50% | 70% | 90% |
| Spend uplift from priority service for top customers | 0% | 1% | 3% |
| Workshop cost saved by better planning | 0.25% | 0.75% | 1.5% |

Programme costs are included: a build cost, a yearly running cost, a three-year cash flow, payback and NPV at 10%.

**Evidence-backed examples (test set, step 18)**
- **Churn campaign:** contacting the riskiest 20% (240 customers) reaches churners worth **$217,337** of next-year spend. At a 25% save rate that is about **$54,000 retained for $4,800 of contact cost**.
- **Break-even:** with a 20% margin, the campaign pays back if the real save rate is above about **11%** (contact cost only).
- **Fraud:** a review queue of 491 transactions covers **$8.24M of the $8.59M** fraud value in the test period, for about $3,900 of review cost at an assumed $8 per review.
- **Service:** budget forecasts within **3.6%** of actual cost.

**Decision guide produced by the script**
- **Approve:** the benefit survives even the conservative case and the model is proven.
- **Pilot with a control group:** it pays in the expected case, but its key assumption is untested.
- **Defer:** it does not pay in the expected case, or the model is not proven.

A churn sensitivity table shows net profit for every contact depth and save rate, so a manager can see exactly what has to be true for the campaign to pay.

**Power BI page:** reads `bi_impact_scenarios`, `bi_impact_programme`, `bi_impact_decisions` and `bi_impact_churn_sensitivity` from SQL Server. Suggested visuals: scenario cards (conservative, expected, optimistic), a three-year cash-flow and payback chart, the approve / pilot / defer table, and a save-rate slider over the sensitivity table.

---

## 24. Conclusion (Power BI)

Five models, one pipeline, one app. The data was audited, cleaned and engineered in SQL, modelled in Python with no test-data leakage, and deployed in Streamlit.

- **All five objectives passed the test-data bar:** three STRONG (fraud, lifetime value, service cost) and two USEFUL (churn, segmentation as tiers).
- **The business value sits in small groups:** the riskiest 20% of customers for retention, the riskiest 5% of transactions for fraud, and one segment that carries 58% of spend.
- **The numbers are honest:** every dollar figure states its assumptions, and the next step is to run controlled pilots and replace those assumptions with measured results.

**Limits**
- Customer reaction to offers is assumed, not measured.
- Drivers shown by the models are associations, not proven causes.
- All data comes from one source and one period, so models need monitoring and retraining.
- Segments are tiers on a continuum, not sharply separated groups.

**Power BI page:** reads `bi_conclusion` (the closing statement, one row per point) next to `bi_ml_scorecard` (verdict for each objective), so the last page of the report shows the same message as this section.

**Next steps:** pilots with control groups, scheduled refresh of the `bi_*` tables, scheduled retraining and monitoring.

---

### Repository structure

```
sql/        01 audit · 02 cleaning · 03 features · 04 relationships · 06 analytical base tables
python/     05 EDA · 06 load ABT · 07 encode · 08 X/y · 09 split · 10 scale
            11 train · 12 predict · 13 compare · 14 evaluate  (one set per objective)
            15 business analysis · 16 scorecard · 17 save · 18 package · 19 findings
            20 recommendations · 21 impact · 22 conclusion
streamlit/  app.py · requirements.txt
powerbi/    Vantage_Automotive.pbix (reads the bi_* tables in SQL Server)
models/     26 joblib files (model, scaler, features, prep, meta per objective + encoding recipe)
```

**Author:** Adewale Oshinukan
