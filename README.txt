VANTAGE MOTORS GLOBAL - DEPLOYMENT PACKAGE
Created: 2026-11-21T14:49:40

OBJECTIVES
churn         champion: Logistic Regression  verdict: USEFUL             status: DEPLOY WITH CARE
fraud         champion: Logistic Regression  verdict: STRONG             status: DEPLOY
segmentation  champion: K-Means (k = 3)      verdict: USEFUL (as tiers)  status: DEPLOY WITH CARE
clv           champion: Ridge Regression     verdict: STRONG             status: DEPLOY
service       champion: Gradient Boosting    verdict: STRONG             status: DEPLOY

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
