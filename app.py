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