import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, ExtraTreesClassifier
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, confusion_matrix,
    average_precision_score
)

st.set_page_config(page_title="AI DC Test Optimizer", page_icon="⚙️", layout="wide")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

@st.cache_data
def load_data():
    die = pd.read_csv(os.path.join(DATA, "dc_test_results.csv"))
    sweep = pd.read_csv(os.path.join(DATA, "settling_sweep.csv"))
    catalog = pd.read_csv(os.path.join(DATA, "test_catalog.csv"))
    return die, sweep, catalog

try:
    df, sweep, catalog = load_data()
except Exception as exc:
    st.error(f"Could not load the project CSV files. Check that the data folder contains all three CSV files. Details: {exc}")
    st.stop()

test_names = catalog["test"].tolist()
pf_cols = [f"{t}_PF" for t in test_names if f"{t}_PF" in df.columns]
measurement_cols = [t for t in test_names if t in df.columns]

st.title("⚙️ AI-Optimized DC Parametric Test Time Reduction")
st.caption("Machine-learning prototype for semiconductor test analysis, predictive measurement and risk-aware optimization.")
st.info("This application uses a synthetic demonstration dataset. Its recommendations are for offline evaluation only—not authorization to change a production semiconductor test program.")

# Top-level metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Devices analyzed", f"{len(df):,}")
m2.metric("Lots", f"{df['lot'].nunique():,}")
m3.metric("Observed sort pass rate", f"{df['sort_pass'].mean()*100:.2f}%")
m4.metric("Mean recorded DC test time", f"{df['dc_test_time_ms'].mean():.1f} ms")

tabs = st.tabs([
    "Overview", "Test Analysis", "Settling Optimization",
    "Predictive Testing", "Anomaly Screening", "Optimization & Risk"
])

with tabs[0]:
    st.subheader("Dataset overview")
    a, b = st.columns(2)
    with a:
        lot_yield = df.groupby("lot", as_index=False)["sort_pass"].mean()
        fig = px.bar(lot_yield, x="lot", y="sort_pass", title="Sort pass rate by manufacturing lot",
                     labels={"sort_pass":"Pass rate", "lot":"Lot"}, range_y=[0,1])
        st.plotly_chart(fig, width="stretch")
    with b:
        bins = df["hard_bin"].value_counts().sort_index().rename_axis("hard_bin").reset_index(name="devices")
        fig = px.bar(bins, x="hard_bin", y="devices", title="Device count by hard bin",
                     labels={"hard_bin":"Hard bin", "devices":"Devices"})
        st.plotly_chart(fig, width="stretch")
    st.markdown("**Dataset dimensions**")
    st.write(f"Main dataset: **{df.shape[0]:,} rows × {df.shape[1]} columns**")
    st.write(f"Settling sweep: **{sweep.shape[0]:,} rows** · Test catalogue: **{len(catalog)} tests**")
    st.dataframe(df.head(10), width="stretch")

with tabs[1]:
    st.subheader("Test failure and correlation analysis")
    fail_summary = []
    for test in test_names:
        col = f"{test}_PF"
        if col in df.columns:
            fails = int((df[col] == "F").sum())
            fail_summary.append({"Test":test, "Failures":fails, "Failure rate (%)":100*fails/len(df)})
    fail_df = pd.DataFrame(fail_summary).sort_values("Failures", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(fail_df, x="Test", y="Failures", title="Failures per DC test")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, width="stretch")
    with c2:
        corr_tests = [c for c in ["IIL_A","IIL_B","IIH_A","IIH_B","IDDQ_V1","IDDQ_V2","IDDQ_V3","IDDQ_V4","IDDQ_V5","IDD_STATIC"] if c in df.columns]
        corr = df[corr_tests].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Measurement correlation matrix", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig, width="stretch")
    st.markdown("**Per-test failure summary**")
    st.dataframe(fail_df, width="stretch")
    st.caption("High correlation can indicate similar behavior, but correlation alone does not prove that a test is safe to remove.")

with tabs[2]:
    st.subheader("Settling time and sample-count analysis")
    selected_test = st.selectbox("Select measurement", sorted(sweep["test"].unique()))
    temp_choice = st.selectbox("Temperature", sorted(sweep["temp_C"].unique()), format_func=lambda x: f"{x} °C")
    sub = sweep[(sweep["test"] == selected_test) & (sweep["temp_C"] == temp_choice)].copy()
    sub["error"] = sub["measured_value"] - sub["reference_value"]
    summary = sub.groupby(["wait_ms","n_samples"], as_index=False).agg(
        bias=("error","mean"), rmse=("error",lambda x: float(np.sqrt(np.mean(np.square(x))))),
        std_error=("error","std"))
    fig = px.line(summary, x="wait_ms", y="rmse", color="n_samples", markers=True,
                  title=f"Measurement RMSE vs wait time — {selected_test} at {temp_choice} °C",
                  labels={"wait_ms":"Wait time (ms)","rmse":"RMSE","n_samples":"Samples"})
    st.plotly_chart(fig, width="stretch")
    st.write("The table reports observed bias and repeatability for each tested wait/sample setting.")
    st.dataframe(summary.sort_values(["rmse","wait_ms"]), width="stretch")
    st.warning("Choose a setting only after checking bias and repeatability against the measurement tolerance. The demo does not establish production Gage R&R or Cpk.")

with tabs[3]:
    st.subheader("Predictive testing: estimate IDDQ_V5")
    st.write("A Random Forest model predicts the expensive IDDQ_V5 measurement using earlier/cheaper measurements and context. Validation holds out entire manufacturing lots to reduce device-level leakage.")
    feature_cols = ["IDDQ_V1","IDDQ_V2","IDDQ_V3","IDDQ_V4","IIL_A","IIL_B","IDD_STATIC","temp_C","die_x","die_y"]
    target = "IDDQ_V5"
    model_df = df[feature_cols + [target,"lot","IDDQ_V5_PF"]].dropna().copy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(model_df[feature_cols], model_df[target], groups=model_df["lot"]))
    train, test = model_df.iloc[train_idx], model_df.iloc[test_idx]
    model = RandomForestRegressor(n_estimators=160, min_samples_leaf=3, random_state=42, n_jobs=-1)
    model.fit(train[feature_cols], train[target])
    pred = model.predict(test[feature_cols])
    mae = mean_absolute_error(test[target], pred)
    rmse = float(np.sqrt(mean_squared_error(test[target], pred)))
    r2 = r2_score(test[target], pred)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Held-out lot MAE", f"{mae:.4f} µA")
    c2.metric("RMSE", f"{rmse:.4f} µA")
    c3.metric("R²", f"{r2:.4f}")
    c4.metric("Held-out devices", f"{len(test):,}")
    st.write("Held-out lots:", ", ".join(sorted(test["lot"].unique())))
    result = pd.DataFrame({"Actual IDDQ_V5":test[target].values, "Predicted IDDQ_V5":pred})
    fig = px.scatter(result.sample(min(1500,len(result)), random_state=42), x="Actual IDDQ_V5", y="Predicted IDDQ_V5",
                     title="Actual vs predicted IDDQ_V5")
    st.plotly_chart(fig, width="stretch")
    st.markdown("**Important safety rule:** this prototype does not automatically skip IDDQ_V5. A real skip decision needs a calibrated prediction interval, held-out escape analysis, and quality approval.")
    st.caption("Model validation is a demonstration on the supplied synthetic data. Results may vary if the data or split changes.")

with tabs[4]:
    st.subheader("AI-based latent-defect screening")
    st.write(
        "An Extra Trees classifier learns patterns from the labeled synthetic "
        "benchmark. Validation holds out entire manufacturing lots so devices "
        "from the same lot do not appear in both training and validation."
    )

    features = [
        "IIL_A", "IIL_B", "IDDQ_V1", "IDDQ_V2",
        "IDDQ_V3", "IDDQ_V4", "IDDQ_V5", "IDD_STATIC"
    ]
    target_col = "is_latent_defect_truth"

    missing_cols = [c for c in features + [target_col, "lot"] if c not in df.columns]

    if missing_cols:
        st.error("The dataset is missing required columns: " + ", ".join(missing_cols))
    else:
        screen_df = (
            df.loc[df["sort_pass"] == 1]
            .dropna(subset=features + [target_col, "lot"])
            .copy()
            .reset_index(drop=True)
        )
        screen_df[target_col] = screen_df[target_col].astype(int)

        X = screen_df[features]
        y = screen_df[target_col]
        groups = screen_df["lot"]

        class_counts = y.value_counts().to_dict()
        n_lots = groups.nunique()

        st.markdown("**Benchmark label distribution among devices that passed initial sort**")
        label_a, label_b, label_c = st.columns(3)
        label_a.metric("Devices screened", f"{len(screen_df):,}")
        label_b.metric("Normal devices", f"{class_counts.get(0, 0):,}")
        label_c.metric("Known latent defects", f"{class_counts.get(1, 0):,}")

        if y.nunique() < 2 or class_counts.get(1, 0) < 2:
            st.error(
                "There are not enough examples from both classes to train and "
                "evaluate a classifier. Check the benchmark labels and filtering."
            )
        elif n_lots < 2:
            st.error("At least two manufacturing lots are required for lot-held-out validation.")
        else:
            n_splits = min(5, n_lots)
            oof_probabilities = np.full(len(screen_df), np.nan)
            fold_messages = []
            validation_ok = True

            cv = GroupKFold(n_splits=n_splits)

            with st.spinner("Training and validating the defect classifier by manufacturing lot..."):
                for fold_number, (train_idx, valid_idx) in enumerate(
                    cv.split(X, y, groups=groups), start=1
                ):
                    y_train = y.iloc[train_idx]

                    # A fold cannot learn a binary decision if its training data
                    # contains only one class.
                    if y_train.nunique() < 2:
                        validation_ok = False
                        fold_messages.append(
                            f"Fold {fold_number}: training data contains only one class."
                        )
                        break

                    fold_model = ExtraTreesClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1
                    )
                    fold_model.fit(X.iloc[train_idx], y_train)

                    positive_class_positions = np.where(fold_model.classes_ == 1)[0]
                    if len(positive_class_positions) != 1:
                        validation_ok = False
                        fold_messages.append(
                            f"Fold {fold_number}: positive class was unavailable."
                        )
                        break

                    positive_position = int(positive_class_positions[0])
                    oof_probabilities[valid_idx] = fold_model.predict_proba(
                        X.iloc[valid_idx]
                    )[:, positive_position]

                    held_out_lots = sorted(groups.iloc[valid_idx].astype(str).unique())
                    fold_messages.append(
                        f"Fold {fold_number}: held-out lot(s): {', '.join(held_out_lots)}"
                    )

            if not validation_ok or np.isnan(oof_probabilities).any():
                st.error(
                    "Lot-held-out validation could not be completed safely. "
                    "Review the defect distribution across manufacturing lots."
                )
                for message in fold_messages:
                    st.write(message)
            else:
                threshold = st.slider(
                    "Defect alert threshold",
                    min_value=0.05,
                    max_value=0.95,
                    value=0.50,
                    step=0.05,
                    help=(
                        "A device is flagged when its predicted defect probability "
                        "is at least this threshold. Lower thresholds usually flag "
                        "more devices and may increase false alarms."
                    )
                )

                predictions = (oof_probabilities >= threshold).astype(int)

                precision = precision_score(y, predictions, zero_division=0)
                recall = recall_score(y, predictions, zero_division=0)
                f1 = f1_score(y, predictions, zero_division=0)
                avg_precision = average_precision_score(y, oof_probabilities)

                cm = confusion_matrix(y, predictions, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()

                st.markdown("### Lot-held-out model performance")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Precision", f"{precision:.2%}")
                k2.metric("Recall", f"{recall:.2%}")
                k3.metric("F1-score", f"{f1:.2%}")
                k4.metric("Average precision", f"{avg_precision:.2%}")

                st.caption(
                    f"At this threshold: {tp:,} known defects detected, "
                    f"{fn:,} missed, and {fp:,} normal devices flagged."
                )

                st.markdown("### Confusion matrix")
                confusion_df = pd.DataFrame(
                    cm,
                    index=["Actual normal", "Actual defect"],
                    columns=["Predicted normal", "Predicted defect"]
                )
                st.dataframe(confusion_df, width="stretch")

                st.markdown("### Defect probability distribution")
                probability_df = pd.DataFrame({
                    "Predicted defect probability": oof_probabilities,
                    "Actual class": np.where(y.to_numpy() == 1, "Defect", "Normal")
                })
                probability_fig = px.histogram(
                    probability_df,
                    x="Predicted defect probability",
                    color="Actual class",
                    nbins=30,
                    barmode="overlay",
                    opacity=0.70,
                    title="Out-of-fold probabilities by benchmark class"
                )
                st.plotly_chart(probability_fig, width="stretch")

                # Fit a final model on all labeled benchmark records for
                # feature-importance inspection and future offline scoring.
                final_model = ExtraTreesClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
                final_model.fit(X, y)

                importance_df = pd.DataFrame({
                    "Measurement": features,
                    "Importance": final_model.feature_importances_
                }).sort_values("Importance", ascending=True)

                st.markdown("### Important defect indicators")
                importance_fig = px.bar(
                    importance_df,
                    x="Importance",
                    y="Measurement",
                    orientation="h",
                    title="Extra Trees feature importance"
                )
                st.plotly_chart(importance_fig, width="stretch")

                # The displayed probabilities are out-of-fold predictions,
                # not predictions from a model trained on those same rows.
                result_columns = [
                    c for c in ["ecid", "lot", "wafer", "tester", "site", "temp_C"]
                    if c in screen_df.columns
                ]
                results = screen_df[result_columns].copy()
                results["Actual benchmark label"] = y.to_numpy()
                results["Predicted defect probability (%)"] = (
                    oof_probabilities * 100
                ).round(2)
                results["Predicted class"] = predictions
                results["Screening result"] = np.where(
                    predictions == 1, "Flag for review", "Not flagged"
                )

                flagged = results.loc[results["Predicted class"] == 1].copy()
                flagged = flagged.sort_values(
                    "Predicted defect probability (%)", ascending=False
                )

                st.markdown("### Devices flagged for review")
                st.write(f"Flagged devices: **{len(flagged):,}**")
                st.dataframe(flagged.head(200), width="stretch")

                csv_data = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download full screening report (CSV)",
                    data=csv_data,
                    file_name="latent_defect_screening_report.csv",
                    mime="text/csv"
                )

                with st.expander("Validation fold details"):
                    for message in fold_messages:
                        st.write(message)

                st.warning(
                    "This is a synthetic benchmark with very few labeled defects. "
                    "Metrics are based on out-of-fold predictions grouped by lot. "
                    "The feature-importance chart is fitted on the full labeled "
                    "benchmark and is descriptive, not an independent validation. "
                    "Do not use these results to skip or alter production tests."
                )

with tabs[5]:
    st.subheader("Optimization candidates and escape-risk checks")
    rows = []
    for test in test_names:
        pf = f"{test}_PF"
        if pf not in df.columns:
            continue
        fail = df[pf] == "F"
        other_pf = [c for c in pf_cols if c != pf]
        passes_others = (df[other_pf] == "P").all(axis=1)
        unique_fails = int((fail & passes_others).sum())
        rows.append({"Test":test, "Failures":int(fail.sum()), "Unique failures if removed":unique_fails,
                     "Test time (ms)":float(catalog.loc[catalog["test"]==test,"test_time_ms"].iloc[0])})
    risk_df = pd.DataFrame(rows)
    risk_df["Decision guidance"] = np.where(risk_df["Unique failures if removed"] > 0,
                                               "Keep full coverage / investigate",
                                               "Candidate for controlled sampling study")
    st.dataframe(risk_df.sort_values(["Unique failures if removed","Test time (ms)"], ascending=[False,False]),
                 width="stretch")
    st.markdown("**Illustrative sampling scenario**")
    chosen = st.selectbox("Test to evaluate", test_names, index=test_names.index("IIL_B") if "IIL_B" in test_names else 0)
    sampling_rate = st.slider("Sampling rate (% of devices still receiving this test)", 1, 100, 5)
    time_ms = float(catalog.loc[catalog["test"]==chosen,"test_time_ms"].iloc[0])
    gross_saving = time_ms * (1 - sampling_rate/100)
    unique = int(risk_df.loc[risk_df["Test"]==chosen,"Unique failures if removed"].iloc[0])
    st.metric("Illustrative gross time saved per device", f"{gross_saving:.1f} ms")
    if unique:
        st.error(f"Risk warning: {unique} devices fail only {chosen} in this dataset. Removing or sampling it could allow unique failures to escape.")
    else:
        st.warning("No unique failures were observed for this test in this synthetic dataset. This is not proof that the test is safe to remove; evaluate confidence bounds and retain monitoring.")
    st.caption("Gross savings assume the test time is paid on every device. Actual savings depend on test order, stop-on-first-fail behavior, multisite effects and production validation.")
