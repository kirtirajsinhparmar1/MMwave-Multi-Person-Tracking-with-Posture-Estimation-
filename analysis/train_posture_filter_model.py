#!/usr/bin/env python
"""Train and evaluate offline second-stage posture filter candidates."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LIVE_SESSION = "sitting_relative_gate_refined_live_test"
LABEL = "expected_pose"
NON_FEATURE_PREFIXES = ("baseline_",)
NON_FEATURE_COLS = {
    "expected_pose",
    "expected_subpose",
    "expected_distance_m",
    "session_id",
    "segment_id",
    "cfg_family",
    "window_start_time_s",
    "window_end_time_s",
    "tid",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def pose_label(value: Any) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    text = str(value).strip().upper()
    if "STAND" in text:
        return "STANDING"
    if "SIT" in text:
        return "SITTING"
    if "MOVE" in text or "WALK" in text:
        return "MOVING"
    if "FALL" in text:
        return "FALLING"
    if "LY" in text or "LAY" in text:
        return "LYING"
    return text if text else "UNKNOWN"


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in NON_FEATURE_COLS or any(col.startswith(prefix) for prefix in NON_FEATURE_PREFIXES):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
            cols.append(col)
    return cols


def md_table(data: pd.DataFrame | pd.Series) -> str:
    if isinstance(data, pd.Series):
        df = data.rename("value").reset_index()
    else:
        df = data.copy()
    if df.empty:
        return "No rows."
    cols = [str(c) for c in df.columns]

    def fmt(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    rows = [[fmt(v) for v in row] for row in df.itertuples(index=False, name=None)]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def metrics_for(y_true: pd.Series, y_pred: pd.Series, rows: pd.DataFrame) -> dict[str, float]:
    y_true = y_true.map(pose_label)
    y_pred = y_pred.map(pose_label)
    stand = y_true == "STANDING"
    sit = y_true == "SITTING"
    standing_acc = float((y_pred[stand] == "STANDING").mean()) if stand.any() else np.nan
    sitting_acc = float((y_pred[sit] == "SITTING").mean()) if sit.any() else np.nan
    false_sit = float((y_pred[stand] == "SITTING").mean()) if stand.any() else np.nan
    false_stand = float((y_pred[sit] == "STANDING").mean()) if sit.any() else np.nan
    standing_3m = stand & (pd.to_numeric(rows.get("expected_distance_m"), errors="coerce") == 3.0)
    upright = rows.get("expected_subpose", pd.Series("", index=rows.index)).astype(str).str.upper().eq("SITTING_UPRIGHT")
    return {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else np.nan,
        "standing_accuracy": standing_acc,
        "sitting_accuracy": sitting_acc,
        "false_SITTING_on_STANDING": false_sit,
        "false_STANDING_on_SITTING": false_stand,
        "false_SITTING_on_STANDING_3m": float((y_pred[standing_3m] == "SITTING").mean()) if standing_3m.any() else np.nan,
        "upright_sitting_accuracy": float((y_pred[upright] == "SITTING").mean()) if upright.any() else np.nan,
    }


def confusion(y_true: pd.Series, y_pred: pd.Series, model: str, validation: str) -> pd.DataFrame:
    labels = ["STANDING", "SITTING", "MOVING", "UNKNOWN", "LYING", "FALLING", "OTHER"]
    rows = []
    yt = y_true.map(pose_label)
    yp = y_pred.map(pose_label)
    for actual in labels:
        for predicted in labels:
            count = int(((yt == actual) & (yp == predicted)).sum())
            if count:
                rows.append({"model": model, "validation": validation, "actual": actual, "predicted": predicted, "count": count})
    return pd.DataFrame(rows)


def make_models():
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "LogisticRegression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "RandomForestClassifier": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample", min_samples_leaf=3)),
            ]
        ),
        "HistGradientBoostingClassifier": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(random_state=42, learning_rate=0.05, max_iter=200)),
            ]
        ),
    }


def fit_predict_grouped(model, x: pd.DataFrame, y: pd.Series, groups: pd.Series) -> tuple[pd.Series, list[str]]:
    preds = pd.Series(index=x.index, dtype=object)
    used_groups: list[str] = []
    for group in sorted(groups.dropna().unique()):
        test = groups == group
        train = ~test
        if y[train].nunique() < 2 or not test.any():
            continue
        model.fit(x[train], y[train])
        preds.loc[test] = model.predict(x[test])
        used_groups.append(str(group))
    return preds.dropna(), used_groups


def evaluate_models(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    from sklearn.base import clone

    x = df[features].replace([np.inf, -np.inf], np.nan)
    y = df[LABEL].map(pose_label)
    rows = []
    confs = []
    importances = []
    detail: dict[str, Any] = {}

    baseline_specs = {
        "baseline_current_display": df["baseline_display_pose"].map(pose_label) if "baseline_display_pose" in df else pd.Series("UNKNOWN", index=df.index),
        "baseline_raw_max_probability": df["baseline_raw_prob_pose"].map(pose_label) if "baseline_raw_prob_pose" in df else pd.Series("UNKNOWN", index=df.index),
        "baseline_refined_gate_display_if_available": df["baseline_display_pose"].map(pose_label) if "relative_gate_trigger_rate_if_available" in df else pd.Series("UNKNOWN", index=df.index),
    }
    for name, pred in baseline_specs.items():
        metric = metrics_for(y, pred, df)
        metric.update({"model": name, "validation": "all_labeled_examples", "heldout_groups": "not_applicable"})
        rows.append(metric)
        confs.append(confusion(y, pred, name, "all_labeled_examples"))

    models = make_models()
    for name, model in models.items():
        try:
            grouped_pred, groups = fit_predict_grouped(clone(model), x, y, df["session_id"].astype(str))
            if not grouped_pred.empty:
                subset = df.loc[grouped_pred.index]
                metric = metrics_for(y.loc[grouped_pred.index], grouped_pred, subset)
                metric.update({"model": name, "validation": "leave_one_session_out", "heldout_groups": ",".join(groups), "notes": ""})
                rows.append(metric)
                confs.append(confusion(y.loc[grouped_pred.index], grouped_pred, name, "leave_one_session_out"))
        except Exception as exc:  # noqa: BLE001 - keep other candidates running
            rows.append({"model": name, "validation": "leave_one_session_out", "heldout_groups": "", "accuracy": np.nan, "notes": f"fit failed: {exc}"})

        live_test = df["session_id"].astype(str).eq(LIVE_SESSION)
        train = ~live_test
        if live_test.any() and train.any() and y[train].nunique() >= 2:
            try:
                candidate = clone(model)
                candidate.fit(x[train], y[train])
                pred = pd.Series(candidate.predict(x[live_test]), index=df.index[live_test])
                metric = metrics_for(y[live_test], pred, df[live_test])
                metric.update({"model": name, "validation": "older_sessions_to_live_session", "heldout_groups": LIVE_SESSION, "notes": ""})
                rows.append(metric)
                confs.append(confusion(y[live_test], pred, name, "older_sessions_to_live_session"))
                if name == "RandomForestClassifier":
                    rf = candidate.named_steps["model"]
                    for feature, value in zip(features, rf.feature_importances_):
                        importances.append({"model": name, "validation": "older_sessions_to_live_session", "feature": feature, "importance": float(value)})
                elif name == "LogisticRegression":
                    coef = candidate.named_steps["model"].coef_
                    for feature, value in zip(features, np.abs(coef).mean(axis=0)):
                        importances.append({"model": name, "validation": "older_sessions_to_live_session", "feature": feature, "importance": float(value)})
            except Exception as exc:  # noqa: BLE001
                rows.append({"model": name, "validation": "older_sessions_to_live_session", "heldout_groups": LIVE_SESSION, "accuracy": np.nan, "notes": f"fit failed: {exc}"})

    detail["baseline_display_live"] = None
    if df["session_id"].astype(str).eq(LIVE_SESSION).any() and "baseline_display_pose" in df:
        mask = df["session_id"].astype(str).eq(LIVE_SESSION)
        detail["baseline_display_live"] = metrics_for(y[mask], df.loc[mask, "baseline_display_pose"], df[mask])
    return pd.DataFrame(rows), pd.concat(confs, ignore_index=True) if confs else pd.DataFrame(), pd.DataFrame(importances), detail


def acceptance_decision(comparison: pd.DataFrame, detail: dict[str, Any]) -> tuple[bool, str, pd.Series | None]:
    live = comparison[comparison["validation"] == "older_sessions_to_live_session"].copy()
    baseline = detail.get("baseline_display_live")
    if live.empty or not baseline:
        return False, "No older-sessions-to-live held-out validation was available.", None
    candidates = live.sort_values(["sitting_accuracy", "accuracy"], ascending=False)
    for _, row in candidates.iterrows():
        standing_ok = row["standing_accuracy"] >= baseline.get("standing_accuracy", np.nan) - 1e-9
        false_3m_ok = pd.isna(row["false_SITTING_on_STANDING_3m"]) or pd.isna(baseline.get("false_SITTING_on_STANDING_3m", np.nan)) or row["false_SITTING_on_STANDING_3m"] <= baseline["false_SITTING_on_STANDING_3m"] + 1e-9
        sitting_ok = row["sitting_accuracy"] > baseline.get("sitting_accuracy", np.nan)
        upright_ok = pd.notna(row["upright_sitting_accuracy"]) and row["upright_sitting_accuracy"] > baseline.get("upright_sitting_accuracy", -1)
        if standing_ok and false_3m_ok and sitting_ok and upright_ok:
            return True, f"{row['model']} passed the offline acceptance criteria on the live held-out session.", row
    return False, "No candidate passed all offline acceptance criteria; no runtime model should be exported.", candidates.iloc[0] if len(candidates) else None


def write_report(out: Path, comparison: pd.DataFrame, feature_importance: pd.DataFrame, passed: bool, decision: str, best: pd.Series | None, features: list[str]) -> None:
    lines = [
        "# Posture Filter Model Report",
        "",
        "This is an offline grouped-validation report. It does not claim runtime improvement.",
        "",
        f"Acceptance criteria passed: {'yes' if passed else 'no'}",
        f"Decision: {decision}",
        "",
        f"Feature count: {len(features)}",
        "",
        "## Model Comparison",
        "",
        md_table(comparison) if not comparison.empty else "No model comparison rows were produced.",
        "",
        "## Best Candidate",
        "",
    ]
    if best is None:
        lines.append("No candidate was selected.")
    else:
        lines.append(md_table(best.to_frame().T))
    lines.extend(
        [
            "",
            "## Feature Importance",
            "",
            md_table(feature_importance.sort_values("importance", ascending=False).head(30)) if not feature_importance.empty else "No feature importance was available.",
            "",
            "## Limitations",
            "",
            "- Validation is grouped by session, but the number of real sessions is small.",
            "- Segment labels come from known protocols, not frame-by-frame human annotation.",
            "- Display pose is used only as a baseline feature/evaluation reference, not as ground truth.",
        ]
    )
    (out / "POSTURE_FILTER_MODEL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def maybe_write_plan(passed: bool, out_root: Path) -> None:
    if not passed:
        return
    path = Path("POSTURE_FILTER_RUNTIME_INTEGRATION_PLAN.md")
    lines = [
        "# Posture Filter Runtime Integration Plan",
        "",
        "No runtime code has been changed. This plan is only for a future integration pass.",
        "",
        "1. Insert the filter after the ONNX posture probabilities and existing motion/gate debug fields are computed.",
        "2. Use the same rolling feature window selected offline, with latency bounded by that window length.",
        "3. Keep original ONNX probabilities and current display label visible in logs/UI for debugging.",
        "4. Apply hysteresis or minimum-hold smoothing around the filter output to avoid flicker.",
        "5. Add a config flag to disable the filter and roll back to the current display path.",
        "6. Log input features, model decision, baseline display decision, and final decision per TID.",
        "7. Re-run live validation before enabling by default.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.dataset, low_memory=False)
    if df.empty:
        pd.DataFrame().to_csv(out / "model_comparison.csv", index=False)
        pd.DataFrame().to_csv(out / "confusion_matrix.csv", index=False)
        pd.DataFrame().to_csv(out / "feature_importance.csv", index=False)
        (out / "POSTURE_FILTER_MODEL_REPORT.md").write_text("No dataset rows available.\n", encoding="utf-8")
        print("No dataset rows available.")
        return 0
    try:
        features = feature_columns(df)
        comparison, conf, importance, detail = evaluate_models(df, features)
        passed, decision, best = acceptance_decision(comparison, detail)
    except ImportError as exc:
        features = feature_columns(df)
        comparison = pd.DataFrame([{"model": "sklearn_unavailable", "validation": "not_run", "accuracy": np.nan, "notes": str(exc)}])
        conf = pd.DataFrame()
        importance = pd.DataFrame()
        passed = False
        decision = f"sklearn was unavailable: {exc}"
        best = None
    comparison.to_csv(out / "model_comparison.csv", index=False)
    conf.to_csv(out / "confusion_matrix.csv", index=False)
    importance.sort_values("importance", ascending=False).to_csv(out / "feature_importance.csv", index=False)
    write_report(out, comparison, importance, passed, decision, best, features)
    maybe_write_plan(passed, out)
    print(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
