"""Train and evaluate the best feasible bounded RadarPostureNet-v2 model.

This script chooses the lite path unless point-cloud tensors were actually
built. Validation is grouped by held-out session; random frame/window splits are
not used for the main result.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from posturenet_v2_common import REPO_ROOT, ensure_dir, safe_div, timestamp_utc, write_csv

try:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    HAVE_SKLEARN = True
    SKLEARN_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - dependency availability guard
    HAVE_SKLEARN = False
    SKLEARN_IMPORT_ERROR = str(exc)


LABELS = ["STANDING", "SITTING"]
BASELINE_DISPLAY = "baseline_current_displayed_pose"
BASELINE_ONNX = "baseline_raw_old_onnx_probability"
BASELINE_GATED = "baseline_old_smoothed_gated_pose"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def read_windows(dataset_root: Path) -> pd.DataFrame:
    windows_path = dataset_root / "posturenet_lite_windows.csv"
    if not windows_path.exists():
        raise FileNotFoundError(f"Missing lite window dataset: {windows_path}")
    df = pd.read_csv(windows_path)
    if df.empty:
        return df
    df["expected_pose"] = df["expected_pose"].astype(str).str.upper()
    return df[df["expected_pose"].isin(LABELS)].reset_index(drop=True)


def baseline_predictions(df: pd.DataFrame) -> dict[str, np.ndarray]:
    stand_display = pd.to_numeric(df.get("display_standing_rate", 0.0), errors="coerce").fillna(0.0).to_numpy()
    sit_display = pd.to_numeric(df.get("display_sitting_rate", 0.0), errors="coerce").fillna(0.0).to_numpy()
    unknown_display = pd.to_numeric(df.get("display_unknown_rate", 1.0), errors="coerce").fillna(1.0).to_numpy()
    stand_prob = pd.to_numeric(df.get("stand_prob_mean", 0.0), errors="coerce").fillna(0.0).to_numpy()
    sit_prob = pd.to_numeric(df.get("sit_prob_mean", 0.0), errors="coerce").fillna(0.0).to_numpy()

    display_pred = np.where((sit_display >= stand_display) & (unknown_display < 0.95), "SITTING", "STANDING")
    onnx_pred = np.where(sit_prob >= stand_prob, "SITTING", "STANDING")
    gated_pred = np.where(sit_display >= 0.5, "SITTING", np.where(stand_display >= 0.5, "STANDING", onnx_pred))
    return {
        BASELINE_DISPLAY: display_pred.astype(object),
        BASELINE_ONNX: onnx_pred.astype(object),
        BASELINE_GATED: gated_pred.astype(object),
    }


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "window_id",
        "session_id",
        "segment_id",
        "person_slot",
        "assigned_tid",
        "window_start_s",
        "window_end_s",
        "expected_pose",
        "expected_subpose",
        "expected_distance_m",
        "expected_position",
        "expected_position_encoded",
        "label_confidence",
        "assignment_confidence",
        "visibility_reliability_label",
        "cfg_family",
    }
    columns: list[str] = []
    for column in df.columns:
        if column in excluded:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().any():
            columns.append(column)
    return columns


def model_factories() -> dict[str, Any]:
    if not HAVE_SKLEARN:
        return {}
    return {
        "LogisticRegression": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=300, class_weight="balanced", random_state=42),
        ),
        "RandomForestClassifier": lambda: RandomForestClassifier(
            n_estimators=180,
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoostingClassifier": lambda: HistGradientBoostingClassifier(
            max_iter=80,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=42,
        ),
        "MLPClassifier_lite": lambda: make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=80,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=42,
            ),
        ),
    }


def grouped_session_predictions(df: pd.DataFrame, columns: list[str], model_name: str, factory: Any) -> tuple[np.ndarray, str]:
    y = df["expected_pose"].astype(str).to_numpy()
    predictions = np.full(len(df), "UNKNOWN", dtype=object)
    notes: list[str] = []
    for held_session in sorted(df["session_id"].astype(str).unique()):
        test_mask = df["session_id"].astype(str) == held_session
        train_mask = ~test_mask
        y_train = y[train_mask]
        if len(set(y_train.tolist())) < 2:
            notes.append(f"{held_session}: skipped fold because training set had one class")
            continue
        X_train = df.loc[train_mask, columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        X_test = df.loc[test_mask, columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        model = factory()
        try:
            model.fit(X_train, y_train)
            predictions[test_mask.to_numpy()] = model.predict(X_test)
        except Exception as exc:
            notes.append(f"{held_session}: {model_name} failed: {exc}")
    return predictions, "; ".join(notes)


def safe_accuracy(y_true: pd.Series, y_pred: np.ndarray, mask: pd.Series | np.ndarray | None = None) -> float:
    if mask is None:
        mask_array = np.ones(len(y_true), dtype=bool)
    else:
        mask_array = np.asarray(mask, dtype=bool)
    if mask_array.sum() == 0:
        return float("nan")
    return float((y_true.to_numpy()[mask_array] == y_pred[mask_array]).mean())


def metric_summary(df: pd.DataFrame, y_pred: np.ndarray, model_name: str, notes: str = "") -> dict[str, Any]:
    y = df["expected_pose"].astype(str)
    standing = y == "STANDING"
    sitting = y == "SITTING"
    standing_3m = standing & (pd.to_numeric(df["expected_distance_m"], errors="coerce").round(1) == 3.0)
    upright = df["expected_subpose"].astype(str).str.upper() == "SITTING_UPRIGHT"
    lean_back = df["expected_subpose"].astype(str).str.upper() == "SITTING_LEAN_BACK"
    lean_forward = df["expected_subpose"].astype(str).str.upper() == "SITTING_LEAN_FORWARD"
    pred = np.asarray(y_pred)
    return {
        "model_name": model_name,
        "validation": "leave_one_session_out",
        "overall_accuracy": safe_accuracy(y, pred),
        "standing_accuracy": safe_accuracy(y, pred, standing),
        "sitting_accuracy": safe_accuracy(y, pred, sitting),
        "upright_sitting_accuracy": safe_accuracy(y, pred, upright),
        "lean_back_sitting_accuracy": safe_accuracy(y, pred, lean_back),
        "lean_forward_sitting_accuracy": safe_accuracy(y, pred, lean_forward),
        "false_sitting_on_standing": safe_div(float(((pred == "SITTING") & standing.to_numpy()).sum()), float(standing.sum())),
        "false_sitting_on_standing_3m": safe_div(float(((pred == "SITTING") & standing_3m.to_numpy()).sum()), float(standing_3m.sum())),
        "false_standing_on_sitting": safe_div(float(((pred == "STANDING") & sitting.to_numpy()).sum()), float(sitting.sum())),
        "unknown_prediction_rate": safe_div(float((pred == "UNKNOWN").sum()), float(len(pred))),
        "notes": notes,
    }


def confusion_rows(df: pd.DataFrame, y_pred: np.ndarray, model_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y = df["expected_pose"].astype(str).to_numpy()
    labels = LABELS + ["UNKNOWN"]
    for actual in LABELS:
        for predicted in labels:
            rows.append(
                {
                    "model_name": model_name,
                    "actual": actual,
                    "predicted": predicted,
                    "count": int(((y == actual) & (y_pred == predicted)).sum()),
                }
            )
    return rows


def grouped_metrics(df: pd.DataFrame, y_pred: np.ndarray, group_column: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value, group in df.groupby(group_column, dropna=False, sort=True):
        idx = group.index.to_numpy()
        y = group["expected_pose"].astype(str).to_numpy()
        pred = y_pred[idx]
        standing = y == "STANDING"
        sitting = y == "SITTING"
        rows.append(
            {
                group_column: value,
                "windows": len(group),
                "accuracy": float((y == pred).mean()) if len(group) else float("nan"),
                "standing_accuracy": safe_div(float(((y == pred) & standing).sum()), float(standing.sum())),
                "sitting_accuracy": safe_div(float(((y == pred) & sitting).sum()), float(sitting.sum())),
                "false_sitting_on_standing": safe_div(float(((pred == "SITTING") & standing).sum()), float(standing.sum())),
            }
        )
    return rows


def select_best_model(comparison: pd.DataFrame) -> str:
    candidates = comparison[~comparison["model_name"].str.startswith("baseline_")].copy()
    if candidates.empty:
        candidates = comparison.copy()
    candidates["selection_score"] = (
        candidates["overall_accuracy"].fillna(0.0)
        + 0.5 * candidates["sitting_accuracy"].fillna(0.0)
        + 0.5 * candidates["standing_accuracy"].fillna(0.0)
        - candidates["false_sitting_on_standing_3m"].fillna(1.0)
    )
    return str(candidates.sort_values("selection_score", ascending=False).iloc[0]["model_name"])


def acceptance_decision(df: pd.DataFrame, comparison: pd.DataFrame, best_pred: np.ndarray, best_model: str) -> tuple[bool, list[dict[str, Any]]]:
    best = comparison[comparison["model_name"] == best_model].iloc[0].to_dict()
    baseline = comparison[comparison["model_name"] == BASELINE_DISPLAY]
    baseline_row = baseline.iloc[0].to_dict() if not baseline.empty else {}
    by_position = grouped_metrics(df, best_pred, "expected_position")
    position_acc = {str(row["expected_position"]).upper(): row["accuracy"] for row in by_position}
    center_acc = position_acc.get("CENTER", float("nan"))
    side_values = [position_acc.get("LEFT", float("nan")), position_acc.get("RIGHT", float("nan"))]
    side_values = [value for value in side_values if not np.isnan(value)]
    side_gap = max([abs(center_acc - value) for value in side_values], default=0.0) if not np.isnan(center_acc) else float("nan")
    side_min = min(side_values) if side_values else float("nan")
    people_rows = grouped_metrics(df, best_pred, "people_count")
    people_acc = {int(row["people_count"]): row["accuracy"] for row in people_rows if not pd.isna(row["people_count"])}
    single_acc = people_acc.get(1, float("nan"))
    two_acc = people_acc.get(2, float("nan"))
    two_person_ok = False if np.isnan(two_acc) else (two_acc >= 0.70 and (np.isnan(single_acc) or two_acc >= single_acc - 0.20))
    distance_values = set(pd.to_numeric(df["expected_distance_m"], errors="coerce").dropna().round(1).tolist())

    rows = [
        {
            "criterion": "Standing accuracy >= 95%",
            "passed": bool(best.get("standing_accuracy", 0.0) >= 0.95),
            "evidence": f"standing_accuracy={best.get('standing_accuracy', float('nan')):.4f}",
        },
        {
            "criterion": "False SITTING on standing_3m <= 5%",
            "passed": bool(best.get("false_sitting_on_standing_3m", 1.0) <= 0.05),
            "evidence": f"false_sitting_on_standing_3m={best.get('false_sitting_on_standing_3m', float('nan')):.4f}",
        },
        {
            "criterion": "Sitting accuracy improves over old runtime",
            "passed": bool(best.get("sitting_accuracy", 0.0) > baseline_row.get("sitting_accuracy", 1.0)),
            "evidence": f"best={best.get('sitting_accuracy', float('nan')):.4f}; baseline={baseline_row.get('sitting_accuracy', float('nan')):.4f}",
        },
        {
            "criterion": "Upright sitting improves over old runtime",
            "passed": bool(best.get("upright_sitting_accuracy", 0.0) > baseline_row.get("upright_sitting_accuracy", 1.0)),
            "evidence": f"best={best.get('upright_sitting_accuracy', float('nan')):.4f}; baseline={baseline_row.get('upright_sitting_accuracy', float('nan')):.4f}",
        },
        {
            "criterion": "Lean-forward sitting improves over old runtime",
            "passed": bool(best.get("lean_forward_sitting_accuracy", 0.0) > baseline_row.get("lean_forward_sitting_accuracy", 1.0)),
            "evidence": f"best={best.get('lean_forward_sitting_accuracy', float('nan')):.4f}; baseline={baseline_row.get('lean_forward_sitting_accuracy', float('nan')):.4f}",
        },
        {
            "criterion": "Left/right position gap is not severe",
            "passed": bool((not np.isnan(side_min)) and side_min >= 0.75 and (np.isnan(side_gap) or side_gap <= 0.20)),
            "evidence": f"center={center_acc:.4f}; side_min={side_min:.4f}; side_gap={side_gap:.4f}",
        },
        {
            "criterion": "Two-person accuracy does not collapse",
            "passed": bool(two_person_ok),
            "evidence": f"single_person={single_acc:.4f}; two_person={two_acc:.4f}",
        },
        {
            "criterion": "5m is reported separately",
            "passed": bool(5.0 in distance_values),
            "evidence": "metrics_by_distance.csv contains 5m rows" if 5.0 in distance_values else "no 5m windows available",
        },
        {
            "criterion": "Validation is grouped, not random frame split",
            "passed": True,
            "evidence": "leave-one-session-out grouped validation used for every trained model",
        },
    ]
    return all(row["passed"] for row in rows), rows


def train_final_model(df: pd.DataFrame, columns: list[str], best_model: str) -> Any:
    factories = model_factories()
    if best_model not in factories:
        return None
    model = factories[best_model]()
    X = df[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df["expected_pose"].astype(str).to_numpy()
    model.fit(X, y)
    return model


def export_candidate_model(
    df: pd.DataFrame,
    columns: list[str],
    best_model: str,
    training_report: Path,
    acceptance_rows: list[dict[str, Any]],
) -> Path | None:
    model = train_final_model(df, columns, best_model)
    if model is None:
        return None
    candidate_dir = ensure_dir(REPO_ROOT / "model_experiments" / "outputs" / "posturenet_v2_candidate")
    with (candidate_dir / "posturenet_v2_lite_model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    with (candidate_dir / "feature_schema.json").open("w", encoding="utf-8") as handle:
        json.dump({"feature_columns": columns, "dataset": "posturenet_lite_windows.csv"}, handle, indent=2)
    with (candidate_dir / "class_map.json").open("w", encoding="utf-8") as handle:
        json.dump({"classes": LABELS, "coarse_head": True, "subtype_head": False}, handle, indent=2)
    with (candidate_dir / "acceptance_report.json").open("w", encoding="utf-8") as handle:
        json.dump({"passed": True, "criteria": acceptance_rows}, handle, indent=2)
    shutil.copy2(training_report, candidate_dir / "POSTURENET_V2_TRAINING_REPORT.md")
    return candidate_dir


def write_runtime_plan(passed: bool, best_model: str, acceptance_rows: list[dict[str, Any]]) -> None:
    path = REPO_ROOT / "POSTURENET_V2_RUNTIME_INTEGRATION_PLAN.md"
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# RadarPostureNet-v2 Runtime Integration Plan\n\n")
        if passed:
            handle.write("The offline candidate passed acceptance and can be integrated behind explicit CLI flags in a separate runtime change.\n\n")
        else:
            handle.write("Runtime replacement was not performed because the v2 candidate did not pass all grouped-validation acceptance criteria.\n\n")
        handle.write("Required flags for any future integration:\n\n")
        handle.write("```powershell\n")
        handle.write("--pose-v2-enable\n--pose-v2-model \"<path>\"\n--pose-v2-mode shadow\n--pose-v2-mode replace\n--pose-v2-log\n--pose-v2-debug\n")
        handle.write("```\n\n")
        handle.write("Default runtime posture behavior must remain old behavior. Shadow mode should log old pose, v2 pose, final pose, confidence, reliability, and reason while the UI continues using the old output. Replace mode should be enabled only after acceptance stays green on grouped validation and live smoke tests.\n\n")
        handle.write(f"Best offline candidate: {best_model}\n\n")
        handle.write("| criterion | passed | evidence |\n")
        handle.write("| --- | --- | --- |\n")
        for row in acceptance_rows:
            handle.write(f"| {row['criterion']} | {row['passed']} | {row['evidence']} |\n")


def write_training_report(
    out: Path,
    dataset_root: Path,
    df: pd.DataFrame,
    comparison: pd.DataFrame,
    best_model: str,
    acceptance_passed: bool,
    acceptance_rows: list[dict[str, Any]],
    full_pointcloud_available: bool,
    lite_available: bool,
    trained_model_names: list[str],
) -> Path:
    path = out / "POSTURENET_V2_TRAINING_REPORT.md"
    best = comparison[comparison["model_name"] == best_model].iloc[0].to_dict() if not comparison.empty else {}
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# RadarPostureNet-v2 Training Report\n\n")
        handle.write(f"Timestamp: {timestamp_utc()}\n\n")
        handle.write("Main validation: leave-one-session-out grouped validation. No random frame/window split was used as the main result.\n\n")
        handle.write(f"Dataset root: `{dataset_root}`\n\n")
        handle.write(f"Windows evaluated: {len(df)}\n\n")
        handle.write(f"Full point-cloud data available: {'yes' if full_pointcloud_available else 'no'}\n\n")
        handle.write(f"Lite dataset available: {'yes' if lite_available else 'no'}\n\n")
        if not HAVE_SKLEARN:
            handle.write(f"scikit-learn unavailable; only baselines were evaluated. Import error: {SKLEARN_IMPORT_ERROR}\n\n")
        handle.write(f"Trained model candidates: {', '.join(trained_model_names) if trained_model_names else 'none'}\n\n")
        handle.write(f"Best model: {best_model}\n\n")
        handle.write(f"Acceptance passed: {'yes' if acceptance_passed else 'no'}\n\n")
        if best:
            handle.write("## Best Metrics\n\n")
            for key in [
                "overall_accuracy",
                "standing_accuracy",
                "sitting_accuracy",
                "upright_sitting_accuracy",
                "lean_forward_sitting_accuracy",
                "false_sitting_on_standing_3m",
                "false_standing_on_sitting",
            ]:
                handle.write(f"- {key}: {best.get(key, float('nan')):.4f}\n")
            handle.write("\n")
        handle.write("## Acceptance Criteria\n\n")
        handle.write("| criterion | passed | evidence |\n")
        handle.write("| --- | --- | --- |\n")
        for row in acceptance_rows:
            handle.write(f"| {row['criterion']} | {row['passed']} | {row['evidence']} |\n")
        handle.write("\n")
        if not acceptance_passed:
            handle.write(
                "No runtime model was exported or integrated. The v2 model remains offline-only until grouped validation protects standing, standing_3m, side positions, and two-person sessions simultaneously.\n"
            )
    return path


def load_cleaning_counts() -> tuple[int, int]:
    summary_path = REPO_ROOT / "analysis_outputs" / "posture_cleaning" / "cleaning_summary.csv"
    quality_path = REPO_ROOT / "analysis_outputs" / "posture_cleaning" / "segment_quality.csv"
    segments = 0
    person_instances = 0
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if "segments" in summary.columns:
            segments = int(pd.to_numeric(summary["segments"], errors="coerce").fillna(0).sum())
    if quality_path.exists():
        person_instances = len(pd.read_csv(quality_path))
    return segments, person_instances


def load_sessions_found() -> int:
    discovery_path = REPO_ROOT / "analysis_outputs" / "posture_registry_full" / "session_discovery.csv"
    if not discovery_path.exists():
        return 0
    discovery = pd.read_csv(discovery_path)
    if "exists" not in discovery.columns:
        return len(discovery)
    return int(discovery["exists"].astype(str).str.upper().isin({"TRUE", "1"}).sum())


def final_decision_rows(
    full_pointcloud_available: bool,
    lite_available: bool,
    comparison: pd.DataFrame,
    best_model: str,
    acceptance_passed: bool,
    runtime_integrated: bool,
) -> list[dict[str, str]]:
    best = comparison[comparison["model_name"] == best_model].iloc[0].to_dict() if not comparison.empty else {}
    baseline = comparison[comparison["model_name"] == BASELINE_DISPLAY]
    baseline_row = baseline.iloc[0].to_dict() if not baseline.empty else {}

    def improved(metric: str) -> str:
        if not best or not baseline_row:
            return "no"
        return "yes" if best.get(metric, 0.0) > baseline_row.get(metric, 1.0) else "no"

    return [
        {"question": "Is the data clean enough to train?", "answer": "yes" if lite_available else "no", "evidence": "cleaned segment files and lite windows were produced" if lite_available else "no usable windows"},
        {"question": "Is full point-cloud architecture possible?", "answer": "yes" if full_pointcloud_available else "no", "evidence": "pointcloud_availability_report.csv"},
        {"question": "Is lite architecture possible?", "answer": "yes" if lite_available else "no", "evidence": "posturenet_lite_windows.csv"},
        {"question": "Did the model improve sitting?", "answer": improved("sitting_accuracy"), "evidence": f"best={best.get('sitting_accuracy', float('nan')):.4f}; old={baseline_row.get('sitting_accuracy', float('nan')):.4f}"},
        {"question": "Did it protect standing?", "answer": "yes" if best.get("standing_accuracy", 0.0) >= 0.95 else "no", "evidence": f"standing_accuracy={best.get('standing_accuracy', float('nan')):.4f}"},
        {"question": "Did it protect standing_3m?", "answer": "yes" if best.get("false_sitting_on_standing_3m", 1.0) <= 0.05 else "no", "evidence": f"false_sitting_on_standing_3m={best.get('false_sitting_on_standing_3m', float('nan')):.4f}"},
        {"question": "Did it improve upright sitting?", "answer": improved("upright_sitting_accuracy"), "evidence": f"best={best.get('upright_sitting_accuracy', float('nan')):.4f}; old={baseline_row.get('upright_sitting_accuracy', float('nan')):.4f}"},
        {"question": "Did it improve lean-forward sitting?", "answer": improved("lean_forward_sitting_accuracy"), "evidence": f"best={best.get('lean_forward_sitting_accuracy', float('nan')):.4f}; old={baseline_row.get('lean_forward_sitting_accuracy', float('nan')):.4f}"},
        {"question": "Did it handle left/right position?", "answer": "see metrics", "evidence": "metrics_by_position.csv"},
        {"question": "Did it handle two-person sessions?", "answer": "see metrics", "evidence": "model comparison and metrics_by_session.csv"},
        {"question": "Is 5m reliable?", "answer": "see metrics", "evidence": "metrics_by_distance.csv reports 5m separately"},
        {"question": "Was runtime integration performed?", "answer": "yes" if runtime_integrated else "no", "evidence": "runtime only changes after acceptance; this pass did not modify UI runtime" if not runtime_integrated else "candidate passed and runtime was updated"},
        {"question": "Is the system production-ready?", "answer": "yes" if acceptance_passed and runtime_integrated else "no", "evidence": "all acceptance criteria must pass and runtime must be integrated safely"},
        {"question": "What is the next fix?", "answer": "add associated point-cloud logging", "evidence": "full architecture requires per-frame per-point xyz/snr/doppler with target association"},
    ]


def write_final_reports(
    dataset_root: Path,
    out: Path,
    comparison: pd.DataFrame,
    best_model: str,
    acceptance_passed: bool,
    runtime_integrated: bool,
    full_pointcloud_available: bool,
    lite_available: bool,
) -> None:
    segments, person_instances = load_cleaning_counts()
    sessions_found = load_sessions_found()
    decision_rows = final_decision_rows(
        full_pointcloud_available,
        lite_available,
        comparison,
        best_model,
        acceptance_passed,
        runtime_integrated,
    )
    registry_path = REPO_ROOT / "analysis_inputs" / "posture_session_registry_full.csv"
    registry = pd.read_csv(registry_path) if registry_path.exists() else pd.DataFrame()
    completion = REPO_ROOT / "POSTURENET_V2_DATA_ARCHITECTURE_COMPLETION.md"
    with completion.open("w", encoding="utf-8") as handle:
        handle.write("# RadarPostureNet-v2 Data Architecture Completion\n\n")
        handle.write(f"Sessions found: {sessions_found}\n\n")
        handle.write(f"Segments labeled: {segments}\n\n")
        handle.write(f"Person-instances labeled: {person_instances}\n\n")
        handle.write(f"Full point-cloud data available: {'yes' if full_pointcloud_available else 'no'}\n\n")
        handle.write(f"Lite dataset available: {'yes' if lite_available else 'no'}\n\n")
        handle.write(f"Best model: {best_model}\n\n")
        handle.write(f"Acceptance passed: {'yes' if acceptance_passed else 'no'}\n\n")
        handle.write(f"Runtime integrated: {'yes' if runtime_integrated else 'no'}\n\n")
        handle.write("The full architecture is specified, but current logs do not provide associated raw point-cloud tensors. The completed offline path therefore uses the lite dataset and grouped validation.\n")

    end_to_end = REPO_ROOT / "POSTURENET_V2_END_TO_END_REPORT.md"
    with end_to_end.open("w", encoding="utf-8") as handle:
        handle.write("# RadarPostureNet-v2 End-to-End Report\n\n")
        handle.write("## 1. Executive Summary\n\n")
        handle.write(
            "One bounded end-to-end pass was completed: old posture assets were preserved, all registered sessions were discovered, protocol segment templates were generated, cleaned segment assignments were produced, data modality was audited, a lite dataset was built, grouped validation was run, and runtime replacement was blocked unless acceptance passed.\n\n"
        )
        handle.write("## 2. Session Registry\n\n")
        if not registry.empty:
            handle.write("| session_id | people_count | positions | distances_m | notes |\n")
            handle.write("| --- | ---: | --- | --- | --- |\n")
            for _, row in registry.iterrows():
                handle.write(
                    f"| {row['session_id']} | {row['people_count']} | {row['positions']} | {row['distances_m']} | {str(row.get('notes', '')).replace('|', '/')} |\n"
                )
        handle.write("\n## 3. Data Cleaning Result\n\n")
        handle.write(f"Segments labeled: {segments}. Person-instances labeled: {person_instances}. See `analysis_outputs/posture_cleaning/DATA_CLEANING_REPORT.md`.\n\n")
        handle.write("## 4. Disappearance/Dropout Summary\n\n")
        handle.write("Disappearance and reliability evidence was retained in `analysis_outputs/posture_cleaning/disappearance_events.csv`; low-confidence segments were not silently discarded.\n\n")
        handle.write("## 5. Old Architecture Snapshot Result\n\n")
        handle.write("Old posture code and model artifacts were copied under `old_architecture`; the manifest and summary are in `old_architecture/manifests` and `old_architecture/reports`.\n\n")
        handle.write("## 6. Data Modality Audit\n\n")
        handle.write(f"Full point-cloud architecture possible: {'yes' if full_pointcloud_available else 'no'}. Lite architecture possible: {'yes' if lite_available else 'no'}. See `POSTURE_DATA_MODALITY_AUDIT.md` and `analysis_outputs/posturenet_v2_dataset/pointcloud_availability_report.csv`.\n\n")
        handle.write("## 7. Architecture Design\n\n")
        handle.write("The final full and lite architectures are specified in `RADAR_POSTURENET_V2_ARCHITECTURE.md`. The old ONNX output is treated as auxiliary input/teacher signal, not ground truth.\n\n")
        handle.write("## 8. Dataset Build Summary\n\n")
        handle.write(f"Lite dataset root: `{dataset_root}`. Lite dataset available: {'yes' if lite_available else 'no'}.\n\n")
        handle.write("## 9. Training Result\n\n")
        handle.write(f"Best model: {best_model}. See `{out / 'POSTURENET_V2_TRAINING_REPORT.md'}`.\n\n")
        handle.write("## 10. Validation Result\n\n")
        handle.write("Validation used leave-one-session-out grouping, with separate metrics by session, distance, position, and subpose. See the CSV outputs in `analysis_outputs/posturenet_v2_model`.\n\n")
        handle.write("## 11. Whether Model Passed Acceptance\n\n")
        handle.write(f"Acceptance passed: {'yes' if acceptance_passed else 'no'}.\n\n")
        handle.write("## 12. Whether Runtime Was Integrated\n\n")
        handle.write(f"Runtime integrated: {'yes' if runtime_integrated else 'no'}.\n\n")
        handle.write("## 13. UI/UX Changes If Any\n\n")
        handle.write("No UI runtime changes were made unless acceptance passed; default old behavior remains unchanged.\n\n")
        handle.write("## 14. What Is Production-Ready\n\n")
        handle.write("The offline registry, cleaning, dataset, modality audit, architecture specification, grouped validation, and reports are production-usable as analysis artifacts.\n\n")
        handle.write("## 15. What Is Not Production-Ready\n\n")
        handle.write("Runtime replacement is not production-ready unless all acceptance criteria pass and a flagged shadow/replace integration is implemented.\n\n")
        handle.write("## 16. Exact Next Step\n\n")
        handle.write("Add non-invasive logging of per-frame associated point-cloud rows: frame, TID/track index, x/y/z, snr, doppler, point quality, and association source. Then repeat the same bounded pipeline for full RadarPostureNet-v2.\n\n")
        handle.write("## 17. Limitations\n\n")
        handle.write("The lite model depends on old model probabilities, track metadata, quality flags, and display stability features. It cannot learn target-centered point geometry without associated point-cloud logs.\n\n")
        handle.write("## Final Decision Table\n\n")
        handle.write("| question | answer | evidence |\n")
        handle.write("| --- | --- | --- |\n")
        for row in decision_rows:
            handle.write(f"| {row['question']} | {row['answer']} | {row['evidence']} |\n")


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    out = ensure_dir(Path(args.out))
    df = read_windows(dataset_root)

    point_report_path = dataset_root / "pointcloud_availability_report.csv"
    point_report = pd.read_csv(point_report_path) if point_report_path.exists() else pd.DataFrame()
    full_pointcloud_available = bool(
        not point_report.empty
        and point_report.get("can_reconstruct_target_centered_tensors", pd.Series(dtype=bool)).astype(str).str.upper().isin({"TRUE", "1"}).any()
    )
    lite_available = not df.empty
    if df.empty:
        comparison = pd.DataFrame(
            [
                {
                    "model_name": "no_model",
                    "validation": "leave_one_session_out",
                    "overall_accuracy": 0.0,
                    "standing_accuracy": 0.0,
                    "sitting_accuracy": 0.0,
                    "upright_sitting_accuracy": 0.0,
                    "lean_back_sitting_accuracy": 0.0,
                    "lean_forward_sitting_accuracy": 0.0,
                    "false_sitting_on_standing": 1.0,
                    "false_sitting_on_standing_3m": 1.0,
                    "false_standing_on_sitting": 1.0,
                    "unknown_prediction_rate": 1.0,
                    "notes": "no usable lite windows",
                }
            ]
        )
        best_model = "no_model"
        best_pred = np.array([], dtype=object)
        trained_names: list[str] = []
    else:
        pred_by_model = baseline_predictions(df)
        notes_by_model = {name: "" for name in pred_by_model}
        columns = feature_columns(df)
        trained_names = []
        for model_name, factory in model_factories().items():
            pred, notes = grouped_session_predictions(df, columns, model_name, factory)
            pred_by_model[model_name] = pred
            notes_by_model[model_name] = notes
            trained_names.append(model_name)
        comparison_rows = [
            metric_summary(df, pred, model_name, notes_by_model.get(model_name, ""))
            for model_name, pred in pred_by_model.items()
        ]
        comparison = pd.DataFrame(comparison_rows)
        best_model = select_best_model(comparison)
        best_pred = pred_by_model[best_model]

    acceptance_passed, acceptance_rows = (
        acceptance_decision(df, comparison, best_pred, best_model) if not df.empty else (False, [])
    )
    comparison["acceptance_passed_for_best"] = comparison["model_name"].eq(best_model) & acceptance_passed
    write_csv(out / "model_comparison.csv", comparison.to_dict("records"), list(comparison.columns))

    if not df.empty:
        write_csv(out / "confusion_matrix.csv", confusion_rows(df, best_pred, best_model), ["model_name", "actual", "predicted", "count"])
        write_csv(out / "metrics_by_session.csv", grouped_metrics(df, best_pred, "session_id"), ["session_id", "windows", "accuracy", "standing_accuracy", "sitting_accuracy", "false_sitting_on_standing"])
        write_csv(out / "metrics_by_distance.csv", grouped_metrics(df, best_pred, "expected_distance_m"), ["expected_distance_m", "windows", "accuracy", "standing_accuracy", "sitting_accuracy", "false_sitting_on_standing"])
        write_csv(out / "metrics_by_position.csv", grouped_metrics(df, best_pred, "expected_position"), ["expected_position", "windows", "accuracy", "standing_accuracy", "sitting_accuracy", "false_sitting_on_standing"])
        write_csv(out / "metrics_by_subpose.csv", grouped_metrics(df, best_pred, "expected_subpose"), ["expected_subpose", "windows", "accuracy", "standing_accuracy", "sitting_accuracy", "false_sitting_on_standing"])
    else:
        for name, fields in [
            ("confusion_matrix.csv", ["model_name", "actual", "predicted", "count"]),
            ("metrics_by_session.csv", ["session_id", "windows", "accuracy", "standing_accuracy", "sitting_accuracy", "false_sitting_on_standing"]),
            ("metrics_by_distance.csv", ["expected_distance_m", "windows", "accuracy", "standing_accuracy", "sitting_accuracy", "false_sitting_on_standing"]),
            ("metrics_by_position.csv", ["expected_position", "windows", "accuracy", "standing_accuracy", "sitting_accuracy", "false_sitting_on_standing"]),
            ("metrics_by_subpose.csv", ["expected_subpose", "windows", "accuracy", "standing_accuracy", "sitting_accuracy", "false_sitting_on_standing"]),
        ]:
            write_csv(out / name, [], fields)

    report_path = write_training_report(
        out,
        dataset_root,
        df,
        comparison,
        best_model,
        acceptance_passed,
        acceptance_rows,
        full_pointcloud_available,
        lite_available,
        trained_names,
    )
    runtime_integrated = False
    if acceptance_passed and not best_model.startswith("baseline_") and best_model != "no_model":
        candidate_dir = export_candidate_model(df, feature_columns(df), best_model, report_path, acceptance_rows)
        runtime_integrated = False
        if candidate_dir is not None:
            with (candidate_dir / "RUNTIME_NEXT_STEP.md").open("w", encoding="utf-8") as handle:
                handle.write("Candidate exported. Runtime integration must still be implemented behind --pose-v2-* flags before replace mode is used.\n")
    write_runtime_plan(acceptance_passed, best_model, acceptance_rows)
    write_final_reports(
        dataset_root,
        out,
        comparison,
        best_model,
        acceptance_passed,
        runtime_integrated,
        full_pointcloud_available,
        lite_available,
    )

    segments, person_instances = load_cleaning_counts()
    sessions_found = load_sessions_found()
    print(f"Sessions found: {sessions_found}")
    print(f"Segments labeled: {segments}")
    print(f"Person-instances labeled: {person_instances}")
    print(f"Full point-cloud data available: {'yes' if full_pointcloud_available else 'no'}")
    print(f"Lite dataset available: {'yes' if lite_available else 'no'}")
    print(f"Model trained: {'yes' if trained_names else 'no'}")
    print(f"Best model: {best_model}")
    print(f"Acceptance passed: {'yes' if acceptance_passed else 'no'}")
    print(f"Runtime integrated: {'yes' if runtime_integrated else 'no'}")
    print("Final report: POSTURENET_V2_END_TO_END_REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
