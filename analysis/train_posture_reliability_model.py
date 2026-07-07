"""Train and evaluate an offline conservative posture reliability model.

The model predicts whether the current/old posture output should be trusted.
It does not predict a replacement posture label and it is not exported for
runtime use by this script.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - fallback is used only when sklearn is absent.
    RandomForestClassifier = None
    LogisticRegression = None
    StandardScaler = None
    confusion_matrix = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--failure-map", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def as_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        parsed = float(value)
        if math.isnan(parsed):
            return default
        return parsed
    except (TypeError, ValueError):
        return default


def norm_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().upper()


def coarse_pose(value: object) -> str:
    text = norm_text(value)
    if "SIT" in text:
        return "SITTING"
    if "STAND" in text:
        return "STANDING"
    if "MOVE" in text or "WALK" in text:
        return "MOVING"
    if "LIE" in text:
        return "LYING"
    if "FALL" in text:
        return "FALLING"
    return "UNKNOWN"


def numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def predicted_display_pose(row: pd.Series) -> str:
    rates = {
        "STANDING": as_float(row.get("display_standing_rate")),
        "SITTING": as_float(row.get("display_sitting_rate")),
        "MOVING": as_float(row.get("display_moving_rate")),
        "UNKNOWN": as_float(row.get("display_unknown_rate")),
    }
    label, value = max(rates.items(), key=lambda item: item[1])
    if value > 0:
        return label
    for col in ("old_display_pose", "display_pose", "displayed_pose", "final_pose"):
        if col in row.index:
            return coarse_pose(row.get(col))
    return "UNKNOWN"


def add_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["expected_pose_norm"] = out.get("expected_pose", "").map(coarse_pose)
    out["expected_subpose_norm"] = out.get("expected_subpose", "").map(norm_text)
    out["expected_position_norm"] = out.get("expected_position", "").map(norm_text)
    out["expected_distance_m_num"] = numeric(out, "expected_distance_m")
    out["people_count_num"] = numeric(out, "people_count")
    out["old_display_pred"] = out.apply(predicted_display_pose, axis=1)
    out["display_correct"] = out["old_display_pred"] == out["expected_pose_norm"]
    out["is_standing_3m"] = (
        (out["expected_pose_norm"] == "STANDING")
        & (out["expected_distance_m_num"].sub(3.0).abs() <= 0.25)
    )
    out["old_confidence_margin"] = (numeric(out, "sit_prob_mean") - numeric(out, "stand_prob_mean")).abs()
    out["max_display_rate"] = pd.concat(
        [
            numeric(out, "display_standing_rate"),
            numeric(out, "display_sitting_rate"),
            numeric(out, "display_moving_rate"),
            numeric(out, "display_unknown_rate"),
        ],
        axis=1,
    ).max(axis=1)
    out["quality_bad"] = (
        (numeric(out, "tracking_presence_rate", default=1.0) < 0.75)
        | (numeric(out, "pose_presence_rate", default=1.0) < 0.75)
        | (numeric(out, "ui_visible_rate", default=1.0) < 0.65)
        | (numeric(out, "disappearance_rate") > 0.20)
        | (numeric(out, "NO_POINTS_rate") > 0.35)
    )
    return out


def trust_label(row: pd.Series) -> str:
    if bool(row.get("quality_bad")):
        return "LOW_VISIBILITY"
    if row.get("old_display_pred") not in {"STANDING", "SITTING"}:
        return "UNCERTAIN"
    if bool(row.get("display_correct")) and as_float(row.get("max_display_rate")) >= 0.45:
        return "TRUST_CORRECT"
    if not bool(row.get("display_correct")):
        return "DO_NOT_TRUST"
    return "UNCERTAIN"


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "stand_prob_mean",
        "sit_prob_mean",
        "move_prob_mean_if_available",
        "lie_prob_mean_if_available",
        "fall_prob_mean_if_available",
        "stand_prob_std",
        "sit_prob_std",
        "sit_minus_stand_mean",
        "sit_minus_stand_std",
        "range_m_mean",
        "range_m_std",
        "target_z_mean",
        "target_z_std",
        "speed_mean",
        "speed_std",
        "geom_pts_mean",
        "geom_pts_std",
        "NO_POINTS_rate",
        "LOW_POINTS_rate",
        "OK_rate",
        "display_standing_rate",
        "display_sitting_rate",
        "display_moving_rate",
        "display_unknown_rate",
        "pose_switch_count",
        "tracking_presence_rate",
        "pose_presence_rate",
        "ui_visible_rate",
        "disappearance_rate",
        "tid_switch_count",
        "people_count",
        "expected_distance_m",
        "old_confidence_margin",
        "max_display_rate",
    ]
    features = pd.DataFrame(index=df.index)
    for col in feature_cols:
        features[col] = numeric(df, col)
    for col in ["old_display_pred", "expected_position_norm", "cfg_family"]:
        if col in df.columns:
            dummies = pd.get_dummies(df[col].fillna("UNKNOWN").astype(str), prefix=col)
            features = pd.concat([features, dummies], axis=1)
    return features.fillna(0.0)


def heuristic_predict(df: pd.DataFrame) -> np.ndarray:
    preds: list[str] = []
    for _, row in df.iterrows():
        if bool(row.get("quality_bad")):
            preds.append("LOW_VISIBILITY")
            continue
        pred = row.get("old_display_pred")
        margin = as_float(row.get("old_confidence_margin"))
        max_rate = as_float(row.get("max_display_rate"))
        low_points = as_float(row.get("LOW_POINTS_rate"))
        geom = as_float(row.get("geom_pts_mean"))
        standing_3m_sitting = bool(row.get("is_standing_3m")) and pred == "SITTING"
        if pred in {"STANDING", "SITTING"} and margin >= 0.22 and max_rate >= 0.60 and low_points < 0.65 and geom >= 4.0 and not standing_3m_sitting:
            preds.append("TRUST_CORRECT")
        elif standing_3m_sitting or low_points >= 0.85 or geom < 2.0:
            preds.append("DO_NOT_TRUST")
        else:
            preds.append("UNCERTAIN")
    return np.array(preds, dtype=object)


def align_features(train_x: pd.DataFrame, test_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = sorted(set(train_x.columns).union(set(test_x.columns)))
    return train_x.reindex(columns=columns, fill_value=0.0), test_x.reindex(columns=columns, fill_value=0.0)


def grouped_predictions(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    sessions = sorted(df["session_id"].dropna().unique())
    for session in sessions:
        train = df[df["session_id"] != session]
        test = df[df["session_id"] == session]
        pred = None
        if model_name == "heuristic_conservative" or LogisticRegression is None:
            pred = heuristic_predict(test)
        else:
            train_x, test_x = align_features(feature_frame(train), feature_frame(test))
            train_y = train["trust_target"].astype(str).to_numpy()
            if len(set(train_y)) < 2:
                pred = heuristic_predict(test)
            elif model_name == "LogisticRegression":
                scaler = StandardScaler()
                train_scaled = scaler.fit_transform(train_x)
                test_scaled = scaler.transform(test_x)
                model = LogisticRegression(max_iter=300, class_weight="balanced", random_state=7)
                model.fit(train_scaled, train_y)
                pred = model.predict(test_scaled)
            elif model_name == "RandomForestClassifier":
                model = RandomForestClassifier(
                    n_estimators=120,
                    max_depth=8,
                    min_samples_leaf=8,
                    class_weight="balanced_subsample",
                    random_state=7,
                    n_jobs=-1,
                )
                model.fit(train_x, train_y)
                pred = model.predict(test_x)
            else:
                pred = heuristic_predict(test)
        out = test.copy()
        out["predicted_trust_label"] = pred
        out["validation_group"] = session
        parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def bool_rate(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else 0.0


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def evaluate(pred_df: pd.DataFrame, model_name: str) -> dict[str, object]:
    trusted = pred_df["predicted_trust_label"] == "TRUST_CORRECT"
    known_pose = pred_df["old_display_pred"].isin(["STANDING", "SITTING"])
    correct = pred_df["display_correct"]
    standing_3m = pred_df["is_standing_3m"]
    false_sitting_3m = standing_3m & (pred_df["old_display_pred"] == "SITTING")
    correct_standing = correct & (pred_df["expected_pose_norm"] == "STANDING") & known_pose
    correct_sitting = correct & (pred_df["expected_pose_norm"] == "SITTING") & known_pose
    wrong = (~correct) & known_pose
    return {
        "model_name": model_name,
        "validation": "leave_one_session_out",
        "rows": len(pred_df),
        "trust_coverage": bool_rate(trusted),
        "trusted_accuracy": safe_div(float((trusted & correct).sum()), float(trusted.sum())),
        "trusted_false_sitting_on_standing_3m": safe_div(
            float((trusted & false_sitting_3m).sum()), float(standing_3m.sum())
        ),
        "baseline_false_sitting_on_standing_3m": safe_div(
            float(false_sitting_3m.sum()), float(standing_3m.sum())
        ),
        "correct_standing_preservation": safe_div(
            float((trusted & correct_standing).sum()), float(correct_standing.sum())
        ),
        "correct_sitting_preservation": safe_div(
            float((trusted & correct_sitting).sum()), float(correct_sitting.sum())
        ),
        "wrong_rejection_rate": safe_div(float(((~trusted) & wrong).sum()), float(wrong.sum())),
        "uncertain_or_low_visibility_rate": bool_rate(pred_df["predicted_trust_label"].isin(["UNCERTAIN", "LOW_VISIBILITY"])),
        "acceptance_passed": False,
    }


def acceptance(row: dict[str, object]) -> bool:
    return (
        as_float(row["trusted_false_sitting_on_standing_3m"]) < as_float(row["baseline_false_sitting_on_standing_3m"])
        and as_float(row["trust_coverage"]) >= 0.15
        and as_float(row["correct_standing_preservation"]) >= 0.35
        and as_float(row["correct_sitting_preservation"]) >= 0.35
        and as_float(row["wrong_rejection_rate"]) >= 0.50
    )


def trusted_case_rows(pred_df: pd.DataFrame, model_name: str) -> list[dict[str, object]]:
    cases = {
        "overall": pd.Series([True] * len(pred_df), index=pred_df.index),
        "standing_3m": pred_df["is_standing_3m"],
        "two_person": pred_df["people_count_num"] >= 2,
        "left_right_position": pred_df["expected_position_norm"].isin(["LEFT", "RIGHT"]),
        "five_meter": pred_df["expected_distance_m_num"].sub(5.0).abs() <= 0.25,
    }
    rows: list[dict[str, object]] = []
    for case, mask in cases.items():
        subset = pred_df[mask]
        trusted = subset["predicted_trust_label"] == "TRUST_CORRECT"
        correct = subset["display_correct"]
        rows.append(
            {
                "model_name": model_name,
                "case": case,
                "rows": len(subset),
                "trust_coverage": bool_rate(trusted),
                "trusted_accuracy": safe_div(float((trusted & correct).sum()), float(trusted.sum())),
                "baseline_accuracy": bool_rate(correct),
                "false_sitting_on_standing": safe_div(
                    float(((subset["expected_pose_norm"] == "STANDING") & (subset["old_display_pred"] == "SITTING")).sum()),
                    float((subset["expected_pose_norm"] == "STANDING").sum()),
                ),
            }
        )
    return rows


def write_report(out: Path, comparison: pd.DataFrame, best_row: pd.Series | None) -> None:
    if best_row is None:
        lines = [
            "# Reliability Model Report",
            "",
            "No reliability model could be evaluated.",
        ]
    else:
        passed = bool(best_row["acceptance_passed"])
        lines = [
            "# Reliability Model Report",
            "",
            "This offline model predicts whether the old/current posture output should be trusted. It does not predict a replacement posture class and no runtime model is exported by this task.",
            "",
            "## Best Candidate",
            "",
            f"- Best model: {best_row['model_name']}",
            f"- Trust coverage: {float(best_row['trust_coverage']) * 100.0:.2f}%",
            f"- Trusted accuracy: {float(best_row['trusted_accuracy']) * 100.0:.2f}%",
            f"- Baseline false SITTING on standing_3m: {float(best_row['baseline_false_sitting_on_standing_3m']) * 100.0:.2f}%",
            f"- Trusted false SITTING on standing_3m: {float(best_row['trusted_false_sitting_on_standing_3m']) * 100.0:.2f}%",
            f"- Correct standing preservation: {float(best_row['correct_standing_preservation']) * 100.0:.2f}%",
            f"- Correct sitting preservation: {float(best_row['correct_sitting_preservation']) * 100.0:.2f}%",
            f"- Wrong rejection rate: {float(best_row['wrong_rejection_rate']) * 100.0:.2f}%",
            f"- Acceptance passed: {'yes' if passed else 'no'}",
            "",
            "## Interpretation",
            "",
        ]
        if passed:
            lines.append("A conservative trust gate is feasible offline: it reduces trusted standing_3m false-sitting errors while preserving some correct standing and sitting windows. Runtime integration is intentionally not performed here.")
        else:
            lines.append("The reliability gate did not meet all acceptance criteria. It should remain offline analysis only until point-cloud logging and stronger validation are available.")
    (out / "RELIABILITY_MODEL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_runtime_plan(out: Path, best_row: pd.Series) -> None:
    lines = [
        "# Posture Reliability Runtime Plan",
        "",
        "Runtime integration was not implemented in this task. If this conservative reliability gate is later promoted, it should be added behind explicit CLI flags and default to shadow/log-only behavior.",
        "",
        "## Preconditions",
        "",
        "- Repeat validation after associated point-cloud logging is added.",
        "- Keep old posture output as the displayed posture unless a separate acceptance review approves replacement or gating behavior.",
        "- Log old posture, reliability label, final displayed posture, and reason in shadow mode first.",
        "",
        "## Current Offline Result",
        "",
        f"- Best reliability model: {best_row['model_name']}",
        f"- Trust coverage: {float(best_row['trust_coverage']) * 100.0:.2f}%",
        f"- Trusted accuracy: {float(best_row['trusted_accuracy']) * 100.0:.2f}%",
        f"- Trusted false SITTING on standing_3m: {float(best_row['trusted_false_sitting_on_standing_3m']) * 100.0:.2f}%",
    ]
    (out / "POSTURE_RELIABILITY_RUNTIME_PLAN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out = ensure_dir(Path(args.out))
    windows_path = Path(args.dataset_root) / "posturenet_lite_windows.csv"
    if not windows_path.exists():
        raise FileNotFoundError(f"Missing lite dataset: {windows_path}")

    df = pd.read_csv(windows_path)
    df = add_base_columns(df)
    df["trust_target"] = df.apply(trust_label, axis=1)

    model_names = ["heuristic_conservative"]
    if LogisticRegression is not None and RandomForestClassifier is not None:
        model_names.extend(["LogisticRegression", "RandomForestClassifier"])

    comparisons: list[dict[str, object]] = []
    all_case_rows: list[dict[str, object]] = []
    best_pred = None
    for model_name in model_names:
        pred_df = grouped_predictions(df, model_name)
        result = evaluate(pred_df, model_name)
        result["acceptance_passed"] = acceptance(result)
        comparisons.append(result)
        all_case_rows.extend(trusted_case_rows(pred_df, model_name))
        pred_df["model_name"] = model_name
        if best_pred is None:
            best_pred = pred_df
        else:
            prev = evaluate(best_pred, str(best_pred["model_name"].iloc[0]))
            prev_score = (as_float(prev["trusted_accuracy"]), as_float(prev["trust_coverage"]))
            new_score = (as_float(result["trusted_accuracy"]), as_float(result["trust_coverage"]))
            if new_score > prev_score:
                best_pred = pred_df

    comparison_df = pd.DataFrame(comparisons)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values(
            ["acceptance_passed", "trusted_accuracy", "trust_coverage"], ascending=[False, False, False]
        )
    comparison_df.to_csv(out / "reliability_model_comparison.csv", index=False)
    pd.DataFrame(all_case_rows).to_csv(out / "trusted_accuracy_by_case.csv", index=False)

    best_row = comparison_df.iloc[0] if not comparison_df.empty else None
    if best_pred is not None and confusion_matrix is not None:
        labels = ["TRUST_CORRECT", "DO_NOT_TRUST", "LOW_VISIBILITY", "UNCERTAIN"]
        cm = confusion_matrix(best_pred["trust_target"], best_pred["predicted_trust_label"], labels=labels)
        cm_df = pd.DataFrame(cm, index=[f"actual_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
        cm_df.to_csv(out / "reliability_confusion_matrix.csv")
    else:
        pd.DataFrame().to_csv(out / "reliability_confusion_matrix.csv", index=False)

    write_report(out, comparison_df, best_row)
    if best_row is not None and bool(best_row["acceptance_passed"]):
        write_runtime_plan(out, best_row)
    print(f"Reliability model analysis written to {out}")
    if best_row is not None:
        print(f"Best reliability model: {best_row['model_name']}")
        print(f"Reliability acceptance passed: {'yes' if bool(best_row['acceptance_passed']) else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
