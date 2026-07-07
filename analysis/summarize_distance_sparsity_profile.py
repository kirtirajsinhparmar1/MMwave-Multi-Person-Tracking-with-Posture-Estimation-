#!/usr/bin/env python
"""Summarize distance-conditioned sparsity and posture failure profiles.

This script uses the cleaned posture outputs when they already exist. It does
not rebuild the registry, redo segment assignment, retrain a model, or touch
runtime posture logic.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from clean_and_label_posture_sessions import load_session_tables, rows_in_segment
from posturenet_v2_common import discover_session_path, ensure_dir, normalized_label, to_float, to_int


RATE_COLUMNS = [
    "tracking_presence_rate",
    "pose_presence_rate",
    "ui_visible_rate",
    "disappearance_rate",
    "NO_POINTS_rate",
    "LOW_POINTS_rate",
    "OK_rate",
]

METRIC_COLUMNS = [
    "tracking_presence_rate",
    "pose_presence_rate",
    "ui_visible_rate",
    "disappearance_rate",
    "NO_POINTS_rate",
    "LOW_POINTS_rate",
    "OK_rate",
    "mean_geom_pts",
    "median_geom_pts",
    "point_count_if_available",
    "mean_snr_if_available",
    "range_error_mean",
    "range_jitter",
    "pose_accuracy",
    "standing_accuracy",
    "sitting_accuracy",
    "false_sitting_on_standing",
    "false_standing_on_sitting",
    "pose_switch_count",
]

LABEL_COLUMNS = [
    "final_display_pose",
    "displayed_label",
    "final_label",
    "smoothed_label",
    "ml_top_label",
    "raw_label",
    "old_display_pose",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--cleaned-root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        match = lower.get(candidate.lower())
        if match is not None:
            return match
    return None


def numeric_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    column = find_column(df, candidates)
    if column is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def clamp01(value: float) -> float:
    if math.isnan(value):
        return math.nan
    return max(0.0, min(1.0, value))


def safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0 or math.isnan(denominator):
        return math.nan
    return numerator / denominator


def rate_from_row(row: pd.Series, name: str, default: float = math.nan) -> float:
    value = to_float(row.get(name), default)
    return clamp01(value)


def distance_band(distance_m: float) -> str:
    if math.isnan(distance_m):
        return "UNKNOWN"
    if distance_m <= 3.0:
        return "NEAR"
    if distance_m <= 5.0:
        return "FAR"
    return "EDGE"


def frame_ids(df: pd.DataFrame) -> set[int]:
    column = find_column(df, ["mmwave_frame_num", "frame"])
    if column is None or df.empty:
        return set()
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return {int(v) for v in values}


def unique_frame_count(df: pd.DataFrame) -> int:
    ids = frame_ids(df)
    return len(ids) if ids else int(len(df))


def filter_tid(df: pd.DataFrame, tid: int | None) -> pd.DataFrame:
    if df.empty or tid is None or "tid" not in df.columns:
        return df.copy()
    values = pd.to_numeric(df["tid"], errors="coerce")
    return df[values == tid].copy()


def label_series(pose: pd.DataFrame) -> pd.Series:
    column = find_column(pose, LABEL_COLUMNS)
    if column is None:
        return pd.Series(dtype=object)
    return pose[column].map(normalized_label)


def quality_rates_from_pose(pose: pd.DataFrame) -> tuple[float, float, float]:
    if pose.empty:
        return math.nan, math.nan, math.nan
    quality_col = find_column(pose, ["quality", "quality_flag", "geom_quality", "quality_label_for_tid"])
    denominator = float(len(pose))
    no_points = low_points = ok = math.nan
    if quality_col is not None:
        values = pose[quality_col].astype(str).str.upper()
        no_points = safe_div(float(values.str.contains("NO_POINTS", regex=False).sum()), denominator)
        low_points = safe_div(float(values.str.contains("LOW_POINTS|LOW QUALITY|LOW_QUALITY", regex=True).sum()), denominator)
        ok = safe_div(float(values.str.contains("OK|POINT_GEOMETRY|GOOD", regex=True).sum()), denominator)
    counts = numeric_series(pose, ["geom_pts", "num_points", "selected_num_points"])
    if counts.notna().any():
        point_low = safe_div(float((counts < 5).sum()), float(counts.notna().sum()))
        low_points = max(low_points if not math.isnan(low_points) else 0.0, point_low)
    return no_points, low_points, ok


def switch_count(labels: pd.Series) -> int:
    visible = labels[~labels.isin(["UNKNOWN"])].reset_index(drop=True)
    if len(visible) < 2:
        return 0
    return int((visible != visible.shift()).sum() - 1)


def mean_median(series: pd.Series) -> tuple[float, float, int]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return math.nan, math.nan, 0
    return float(clean.mean()), float(clean.median()), int(len(clean))


def resolve_session_path(registry_row: pd.Series) -> Path | None:
    raw_path = str(registry_row.get("session_path") or "").strip()
    if raw_path:
        path = Path(raw_path)
        if path.exists():
            return path
    return discover_session_path(str(registry_row.get("session_id") or ""))


def load_associated_points(session_path: Path | None, cache: dict[Path, pd.DataFrame]) -> pd.DataFrame:
    if session_path is None:
        return pd.DataFrame()
    path = session_path / "mmwave_associated_points.csv"
    if not path.exists():
        return pd.DataFrame()
    if path not in cache:
        try:
            cache[path] = pd.read_csv(path, low_memory=False)
        except Exception:
            cache[path] = pd.DataFrame()
    return cache[path]


def associated_point_metrics(
    points: pd.DataFrame,
    tid: int | None,
    segment_frames: set[int],
) -> dict[str, float]:
    if points.empty or not segment_frames:
        return {
            "point_count_if_available": math.nan,
            "mean_snr_if_available": math.nan,
            "point_frame_count": 0,
            "snr_count": 0,
        }

    frame_col = find_column(points, ["frame", "mmwave_frame_num"])
    if frame_col is None:
        return {
            "point_count_if_available": math.nan,
            "mean_snr_if_available": math.nan,
            "point_frame_count": 0,
            "snr_count": 0,
        }

    work = points[pd.to_numeric(points[frame_col], errors="coerce").isin(segment_frames)].copy()
    if tid is not None and "tid" in work.columns:
        work = work[pd.to_numeric(work["tid"], errors="coerce") == tid].copy()
    if work.empty:
        return {
            "point_count_if_available": 0.0,
            "mean_snr_if_available": math.nan,
            "point_frame_count": 0,
            "snr_count": 0,
        }

    valid = work
    valid_col = find_column(work, ["is_valid_point"])
    if valid_col is not None:
        valid = valid[pd.to_numeric(valid[valid_col], errors="coerce").fillna(0) != 0]
    quality_col = find_column(valid, ["point_quality", "quality_label_for_tid"])
    if quality_col is not None:
        valid = valid[~valid[quality_col].astype(str).str.upper().eq("NO_POINTS")]

    if valid.empty:
        counts = pd.Series(dtype=float)
    else:
        counts = valid.groupby(frame_col).size().astype(float)

    geom_col = find_column(work, ["geom_pts_for_tid"])
    if geom_col is not None:
        geom_counts = pd.to_numeric(work[geom_col], errors="coerce").dropna()
        if not geom_counts.empty:
            frame_geom = work.assign(_geom=pd.to_numeric(work[geom_col], errors="coerce")).groupby(frame_col)["_geom"].max()
            counts = frame_geom if counts.empty else counts.combine(frame_geom, max, fill_value=0)

    snr_col = find_column(valid, ["point_snr", "snr"])
    snr = pd.to_numeric(valid[snr_col], errors="coerce").dropna() if snr_col is not None and not valid.empty else pd.Series(dtype=float)
    return {
        "point_count_if_available": float(counts.mean()) if not counts.empty else 0.0,
        "mean_snr_if_available": float(snr.mean()) if not snr.empty else math.nan,
        "point_frame_count": int(len(counts)),
        "snr_count": int(len(snr)),
    }


def segment_metrics(
    row: pd.Series,
    registry_row: pd.Series,
    session_path: Path | None,
    tables: dict[str, pd.DataFrame],
    associated_cache: dict[Path, pd.DataFrame],
) -> dict[str, Any]:
    start = to_float(row.get("start_time_s"), math.nan)
    end = to_float(row.get("end_time_s"), math.nan)
    tid = to_int(row.get("assigned_tid"))
    expected_pose = normalized_label(row.get("expected_pose"))
    distance = to_float(row.get("expected_distance_m"), math.nan)
    expected_frames_value = to_float(row.get("expected_frames"), math.nan)
    expected_frames = max(1, int(expected_frames_value) if not math.isnan(expected_frames_value) else 1)

    pose_all = rows_in_segment(tables.get("pose", pd.DataFrame()), start, end)
    tracks_all = rows_in_segment(tables.get("tracks", pd.DataFrame()), start, end)
    pose = filter_tid(pose_all, tid)
    tracks = filter_tid(tracks_all, tid)
    if expected_frames <= 1:
        expected_frames = max(unique_frame_count(pose_all), unique_frame_count(tracks_all), 1)

    pose_frames = unique_frame_count(pose)
    track_frames = unique_frame_count(tracks)
    tracking_presence = rate_from_row(row, "tracking_presence_rate", safe_div(track_frames, expected_frames))
    pose_presence = rate_from_row(row, "pose_presence_rate", safe_div(pose_frames, expected_frames))
    if math.isnan(tracking_presence):
        tracking_presence = 0.0
    if math.isnan(pose_presence):
        pose_presence = 0.0

    no_points = rate_from_row(row, "NO_POINTS_rate")
    low_points = rate_from_row(row, "LOW_POINTS_rate")
    ok_rate = rate_from_row(row, "OK_rate")
    if any(math.isnan(v) for v in [no_points, low_points, ok_rate]):
        q_no, q_low, q_ok = quality_rates_from_pose(pose)
        no_points = q_no if math.isnan(no_points) else no_points
        low_points = q_low if math.isnan(low_points) else low_points
        ok_rate = q_ok if math.isnan(ok_rate) else ok_rate

    geom = numeric_series(pose, ["geom_pts", "num_points", "selected_num_points"])
    if not geom.notna().any():
        geom = numeric_series(tracks, ["num_associated_points", "geom_pts_for_tid"])
    mean_geom, median_geom, geom_count = mean_median(geom)

    range_series = numeric_series(tracks, ["target_range_m", "range_m"])
    if not range_series.notna().any():
        range_series = numeric_series(pose, ["target_range_m", "range_m"])
    range_clean = range_series.dropna()
    range_clean = range_clean[(range_clean >= 0.0) & (range_clean <= 20.0)]
    if not range_clean.empty and not math.isnan(distance):
        range_error_mean = float((range_clean - distance).abs().mean())
        range_jitter = float(range_clean.std(ddof=0))
        range_count = int(len(range_clean))
    else:
        range_error_mean = math.nan
        range_jitter = math.nan
        range_count = 0

    labels = label_series(pose)
    label_count = int(len(labels))
    visible = labels[~labels.isin(["UNKNOWN"])]
    visible_count = int(len(visible))
    correct_count = int((labels == expected_pose).sum()) if label_count else 0
    standing_denom = label_count if expected_pose == "STANDING" else 0
    sitting_denom = label_count if expected_pose == "SITTING" else 0
    standing_correct = int((labels == "STANDING").sum()) if standing_denom else 0
    sitting_correct = int((labels == "SITTING").sum()) if sitting_denom else 0
    false_sitting = int((labels == "SITTING").sum()) if standing_denom else 0
    false_standing = int((labels == "STANDING").sum()) if sitting_denom else 0
    ui_visible_rate = safe_div(float(visible_count), float(expected_frames))
    row_ui = rate_from_row(row, "ui_visible_rate")
    if not math.isnan(row_ui) and visible_count == 0:
        ui_visible_rate = row_ui

    segment_frames = frame_ids(pose) | frame_ids(tracks)
    point_stats = associated_point_metrics(load_associated_points(session_path, associated_cache), tid, segment_frames)

    return {
        "session_id": row.get("session_id", ""),
        "segment_id": row.get("segment_id", ""),
        "person_slot": row.get("person_slot", ""),
        "expected_pose": expected_pose,
        "expected_subpose": str(row.get("expected_subpose") or "").upper(),
        "expected_distance_m": distance,
        "distance_band": distance_band(distance),
        "expected_position": str(row.get("expected_position") or registry_row.get("positions") or "").upper(),
        "people_count": to_int(registry_row.get("people_count"), to_int(row.get("people_count"), 1)) or 1,
        "assigned_tid": tid if tid is not None else "",
        "duration_s": max(0.0, end - start) if not math.isnan(start) and not math.isnan(end) else math.nan,
        "expected_frames": expected_frames,
        "pose_frame_count": pose_frames,
        "track_frame_count": track_frames,
        "tracking_presence_rate": tracking_presence,
        "pose_presence_rate": pose_presence,
        "ui_visible_rate": clamp01(ui_visible_rate),
        "disappearance_rate": clamp01(1.0 - min(tracking_presence, pose_presence)),
        "NO_POINTS_rate": no_points,
        "LOW_POINTS_rate": low_points,
        "OK_rate": ok_rate,
        "mean_geom_pts": mean_geom,
        "median_geom_pts": median_geom,
        "geom_count": geom_count,
        "point_count_if_available": point_stats["point_count_if_available"],
        "mean_snr_if_available": point_stats["mean_snr_if_available"],
        "point_frame_count": point_stats["point_frame_count"],
        "snr_count": point_stats["snr_count"],
        "range_error_mean": range_error_mean,
        "range_jitter": range_jitter,
        "range_count": range_count,
        "pose_accuracy": safe_div(float(correct_count), float(label_count)),
        "standing_accuracy": safe_div(float(standing_correct), float(standing_denom)),
        "sitting_accuracy": safe_div(float(sitting_correct), float(sitting_denom)),
        "false_sitting_on_standing": safe_div(float(false_sitting), float(standing_denom)),
        "false_standing_on_sitting": safe_div(float(false_standing), float(sitting_denom)),
        "pose_switch_count": switch_count(labels),
        "pose_label_count": label_count,
        "pose_correct_count": correct_count,
        "standing_label_count": standing_denom,
        "standing_correct_count": standing_correct,
        "sitting_label_count": sitting_denom,
        "sitting_correct_count": sitting_correct,
        "false_sitting_on_standing_count": false_sitting,
        "false_standing_on_sitting_count": false_standing,
    }


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    data = pd.DataFrame({"value": pd.to_numeric(values, errors="coerce"), "weight": pd.to_numeric(weights, errors="coerce")})
    data = data.dropna(subset=["value"])
    if data.empty:
        return math.nan
    data["weight"] = data["weight"].fillna(0.0).clip(lower=0.0)
    if float(data["weight"].sum()) <= 0.0:
        return float(data["value"].mean())
    return float(np.average(data["value"], weights=data["weight"]))


def aggregate_segments(segments: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if segments.empty:
        return pd.DataFrame(columns=group_cols + ["segment_count", "session_count", *METRIC_COLUMNS])

    grouped = segments.groupby(group_cols, dropna=False, sort=True)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        out = {col: value for col, value in zip(group_cols, key)}
        out["segment_count"] = int(len(group))
        out["session_count"] = int(group["session_id"].nunique()) if "session_id" in group.columns else 0

        expected_weights = group["expected_frames"].fillna(1.0)
        pose_weights = group["pose_label_count"].replace(0, np.nan).fillna(group["pose_frame_count"]).fillna(1.0)
        geom_weights = group["geom_count"].replace(0, np.nan).fillna(group["pose_frame_count"]).fillna(1.0)
        point_weights = group["point_frame_count"].replace(0, np.nan)
        snr_weights = group["snr_count"].replace(0, np.nan)
        range_weights = group["range_count"].replace(0, np.nan)

        for column in RATE_COLUMNS:
            out[column] = weighted_mean(group[column], expected_weights)
        out["mean_geom_pts"] = weighted_mean(group["mean_geom_pts"], geom_weights)
        out["median_geom_pts"] = float(pd.to_numeric(group["median_geom_pts"], errors="coerce").median()) if group["median_geom_pts"].notna().any() else math.nan
        out["point_count_if_available"] = weighted_mean(group["point_count_if_available"], point_weights)
        out["mean_snr_if_available"] = weighted_mean(group["mean_snr_if_available"], snr_weights)
        out["range_error_mean"] = weighted_mean(group["range_error_mean"], range_weights)
        out["range_jitter"] = weighted_mean(group["range_jitter"], range_weights)
        out["pose_accuracy"] = safe_div(float(group["pose_correct_count"].sum()), float(group["pose_label_count"].sum()))
        out["standing_accuracy"] = safe_div(float(group["standing_correct_count"].sum()), float(group["standing_label_count"].sum()))
        out["sitting_accuracy"] = safe_div(float(group["sitting_correct_count"].sum()), float(group["sitting_label_count"].sum()))
        out["false_sitting_on_standing"] = safe_div(
            float(group["false_sitting_on_standing_count"].sum()),
            float(group["standing_label_count"].sum()),
        )
        out["false_standing_on_sitting"] = safe_div(
            float(group["false_standing_on_sitting_count"].sum()),
            float(group["sitting_label_count"].sum()),
        )
        out["pose_switch_count"] = int(group["pose_switch_count"].sum())
        rows.append(out)

    return pd.DataFrame(rows)


def round_output(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_float_dtype(out[column]):
            out[column] = out[column].round(4)
    return out


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    clean = df.fillna("")
    headers = [str(col) for col in clean.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in clean.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in clean.columns) + " |")
    return "\n".join(lines)


def metric_at_distance(by_distance: pd.DataFrame, distance: float, metric: str) -> float:
    rows = by_distance[pd.to_numeric(by_distance["expected_distance_m"], errors="coerce") == distance]
    if rows.empty or metric not in rows.columns:
        return math.nan
    return float(pd.to_numeric(rows.iloc[0][metric], errors="coerce"))


def grouped_weighted_metrics(df: pd.DataFrame, group_col: str, metric_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in df.groupby(group_col, dropna=False):
        row: dict[str, Any] = {group_col: key}
        weights = group["segment_count"] if "segment_count" in group.columns else pd.Series([1.0] * len(group))
        for metric in metric_cols:
            row[metric] = weighted_mean(group[metric], weights)
        rows.append(row)
    return pd.DataFrame(rows)


def first_problem_distance(by_distance: pd.DataFrame) -> str:
    if by_distance.empty:
        return "Unknown; no distance aggregates were available."
    near_rows = by_distance[pd.to_numeric(by_distance["expected_distance_m"], errors="coerce") <= 3.0]
    near_acc = float(pd.to_numeric(near_rows["pose_accuracy"], errors="coerce").mean()) if not near_rows.empty else math.nan
    candidates: list[str] = []
    for _, row in by_distance.sort_values("expected_distance_m").iterrows():
        distance = to_float(row.get("expected_distance_m"), math.nan)
        if math.isnan(distance) or distance <= 3.0:
            continue
        acc = to_float(row.get("pose_accuracy"), math.nan)
        ok_rate = to_float(row.get("OK_rate"), math.nan)
        geom = to_float(row.get("mean_geom_pts"), math.nan)
        no_points = to_float(row.get("NO_POINTS_rate"), math.nan)
        if (
            (not math.isnan(acc) and not math.isnan(near_acc) and acc <= near_acc - 0.10)
            or (not math.isnan(ok_rate) and ok_rate < 0.20)
            or (not math.isnan(geom) and geom < 5.0)
            or (not math.isnan(no_points) and no_points >= 0.50)
        ):
            candidates.append(f"{distance:g}m")
            break
    return candidates[0] if candidates else "No clear severe break beyond 3m in the available cleaned metrics."


def answer_questions(by_distance: pd.DataFrame, by_band: pd.DataFrame, by_subpose: pd.DataFrame) -> list[str]:
    severe = first_problem_distance(by_distance)
    acc_4 = metric_at_distance(by_distance, 4.0, "pose_accuracy")
    acc_5 = metric_at_distance(by_distance, 5.0, "pose_accuracy")
    acc_6 = metric_at_distance(by_distance, 6.0, "pose_accuracy")

    standing = by_subpose[by_subpose["expected_pose"] == "STANDING"] if not by_subpose.empty else pd.DataFrame()
    sitting = by_subpose[by_subpose["expected_pose"] == "SITTING"] if not by_subpose.empty else pd.DataFrame()
    standing_near = weighted_mean(standing[standing["distance_band"] == "NEAR"]["standing_accuracy"], standing[standing["distance_band"] == "NEAR"]["segment_count"]) if not standing.empty else math.nan
    standing_far = weighted_mean(standing[standing["distance_band"] == "FAR"]["standing_accuracy"], standing[standing["distance_band"] == "FAR"]["segment_count"]) if not standing.empty else math.nan
    sitting_near = weighted_mean(sitting[sitting["distance_band"] == "NEAR"]["sitting_accuracy"], sitting[sitting["distance_band"] == "NEAR"]["segment_count"]) if not sitting.empty else math.nan
    sitting_far = weighted_mean(sitting[sitting["distance_band"] == "FAR"]["sitting_accuracy"], sitting[sitting["distance_band"] == "FAR"]["segment_count"]) if not sitting.empty else math.nan

    subtype_lines: list[str] = []
    if not sitting.empty:
        far_sitting = sitting[sitting["distance_band"] == "FAR"].copy()
        if not far_sitting.empty:
            subtype_summary = grouped_weighted_metrics(
                far_sitting,
                "expected_subpose",
                ["sitting_accuracy", "disappearance_rate"],
            )
            subtype_summary = subtype_summary.sort_values(["sitting_accuracy", "disappearance_rate"], ascending=[True, False])
            first = subtype_summary.iloc[0]
            subtype_lines.append(
                f"The most sensitive FAR sitting subtype is `{first['expected_subpose']}` "
                f"(sitting_accuracy={first['sitting_accuracy']:.3f}, disappearance_rate={first['disappearance_rate']:.3f})."
            )

    position_lines: list[str] = []
    if not by_subpose.empty:
        far = by_subpose[by_subpose["distance_band"] == "FAR"]
        if not far.empty:
            pos = grouped_weighted_metrics(
                far,
                "expected_position",
                ["pose_accuracy", "disappearance_rate", "NO_POINTS_rate"],
            )
            pos = pos.sort_values(["pose_accuracy", "disappearance_rate"], ascending=[True, False])
            first = pos.iloc[0]
            position_lines.append(
                f"Worst FAR position in this pass is `{first['expected_position']}` "
                f"(pose_accuracy={first['pose_accuracy']:.3f}, disappearance_rate={first['disappearance_rate']:.3f}, "
                f"NO_POINTS_rate={first['NO_POINTS_rate']:.3f})."
            )

    people_lines: list[str] = []
    if not by_subpose.empty:
        ppl = grouped_weighted_metrics(
            by_subpose,
            "people_count",
            ["pose_accuracy", "disappearance_rate", "NO_POINTS_rate"],
        )
        if len(ppl) > 1:
            people_lines.append(
                "People-count comparison: "
                + "; ".join(
                    f"{int(r.people_count)} person(s): accuracy={r.pose_accuracy:.3f}, disappearance={r.disappearance_rate:.3f}, NO_POINTS={r.NO_POINTS_rate:.3f}"
                    for r in ppl.itertuples()
                )
                + "."
            )

    indicators = []
    if not by_subpose.empty:
        candidate_metrics = ["NO_POINTS_rate", "LOW_POINTS_rate", "OK_rate", "mean_geom_pts", "ui_visible_rate", "range_jitter"]
        correlations = []
        work = by_subpose[["pose_accuracy", *candidate_metrics]].copy()
        for metric in candidate_metrics:
            pair = work[["pose_accuracy", metric]].dropna()
            if len(pair) >= 3 and pair[metric].std(ddof=0) > 0:
                corr = float(pair["pose_accuracy"].corr(pair[metric]))
                correlations.append((abs(corr), corr, metric))
        correlations.sort(reverse=True)
        indicators = [f"`{metric}` (corr={corr:.3f})" for _, corr, metric in correlations[:3]]

    lines = [
        f"1. Severe sparsity/failure begins at: {severe}.",
        (
            "2. 4m/5m/6m check: "
            f"pose_accuracy_4m={acc_4:.3f}" if not math.isnan(acc_4) else "2. 4m/5m/6m check: 4m unavailable"
        )
        + (f", pose_accuracy_5m={acc_5:.3f}" if not math.isnan(acc_5) else ", 5m unavailable")
        + (f", pose_accuracy_6m={acc_6:.3f}." if not math.isnan(acc_6) else ", 6m unavailable."),
        (
            "3. Standing vs sitting degradation: "
            f"standing NEAR={standing_near:.3f}, standing FAR={standing_far:.3f}; "
            f"sitting NEAR={sitting_near:.3f}, sitting FAR={sitting_far:.3f}."
            if not all(math.isnan(v) for v in [standing_near, standing_far, sitting_near, sitting_far])
            else "3. Standing vs sitting degradation: unavailable in current aggregates."
        ),
        "4. " + (subtype_lines[0] if subtype_lines else "Sitting subtype sensitivity is unavailable in current aggregates."),
        "5. " + (position_lines[0] if position_lines else "Left/right position comparison is unavailable in current aggregates."),
        "6. " + (people_lines[0] if people_lines else "Two-person comparison is unavailable in current aggregates."),
        "7. Sparse indicators most associated with failure: " + (", ".join(indicators) if indicators else "unavailable; too few non-null aggregate rows."),
    ]
    return lines


def write_missing_report(out_dir: Path, missing: list[str]) -> None:
    ensure_dir(out_dir)
    empty = pd.DataFrame(columns=["missing_input", *METRIC_COLUMNS])
    empty.to_csv(out_dir / "sparsity_by_distance.csv", index=False)
    empty.to_csv(out_dir / "sparsity_by_band.csv", index=False)
    empty.to_csv(out_dir / "sparsity_by_subpose.csv", index=False)
    lines = [
        "# Sparse Distance Profile Report",
        "",
        "The sparse-distance profile could not be computed because required existing outputs are missing.",
        "",
        "Missing inputs:",
        *[f"- `{item}`" for item in missing],
        "",
        "Per instruction, this script did not rebuild the registry or cleaning outputs.",
    ]
    (out_dir / "sparsity_profile_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    out_dir: Path,
    registry_path: Path,
    cleaned_root: Path,
    segment_count: int,
    by_distance: pd.DataFrame,
    by_band: pd.DataFrame,
    by_subpose: pd.DataFrame,
    missing_optional: list[str],
) -> None:
    lines = [
        "# Sparse Distance Profile Report",
        "",
        "## Inputs",
        "",
        f"- Registry: `{registry_path}`",
        f"- Cleaned root: `{cleaned_root}`",
        f"- Segments analyzed: {segment_count}",
    ]
    if missing_optional:
        lines.extend(["", "Optional inputs not found:"])
        lines.extend(f"- `{item}`" for item in missing_optional)

    lines.extend(
        [
            "",
            "## Distance Band Definitions",
            "",
            "- NEAR: `<= 3m`",
            "- FAR: `> 3m and <= 5m`",
            "- EDGE: `> 5m`",
            "",
            "## Band Summary",
            "",
        ]
    )
    if by_band.empty:
        lines.append("No band summary rows were produced.")
    else:
        display_cols = ["distance_band", "segment_count", "pose_accuracy", "standing_accuracy", "sitting_accuracy", "NO_POINTS_rate", "LOW_POINTS_rate", "OK_rate", "mean_geom_pts", "range_error_mean", "range_jitter"]
        lines.append(markdown_table(round_output(by_band[[c for c in display_cols if c in by_band.columns]])))

    lines.extend(["", "## Required Questions", ""])
    lines.extend(answer_questions(by_distance, by_band, by_subpose))
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `range_error_mean` is the mean absolute error between measured target range and protocol distance.",
            "- `range_jitter` is the within-segment standard deviation of target range.",
            "- `point_count_if_available` and `mean_snr_if_available` are populated only for sessions that include `mmwave_associated_points.csv` aligned to cleaned segment frames.",
            "- The current registry mostly predates full associated-point logging, so sparse architecture design should treat these aggregates as a failure profile, not as the final full point-cloud training set.",
        ]
    )
    (out_dir / "sparsity_profile_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    registry_path = Path(args.registry)
    cleaned_root = Path(args.cleaned_root)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    missing: list[str] = []
    if not registry_path.exists():
        missing.append(str(registry_path))
    if not cleaned_root.exists():
        missing.append(str(cleaned_root))
    segment_quality_path = cleaned_root / "segment_quality.csv"
    if not segment_quality_path.exists():
        missing.append(str(segment_quality_path))
    if missing:
        write_missing_report(out_dir, missing)
        print("Sparse profile created: no")
        print("Missing inputs: " + "; ".join(missing))
        return 0

    registry = pd.read_csv(registry_path, low_memory=False)
    segment_quality = pd.read_csv(segment_quality_path, low_memory=False)
    registry_map = {str(row.get("session_id")): row for _, row in registry.iterrows()}
    associated_cache: dict[Path, pd.DataFrame] = {}
    table_cache: dict[str, dict[str, pd.DataFrame]] = {}
    path_cache: dict[str, Path | None] = {}

    rows: list[dict[str, Any]] = []
    for _, segment in segment_quality.iterrows():
        session_id = str(segment.get("session_id") or "")
        registry_row = registry_map.get(session_id, pd.Series({"session_id": session_id}))
        if session_id not in path_cache:
            path_cache[session_id] = resolve_session_path(registry_row)
        session_path = path_cache[session_id]
        if session_id not in table_cache:
            table_cache[session_id] = load_session_tables(session_path) if session_path is not None else {"pose": pd.DataFrame(), "tracks": pd.DataFrame(), "frames": pd.DataFrame()}
        rows.append(segment_metrics(segment, registry_row, session_path, table_cache[session_id], associated_cache))

    segments = pd.DataFrame(rows)
    by_distance = aggregate_segments(segments, ["expected_distance_m", "distance_band"])
    by_band = aggregate_segments(segments, ["distance_band"])
    by_subpose = aggregate_segments(
        segments,
        [
            "session_id",
            "expected_pose",
            "expected_subpose",
            "expected_distance_m",
            "distance_band",
            "expected_position",
            "people_count",
        ],
    )

    round_output(by_distance).to_csv(out_dir / "sparsity_by_distance.csv", index=False)
    round_output(by_band).to_csv(out_dir / "sparsity_by_band.csv", index=False)
    round_output(by_subpose).to_csv(out_dir / "sparsity_by_subpose.csv", index=False)

    optional_missing: list[str] = []
    if not (Path("analysis_outputs") / "range_sparse_posture_audit").exists():
        optional_missing.append("analysis_outputs/range_sparse_posture_audit")
    write_report(out_dir, registry_path, cleaned_root, len(segments), by_distance, by_band, by_subpose, optional_missing)
    print("Sparse profile created: yes")
    print(f"Segments analyzed: {len(segments)}")
    print(f"Output: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
