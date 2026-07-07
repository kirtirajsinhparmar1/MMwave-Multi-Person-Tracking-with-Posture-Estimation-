"""Validate mmwave_associated_points.csv for full posture-model data collection."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = [
    "session_id",
    "frame",
    "timestamp_s",
    "tid",
    "track_index",
    "point_index",
    "association_source",
    "association_confidence",
    "point_x_m",
    "point_y_m",
    "point_z_m",
    "point_range_m",
    "point_azimuth_deg",
    "point_elevation_deg",
    "point_doppler_mps",
    "point_snr",
    "point_noise",
    "point_quality",
    "target_x_m",
    "target_y_m",
    "target_z_m",
    "target_range_m",
    "target_azimuth_deg",
    "target_elevation_deg",
    "target_vx_mps",
    "target_vy_mps",
    "target_vz_mps",
    "relative_x_m",
    "relative_y_m",
    "relative_z_m",
    "relative_range_m",
    "relative_radial_m",
    "relative_lateral_m",
    "height_above_ground_m",
    "is_valid_point",
    "geom_pts_for_tid",
    "points_total_frame",
    "quality_label_for_tid",
    "old_display_pose",
    "old_model_stand_prob",
    "old_model_sit_prob",
    "old_model_move_prob",
    "old_model_lie_prob",
    "old_model_fall_prob",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a mmwave_associated_points.csv session log."
    )
    parser.add_argument("--session", required=True, help="Session folder containing the CSV.")
    parser.add_argument(
        "--out",
        default="analysis_outputs/associated_point_log_validation",
        help="Output directory for validation summary files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_dir = Path(args.session).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = session_dir / "mmwave_associated_points.csv"
    summary_path = out_dir / "associated_point_log_summary.csv"
    report_path = out_dir / "ASSOCIATED_POINT_LOG_VALIDATION_REPORT.md"

    if not log_path.exists():
        write_failure(summary_path, report_path, log_path, "mmwave_associated_points.csv does not exist")
        return 1

    stats = scan_log(log_path)
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in stats["columns"]]
    write_summary(summary_path, log_path, stats, missing_columns)
    write_report(report_path, log_path, stats, missing_columns)
    return 1 if missing_columns else 0


def scan_log(log_path: Path) -> dict:
    row_count = 0
    frames = set()
    tids = set()
    valid_by_frame: Counter[str] = Counter()
    rows_by_frame: Counter[str] = Counter()
    valid_by_tid: Counter[str] = Counter()
    rows_by_tid: Counter[str] = Counter()
    no_points = 0
    association_sources: Counter[str] = Counter()
    quality_labels: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    populated_old_probs: Counter[str] = Counter()
    relative_populated = 0

    with log_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        for row in reader:
            row_count += 1
            frame = row.get("frame", "")
            tid = row.get("tid", "")
            if frame:
                frames.add(frame)
                rows_by_frame[frame] += 1
            if tid:
                tids.add(tid)
                rows_by_tid[tid] += 1
            if as_int(row.get("is_valid_point")) == 1:
                if frame:
                    valid_by_frame[frame] += 1
                if tid:
                    valid_by_tid[tid] += 1
            if as_int(row.get("is_valid_point")) == 0 or row.get("quality_label_for_tid") == "NO_POINTS":
                no_points += 1
            association_sources[row.get("association_source", "") or "<blank>"] += 1
            quality_labels[row.get("quality_label_for_tid", "") or "<blank>"] += 1
            if all(populated(row.get(name, "")) for name in ("relative_x_m", "relative_y_m", "relative_z_m")):
                relative_populated += 1
            for column in columns:
                if not populated(row.get(column, "")):
                    missing_counts[column] += 1
            for column in (
                "old_model_stand_prob",
                "old_model_sit_prob",
                "old_model_move_prob",
                "old_model_lie_prob",
                "old_model_fall_prob",
            ):
                if populated(row.get(column, "")):
                    populated_old_probs[column] += 1

    return {
        "columns": columns,
        "row_count": row_count,
        "frames": frames,
        "tids": tids,
        "valid_by_frame": valid_by_frame,
        "rows_by_frame": rows_by_frame,
        "valid_by_tid": valid_by_tid,
        "rows_by_tid": rows_by_tid,
        "no_points": no_points,
        "association_sources": association_sources,
        "quality_labels": quality_labels,
        "missing_counts": missing_counts,
        "relative_populated": relative_populated,
        "populated_old_probs": populated_old_probs,
        "file_size_bytes": log_path.stat().st_size,
    }


def write_failure(summary_path: Path, report_path: Path, log_path: Path, message: str) -> None:
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["log_path", str(log_path)])
        writer.writerow(["status", "FAIL"])
        writer.writerow(["error", message])
    report_path.write_text(
        "# Associated Point Log Validation Report\n\n"
        f"Status: FAIL\n\nLog path: `{log_path}`\n\nError: {message}\n",
        encoding="utf-8",
    )


def write_summary(
    summary_path: Path,
    log_path: Path,
    stats: dict,
    missing_columns: list[str],
) -> None:
    rows = [
        ("log_path", str(log_path)),
        ("status", "FAIL" if missing_columns else "PASS"),
        ("missing_required_columns", ";".join(missing_columns)),
        ("row_count", stats["row_count"]),
        ("file_size_bytes", stats["file_size_bytes"]),
        ("frames_covered", len(stats["frames"])),
        ("tids_covered", len(stats["tids"])),
        ("no_points_rows", stats["no_points"]),
        ("relative_coordinate_rows_populated", stats["relative_populated"]),
        ("points_per_frame_valid_min", describe(stats["valid_by_frame"].values())["min"]),
        ("points_per_frame_valid_p50", describe(stats["valid_by_frame"].values())["p50"]),
        ("points_per_frame_valid_p95", describe(stats["valid_by_frame"].values())["p95"]),
        ("points_per_frame_valid_max", describe(stats["valid_by_frame"].values())["max"]),
        ("points_per_tid_valid_min", describe(stats["valid_by_tid"].values())["min"]),
        ("points_per_tid_valid_p50", describe(stats["valid_by_tid"].values())["p50"]),
        ("points_per_tid_valid_p95", describe(stats["valid_by_tid"].values())["p95"]),
        ("points_per_tid_valid_max", describe(stats["valid_by_tid"].values())["max"]),
        ("association_source_distribution", counter_text(stats["association_sources"])),
        ("quality_label_distribution", counter_text(stats["quality_labels"])),
    ]
    for column in REQUIRED_COLUMNS:
        missing_rate = 0.0
        if stats["row_count"]:
            missing_rate = stats["missing_counts"].get(column, 0) / stats["row_count"]
        rows.append((f"missing_rate:{column}", f"{missing_rate:.6f}"))
    for column, count in sorted(stats["populated_old_probs"].items()):
        rows.append((f"populated_rows:{column}", count))

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def write_report(
    report_path: Path,
    log_path: Path,
    stats: dict,
    missing_columns: list[str],
) -> None:
    per_frame = describe(stats["valid_by_frame"].values())
    per_tid = describe(stats["valid_by_tid"].values())
    lines = [
        "# Associated Point Log Validation Report",
        "",
        f"Status: {'FAIL' if missing_columns else 'PASS'}",
        "",
        f"Log path: `{log_path}`",
        f"File size bytes: {stats['file_size_bytes']}",
        f"Row count: {stats['row_count']}",
        f"Frames covered: {len(stats['frames'])}",
        f"TIDs covered: {len(stats['tids'])}",
        f"Missing required columns: {', '.join(missing_columns) if missing_columns else 'none'}",
        "",
        "## Points Per Frame",
        "",
        f"Valid associated point rows per frame: min={per_frame['min']}, p50={per_frame['p50']}, p95={per_frame['p95']}, max={per_frame['max']}",
        "",
        "## Points Per TID",
        "",
        f"Valid associated point rows per TID: min={per_tid['min']}, p50={per_tid['p50']}, p95={per_tid['p95']}, max={per_tid['max']}",
        "",
        "## NO_POINTS Summary",
        "",
        f"NO_POINTS or invalid summary rows: {stats['no_points']}",
        "",
        "## Relative Coordinates",
        "",
        f"Rows with relative_x_m, relative_y_m, and relative_z_m populated: {stats['relative_populated']}",
        "",
        "## Old Model Probability Columns",
        "",
    ]
    for column in (
        "old_model_stand_prob",
        "old_model_sit_prob",
        "old_model_move_prob",
        "old_model_lie_prob",
        "old_model_fall_prob",
    ):
        lines.append(f"- {column}: {stats['populated_old_probs'].get(column, 0)} populated rows")
    lines.extend(
        [
            "",
            "## Association Source Distribution",
            "",
            counter_markdown(stats["association_sources"]),
            "",
            "## Missing Value Rates",
            "",
        ]
    )
    for column in REQUIRED_COLUMNS:
        count = stats["missing_counts"].get(column, 0)
        rate = count / stats["row_count"] if stats["row_count"] else 0.0
        lines.append(f"- {column}: {count} missing ({rate:.2%})")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def describe(values: Iterable[int]) -> dict[str, int]:
    data = sorted(int(value) for value in values)
    if not data:
        return {"min": 0, "p50": 0, "p95": 0, "max": 0}
    return {
        "min": data[0],
        "p50": percentile(data, 0.50),
        "p95": percentile(data, 0.95),
        "max": data[-1],
    }


def percentile(data: list[int], fraction: float) -> int:
    if not data:
        return 0
    index = min(len(data) - 1, max(0, int(math.ceil(fraction * len(data))) - 1))
    return data[index]


def counter_text(counter: Counter[str]) -> str:
    return ";".join(f"{key}:{value}" for key, value in sorted(counter.items()))


def counter_markdown(counter: Counter[str]) -> str:
    if not counter:
        return "No rows."
    return "\n".join(f"- {key}: {value}" for key, value in sorted(counter.items()))


def populated(value: str | None) -> bool:
    return value not in (None, "", "NaN", "nan")


def as_int(value: str | None) -> int | None:
    try:
        return int(float(value or ""))
    except ValueError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
