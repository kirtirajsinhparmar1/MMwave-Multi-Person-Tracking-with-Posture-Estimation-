from __future__ import annotations

import argparse
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd


SITTING_RELATIVE_MIN_PROB = 0.50
SITTING_RELATIVE_MARGIN = 0.12
SITTING_RELATIVE_FRAMES = 8
SITTING_RELATIVE_RANGE_MIN_M = 0.0
STANDING_VETO_PROB = 1.01
STANDING_VETO_MARGIN = 999.0
MOVING_TRANSLATION_WINDOW = 8
MOVING_TRANSLATION_MIN_M = 0.25
MOVING_SPEED_THRESHOLD = 0.18


def _norm_label(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text == "WALKING":
        return "MOVING"
    return text or "UNKNOWN"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_pose(session: Path) -> pd.DataFrame:
    path = session / "mmwave_pose.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing pose log: {path}")
    df = pd.read_csv(path)
    if "mmwave_frame_num" in df.columns:
        df["frame"] = pd.to_numeric(df["mmwave_frame_num"], errors="coerce")
    elif "frame" not in df.columns:
        raise ValueError(f"{path} has no frame column")
    if "tid" not in df.columns:
        raise ValueError(f"{path} has no tid column")
    df["tid"] = pd.to_numeric(df["tid"], errors="coerce").fillna(-1).astype(int)
    if "host_monotonic_ns" in df.columns:
        ns = pd.to_numeric(df["host_monotonic_ns"], errors="coerce")
        df["time_s"] = (ns - ns.min()) / 1_000_000_000.0
    else:
        df["time_s"] = pd.RangeIndex(len(df)).astype(float)

    rename = {
        "prob_standing": "stand_prob",
        "prob_sitting": "sit_prob",
        "prob_lying": "lying_prob",
        "prob_falling": "falling_prob",
        "speed_mps": "horizontal_speed",
        "quality_flag": "quality",
    }
    for old, new in rename.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    if "final_label" in df.columns:
        df["old_display_pose"] = df["final_label"].map(_norm_label)
    elif "displayed_label" in df.columns:
        df["old_display_pose"] = df["displayed_label"].map(_norm_label)
    else:
        df["old_display_pose"] = "UNKNOWN"
    for col in ["stand_prob", "sit_prob", "lying_prob", "falling_prob", "horizontal_speed"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "quality" not in df.columns:
        df["quality"] = ""
    return df


def _load_tracks(session: Path, pose: pd.DataFrame) -> pd.DataFrame:
    path = session / "mmwave_tracks.csv"
    if not path.exists():
        return pose
    tracks = pd.read_csv(path)
    if "mmwave_frame_num" in tracks.columns:
        tracks["frame"] = pd.to_numeric(tracks["mmwave_frame_num"], errors="coerce")
    if "tid" not in tracks.columns or "frame" not in tracks.columns:
        return pose
    tracks["tid"] = pd.to_numeric(tracks["tid"], errors="coerce").fillna(-1).astype(int)
    keep = ["frame", "tid"]
    for col in ["x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps", "num_associated_points"]:
        if col in tracks.columns:
            tracks[col] = pd.to_numeric(tracks[col], errors="coerce")
            keep.append(col)
    tracks = tracks[keep].drop_duplicates(["frame", "tid"], keep="last")
    merged = pose.merge(tracks, on=["frame", "tid"], how="left")
    if {"x_m", "y_m", "z_m"}.issubset(merged.columns):
        merged["range_m"] = (
            merged["x_m"].fillna(0.0) ** 2
            + merged["y_m"].fillna(0.0) ** 2
            + merged["z_m"].fillna(0.0) ** 2
        ) ** 0.5
    else:
        merged["range_m"] = pd.NA
    return merged


def _fallback_segments(session: Path, pose: pd.DataFrame) -> pd.DataFrame:
    latest = Path("analysis_outputs/latest_distance_posture_analysis_v2/segments_auto.csv")
    candidates = Path("analysis_outputs/latest_distance_posture_analysis_v2/candidate_sessions.csv")
    if latest.exists() and candidates.exists():
        try:
            cand = pd.read_csv(candidates)
            if any(Path(str(p)).resolve() == session.resolve() for p in cand.get("path", [])):
                seg = pd.read_csv(latest)
                seg["segmentation_method"] = seg.get("method", "auto_range_plateau_trimmed")
                return seg
        except Exception:
            pass
    return pd.DataFrame(
        [
            {
                "segment_id": "session_all",
                "expected_pose": "UNKNOWN",
                "expected_distance_m": pd.NA,
                "start_time_s": float(pose["time_s"].min()),
                "end_time_s": float(pose["time_s"].max()),
                "segmentation_method": "session_all_no_segments",
                "confidence": 0.0,
                "notes": "No segment file was supplied or discovered; accuracy metrics are limited.",
            }
        ]
    )


def _load_segments(path: Path | None, session: Path, pose: pd.DataFrame) -> pd.DataFrame:
    if path is not None and path.exists():
        seg = pd.read_csv(path)
    else:
        seg = _fallback_segments(session, pose)
    for col in ["start_time_s", "end_time_s", "expected_distance_m"]:
        if col in seg.columns:
            seg[col] = pd.to_numeric(seg[col], errors="coerce")
    if "segmentation_method" not in seg.columns:
        seg["segmentation_method"] = seg.get("method", "manual")
    if "confidence" not in seg.columns:
        seg["confidence"] = 1.0
    return seg


def _translation_confirmed(history: deque[tuple[float, float, float]], speed: float) -> tuple[bool, float]:
    displacement = 0.0
    if len(history) >= 2:
        first = history[0]
        last = history[-1]
        displacement = math.sqrt(
            (last[0] - first[0]) ** 2
            + (last[1] - first[1]) ** 2
            + (last[2] - first[2]) ** 2
        )
    confirmed = bool(
        (len(history) >= MOVING_TRANSLATION_WINDOW and displacement >= MOVING_TRANSLATION_MIN_M)
        or speed >= MOVING_SPEED_THRESHOLD * 1.5
    )
    return confirmed, displacement


def replay_decisions(
    df: pd.DataFrame,
    *,
    range_min_m: float,
    min_prob: float,
    margin: float,
    frames: int,
    standing_veto_prob: float,
    standing_veto_margin: float,
    moving_guard: bool,
) -> pd.DataFrame:
    histories: dict[int, deque[bool]] = defaultdict(lambda: deque(maxlen=frames))
    positions: dict[int, deque[tuple[float, float, float]]] = defaultdict(
        lambda: deque(maxlen=MOVING_TRANSLATION_WINDOW)
    )
    rows: list[dict[str, Any]] = []
    ordered = df.sort_values(["time_s", "frame", "tid"]).copy()
    for row in ordered.to_dict("records"):
        tid = int(row.get("tid", -1))
        x = _safe_float(row.get("x_m"))
        y = _safe_float(row.get("y_m"))
        z = _safe_float(row.get("z_m"))
        positions[tid].append((x, y, z))
        speed = _safe_float(row.get("horizontal_speed"))
        translation_confirmed, displacement = _translation_confirmed(positions[tid], speed)

        stand_prob = _safe_float(row.get("stand_prob"))
        sit_prob = _safe_float(row.get("sit_prob"))
        falling_prob = _safe_float(row.get("falling_prob"))
        lying_prob = _safe_float(row.get("lying_prob"))
        sit_minus_stand = sit_prob - stand_prob
        falling_lying_dominant = max(falling_prob, lying_prob) > max(stand_prob, sit_prob)
        range_m = _safe_float(row.get("range_m"), float("nan"))
        range_ok = not math.isnan(range_m) and range_m >= range_min_m
        standing_veto_ok = bool(
            stand_prob < standing_veto_prob
            and (stand_prob - sit_prob) < standing_veto_margin
        )
        evidence = bool(
            range_ok
            and sit_prob >= min_prob
            and sit_minus_stand >= margin
            and standing_veto_ok
            and not translation_confirmed
            and not falling_lying_dominant
        )
        histories[tid].append(evidence)
        stable_count = 0
        for value in reversed(histories[tid]):
            if not value:
                break
            stable_count += 1
        gate_passed = stable_count >= frames

        old_pose = _norm_label(row.get("old_display_pose"))
        new_pose = old_pose
        reason = "unchanged"
        if old_pose == "STANDING" and gate_passed:
            new_pose = "SITTING"
            reason = "sitting_relative_gate"
        elif old_pose == "MOVING" and moving_guard and gate_passed and not translation_confirmed:
            new_pose = "SITTING"
            reason = "moving_override_blocked_body_still_sitting"
        elif old_pose == "MOVING" and translation_confirmed:
            reason = "moving_override_translation_confirmed"

        out = dict(row)
        out.update(
            {
                "new_display_pose": new_pose,
                "replay_reason": reason,
                "sit_minus_stand_margin": sit_minus_stand,
                "relative_range_min_m": range_min_m,
                "relative_range_ok": range_ok,
                "standing_veto_prob": standing_veto_prob,
                "standing_veto_margin": standing_veto_margin,
                "standing_veto_ok": standing_veto_ok,
                "sitting_relative_gate_state": "PASS" if gate_passed else ("WAIT" if evidence else "NA"),
                "sitting_relative_gate_stable_count": stable_count,
                "sitting_relative_gate_required_frames": frames,
                "sitting_relative_gate_passed": gate_passed,
                "moving_translation_displacement_m": displacement,
                "moving_translation_confirmed": translation_confirmed,
            }
        )
        rows.append(out)
    return pd.DataFrame(rows)


def _pick_primary_tid(segment_rows: pd.DataFrame, expected_distance: float | None) -> int | None:
    if segment_rows.empty:
        return None
    rows = segment_rows.copy()
    rows["range_m_num"] = pd.to_numeric(rows.get("range_m"), errors="coerce")
    total_frames = max(1, rows["frame"].nunique())
    scores = []
    for tid, group in rows.groupby("tid"):
        presence = group["frame"].nunique() / total_frames
        mean_range = group["range_m_num"].mean()
        if expected_distance is None or pd.isna(expected_distance) or pd.isna(mean_range):
            range_score = 0.0
        else:
            range_score = -abs(float(mean_range) - float(expected_distance))
        scores.append((presence + range_score, int(tid)))
    if not scores:
        return None
    return max(scores)[1]


def _switch_count(labels: pd.Series) -> int:
    values = [_norm_label(v) for v in labels.tolist()]
    return sum(1 for prev, cur in zip(values, values[1:]) if prev != cur)


def _rate(labels: pd.Series, target: str) -> float:
    if len(labels) == 0:
        return float("nan")
    return float((labels.map(_norm_label) == target).mean())


def summarize_segments(replayed: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seg in segments.to_dict("records"):
        start = _safe_float(seg.get("start_time_s"), float("nan"))
        end = _safe_float(seg.get("end_time_s"), float("nan"))
        if math.isnan(start) or math.isnan(end):
            continue
        expected_pose = _norm_label(seg.get("expected_pose"))
        expected_distance = seg.get("expected_distance_m")
        window = replayed[(replayed["time_s"] >= start) & (replayed["time_s"] <= end)]
        primary_tid = _pick_primary_tid(window, expected_distance)
        if primary_tid is not None:
            window = window[window["tid"] == primary_tid].copy()
        frames = len(window)
        old = window["old_display_pose"] if frames else pd.Series(dtype=object)
        new = window["new_display_pose"] if frames else pd.Series(dtype=object)
        old_correct = old.map(_norm_label) == expected_pose
        new_correct = new.map(_norm_label) == expected_pose
        has_expected = expected_pose in {"STANDING", "SITTING", "MOVING", "LYING", "FALLING"}
        rows.append(
            {
                "segment_id": seg.get("segment_id", ""),
                "expected_pose": expected_pose,
                "expected_distance_m": expected_distance,
                "start_time_s": start,
                "end_time_s": end,
                "duration_s": end - start,
                "segmentation_method": seg.get("segmentation_method", seg.get("method", "")),
                "confidence": seg.get("confidence", ""),
                "primary_tid": primary_tid,
                "frames": frames,
                "old_accuracy": float(old_correct.mean()) if frames and has_expected else float("nan"),
                "new_accuracy": float(new_correct.mean()) if frames and has_expected else float("nan"),
                "old_display_sitting_rate": _rate(old, "SITTING"),
                "new_display_sitting_rate": _rate(new, "SITTING"),
                "old_display_standing_rate": _rate(old, "STANDING"),
                "new_display_standing_rate": _rate(new, "STANDING"),
                "old_display_moving_rate": _rate(old, "MOVING"),
                "new_display_moving_rate": _rate(new, "MOVING"),
                "pose_switch_count_old": _switch_count(old),
                "pose_switch_count_new": _switch_count(new),
                "standing_false_sitting_rate_old": (
                    _rate(old, "SITTING") if expected_pose == "STANDING" else float("nan")
                ),
                "standing_false_sitting_rate_new": (
                    _rate(new, "SITTING") if expected_pose == "STANDING" else float("nan")
                ),
                "sitting_false_standing_rate_old": (
                    _rate(old, "STANDING") if expected_pose == "SITTING" else float("nan")
                ),
                "sitting_false_standing_rate_new": (
                    _rate(new, "STANDING") if expected_pose == "SITTING" else float("nan")
                ),
                "mean_stand_prob": window["stand_prob"].mean() if frames else float("nan"),
                "mean_sit_prob": window["sit_prob"].mean() if frames else float("nan"),
                "mean_sit_minus_stand_margin": window["sit_minus_stand_margin"].mean() if frames else float("nan"),
                "relative_gate_pass_rate": window["sitting_relative_gate_passed"].mean() if frames else float("nan"),
                "moving_translation_confirmed_rate": window["moving_translation_confirmed"].mean() if frames else float("nan"),
                "notes": (
                    "Closest/highest-presence TID selected for segment; replay uses logged probabilities "
                    "and approximated translation evidence."
                ),
            }
        )
    return pd.DataFrame(rows)


def make_plots(replayed: pd.DataFrame, summary: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    label_map = {"UNKNOWN": 0, "NO_POSE": 0, "WARMUP": 0, "STANDING": 1, "SITTING": 2, "MOVING": 3, "LYING": 4, "FALLING": 5}
    sample = replayed.sort_values("time_s")
    if len(sample) > 5000:
        sample = sample.iloc[:: max(1, len(sample) // 5000)]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(sample["time_s"], sample["old_display_pose"].map(lambda v: label_map.get(_norm_label(v), 0)), label="old", alpha=0.8)
    ax.plot(sample["time_s"], sample["new_display_pose"].map(lambda v: label_map.get(_norm_label(v), 0)), label="new", alpha=0.8)
    ax.set_yticks(list(label_map.values()), list(label_map.keys()))
    ax.set_xlabel("time_s")
    ax.set_title("Old vs new display pose timeline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "old_vs_new_display_pose_timeline.png", dpi=150)
    plt.close(fig)

    if not summary.empty:
        x = range(len(summary))
        for metric, filename, title in [
            ("accuracy", "old_vs_new_accuracy_by_segment.png", "Old vs new accuracy by segment"),
            ("display_sitting_rate", "old_vs_new_sit_rate_by_segment.png", "Old vs new SITTING display rate"),
            ("standing_false_sitting_rate", "old_vs_new_standing_false_sitting_rate.png", "Standing false-SITTING rate"),
        ]:
            old_col = f"old_{metric}"
            new_col = f"new_{metric}"
            if old_col not in summary.columns or new_col not in summary.columns:
                continue
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar([i - 0.2 for i in x], summary[old_col], width=0.4, label="old")
            ax.bar([i + 0.2 for i in x], summary[new_col], width=0.4, label="new")
            ax.set_xticks(list(x), summary["segment_id"], rotation=30, ha="right")
            ax.set_ylim(0, 1)
            ax.set_title(title)
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / filename, dpi=150)
            plt.close(fig)


def _markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy()
    cols = [str(col) for col in view.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in view.to_dict("records"):
        values = [str(row.get(col, "")) for col in view.columns]
        values = [value.replace("|", "\\|").replace("\n", " ") for value in values]
        lines.append("| " + " | ".join(values) + " |")
    if len(df) > max_rows:
        lines.append(f"| ... | {' | '.join([''] * (len(cols) - 1))} |")
    return "\n".join(lines)


def write_report(
    out_dir: Path,
    session: Path,
    segments: pd.DataFrame,
    summary: pd.DataFrame,
    params: dict[str, Any],
) -> None:
    report = out_dir / "DECISION_REPLAY_REPORT.md"
    lines = [
        "# Posture Decision Fix Replay",
        "",
        f"Session: `{session}`",
        "",
        "Replay limitation: this is an offline approximation using logged probabilities, logged old display pose, TID, range/track position, and segment labels. It does not rerun the ONNX model or every live smoother state.",
        "",
        "## Parameters",
        "",
        _markdown_table(pd.DataFrame([params])),
        "",
        "## Segments",
        "",
        _markdown_table(segments),
        "",
        "## Segment Results",
        "",
        _markdown_table(summary),
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay the evidence-based sitting decision fix on logged sessions.")
    parser.add_argument("--session", required=True, help="Session folder containing mmwave_pose.csv.")
    parser.add_argument("--segments", help="Optional segment CSV with expected_pose/start/end.")
    parser.add_argument("--out", required=True, help="Output folder.")
    parser.add_argument("--expected-distances", default="", help="Accepted for command compatibility; segment files remain preferred.")
    parser.add_argument("--make-plots", action="store_true", help="Generate replay plots.")
    parser.add_argument("--relative-range-min-m", type=float, default=SITTING_RELATIVE_RANGE_MIN_M)
    parser.add_argument("--relative-min-prob", type=float, default=None)
    parser.add_argument("--relative-margin", type=float, default=None)
    parser.add_argument("--relative-frames", type=int, default=None)
    parser.add_argument("--standing-veto-prob", type=float, default=STANDING_VETO_PROB)
    parser.add_argument("--standing-veto-margin", type=float, default=STANDING_VETO_MARGIN)
    parser.add_argument("--sitting-relative-min-prob", type=float, default=SITTING_RELATIVE_MIN_PROB)
    parser.add_argument("--sitting-relative-margin", type=float, default=SITTING_RELATIVE_MARGIN)
    parser.add_argument("--sitting-relative-frames", type=int, default=SITTING_RELATIVE_FRAMES)
    parser.add_argument("--disable-moving-guard", action="store_true")
    args = parser.parse_args()

    session = Path(args.session).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pose = _load_pose(session)
    pose = _load_tracks(session, pose)
    segments = _load_segments(Path(args.segments) if args.segments else None, session, pose)
    min_prob = args.sitting_relative_min_prob if args.relative_min_prob is None else args.relative_min_prob
    margin = args.sitting_relative_margin if args.relative_margin is None else args.relative_margin
    frames = args.sitting_relative_frames if args.relative_frames is None else args.relative_frames
    params = {
        "relative_range_min_m": args.relative_range_min_m,
        "relative_min_prob": min_prob,
        "relative_margin": margin,
        "relative_frames": max(1, int(frames)),
        "standing_veto_prob": args.standing_veto_prob,
        "standing_veto_margin": args.standing_veto_margin,
        "moving_guard": not args.disable_moving_guard,
    }
    replayed = replay_decisions(
        pose,
        range_min_m=float(params["relative_range_min_m"]),
        min_prob=float(params["relative_min_prob"]),
        margin=float(params["relative_margin"]),
        frames=int(params["relative_frames"]),
        standing_veto_prob=float(params["standing_veto_prob"]),
        standing_veto_margin=float(params["standing_veto_margin"]),
        moving_guard=bool(params["moving_guard"]),
    )
    summary = summarize_segments(replayed, segments)

    replayed.to_csv(out_dir / "decision_replay_frames.csv", index=False)
    summary.to_csv(out_dir / "decision_replay_by_segment.csv", index=False)
    segments.to_csv(out_dir / "segments_used.csv", index=False)
    make_plots(replayed, summary, out_dir)
    write_report(out_dir, session, segments, summary, params)

    print(f"Replay rows: {len(replayed)}")
    print(f"Segments: {len(summary)}")
    print(f"Output: {out_dir}")
    if not summary.empty:
        cols = ["segment_id", "expected_pose", "old_accuracy", "new_accuracy", "old_display_sitting_rate", "new_display_sitting_rate"]
        print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
