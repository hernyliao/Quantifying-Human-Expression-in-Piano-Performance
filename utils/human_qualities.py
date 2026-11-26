'''
    1. rubato                  - subtle stretching / compressing of time
    2. micro-timing variations - imperceptible early / late arrivals
    3. breath                  - inhale/exhale phrasing gaps between measures
    4. dynamic shaping         - living gradient within & across measures
    6. tension and release     - build toward climax, resolve
'''

import csv
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_INPUT_DIR  = PROJECT_ROOT / "output" / "csv_files"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_VISUAL_DIR = PROJECT_ROOT / "output" / "visuals"

NUMERIC_COLUMNS = {"measure_index", "note_count", "recorded_dominant_frequency_hz", "expected_start_sec", 
                    "start_sec", "end_sec", "measure_length_sec", "timing_offset_sec", 
                    "inter_measure_interval_sec", "gap_to_next_sec", "volume",
                    "attack_volume", "sustain_volume", "decay_volume", "volume_range_within", 
                    "volume_std_within", "energy_spike", "pitch_confidence", "confidence"}

def read_measure_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                if k in NUMERIC_COLUMNS:
                    parsed[k] = float(v) if v not in ("", None) else float("nan")
                else:
                    parsed[k] = v
            # tolerate older measure CSVs that pre-date columns like
            # volume_range_within / volume_std_within
            for k in NUMERIC_COLUMNS:
                parsed.setdefault(k, float("nan"))
            rows.append(parsed)
    return rows

read_note_csv = read_measure_csv

def moving_average(x: np.ndarray, window: int) -> np.ndarray: # centered moving average that ignores NaNs; edges shrink window instead of zero-padding
    n = len(x)
    if n == 0:
        return x.copy()
    half = max(1, window // 2)
    out = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_vals = x[lo:hi]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) > 0:
            out[i] = float(np.mean(valid))
    return out

def zscore(x: np.ndarray) -> np.ndarray:
    valid = x[~np.isnan(x)]
    if len(valid) < 2:
        return np.zeros_like(x)
    mu = float(np.mean(valid))
    sd = float(np.std(valid))
    if sd == 0:
        return np.zeros_like(x)
    return (x - mu) / sd

def clamp01(x):
    if isinstance(x, np.ndarray):
        return np.clip(x, 0.0, 1.0)
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return float("nan")
    return max(0.0, min(1.0, x))

def fit_baseline_volume(rows: list[dict]) -> tuple[float, float]:
    # Least-squares fit of volume vs. note_count on a baseline recording.
    # The line a*note_count + b is the "score-structural" volume the piece
    # mechanically demands (chord density). Used by analyse_human_qualities
    # as the reference dynamic-shaping should deviate from.
    nc = np.array([r.get("note_count", float("nan")) for r in rows], dtype=float)
    v  = np.array([r.get("volume",     float("nan")) for r in rows], dtype=float)
    m = ~np.isnan(nc) & ~np.isnan(v) & (nc > 0)
    if m.sum() >= 2:
        a, b = np.polyfit(nc[m], v[m], 1)
        return float(a), float(b)
    if m.sum() == 1:
        return 0.0, float(v[m][0])
    return 0.0, 1.0

def analyse_human_qualities(rows: list[dict], volume_fit: tuple[float, float] | None = None) -> list[dict]:
    n = len(rows)

    measure_index = np.array([r["measure_index"] for r in rows])
    note_count = np.array([r["note_count"] for r in rows])
    start_sec = np.array([r["start_sec"] for r in rows])
    end_sec = np.array([r["end_sec"] for r in rows])
    expected_start = np.array([r["expected_start_sec"] for r in rows])
    measure_len = np.array([r["measure_length_sec"] for r in rows])
    ioi = np.array([r["inter_measure_interval_sec"] for r in rows])
    gap = np.array([r["gap_to_next_sec"] for r in rows])
    timing_off = np.array([r["timing_offset_sec"] for r in rows])
    volume = np.array([r["volume"] for r in rows])
    attack_v = np.array([r["attack_volume"] for r in rows])
    sustain_v = np.array([r["sustain_volume"] for r in rows])
    decay_v = np.array([r["decay_volume"] for r in rows])
    vol_range = np.array([r["volume_range_within"] for r in rows])
    vol_std = np.array([r["volume_std_within"] for r in rows])
    freq = np.array([r["recorded_dominant_frequency_hz"] for r in rows])

    # RUBATO — recorded interonset interval (gap between ) vs ioi from xml
    expected_ioi = np.full(n, np.nan)
    for i in range(n - 1):
        expected_ioi[i] = expected_start[i + 1] - expected_start[i]

    # >1 means lingering on this measure, <1 means rushing through it
    rubato_ratio = np.full(n, np.nan)
    for i in range(n):
        if expected_ioi[i] and expected_ioi[i] > 0:
            rubato_ratio[i] = ioi[i] / expected_ioi[i]
    rubato_ratio = np.clip(rubato_ratio, 0.25, 4.0)

    # 3-measure window of rubato trend
    rubato_smoothed = moving_average(rubato_ratio, window=3)

    # MICRO-TIMING — per-measure deviation of rubato_ratio from the local
    # smoothed tempo, converted back to seconds. This is the high-frequency
    # component of the tempo signal and is immune to score-vs-recording tempo
    # mismatches (e.g. a baseline that ignores XML tempo markings shows step
    # changes in rubato_ratio, but rubato_smoothed tracks them within ~3
    # measures, so the residual stays near zero everywhere except briefly at
    # the step itself).
    micro_jitter = rubato_ratio - rubato_smoothed
    micro_timing = micro_jitter * np.where(np.isnan(expected_ioi), np.nan, expected_ioi)
    timing_trend = timing_off - micro_timing  # only used as a "what the trend says" series for plots
    EDGE_DROP = 2
    if n > 2 * EDGE_DROP:
        micro_timing[:EDGE_DROP] = 0.0
        micro_timing[-EDGE_DROP:] = 0.0

    # local variability of the rubato signal — a metronomic baseline has
    # ~zero within any constant-tempo segment. A pro varies continuously.
    rubato_local_std = np.full(n, np.nan)
    half_w = 2
    for i in range(n):
        lo, hi = max(0, i - half_w), min(n, i + half_w + 1)
        v = rubato_ratio[lo:hi]
        v = v[~np.isnan(v)]
        if len(v) >= 2:
            rubato_local_std[i] = float(np.std(v))

    # BREATH — inter-measure gap large enough to be intentional phrasing.
    # Use a fixed 150ms threshold and normalise depth against a measure-length
    # reference rather than the file's max gap (which is usually startup latency).
    valid_mlen = measure_len[~np.isnan(measure_len)]
    ref_mlen = float(np.median(valid_mlen)) if len(valid_mlen) > 0 else 1.0
    breath_ref = max(0.2, ref_mlen * 0.15)
    breath_threshold = 0.15
    is_breath = np.array([1 if (not np.isnan(g) and g >= breath_threshold) else 0 for g in gap])
    breath_depth = np.where(np.isnan(gap), 0.0, np.clip(gap / breath_ref, 0.0, 1.0))
    # final measure has no meaningful gap-to-next; don't reward it for breath
    if n > 0:
        is_breath[-1] = 0
        breath_depth[-1] = 0.0

    # DYNAMIC SHAPING:
    #   intra_decay / intra_shape_range: kept for visualisation, but the score
    #   uses *relative* dynamics so chord-density-driven volume swings don't
    #   give a baseline credit for "shaping".
    intra_decay = attack_v - decay_v
    intra_shape_range = np.full(n, np.nan)
    for i in range(n):
        triple = np.array([attack_v[i], sustain_v[i], decay_v[i]])
        if np.all(np.isnan(triple)):
            continue
        intra_shape_range[i] = float(np.nanmax(triple) - np.nanmin(triple))

    phrase_volume = moving_average(volume, window=3)

    # per-note volume — divides out the mechanical contribution of chord density
    # to RMS so the dynamic-shape score reflects per-note striking intent rather
    # than how many notes happened to sound this measure. note_count comes from
    # the parsed XML score, so missed/extra recorded notes don't distort it.
    notes_per_measure = np.where(np.isnan(note_count) | (note_count <= 0), 1.0, note_count)
    volume_per_note = volume / notes_per_measure
    phrase_volume_per_note = moving_average(volume_per_note, window=3)

    # TENSION & RELEASE — volume + log-pitch + (inverted) IOI, z-scored
    vol_z = zscore(np.nan_to_num(volume, nan=np.nanmean(volume) if np.any(~np.isnan(volume)) else 0))
    log_freq = np.log2(np.where((freq > 0) & ~np.isnan(freq), freq, np.nan))
    pitch_z = zscore(np.nan_to_num(log_freq, nan=np.nanmean(log_freq) if np.any(~np.isnan(log_freq)) else 0))
    # invert so faster (smaller) IOI -> more tension
    ioi_inv = -zscore(np.nan_to_num(ioi, nan=np.nanmean(ioi) if np.any(~np.isnan(ioi)) else 0))

    tension_raw = (vol_z + pitch_z + ioi_inv) / 3.0
    tension = moving_average(tension_raw, window=3)
    release = np.full(n, 0.0)
    for i in range(n - 1):
        release[i] = max(0.0, tension[i] - tension[i + 1])

    # tension is z-scored per-file, so a metronomic recording's microscopic
    # variation gets stretched to fill [-1.5, +1.5] anyway. Gate by the raw
    # pre-normalisation variability so flatlines score near zero.
    raw_vol_std = float(np.nanstd(volume)) if np.any(~np.isnan(volume)) else 0.0
    raw_pitch_std = float(np.nanstd(log_freq)) if np.any(~np.isnan(log_freq)) else 0.0
    raw_ioi_std = float(np.nanstd(ioi)) if np.any(~np.isnan(ioi)) else 0.0
    # mean global volume — denominator for relative volume normalisation
    mean_vol = float(np.nanmean(volume)) if np.any(~np.isnan(volume)) else 1.0
    mean_vol = max(0.1, mean_vol)
    vol_gate = clamp01(raw_vol_std / mean_vol / 0.15)  # 15% relative vol std = full credit
    # per-note mean — denominator for dyn_score so its normalisation is
    # consistent with the per-note volume series
    mean_vpn = float(np.nanmean(volume_per_note)) if np.any(~np.isnan(volume_per_note)) else 1.0
    mean_vpn = max(0.01, mean_vpn)
    pitch_gate = clamp01(raw_pitch_std / 0.6)          # 0.6 octaves of log2-freq spread
    ioi_gate = clamp01(raw_ioi_std / 0.3)              # 300ms ioi spread
    tension_gate = float(max(vol_gate, pitch_gate, ioi_gate))

    # per-measure humanness summary — rough heuristic, 0–1 each
    # micro: how much THIS measure jitters off the locally-smoothed tempo
    micro_score = clamp01(np.abs(np.nan_to_num(micro_timing, nan=0.0)) / 0.15)
    # rubato: how much the tempo VARIES across a local window. A baseline
    # at constant tempo (even if not matching the XML's tempo) scores ~0.
    rubato_score = clamp01(np.nan_to_num(rubato_local_std, nan=0.0) / 0.10)
    # DYNAMIC SHAPING — phrase volume relative to a score-structural reference.
    # The reference is volume = a*note_count + b, fit on the piece's baseline
    # recording (a synthesised constant-velocity render). That line captures
    # the volume the score mechanically demands from chord density, so the
    # residual is the human's expressive deviation on top of it.
    # Earlier versions averaged this with a within-measure RMS range term,
    # but that term was dominated by piano attack/decay envelope mechanics
    # and actually rewarded the baseline more than the human; it's dropped.
    # Fallback when no fit is available (e.g. no baseline yet): compare
    # phrase per-note volume to its own global mean.
    if volume_fit is not None:
        a_vol, b_vol = volume_fit
        predicted_vol = a_vol * notes_per_measure + b_vol
        phrase_dev_score = np.abs(np.nan_to_num(phrase_volume, nan=mean_vol) - predicted_vol) / mean_vol
        dyn_score = clamp01(phrase_dev_score / 0.3)
    else:
        phrase_dev = np.abs(np.nan_to_num(phrase_volume_per_note, nan=mean_vpn) - mean_vpn) / mean_vpn
        dyn_score = clamp01(phrase_dev / 0.3)
    breath_score = is_breath * 0.5 + breath_depth * 0.5
    tension_score = clamp01(np.abs(np.nan_to_num(tension, nan=0.0)) / 1.5) * tension_gate

    # reweighted: rubato (most reliable feature) gets more pull; dyn (a weak
    # discriminator because piano envelopes look mechanical too) gets less.
    humanness = (0.10 * micro_score + 0.45 * rubato_score + 0.15 * dyn_score + 0.10 * breath_score + 0.20 * tension_score)

    out = []
    round_digits = 4
    for i, r in enumerate(rows):
        out.append({"measure_index": int(measure_index[i]) if not np.isnan(measure_index[i]) else i + 1,
                    "note_count": int(note_count[i]) if not np.isnan(note_count[i]) else 0,
                    "expected_pitches": r.get("expected_pitches", ""),
                    "start_sec": round(float(start_sec[i]), round_digits),
                    "end_sec": round(float(end_sec[i]), round_digits),
                    "measure_length_sec": round(float(measure_len[i]), round_digits),
                    "rubato_ratio": round(float(rubato_ratio[i]), round_digits),
                    "rubato_smoothed": round(float(rubato_smoothed[i]), round_digits),
                    "micro_timing_sec": round(float(micro_timing[i]), round_digits),
                    "timing_trend_sec": round(float(timing_trend[i]), round_digits),
                    "gap_to_next_sec": round(float(gap[i]), round_digits),
                    "is_breath": int(is_breath[i]),
                    "breath_depth": round(float(breath_depth[i]), round_digits),
                    "intra_decay": round(float(intra_decay[i]), round_digits),
                    "intra_shape_range": round(float(intra_shape_range[i]), round_digits),
                    "within_measure_volume_range": round(float(vol_range[i]), round_digits),
                    "within_measure_volume_std": round(float(vol_std[i]), round_digits),
                    "phrase_volume": round(float(phrase_volume[i]), round_digits),
                    "tension": round(float(tension[i]), round_digits),
                    "release": round(float(release[i]), round_digits),
                    "rubato_score": round(float(rubato_score[i]), round_digits),
                    "micro_score": round(float(micro_score[i]), round_digits),
                    "breath_score": round(float(breath_score[i]), round_digits),
                    "dyn_score": round(float(dyn_score[i]), round_digits),
                    "tension_score": round(float(tension_score[i]), round_digits),
                    "humanness_score": round(float(humanness[i]), round_digits)})
    return out

HUMAN_QUALITY_COLUMNS = ["measure_index", "note_count", "expected_pitches", "start_sec", "end_sec", "measure_length_sec",
                         "rubato_ratio", "rubato_smoothed", "micro_timing_sec", "timing_trend_sec", "gap_to_next_sec",
                         "is_breath", "breath_depth", "intra_decay", "intra_shape_range", "within_measure_volume_range",
                         "within_measure_volume_std", "phrase_volume", "tension", "release",
                         "rubato_score", "micro_score", "breath_score", "dyn_score", "tension_score",
                         "humanness_score",]

def write_human_quality_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HUMAN_QUALITY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write_human_quality_csv] CSV written to {path} ({len(rows)} rows)")

def _x_axis(rows: list[dict]): # measure start time on x-axis; measure_index fallback when start_sec is missing
    xs = np.array([r["start_sec"] if r["start_sec"] != "" else np.nan for r in rows], dtype=float)
    if np.all(np.isnan(xs)):
        idx = np.array([r.get("measure_index", i + 1) for i, r in enumerate(rows)], dtype=float)
        return idx, "measure index"
    nan_mask = np.isnan(xs)
    if np.any(nan_mask):
        idx = np.array([r.get("measure_index", i + 1) for i, r in enumerate(rows)], dtype=float)
        xs[nan_mask] = idx[nan_mask]
    return xs, "time (s)"

def _val(row, key):
    v = row.get(key, "")
    if v == "" or v is None:
        return float("nan")
    return float(v)

def _arr(rows, key):
    return np.array([_val(r, key) for r in rows], dtype=float)

def _bar_width(x: np.ndarray) -> float:
    return (x[-1] - x[0]) / max(1, len(x) * 1.8) if len(x) > 1 else 0.3

def plot_rubato(rows, out_path, title):
    x, xlabel = _x_axis(rows)
    raw = _arr(rows, "rubato_ratio")
    smth = _arr(rows, "rubato_smoothed")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="metronomic")
    ax.plot(x, raw, "o-", alpha=0.4, label="per-measure rubato ratio")
    ax.plot(x, smth, "-", linewidth=2.2, color="C3", label="smoothed (slow ebb/flow)")
    ax.fill_between(x, 1.0, smth, where=~np.isnan(smth), alpha=0.15, color="C3")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("actual IOI / expected IOI (per measure)")
    ax.set_title(f"Rubato -- {title}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

def plot_micro_timing(rows, out_path, title):
    x, xlabel = _x_axis(rows)
    mt = _arr(rows, "micro_timing_sec")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    colors = ["C2" if v >= 0 else "C0" for v in mt]
    ax.bar(x, mt * 1000.0, width=_bar_width(x), color=colors, alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("micro-timing (ms) -- early < 0 < late")
    ax.set_title(f"Micro-timing variations (per measure onset) -- {title}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

def plot_breath(rows, out_path, title):
    x, xlabel = _x_axis(rows)
    gaps = _arr(rows, "gap_to_next_sec")
    breath = np.array([int(r.get("is_breath") or 0) for r in rows])
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["C3" if b else "C7" for b in breath]
    ax.bar(x, gaps, width=_bar_width(x), color=colors, alpha=0.9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("gap to next measure (s)")
    ax.set_title(f"Breath / phrasing gaps -- {title} (red = flagged breath)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

def plot_dynamic_shaping(rows, out_path, title):
    x, xlabel = _x_axis(rows)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, _arr(rows, "phrase_volume"), "-", linewidth=2.4, color="C3", label="phrase-level volume (across measures)")
    ax.plot(x, _arr(rows, "intra_shape_range"), "--", linewidth=1.5, color="C0", label="attack/sustain/decay range (within measure)")
    ax.plot(x, _arr(rows, "within_measure_volume_range"), ":", linewidth=1.5, color="C2", label="broadband RMS range (within measure)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("normalised volume")
    ax.set_title(f"Dynamic shaping -- {title}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

def plot_tension_release(rows, out_path, title):
    x, xlabel = _x_axis(rows)
    tension = _arr(rows, "tension")
    release = _arr(rows, "release")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.plot(x, tension, "-", linewidth=2.2, color="C3", label="tension")
    ax.fill_between(x, 0, tension, where=tension > 0, alpha=0.20, color="C3")
    ax.fill_between(x, 0, tension, where=tension < 0, alpha=0.20, color="C0")
    # mark releases as down-arrows above the tension curve
    for xi, ti, ri in zip(x, tension, release):
        if not np.isnan(ri) and ri > 0.3:
            ax.annotate("", xy=(xi, ti - ri), xytext=(xi, ti), arrowprops=dict(arrowstyle="->", color="C2", alpha=0.6))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("tension (z-score of vol+pitch+drive)")
    ax.set_title(f"Tension & release -- {title}")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

def plot_summary(rows, out_path, title): # one-page dashboard, small multiples of every quality
    x, xlabel = _x_axis(rows)
    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    fig.suptitle(f"Human-quality dashboard (measure-level) -- {title}", fontsize=14)

    ax = axes[0, 0]
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.plot(x, _arr(rows, "rubato_smoothed"), "-", color="C3")
    ax.plot(x, _arr(rows, "rubato_ratio"), "o", alpha=0.4, color="C0")
    ax.set_title("Rubato"); ax.set_ylabel("ratio"); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    mt = _arr(rows, "micro_timing_sec") * 1000.0
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.bar(x, mt, width=_bar_width(x),
           color=["C2" if v >= 0 else "C0" for v in mt], alpha=0.85)
    ax.set_title("Micro-timing (ms)"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    gaps = _arr(rows, "gap_to_next_sec")
    breath = np.array([int(r.get("is_breath") or 0) for r in rows])
    ax.bar(x, gaps, width=_bar_width(x),
           color=["C3" if b else "C7" for b in breath], alpha=0.85)
    ax.set_title("Breath / gaps (s)"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x, _arr(rows, "phrase_volume"), "-", color="C3", linewidth=2.0,
            label="phrase volume")
    ax.plot(x, _arr(rows, "within_measure_volume_range"), ":", color="C2",
            label="within-measure range")
    ax.plot(x, _arr(rows, "intra_shape_range"), "--", color="C0",
            label="a/s/d range")
    ax.legend(fontsize=7)
    ax.set_title("Dynamic shaping"); ax.grid(alpha=0.3)

    ax = axes[2, 1]
    tension = _arr(rows, "tension")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.plot(x, tension, "-", color="C3", linewidth=2.0)
    ax.fill_between(x, 0, tension, where=tension > 0, alpha=0.20, color="C3")
    ax.fill_between(x, 0, tension, where=tension < 0, alpha=0.20, color="C0")
    ax.set_title("Tension & release"); ax.grid(alpha=0.3)

    # bottom row: humanness + a blank slot to keep the same 4x2 layout
    ax = axes[3, 0]
    ax.plot(x, _arr(rows, "humanness_score"), "o-", color="C4", linewidth=2.0)
    ax.set_ylim(0, 1)
    ax.set_title("Per-measure humanness score"); ax.set_xlabel(xlabel); ax.grid(alpha=0.3)

    axes[3, 1].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

def plot_comparison(rows_a, rows_b, out_path, piece_name,
                    label_a="performance", label_b="baseline",
                    color_a="C3", color_b="C0"): # overlay any two role recordings on the same axes
    if not rows_a or not rows_b:
        return
    xa, _ = _x_axis(rows_a)
    xb, _ = _x_axis(rows_b)

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    fig.suptitle(f"{label_a.capitalize()} vs {label_b} (measure-level) -- {piece_name}", fontsize=14)

    ax = axes[0, 0]
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.plot(xa, _arr(rows_a, "rubato_smoothed"), "-", color=color_a, linewidth=2.0, label=label_a)
    ax.plot(xb, _arr(rows_b, "rubato_smoothed"), "-", color=color_b, linewidth=2.0, label=label_b)
    ax.set_title("Rubato (smoothed)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.plot(xa, _arr(rows_a, "micro_timing_sec") * 1000.0, "-", color=color_a, linewidth=1.6, label=label_a)
    ax.plot(xb, _arr(rows_b, "micro_timing_sec") * 1000.0, "-", color=color_b, linewidth=1.6, label=label_b)
    ax.set_title("Micro-timing (ms)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(xa, _arr(rows_a, "gap_to_next_sec"), "-", color=color_a, linewidth=1.6, label=label_a)
    ax.plot(xb, _arr(rows_b, "gap_to_next_sec"), "-", color=color_b, linewidth=1.6, label=label_b)
    ax.set_title("Breath (gap to next measure, s)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(xa, _arr(rows_a, "phrase_volume"), "-", color=color_a, linewidth=2.0, label=label_a)
    ax.plot(xb, _arr(rows_b, "phrase_volume"), "-", color=color_b, linewidth=2.0, label=label_b)
    ax.set_title("Dynamic shaping"); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    ax = axes[2, 1]
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.plot(xa, _arr(rows_a, "tension"), "-", color=color_a, linewidth=2.0, label=label_a)
    ax.plot(xb, _arr(rows_b, "tension"), "-", color=color_b, linewidth=2.0, label=label_b)
    ax.set_title("Tension"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

def plot_comparison_all(roles_to_rows: dict, out_path, piece_name, colors: dict): # overlay every available role on shared axes
    items = [(role, rows) for role, rows in roles_to_rows.items() if rows]
    if len(items) < 2:
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    fig.suptitle(f"All recordings (measure-level) -- {piece_name}", fontsize=14)

    def overlay(ax, key, scale=1.0):
        for role, rows in items:
            x, _ = _x_axis(rows)
            ax.plot(x, _arr(rows, key) * scale, "-", color=colors.get(role, "C7"), linewidth=2.0, label=role)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[0, 0]
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    overlay(ax, "rubato_smoothed")
    ax.set_title("Rubato (smoothed)")

    ax = axes[0, 1]
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    overlay(ax, "micro_timing_sec", scale=1000.0)
    ax.set_title("Micro-timing (ms)")

    ax = axes[1, 0]
    overlay(ax, "gap_to_next_sec")
    ax.set_title("Breath (gap to next measure, s)")

    ax = axes[1, 1]
    for role, rows in items:
        x, _ = _x_axis(rows)
        color = colors.get(role, "C7")
        ax.plot(x, _arr(rows, "phrase_volume"), "-", color=color, linewidth=2.0, label=role)
        
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    ax.set_title("Dynamic shaping")

    ax = axes[2, 1]
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    overlay(ax, "tension")
    ax.set_title("Tension")

    ax = axes[2, 0]
    overlay(ax, "humanness_score")
    ax.set_ylim(0, 1)
    ax.set_title("Per-measure humanness score")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

ROLE_COLORS = {"performance": "C3", "baseline": "C0", "pro": "C2"}

SUMMARY_METRICS = [("rubato", "rubato_score"),
                   ("micro", "micro_score"),
                   ("breath", "breath_score"),
                   ("dyn", "dyn_score"),
                   ("tension", "tension_score"),
                   ("humanness", "humanness_score")]

SUMMARY_COLUMNS = (["piece", "role", "n_measures"]
                   + [k for k, _ in SUMMARY_METRICS]
                   + [f"{k}_p10" for k, _ in SUMMARY_METRICS]
                   + [f"{k}_p90" for k, _ in SUMMARY_METRICS]
                   + ["humanness_range"])

def summarise_recording(rows: list[dict], piece: str, role: str) -> dict:
    # note-count-weighted means of the 0-1 sub-scores produced by analyse_human_qualities.
    # measures with 0 notes (rests, silence) contribute nothing.
    weights = np.array([float(r.get("note_count") or 0) for r in rows], dtype=float)
    weights = np.where(np.isnan(weights) | (weights <= 0), 0.0, weights)
    if weights.sum() == 0:
        weights = np.ones(len(rows), dtype=float)

    def vals_for(key: str) -> np.ndarray:
        return np.array([float(r.get(key)) if r.get(key) not in (None, "") else float("nan") for r in rows], dtype=float)

    def wmean(key: str) -> float:
        vals = vals_for(key)
        mask = ~np.isnan(vals)
        w = weights[mask]
        if not mask.any() or w.sum() == 0:
            return float("nan")
        return float(np.average(vals[mask], weights=w))

    def percentiles(key: str) -> tuple[float, float]:
        arr = vals_for(key)
        arr = arr[~np.isnan(arr)]
        if not len(arr):
            return float("nan"), float("nan")
        return float(np.percentile(arr, 10)), float(np.percentile(arr, 90))

    rd = 4
    out = {"piece": piece, "role": role, "n_measures": len(rows)}
    for short, raw in SUMMARY_METRICS:
        out[short] = round(wmean(raw), rd)
        p10, p90 = percentiles(raw)
        out[f"{short}_p10"] = round(p10, rd)
        out[f"{short}_p90"] = round(p90, rd)
    out["humanness_range"] = (round(out["humanness_p90"] - out["humanness_p10"], rd)
                              if not (math.isnan(out["humanness_p10"]) or math.isnan(out["humanness_p90"]))
                              else float("nan"))
    return out

def write_summary_csv(summaries: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"[write_summary_csv] {path} ({len(summaries)} rows)")

def plot_humanness_summary(summaries: list[dict], out_path: Path):
    if not summaries:
        return
    pieces = sorted({s["piece"] for s in summaries})
    role_order = [r for r in ("baseline", "performance", "pro") if any(s["role"] == r for s in summaries)]
    by_pr = {(s["piece"], s["role"]): s for s in summaries}

    panels = [("rubato", "Rubato"),
              ("micro", "Micro-timing"),
              ("breath", "Breath"),
              ("dyn", "Dynamics"),
              ("tension", "Tension"),
              ("humanness", "Overall humanness")]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Humanness summary across pieces (baseline = robotic, performance / pro = human)", fontsize=13)

    n_roles = max(1, len(role_order))
    bar_w = 0.8 / n_roles
    x = np.arange(len(pieces))

    for ax_i, (key, label) in enumerate(panels):
        ax = axes[ax_i // 3, ax_i % 3]
        for r_i, role in enumerate(role_order):
            heights = [float(by_pr[(p, role)][key]) if (p, role) in by_pr and not math.isnan(float(by_pr[(p, role)][key])) else 0.0
                       for p in pieces]
            offset = (r_i - (n_roles - 1) / 2.0) * bar_w
            ax.bar(x + offset, heights, bar_w, color=ROLE_COLORS.get(role, "C7"),
                   label=role if ax_i == 0 else None, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(pieces, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(label)
        ax.grid(alpha=0.3, axis="y")
        if ax_i == 0:
            ax.legend(fontsize=9, loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[plot_humanness_summary] {out_path}")

def generate_plots(rows, stem: str, visuals_dir: Path):
    visuals_dir.mkdir(parents=True, exist_ok=True)
    title = stem
    plot_rubato(rows, visuals_dir / f"{stem}__1_rubato.png", title)
    plot_micro_timing(rows, visuals_dir / f"{stem}__2_micro_timing.png", title)
    plot_breath(rows, visuals_dir / f"{stem}__3_breath.png", title)
    plot_dynamic_shaping(rows, visuals_dir / f"{stem}__4_dynamic_shaping.png", title)
    plot_tension_release(rows, visuals_dir / f"{stem}__6_tension_release.png", title)
    plot_summary(rows, visuals_dir / f"{stem}__0_summary.png", title)
    print(f"[generate_plots] wrote 7 plots for {stem} -> {visuals_dir}")

PIECE_STEM_RE = re.compile(r"^(?P<piece>.+?)_(performance|baseline|pro)_recording$")

def piece_name(stem: str) -> Optional[str]:
    m = PIECE_STEM_RE.match(stem)
    return m.group("piece") if m else None

def run(input_dir: Path, output_dir: Path, visuals_dir: Path):
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    visuals_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(input_dir.glob("*_measure.csv"))
    if not csv_paths:
        print(f"[run] no measure CSVs in {input_dir} — run measure_isolator first")
        return

    print(f"[run] found {len(csv_paths)} measure CSV(s) in {input_dir}")

    # Two-pass: fit each piece's baseline volume->note_count line first so we
    # can pass it as a score-structural reference into analyse_human_qualities.
    source_rows: dict[str, list[dict]] = {}
    for path in csv_paths:
        rows = read_measure_csv(path)
        if not rows:
            print(f"[run] skipping empty {path.name}")
            continue
        source_rows[path.stem] = rows
    volume_fits: dict[str, tuple[float, float]] = {}
    for stem, rows in source_rows.items():
        if "_baseline_" in stem:
            piece = piece_name(stem)
            if piece is not None:
                volume_fits[piece] = fit_baseline_volume(rows)

    analyses: dict[str, list[dict]] = {}
    for path in csv_paths:
        rows = source_rows.get(path.stem)
        if not rows:
            continue
        print(f"\n=== Analysing {path.name} ({len(rows)} measures) ===")
        fit = volume_fits.get(piece_name(path.stem) or "")
        analysed = analyse_human_qualities(rows, volume_fit=fit)
        analyses[path.stem] = analysed

        out_csv = output_dir / f"{path.stem}_human_qualities.csv"
        write_human_quality_csv(analysed, out_csv)
        generate_plots(analysed, path.stem, visuals_dir)

    # pair performance/baseline/pro if both exist for the same piece
    by_piece: dict[str, dict[str, list[dict]]] = {}
    for stem, rows in analyses.items():
        p = piece_name(stem)
        if not p:
            continue
        kind = next((k for k in ("performance", "baseline", "pro") if k in stem), None)
        if kind is None:
            continue
        by_piece.setdefault(p, {})[kind] = rows

    comparison_pairs = [("performance", "baseline"), ("pro", "performance"), ("pro", "baseline")]
    for piece, kinds in by_piece.items():
        for a, b in comparison_pairs:
            if a in kinds and b in kinds:
                cmp_path = visuals_dir / f"{piece}__comparison_{a}_vs_{b}.png"
                plot_comparison(kinds[a], kinds[b], cmp_path, piece,
                                label_a=a, label_b=b,
                                color_a=ROLE_COLORS[a], color_b=ROLE_COLORS[b])
                print(f"[run] comparison ({a} vs {b}) -> {cmp_path}")
        if all(r in kinds for r in ("baseline", "performance", "pro")):
            cmp_all_path = visuals_dir / f"{piece}__comparison_all.png"
            plot_comparison_all({r: kinds[r] for r in ("baseline", "performance", "pro")},
                                cmp_all_path, piece, ROLE_COLORS)
            print(f"[run] comparison (all) -> {cmp_all_path}")

    print("\n[Done]")
