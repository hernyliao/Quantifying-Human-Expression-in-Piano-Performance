import csv
import json
import math
from pathlib import Path

import librosa
import sounddevice as sd
import webview

SCRIPT_DIR = Path(__file__).resolve().parent.parent

PIECE = "furelise"
ROLE = "baseline"
ISOLATOR = "sm"
CSV_PATH = SCRIPT_DIR / "files" / PIECE / "output" / "csv_files" / f"{PIECE}_{ROLE}_{ISOLATOR}.csv"
AUDIO_PATH = SCRIPT_DIR / "files" / PIECE / f"{PIECE}_{ROLE}.wav"
SCORE_XML_PATH = SCRIPT_DIR / "files" / PIECE / f"{PIECE}.xml"
VIEWER_HTML = SCRIPT_DIR / "viewer" / "score_viewer.html"

LOW_CONFIDENCE_THRESHOLD = 45.0
PLAYBACK_TAIL_PAD_SEC = 0.00


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def is_measure_csv(rows: list[dict]) -> bool:
    return bool(rows) and "measure_index" in rows[0]


def _safe_float(v, default=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class Api:
    """Exposed to the JS side via pywebview.api."""

    def __init__(self):
        self.segments = []   # list of {"start_sec", "end_sec"}
        self.y = None
        self.sr = None

    def load_session(self):
        rows = load_rows(CSV_PATH)
        y, sr = librosa.load(str(AUDIO_PATH), sr=None, mono=True)
        self.y = y
        self.sr = sr
        self.segments = []
        out_rows = []

        if is_measure_csv(rows):
            session_type = "measure"
            for r in rows:
                measure = int(r["measure_index"])
                note_count = int(r.get("note_count") or 0)
                dom = r.get("recorded_dominant_pitch") or "?"
                conf = _safe_float(r["confidence"])
                start = _safe_float(r["start_sec"], 0.0)
                end = _safe_float(r["end_sec"], 0.0)
                length = _safe_float(r.get("measure_length_sec"), 0.0)
                low_conf = (not math.isnan(conf)) and conf < LOW_CONFIDENCE_THRESHOLD
                info = f"{note_count} notes · {length:.2f}s · dom {dom} · conf {conf:.0f}"
                self.segments.append({"start_sec": start, "end_sec": end})
                out_rows.append({"measure": measure, "info": info, "low_conf": low_conf})
        else:
            session_type = "note"

            # Pass 1: measure timing spans, derived from each note's expected_start_sec
            measure_spans: dict[int, dict[str, float]] = {}
            for r in rows:
                m = int(r.get("measure") or 1)
                s = _safe_float(r.get("expected_start_sec"), 0.0)
                if m not in measure_spans:
                    measure_spans[m] = {"start": s, "end": s}
                else:
                    measure_spans[m]["start"] = min(measure_spans[m]["start"], s)
                    measure_spans[m]["end"] = max(measure_spans[m]["end"], s)
            # End of measure M = start of measure M+1 (fall back to last seen end for the final measure)
            sorted_ms = sorted(measure_spans.keys())
            for i, m in enumerate(sorted_ms):
                if i + 1 < len(sorted_ms):
                    measure_spans[m]["end"] = measure_spans[sorted_ms[i + 1]]["start"]

            prev_measure = None
            note_in_measure = 0
            for r in rows:
                note_index = int(r["note_index"])
                measure = int(r.get("measure") or 1)
                if measure != prev_measure:
                    note_in_measure = 0
                    prev_measure = measure
                note_in_measure += 1

                exp = r["expected_pitch"]
                rec = r["recorded_pitch"] or "?"
                conf = _safe_float(r["confidence"])
                start = _safe_float(r["start_sec"], 0.0)
                end = _safe_float(r["end_sec"], 0.0)
                exp_start = _safe_float(r.get("expected_start_sec"), 0.0)
                low_conf = (not math.isnan(conf)) and conf < LOW_CONFIDENCE_THRESHOLD

                span = measure_spans.get(measure, {"start": exp_start, "end": exp_start})
                span_len = span["end"] - span["start"]
                if span_len > 1e-6:
                    frac = (exp_start - span["start"]) / span_len
                    frac = max(0.0, min(0.999, frac))
                else:
                    frac = 0.0

                info = f"n{note_index} · exp {exp} / rec {rec} · conf {conf:.0f}"
                self.segments.append({"start_sec": start, "end_sec": end})
                out_rows.append({
                    "measure": measure,
                    "note_in_measure": note_in_measure,
                    "frac_in_measure": round(frac, 4),
                    "info": info,
                    "low_conf": low_conf,
                })

        print(f"[api] loaded {len(out_rows)} {session_type} rows")
        return {
            "rows": out_rows,
            "type": session_type,
            "title": f"{PIECE} — {ROLE} ({ISOLATOR})",
        }

    def play_row(self, idx):
        """Start non-blocking playback. Returns segment duration in seconds (0 on failure)."""
        if self.y is None or idx < 0 or idx >= len(self.segments):
            return 0.0
        seg = self.segments[idx]
        sd.stop()
        start_sample = int(seg["start_sec"] * self.sr)
        end_sample = min(len(self.y), int((seg["end_sec"] + PLAYBACK_TAIL_PAD_SEC) * self.sr))
        if end_sample <= start_sample:
            return 0.0
        sd.play(self.y[start_sample:end_sample], samplerate=self.sr)
        return (end_sample - start_sample) / float(self.sr)

    def stop(self):
        sd.stop()
        return True


def js_call(window, fn: str, *args):
    args_json = ", ".join(json.dumps(a) for a in args)
    window.evaluate_js(f"{fn}({args_json})")


def run(window):
    title = f"{PIECE} — {ROLE} ({ISOLATOR})"
    print(f"Loading score: {SCORE_XML_PATH}")
    xml_content = Path(SCORE_XML_PATH).read_text(encoding="utf-8")
    js_call(window, "loadScore", xml_content, title)
    print("Score sent to viewer. Use the window to control playback.")
    print("  Space/Enter  play current & advance")
    print("  ← / →        prev / next (no audio)")
    print("  R            replay current")
    print("  Esc          stop audio")


def main():
    api = Api()
    window = webview.create_window(
        f"{PIECE} — Score Viewer",
        str(VIEWER_HTML),
        js_api=api,
        width=1400,
        height=900,
    )
    webview.start(run, window)


if __name__ == "__main__":
    main()
