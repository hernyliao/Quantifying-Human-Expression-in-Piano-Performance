'''
note_isolator_sm.py

Sequential-matching note isolator with chord-aware event grouping. Score
notes that share a start_beat (chords / two-hand strikes) are treated as a
single event and matched to one audio position; per-note features are then
extracted from each note's own CQT band so chord members get individually
meaningful volume/attack/sustain/decay.
'''

from dataclasses import dataclass
from typing import Optional

import librosa
import numpy as np

from note_isolator_dtw import XmlNote, parse_xml_dtw
from note_isolator_obo import (
    DetectedOnset,
    MatchedNote,
    ScoreMetadata,
    freq_to_note_name,
    sec_to_note_name,
    analyze_audio,
    get_dominant_pitch,
    get_energy_spike,
    write_csv,
    band_volume,
    band_presence,
    band_energy_spike,
    compute_band_dynamic_shape,
)

TEMPLATE_SR = 22050
HOP_LENGTH = 512

# Search window is bidirectional around the score-predicted time of each
# event. The half-window scales with the expected score gap so a triplet
# in a fast passage doesn't search a 1.5 s neighborhood that covers two
# or three subsequent events. Bounds keep the window sensible at both
# extremes (very fast runs vs. long sustained notes).
# MIN kept tight (0.06) so fast passages (250bpm eighths = 0.12s gap) don't
# blow the window wide enough to swallow the neighbouring note.
SEARCH_HALF_WINDOW_MIN_SEC = 0.06
SEARCH_HALF_WINDOW_MAX_SEC = 0.50
SEARCH_HALF_WINDOW_GAP_MULT = 0.75
SEARCH_HALF_WINDOW_GAP_CAP = 0.90  # half-window capped at this fraction of gap

# cursor advance — push the next event's search at least most of the way to
# the score-expected position, so consecutive notes can't collapse onto the
# same chord's decay
CURSOR_ADVANCE_RATIO = 0.60
MIN_CURSOR_ADVANCE = 0.04

# Scoring window for evaluating a candidate position. Caps at 0.20s for slow
# pieces (ballade-style), but in fast passages scales down to roughly the
# event gap so the window doesn't bleed into the next note's chroma.
ID_WINDOW_SEC_MAX = 0.20
ID_WINDOW_SEC_MIN = 0.05
ID_WINDOW_GAP_MULT = 0.85

# Silence/noise-floor gate for sweep-fallback candidates. Without this,
# silent frames look like high-confidence matches: chroma of near-silent
# audio normalises to a near-uniform vector that, dotted with a chord
# template, yields ~0.7 similarity — beating real onsets and trapping
# leading events in the pre-roll silence. Only sweep-generated positions
# are gated; positions from real detected onsets pass through unconditionally.
SILENCE_ONSET_STRENGTH_MIN = 0.02   # fraction of p95 onset envelope
SILENCE_GATE_PENALTY = 0.0          # score returned for silent positions

LOW_CONFIDENCE_THRESHOLD = 12.0
TOP_K_CANDIDATES = 3
MIN_NOTE_SEC = 0.04

# scoring weights — pyin gives a single dominant pitch which is meaningless
# for chord events, so chord-chroma + onset carry most of the weight. PROX
# penalises candidates that score well on chroma/onset but sit far from the
# score-projected position — without it, a chord-similar real onset belonging
# to the *next* event (the only one librosa flagged in a sparse arpeggio run)
# can win over a sweep position near the actual expected time, off-by-one'ing
# the rest of the passage.
W_CHROMA = 0.45
W_ONSET  = 0.15
W_PROX   = 0.40

NEAR_TOP_RATIO = 0.80 # tighter = stronger earliest-candidate bias

# Backtracking — when N event matches in a row all fall below the off-score,
# roll back to where that streak started and retry that event with its
# next-best candidate. Recovers from a wrong early pick that cascades through
# subsequent events. Toggle with BACKTRACK_ENABLED; tune the streak length to
# trade off responsiveness vs. churn (lower = more aggressive rollbacks).
BACKTRACK_ENABLED = True
BACKTRACK_STREAK_LEN = 3      # # of consecutive "off" events that trigger a rollback
BACKTRACK_OFF_SCORE = 25.0    # event score below this counts as "off"
BACKTRACK_MAX_PER_EVENT = 2   # cap retries anchored at the same event
BACKTRACK_MAX_TOTAL = 20      # global safety cap to prevent runaway loops

# parse_xml_local stays as the public entry point that main.py imports;
# delegate to the chord-aware DTW parser so SM sees chord members too.
def parse_xml_local(xml_path: str) -> tuple[list[XmlNote], ScoreMetadata]:
    return parse_xml_dtw(xml_path)

def synthesize_note_chroma(freq_hz: float, duration_sec: float) -> np.ndarray: # single-pitch chroma fingerprint (sine fundamental + harmonics)
    n = max(2, int(duration_sec * TEMPLATE_SR))
    t = np.arange(n, dtype=np.float32) / TEMPLATE_SR
    wave = np.zeros(n, dtype=np.float32)
    for harmonic, amp in [(1, 1.0), (2, 0.5), (3, 0.3), (4, 0.15), (5, 0.08)]:
        wave += amp * np.sin(2 * np.pi * freq_hz * harmonic * t)
    peak = float(np.max(np.abs(wave)))
    if peak > 0:
        wave /= peak
    attack = min(max(1, int(0.01 * TEMPLATE_SR)), n // 4)
    env = np.ones(n, dtype=np.float32)
    env[:attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    decay_len = n - attack
    if decay_len > 0:
        env[attack:] = np.exp(-3.0 * np.arange(decay_len, dtype=np.float32) / decay_len)
    wave = (wave * env).astype(np.float32)

    chroma = librosa.feature.chroma_cqt(y=wave, sr=TEMPLATE_SR, hop_length=HOP_LENGTH)
    if chroma.size == 0:
        return np.zeros(12, dtype=np.float32)
    template = chroma.mean(axis=1).astype(np.float32)
    norm = float(np.linalg.norm(template))
    return template / norm if norm > 0 else template

def event_composite_template(notes: list[XmlNote], duration_sec: float) -> np.ndarray: # sum of per-pitch chroma templates → fingerprint for the whole chord
    template = np.zeros(12, dtype=np.float32)
    for n in notes:
        template += synthesize_note_chroma(n.frequency_hz, duration_sec)
    norm = float(np.linalg.norm(template))
    return template / norm if norm > 0 else template

def compute_audio_features(audio_data: dict):
    y  = audio_data["y"]
    sr = audio_data["sr"]

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP_LENGTH)
    norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    chroma_norm = chroma / norms

    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=HOP_LENGTH, aggregate=np.median,
    )
    p95 = float(np.percentile(onset_env, 95)) if onset_env.size else 1.0
    onset_norm = onset_env / p95 if p95 > 0 else onset_env

    hop_sec = HOP_LENGTH / sr
    return chroma_norm, onset_norm, hop_sec

@dataclass
class ScoreEvent: # one beat-aligned group of simultaneous notes (chord / two-hand strike)
    start_beat: float
    notes: list[XmlNote]
    expected_gap_sec: float # score-time distance to the next event (or own duration if last)
    template: np.ndarray # composite chroma fingerprint

@dataclass
class Candidate:
    start_sec: float
    score: float

def group_into_events(xml_notes: list[XmlNote], bpm: float) -> list[ScoreEvent]:
    sec_per_beat = 60.0 / bpm if bpm > 0 else 0.5
    sorted_notes = sorted(xml_notes, key=lambda n: (n.start_beat, n.voice, n.index))

    # bucket by start_beat using a small float tolerance — XML divisions are
    # exact, but float arithmetic in parse can introduce ULP-level drift.
    events: list[ScoreEvent] = []
    cur_beat: Optional[float] = None
    cur_notes: list[XmlNote] = []
    for n in sorted_notes:
        if cur_beat is None or abs(n.start_beat - cur_beat) < 1e-6:
            cur_notes.append(n)
            cur_beat = n.start_beat if cur_beat is None else cur_beat
        else:
            events.append(ScoreEvent(cur_beat, cur_notes, 0.0, np.zeros(12, dtype=np.float32)))
            cur_notes = [n]
            cur_beat = n.start_beat
    if cur_notes:
        events.append(ScoreEvent(cur_beat, cur_notes, 0.0, np.zeros(12, dtype=np.float32)))

    for i, ev in enumerate(events):
        if i + 1 < len(events):
            gap_beats = events[i + 1].start_beat - ev.start_beat
        else:
            gap_beats = max(n.duration_beats for n in ev.notes)
        ev.expected_gap_sec = max(MIN_NOTE_SEC, gap_beats * sec_per_beat)
        ev.template = event_composite_template(ev.notes, ev.expected_gap_sec)

    return events

def _proximity(start_sec: float, expected_sec: float, half_window: float) -> float:
    if half_window <= 0:
        return 1.0
    return max(0.0, 1.0 - abs(start_sec - expected_sec) / half_window)

def score_event_position(template: np.ndarray, audio_chroma: np.ndarray,
                         start_frame: int, end_frame: int, onset_strength: float,
                         start_sec: float, expected_sec: float,
                         half_window: float) -> float:
    prox = _proximity(start_sec, expected_sec, half_window)
    if end_frame <= start_frame:
        return round(100.0 * W_PROX * prox, 1)
    mean = audio_chroma[:, start_frame:end_frame].mean(axis=1)
    norm = float(np.linalg.norm(mean))
    if norm == 0:
        chroma_sim = 0.0
    else:
        chroma_sim = float(np.clip(np.dot(template, mean / norm), 0.0, 1.0))
    onset = float(np.clip(onset_strength, 0.0, 1.0))
    return round(100.0 * (W_CHROMA * chroma_sim + W_ONSET * onset + W_PROX * prox), 1)

def find_event_candidates(event: ScoreEvent, audio_chroma: np.ndarray,
                          onset_env: np.ndarray, hop_sec: float,
                          cursor_sec: float, expected_sec: float,
                          audio_duration_sec: float,
                          onset_times: np.ndarray) -> list[Candidate]:
    # Bidirectional window centered on the score-predicted time, then
    # clipped forward of the cursor so we never re-pick a past onset.
    # Without this, fast passages drift: each event's search starts wherever
    # the cursor landed, and picking a slightly-late candidate cascades
    # because the next search starts from that already-late position.
    raw_hw = SEARCH_HALF_WINDOW_GAP_MULT * event.expected_gap_sec
    half_window = min(SEARCH_HALF_WINDOW_MAX_SEC,
                      max(SEARCH_HALF_WINDOW_MIN_SEC, raw_hw))
    if event.expected_gap_sec > 0:
        half_window = min(half_window, SEARCH_HALF_WINDOW_GAP_CAP * event.expected_gap_sec)

    # ID window also scales with gap so fast-passage scoring doesn't bleed into
    # the next event's chroma. Slow pieces keep the original 0.20s window.
    id_window_sec = float(np.clip(ID_WINDOW_GAP_MULT * event.expected_gap_sec,
                                  ID_WINDOW_SEC_MIN, ID_WINDOW_SEC_MAX))

    earliest = max(cursor_sec, expected_sec - half_window)
    latest = min(expected_sec + half_window,
                 max(earliest, audio_duration_sec - id_window_sec))
    if latest < earliest:
        latest = earliest

    window_frames = max(1, int(round(id_window_sec / hop_sec)))
    n_frames = audio_chroma.shape[1]

    in_range = onset_times[(onset_times >= earliest) & (onset_times <= latest)]
    # mark which candidate positions came from real detected onsets vs sweep —
    # real onsets bypass the silence gate; sweep positions don't.
    starts: list[tuple[float, bool]] = [(float(t), True) for t in in_range]

    # Always also generate sweep positions across the window. Previously
    # sweep was a *fallback* only fired when no real onset was in range;
    # but in sparsely-detected dense runs (rach arpeggios) the single real
    # onset present often belongs to the *next* event, and being the only
    # candidate it wins by default. With sweep always present, the
    # proximity term in score_event_position lets a near-expected sweep
    # outscore a far-from-expected real onset that scored well on chroma
    # for the wrong reason.
    sf_lo = int(np.floor(earliest / hop_sec))
    sf_hi = int(np.ceil(latest / hop_sec))
    step = max(1, window_frames // 4)
    for sf in range(sf_lo, sf_hi + 1, step):
        starts.append((sf * hop_sec, False))

    candidates: list[Candidate] = []
    for start_sec, is_real_onset in starts:
        sf = int(round(start_sec / hop_sec))
        if sf < 0 or sf >= n_frames:
            continue
        ef = min(sf + window_frames, n_frames)
        if ef <= sf:
            continue
        onset_at_start = float(onset_env[sf]) if sf < len(onset_env) else 0.0
        # Silence gate: a sweep position with effectively no audio energy
        # would otherwise pick up a high "fake" chroma score off the
        # near-uniform chroma of silence (this is what stranded the first
        # 3 scale events in pre-roll silence and cascaded the whole match).
        # Gated candidates skip chroma/onset (no signal) but still receive
        # the proximity component — that gives projection-to-expected
        # behaviour when no real onset is in the window, without giving
        # silent pre-roll a free chroma win (it's far from expected once
        # match_events anchors event 0 to first_onset_sec).
        if (not is_real_onset) and onset_at_start < SILENCE_ONSET_STRENGTH_MIN:
            prox = _proximity(start_sec, expected_sec, half_window)
            candidates.append(Candidate(start_sec=float(start_sec),
                                        score=round(100.0 * W_PROX * prox, 1)))
            continue
        score = score_event_position(event.template, audio_chroma, sf, ef,
                                      onset_at_start, start_sec, expected_sec,
                                      half_window)
        candidates.append(Candidate(start_sec=float(start_sec), score=score))

    # Primary: highest score wins. Secondary tiebreak: closest to the
    # score's expected_sec. The secondary key matters most for the score=0
    # tied group produced by sweep-fallback when no real onset is in the
    # window — without it, Python's stable sort keeps iteration order
    # (earliest first), so cands[0] lands at last_match_sec + 0.6*gap and
    # the next event's anchor is dragged early, compounding into seconds
    # of backward drift on rach's dense arpeggios. Sorting by |start -
    # expected| within ties means score-0 picks fall at score-projected
    # positions instead, breaking the drift loop.
    candidates.sort(key=lambda c: (-c.score, abs(c.start_sec - expected_sec)))
    if candidates:
        top = candidates[0].score
        threshold = max(top * NEAR_TOP_RATIO, LOW_CONFIDENCE_THRESHOLD * 0.6)
        near_top = [c for c in candidates if c.score >= threshold]
        near_top.sort(key=lambda c: c.start_sec)
        rest = [c for c in candidates if c.score < threshold]
        candidates = near_top + rest
    return candidates[:TOP_K_CANDIDATES]

def match_events(events: list[ScoreEvent], audio_chroma: np.ndarray,
                 onset_env: np.ndarray, hop_sec: float,
                 audio_data: dict, onset_times: np.ndarray,
                 first_onset_sec: Optional[float] = None) -> list[Candidate]:
    duration_sec_audio = audio_data["duration_sec"]
    n_events = len(events)

    # Greedy-forward pass with optional backtracking. Each event's search is
    # centered on its score-predicted time, extrapolated from the last matched
    # event plus cumulative expected_gap_sec. Candidate lists are cached per
    # event so a rollback can advance to the next-best pick without re-searching.
    event_candidates: list[Optional[list[Candidate]]] = [None] * n_events
    event_chosen_idx: list[int] = [-1] * n_events
    accepted: list[Candidate] = [Candidate(float("nan"), 0.0) for _ in range(n_events)]
    backtrack_attempts: dict[int, int] = {}

    cursor_sec = 0.0
    last_match_sec: Optional[float] = None
    last_match_idx = -1
    streak_start: Optional[int] = None
    streak_len = 0
    n_backtracks = 0

    idx = 0
    while idx < n_events:
        if last_match_sec is not None:
            cum_gap = sum(events[j].expected_gap_sec for j in range(last_match_idx, idx))
            expected_sec = last_match_sec + cum_gap
        elif first_onset_sec is not None:
            # Before any match is anchored, predict event 0 lands at the first
            # detected audio onset rather than at audio t=0. Skips leading
            # silence so event 0 doesn't fake-match silent frames and cascade
            # an N-event offset through the whole piece.
            expected_sec = first_onset_sec
        else:
            expected_sec = cursor_sec

        # First visit: run the search. Re-visit (after rollback): step to the
        # next-best cached candidate instead of researching.
        if event_candidates[idx] is None:
            event_candidates[idx] = find_event_candidates(
                events[idx], audio_chroma, onset_env, hop_sec,
                cursor_sec, expected_sec, duration_sec_audio, onset_times,
            )
            event_chosen_idx[idx] = 0
        else:
            event_chosen_idx[idx] += 1

        cands = event_candidates[idx]
        chose_i = event_chosen_idx[idx]
        chosen = cands[chose_i] if cands and 0 <= chose_i < len(cands) else None

        advance = max(MIN_CURSOR_ADVANCE,
                      CURSOR_ADVANCE_RATIO * events[idx].expected_gap_sec)

        if chosen is None:
            accepted[idx] = Candidate(float("nan"), 0.0)
            cursor_sec = cursor_sec + advance
        else:
            accepted[idx] = chosen
            last_match_sec = chosen.start_sec
            last_match_idx = idx
            cursor_sec = chosen.start_sec + advance

        # streak tracking: missing match or score below threshold counts as "off"
        is_off = chosen is None or chosen.score < BACKTRACK_OFF_SCORE
        if is_off:
            if streak_start is None:
                streak_start = idx
            streak_len += 1
        else:
            streak_start = None
            streak_len = 0

        # rollback when the streak is long enough AND the streak-start event
        # has another candidate to try AND we haven't blown the retry budget
        if (BACKTRACK_ENABLED
                and streak_start is not None
                and streak_len >= BACKTRACK_STREAK_LEN
                and n_backtracks < BACKTRACK_MAX_TOTAL):
            target = streak_start
            target_cands = event_candidates[target]
            attempts = backtrack_attempts.get(target, 0)
            has_more = (target_cands is not None
                        and event_chosen_idx[target] + 1 < len(target_cands))

            if attempts < BACKTRACK_MAX_PER_EVENT and has_more:
                prev_streak = streak_len
                backtrack_attempts[target] = attempts + 1
                n_backtracks += 1

                # restore anchor to the last good match BEFORE the streak
                prev_good = None
                for j in range(target - 1, -1, -1):
                    if not np.isnan(accepted[j].start_sec):
                        prev_good = j
                        break
                if prev_good is not None:
                    last_match_sec = accepted[prev_good].start_sec
                    last_match_idx = prev_good
                    cursor_sec = (last_match_sec
                                  + max(MIN_CURSOR_ADVANCE,
                                        CURSOR_ADVANCE_RATIO * events[prev_good].expected_gap_sec))
                else:
                    last_match_sec = None
                    last_match_idx = -1
                    cursor_sec = 0.0

                # forward events lose their cached search — they'll re-search
                # against the new anchor on the next pass
                for j in range(target + 1, idx + 1):
                    event_candidates[j] = None
                    event_chosen_idx[j] = -1
                    accepted[j] = Candidate(float("nan"), 0.0)

                streak_start = None
                streak_len = 0
                next_pick = event_chosen_idx[target] + 2  # 1-indexed for display
                print(f"[match_events] backtrack #{n_backtracks}: "
                      f"{prev_streak}-deep streak below {BACKTRACK_OFF_SCORE} starting at event {target}; "
                      f"retrying event {target} with candidate #{next_pick}")
                idx = target
                continue

        idx += 1

    matched_count = sum(1 for c in accepted if not np.isnan(c.start_sec))
    mode = "with backtracking" if BACKTRACK_ENABLED else "greedy-forward only"
    print(f"[match_events] {mode} — matched {matched_count}/{n_events} events  "
          f"|  backtracks: {n_backtracks}")
    return accepted

def build_pseudo_onset(idx: int, start_sec: float, end_sec: float, audio_data: dict,
                       expected_freq: float) -> Optional[DetectedOnset]:
    if np.isnan(start_sec):
        return None
    f0 = audio_data["f0"]
    f0_times = audio_data["f0_times"]
    voiced_prob = audio_data["voiced_prob"]
    rms = audio_data["rms"]
    rms_times = audio_data["rms_times"]

    safe_end = end_sec if not np.isnan(end_sec) and end_sec > start_sec else start_sec + ID_WINDOW_SEC_MAX
    ioi = safe_end - start_sec
    pitch_win = max(0.05, min(0.8 * ioi, 0.8))
    pitch_win_end = min(safe_end, start_sec + pitch_win)
    rec_freq, conf = get_dominant_pitch(f0, voiced_prob, f0_times, start_sec, pitch_win_end)
    # band-localised spike for this pitch's own attack
    spike = band_energy_spike(audio_data, expected_freq, start_sec) if expected_freq > 0 else get_energy_spike(rms, rms_times, start_sec)
    rmask = (rms_times >= start_sec) & (rms_times < safe_end)
    mean_rms = float(np.mean(rms[rmask])) if np.any(rmask) else 0.0

    return DetectedOnset(
        index=idx,
        time_sec=start_sec,
        pitch=freq_to_note_name(rec_freq) if rec_freq > 0 else "?",
        frequency_hz=rec_freq,
        pitch_confidence=conf,
        energy_spike=spike,
        rms=mean_rms,
    )

def _sum_skip_nan(vals: list[float]) -> float: # sum treating NaN as missing; returns NaN only if every value is NaN
    clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
    return round(float(np.sum(clean)), 4) if clean else float("nan")

def assemble_matches(events: list[ScoreEvent], event_matches: list[Candidate],
                     audio_data: dict, bpm: float) -> list[MatchedNote]:
    duration_sec_audio = audio_data["duration_sec"]

    # event_end_sec = next matched event's start, capped at start + this
    # event's score-expected gap. Without the cap, if the next match drifts
    # late, this event's window bleeds into the next attack — playback
    # captures two notes and the feature-extraction window (sustain/decay)
    # picks up contamination from the next pitch. The cap is score-derived
    # (per-event expected_gap_sec), so it scales naturally with tempo.
    event_end_sec: list[float] = []
    for i, cand in enumerate(event_matches):
        if np.isnan(cand.start_sec):
            event_end_sec.append(float("nan"))
            continue
        nxt = float("nan")
        for j in range(i + 1, len(event_matches)):
            if not np.isnan(event_matches[j].start_sec):
                nxt = event_matches[j].start_sec
                break
        natural_end = nxt if not np.isnan(nxt) else duration_sec_audio
        score_end = cand.start_sec + events[i].expected_gap_sec
        event_end_sec.append(min(natural_end, score_end))

    sec_per_beat = 60.0 / bpm
    anchor_start_sec = next((c.start_sec for c in event_matches if not np.isnan(c.start_sec)), 0.0)
    anchor_beat = events[0].start_beat if events else 0.0

    # One row per event: chord pitches join with '+', band features sum across
    # the event's notes so volume reflects total chord energy.
    matched: list[MatchedNote] = []
    for ev_idx, (ev, cand) in enumerate(zip(events, event_matches)):
        start_sec = cand.start_sec
        end_sec = event_end_sec[ev_idx]
        score = cand.score

        rep_note = XmlNote(
            index=ev_idx,
            pitch="+".join(n.pitch for n in ev.notes),
            frequency_hz=min(n.frequency_hz for n in ev.notes),
            duration_beats=max(n.duration_beats for n in ev.notes),
            start_beat=ev.start_beat,
            measure=min(n.measure for n in ev.notes),
            voice=min(n.voice for n in ev.notes),
        )

        if not np.isnan(start_sec) and not np.isnan(end_sec):
            note_length_sec = end_sec - start_sec
        else:
            note_length_sec = float("nan")

        onset = build_pseudo_onset(ev_idx, start_sec, end_sec, audio_data, rep_note.frequency_hz)
        expected_start = anchor_start_sec + (ev.start_beat - anchor_beat) * sec_per_beat

        if onset is not None and not np.isnan(start_sec):
            note_name = sec_to_note_name(note_length_sec, bpm) if note_length_sec > 0 else "unknown"
            volume = round(sum(band_volume(audio_data, n.frequency_hz, start_sec, end_sec)
                               for n in ev.notes), 4)
            pitch_presence = round(sum(band_presence(audio_data, n.frequency_hz, start_sec, end_sec)
                                       for n in ev.notes), 4)
            shape_vals = [compute_band_dynamic_shape(start_sec, end_sec, audio_data, n.frequency_hz)
                          for n in ev.notes]
            attack_v  = _sum_skip_nan([s[0] for s in shape_vals])
            sustain_v = _sum_skip_nan([s[1] for s in shape_vals])
            decay_v   = _sum_skip_nan([s[2] for s in shape_vals])
            energy_spike = _sum_skip_nan([band_energy_spike(audio_data, n.frequency_hz, start_sec)
                                          for n in ev.notes])
            timing_offset = round(start_sec - expected_start, 4)
            # blend chord-level chroma score with the event's summed band presence
            event_conf = round(0.5 * score + 0.5 * min(pitch_presence, 1.0) * 100.0, 1)
        else:
            note_name = "unknown"
            volume = 0.0
            attack_v = sustain_v = decay_v = float("nan")
            pitch_presence = float("nan")
            energy_spike = float("nan")
            timing_offset = float("nan")
            event_conf = 0.0

        matched.append(MatchedNote(
            xml_note=rep_note,
            onset=onset,
            start_sec=round(start_sec, 4) if not np.isnan(start_sec) else float("nan"),
            end_sec=round(end_sec, 4) if not np.isnan(end_sec) else float("nan"),
            note_length_sec=round(note_length_sec, 4) if not np.isnan(note_length_sec) else float("nan"),
            note_name=note_name,
            volume=volume,
            confidence=event_conf,
            expected_start_sec=round(expected_start, 4),
            timing_offset_sec=timing_offset,
            attack_volume=attack_v,
            sustain_volume=sustain_v,
            decay_volume=decay_v,
            energy_spike=energy_spike,
            pitch_confidence=pitch_presence,
        ))

    # IOI / gap  between consecutive events
    for i, m in enumerate(matched):
        if np.isnan(m.start_sec) or i + 1 >= len(matched):
            continue
        next_start = float("nan")
        for j in range(i + 1, len(matched)):
            if not np.isnan(matched[j].start_sec):
                next_start = matched[j].start_sec
                break
        if np.isnan(next_start):
            continue
        ioi = next_start - m.start_sec
        m.inter_onset_interval_sec = round(ioi, 4)
        m.gap_to_next_sec = round(next_start - m.end_sec, 4) if not np.isnan(m.end_sec) else float("nan")

    return matched

def process_recording_sm(xml_notes: list[XmlNote], score_meta: ScoreMetadata, audio_path: str, output_path: str) -> list[MatchedNote]:
    print(f"[process_recording_sm] processing {audio_path}")
    audio_data = analyze_audio(audio_path)

    # Use raw librosa onsets (not clean_onsets) — sm scores candidates with
    # chord-aware chroma, so pyin's per-onset pitch_confidence is the wrong
    # filter for polyphonic audio and throws away most legitimate chord
    # attacks. With too few candidates, multiple score events collapse onto
    # the same audio position and cascade.
    onset_times = np.asarray(audio_data["onset_times"], dtype=np.float64)
    print(f"[process_recording_sm] using {len(onset_times)} raw onsets as candidate positions")

    # chord-aware BPM: align first-onset to last-note-start so we measure
    # beats actually elapsed in audio, not start+duration (which over-counts
    # the trailing sustain when there's no onset for it) and not total span
    # (which over-counts when the audio has leading/trailing silence).
    if xml_notes:
        first_start = min(n.start_beat for n in xml_notes)
        last_start = max(n.start_beat for n in xml_notes)
        beats_between_starts = last_start - first_start
        if onset_times.size >= 2 and beats_between_starts > 0:
            audio_span = float(onset_times[-1] - onset_times[0])
            bpm = round(60.0 * beats_between_starts / audio_span, 1) if audio_span > 0 else 120.0
        else:
            total_beats = max(n.start_beat + n.duration_beats for n in xml_notes)
            bpm = round(60.0 * total_beats / audio_data["duration_sec"], 1) if audio_data["duration_sec"] > 0 else 120.0
    else:
        bpm = 120.0
    print(f"[process_recording_sm] estimated BPM: {bpm}")

    print("[process_recording_sm] computing audio chroma + onset envelope ...")
    audio_chroma, onset_env, hop_sec = compute_audio_features(audio_data)

    print("[process_recording_sm] grouping score into chord events ...")
    events = group_into_events(xml_notes, bpm)
    print(f"[process_recording_sm] {len(events)} events for {len(xml_notes)} notes")

    # First-onset anchor for event 0 — see match_events docstring.
    first_onset_sec = float(onset_times[0]) if onset_times.size else None

    mode = "with backtracking" if BACKTRACK_ENABLED else "greedy-forward"
    print(f"[process_recording_sm] running event-level matcher ({mode}) ...")
    event_matches = match_events(events, audio_chroma, onset_env, hop_sec,
                                 audio_data, onset_times,
                                 first_onset_sec=first_onset_sec)

    matched = assemble_matches(events, event_matches, audio_data, bpm)
    matched_count = sum(1 for m in matched if not np.isnan(m.start_sec))
    print(f"[process_recording_sm] {matched_count}/{len(events)} event rows have a start_sec")

    write_csv(matched, output_path)
    return matched

if __name__ == "__main__":
    xmlnotes, meta = parse_xml_local("files/furelise/furelise.xml")
    process_recording_sm(xmlnotes, meta, "files/furelise/furelise_baseline.wav", "files/furelise/output/csv_files/furelise_baseline_sm.csv")