'''
note_isolator_obo.py

Parses a MusicXML file and an audio recording, matches each expected note
to a detected onset using obo scoring, and writes a per-note CSV.

Entry points (called by main.py):
    parse_musicxml(xml_path) -> (xml_notes, score_meta)
    process_recording(xml_notes, score_meta, audio_path, output_path)
'''

import csv
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

try:
    import sounddevice as sd
except Exception:
    sd = None

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "csv_files"

'''Comparing XmlNote to DetectedOnset --> MatchedNote'''

@dataclass
class XmlNote: # note parsed from xml file
    index: int
    pitch: str # e.g. "C4", "D#4", or "rest"
    frequency_hz: float # theoretical/expected frequency
    duration_beats: float # in quarter notes
    measure: int
    is_rest: bool = False

@dataclass
class DetectedOnset: # single onset detected in the audio.
    index: int
    time_sec: float # duration
    pitch: str
    frequency_hz: float
    pitch_confidence: float # pyin confidence 0.0–1.0
    energy_spike: float # relative RMS spike at onset 0.0–1.0
    rms: float # mean RMS in the note window

@dataclass
class MatchedNote: # matching an XmlNote to a DetectedOnset.
    xml_note: XmlNote
    onset: Optional[DetectedOnset]
    start_sec: float
    end_sec: float
    note_length_sec: float
    note_name: str # nearest musical duration e.g. "quarter"
    volume: float # mean RMS over window, normalised 0–1
    confidence: float # obo 0–100

    # timing relative to score (rubato + micro-timing)
    expected_start_sec: float = float("nan") # score-predicted start using estimated BPM, anchored on first matched onset
    timing_offset_sec: float = float("nan")  # actual start - expected start; + => late, - => early

    # phrasing (breath / touch)
    inter_onset_interval_sec: float = float("nan") # this start -> next matched start
    gap_to_next_sec: float = float("nan")          # this end   -> next matched start (silence)

    # intra-note dynamic shaping (attack -> sustain -> decay curve)
    attack_volume: float = float("nan")  # peak RMS in first ~100ms (or 30% of note), normalised 0–1
    sustain_volume: float = float("nan") # mean RMS in middle portion, normalised 0–1
    decay_volume: float = float("nan")   # mean RMS in last 25% of note, normalised 0–1
    # touch quality
    energy_spike: float = float("nan")      # attack sharpness 0–1 from onset
    pitch_confidence: float = float("nan")  # pyin confidence 0–1 within note window

@dataclass
class ScoreMetadata: # Score-level info extracted from MusicXML.
    time_sig_num: int = 4          # numerator (e.g., 3 in 3/4)
    time_sig_den: int = 4          # denominator (e.g., 8 in 6/8)
    total_beats: float = 0.0       # total piece length in quarter notes
    measure_count: int = 0         # total measures parsed

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def midi_to_freq(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))

def note_to_midi(step: str, octave: int, alter: float = 0.0) -> int: # Convert MusicXML pitch components to a MIDI note number.
    semitone = NOTE_NAMES.index(step.upper())
    return 12 * (octave + 1) + semitone + int(alter)

def midi_to_note_name(midi: int) -> str: # e.g. 'C4' for MIDI 60.
    octave = (midi // 12) - 1
    name = NOTE_NAMES[midi % 12]
    return f"{name}{octave}"

def freq_to_midi(freq: float) -> float: # convert Hz to (possibly fractional) MIDI note number.
    return 69 + 12 * np.log2(freq / 440.0)

def freq_to_note_name(freq: float) -> str: # nearest note name for a frequency.
    midi = int(round(freq_to_midi(freq)))
    return midi_to_note_name(midi)

def semitone_distance(freq_a: float, freq_b: float) -> float: # absolute semitone distance between two frequencies.
    return abs(freq_to_midi(freq_a) - freq_to_midi(freq_b))

DURATION_MAP = [
    (4.0,  "whole"),
    (3.0,  "dotted-half"),
    (2.0,  "half"),
    (1.5,  "dotted-quarter"),
    (1.0,  "quarter"),
    (0.75, "dotted-eighth"),
    (0.5,  "eighth"),
    (0.25, "sixteenth"),
]

def sec_to_note_name(duration_sec: float, bpm: float) -> str: # convert a duration in seconds to the nearest musical name from DURATION_MAP, given the tempo in BPM.
    beats = duration_sec * bpm / 60.0
    closest = min(DURATION_MAP, key=lambda x: abs(x[0] - beats))
    return closest[1]

def parse_xml(xml_path: str) -> list[XmlNote]: # xml file --> ordered list of XmlNote objects.

    tree = ET.parse(xml_path)
    root = tree.getroot()

    def strip_ns(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    notes = []
    note_index = 0

    metadata = ScoreMetadata()

    # Pre-scan for <divisions> — some MusicXML exports put it after the first
    # measure's notes; default 1 inflates those notes' durations and breaks BPM.
    initial_divisions = None
    for elem in root.iter():
        if strip_ns(elem.tag) == "divisions":
            initial_divisions = int(elem.text)
            break

    for part in root.iter():
        if strip_ns(part.tag) != "part":
            continue
        divisions_per_quarter = initial_divisions if initial_divisions is not None else 1
        measure_number = 0

        for measure in part:
            if strip_ns(measure.tag) != "measure":
                continue
            measure_number += 1

            for child in measure:
                tag = strip_ns(child.tag)

                if tag == "attributes":
                    for attr_child in child:
                        if strip_ns(attr_child.tag) == "divisions":
                            divisions_per_quarter = int(attr_child.text)
                        elif strip_ns(attr_child.tag) == "time":
                            # Time signature — only set if not already set (first encountered)
                            beats_el     = next((c for c in attr_child if strip_ns(c.tag) == "beats"),     None)
                            beat_type_el = next((c for c in attr_child if strip_ns(c.tag) == "beat-type"), None)
                            if beats_el is not None:
                                metadata.time_sig_num = int(beats_el.text)
                            if beat_type_el is not None:
                                metadata.time_sig_den = int(beat_type_el.text)

                elif tag == "note":
                    # skip chord notes — melody only
                    is_chord = any(strip_ns(c.tag) == "chord" for c in child)

                    if is_chord:
                        continue

                    dur_el = next((c for c in child if strip_ns(c.tag) == "duration"), None)
                    duration_divisions = int(dur_el.text) if dur_el is not None else divisions_per_quarter
                    duration_beats = duration_divisions / divisions_per_quarter

                    # skip all rests — no audio
                    rest_el = next((c for c in child if strip_ns(c.tag) == "rest"), None)

                    if rest_el is not None:
                        continue

                    # pitched note
                    pitch_el = next((c for c in child if strip_ns(c.tag) == "pitch"), None)

                    if pitch_el is None:
                        continue

                    step = next((c.text for c in pitch_el if strip_ns(c.tag) == "step"),   "C")
                    octave = int(next((c.text for c in pitch_el if strip_ns(c.tag) == "octave"), "4"))
                    alter = float(next((c.text for c in pitch_el if strip_ns(c.tag) == "alter"), "0"))

                    midi = note_to_midi(step, octave, alter)
                    freq = midi_to_freq(midi)
                    pitch_name = midi_to_note_name(midi)

                    notes.append(XmlNote(
                        index=note_index,
                        pitch=pitch_name,
                        frequency_hz=freq,
                        duration_beats=duration_beats,
                        measure=measure_number,
                        is_rest=False,
                    ))

                    note_index += 1

    metadata.total_beats   = sum(n.duration_beats for n in notes)
    metadata.measure_count = notes[-1].measure if notes else 0

    print(f"[parse_xml] Parsed {len(notes)} pitched notes  |  "
          f"Time signature: {metadata.time_sig_num}/{metadata.time_sig_den}  |  "
          f"Total: {metadata.total_beats:.1f} quarter-note beats over {metadata.measure_count} measures")
    return notes, metadata

'''Audio Analysis'''

def analyze_audio(audio_path: str): # pyin pitch detection (per-frame pitch + confidence), onset detection (high sensitivity), RMS energy envelope
    print(f"[analyze_audio] Loading {audio_path} ...")
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration_sec = len(y) / sr
    print(f"[analyze_audio] Duration: {duration_sec:.2f}s  SR: {sr} Hz")

    # Pitch detection via pyin — covers full piano range A0–C8
    print("[analyze_audio] Running pyin pitch detection ...")
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("A0"),
        fmax=librosa.note_to_hz("C8"),
        sr=sr,
        frame_length=2048,
    )
    hop_length = 512
    times      = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    f0_voiced  = np.where(voiced_flag, f0, np.nan)

    # RMS energy envelope
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    rms_times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

    # Onset detection — high sensitivity to over-detect rather than under-detect.
    # Spurious onsets are filtered by scoring; missed onsets break alignment permanently.
    print("[analyze_audio] Running onset detection ...")
    onset_envelope = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=hop_length,
        aggregate=np.median, # median more robust than mean for piano
    )
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sr,
        hop_length=hop_length,
        backtrack=True, # snap back to true attack
        units="frames",
        delta=0.03, # lower than default 0.07 — catches soft notes
        wait=8, # minimum frames between onsets (~23ms at 512 hop)
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    print(f"[analyze_audio] Detected {len(onset_times)} onsets")

    # Per-MIDI-pitch CQT magnitude — one bin per semitone across the full piano
    # range so chord members can be measured independently. cqt_mag[i, t] is the
    # energy at MIDI note (CQT_MIDI_OFFSET + i) at frame t.
    print("[analyze_audio] Computing per-pitch CQT envelope ...")
    cqt_mag = np.abs(librosa.cqt(
        y, sr=sr, hop_length=hop_length,
        fmin=librosa.note_to_hz("A0"),
        n_bins=88, bins_per_octave=12,
    )).astype(np.float32)
    cqt_times = librosa.frames_to_time(np.arange(cqt_mag.shape[1]), sr=sr, hop_length=hop_length)
    # per-bin peak across the whole piece for normalisation
    band_peak = cqt_mag.max(axis=1)
    band_peak[band_peak <= 0] = 1.0

    return {
        "y": y,
        "sr": sr,
        "duration_sec": duration_sec,
        "f0": f0_voiced,
        "f0_times": times,
        "voiced_prob": voiced_prob,
        "rms": rms,
        "rms_times": rms_times,
        "onset_frames": onset_frames,
        "onset_times": onset_times,
        "hop_length": hop_length,
        "cqt_mag": cqt_mag,
        "cqt_times": cqt_times,
        "band_peak": band_peak,
    }

# CQT bin 0 corresponds to MIDI 21 (A0); bin 87 to MIDI 108 (C8).
CQT_MIDI_OFFSET = 21
CQT_N_BINS = 88

def freq_to_cqt_bin(freq_hz: float) -> int:
    if freq_hz <= 0:
        return 0
    midi = int(round(freq_to_midi(freq_hz)))
    return int(np.clip(midi - CQT_MIDI_OFFSET, 0, CQT_N_BINS - 1))

def band_envelope(audio_data: dict, freq_hz: float, start_sec: float, end_sec: float): # CQT magnitude slice for a single pitch within a time window — empty if window is invalid
    if np.isnan(start_sec) or np.isnan(end_sec) or end_sec <= start_sec:
        return np.array([], dtype=np.float32), 1.0
    cqt = audio_data["cqt_mag"]
    times = audio_data["cqt_times"]
    bin_idx = freq_to_cqt_bin(freq_hz)
    mask = (times >= start_sec) & (times < end_sec)
    if not np.any(mask):
        return np.array([], dtype=np.float32), float(audio_data["band_peak"][bin_idx])
    return cqt[bin_idx, mask], float(audio_data["band_peak"][bin_idx])

def band_volume(audio_data: dict, freq_hz: float, start_sec: float, end_sec: float) -> float: # mean band energy in window, normalised to per-bin peak
    env, peak = band_envelope(audio_data, freq_hz, start_sec, end_sec)
    if env.size == 0 or peak <= 0:
        return 0.0
    return round(float(env.mean()) / peak, 4)

def band_peak_volume(audio_data: dict, freq_hz: float, start_sec: float, end_sec: float) -> float:
    env, peak = band_envelope(audio_data, freq_hz, start_sec, end_sec)
    if env.size == 0 or peak <= 0:
        return 0.0
    return round(float(env.max()) / peak, 4)

def compute_band_dynamic_shape(start: float, end: float, audio_data: dict, freq_hz: float):
    # Per-pitch attack / sustain / decay using the CQT band for just this note's
    # fundamental, so chord members get individually meaningful envelopes
    # instead of all sharing the broadband RMS shape.
    if np.isnan(start) or np.isnan(end) or end <= start:
        return float("nan"), float("nan"), float("nan")
    length = end - start
    attack_end = start + min(0.1, length * 0.3)
    decay_start = end - length * 0.25
    attack_v = band_peak_volume(audio_data, freq_hz, start, attack_end)
    sustain_v = (band_volume(audio_data, freq_hz, attack_end, decay_start)
                 if decay_start > attack_end else float("nan"))
    decay_v = band_volume(audio_data, freq_hz, decay_start, end)
    return attack_v, sustain_v, decay_v

def band_energy_spike(audio_data: dict, freq_hz: float, onset_time: float, pre_window: float = 0.05, post_window: float = 0.05) -> float: # attack sharpness within the note's own pitch band
    cqt = audio_data["cqt_mag"]
    times = audio_data["cqt_times"]
    bin_idx = freq_to_cqt_bin(freq_hz)
    pre_mask = (times >= onset_time - pre_window) & (times < onset_time)
    post_mask = (times >= onset_time) & (times < onset_time + post_window)
    pre = float(cqt[bin_idx, pre_mask].mean()) if np.any(pre_mask) else 0.0
    post = float(cqt[bin_idx, post_mask].mean()) if np.any(post_mask) else 0.0
    ratio = post / (pre + 1e-9)
    return float(np.clip(np.log2(ratio + 1e-9) / 4.0, 0.0, 1.0))

def band_presence(audio_data: dict, freq_hz: float, start_sec: float, end_sec: float) -> float: # 0–1: expected pitch's band energy in window relative to loudest *any-pitch* band at the same time — separates "this note is present" from "this note is loud globally"
    if np.isnan(start_sec) or np.isnan(end_sec) or end_sec <= start_sec:
        return 0.0
    cqt = audio_data["cqt_mag"]
    times = audio_data["cqt_times"]
    bin_idx = freq_to_cqt_bin(freq_hz)
    mask = (times >= start_sec) & (times < end_sec)
    if not np.any(mask):
        return 0.0
    slab = cqt[:, mask]
    own = float(slab[bin_idx].mean())
    loudest = float(slab.max(axis=1).max())
    if loudest <= 0:
        return 0.0
    return float(np.clip(own / loudest, 0.0, 1.0))

def get_dominant_pitch(f0: np.ndarray, voiced_prob: np.ndarray, f0_times: np.ndarray, start_sec: float, end_sec: float): # return frequency and mean confidence for a time window
    # confidence weighted median to reduce octave-error sensitivity
    mask = (f0_times >= start_sec) & (f0_times < end_sec) & ~np.isnan(f0)

    if not np.any(mask):
        return 0.0, 0.0

    freqs = f0[mask]
    confs = voiced_prob[mask]

    if np.sum(confs) == 0:
        return float(np.nanmedian(freqs)), 0.0

    order = np.argsort(freqs)
    freqs_s = freqs[order]
    confs_s = confs[order]
    cumulative_weight = np.cumulative_sum(confs_s)
    median_freq = freqs_s[np.searchsorted(cumulative_weight, cumulative_weight[-1] / 2)]

    mean_conf = float(np.mean(confs))
    return float(median_freq), mean_conf

def get_energy_spike(rms: np.ndarray, rms_times: np.ndarray, onset_time: float, pre_window: float = 0.05, post_window: float = 0.05) -> float: # RMS spike at an onset (key being pressed) (0–1: 1 = strong clear attack, 0 = no spike)
    pre_mask  = (rms_times >= onset_time - pre_window) & (rms_times < onset_time)
    post_mask = (rms_times >= onset_time) & (rms_times < onset_time + post_window)

    pre_rms  = float(np.mean(rms[pre_mask]))  if np.any(pre_mask)  else 0.0
    post_rms = float(np.mean(rms[post_mask])) if np.any(post_mask) else 0.0

    ratio = post_rms / (pre_rms + 1e-9)
    spike = np.clip(np.log2(ratio) / 4.0, 0.0, 1.0)
    return float(spike)

def build_onset_list(audio_data: dict) -> list[DetectedOnset]: # onsets --> list of DetectedOnset objects
    onset_times = audio_data["onset_times"]
    f0          = audio_data["f0"]
    f0_times    = audio_data["f0_times"]
    voiced_prob = audio_data["voiced_prob"]
    rms         = audio_data["rms"]
    rms_times   = audio_data["rms_times"]
    duration    = audio_data["duration_sec"]

    onsets: list[DetectedOnset] = []
    for i, t in enumerate(onset_times):
        next_t     = onset_times[i + 1] if i + 1 < len(onset_times) else duration
        window_end = min(t + (next_t - t) * 0.8, next_t)

        freq, conf = get_dominant_pitch(f0, voiced_prob, f0_times, t, window_end)
        spike      = get_energy_spike(rms, rms_times, t)

        rmask    = (rms_times >= t) & (rms_times < next_t)
        mean_rms = float(np.mean(rms[rmask])) if np.any(rmask) else 0.0

        pitch_name = freq_to_note_name(freq) if freq > 0 else "?"

        onsets.append(DetectedOnset(
            index=i,
            time_sec=t,
            pitch=pitch_name,
            frequency_hz=freq,
            pitch_confidence=conf,
            energy_spike=spike,
            rms=mean_rms,
        ))

    return onsets

def clean_onsets(onsets: list[DetectedOnset], min_gap_sec: float = 0.08, min_energy_spike: float = 0.05, min_pitch_confidence: float = 0.15) -> list[DetectedOnset]:
    # filter onsets with very weak energy spike OR very low pitch confidence (usually mid-note artifacts or noise)
    # merge onsets that are too close together (split-notes). keep the one with the stronger energy spike (true key press).
    # tune min_gap_sec based on tempo: 0.08s works for ~120 BPM with notes no shorter than eighth notes. lower it for faster pieces with sixteenths.
    
    if not onsets:
        print("[clean_onsets] no onsets to clean")
        return onsets

    original_count = len(onsets)

    # filtering
    filtered = [
        o for o in onsets
        if o.energy_spike >= min_energy_spike
        and o.pitch_confidence >= min_pitch_confidence
    ]

    if not filtered:
        print("[clean_onsets] all onsets filtered out — thresholds may be too strict")
        return filtered

    # merging
    merged = [filtered[0]]
    for o in filtered[1:]:
        if o.time_sec - merged[-1].time_sec < min_gap_sec:
            if o.energy_spike > merged[-1].energy_spike:
                merged[-1] = o
        else:
            merged.append(o)

    # re-index sequentially so onset.index matches list position
    for i, o in enumerate(merged):
        o.index = i

    removed = original_count - len(merged)
    print(f"[clean_onsets] {original_count} --> {len(merged)} onsets ({removed} removed)")
    return merged

# obo (one-by-one) scoring

obo_PITCH_WEIGHT  = 0.7
obo_SPIKE_WEIGHT  = 0.25
obo_TIMING_WEIGHT = 0.05  # reintroduce timing to prevent distant matches
MAX_TIME_JUMP_SEC    = 0.5   # hard limit: don't match onsets more than 0.5s ahead

def score_candidate(xml_note: XmlNote, onset: DetectedOnset, expected_time: float, timing_tolerance_sec: float = 1.0) -> float: # score an onset candidate for an XML note. Returns 0–100.
    # reject onsets too far ahead (cascade prevention)
    if onset.time_sec > expected_time + MAX_TIME_JUMP_SEC:
        return 0.0

    # Pitch score (0–100)
    dist = semitone_distance(xml_note.frequency_hz, onset.frequency_hz)
    if dist < 0.5:
        pitch_score = 100.0
    elif dist < 1.5:
        pitch_score = 60.0 * (1.0 - (dist - 0.5))
    elif dist < 3.0:
        pitch_score = 20.0 * (1.0 - (dist - 1.5) / 1.5)
    else:
        pitch_score = 0.0
    pitch_score *= (0.5 + 0.5 * onset.pitch_confidence)

    # Energy spike score (0–100)
    spike_score = onset.energy_spike * 100.0

    # Timing score (0–100) — now weighted to penalize distant matches
    time_diff    = abs(onset.time_sec - expected_time)
    timing_score = max(0.0, 100.0 * (1.0 - time_diff / timing_tolerance_sec))

    total = (obo_PITCH_WEIGHT  * pitch_score
           + obo_SPIKE_WEIGHT  * spike_score
           + obo_TIMING_WEIGHT * timing_score)
    return round(total, 1)

def obo_match(xml_notes: list[XmlNote], onsets: list[DetectedOnset], bpm: float) -> list[tuple]: # match each XML note to a detected onset.
    if not onsets:
        print("[obo_match] no onsets detected")
        return [(xn, None, 0.0) for xn in xml_notes]

    # 1:1
    if len(onsets) == len(xml_notes):
        print(f"[obo_match] onset count matches note count ({len(onsets)}) using direct 1:1 matching")
        results = []
        for xml_note, onset in zip(xml_notes, onsets):
            score = score_candidate(xml_note, onset, onset.time_sec)
            results.append((xml_note, onset, score))
        return results

    # fallback: scored sequential matching for mismatched counts
    LOOKAHEAD = 8
    results = []
    onset_cursor = 0
    sec_per_beat = 60.0 / bpm

    # remove leading silence
    current_time = onsets[0].time_sec

    for xml_note in xml_notes:
        expected_time = current_time

        candidates = []
        for look in range(LOOKAHEAD):
            candidate_idx = onset_cursor + look
            if candidate_idx >= len(onsets):
                break
            candidate = onsets[candidate_idx]
            score = score_candidate(xml_note, candidate, expected_time)
            candidates.append((score, candidate_idx, candidate))

        if not candidates:
            current_time += xml_note.duration_beats * sec_per_beat
            results.append((xml_note, None, 0.0))
            continue

        best_score, best_idx, best_onset = max(candidates, key=lambda x: x[0])

        if best_score == 0.0:
            current_time += xml_note.duration_beats * sec_per_beat
            results.append((xml_note, None, 0.0))
            continue

        onset_cursor = best_idx + 1
        # current_time to actual matched onset time prevents drift when the assumed BPM doesn't match the actual playing tempo
        current_time = best_onset.time_sec + xml_note.duration_beats * sec_per_beat

        results.append((xml_note, best_onset, best_score))

    return results

def estimate_bpm(xml_notes: list[XmlNote], audio_duration_sec: float, onsets: Optional[list[DetectedOnset]] = None) -> float:
    # method 1: excludes leading/trailing silence
    # fallback: total audio duration / total beats (less accurate if there's leading/trailing silence)
    # returns bpm in quarter-notes per minute
    total_beats = sum(n.duration_beats for n in xml_notes)

    # first-to-last onset span
    if onsets and len(onsets) >= 2:
        last_note_beats = xml_notes[-1].duration_beats
        beats_between = total_beats - last_note_beats
        audio_span = onsets[-1].time_sec - onsets[0].time_sec
        if beats_between > 0 and audio_span > 0:
            return round(60.0 * beats_between / audio_span, 1)

    # fallback
    if audio_duration_sec > 0:
        return round(60.0 * total_beats / audio_duration_sec, 1)

    return 120.0

def normalise_rms(onsets: list[DetectedOnset]) -> float: # max RMS across all onsets for normalization
    return max(o.rms for o in onsets) or 1.0

def _window_mean_volume(rms, rms_times, start, end, max_rms) -> float:
    mask = (rms_times >= start) & (rms_times < end)
    return round(float(np.mean(rms[mask])) / max_rms, 4)

def _window_peak_volume(rms, rms_times, start, end, max_rms) -> float:
    mask = (rms_times >= start) & (rms_times < end)
    return round(float(np.max(rms[mask])) / max_rms, 4)

def compute_dynamic_shape(start, end, rms, rms_times, max_rms):
    # Split a matched note into attack / sustain / decay sub-windows and report
    # normalised volume in each. Lets downstream code see the *shape* of a single
    # note's envelope (not just its mean), which is what "dynamic shaping within
    # a single note's decay" actually requires.
    if np.isnan(start) or np.isnan(end):
        return float("nan"), float("nan"), float("nan")
    length = end - start
    if length <= 0:
        return float("nan"), float("nan"), float("nan")

    attack_end   = start + min(0.1, length * 0.3)
    decay_start  = end - length * 0.25
    sustain_lo   = attack_end
    sustain_hi   = decay_start

    attack_v  = _window_peak_volume(rms, rms_times, start, attack_end, max_rms)
    sustain_v = (_window_mean_volume(rms, rms_times, sustain_lo, sustain_hi, max_rms)
                 if sustain_hi > sustain_lo else float("nan"))
    decay_v   = _window_mean_volume(rms, rms_times, decay_start, end, max_rms)
    return attack_v, sustain_v, decay_v

def compute_expected_starts(xml_notes: list[XmlNote], bpm: float, anchor_time: float) -> list[float]:
    # Score-time start for each note, anchored on the first matched onset so
    # leading silence is absorbed by the anchor instead of pretending the
    # performer started at t=0. Anything past the anchor uses cumulative XML
    # beats * sec_per_beat, which is the reference clock rubato is measured against.
    sec_per_beat = 60.0 / bpm if bpm > 0 else 0.5
    expected = []
    cumulative = anchor_time
    for n in xml_notes:
        expected.append(cumulative)
        cumulative += n.duration_beats * sec_per_beat
    return expected

def assemble_matches(obo_results: list[tuple], onsets: list[DetectedOnset], audio_data: dict, bpm: float) -> list[MatchedNote]: # obo results to final MatchedNote object
    duration_sec = audio_data["duration_sec"]
    rms = audio_data["rms"]
    rms_times = audio_data["rms_times"]
    f0 = audio_data["f0"]
    f0_times = audio_data["f0_times"]
    voiced_prob = audio_data["voiced_prob"]
    max_rms = normalise_rms(onsets)

    # Anchor expected timing on the first matched onset
    anchor_time = next(
        (o.time_sec for _, o, _ in obo_results if o is not None),
        onsets[0].time_sec if onsets else 0.0,
    )
    xml_notes_in_order = [xn for xn, _, _ in obo_results]
    expected_starts = compute_expected_starts(xml_notes_in_order, bpm, anchor_time)

    matched: list[MatchedNote] = []

    for i, (xml_note, onset, obo_score) in enumerate(obo_results):

        if onset is not None:
            start_sec = onset.time_sec
            # End = the very next detected onset in the full list.
            # Using the full onset list (not just matched ones) prevents
            # one segment bleeding into the next physical note.
            next_onset_idx = onset.index + 1
            end_sec = onsets[next_onset_idx].time_sec if next_onset_idx < len(onsets) else duration_sec
        else:
            start_sec = float("nan")
            end_sec   = float("nan")

        note_length_sec = (
            end_sec - start_sec
            if not (np.isnan(start_sec) or np.isnan(end_sec))
            else float("nan")
        )

        note_name = (
            sec_to_note_name(note_length_sec, bpm)
            if not np.isnan(note_length_sec)
            else "unknown"
        )

        if onset is not None and not np.isnan(start_sec):
            rec_freq, pitch_conf = get_dominant_pitch(f0, voiced_prob, f0_times, start_sec, end_sec)
            rec_pitch   = freq_to_note_name(rec_freq) if rec_freq > 0 else "?"

            rmask   = (rms_times >= start_sec) & (rms_times < end_sec)
            seg_rms = float(np.mean(rms[rmask])) if np.any(rmask) else 0.0
            volume  = round(seg_rms / max_rms, 4)
            attack_v, sustain_v, decay_v = compute_dynamic_shape(
                start_sec, end_sec, rms, rms_times, max_rms
            )
            energy_spike = onset.energy_spike
        else:
            rec_freq     = 0.0
            rec_pitch    = "?"
            volume       = 0.0
            attack_v     = float("nan")
            sustain_v    = float("nan")
            decay_v      = float("nan")
            energy_spike = float("nan")
            pitch_conf   = float("nan")

        expected_start = expected_starts[i]
        timing_offset = (
            round(start_sec - expected_start, 4)
            if not np.isnan(start_sec) else float("nan")
        )

        matched.append(MatchedNote(
            xml_note=xml_note,
            onset=onset,
            start_sec=round(start_sec, 4) if not np.isnan(start_sec) else float("nan"),
            end_sec=round(end_sec, 4)     if not np.isnan(end_sec)   else float("nan"),
            note_length_sec=round(note_length_sec, 4) if not np.isnan(note_length_sec) else float("nan"),
            note_name=note_name,
            volume=volume,
            confidence=obo_score,
            expected_start_sec=round(expected_start, 4),
            timing_offset_sec=timing_offset,
            attack_volume=attack_v,
            sustain_volume=sustain_v,
            decay_volume=decay_v,
            energy_spike=round(energy_spike, 4) if not np.isnan(energy_spike) else float("nan"),
            pitch_confidence=round(pitch_conf, 4) if not np.isnan(pitch_conf) else float("nan"),
        ))

    # Second pass: IOI / gap  depend on the *next matched* note,
    # so they need the whole sequence assembled first.
    for i, m in enumerate(matched):
        if np.isnan(m.start_sec):
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

def play_onsets(onsets: list[DetectedOnset], audio_data: dict): # play detected onsets one by one
    y  = audio_data["y"]
    sr = audio_data["sr"]
    duration_sec = audio_data["duration_sec"]

    for i, onset in enumerate(onsets):
        next_onset_time = onsets[i + 1].time_sec if i + 1 < len(onsets) else duration_sec
        window_end = min(onset.time_sec + (next_onset_time - onset.time_sec) * 0.8, next_onset_time)

        print(f"Onset {i + 1:>3}  |  "
              f"Time: {onset.time_sec:.3f}s  |  "
              f"Pitch: {onset.pitch:<4} ({onset.frequency_hz:.1f} Hz)  |  "
              f"Confidence: {onset.pitch_confidence:.2f}  |  "
              f"Energy Spike: {onset.energy_spike:.2f}")

        user = input("Press Enter to play, 'q' to quit: ").strip().lower()
        if user == "q":
            print("Quitting playback.\n")
            break

        start_sample = int(onset.time_sec * sr)
        end_sample   = int(window_end * sr)
        segment      = y[start_sample:end_sample]

        if len(segment) > 0:
            print(f"Playing {len(segment) / sr:.3f}s segment...")
            sd.play(segment, samplerate=sr)
            sd.wait()
        else:
            print("[empty segment]\n")
            continue

        print()

CSV_COLUMNS = [
    # basic info
    "note_index",
    "measure",
    "expected_pitch",
    "expected_frequency_hz",
    "recorded_pitch",
    "recorded_frequency_hz",
    # timing — score reference + actual + offset (rubato, micro-timing)
    "expected_start_sec",
    "start_sec",
    "end_sec",
    "note_length_sec",
    "timing_offset_sec",
    "inter_onset_interval_sec",
    "gap_to_next_sec", # silence to next note (breath / phrasing)
    "note_name",
    # dynamic shape — mean + intra-note envelope (dynamic shaping, tension/release)
    "volume",
    "attack_volume",
    "sustain_volume",
    "decay_volume",
    "energy_spike", # attack sharpness (touch)
    "pitch_confidence",
    "confidence",
    "low_confidence_flag",
]

LOW_CONFIDENCE_THRESHOLD = 50.0

def write_csv(matched: list[MatchedNote], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    rows = []
    for m in matched:
        xn = m.xml_note
        onset = m.onset

        rec_freq = onset.frequency_hz if onset else 0.0
        low_conf = m.confidence < LOW_CONFIDENCE_THRESHOLD

        rows.append({
            "note_index":               xn.index + 1,
            "measure":                  xn.measure,
            "expected_pitch":           xn.pitch,
            "expected_frequency_hz":    round(xn.frequency_hz, 2),
            "recorded_pitch":           onset.pitch if onset else "?",
            "recorded_frequency_hz":    round(rec_freq, 2) if onset else 0.0,
            "expected_start_sec":       m.expected_start_sec,
            "start_sec":                m.start_sec,
            "end_sec":                  m.end_sec,
            "note_length_sec":          m.note_length_sec,
            "timing_offset_sec":        m.timing_offset_sec,
            "inter_onset_interval_sec": m.inter_onset_interval_sec,
            "gap_to_next_sec":          m.gap_to_next_sec,
            "note_name":                m.note_name,
            "volume":                   m.volume,
            "attack_volume":            m.attack_volume,
            "sustain_volume":           m.sustain_volume,
            "decay_volume":             m.decay_volume,
            "energy_spike":             m.energy_spike,
            "pitch_confidence":         m.pitch_confidence,
            "confidence":               m.confidence,
            "low_confidence_flag":      "YES" if low_conf else "",
        })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[write_csv] CSV written to {output_path}  ({len(rows)} rows)")
    low_count = sum(1 for r in rows if r["low_confidence_flag"] == "YES")
    if low_count:
        print(f"[write_csv] {low_count} notes flagged as low confidence")

def process_recording(xml_notes, score_meta, audio_path: str, output_path: str):
    print(f"[process_recording] processing {audio_path}")

    audio_data = analyze_audio(audio_path)

    print(f"[process_recording] searching for onsets...")
    onsets = build_onset_list(audio_data)
    onsets = clean_onsets(onsets)

    bpm = estimate_bpm(xml_notes, audio_data["duration_sec"], onsets)
    print(f"[process_recording] estimated BPM: {bpm} (quarter-notes per minute)")

    if score_meta.time_sig_den == 8 and score_meta.time_sig_num in (6, 9, 12):
        conventional_bpm = round(bpm / 1.5, 1)
        print(f"[process_recording] compound time {score_meta.time_sig_num}/{score_meta.time_sig_den}: "
              f"conventional BPM (dotted-quarter beat) ≈ {conventional_bpm}")

    print(f"[process_recording] XML notes: {len(xml_notes)}  |  Detected onsets: {len(onsets)}")
    if len(onsets) < len(xml_notes):
        print(f"[process_recording] fewer onsets than notes — some notes will be missed.")
        print(f"[process_recording] lower 'delta' in analyze_audio if this persists.")

    print(f"[process_recording] running obo sequential matching ...")
    obo_results = obo_match(xml_notes, onsets, bpm)
    matched_count  = sum(1 for _, o, _ in obo_results if o is not None)
    print(f"[process_recording] obo matched {matched_count}/{len(xml_notes)} notes")

    matched = assemble_matches(obo_results, onsets, audio_data, bpm)

    write_csv(matched, output_path)
    return matched
