import csv
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable, Optional
import librosa
import numpy as np

LOW_CONFIDENCE_THRESHOLD = 35.0

CSV_COLUMNS = ["measure_index", "note_count", "expected_pitches", "recorded_dominant_pitch", "recorded_dominant_frequency_hz",
                "expected_start_sec", "start_sec", "end_sec", "audio_end_sec", "measure_length_sec", "timing_offset_sec",
                "inter_measure_interval_sec", "gap_to_next_sec", "volume", "attack_volume", "sustain_volume", "decay_volume",
                "volume_range_within", "volume_std_within", "energy_spike", "pitch_confidence", "confidence", "low_confidence_flag"]

@dataclass
class XmlNote:
    index: int
    pitch: str
    frequency_hz: float
    duration_beats: float
    start_beat: float
    measure: int
    voice: int = 1

@dataclass
class MatchedMeasure:
    measure_index: int
    note_count: int
    expected_pitches: str
    recorded_dominant_pitch: str
    recorded_dominant_frequency_hz: float
    expected_start_sec: float
    start_sec: float
    end_sec: float
    audio_end_sec: float
    measure_length_sec: float
    timing_offset_sec: float
    inter_measure_interval_sec: float
    gap_to_next_sec: float
    volume: float
    attack_volume: float
    sustain_volume: float
    decay_volume: float
    volume_range_within: float
    volume_std_within: float
    energy_spike: float
    pitch_confidence: float
    confidence: float

@dataclass
class ScoreMetadata:
    time_sig_num: int = 4
    time_sig_den: int = 4
    total_beats: float = 0.0
    measure_count: int = 0 

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

def freq_to_midi(freq: float) -> float:
    return 69 + 12 * np.log2(freq / 440.0)

def freq_to_note_name(freq: float) -> str: # nearest note name for a frequency.
    midi = int(round(freq_to_midi(freq)))
    return midi_to_note_name(midi)

def parse_xml_measure(xml_path: str) -> tuple[list[XmlNote], ScoreMetadata]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def strip_ns(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    notes: list[XmlNote] = []
    note_index = 0
    metadata = ScoreMetadata()

    # Pre-scan for <divisions> — some divisions blocks come after the first measure's notes; the default of 1 then inflates
    # those notes' start_beat by 64x and the global sort breaks alignment.
    initial_divisions: Optional[int] = None
    for elem in root.iter():
        if strip_ns(elem.tag) == "divisions":
            initial_divisions = int(elem.text)
            break

    for part in root.iter():
        if strip_ns(part.tag) != "part":
            continue

        divisions_per_quarter = initial_divisions if initial_divisions is not None else 1
        measure_number = 0
        measure_start_beat = 0.0

        for measure in part:
            if strip_ns(measure.tag) != "measure":
                continue
            measure_number += 1

            cursor_div = 0 # position within this measure, in divisions
            max_cursor_div = 0 # furthest position reached (= measure length)
            last_note_start_div = 0 # start of the most recent non-chord note

            for child in measure:
                tag = strip_ns(child.tag)

                if tag == "attributes":
                    for ac in child:
                        atag = strip_ns(ac.tag)
                        if atag == "divisions":
                            divisions_per_quarter = int(ac.text)
                        elif atag == "time":
                            beats_el = next((c for c in ac if strip_ns(c.tag) == "beats"), None)
                            beat_type_el = next((c for c in ac if strip_ns(c.tag) == "beat-type"), None)
                            if beats_el is not None:
                                metadata.time_sig_num = int(beats_el.text)
                            if beat_type_el is not None:
                                metadata.time_sig_den = int(beat_type_el.text)

                elif tag == "backup":
                    dur_el = next((c for c in child if strip_ns(c.tag) == "duration"), None)
                    if dur_el is not None:
                        cursor_div -= int(dur_el.text)

                elif tag == "forward":
                    dur_el = next((c for c in child if strip_ns(c.tag) == "duration"), None)
                    if dur_el is not None:
                        cursor_div += int(dur_el.text)

                elif tag == "note":
                    is_chord = any(strip_ns(c.tag) == "chord" for c in child)
                    is_rest = any(strip_ns(c.tag) == "rest" for c in child)

                    dur_el = next((c for c in child if strip_ns(c.tag) == "duration"), None)
                    duration_div = int(dur_el.text) if dur_el is not None else 0

                    note_start_div = last_note_start_div if is_chord else cursor_div

                    if not is_rest:
                        pitch_el = next((c for c in child if strip_ns(c.tag) == "pitch"), None)
                        if pitch_el is not None:
                            step = next((c.text for c in pitch_el if strip_ns(c.tag) == "step"), "C")
                            octave_text = next((c.text for c in pitch_el if strip_ns(c.tag) == "octave"), "4")
                            alter_el = next((c for c in pitch_el if strip_ns(c.tag) == "alter"), None)
                            alter = float(alter_el.text) if alter_el is not None else 0.0
                            voice_el = next((c for c in child if strip_ns(c.tag) == "voice"), None)
                            voice = int(voice_el.text) if voice_el is not None else 1

                            midi = note_to_midi(step, int(octave_text), alter)
                            notes.append(XmlNote(
                                index=note_index,
                                pitch=midi_to_note_name(midi),
                                frequency_hz=midi_to_freq(midi),
                                duration_beats=duration_div / divisions_per_quarter,
                                start_beat=measure_start_beat + note_start_div / divisions_per_quarter,
                                measure=measure_number,
                                voice=voice,
                            ))
                            note_index += 1

                    # chord notes share previous note's start, so cursor must not advance
                    if not is_chord:
                        last_note_start_div = cursor_div
                        cursor_div += duration_div

                    if cursor_div > max_cursor_div:
                        max_cursor_div = cursor_div

            measure_start_beat += max_cursor_div / divisions_per_quarter

        metadata.total_beats = max(metadata.total_beats, measure_start_beat)
        metadata.measure_count = max(metadata.measure_count, measure_number)

    # sort onto single absolute timeline; stable secondary sort keeps chord notes adjacent
    notes.sort(key=lambda n: (n.start_beat, n.voice, n.index))
    for i, n in enumerate(notes):
        n.index = i

    print(f"[parse_xml_measure] Parsed {len(notes)} notes (incl. chord/poly)  |  "
          f"Time sig: {metadata.time_sig_num}/{metadata.time_sig_den}  |  "
          f"{metadata.total_beats:.1f} beats over {metadata.measure_count} measures")
    return notes, metadata

def group_by_measure(xml_notes: list[XmlNote]) -> list[tuple[int, float, list[XmlNote]]]:
    by_measure: dict[int, list[XmlNote]] = {}
    for n in xml_notes:
        by_measure.setdefault(n.measure, []).append(n)
    items: list[tuple[int, float, list[XmlNote]]] = []
    for m in sorted(by_measure.keys()):
        notes = by_measure[m]
        start_beat = min(n.start_beat for n in notes)
        items.append((m, start_beat, notes))
    return items

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
        "hop_length": hop_length,
        "cqt_mag": cqt_mag,
        "cqt_times": cqt_times,
        "band_peak": band_peak,
    }

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

def note_name_to_pitch_class(name: str) -> int: # 'C', 'C#', ... 'B' + octave digit --> 0–11
    if len(name) >= 2 and name[1] == "#":
        return NOTE_NAMES.index(name[:2])
    return NOTE_NAMES.index(name[:1])

# harmonic pitch-class offsets + relative weights — matches piano chroma shape
# (octave = same pc, fifth = +7, major third = +4, b7 = +10)
HARMONIC_PCS = [(0, 1.0), (0, 0.5), (7, 0.3), (4, 0.15), (10, 0.08)]

def synthesize_score_chroma(xml_notes: list[XmlNote], bpm: float, hop_sec: float, pad_sec: float = 1.0) -> np.ndarray: # (12, n_frames) — each note stamps its harmonic stack (fundamental + 4 partials) onto the pitch-class grid for every frame it sounds. matches real piano chroma shape.
    if not xml_notes:
        return np.zeros((12, 1), dtype=np.float32)

    sec_per_beat = 60.0 / bpm
    end_sec = max(n.start_beat + n.duration_beats for n in xml_notes) * sec_per_beat + pad_sec
    n_frames = int(np.ceil(end_sec / hop_sec)) + 1

    chroma = np.zeros((12, n_frames), dtype=np.float32)
    for n in xml_notes:
        start_f = max(0, int(np.floor(n.start_beat * sec_per_beat / hop_sec)))
        end_f = min(n_frames, int(np.ceil((n.start_beat + n.duration_beats) * sec_per_beat / hop_sec)))
        if end_f <= start_f:
            continue
        pc = note_name_to_pitch_class(n.pitch)
        for offset, weight in HARMONIC_PCS:
            chroma[(pc + offset) % 12, start_f:end_f] += weight

    # tiny uniform offset prevents zero columns (which break cosine DTW)
    chroma += 1e-3
    chroma /= np.linalg.norm(chroma, axis=0, keepdims=True)
    return chroma

def compute_audio_chroma(audio_data: dict, hop_length: int = 512) -> tuple[np.ndarray, float]: # CQT chroma — log-spaced bins pool harmonics into pitch classes
    y = audio_data["y"]
    sr = audio_data["sr"]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma = chroma + 1e-6
    chroma /= np.linalg.norm(chroma, axis=0, keepdims=True)
    return chroma, hop_length / sr

def dtw_align(score_chroma: np.ndarray, audio_chroma: np.ndarray) -> np.ndarray:
    print(f"[dtw_align] aligning score ({score_chroma.shape[1]} frames) "
          f"-> audio ({audio_chroma.shape[1]} frames) ...")
    # Penalise off-diagonal steps so the path can't sit horizontal/vertical
    # for free across harmonically-ambiguous boundaries (rach2pt2 m5 was
    # snapping ~1s early because m4's pedal blur shared pitch classes with
    # m5's downbeat). Diagonal stays cost-equal to cosine distance; the
    # 0.05 toll is small enough not to fight legitimate rubato.
    _, wp = librosa.sequence.dtw(
        X=score_chroma, Y=audio_chroma,
        metric="cosine",
        subseq=False,
        step_sizes_sigma=np.array([[1, 1], [1, 0], [0, 1]]),
        weights_add=np.array([0.0, 0.05, 0.05]),
    )
    # librosa returns end -> start; flip
    return wp[::-1]

def build_time_mapper(wp: np.ndarray, hop_sec: float) -> Callable[[float], float]: # average audio frames per unique score frame for a monotonic score -> audio curve
    score_frames = wp[:, 0].astype(np.int64)
    audio_frames = wp[:, 1].astype(np.float64)

    n_score = int(score_frames.max()) + 1
    counts = np.bincount(score_frames, minlength=n_score)
    sums = np.bincount(score_frames, weights=audio_frames, minlength=n_score)
    mask = counts > 0
    unique_sf = np.arange(n_score)[mask].astype(np.float64)
    mean_af = sums[mask] / counts[mask]

    def map_time(score_sec: float) -> float:
        score_f = score_sec / hop_sec
        audio_f = float(np.interp(score_f, unique_sf, mean_af))
        return audio_f * hop_sec

    return map_time

def estimate_initial_bpm(xml_notes: list[XmlNote], audio_duration_sec: float) -> float: # rough BPM — DTW warps from here, doesn't have to be exact
    if not xml_notes:
        return 120.0
    total_beats = max(n.start_beat + n.duration_beats for n in xml_notes)
    if total_beats <= 0 or audio_duration_sec <= 0:
        return 120.0
    return round(60.0 * total_beats / audio_duration_sec, 1)

def measure_chroma_template(notes: list[XmlNote]) -> np.ndarray:
    # weight each pitch class by its total sounding-time in the measure so
    # long sustained notes carry more weight than brief ornaments
    template = np.zeros(12, dtype=np.float32)
    for n in notes:
        pc = note_name_to_pitch_class(n.pitch)
        template[pc] += max(0.1, n.duration_beats)
    norm = float(np.linalg.norm(template))
    return template / norm if norm > 0 else template

def correct_anomalous_measure_lengths(starts_sec: list[float], measures_grouped: list[tuple[int, float, list[XmlNote]]], duration_sec: float) -> list[float]:
    # DTW slides across boundaries where adjacent measures share most of
    # their pitch classes (rach1 m3->m4, m15->m16). Symptom: one measure
    # absorbs audio from its neighbor, ending up much longer per score
    # beat than the measures around it.
    #
    # Detect this by computing seconds-per-beat for each measure (using
    # the XML's beat count, not absolute durations, so we're robust to
    # rubato) and flagging measures whose s/beat is much higher than the
    # local median across its non-self neighbors. The chroma signal is
    # too polluted in these passages to relocate the boundary precisely,
    # so we just shrink the anomalous measure to the neighbor-implied
    # length and shift the excess into the preceding measure (which is
    # the side observed to be cut off).
    if len(starts_sec) < 4:
        return list(starts_sec)

    NEIGHBOR_HALF_WINDOW = 2  # use up to 2 measures on each side
    RATIO_THRESHOLD = 1.4     # how much higher than the local median triggers a correction
    MIN_GAP_SEC = 0.2

    n = len(measures_grouped)
    beats = [0.0] * n
    for i in range(n):
        if i + 1 < n:
            beats[i] = measures_grouped[i + 1][1] - measures_grouped[i][1]
        else:
            last_end = max(no.start_beat + no.duration_beats for no in measures_grouped[i][2])
            beats[i] = max(0.0, last_end - measures_grouped[i][1])

    lengths = [
        (starts_sec[i + 1] - starts_sec[i]) if i + 1 < n else (duration_sec - starts_sec[i])
        for i in range(n)
    ]
    spb = [(lengths[i] / beats[i]) if beats[i] > 0 else 0.0 for i in range(n)]

    refined = list(starts_sec)
    corrections: list[tuple[int, float, float, float]] = []

    # First and last measure are excluded — they often have fermatas, anacruses,
    # or trailing release that legitimately deviate from the local rubato.
    for i in range(1, n - 1):
        if spb[i] <= 0:
            continue
        nbrs = [spb[j] for j in range(max(0, i - NEIGHBOR_HALF_WINDOW), min(n, i + NEIGHBOR_HALF_WINDOW + 1))
                if j != i and spb[j] > 0]
        if len(nbrs) < 2:
            continue
        local_median = float(np.median(nbrs))
        if local_median <= 0:
            continue
        if spb[i] / local_median <= RATIO_THRESHOLD:
            continue

        target_length = local_median * beats[i]
        shift = lengths[i] - target_length
        new_start = starts_sec[i] + shift
        new_start = min(new_start, starts_sec[i + 1] - MIN_GAP_SEC)
        new_start = max(new_start, refined[i - 1] + MIN_GAP_SEC)
        refined[i] = new_start
        corrections.append((measures_grouped[i][0], spb[i], local_median, new_start - starts_sec[i]))

    for i in range(1, len(refined)):
        if refined[i] < refined[i - 1]:
            refined[i] = refined[i - 1]

    if corrections:
        for measure_num, this_spb, med_spb, shift in corrections:
            print(f"[correct_anomalous_measure_lengths] m{measure_num}: shifted start "
                  f"+{shift:.2f}s (s/beat {this_spb:.3f} vs local median {med_spb:.3f})")
    else:
        print("[correct_anomalous_measure_lengths] no anomalous measures detected")

    return refined

def _sum_skip_nan(vals: list[float]) -> float:
    clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
    return round(float(np.sum(clean)), 4) if clean else float("nan")

def find_inter_measure_silence(audio_data: dict, start_sec: float, slot_end_sec: float, noise_floor_rms: float) -> tuple[float, float]:
    # Locate the longest silence run inside [start_sec, slot_end_sec) and treat it as
    # the inter-measure gap. Returns (audio_end_sec, gap_to_next_sec):
    #   - audio_end_sec: time at which audible content in this measure stops (start of
    #     the silence run). Equals slot_end_sec when no significant silence is found.
    #   - gap_to_next_sec: duration of that silence run. 0 when audio sustains into
    #     the next measure's slot.
    # Scans the whole slot rather than just the tail because DTW boundaries can be
    # off by a beat or more in passages where adjacent measures share pitch content:
    # the audible breath then lives mid-slot, not at the slot edge (rach1_pro m14
    # is exactly this case — clear ~100ms dip near t=61.6 but DTW puts m15 onset
    # at t=62.94, well after the actual m15 has started).
    if np.isnan(start_sec) or np.isnan(slot_end_sec) or slot_end_sec <= start_sec:
        return slot_end_sec, 0.0
    rms = audio_data["rms"]
    times = audio_data["rms_times"]
    mask = (times >= start_sec) & (times < slot_end_sec)
    if not np.any(mask):
        return slot_end_sec, 0.0
    slot_rms = rms[mask]
    slot_times = times[mask]
    peak = float(slot_rms.max())
    # Threshold has to clear both the room-noise floor AND a fraction of this
    # measure's own peak — soft phrases shouldn't decay into "silence" by their
    # own standards.
    threshold = max(noise_floor_rms * 1.5, peak * 0.10)

    # ~70ms minimum (6 frames @ hop=512, sr~44k). Shorter dips are normal piano
    # articulation, not an inter-measure breath.
    MIN_SILENCE_FRAMES = 6

    below = slot_rms < threshold
    runs: list[tuple[int, int]] = []
    i, n = 0, int(below.size)
    while i < n:
        if below[i]:
            j = i + 1
            while j < n and below[j]:
                j += 1
            if j - i >= MIN_SILENCE_FRAMES:
                runs.append((i, j))
            i = j
        else:
            i += 1

    if not runs:
        return slot_end_sec, 0.0

    # Longest run wins; later-in-slot wins on ties (closer to the DTW boundary)
    best = max(runs, key=lambda r: (r[1] - r[0], r[0]))
    audio_end = float(slot_times[best[0]])
    if best[1] >= len(slot_times):
        gap = slot_end_sec - audio_end
    else:
        gap = float(slot_times[best[1]]) - audio_end
    return audio_end, gap

def measure_rms_variability(audio_data: dict, start_sec: float, end_sec: float) -> tuple[float, float]:
    # Broadband RMS sampled across the measure window, normalised by the
    # piece's peak RMS so values are comparable across recordings.
    # Captures the "living gradient" within a phrase that the 3-point
    # attack/sustain/decay envelope summarises too coarsely.
    if np.isnan(start_sec) or np.isnan(end_sec) or end_sec <= start_sec:
        return float("nan"), float("nan")
    rms = audio_data["rms"]
    times = audio_data["rms_times"]
    mask = (times >= start_sec) & (times < end_sec)
    if not np.any(mask):
        return float("nan"), float("nan")
    slice_rms = rms[mask]
    if len(slice_rms) < 2:
        return 0.0, 0.0
    peak = float(rms.max()) if rms.max() > 0 else 1.0
    rng = float(slice_rms.max() - slice_rms.min()) / peak
    std = float(slice_rms.std()) / peak
    return round(rng, 4), round(std, 4)

def assemble_measures(measures_grouped: list[tuple[int, float, list[XmlNote]]], audio_data: dict, bpm: float, map_time: Callable[[float], float], audio_chroma: np.ndarray, hop_sec: float) -> list[MatchedMeasure]:
    duration_sec = audio_data["duration_sec"]
    sec_per_beat = 60.0 / bpm
    f0 = audio_data["f0"]
    f0_times = audio_data["f0_times"]
    voiced_prob = audio_data["voiced_prob"]

    measure_starts_beat = [g[1] for g in measures_grouped]

    # Boundaries: a measure starts at its first-note beat and ends at the next
    # measure's first-note beat. For the last measure, use the latest note end.
    starts_sec = [float(np.clip(map_time(b * sec_per_beat), 0.0, duration_sec))
                  for b in measure_starts_beat]
    # DTW path is monotonic, but float interp can produce ties going backward
    for i in range(1, len(starts_sec)):
        if starts_sec[i] < starts_sec[i - 1]:
            starts_sec[i] = starts_sec[i - 1]

    starts_sec = correct_anomalous_measure_lengths(starts_sec, measures_grouped, duration_sec)

    ends_sec: list[float] = []
    for i in range(len(measures_grouped)):
        if i + 1 < len(measures_grouped):
            ends_sec.append(starts_sec[i + 1])
        else:
            last_end_beat = max(n.start_beat + n.duration_beats for n in measures_grouped[i][2])
            ends_sec.append(min(duration_sec, float(map_time(last_end_beat * sec_per_beat))))
    for i in range(len(ends_sec)):
        if ends_sec[i] < starts_sec[i]:
            ends_sec[i] = starts_sec[i]

    anchor_audio = starts_sec[0] if starts_sec else 0.0
    anchor_beat = measure_starts_beat[0] if measure_starts_beat else 0.0

    matched: list[MatchedMeasure] = []
    for i, (measure_num, start_beat, notes) in enumerate(measures_grouped):
        start_sec = starts_sec[i]
        end_sec = ends_sec[i]
        length_sec = max(0.0, end_sec - start_sec)

        expected_start = anchor_audio + (start_beat - anchor_beat) * sec_per_beat
        timing_offset = round(start_sec - expected_start, 4)

        # dict preserves insertion order, so it gives us deduped pitches and
        # their first-seen frequency in one pass — used both for the CSV's
        # expected_pitches column and for per-band feature accumulation.
        distinct_freqs: dict[str, float] = {}
        for n in notes:
            distinct_freqs.setdefault(n.pitch, n.frequency_hz)
        unique_pitches = list(distinct_freqs.keys())

        if length_sec > 0:
            # Sum each distinct pitch's per-band features so dense chords/runs
            # accumulate, but a hot single pitch can't dominate the measure.
            freqs = list(distinct_freqs.values())

            vol = round(sum(band_volume(audio_data, fz, start_sec, end_sec) for fz in freqs), 4)
            presence = round(sum(band_presence(audio_data, fz, start_sec, end_sec) for fz in freqs), 4)
            shape_vals = [compute_band_dynamic_shape(start_sec, end_sec, audio_data, fz) for fz in freqs]
            attack_v = _sum_skip_nan([s[0] for s in shape_vals])
            sustain_v = _sum_skip_nan([s[1] for s in shape_vals])
            decay_v = _sum_skip_nan([s[2] for s in shape_vals])
            vol_range, vol_std = measure_rms_variability(audio_data, start_sec, end_sec)
            energy_spike = _sum_skip_nan([band_energy_spike(audio_data, fz, start_sec) for fz in freqs])

            pitch_win_end = min(end_sec, start_sec + 0.25)
            rec_freq, _ = get_dominant_pitch(f0, voiced_prob, f0_times, start_sec, pitch_win_end)
            rec_pitch = freq_to_note_name(rec_freq) if rec_freq > 0 else "?"

            # Confidence = chroma similarity between the score's expected
            # pitch-class profile for the measure and the audio chroma over
            # the same window. Scales 0-100.
            tpl = measure_chroma_template(notes)
            sf = int(np.clip(round(start_sec / hop_sec), 0, audio_chroma.shape[1] - 1))
            ef = int(np.clip(round(end_sec / hop_sec), sf + 1, audio_chroma.shape[1]))
            measured = audio_chroma[:, sf:ef].mean(axis=1)
            mnorm = float(np.linalg.norm(measured))
            chroma_sim = float(np.clip(np.dot(tpl, measured / mnorm), 0.0, 1.0)) if mnorm > 0 else 0.0
            confidence = round(chroma_sim * 100.0, 1)
        else:
            vol = 0.0
            presence = 0.0
            attack_v = sustain_v = decay_v = float("nan")
            vol_range = vol_std = float("nan")
            energy_spike = float("nan")
            rec_freq = 0.0
            rec_pitch = "?"
            confidence = 0.0

        matched.append(MatchedMeasure(
            measure_index=measure_num,
            note_count=len(notes),
            expected_pitches="+".join(unique_pitches),
            recorded_dominant_pitch=rec_pitch,
            recorded_dominant_frequency_hz=round(rec_freq, 2),
            expected_start_sec=round(expected_start, 4),
            start_sec=round(start_sec, 4),
            end_sec=round(end_sec, 4),
            audio_end_sec=round(end_sec, 4),
            measure_length_sec=round(length_sec, 4),
            timing_offset_sec=timing_offset,
            inter_measure_interval_sec=float("nan"),
            gap_to_next_sec=float("nan"),
            volume=vol,
            attack_volume=attack_v,
            sustain_volume=sustain_v,
            decay_volume=decay_v,
            volume_range_within=vol_range,
            volume_std_within=vol_std,
            energy_spike=energy_spike,
            pitch_confidence=presence,
            confidence=confidence,
        ))

    # piece-wide silence floor — 5th-percentile RMS catches the quietest moments
    # without being thrown off by a stray glitch sample. The absolute floor keeps
    # very clean recordings (whose 5th percentile can be ~0) from setting a
    # threshold so low that decaying notes never look "silent".
    noise_floor_rms = max(0.001, float(np.percentile(audio_data["rms"], 5)))

    # second pass: real audio_end_sec, IOI, and gap between consecutive measures
    for i in range(len(matched) - 1):
        audio_end, gap = find_inter_measure_silence(audio_data, matched[i].start_sec, matched[i + 1].start_sec, noise_floor_rms)
        matched[i].audio_end_sec = round(audio_end, 4)
        ioi = matched[i + 1].start_sec - matched[i].start_sec
        matched[i].inter_measure_interval_sec = round(ioi, 4)
        matched[i].gap_to_next_sec = round(gap, 4)

    return matched

def write_measure_csv(matched: list[MatchedMeasure], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    rows = []
    for m in matched:
        low_conf = m.confidence < LOW_CONFIDENCE_THRESHOLD
        rows.append({
            "measure_index": m.measure_index,
            "note_count": m.note_count,
            "expected_pitches": m.expected_pitches,
            "recorded_dominant_pitch": m.recorded_dominant_pitch,
            "recorded_dominant_frequency_hz": m.recorded_dominant_frequency_hz,
            "expected_start_sec": m.expected_start_sec,
            "start_sec": m.start_sec,
            "end_sec": m.end_sec,
            "audio_end_sec": m.audio_end_sec,
            "measure_length_sec": m.measure_length_sec,
            "timing_offset_sec": m.timing_offset_sec,
            "inter_measure_interval_sec": m.inter_measure_interval_sec,
            "gap_to_next_sec": m.gap_to_next_sec,
            "volume": m.volume,
            "attack_volume": m.attack_volume,
            "sustain_volume": m.sustain_volume,
            "decay_volume": m.decay_volume,
            "volume_range_within": m.volume_range_within,
            "volume_std_within": m.volume_std_within,
            "energy_spike": m.energy_spike,
            "pitch_confidence": m.pitch_confidence,
            "confidence": m.confidence,
            "low_confidence_flag": "YES" if low_conf else "",
        })
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[write_measure_csv] CSV written to {output_path}  ({len(rows)} rows)")
    low_count = sum(1 for r in rows if r["low_confidence_flag"] == "YES")
    if low_count:
        print(f"[write_measure_csv] {low_count} measures flagged as low confidence")

def process_recording_measure(xml_notes: list[XmlNote], score_meta: ScoreMetadata, audio_path: str, output_path: str) -> list[MatchedMeasure]:
    print(f"[process_recording_measure] processing {audio_path}")
    audio_data = analyze_audio(audio_path)

    bpm = estimate_initial_bpm(xml_notes, audio_data["duration_sec"])
    print(f"[process_recording_measure] initial BPM estimate: {bpm} (DTW will warp from here)")

    print("[process_recording_measure] computing audio CQT chroma ...")
    audio_chroma, hop_sec = compute_audio_chroma(audio_data)
    print("[process_recording_measure] synthesizing score chroma ...")
    score_chroma = synthesize_score_chroma(xml_notes, bpm, hop_sec)

    wp = dtw_align(score_chroma, audio_chroma)
    map_time = build_time_mapper(wp, hop_sec)

    measures_grouped = group_by_measure(xml_notes)
    print(f"[process_recording_measure] {len(measures_grouped)} measures "
          f"covering {len(xml_notes)} notes")

    matched = assemble_measures(measures_grouped, audio_data, bpm,
                                map_time, audio_chroma, hop_sec)

    confs = [m.confidence for m in matched]
    if confs:
        low = sum(1 for c in confs if c < LOW_CONFIDENCE_THRESHOLD)
        print(f"[process_recording_measure] mean chroma confidence: {np.mean(confs):.1f}  |  "
              f"{low}/{len(confs)} below {LOW_CONFIDENCE_THRESHOLD}")

    write_measure_csv(matched, output_path)
    return matched
