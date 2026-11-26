'''
note_isolator_dtw.py

Note isolator using DTW score-to-audio alignment. Handles chords, two-hand
passages, and fast runs where onsets blur into each other.

  1. Parse xml *including chord notes*, tracking each note's absolute start
     beat (handles <backup>/<forward> for multi-voice).
  2. Synthesize a 12-bin chroma sequence from the score.
  3. Compute a CQT chromagram from the audio.
  4. DTW-align the two chroma sequences -> warping path (score_f, audio_f).
  5. Map each note's score-time window through the path to recover its
     audio-time window, then compute per-note features.

Entry points (called by main.py):
    parse_xml_dtw(xml_path) -> (xml_notes, score_meta)
    process_recording_dtw(xml_notes, score_meta, audio_path, output_path)
'''

from os import name
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Callable, Optional

import librosa
import numpy as np

from note_isolator_obo import (
    DetectedOnset,
    MatchedNote,
    ScoreMetadata,
    NOTE_NAMES,
    midi_to_freq,
    note_to_midi,
    midi_to_note_name,
    freq_to_note_name,
    sec_to_note_name,
    analyze_audio,
    get_dominant_pitch,
    get_energy_spike,
    compute_dynamic_shape,
    normalise_rms,
    write_csv,
    band_volume,
    band_presence,
    band_energy_spike,
    compute_band_dynamic_shape,
)

'''XML note with absolute start beat (chord notes share their parent's start)'''

@dataclass
class XmlNote:
    index: int
    pitch: str
    frequency_hz: float
    duration_beats: float
    start_beat: float
    measure: int
    voice: int = 1
    is_rest: bool = False

'''Parsing'''

def parse_xml_dtw(xml_path: str) -> tuple[list[XmlNote], ScoreMetadata]: # tracks in-measure cursor (divisions) so <backup>/<forward> and chord notes get correct absolute beat positions
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def strip_ns(tag: str) -> str:
        return tag.split("}")[-1] if "}" in tag else tag

    notes: list[XmlNote] = []
    note_index = 0
    metadata = ScoreMetadata()

    # Pre-scan for <divisions> — some MusicXML exports place the divisions
    # block after the first measure's notes; the default of 1 then inflates
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

            cursor_div = 0           # position within this measure, in divisions
            max_cursor_div = 0       # furthest position reached (= measure length)
            last_note_start_div = 0  # start of the most recent non-chord note

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

    print(f"[parse_xml_dtw] Parsed {len(notes)} notes (incl. chord/poly)  |  "
          f"Time sig: {metadata.time_sig_num}/{metadata.time_sig_den}  |  "
          f"{metadata.total_beats:.1f} beats over {metadata.measure_count} measures")
    return notes, metadata

'''Chroma synthesis and alignment'''

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

'''Per-note feature computation'''

def build_pseudo_onset(start_sec: float, end_sec: float, audio_data: dict, index: int, expected_freq: float = 0.0) -> DetectedOnset: # DetectedOnset built from a DTW-aligned window; per-pitch spike + recorded freq fall back to the expected pitch's band when pyin is dominated by another chord member
    f0 = audio_data["f0"]
    f0_times = audio_data["f0_times"]
    voiced_prob = audio_data["voiced_prob"]
    rms = audio_data["rms"]
    rms_times = audio_data["rms_times"]

    pitch_win_end = min(end_sec, start_sec + 0.2) if end_sec > start_sec else start_sec + 0.2
    freq, conf = get_dominant_pitch(f0, voiced_prob, f0_times, start_sec, pitch_win_end)
    # band-localised attack sharpness for this pitch — chord members no longer share one spike
    spike = band_energy_spike(audio_data, expected_freq, start_sec) if expected_freq > 0 else get_energy_spike(rms, rms_times, start_sec)

    rmask = (rms_times >= start_sec) & (rms_times < end_sec)
    mean_rms = float(np.mean(rms[rmask])) if np.any(rmask) else 0.0

    return DetectedOnset(
        index=index,
        time_sec=start_sec,
        pitch=freq_to_note_name(freq) if freq > 0 else "?",
        frequency_hz=freq,
        pitch_confidence=conf,
        energy_spike=spike,
        rms=mean_rms,
    )

def note_chroma_confidence(audio_chroma: np.ndarray, audio_sec: float, expected_pitch: str, hop_sec: float) -> float: # 0–100: how strong the expected pitch class is vs. the loudest pitch class at the aligned time
    af = int(np.clip(round(audio_sec / hop_sec), 0, audio_chroma.shape[1] - 1))
    col = audio_chroma[:, af]
    peak = float(col.max())
    if peak <= 0:
        return 0.0
    pc = note_name_to_pitch_class(expected_pitch)
    return round(float(col[pc]) / peak * 100.0, 1)

'''Assembly'''

def _sum_skip_nan(vals: list[float]) -> float: # sum treating NaN as missing; returns NaN only if every value is NaN
    clean = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
    return round(float(np.sum(clean)), 4) if clean else float("nan")

def assemble_dtw_matches(xml_notes: list[XmlNote], audio_data: dict, bpm: float, map_time: Callable[[float], float], audio_chroma: np.ndarray, hop_sec: float) -> list[MatchedNote]:
    duration_sec = audio_data["duration_sec"]
    sec_per_beat = 60.0 / bpm

    # Group notes into score events (same start_beat = chord / simultaneous strike).
    # All members of one event share an audio window so the CSV is chronological:
    # end_sec for note i is the START of event i+1, not note i's own score length.
    # Without this, a treble whole and a voice-2 quarter at the same beat produce
    # rows where end_sec jumps backwards from the long note to the short one.
    events: list[tuple[float, list[XmlNote]]] = []
    cur_beat: Optional[float] = None
    cur_notes: list[XmlNote] = []
    for n in xml_notes:
        if cur_beat is None or abs(n.start_beat - cur_beat) < 1e-6:
            if cur_beat is None:
                cur_beat = n.start_beat
            cur_notes.append(n)
        else:
            events.append((cur_beat, cur_notes))
            cur_notes = [n]
            cur_beat = n.start_beat
    if cur_notes:
        events.append((cur_beat, cur_notes))

    event_starts_sec: list[float] = []
    for beat, _ in events:
        s_audio = float(np.clip(map_time(beat * sec_per_beat), 0.0, duration_sec))
        event_starts_sec.append(s_audio)
    # monotonic guard: DTW path is monotonic but float interp can produce
    # ties; force strict non-decreasing so end_sec[i] >= start_sec[i]
    for i in range(1, len(event_starts_sec)):
        if event_starts_sec[i] < event_starts_sec[i - 1]:
            event_starts_sec[i] = event_starts_sec[i - 1]

    event_ends_sec: list[float] = []
    for i in range(len(events)):
        event_ends_sec.append(event_starts_sec[i + 1] if i + 1 < len(events) else duration_sec)

    anchor_audio = event_starts_sec[0] if event_starts_sec else 0.0
    anchor_beat = events[0][0] if events else 0.0

    matched: list[MatchedNote] = []
    for ev_idx, (ev_beat, ev_notes) in enumerate(events):
        start_sec = event_starts_sec[ev_idx]
        end_sec = event_ends_sec[ev_idx]
        note_length_sec = end_sec - start_sec
        note_name = sec_to_note_name(note_length_sec, bpm) if note_length_sec > 0 else "unknown"
        expected_start = anchor_audio + (ev_beat - anchor_beat) * sec_per_beat
        timing_offset = round(start_sec - expected_start, 4)

        # One row per event: chord pitches join with '+', band features sum
        # across the event's notes so volume reflects total chord energy.
        rep_note = XmlNote(
            index=ev_idx,
            pitch="+".join(n.pitch for n in ev_notes),
            frequency_hz=min(n.frequency_hz for n in ev_notes),
            duration_beats=max(n.duration_beats for n in ev_notes),
            start_beat=ev_beat,
            measure=min(n.measure for n in ev_notes),
            voice=min(n.voice for n in ev_notes),
        )

        onset = build_pseudo_onset(start_sec, end_sec, audio_data, ev_idx, rep_note.frequency_hz)

        volume = round(sum(band_volume(audio_data, n.frequency_hz, start_sec, end_sec)
                           for n in ev_notes), 4)
        pitch_conf = round(sum(band_presence(audio_data, n.frequency_hz, start_sec, end_sec)
                               for n in ev_notes), 4)
        shape_vals = [compute_band_dynamic_shape(start_sec, end_sec, audio_data, n.frequency_hz)
                      for n in ev_notes]
        attack_v  = _sum_skip_nan([s[0] for s in shape_vals])
        sustain_v = _sum_skip_nan([s[1] for s in shape_vals])
        decay_v   = _sum_skip_nan([s[2] for s in shape_vals])
        energy_spike = _sum_skip_nan([band_energy_spike(audio_data, n.frequency_hz, start_sec)
                                      for n in ev_notes])
        conf = round(pitch_conf * 100.0, 1)

        matched.append(MatchedNote(
            xml_note=rep_note,
            onset=onset,
            start_sec=round(start_sec, 4),
            end_sec=round(end_sec, 4),
            note_length_sec=round(note_length_sec, 4),
            note_name=note_name,
            volume=volume,
            confidence=conf,
            expected_start_sec=round(expected_start, 4),
            timing_offset_sec=timing_offset,
            attack_volume=attack_v,
            sustain_volume=sustain_v,
            decay_volume=decay_v,
            energy_spike=energy_spike,
            pitch_confidence=pitch_conf,
        ))

    # second pass: IOI / gap — needs the full sequence
    for i, m in enumerate(matched):
        if i + 1 >= len(matched):
            continue
        next_start = matched[i + 1].start_sec
        ioi = next_start - m.start_sec
        m.inter_onset_interval_sec = round(ioi, 4)
        m.gap_to_next_sec = round(next_start - m.end_sec, 4)

    return matched

def estimate_initial_bpm(xml_notes: list[XmlNote], audio_duration_sec: float) -> float: # rough BPM — DTW warps from here, doesn't have to be exact
    if not xml_notes:
        return 120.0
    total_beats = max(n.start_beat + n.duration_beats for n in xml_notes)
    if total_beats <= 0 or audio_duration_sec <= 0:
        return 120.0
    return round(60.0 * total_beats / audio_duration_sec, 1)

def process_recording_dtw(xml_notes: list[XmlNote], score_meta: ScoreMetadata, audio_path: str, output_path: str) -> list[MatchedNote]:
    print(f"[process_recording_dtw] processing {audio_path}")
    audio_data = analyze_audio(audio_path)

    bpm = estimate_initial_bpm(xml_notes, audio_data["duration_sec"])
    print(f"[process_recording_dtw] initial BPM estimate: {bpm} (DTW will warp from here)")

    print("[process_recording_dtw] computing audio CQT chroma ...")
    audio_chroma, hop_sec = compute_audio_chroma(audio_data)
    print("[process_recording_dtw] synthesizing score chroma ...")
    score_chroma = synthesize_score_chroma(xml_notes, bpm, hop_sec)

    wp = dtw_align(score_chroma, audio_chroma)
    map_time = build_time_mapper(wp, hop_sec)

    print("[process_recording_dtw] assembling note records ...")
    matched = assemble_dtw_matches(xml_notes, audio_data, bpm, map_time, audio_chroma, hop_sec)

    confs = [m.confidence for m in matched]
    if confs:
        low = sum(1 for c in confs if c < 45.0)
        print(f"[process_recording_dtw] mean chroma confidence: {np.mean(confs):.1f}  |  "
              f"{low}/{len(confs)} below 45")

    write_csv(matched, output_path)
    return matched

if __name__ == "__main__":
    xmlnotes, score_meta = parse_xml_dtw("files/furelise/furelise.xml")
    matched = process_recording_dtw(xmlnotes, score_meta, "files/furelise/furelise_baseline.wav", "files/furelise/output/csv_files/furelise_baseline_dtw.csv")