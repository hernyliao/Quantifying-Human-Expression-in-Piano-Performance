# Quantifying Human Expression in Piano Performance

**Henry Liao · Senior Final Project · Geffen Academy at UCLA**

A Python project exploring how expressive qualities in piano performances can be represented numerically. It aligns a MusicXML score with recorded audio, extracts timing and sound-energy measurements, and generates visual comparisons between a mechanically generated baseline, my performance, and a professional performance of the same passage.

The project brings together my interests in piano, programming, and data analysis: can a computer describe some of the timing and dynamic choices that make a performance sound human?

## Project overview

The starting point was a score and recordings of the same music. Before comparing performances, I needed to determine where each note or measure occurred in the audio, even when performers changed tempo or lingered over a phrase.

I explored three note-level alignment methods before using a measure-level Dynamic Time Warping (DTW) approach for the final analysis. The pipeline exports structured CSV data, calculates experimental expression metrics, and produces plots for individual recordings and comparisons across recordings and pieces.

The score viewer is a presentation and inspection tool. It connects highlighted sections of the score to audio playback to demonstrate both measure-level and note-level isolation.

## Alignment methods explored

| Method | Approach | Role in the project |
| --- | --- | --- |
| One-by-one matching (`obo`) | Detect audio onsets and match them sequentially to score notes using pitch, attack, and timing information. | Early experiment; vulnerable to ambiguous chords and cascading matching errors. |
| Simulated matching (`sm`) | Compare score-derived pitch-class templates with candidate audio positions, group simultaneous notes, and backtrack after poor matches. | Chord-aware note/event alignment experiment and demonstration. |
| Note-level DTW (`dtw`) | Align score-derived and recorded pitch-class sequences globally, then map score notes onto audio times. | Explored alignment across tempo changes and complex passages. |
| Measure-level DTW (`measure`) | Use global alignment to estimate measure boundaries and extract measurements for each measure. | Final analysis pipeline used by `main.py` and `main_all.py`. |

“Isolation” here means locating time spans in a recording. It does not separate an individual note or instrument from the mixed audio.

## Final analysis pipeline

1. **Parse the score.** Read pitches, durations, voices, chord membership, and measure positions from MusicXML.
2. **Analyze the recording.** Extract pitch estimates, RMS energy, and constant-Q transform (CQT) features with librosa.
3. **Align score and audio.** Construct a score-derived chroma representation and use DTW to map it to the recording despite timing differences.
4. **Extract measure data.** Estimate boundaries and calculate duration, timing offsets, pitch-band energy, attack/sustain/decay measurements, silence estimates, and alignment confidence.
5. **Compare expression.** Calculate the metrics below and export analysis CSVs, plots, and cross-piece summaries.

## Expression metrics

| Concept | Implementation |
| --- | --- |
| **Rubato** | Ratio of recorded to expected intervals between measure starts, a three-measure smoothed trend, and local variability. Expected timing is derived from score beats and a recording-specific initial tempo estimate. |
| **Micro-timing** | Measure-level timing variation remaining after subtracting the locally smoothed rubato trend, expressed in seconds. |
| **Breath** | A silence-based phrasing proxy, using the longest qualifying low-energy run within an aligned measure slot. Despite the name `gap_to_next_sec`, the detected silence can occur inside the slot. |
| **Dynamic shaping** | Pitch-band energy summaries, attack/sustain/decay differences, and smoothed volume trends. Scoring uses per-note normalization or a baseline volume-versus-note-count reference when supplied. |
| **Tension and release** | A heuristic combination of standardized volume, estimated pitch height, and measure-start spacing, followed by changes in that combined signal. |
| **Humanness score** | A weighted combination of normalized expression sub-scores, intended for exploration and visualization. |

The current composite weights are **45% rubato, 20% tension, 15% dynamics, 10% micro-timing, and 10% breath**. These weights and thresholds are manually chosen. The score is not a trained classifier, a probability that a performance is human, or an objective judgment of musical quality.

## Repository organization

The paths below use the original Python module names. If downloaded files have upload suffixes such as `(1)`, remove those suffixes when arranging the repository so imports resolve correctly.

| Path | Purpose |
| --- | --- |
| `main.py` | Run the final measure-based pipeline for one piece. |
| `main_all.py` | Process piece folders, generate comparison plots, and build a cross-piece summary. |
| `utils/measure_isolator.py` | Score parsing, DTW alignment, and raw measure-feature extraction. |
| `utils/human_qualities.py` | Expression metrics, plots, comparison figures, and summary tables. |
| `utils/note_isolator_obo.py` | Earlier one-by-one note matching. |
| `utils/note_isolator_sm.py` | Simulated/chord-aware matching with backtracking. |
| `utils/note_isolator_dtw.py` | Note-level DTW alignment. |
| `utils/play_and_verify.py` | Python bridge between the score viewer, alignment CSVs, and audio playback. |
| `utils/fix_score_xml.py` | Repair helper for misplaced MusicXML attribute blocks in certain exports. |
| `viewer/score_viewer.html` | Presentation interface for highlighting and playback. |
| `viewer/opensheetmusicdisplay.min.js` | Local OpenSheetMusicDisplay dependency for rendering notation. |
| `files/<piece>/` | Each piece’s MusicXML score, WAV recordings, and generated outputs. |

## Setup

Use **Python 3.10 or newer**; the code uses modern type annotations. The analysis imports NumPy, librosa, and Matplotlib:

```bash
python -m pip install numpy librosa matplotlib
```

For the optional presentation viewer, also install:

```bash
python -m pip install sounddevice pywebview
```

The viewer requires desktop webview support and an available audio output device. These dependency lists are inferred from the supplied source; exact versions from the original working environment still need to be recorded.

### Prepare a piece folder

Place one uncompressed `.xml` MusicXML score and the corresponding `.wav` recordings in `files/<piece>/`. Use the same passage and ordering in the score and recordings.

For example:

| Input | Example path |
| --- | --- |
| Score | `files/furelise/furelise.xml` |
| Mechanical baseline | `files/furelise/furelise_baseline.wav` |
| My performance | `files/furelise/furelise_performance.wav` |
| Professional performance | `files/furelise/furelise_pro.wav` |

The role suffixes `_baseline`, `_performance`, and `_pro` allow the runners to pair recordings for comparison plots. Comparisons are produced when the relevant roles are available. The runner takes the first `.xml` file it finds, so keep only the intended score in each piece folder.

## Run the analysis

### One piece

In `main.py`, set the target folder:

```python
FOLDER_NAME = "furelise"
```

Then run from the project root:

```bash
python main.py
```

The supplied single-piece runner defaults to `rach2pt2`. It runs the measure isolator for every WAV in that folder, then generates expression analyses and pairwise comparison plots.

### All pieces

For a first run from audio, change this setting in `main_all.py`:

```python
RUN_ISOLATOR = True
```

Then run:

```bash
python main_all.py
```

Set `RUN_ISOLATOR = False` to reuse existing measure CSVs and regenerate analyses without repeating audio alignment. The supplied batch runner defaults to this reuse mode and skips folders without existing CSVs. It excludes `archive` and `_summary` from normal piece processing.

### Outputs

| Output path | Contents |
| --- | --- |
| `files/<piece>/output/csv_files/<recording>_measure.csv` | Raw alignment and measure measurements. |
| `files/<piece>/output/<recording>_measure_human_qualities.csv` | Derived expression metrics and sub-scores. |
| `files/<piece>/output/visuals/` | Individual metric plots, overview plots, and available recording comparisons. |
| `files/_summary/humanness_summary.csv` | Batch summary across pieces and roles, including note-count-weighted mean scores and percentile ranges. |
| `files/_summary/humanness_summary.png` | Batch comparison figure. |

The batch runner also generates a three-recording comparison when baseline, performance, and pro recordings are all available.

## Presentation score viewer

The viewer displays the score, highlights the current measure or approximate note position, and plays its aligned audio segment. It reads an existing alignment CSV; it does not run the isolators itself.

In `utils/play_and_verify.py`, select the piece, recording role, and alignment method. For example, after generating the measure CSV:

```python
PIECE = "furelise"
ROLE = "baseline"
ISOLATOR = "measure"
```

Then launch from the project root:

```bash
python utils/play_and_verify.py
```

The supplied configuration uses `ISOLATOR = "sm"`, which expects a previously generated simulated-matching CSV. The final main runners generate only measure CSVs. For the supplied Für Elise baseline example, the earlier note experiments can be run separately with:

```bash
python utils/note_isolator_sm.py
python utils/note_isolator_dtw.py
```

Their example input and output paths are set in the scripts. Select `sm` or `dtw` in the viewer to inspect the corresponding result. Keep `score_viewer.html` and `opensheetmusicdisplay.min.js` together in `viewer/`.

| Key | Action |
| --- | --- |
| Space or Enter | Play the current segment and advance the selection when playback finishes. |
| Left / Right | Move to the previous / next segment without audio. |
| R | Replay the current segment. |
| Escape | Stop playback and cancel pending advancement. |

Note highlighting uses an approximate position within a measure; it is a visual aid for demonstrating alignment rather than a precise mapping to every engraved notehead.

## Interpretation and current limitations

- Alignment quality determines the usefulness of every downstream measurement. Pedal blur, repeated pitch patterns, quiet attacks, and mismatched passages can shift boundaries; use the viewer and confidence indicators to inspect uncertain sections.
- Chroma groups pitches by pitch class, which can make octaves and repeated harmonies ambiguous. Dominant-pitch estimates are also limited in polyphonic piano audio.
- Recordings differ in microphone placement, room acoustics, gain, instrument, and processing. Energy-based features should not be interpreted as calibrated loudness or pure performer intent.
- The metrics are musical proxies. Detected silence is not necessarily intentional phrasing, and a numerical tension signal does not establish a listener’s experience of tension.
- Some code estimates expression at measure resolution, so the label “micro-timing” does not imply a validated analysis of individual keypresses.
- The per-piece baseline-fit lookup currently expects a different filename pattern from the generated measure CSVs, so dynamics scoring can fall back to within-recording normalization. The batch cross-piece summary supplies a baseline fit directly when one is present.
- The XML repair helper has its own path assumptions. Check its `ROOT` and target paths before using it with the `files/<piece>/` layout; it modifies matching scores and creates `.xml.bak` backups.

This README was checked against the supplied source and presentation. Audio recordings, scores, and generated result files were not supplied for this review, so no performance rankings, alignment-accuracy measurements, or end-to-end execution results are claimed here.

## Reflection and next steps

A central lesson was that meaningful comparisons depend on reliable alignment. Exploring several isolators made the tradeoffs between local matching, error recovery, and global alignment visible before moving to the final measure-based analysis.

Future work could validate boundaries against manually annotated recordings, evaluate the expression metrics against listener judgments, standardize recording conditions, and replace hand-selected composite weights with empirically tested alternatives.

The presentation proposes broader questions about recognizing human performance and evaluating musical interpretation. The implemented project provides tools for exploring those questions, rather than establishing an objective ranking of performances.

## Third-party component

The presentation viewer uses OpenSheetMusicDisplay. Preserve its upstream license notices when including the JavaScript bundle; the supplied bundle header also references `opensheetmusicdisplay.min.js.LICENSE.txt`.
