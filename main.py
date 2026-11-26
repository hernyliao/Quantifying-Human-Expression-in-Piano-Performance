from pathlib import Path
from utils import measure_isolator, human_qualities

FOLDER_NAME = "rach2pt2"

PROJECT_ROOT = Path(__file__).resolve().parent
FILES_DIR = PROJECT_ROOT / "files" / FOLDER_NAME
OUTPUT_DIR = FILES_DIR / "output"
CSV_DIR = OUTPUT_DIR / "csv_files"
VISUALS_DIR = OUTPUT_DIR / "visuals"

# Measure-based pipeline only — note-by-note isolators have been retired.
ISOLATORS = [("measure", measure_isolator.parse_xml_measure, measure_isolator.process_recording_measure)]

def find_xml(files_dir: Path) -> Path:
    return list(files_dir.glob("*.xml"))[0]

def find_wavs(files_dir: Path) -> list[Path]:
    return sorted(files_dir.glob("*.wav"))

def split_role(stem: str, isolator: str) -> tuple[str, str] | None: # (piece, role) role is "baseline" or "performance"
    suffix = f"_{isolator}"
    if not stem.endswith(suffix):
        return None
    base = stem[: -len(suffix)]
    for role in ("baseline", "performance", "pro"):
        marker = f"_{role}"
        if base.endswith(marker):
            return base[: -len(marker)], role
    return None

def main():
    xml_path  = find_xml(FILES_DIR)
    wav_paths = find_wavs(FILES_DIR)

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[main] Folder: {FOLDER_NAME}")
    print(f"[main] XML:    {xml_path.name}")
    print(f"[main] WAVS:   {[m.name for m in wav_paths]}")

    # 1) run measure isolator on every recording
    for isolator_name, parser, processor in ISOLATORS:
        print(f"\n##### Isolator: {isolator_name} #####")
        xml_notes, score_meta = parser(str(xml_path))
        for wav in wav_paths:
            csv_path = CSV_DIR / f"{wav.stem}_{isolator_name}.csv"
            processor(xml_notes, score_meta, str(wav), str(csv_path))

    # 2) human-quality analysis per measure CSV
    print("\n##### Human qualities #####")
    csv_paths = sorted(CSV_DIR.glob("*_measure.csv"))
    source_rows: dict[str, list[dict]] = {}
    for path in csv_paths:
        rows = human_qualities.read_measure_csv(path)
        if rows:
            source_rows[path.stem] = rows
    # fit each piece's baseline volume->note_count line as a score-structural
    # reference for dyn_score (see human_qualities.fit_baseline_volume)
    volume_fits: dict[str, tuple[float, float]] = {}
    for stem, rows in source_rows.items():
        if "_baseline_" in stem:
            piece = human_qualities.piece_name(stem)
            if piece is not None:
                volume_fits[piece] = human_qualities.fit_baseline_volume(rows)
    analyses: dict[str, list[dict]] = {}
    for path in csv_paths:
        rows = source_rows.get(path.stem)
        if not rows:
            continue
        print(f"\n=== Analysing {path.name} ({len(rows)} measures) ===")
        fit = volume_fits.get(human_qualities.piece_name(path.stem) or "")
        analysed = human_qualities.analyse_human_qualities(rows, volume_fit=fit)
        analyses[path.stem] = analysed

        out_csv = OUTPUT_DIR / f"{path.stem}_human_qualities.csv"
        human_qualities.write_human_quality_csv(analysed, out_csv)
        human_qualities.generate_plots(analysed, path.stem, VISUALS_DIR)

    # 3) pair recordings by isolator suffix and emit comparison plots
    #    baseline vs performance keeps its legacy filename; pro pairings get explicit suffixes
    comparison_pairs = [
        ("performance", "baseline", "__comparison"),
        ("pro", "performance", "__comparison_pro_vs_performance"),
        ("pro", "baseline", "__comparison_pro_vs_baseline"),
    ]
    for isolator_name, _, _ in ISOLATORS:
        pairs: dict[str, dict[str, str]] = {}
        for stem in analyses:
            split = split_role(stem, isolator_name)
            if split is None:
                continue
            piece, role = split
            pairs.setdefault(piece, {})[role] = stem

        for piece, roles in pairs.items():
            for role_a, role_b, suffix in comparison_pairs:
                if role_a in roles and role_b in roles:
                    cmp_path = VISUALS_DIR / f"{piece}_{isolator_name}{suffix}.png"
                    human_qualities.plot_comparison(
                        analyses[roles[role_a]],
                        analyses[roles[role_b]],
                        cmp_path,
                        f"{piece} ({isolator_name})",
                        label_a=role_a,
                        label_b=role_b,
                        color_a=human_qualities.ROLE_COLORS[role_a],
                        color_b=human_qualities.ROLE_COLORS[role_b],
                    )
                    print(f"[main] comparison ({role_a} vs {role_b}) -> {cmp_path}")

    print("\n[Done]")

main()
