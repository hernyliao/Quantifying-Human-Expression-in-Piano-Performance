from pathlib import Path
from utils import measure_isolator, human_qualities

RUN_ISOLATOR = False  # set to False to skip the isolator and only run human qualities on existing CSVs

PROJECT_ROOT = Path(__file__).resolve().parent
FILES_ROOT = PROJECT_ROOT / "files"
# SKIP_FOLDERS = {"archive", "_summary", "aeolianharp", "ballade", "brahms", "furelise", "grieg", "chopinconcerto", "rach1", "rach2pt2"}
SKIP_FOLDERS = {"archive", "_summary"}  # for quick testing on a subset of folders

ISOLATORS = [("measure", measure_isolator.parse_xml_measure, measure_isolator.process_recording_measure)]

def find_xml(files_dir: Path) -> Path:
    return list(files_dir.glob("*.xml"))[0]

def find_wavs(files_dir: Path) -> list[Path]:
    return sorted(files_dir.glob("*.wav"))

def split_role(stem: str, isolator: str) -> tuple[str, str] | None:
    suffix = f"_{isolator}"
    if not stem.endswith(suffix):
        return None
    base = stem[: -len(suffix)]
    for role in ("baseline", "performance", "pro"):
        marker = f"_{role}"
        if base.endswith(marker):
            return base[: -len(marker)], role
    return None

def run_folder(folder_name: str, run_isolator: bool = True):
    files_dir = FILES_ROOT / folder_name
    output_dir = files_dir / "output"
    csv_dir = output_dir / "csv_files"
    visuals_dir = output_dir / "visuals"

    if run_isolator:
        xml_matches = list(files_dir.glob("*.xml"))
        wav_paths = find_wavs(files_dir)
        if not xml_matches or not wav_paths:
            print(f"[main_all] Skipping {folder_name}: missing xml or wavs")
            return

        xml_path = xml_matches[0]
        csv_dir.mkdir(parents=True, exist_ok=True)
        visuals_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n############## Folder: {folder_name} ##############")
        print(f"[main_all] XML:    {xml_path.name}")
        print(f"[main_all] WAVS:   {[m.name for m in wav_paths]}")

        for isolator_name, parser, processor in ISOLATORS:
            print(f"\n##### Isolator: {isolator_name} #####")
            xml_notes, score_meta = parser(str(xml_path))
            for wav in wav_paths:
                csv_path = csv_dir / f"{wav.stem}_{isolator_name}.csv"
                processor(xml_notes, score_meta, str(wav), str(csv_path))
    else:
        if not csv_dir.exists():
            print(f"[main_all] Skipping {folder_name}: no existing csv_files dir")
            return
        visuals_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n############## Folder: {folder_name} (skip isolator) ##############")

    print("\n##### Human qualities #####")
    csv_paths = sorted(csv_dir.glob("*_measure.csv"))
    source_rows: dict[str, list[dict]] = {}
    for path in csv_paths:
        rows = human_qualities.read_measure_csv(path)
        if rows:
            source_rows[path.stem] = rows
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

        out_csv = output_dir / f"{path.stem}_human_qualities.csv"
        human_qualities.write_human_quality_csv(analysed, out_csv)
        human_qualities.generate_plots(analysed, path.stem, visuals_dir)

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
                    cmp_path = visuals_dir / f"{piece}_{isolator_name}{suffix}.png"
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
                    print(f"[main_all] comparison ({role_a} vs {role_b}) -> {cmp_path}")
            if all(r in roles for r in ("baseline", "performance", "pro")):
                cmp_all_path = visuals_dir / f"{piece}_{isolator_name}__comparison_all.png"
                human_qualities.plot_comparison_all(
                    {r: analyses[roles[r]] for r in ("baseline", "performance", "pro")},
                    cmp_all_path,
                    f"{piece} ({isolator_name})",
                    human_qualities.ROLE_COLORS,
                )
                print(f"[main_all] comparison (all) -> {cmp_all_path}")

def build_cross_piece_summary():
    print("\n############## Cross-piece humanness summary ##############")
    summaries = []
    for piece_dir in sorted(p for p in FILES_ROOT.iterdir() if p.is_dir() and p.name not in {"archive", "_summary"}):
        csv_dir = piece_dir / "output" / "csv_files"
        if not csv_dir.exists():
            continue
        # gather rows once so we can fit this piece's baseline volume->note_count
        # line before scoring (see human_qualities.fit_baseline_volume)
        piece_rows: dict[str, tuple[str, list[dict]]] = {}
        baseline_fit: tuple[float, float] | None = None
        for measure_csv in sorted(csv_dir.glob("*_measure.csv")):
            split = split_role(measure_csv.stem, "measure")
            if split is None:
                continue
            piece, role = split
            rows = human_qualities.read_measure_csv(measure_csv)
            if not rows:
                continue
            piece_rows[measure_csv.stem] = (role, rows)
            if role == "baseline":
                baseline_fit = human_qualities.fit_baseline_volume(rows)
        for stem, (role, rows) in piece_rows.items():
            split = split_role(stem, "measure")
            if split is None:
                continue
            piece, _ = split
            analysed = human_qualities.analyse_human_qualities(rows, volume_fit=baseline_fit)
            summaries.append(human_qualities.summarise_recording(analysed, piece, role))

    if not summaries:
        print("[main_all] no measure CSVs found for summary")
        return

    summary_dir = FILES_ROOT / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    human_qualities.write_summary_csv(summaries, summary_dir / "humanness_summary.csv")
    human_qualities.plot_humanness_summary(summaries, summary_dir / "humanness_summary.png")

def main():
    folders = sorted(p.name for p in FILES_ROOT.iterdir() if p.is_dir() and p.name not in SKIP_FOLDERS)
    print(f"[main_all] Running folders: {folders} (run_isolator={RUN_ISOLATOR})")
    for folder in folders:
        run_folder(folder, run_isolator=RUN_ISOLATOR)
    build_cross_piece_summary()
    print("\n[Done]")

main()
