"""Repair MusicXML files exported by Noteflight 0.4.

The exporter emits each measure's <attributes> block (clef, key, time, staves)
AFTER the notes instead of before them. OSMD then lays out the notes without
knowing the time signature or clefs and the affected measure renders squashed.
This script moves every misplaced <attributes> block to the start of its
measure, writing a <name>.xml.bak alongside each modified file.

Idempotent: running again after a successful fix is a no-op.
"""
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIECES = ["aeolianharp", "ballade", "brahms", "furelise", "grieg", "rach1", "rach2pt2"]

MEASURE_RE = re.compile(r'(<measure\s+number="[^"]+"[^>]*>)(.*?)(</measure>)', re.DOTALL)
ATTRS_RE = re.compile(r'<attributes>.*?</attributes>', re.DOTALL)
NOTE_RE = re.compile(r'<note[\s>]')


def fix_measure_body(body: str) -> tuple[str, bool]:
    attrs = ATTRS_RE.search(body)
    if not attrs:
        return body, False
    note = NOTE_RE.search(body)
    if note is None or note.start() < attrs.start():
        if note is None:
            return body, False
        block = attrs.group(0)
        without = body[: attrs.start()] + body[attrs.end():]
        return block + without, True
    return body, False


def fix_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    fixed_count = 0
    out = []
    last = 0
    for m in MEASURE_RE.finditer(text):
        open_tag, body, close_tag = m.group(1), m.group(2), m.group(3)
        new_body, changed = fix_measure_body(body)
        out.append(text[last:m.start()])
        out.append(open_tag + new_body + close_tag)
        last = m.end()
        if changed:
            fixed_count += 1
    out.append(text[last:])
    if fixed_count == 0:
        return 0
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text("".join(out), encoding="utf-8")
    return fixed_count


def main() -> int:
    total = 0
    for piece in PIECES:
        xml_path = ROOT / piece / f"{piece}.xml"
        if not xml_path.exists():
            print(f"{piece}: file not found at {xml_path}")
            continue
        n = fix_file(xml_path)
        total += n
        if n:
            print(f"{piece}: fixed {n} measure(s)")
        else:
            print(f"{piece}: nothing to fix")
    print(f"\nTotal measures repaired: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
