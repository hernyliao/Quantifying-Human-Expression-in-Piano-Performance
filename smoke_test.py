"""Validate note-level highlight + scroll fix. Reads ISOLATOR from play_and_verify."""
import json
import time
from pathlib import Path

import webview

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.play_and_verify import Api, SCORE_XML_PATH, VIEWER_HTML, PIECE, ROLE, ISOLATOR  # noqa: E402


def js_call(window, fn, *args):
    args_json = ", ".join(json.dumps(a) for a in args)
    return window.evaluate_js(f"{fn}({args_json})")


def run(window):
    xml_content = Path(SCORE_XML_PATH).read_text(encoding="utf-8")
    js_call(window, "loadScore", xml_content, f"{PIECE} — {ROLE} ({ISOLATOR})")

    for _ in range(300):
        if window.evaluate_js("scoreReady === true && sessionReady === true"):
            break
        time.sleep(0.1)
    else:
        print("[smoke] never reached ready state")
        window.destroy(); return

    typ = window.evaluate_js("sessionData.type")
    n = window.evaluate_js("sessionData.rows.length")
    print(f"[smoke] session type={typ}, rows={n}")

    # Show first row's note_in_measure (None for measure mode)
    nim = window.evaluate_js("sessionData.rows[0].note_in_measure || null")
    print(f"[smoke] rows[0].note_in_measure={nim}")

    # Verify frac_in_measure populated and within range
    fracs = window.evaluate_js(
        "sessionData.rows.slice(0, 12).map(r => [r.measure, r.note_in_measure, r.frac_in_measure])"
    )
    print(f"[smoke] first 12 rows (measure, n_in_m, frac): {fracs}")

    # Advance through 4 rows checking scroll & highlight
    for i in range(4):
        cur = window.evaluate_js("currentIdx")
        m = window.evaluate_js("sessionData.rows[currentIdx].measure")
        nim = window.evaluate_js("sessionData.rows[currentIdx].note_in_measure || null")
        scroll = window.evaluate_js("document.getElementById('score-wrapper').scrollTop")
        rects = window.evaluate_js("document.querySelectorAll('.highlight-rect').length")
        rect_cls = window.evaluate_js(
            "(document.querySelector('.highlight-rect') || {}).getAttribute && "
            "document.querySelector('.highlight-rect').getAttribute('class')"
        )
        print(f"[smoke] step{i}: idx={cur} m={m} n_in_m={nim} scrollTop={scroll:.0f} rects={rects} cls={rect_cls!r}")
        # next without playing (avoid audio dep)
        window.evaluate_js("next()")
        time.sleep(0.5)

    # Jump to a later measure to confirm scroll moves down (not to top)
    window.evaluate_js("cancelAdvance(); currentIdx = Math.min(sessionData.rows.length-1, 30); showCurrent();")
    time.sleep(0.8)
    scroll = window.evaluate_js("document.getElementById('score-wrapper').scrollTop")
    max_scroll = window.evaluate_js("document.getElementById('score-wrapper').scrollHeight - document.getElementById('score-wrapper').clientHeight")
    print(f"[smoke] after jump to later row: scrollTop={scroll:.0f} / maxScroll={max_scroll:.0f}")

    time.sleep(0.5)
    window.destroy()


def main():
    api = Api()
    window = webview.create_window(
        "Smoke Test", str(VIEWER_HTML), js_api=api, width=1400, height=900,
    )
    webview.start(run, window)


if __name__ == "__main__":
    main()
