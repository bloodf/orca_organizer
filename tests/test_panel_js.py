"""The Organizer panel is a Vue 3 + PrimeVue app (ui/) built into one offline HTML page that is embedded in
shape_arrange.py as PANEL_HTML. This file checks the three links of that chain:

1. the embedded page is exactly what ui/ builds, and is self-contained (no network, no eval, no sibling files);
2. the plugin's English wording and the page's English wording are the same (fixtures written by Python);
3. the Vue component tests (ui/tests, vitest + jsdom) pass: all panel states, messages, focus, protocol.

Run: uv run --python 3.12 --with numpy --with pillow python tests/test_panel_js.py
Regenerate the English fixtures after changing plugin wording: ... python tests/test_panel_js.py --write-fixtures
Needs `node`/`npm` and `npm ci` in ui/; without them the test fails rather than passing silently.
"""

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "ui"
FIXTURES = UI / "tests" / "fixtures" / "english.json"
sys.path.insert(0, str(ROOT))

import shape_arrange as sa  # noqa: E402  (no `orca` module here: only the page source is used)


def english_fixtures() -> list[dict]:
    """(structured message, English text) pairs built by the plugin's own Python, so the page's
    English can never drift from what the plugin writes."""
    label, ref = "P, plate 2", {"stem": "P", "plate": 2}
    out = []
    for code in sa.ERRORS_EN:
        cause = {"code": code, "plate": 3, "plates": 2, "unit": "inch", "res": 0.5}
        for action in ("arrange", "fill"):
            out.append({"msg": {**ref, "action": action, "code": "failed", "cause": cause, "detail": "d"},
                        "text": sa.failure_text(action, label, cause, "d"), "error": True})
        out.append({"msg": {"code": "start_failed", "cause": cause, "detail": "d"},
                    "text": sa.start_failed_text(cause, "d"), "error": True})
    cause = {"code": "unexpected"}
    out.append({"msg": {**ref, "action": "fill", "code": "failed", "cause": cause, "detail": "boom"},
                "text": sa.failure_text("fill", label, cause, "boom"), "error": True})
    out.append({"msg": {"code": "start_failed", "cause": cause, "detail": "boom"},
                "text": sa.start_failed_text(cause, "boom"), "error": True})
    out.append({"msg": {**ref, "action": "arrange", "code": "save_failed", "detail": "disk full"},
                "text": sa.save_failed_text(label, "disk full"), "error": True})
    for code, text in (("cancelled", sa.CANCELLED), ("cancelling", "cancelling…"), ("saving", "saving the arranged copy…")):
        out.append({"msg": {**ref, "action": "arrange", "code": code}, "text": f"{label}: {text}"})
    out.append({"msg": {"stem": None, "plate": 1, "action": "arrange", "code": "cancelled"},
                "text": f"untitled, plate 1: {sa.CANCELLED}"})
    out.append({"msg": None, "text": "unused", "stale": f"The run below is for {label}, not the current selection.",
                "stale_ref": {**ref, "action": "arrange"}})
    reports = [
        ({"placed": 2, "added": sa.MAX_COPIES, "added_by_object": {"a": 150, "b": 150}, "prime_tower": True,
          "spacing": 20.0, "unplaced": ["c", "d"]}, sa.Params(spacing=2, step=0)),
        ({"placed": 1, "added": 0, "added_by_object": {}, "prime_tower": False, "spacing": 2.0, "unplaced": []},
         sa.Params(spacing=2)),
        ({"placed": 5, "added": 1, "added_by_object": {"block": 1}, "prime_tower": False, "spacing": 12.5,
          "unplaced": []}, sa.Params(spacing=2)),
    ]
    for report, p in reports:
        for how in ("asked", "later", "failed", "none"):
            tail = sa.OPEN_EN[how].format(detail="no permission")
            msg = {**ref, "action": "fill", "code": "done", **sa.result_msg(report, p), "open": how,
                   "detail": "no permission"}
            text = sa._summary(report, p, label) + "\nSaved as: /s/P-arranged.3mf\n" + tail
            out.append({"msg": msg, "text": "\n".join(line for line in text.split("\n") if not line.startswith("Saved as: ")),
                        "path": "/s/P-arranged.3mf"})
    return out


def fixtures_json() -> str:
    return json.dumps({"error_codes": sorted(sa.ERRORS_EN), "cases": english_fixtures()}, indent=1, ensure_ascii=False) + "\n"


def tool(name: str) -> str:
    exe = shutil.which(name)
    assert exe, f"{name} is required for the panel tests"
    return exe


def test_english_fixtures_are_current():
    assert FIXTURES.is_file(), f"{FIXTURES} is missing: run this file with --write-fixtures"
    assert FIXTURES.read_text(encoding="utf-8") == fixtures_json(), (
        "plugin wording changed: run `python tests/test_panel_js.py --write-fixtures` and make the Vue panel match")


def test_embedded_page_is_the_built_page():
    html = sa.PANEL_HTML
    sha = hashlib.sha256(html.encode("utf-8")).hexdigest()
    assert sha == sa.PANEL_HTML_SHA256, "embedded page does not match its declared hash"
    built = UI / "dist" / "index.html"
    if built.is_file():  # always the case in CI, after `npm run build`
        assert hashlib.sha256(built.read_bytes()).hexdigest() == sha, "stale: run `npm run build` in ui/"
    assert html.startswith("<!doctype html>") and '<html lang="en">' in html and "<title>Organizer</title>" in html
    assert '<div id="app"></div>' in html


def test_embedded_page_is_offline_and_needs_no_unsafe_eval():
    html = sa.PANEL_HTML
    assert not re.search(r"<script[^>]*\ssrc=", html), "external script"
    assert not re.search(r"<link[^>]*\shref=", html), "external stylesheet or preload"
    assert not re.search(r"@import|url\(\s*[\"']?(?!data:)", html), "external CSS resource"
    assert not re.search(r"<iframe|<object|<embed", html, re.I)
    assert not re.search(r"\beval\(|new Function|\bFunction\(", html), "needs unsafe-eval: templates must be compiled ahead of time"
    assert not re.search(r"\bfetch\(|XMLHttpRequest|WebSocket|sendBeacon|importScripts", html), "the page only talks to the host bridge"
    urls = set(re.findall(r"https?://[^\s\"'`)<>]+", html))
    allowed = ("http://www.w3.org/", "https://vuejs.org/error-reference")  # XML namespaces and a Vue error-doc string, never loaded
    assert all(u.startswith(allowed) for u in urls), urls
    assert "bootstrap" not in html.lower() and "cdn." not in html.lower()


def test_vue_component_tests_pass():
    assert (UI / "node_modules").is_dir(), "run `npm ci` in ui/ first"
    r = subprocess.run([tool("npm"), "test", "--silent"], cwd=UI, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-6000:] + r.stderr[-2000:]
    print(re.sub(r"\x1b\[[0-9;]*m", "", r.stdout).strip().splitlines()[-4:])


if __name__ == "__main__":
    if "--write-fixtures" in sys.argv:
        FIXTURES.parent.mkdir(parents=True, exist_ok=True)
        FIXTURES.write_text(fixtures_json(), encoding="utf-8")
        print(f"wrote {FIXTURES}")
        sys.exit(0)
    test_english_fixtures_are_current()
    test_embedded_page_is_the_built_page()
    test_embedded_page_is_offline_and_needs_no_unsafe_eval()
    test_vue_component_tests_pass()
    print("ok")
