"""Plugin glue against a stub `orca` module: UI-thread export, worker nesting, UI-thread finish.

Run: uv run --python 3.12 --with numpy --with pillow python tests/test_plugin.py
"""

import shutil
import sys
import tempfile
import threading
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def install_stub(tmp: Path, project: Path):
    ui_thread = threading.get_ident()
    log = {"export_thread": None, "open_thread": None, "posts": [], "perms": None}

    class Panel:
        def __init__(self, on_message, **_):
            self.on_message = on_message
            self.done = threading.Event()

        def post(self, msg):
            log["posts"].append(msg)
            # the real page relays {"relay": ...} back as a message on the UI thread;
            # emulate by queueing it for the "UI loop" below
            if msg.get("relay"):
                pending.append({"action": msg["relay"]})
            if msg.get("done"):
                self.done.set()

        def is_open(self):
            return True

    class Plater:
        project_path = str(tmp / "My Project.3mf")

        def export_3mf_copy(self, path):
            log["export_thread"] = threading.get_ident()
            shutil.copy(project, path)

    pending = []
    orca = types.ModuleType("orca")
    orca.base = object
    orca.plugin = lambda c: log.setdefault("plugin", c)
    orca.register_capability = lambda c: log.setdefault("cap", c)
    orca.request_permissions = lambda **kw: log.update(perms=kw)
    orca.script = types.SimpleNamespace(ScriptPluginCapabilityBase=object)
    orca.ExecutionResult = types.SimpleNamespace(success=lambda m: m)
    orca.host = types.SimpleNamespace(
        model=lambda: types.SimpleNamespace(current_plate_index=1),  # 0-based -> plate 2
        plater=lambda: Plater(),
        plugin=types.SimpleNamespace(storage=lambda: str(tmp / "storage")),
        ui=types.SimpleNamespace(create_dock_panel=lambda **kw: Panel(**kw)),
    )
    sys.modules["orca"] = orca
    return log, pending, ui_thread


def test_arrange_runs_off_ui_thread_and_finishes_on_it():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        import test_shape_arrange as t  # builds projects; import before the stub is fine

        project = tmp / "proj.3mf"
        t.make_project(project)
        log, pending, ui_thread = install_stub(tmp, project)
        sys.modules.pop("shape_arrange", None)
        import shape_arrange as sa

        opened = []
        sa._open_in_orca = lambda p: (opened.append((p, threading.get_ident())), True)[1]
        log["plugin"]().register_capabilities()
        assert log["perms"] == {"threading": True}
        log["cap"]().execute()
        panel = sa._panel

        panel.on_message({"action": "arrange", "spacing": 1})
        panel.on_message({"action": "arrange", "spacing": 1})  # double click while busy: ignored
        while not panel.done.is_set():  # the "UI loop": deliver relayed messages on this thread
            while pending:
                panel.on_message(pending.pop(0))
            panel.done.wait(0.05)

        final = log["posts"][-1]
        assert final["done"] and final["text"].startswith("Plate 2: placed 3"), final
        assert any("Placing" in m["text"] for m in log["posts"]), "no progress posted"
        assert log["export_thread"] == ui_thread
        assert opened and opened[0][1] == ui_thread, "launch must happen on the UI thread"
        assert opened[0][0].name == "My Project-arranged.3mf" and opened[0][0].exists()
        assert sum(1 for m in log["posts"] if m.get("relay")) == 1, "second click started a second job"


if __name__ == "__main__":
    test_arrange_runs_off_ui_thread_and_finishes_on_it()
    print("ok")
