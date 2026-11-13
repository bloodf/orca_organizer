"""Plugin glue against a stub `orca` module: UI-thread export, worker nesting, UI-thread finish,
plus the job lifecycle: close/reopen, cancel, failed writes, earlier results kept.

Run: uv run --python 3.12 --with numpy --with pillow python tests/test_plugin.py

Races are driven with events (a gate inside the nesting call), never with sleeps.
Every wait has a deadline, so a broken finish fails the test instead of hanging it.
"""

import hashlib
import os
import shutil
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import test_shape_arrange as t  # noqa: E402  builds projects; import before the stub is fine

DEADLINE = 60.0


def install_stub(tmp: Path, project: Path, pages: bool = True, base_hooks: bool = True,
                 page_register_fails: bool = False):
    """`pages=False` is a host without Pages; `base_hooks=False` a Pages base without on_load/on_unload."""
    ui_thread = threading.get_ident()
    log = {"export_thread": None, "exports": [], "posts": [], "perms": None, "panels": [],
           "project": project, "project_path": str(tmp / "My Project.3mf"), "plate_index": 1,
           "objects": 3, "caps": [], "page_posts": [], "hooks": [], "lang": "pt_BR"}

    class Panel:
        def __init__(self, on_message, **_):
            self.on_message = on_message
            self.open = True
            self.done = threading.Event()
            log["panels"].append(self)

        def post(self, msg):
            if not self.open:  # a closed page drops messages, relays included
                return
            log["posts"].append(msg)
            # the real page relays {"relay": ...} back as a message on the UI thread;
            # emulate by queueing it for the "UI loop" below
            if msg.get("relay"):
                pending.append((self, {"action": msg["relay"], "job": msg["job"]}))
            if msg.get("done"):
                self.done.set()

        def is_open(self):
            return self.open

        def show(self):
            pass

        def close(self):
            self.open = False

    class Plater:
        project_path = property(lambda self: log["project_path"])

        def export_3mf_copy(self, path):
            log["export_thread"] = threading.get_ident()
            log["exports"].append(Path(path))
            shutil.copy(log["project"], path)

    def storage():
        # Like OrcaSlicer 2.5: only answered inside a plugin callback (here: while registering), never from a
        # panel or page message, where the real host raises this exact error.
        if not log.get("in_callback"):
            raise RuntimeError("plugin.storage() must be called from a plugin callback")
        return str(tmp / "storage")

    pending = []
    orca = types.ModuleType("orca")
    orca.base = object
    orca.plugin = lambda c: log.setdefault("plugin", c)

    def register(c):
        if page_register_fails and c.__name__ == "OrganizerPage":
            raise ValueError("Pages is not available in this host")
        log["caps"].append(c)

    orca.register_capability = register
    orca.request_permissions = lambda **kw: log.update(perms=kw)
    orca.script = types.SimpleNamespace(ScriptPluginCapabilityBase=object)
    orca.ExecutionResult = types.SimpleNamespace(success=lambda m: m)
    orca.host = types.SimpleNamespace(
        model=lambda: types.SimpleNamespace(current_plate_index=log["plate_index"],  # 0-based
                                            objects=lambda: list(range(log["objects"]))),
        plater=lambda: Plater(),
        plugin=types.SimpleNamespace(storage=storage),
        app_language=lambda: log["lang"],
        ui=types.SimpleNamespace(create_dock_panel=lambda **kw: Panel(**kw)),
    )
    if pages:
        class PagesBase:
            def post_message(self, data):  # the host's bridge to orca.onMessage
                log["page_posts"].append((dict(data), threading.get_ident()))

        if base_hooks:
            # Like OrcaSlicer's pybind11 base: on_load/on_unload are C++ virtuals, so calling the base from an
            # override dispatches back to the override. The real host recurses until the stack overflows.
            def virtual(name):
                def hook(self):
                    log["hooks"].append("base " + name)
                    if len(log["hooks"]) > 20:
                        raise AssertionError("base %s re-entered the override: infinite recursion" % name)
                    getattr(type(self), name)(self)
                return hook

            PagesBase.on_load = virtual("on_load")
            PagesBase.on_unload = virtual("on_unload")
        orca.pages = types.SimpleNamespace(PagesPluginCapabilityBase=PagesBase)
    sys.modules["orca"] = orca
    return log, pending, ui_thread


def load(tmp: Path, project: Path, **host):
    log, pending, ui_thread = install_stub(tmp, project, **host)
    sys.modules.pop("shape_arrange", None)
    import shape_arrange as sa

    opened = []
    sa._open_in_orca = lambda p: (opened.append((p, threading.get_ident())), True)[1]
    log["in_callback"] = True
    log["plugin"]().register_capabilities()
    log["in_callback"] = False
    log["cap"] = log["caps"][0]  # the Script
    log["cap"]().execute()
    return sa, log, pending, opened, ui_thread


def pump(pending, until, deliver=True):
    """The "UI loop": deliver relayed messages on this thread until `until()` holds."""
    end = time.monotonic() + DEADLINE
    while not until():
        assert time.monotonic() < end, "timed out waiting for the plugin"
        while deliver and pending:
            panel, msg = pending.pop(0)
            panel.on_message(msg)
        time.sleep(0.01)


def gate(sa):
    """Hold the worker inside plan_3mf until released. Returns (started, release)."""
    real = sa.plan_3mf
    started, release = threading.Event(), threading.Event()

    def gated(src, p, progress=None, **kw):
        started.set()
        assert release.wait(DEADLINE), "test never released the worker"
        return real(src, p, progress=progress, **kw)

    sa.plan_3mf = gated
    return started, release


def join(job):
    job.thread.join(DEADLINE)
    assert not job.thread.is_alive(), "worker did not stop"


def outputs(tmp: Path):
    return sorted((tmp / "storage").glob("*.3mf"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_arrange_runs_off_ui_thread_and_finishes_on_it():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, ui_thread = load(tmp, project)
        panel = sa._panel

        panel.on_message({"action": "arrange", "spacing": 1})
        panel.on_message({"action": "arrange", "spacing": 1})  # double click while busy: ignored
        pump(pending, panel.done.is_set)

        final = log["posts"][-1]
        assert final["done"] and final["text"].startswith("My Project, plate 2: placed 3"), final
        assert any("Placing" in m.get("text", "") for m in log["posts"]), "no progress posted"
        assert log["export_thread"] == ui_thread and len(log["exports"]) == 1
        assert opened and opened[0][1] == ui_thread, "launch must happen on the UI thread"
        out = opened[0][0]
        assert out.name.startswith("My Project-arranged-") and out.exists()
        assert f"Saved as: {out}" in final["text"] and final["path"] == str(out)
        assert sum(1 for m in log["posts"] if m.get("relay")) == 1, "second click started a second job"
        assert not log["exports"][0].exists(), "a finished job keeps no snapshot"


def test_fill_button_adds_copies():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, _ = load(tmp, project)
        sa._current_plate = lambda: 1
        panel = sa._panel
        panel.on_message({"action": "fill", "spacing": 2})
        pump(pending, panel.done.is_set)
        final = log["posts"][-1]["text"]
        assert "Filled the plate with" in final and "block ×" in final, final
        [out] = outputs(tmp)
        pkg = sa.Package(out)
        assert len(pkg.items()) > 4
        pkg.close()


def test_closed_panel_keeps_the_job_and_reopening_recovers_the_result():
    """C2: closing the panel loses the relay; the reopened panel's hello saves the result (no surprise
    launch), shows its path and can open it. finish is tied to the job id and idempotent."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, _ = load(tmp, project)
        started, release = gate(sa)
        first = sa._panel
        first.on_message({"action": "arrange", "spacing": 1})
        job = sa._session.job
        assert started.wait(DEADLINE)
        first.close()
        release.set()
        join(job)
        assert job.state == "ready" and not pending and outputs(tmp) == [], "result must wait, not vanish"

        log["cap"]().execute()  # reopen: a new panel joins the same session
        second = sa._panel
        assert second is not first and second in sa._session.sinks
        second.on_message({"action": "hello"})
        assert first not in sa._session.sinks, "a closed panel is dropped"
        assert job.state == "done" and not opened, "recovery saves but does not launch by itself"
        [out] = outputs(tmp)
        final = log["posts"][-1]
        assert final["job"] == job.id and final["path"] == str(out) and "Finished while" in final["text"], final

        second.on_message({"action": "finish", "job": job.id})  # duplicate relay: harmless
        second.on_message({"action": "finish", "job": "stale"})
        assert outputs(tmp) == [out] and not opened
        second.on_message({"action": "open", "job": job.id})
        assert [p for p, _ in opened] == [out]

        second.on_message({"action": "hello"})  # reload after done: same state comes back
        sync = log["posts"][-1]
        assert sync["sync"] and sync["job"] == job.id and sync["path"] == str(out) and not sync["busy"], sync


def test_one_job_at_a_time_with_its_own_files_and_earlier_results_kept():
    """C2/C3: a second request while a job runs exports nothing; each job reads its own snapshot and
    writes a new file; the earlier result is never overwritten, even for the same project name."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        one, two = tmp / "one.3mf", tmp / "two.3mf"
        cfg = {"printable_area": ["0x0", "60x0", "60x60", "0x60"]}
        t.simple_project(one, [t.box(0, 0, 10, 10)], [(0, t.tf(0, 0, 0))], cfg)
        t.simple_project(two, [t.box(0, 0, 10, 10)], [(0, t.tf(0, 0, 0))] * 2, cfg)
        sa, log, pending, opened, _ = load(tmp, one)
        sa._current_plate = lambda: 1
        panel = sa._panel
        started, release = gate(sa)
        panel.on_message({"action": "arrange", "spacing": 1})
        a = sa._session.job
        assert started.wait(DEADLINE)
        log["project"] = two  # the user switches project while A runs
        panel.on_message({"action": "arrange", "spacing": 1})
        assert len(log["exports"]) == 1 and sa._session.job is a, "second job started while the first ran"
        assert log["posts"][-1]["sync"] and log["posts"][-1]["busy"]
        release.set()
        pump(pending, lambda: a.state == "done")
        assert "placed 1 object" in a.summary, a.summary  # A computed its own snapshot
        [out_a] = outputs(tmp)
        before = digest(out_a)

        panel.on_message({"action": "arrange", "spacing": 1})
        b = sa._session.job
        pump(pending, lambda: b.state == "done")
        assert "placed 2 object" in b.summary, b.summary
        assert b.src != a.src and b.dst != a.dst and b.dst.name.startswith("My Project-arranged-")
        assert outputs(tmp) == sorted([out_a, b.dst]) and digest(out_a) == before


def test_cancel_while_running_writes_and_opens_nothing():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, _ = load(tmp, project)
        panel = sa._panel
        started, release = gate(sa)
        panel.on_message({"action": "arrange", "spacing": 1})
        job = sa._session.job
        assert started.wait(DEADLINE)
        panel.on_message({"action": "cancel", "job": "other"})  # stale id: ignored
        assert not job.cancelled.is_set()
        panel.on_message({"action": "cancel", "job": job.id})
        n = len(log["posts"])
        assert "cancelling" in log["posts"][-1]["text"]
        release.set()
        join(job)
        pump(pending, lambda: True)
        panel.on_message({"action": "finish", "job": job.id})
        assert job.state == "cancelled" and outputs(tmp) == [] and not opened
        after = [m for m in log["posts"][n:] if m.get("job") == job.id]
        assert len(after) == 1 and after[0]["done"] and "cancelled" in after[0]["text"], after
        assert not job.src.exists() and not job.src.parent.exists(), "cancelled job keeps no snapshot"
        panel.on_message({"action": "arrange", "spacing": 1})  # a new job can start afterwards
        pump(pending, lambda: sa._session.job.state == "done")


def test_cancel_racing_the_end_of_the_calculation_wins_cleanly():
    """The worker has its result but has not published it yet when Cancel arrives: no relay, no file."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, _ = load(tmp, project)
        real = sa.plan_3mf
        computed, release = threading.Event(), threading.Event()

        def held(src, p, progress=None, **kw):
            out = real(src, p, progress=progress, **kw)
            computed.set()
            assert release.wait(DEADLINE), "test never released the worker"
            return out

        sa.plan_3mf = held
        panel = sa._panel
        panel.on_message({"action": "arrange", "spacing": 1})
        job = sa._session.job
        assert computed.wait(DEADLINE)
        panel.on_message({"action": "cancel", "job": job.id})
        release.set()
        join(job)
        assert job.state == "cancelled" and not pending and job.result is None
        assert not any(m.get("relay") for m in log["posts"]) and outputs(tmp) == [] and not opened


def test_cancel_after_the_result_and_before_finish_writes_nothing():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, _ = load(tmp, project)
        panel = sa._panel
        panel.on_message({"action": "arrange", "spacing": 1})
        job = sa._session.job
        join(job)
        assert job.state == "ready" and pending, "relay should be waiting"
        panel.on_message({"action": "cancel", "job": job.id})
        pump(pending, lambda: not pending)  # the old relay arrives late
        assert job.state == "cancelled" and outputs(tmp) == [] and not opened


def test_write_failure_reports_and_keeps_earlier_results():
    if os.name == "nt" or os.geteuid() == 0:
        print("skip: needs POSIX permissions as a normal user")
        return
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, _ = load(tmp, project)
        panel = sa._panel
        storage = tmp / "storage"
        storage.mkdir()
        earlier = storage / "My Project-arranged-earlier.3mf"
        earlier.write_bytes(b"earlier result")
        panel.on_message({"action": "arrange", "spacing": 1})
        job = sa._session.job
        join(job)
        storage.chmod(0o555)
        try:
            pump(pending, panel.done.is_set)
        finally:
            storage.chmod(0o755)
        final = log["posts"][-1]
        assert job.state == "failed" and "could not save" in final["text"] and not final["busy"], final
        assert not opened and earlier.read_bytes() == b"earlier result" and outputs(tmp) == [earlier]
        assert job.src.exists() and project.exists(), "snapshot kept for diagnosis, project untouched"
        assert not list(storage.glob("*.tmp"))


def test_open_failure_still_shows_the_saved_path():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, _ = load(tmp, project)

        def broken(path):
            raise OSError("no launcher permission")

        sa._open_in_orca = broken
        panel = sa._panel
        panel.on_message({"action": "arrange", "spacing": 1})
        pump(pending, panel.done.is_set)
        final = log["posts"][-1]
        [out] = outputs(tmp)
        assert sa._session.job.state == "done" and final["path"] == str(out), final
        assert f"Saved as: {out}" in final["text"] and "Could not ask OrcaSlicer to open it" in final["text"]
        assert final["msg"]["open"] == "failed" and "no launcher permission" in final["msg"]["detail"], final


def test_bad_spacing_is_rejected_before_export():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, _ = load(tmp, project)
        panel = sa._panel
        for bad in (-1, None, "abc", float("inf"), float("nan"), True):
            panel.on_message({"action": "arrange", "spacing": bad})
            msg = log["posts"][-1]
            assert "spacing" in msg["text"].lower() and not msg["busy"], (bad, msg)
        assert log["exports"] == [] and sa._session.job is None


def test_result_text_is_honest_about_estimates_and_limits():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, *_ = load(tmp, project)
        report = {"placed": 2, "added": sa.MAX_COPIES, "added_by_object": {"a": 150, "b": 150},
                  "prime_tower": True, "spacing": 20.0, "unplaced": ["c"]}
        text = sa._summary(report, sa.Params(spacing=2), "P, plate 1")
        for needle in ("estimated prime tower", "only known after slicing", "XY clearance",
                       "Orca still checks", f"{sa.MAX_COPIES}-copy limit", "a ×150, b ×150", "moved above"):
            assert needle in text, (needle, text)
        # The page is the compiled Vue app: its live region is a compiled prop, not literal markup. The
        # states of the region are tested on the real components (ui/tests); here only that it ships.
        import re

        assert re.search(r"role:\s*[\"'`]status[\"'`]", sa.PANEL_HTML) and re.search(r"[\"']?aria-live[\"']?:\s*[\"'`]polite[\"'`]", sa.PANEL_HTML)
        assert '<html lang="en">' in sa.PANEL_HTML and "up to 300" in sa.PANEL_HTML


# ---------------------------------------------------------------- the Organizer tab (Pages)

def page_of(sa, log):
    """The host builds the tab's capability, then shows the tab, which calls get_ui()."""
    page = log["caps"][1]()
    page.on_load()
    return page


def page_polls(log, ui_thread):
    return [m for m, th in log["page_posts"] if th != ui_thread]


def drive_page(page, log, until, ui_thread):
    """What the tab's JavaScript does: poll, on the UI thread, until `until()` holds."""
    end = time.monotonic() + DEADLINE
    while not until():
        assert time.monotonic() < end, "timed out waiting for the plugin"
        page.on_message({"action": "poll"})
        time.sleep(0.01)
    assert page_polls(log, ui_thread) == [], "post_message was called off the UI thread"


def test_pages_capability_registered_next_to_the_script():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, *_ = load(tmp, project)
        assert [c.__name__ for c in log["caps"]] == ["ArrangeScript", "OrganizerPage"]
        assert log["perms"] == {"threading": True}
        page = log["caps"][1]()
        assert page.get_name() == "Organizer" and ";" not in page.get_name()
        assert page.get_ui() == sa.PANEL_HTML and sa._page_sink in sa._session.sinks
        page.on_load()  # must not call the virtual base hook: in Orca that recursed and crashed the app
        page.on_unload()
        assert log["hooks"] == [] and sa._page_sink not in sa._session.sinks


def test_old_hosts_keep_only_the_script():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, _ = load(tmp, project, pages=False)
        assert [c.__name__ for c in log["caps"]] == ["ArrangeScript"] and sa.OrganizerPage is None
        sa._panel.on_message({"action": "arrange", "spacing": 1})  # the Script path still works
        pump(pending, sa._panel.done.is_set)
        assert opened and sa._session.job.state == "done"
    with tempfile.TemporaryDirectory() as d:  # Pages exists but registering it is refused
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, _ = load(tmp, project, page_register_fails=True)
        assert [c.__name__ for c in log["caps"]] == ["ArrangeScript"]


def test_page_hooks_are_safe_without_base_hooks():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, *_ = load(tmp, project, base_hooks=False)
        page = log["caps"][1]()
        page.on_load()
        page.get_ui()
        page.on_unload()  # the base has no hooks: must not raise
        page.on_unload()  # and a second unload is harmless
        assert sa._session.sinks == [log["panels"][0]]


def test_page_arranges_and_opens_without_the_plugin_manager():
    """Hello, Arrange, progress by polling, finish and open: all through the tab, on the UI thread."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, ui = load(tmp, project)
        page = page_of(sa, log)
        page.get_ui()
        page.on_message({"action": "hello"})
        (hello, _), = log["page_posts"]
        assert hello["sync"] and hello["job"] is None and not hello["busy"], hello
        assert hello["context"] == "My Project · plate 2 selected · 3 objects in the project", hello
        assert hello["ctx"] == {"plate": 2, "objects": 3, "name": "My Project", "problem": None} and hello["lang"] == "pt_BR" and hello["stale"] == ""

        page.on_message({"action": "arrange", "spacing": 1})
        job = sa._session.job
        page.on_message({"action": "fill", "spacing": 1})  # a second click: shows the running job
        assert len(log["exports"]) == 1 and sa._session.job is job
        drive_page(page, log, lambda: job.state == "done", ui)
        page.on_message({"action": "poll"})
        final = log["page_posts"][-1][0]
        [out] = outputs(tmp)
        assert final["sync"] and final["job"] == job.id and final["path"] == str(out) and not final["busy"], final
        assert f"Saved as: {out}" in final["text"] and not final["error"]
        assert [p for p, _ in opened] == [out] and opened[0][1] == ui, "opened once, on the UI thread"
        assert not any(m.get("relay") for m, _ in log["page_posts"]), "the tab needs no relay"
        page.on_message({"action": "finish", "job": job.id})  # nothing more to write
        page.on_message({"action": "open", "job": job.id})
        assert [p for p, _ in opened] == [out, out]


def test_script_panel_and_page_share_one_session():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, ui = load(tmp, project)
        dock, page = sa._panel, page_of(sa, log)
        page.get_ui()
        started, release = gate(sa)
        page.on_message({"action": "arrange", "spacing": 1})  # started from the tab
        job = sa._session.job
        assert started.wait(DEADLINE)
        dock.on_message({"action": "arrange", "spacing": 1})  # the Script's panel: no second job
        dock.on_message({"action": "hello"})
        assert len(log["exports"]) == 1 and sa._session.job is job
        assert log["posts"][-1]["sync"] and log["posts"][-1]["job"] == job.id and log["posts"][-1]["busy"]
        page.on_message({"action": "poll"})
        assert log["page_posts"][-1][0]["job"] == job.id and log["page_posts"][-1][0]["busy"]
        dock.on_message({"action": "cancel", "job": job.id})  # cancelled from the other entry
        release.set()
        join(job)
        page.on_message({"action": "poll"})
        last = log["page_posts"][-1][0]
        assert job.state == "cancelled" and not last["busy"] and "cancelled" in last["text"], last
        assert outputs(tmp) == [] and not opened and page_polls(log, ui) == []
        page.on_message({"action": "arrange", "spacing": 1})  # and the next run can start from either
        job2 = sa._session.job
        assert job2 is not job
        pump(pending, lambda: job2.state == "done")  # the dock relays finish; the page's poll is a no-op
        page.on_message({"action": "poll"})
        assert len(opened) == 1 and len(outputs(tmp)) == 1


def test_page_stale_ids_are_rejected():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, ui = load(tmp, project)
        page = page_of(sa, log)
        page.get_ui()
        started, release = gate(sa)
        page.on_message({"action": "arrange", "spacing": 1})
        job = sa._session.job
        assert started.wait(DEADLINE)
        for action in ("cancel", "finish", "open"):
            for bad in ("old-job", None, 7):
                page.on_message({"action": action, "job": bad})
        page.on_message({"action": "finish", "job": job.id})  # right id, wrong state: nothing to finish
        assert job.state == "running" and not job.cancelled.is_set() and not opened and outputs(tmp) == []
        release.set()
        drive_page(page, log, lambda: job.state == "done", ui)
        page.on_message({"action": "cancel", "job": "old-job"})
        page.on_message({"action": "cancel", "job": job.id})  # done: cancelling changes nothing
        assert job.state == "done" and len(outputs(tmp)) == 1


def test_page_that_looked_away_saves_without_opening():
    """Silence longer than WATCH_GAP means nobody watched: save, show the path, do not launch."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, ui = load(tmp, project)
        now = [100.0]
        sa._session.clock = lambda: now[0]
        page = page_of(sa, log)
        page.get_ui()
        page.on_message({"action": "arrange", "spacing": 1})
        job = sa._session.job
        join(job)
        assert job.state == "ready"
        now[0] += sa.WATCH_GAP + 1  # the tab was hidden meanwhile
        page.on_message({"action": "poll"})
        [out] = outputs(tmp)
        assert job.state == "done" and not opened
        assert "Finished while the panel was closed" in job.text and str(out) in job.text
        page.on_message({"action": "open", "job": job.id})
        assert [p for p, _ in opened] == [out]


def test_reloaded_tab_recovers_state_and_unload_stops_posts():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, ui = load(tmp, project)
        page = page_of(sa, log)
        page.get_ui()
        started, release = gate(sa)
        page.on_message({"action": "arrange", "spacing": 1})
        job = sa._session.job
        assert started.wait(DEADLINE)
        page.on_message({"action": "hello"})  # the user reloaded the tab mid-run
        sync = log["page_posts"][-1][0]
        assert sync["sync"] and sync["job"] == job.id and sync["busy"] and "reading project" in sync["text"], sync
        page.on_unload()  # plugin disabled: the tab goes away, the worker keeps going
        n = len(log["page_posts"])
        release.set()
        join(job)
        sa._session.post(text="x")
        assert len(log["page_posts"]) == n, "an unloaded tab received a post"
        page2 = page_of(sa, log)  # enabled again: tab rebuilt, asks for state
        page2.get_ui()
        page2.on_message({"action": "hello"})  # recovery: saved, not launched
        assert job.state == "done" and not opened and len(outputs(tmp)) == 1
        assert log["page_posts"][-1][0]["path"] == str(outputs(tmp)[0])


def test_context_reports_stale_runs_and_empty_projects():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, ui = load(tmp, project)
        page = page_of(sa, log)
        page.get_ui()
        page.on_message({"action": "arrange", "spacing": 1})
        job = sa._session.job
        drive_page(page, log, lambda: job.state == "done", ui)
        page.on_message({"action": "poll"})
        assert log["page_posts"][-1][0]["stale"] == ""
        log["plate_index"] = 0  # the user selects another plate
        page.on_message({"action": "poll"})
        stale = log["page_posts"][-1][0]["stale"]
        assert stale == "The run below is for My Project, plate 2, not the current selection.", stale
        log["project_path"] = str(tmp / "Other.3mf")
        log["plate_index"] = 1
        log["objects"] = 0
        page.on_message({"action": "poll"})
        msg = log["page_posts"][-1][0]
        assert "Other · plate 2 selected · 0 objects" in msg["context"] and msg["ctx"]["objects"] == 0
        assert msg["stale"].startswith("The run below is for My Project, plate 2"), msg["stale"]
        sa.orca.host.model = lambda: (_ for _ in ()).throw(RuntimeError("no model"))  # count unavailable
        page.on_message({"action": "poll"})
        assert log["page_posts"][-1][0]["ctx"]["objects"] is None


def test_keep_orientation_option():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, ui = load(tmp, project)
        page = page_of(sa, log)
        page.get_ui()
        page.on_message({"action": "arrange", "spacing": 1, "rotation": "sideways"})
        assert log["exports"] == [] and sa._session.job is None, "bad option must not start a run"
        assert "rotation" in log["page_posts"][-1][0]["text"].lower() and log["page_posts"][-1][0]["error"]
        page.on_message({"action": "arrange", "spacing": 1, "rotation": "keep"})
        job = sa._session.job
        assert job.p.step == 0 and job.p.res == sa.AUTO["res"] and job.p.refine
        drive_page(page, log, lambda: job.state == "done", ui)
        assert "Rotation was off" in job.summary
        page.on_message({"action": "arrange", "spacing": 1})  # default stays automatic
        assert sa._session.job.p.step == sa.AUTO["step"]
        drive_page(page, log, lambda: sa._session.job.state == "done", ui)


def test_failed_run_is_flagged_as_error_for_the_page():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, ui = load(tmp, project)
        sa._current_plate = lambda: 9  # no such plate: the plan fails
        page = page_of(sa, log)
        page.get_ui()
        page.on_message({"action": "arrange", "spacing": 1})
        job = sa._session.job
        drive_page(page, log, lambda: job.state == "failed", ui)
        page.on_message({"action": "poll"})
        msg = log["page_posts"][-1][0]
        assert msg["error"] and not msg["busy"], msg
        assert not msg["text"].startswith("error:") and ": error:" not in msg["text"], "no machine prefix"
        assert "Plate 9 does not exist: the project has 2 plates." in msg["text"], msg["text"]
        assert "Next: Select one of the existing plates" in msg["text"] and "Details:" not in msg["text"]
        assert msg["msg"]["code"] == "failed" and msg["msg"]["cause"] == {"code": "plate_gone", "plate": 9, "plates": 2}
        assert msg["msg"]["action"] == "arrange" and msg["msg"]["plate"] == 9 and msg["msg"]["stem"] == "My Project"
        assert job.src.exists(), "snapshot of a failed run stays for diagnosis"
        page.on_message({"action": "arrange", "spacing": 1})  # a failed run does not block the next one
        assert sa._session.job is not job


def test_unexpected_failure_keeps_the_raw_error_as_details():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, ui = load(tmp, project)

        def boom(*a, **k):
            raise RuntimeError("zip is corrupt")

        sa.plan_3mf = boom
        page = page_of(sa, log)
        page.get_ui()
        page.on_message({"action": "fill", "spacing": 1})
        job = sa._session.job
        drive_page(page, log, lambda: job.state == "failed", ui)
        page.on_message({"action": "poll"})
        msg = log["page_posts"][-1][0]
        assert msg["msg"]["cause"] == {"code": "unexpected"} and msg["msg"]["detail"] == "zip is corrupt"
        assert msg["msg"]["action"] == "fill"
        assert "could not fill." in msg["text"] and "Details: zip is corrupt" in msg["text"], msg["text"]
        assert "Next: Try again" in msg["text"]


def test_progress_is_structured_and_counts_only_real_totals():
    """The page draws a determinate bar from `i`/`n`, never from parsing the English text."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, ui = load(tmp, project)
        sa._current_plate = lambda: 1
        panel = sa._panel
        panel.on_message({"action": "fill", "spacing": 2})
        pump(pending, panel.done.is_set)
        prog = [m["msg"] for m in log["posts"] if m.get("msg") and m["msg"]["code"] == "progress"]
        phases = {m["phase"] for m in prog}
        assert {"read", "place"} <= phases and "fill" in phases, phases
        for m in prog:
            if m["phase"] in ("place", "footprint", "patterns"):
                assert m["n"] and 0 <= m["i"] < m["n"] and m["count"] is None, m
            else:  # no stable total: a count at most, never a bar
                assert m["n"] is None, m
            assert m["plate"] == 1 and m["stem"] == "My Project"
        fills = [m for m in prog if m["phase"] == "fill"]
        assert fills and all(isinstance(m["count"], int) for m in fills), fills
        texts = [m["text"] for m in log["posts"] if m.get("msg") and m["msg"]["code"] == "progress"]
        assert any("Placing" in x and "(1/" in x for x in texts)
        assert all("(" not in x.split(": ", 1)[1] for x in texts if "Filling" in x or "Reading" in x), texts


def test_host_language_is_passed_to_the_page():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, ui = load(tmp, project)
        page = page_of(sa, log)
        page.get_ui()
        page.on_message({"action": "hello"})
        assert log["page_posts"][-1][0]["lang"] == "pt_BR"
        assert log["page_posts"][-1][0]["ctx"] == {"plate": 2, "objects": 3, "name": "My Project", "problem": None}
        log["lang"] = None  # not offered: nothing invented, the page falls back to the browser's language
        sa._session.lang = None
        page.on_message({"action": "hello"})
        assert log["page_posts"][-1][0]["lang"] is None
        del sa.orca.host.app_language  # a host without the API
        page.on_message({"action": "poll"})
        assert log["page_posts"][-1][0]["lang"] is None
        log["lang"] = "pt_BR"
        sa.orca.host.app_language = lambda: log["lang"]
        panel = sa._panel
        panel.on_message({"action": "arrange", "spacing": 1})
        pump(pending, panel.done.is_set)
        assert all(m.get("lang") == "pt_BR" for m in log["posts"] if "text" in m and m.get("job")), "worker posts carry the cached language"


def test_every_user_reachable_error_has_cause_and_next_step():
    """Each PlanError code has English cause + next step; the page's pt-BR/en tables are checked in test_panel_js."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, *_ = load(tmp, project)
        import ast

        tree = ast.parse(Path(sa.__file__).read_text())
        codes = {c.args[1].value for c in ast.walk(tree) if isinstance(c, ast.Call)
                 and getattr(c.func, "id", "") == "PlanError" and len(c.args) > 1 and isinstance(c.args[1], ast.Constant)}
        assert codes == set(sa.ERRORS_EN), (codes ^ set(sa.ERRORS_EN))
        for code in codes:
            why, nxt = sa.error_en({"code": code, "plate": 3, "plates": 1, "unit": "inch", "res": 0.5})
            assert why.endswith(".") and nxt.endswith(".") and "{" not in why + nxt, (code, why, nxt)
        assert sa.error_en({"code": "plate_gone", "plate": 3, "plates": 1})[0].endswith("the project has 1 plate.")
        assert isinstance(sa.PlanError("x", "plate_empty", plate=1), ValueError)  # old callers still catch ValueError


def test_result_message_is_structured_and_open_wording_is_neutral():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, _ = load(tmp, project)
        sa._current_plate = lambda: 1
        panel = sa._panel
        panel.on_message({"action": "arrange", "spacing": 1})
        pump(pending, panel.done.is_set)
        final = log["posts"][-1]
        msg = final["msg"]
        assert msg["code"] == "done" and msg["open"] == "asked" and msg["placed"] > 0 and msg["added"] == 0
        assert msg["keep"] is False and msg["unplaced"] == [] and msg["tower"] in (True, False)
        low = final["text"].lower()
        assert "asked orcaslicer to open the copy" in low and "open this copy in orcaslicer" in low
        for claim in ("saved your project", "replaces", "new window", "project is saved", "unsaved"):
            assert claim not in low, claim  # what Orca does with its open project is not known
        panel.on_message({"action": "open", "job": sa._session.job.id})
        again = log["posts"][-1]
        assert again["msg"]["code"] == "done" and again["msg"]["open"] == "asked" and again["path"] == final["path"]
        sa._open_in_orca = lambda p: False
        panel.on_message({"action": "open", "job": sa._session.job.id})
        assert log["posts"][-1]["msg"]["open"] == "none" and "open the saved copy by hand" in log["posts"][-1]["text"]


def test_start_error_is_explained_for_the_page():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        project = tmp / "proj.3mf"
        t.make_project(project)
        sa, log, pending, opened, _ = load(tmp, project)
        panel = sa._panel
        panel.on_message({"action": "arrange", "spacing": -1})
        msg = log["posts"][-1]
        assert msg["error"] and msg["msg"]["code"] == "start_failed" and msg["msg"]["cause"]["code"] == "bad_number"
        assert msg["text"].startswith("Could not start: Spacing must be a number, 0 mm or more.\nNext: Enter a value")
        assert "Error:" not in msg["text"] and "Details:" not in msg["text"]
        panel.on_message({"action": "arrange", "spacing": 1, "rotation": "sideways"})
        assert log["posts"][-1]["msg"]["cause"]["code"] == "bad_rotation" and "rotation" in log["posts"][-1]["text"].lower()


if __name__ == "__main__":
    test_arrange_runs_off_ui_thread_and_finishes_on_it()
    test_fill_button_adds_copies()
    test_closed_panel_keeps_the_job_and_reopening_recovers_the_result()
    test_one_job_at_a_time_with_its_own_files_and_earlier_results_kept()
    test_cancel_while_running_writes_and_opens_nothing()
    test_cancel_racing_the_end_of_the_calculation_wins_cleanly()
    test_cancel_after_the_result_and_before_finish_writes_nothing()
    test_write_failure_reports_and_keeps_earlier_results()
    test_open_failure_still_shows_the_saved_path()
    test_bad_spacing_is_rejected_before_export()
    test_result_text_is_honest_about_estimates_and_limits()
    test_pages_capability_registered_next_to_the_script()
    test_old_hosts_keep_only_the_script()
    test_page_hooks_are_safe_without_base_hooks()
    test_page_arranges_and_opens_without_the_plugin_manager()
    test_script_panel_and_page_share_one_session()
    test_page_stale_ids_are_rejected()
    test_page_that_looked_away_saves_without_opening()
    test_reloaded_tab_recovers_state_and_unload_stops_posts()
    test_context_reports_stale_runs_and_empty_projects()
    test_keep_orientation_option()
    test_failed_run_is_flagged_as_error_for_the_page()
    test_unexpected_failure_keeps_the_raw_error_as_details()
    test_progress_is_structured_and_counts_only_real_totals()
    test_host_language_is_passed_to_the_page()
    test_every_user_reachable_error_has_cause_and_next_step()
    test_result_message_is_structured_and_open_wording_is_neutral()
    test_start_error_is_explained_for_the_page()
    print("ok")
