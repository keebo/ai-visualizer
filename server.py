#!/usr/bin/env python3
# ai-visualizer: give your AI agent a face.
# Copyright (C) 2026 Jared Rhodenizer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ai-visualizer server. Python standard library only, nothing to install.

Serves the face gallery at http://127.0.0.1:8790/ and exposes:

  /state   polled by the faces (~8x/sec):
           {"state":  "idle|listening|thinking|working|speaking",
            "level":  0.0-1.0,       voice loudness while speaking
            "samples": [64 floats],  raw waveform snapshot (0s when quiet)
            "alert":  bool,          optional attention signal
            "loading": bool,         true while the voice line plays its
                                     own thinking sound (we stay quiet)
            "note": {...},           see .agent_note below
            "activity": [...]}       see .agent_activity below
  /config  the merged ai-visualizer.json plus the list of installed
           faces, discovered by scanning the faces/ folder. Drop a new
           folder with an index.html into faces/ and it appears in the
           gallery. That is the whole plugin system.
  /inbox   POST, image bytes with an image/* Content-Type. Writes
           inbox/paste-<timestamp>.<ext> AND inbox/latest.<ext> under the
           bus dir, so a paste (Ctrl+V, e.g. a Win+Shift+S snip) from any
           face reaches whatever is driving the session at a fixed path.

READ-ONLY on the signal bus. The bus is three tiny files written by a
voice line (backtalk writes them natively, github.com/jaredrhod/backtalk):

  .voice_state        idle | listening | thinking | working | speaking
  .voice_waveform     JSON {ts, samples: [64 floats]} while audio plays
  .voice_loading_pid  exists while the voice line plays a thinking sound
  .voice_alert        optional: non-empty file = attention needed
  .agent_note         optional, NOT written by backtalk: JSON
                       {"title": str, "text": str} that whatever agent is
                       driving the session (Claude Code, a script, anything
                       with filesystem access) can drop on the bus so a
                       face can surface it as a HUD panel. Empty/missing
                       file = no panel. Overwrite to update it, delete or
                       empty it to dismiss.
  .voice_transcript   optional, written by backtalk: JSON array of the last
                       ~60 {"ts", "role", "text"} spoken lines, role being
                       "you" or the agent's name. Subtitles, and a scrollback
                       for when the listener was not listening.
  .agent_activity      optional, NOT written by backtalk: JSON array of
                       the last ~30 {"ts", "tool", "detail"} entries, kept
                       current by the project's PreToolUse hook
                       (.claude/hooks/activity_log.py) on every tool call,
                       so a face can show a live "what is Jarvis doing"
                       feed instead of looking idle during silent work.

"working" is a tool call actually running (a file write, a shell
command, a dispatched subagent) — distinct from "thinking", which is
the model composing with nothing to show yet.

BACKGROUND JOBS (anything outside a live model turn): drop any file
into a "background/" folder next to the bus files while your job runs,
and remove it when done — no Python, no API, works from a plain shell
script:

  mkdir -p background && touch background/my-job    # starting
  rm -f background/my-job                           # finished

While ANY file sits in that folder, /state reports "working" instead
of "idle" (a live turn's own state always takes priority). A marker
older than 6 hours is ignored, so a job that crashed without cleaning
up can't wedge the face in "working" for that long.

Optional, Python-only upgrade: write your PID as the marker's content
instead of leaving it empty --

  BG_MARKER.write_text(str(os.getpid()))

-- and a marker whose owning process has already died gets ignored
immediately instead of waiting up to 6 hours (confirmed real 2026-09-02:
a launchd photo-tagging job got killed by a reboot before its own
`finally` cleanup ran, wedging the face on "working" for the rest of
that stretch). A marker with no PID, or one that can't be read as an
int, just falls back to the age-only check above -- so the plain
`touch`/`rm` shell-script path stays exactly as simple as it always was.

Where the bus lives comes from "bus_dir" in ai-visualizer.json (default:
this folder). Point it at your backtalk folder, or point backtalk's
"signals_dir" here. Either direction works.

Run:
  python3 server.py             the real bus
  python3 server.py --mock speaking
                                no voice line needed: /state synthesizes
                                the chosen state (idle|listening|thinking
                                |working|speaking) so you can see a face
                                perform
  python3 server.py --no-open   do not auto-open the browser
Ctrl-C stops.
"""
import json
import math
import mimetypes
import os
import plistlib
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
import urllib.request
import errno
from collections import deque
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATES = {"idle", "listening", "thinking", "working", "speaking"}
WAVEFORM_STALE_S = 0.6
BG_STALE_S = 6 * 60 * 60   # ignore a marker file older than this (orphaned by a crash)

DEFAULTS = {
    "name": "JARVIS",       # shown on the chip / headers, yours to change
    "badge": "",            # optional handle shown in some faces' chrome
    "face": "board",        # the default face the root URL opens
    "port": 8790,
    "bus_dir": "",          # where the .voice_* files live ("" = here)
    "thinking_sound": True, # play assets/thinking.wav while thinking
    # Label a face shows when a local model (backtalk's local_llm) is
    # answering instead of the main agent — only meaningful if that
    # feature is on. "" falls back to a generic "LOCAL" in the face.
    "local_name": "",
}


def load_config():
    cfg = dict(DEFAULTS)
    try:
        user = json.loads((HERE / "ai-visualizer.json").read_text(encoding="utf-8"))
        for k, v in user.items():
            cfg[k] = v
    except FileNotFoundError:
        pass
    except ValueError as e:
        print(f"[config] ai-visualizer.json is not valid JSON ({e}), "
              f"using defaults")
    return cfg


CFG = load_config()
BUS = Path(CFG["bus_dir"]).expanduser() if CFG.get("bus_dir") else HERE
BG_DIR = BUS / "background"

MOCK = None
NO_OPEN = "--no-open" in sys.argv
if "--mock" in sys.argv:
    i = sys.argv.index("--mock")
    MOCK = sys.argv[i + 1] if len(sys.argv) > i + 1 else "speaking"
    if MOCK not in STATES:
        MOCK = "speaking"
MOCK_SOURCE = "local" if "--mock-local" in sys.argv else ""
PORT = int(CFG.get("port", 8790))
if "--port" in sys.argv:
    i = sys.argv.index("--port")
    PORT = int(sys.argv[i + 1])


def list_faces():
    faces = []
    fdir = HERE / "faces"
    if fdir.is_dir():
        for p in sorted(fdir.iterdir()):
            if p.is_dir() and (p / "index.html").exists():
                meta = {"id": p.name, "title": p.name.title(), "tagline": ""}
                try:
                    meta.update(json.loads((p / "face.json").read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    pass
                meta["id"] = p.name
                faces.append(meta)
    return faces


def mock_bus():
    t = time.time()
    level = 0.0
    samples = [0.0] * 64
    if MOCK == "speaking":
        level = abs(math.sin(t * 6.0)) * 0.85
        samples = [
            (math.sin(i * 0.55 + t * 9.0) * 0.6
             + math.sin(i * 1.7 - t * 13.0) * 0.4)
            * 9000.0 * (0.35 + 0.65 * abs(math.sin(t * 2.6)))
            for i in range(64)
        ]
    return {"state": MOCK, "level": level, "samples": samples,
            "alert": False, "loading": MOCK in ("thinking", "working"),
            # Faked so the usage readout can be looked at without
            # spending a real session to make it appear.
            "rate_limits": {
                "five_hour": {"utilization": 0.34, "resets_at": t + 9200},
                "seven_day": {"utilization": 0.61, "resets_at": t + 288000},
            },
            # Real reads, not faked -- a slider should still work against
            # --mock for UI testing without a live voice line.
            "thinking_volume": _read_volume(".thinking_volume", 0.35),
            "voice_volume": _read_volume(".voice_volume", 1.0),
            "silent_mode": _read_silent_mode(),
            "model": "fast",
            "system": read_sys_stats(),
            "source": MOCK_SOURCE, "note": {}, "activity": []}


def _read_silent_mode() -> bool:
    try:
        return (BUS / ".silent_mode").read_text().strip() == "1"
    except OSError:
        return False


def _marker_pid(path: Path) -> int | None:
    """The marker's content as a PID, if it has one -- see the module
    docstring's optional PID-marker upgrade. None for an empty/plain
    touch()'d marker or one that doesn't parse, which is the expected,
    still-supported common case."""
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just owned by someone else
    return True


def _background_job_active():
    """Any live, non-stale, non-orphaned file in BG_DIR counts as work
    in progress. A PID-marker whose process has already died is treated
    as inactive immediately rather than waiting out BG_STALE_S."""
    try:
        now = time.time()
        for p in BG_DIR.iterdir():
            if now - p.stat().st_mtime >= BG_STALE_S:
                continue
            pid = _marker_pid(p)
            if pid is not None and not _pid_alive(pid):
                continue
            return True
        return False
    except OSError:
        return False


# --------------------------- system stats sampler ---------------------------
# CPU/memory/disk/network for a face's own resource widgets. Kevin's ask,
# 2026-09-06 -- GPU deliberately left out: live GPU utilization needs
# `powermetrics`, which is sudo-gated on macOS and can't be run
# unattended without a password prompt. Sampled on a background thread,
# not per-request: `top -l 1` alone takes close to a second, and
# `iostat -w 1` deliberately blocks for a real second to compute a rate
# -- both far too slow for a poll hit ~8x/sec. Every command here is a
# standard macOS CLI tool (no new Python dependency), consistent with
# this file's own "Python standard library only" design.
SYS_SAMPLE_S = 10.0
SYS_HISTORY_LEN = 36   # * SYS_SAMPLE_S = 6 minutes of trend, Kevin's ask
_sys_lock = threading.Lock()
_sys_stats = {"cpu_pct": None, "mem_pct": None, "disk_pct": None,
              "disk_io_mbs": None, "net_kbs": None, "gpu_pct": None}
_sys_history = {k: deque(maxlen=SYS_HISTORY_LEN) for k in _sys_stats}
_sys_prev_net = None   # (bytes_total, monotonic_ts), for a manual rate calc
_sys_cpu_cores = []    # per-core %, latest reading only -- no history needed
                       # for an accordion that only ever shows "right now"
# Top-contributor lists (latest reading only, same reasoning as cores above --
# a detail window's process list only ever needs to show "right now").
_sys_top_cpu = []
_sys_top_mem_list = []
_sys_top_disk_io = []
_sys_top_net = []

_TOP_N = 5   # rows shown in each detail window's "top contributors" list


def _sys_powermetrics():
    # Per-core CPU, GPU, and per-process CPU/disk-IO/network breakdowns all
    # come from this one `powermetrics` call -- no other standard macOS
    # tool exposes per-core or per-process network/disk-IO attribution.
    # Runs passwordless via the sudoers rule scoped to exactly this binary
    # (/etc/sudoers.d/powermetrics-nopasswd, 2026-09-06). `-n` (never
    # prompt) means a broken/missing grant fails silently to empty results
    # instead of hanging this background thread on a password prompt
    # nobody's watching. `--format plist` -- structured fields, not the
    # human-readable table (fixed-width columns, names truncated/collide
    # with the grid, multi-word process names break naive parsing).
    # Per-process GPU time was tried (--show-process-gpu) and never
    # populates on this machine/OS build, even for a genuinely GPU-heavy
    # process -- a platform gap, not a parsing bug, so GPU gets no
    # top-contributor list (Kevin's call, 2026-09-06: skip it, don't fake it).
    try:
        out = subprocess.run(
            ["sudo", "-n", "/usr/bin/powermetrics", "-n", "1", "-i", "1000",
             "--samplers", "cpu_power,gpu_power,tasks",
             "--show-process-io", "--show-process-netstats",
             "--format", "plist"],
            capture_output=True, timeout=4).stdout
        data = plistlib.loads(out)
    except Exception:
        return None, None, [], [], []
    cores = None
    try:
        by_idx = {}
        for cluster in data["processor"]["clusters"]:
            for cpu in cluster["cpus"]:
                by_idx[cpu["cpu"]] = round(100.0 * (1 - cpu["idle_ratio"]), 1)
        cores = [by_idx[i] for i in sorted(by_idx)] if by_idx else None
    except Exception:
        pass
    gpu = None
    try:
        gpu = round(100.0 * (1 - data["gpu"]["idle_ratio"]), 1)
    except Exception:
        pass
    tasks = data.get("tasks", [])

    def top(key_fn):
        # DEAD_TASKS is powermetrics' own aggregate bucket for already-exited
        # processes' residual accounting -- real, but not a single app Kevin
        # could act on, so it's noise in a "what's using this" list.
        real = [t for t in tasks if t.get("name") != "DEAD_TASKS"]
        ranked = sorted(real, key=key_fn, reverse=True)[:_TOP_N]
        return [{"name": t.get("name", "?"), "value": round(key_fn(t), 1)}
                for t in ranked if key_fn(t) > 0]

    top_cpu = top(lambda t: t.get("cputime_ms_per_s") or 0)
    top_disk_io = top(lambda t: (t.get("diskio_bytesread_per_s") or 0)
                                + (t.get("diskio_byteswritten_per_s") or 0))
    top_net = top(lambda t: (t.get("bytes_received_per_s") or 0)
                            + (t.get("bytes_sent_per_s") or 0))
    return cores, gpu, top_cpu, top_disk_io, top_net


def _sys_top_mem():
    # Top processes by memory share -- `ps -m` sorts by memory usage
    # descending on macOS, no sudo needed (unlike everything routed
    # through powermetrics above).
    try:
        out = subprocess.run(["ps", "-Ao", "pid,%mem,comm", "-m"],
                              capture_output=True, text=True, timeout=3).stdout
        rows = []
        for line in out.splitlines()[1:]:   # skip header
            parts = line.split(None, 2)
            if len(parts) == 3:
                rows.append({"name": os.path.basename(parts[2]),
                            "value": float(parts[1])})
        return [r for r in rows if r["value"] > 0][:_TOP_N]
    except Exception:
        return []


# Largest files under the home folder -- a disk-SPACE question, not a
# process-load one, so it doesn't belong in the 3s sampler loop at all:
# a full home-folder walk is far too slow to repeat every few seconds,
# and unlike CPU/mem/network this genuinely doesn't change minute to
# minute. Kevin's explicit call, 2026-09-06: scan on its own 24-hour
# timer, cached, so the detail window always has an instant answer.
SYS_DISK_SCAN_S = 24 * 60 * 60
_sys_disk_files_lock = threading.Lock()
_sys_disk_files = []


def _sys_scan_top_files(n=10):
    home = Path.home()
    biggest = []   # small n -- a plain list + sort beats a heap for clarity
    for root, dirs, files in os.walk(home, onerror=lambda e: None):
        for fn in files:
            p = os.path.join(root, fn)
            try:
                if os.path.islink(p):
                    continue
                size = os.path.getsize(p)
            except OSError:
                continue
            biggest.append((size, p))
    biggest.sort(reverse=True)
    rel = lambda p: os.path.relpath(p, home)
    return [{"path": rel(p), "bytes": s} for s, p in biggest[:n]]


def _sys_disk_scan_loop():
    global _sys_disk_files
    while True:
        try:
            files = _sys_scan_top_files()
        except Exception:
            files = []
        with _sys_disk_files_lock:
            _sys_disk_files = files
        time.sleep(SYS_DISK_SCAN_S)


# Calendar widget, 2026-09-07 -- Spark Mail (Readdle's Electron mail client)
# turns out to keep a real, live, WAL-mode SQLite cache of every calendar
# it syncs (`calendarsapi.sqlite`), confirmed directly against Kevin's own
# real data before building anything here. Read-only, WAL mode means we
# never contend with Spark's own writer for a lock. Kevin's own choice: the
# picker starts empty, nothing shown until he selects calendars himself.
CAL_DB_PATH = Path.home() / "Library/Application Support/Spark Mail/core-data/calendarsapi.sqlite"
CAL_POLL_S = 60.0
CAL_MAX_EVENTS = 3
_cal_lock = threading.Lock()
_cal_calendars = []
_cal_events = []
_cal_month_days = []


def _cal_read_selection():
    try:
        data = json.loads((BUS / ".calendar_selection").read_text(encoding="utf-8"))
        return [int(x) for x in data] if isinstance(data, list) else []
    except (OSError, ValueError, TypeError):
        return []


def _cal_write_selection(pks):
    (BUS / ".calendar_selection").write_text(
        json.dumps(sorted(set(int(p) for p in pks))))


def _cal_query(selected):
    calendars, events, month_days = [], [], []
    try:
        conn = sqlite3.connect(f"file:{CAL_DB_PATH}?mode=ro", uri=True, timeout=2)
        try:
            rows = conn.execute(
                "SELECT pk, displayname, backgroundColor FROM RDCALAPICollection "
                "WHERE hidden=0 ORDER BY displayname COLLATE NOCASE").fetchall()
            calendars = [{"pk": r[0], "name": r[1] or "(untitled)",
                          "color": r[2] or "#5ae1ff"} for r in rows]
            if selected:
                placeholders = ",".join("?" * len(selected))
                rows = conn.execute(
                    "SELECT e.summary, e.dstart, e.allDay, c.displayname, "
                    "c.backgroundColor FROM RDCALAPIEvent e "
                    "JOIN RDCALAPICollection c ON c.pk = e.refCollectionPK "
                    f"WHERE e.refCollectionPK IN ({placeholders}) "
                    "AND e.dstart > strftime('%s','now') "
                    f"ORDER BY e.dstart ASC LIMIT {CAL_MAX_EVENTS}",
                    list(selected)).fetchall()
                events = [{"summary": r[0] or "(untitled)", "start": r[1],
                          "allDay": bool(r[2]), "calendar": r[3],
                          "color": r[4] or "#5ae1ff"} for r in rows]
                # Every day in the CURRENT LOCAL month that has at least one
                # event, across the selected calendars -- feeds the month
                # grid's highlight, not just the next-3 list. 'localtime'
                # matches how the frontend already displays dstart (via a
                # plain JS `new Date(start*1000)`, which is local-zone too).
                rows = conn.execute(
                    "SELECT DISTINCT CAST(strftime('%d', e.dstart, 'unixepoch', "
                    "'localtime') AS INTEGER) FROM RDCALAPIEvent e "
                    f"WHERE e.refCollectionPK IN ({placeholders}) "
                    "AND strftime('%Y-%m', e.dstart, 'unixepoch', 'localtime') "
                    "= strftime('%Y-%m', 'now', 'localtime')",
                    list(selected)).fetchall()
                month_days = sorted(r[0] for r in rows)
        finally:
            conn.close()
    except Exception:
        # Spark not installed, db missing, or a schema Spark's own next
        # update changes underneath us -- degrade to an empty widget
        # rather than take the whole face down over an optional feature.
        pass
    return calendars, events, month_days


def _cal_poll_loop():
    global _cal_calendars, _cal_events, _cal_month_days
    while True:
        calendars, events, month_days = _cal_query(_cal_read_selection())
        with _cal_lock:
            _cal_calendars = calendars
            _cal_events = events
            _cal_month_days = month_days
        time.sleep(CAL_POLL_S)


def read_cal_state():
    with _cal_lock:
        return {"calendars": list(_cal_calendars), "events": list(_cal_events),
                "month_days": list(_cal_month_days),
                "selected": _cal_read_selection()}


def _cal_day_events(date_str):
    # date_str: "YYYY-MM-DD", as produced by the reactor face's own month
    # grid (built off the browser's local Date, same zone this machine's
    # SQLite 'localtime' modifier resolves to). On-demand only -- clicking
    # a day, not part of the 60s poll -- so no caching, straight to Spark's
    # db each time.
    selected = _cal_read_selection()
    if not selected:
        return []
    events = []
    try:
        conn = sqlite3.connect(f"file:{CAL_DB_PATH}?mode=ro", uri=True, timeout=2)
        try:
            placeholders = ",".join("?" * len(selected))
            rows = conn.execute(
                "SELECT e.summary, e.dstart, e.dend, e.allDay, e.location, "
                "c.displayname, c.backgroundColor FROM RDCALAPIEvent e "
                "JOIN RDCALAPICollection c ON c.pk = e.refCollectionPK "
                f"WHERE e.refCollectionPK IN ({placeholders}) "
                "AND date(e.dstart, 'unixepoch', 'localtime') = ? "
                "ORDER BY e.dstart ASC",
                list(selected) + [date_str]).fetchall()
            events = [{"summary": r[0] or "(untitled)", "start": r[1], "end": r[2],
                      "allDay": bool(r[3]), "location": r[4] or "",
                      "calendar": r[5], "color": r[6] or "#5ae1ff"} for r in rows]
        finally:
            conn.close()
    except Exception:
        pass
    return events


# Rosa job queue, 2026-09-07 -- Kevin's ask: a widget where he hands Rosa
# (backtalk's local model) a specific file or pasted text plus an
# instruction, and gets the result back without a live conversation turn.
# This file is the one shared hand-off point: this server only ever reads
# and writes it on Kevin's behalf (submit, list, fetch a result) -- the
# actual model call happens over in backtalk's own process (main.py's
# _rosa_queue_loop), the same "backtalk owns every real file read/write,
# Rosa never touches the filesystem herself" boundary every other Rosa
# feature already follows. Known limitation, same category as the BTT
# USB automation's "BTT must already be running" note: a job only
# processes while backtalk itself is running, since nothing else is
# watching this file. Revisit as a standalone poller only if that's ever
# actually a problem in practice.
ROSA_QUEUE_FILE = BUS / ".rosa_queue.json"
ROSA_JOBS_SHOWN = 30


def _rosa_read_queue() -> list:
    try:
        data = json.loads(ROSA_QUEUE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _rosa_write_queue(jobs: list):
    ROSA_QUEUE_FILE.write_text(json.dumps(jobs), encoding="utf-8")


def read_rosa_state() -> dict:
    jobs = _rosa_read_queue()
    # Newest first, capped -- a face shows recent activity, not a
    # forever-growing history; the underlying file keeps everything.
    return {"jobs": list(reversed(jobs))[:ROSA_JOBS_SHOWN]}


def _rosa_result_text(job_id: str) -> str | None:
    for job in _rosa_read_queue():
        if job.get("id") == job_id and job.get("result_path"):
            try:
                return Path(job["result_path"]).read_text(
                    encoding="utf-8", errors="replace")
            except OSError:
                return None
    return None


def _rosa_delete_result_file(job: dict):
    # Kevin's explicit ask, 2026-09-07: clearing must actually delete the
    # output file from disk, not just drop the queue entry and leave an
    # orphaned .txt sitting in ~/Documents/Rosa/ forever.
    path = job.get("result_path")
    if path:
        Path(path).unlink(missing_ok=True)


def rosa_delete_job(job_id: str) -> bool:
    jobs = _rosa_read_queue()
    keep, found = [], False
    for job in jobs:
        if job.get("id") == job_id:
            found = True
            _rosa_delete_result_file(job)
        else:
            keep.append(job)
    if found:
        _rosa_write_queue(keep)
    return found


def rosa_clear_outbox() -> int:
    # Only done/error jobs -- exactly what the outbox displays. A pending
    # or running job is left untouched; "clear the outbox" should never
    # be able to nuke work still in flight.
    jobs = _rosa_read_queue()
    keep, cleared = [], 0
    for job in jobs:
        if job.get("status") in ("done", "error"):
            _rosa_delete_result_file(job)
            cleared += 1
        else:
            keep.append(job)
    _rosa_write_queue(keep)
    return cleared


def _sys_cpu_pct():
    try:
        out = subprocess.run(["top", "-l", "1", "-n", "0"],
                              capture_output=True, text=True,
                              timeout=3).stdout
        m = re.search(r"([\d.]+)% idle", out)
        return round(100.0 - float(m.group(1)), 1) if m else None
    except Exception:
        return None


def _sys_mem_pct():
    # Approximates Activity Monitor's own notion of "used": everything
    # that ISN'T immediately free or speculative (about-to-be-reclaimed
    # cache), out of physical RAM. A rough gauge, not an exact accounting
    # of wired/compressed/active -- good enough for a green/yellow/red
    # widget, not meant to replace Activity Monitor.
    try:
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                    capture_output=True, text=True,
                                    timeout=2).stdout.strip())
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                              timeout=2).stdout
        m = re.search(r"page size of (\d+) bytes", out)
        page_size = int(m.group(1)) if m else 4096
        free = 0
        for label in ("free", "speculative"):
            mm = re.search(rf"Pages {label}:\s+(\d+)\.", out)
            if mm:
                free += int(mm.group(1))
        used = total - free * page_size
        return round(100.0 * used / total, 1)
    except Exception:
        return None


def _sys_disk_pct():
    try:
        du = shutil.disk_usage("/")
        return round(100.0 * du.used / du.total, 1)
    except OSError:
        return None


def _sys_disk_io_mbs():
    # iostat's own -w interval sampling does the rate math, so there's no
    # manual delta to track here (unlike network below) -- the first row
    # is the average since boot, the second is the real last-second
    # sample, which is the one we want.
    try:
        out = subprocess.run(["iostat", "-d", "-c", "2", "-w", "1"],
                              capture_output=True, text=True,
                              timeout=4).stdout
        rows = [ln.split() for ln in out.splitlines()
                if ln.strip() and re.match(r"^\s*[\d.]+", ln)]
        if len(rows) >= 2:
            return round(float(rows[-1][-1]), 2)
    except Exception:
        pass
    return None


def _sys_default_iface():
    try:
        out = subprocess.run(["route", "get", "default"],
                              capture_output=True, text=True,
                              timeout=2).stdout
        m = re.search(r"interface:\s*(\S+)", out)
        return m.group(1) if m else "en0"
    except Exception:
        return "en0"


def _sys_net_total_bytes(iface: str):
    # The <Link#N> row is the authoritative per-interface total; the
    # other rows for the same interface (one per address family) repeat
    # the identical counters, so summing every match would double- or
    # triple-count. The Address field is BLANK for an interface with no
    # link-layer address (loopback) but POPULATED with a MAC for a real
    # one (en0 etc.), which shifts every column after it right by one --
    # a real bug caught by testing against this machine's actual en0
    # line, not assumed from the loopback shape alone. Branch on the
    # real field count instead of a fixed offset so both shapes parse
    # correctly.
    try:
        out = subprocess.run(["netstat", "-ib"], capture_output=True,
                              text=True, timeout=3).stdout
        for line in out.splitlines():
            f = line.split()
            if len(f) >= 8 and f[0] == iface and f[2].startswith("<Link"):
                if len(f) >= 11:      # has a MAC address field
                    return int(f[6]) + int(f[9])
                return int(f[5]) + int(f[8])   # no address field (loopback)
    except Exception:
        pass
    return None


def _sys_sample_loop():
    global _sys_prev_net, _sys_cpu_cores, _sys_top_cpu, _sys_top_mem_list, _sys_top_disk_io, _sys_top_net
    iface = _sys_default_iface()
    while True:
        cpu = _sys_cpu_pct()
        mem = _sys_mem_pct()
        disk_pct = _sys_disk_pct()
        disk_io = _sys_disk_io_mbs()
        cpu_cores, gpu_pct, top_cpu, top_disk_io, top_net = _sys_powermetrics()
        top_mem = _sys_top_mem()
        net_kbs = None
        total = _sys_net_total_bytes(iface)
        now = time.monotonic()
        if total is not None:
            if _sys_prev_net is not None:
                prev_bytes, prev_t = _sys_prev_net
                elapsed = now - prev_t
                if elapsed > 0:
                    net_kbs = round((total - prev_bytes) / elapsed / 1024, 1)
            _sys_prev_net = (total, now)
        sample = dict(cpu_pct=cpu, mem_pct=mem, disk_pct=disk_pct,
                     disk_io_mbs=disk_io, net_kbs=net_kbs, gpu_pct=gpu_pct)
        with _sys_lock:
            _sys_stats.update(sample)
            if cpu_cores:
                _sys_cpu_cores = cpu_cores
            _sys_top_cpu = top_cpu
            _sys_top_mem_list = top_mem
            _sys_top_disk_io = top_disk_io
            _sys_top_net = top_net
            # None (a metric that failed or hasn't produced its first
            # real sample yet) is skipped rather than plotted as 0 --
            # a real 0% reads identically to "no data" otherwise, and
            # a graph that dips to the floor every time one shell-out
            # hiccups is worse than a short gap in the line.
            for k, v in sample.items():
                if v is not None:
                    _sys_history[k].append(v)
        time.sleep(SYS_SAMPLE_S)


def read_sys_stats() -> dict:
    with _sys_lock:
        out = dict(_sys_stats)
        out["history"] = {k: list(v) for k, v in _sys_history.items()}
        out["cpu_cores"] = list(_sys_cpu_cores)
        out["top_cpu"] = list(_sys_top_cpu)
        out["top_mem"] = list(_sys_top_mem_list)
        out["top_disk_io"] = list(_sys_top_disk_io)
        out["top_net"] = list(_sys_top_net)
    with _sys_disk_files_lock:
        out["top_disk_files"] = list(_sys_disk_files)
    return out


def read_bus():
    if MOCK:
        return mock_bus()
    try:
        state = (BUS / ".voice_state").read_text(encoding="utf-8").strip().lower()
        if state not in STATES:
            state = "idle"
    except OSError:
        state = "idle"
    if state == "idle" and _background_job_active():
        state = "working"
    level = 0.0
    samples = [0.0] * 64
    try:
        payload = json.loads((BUS / ".voice_waveform").read_text(encoding="utf-8"))
        age = time.time() - float(payload.get("ts", 0))
        raw = payload.get("samples") or []
        if raw and age < WAVEFORM_STALE_S:
            # A fresh waveform IS speech, whatever the state file says.
            state = "speaking"
            samples = [float(s) for s in raw[:64]]
            mean = sum(abs(s) for s in samples) / len(samples)
            level = min(1.0, mean / 3000.0)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        alert = (BUS / ".voice_alert").stat().st_size > 0
    except OSError:
        alert = False
    loading = (BUS / ".voice_loading_pid").exists()
    # Absent unless the voice line was told to publish it, which is the
    # normal case: it is the account holder's own spend and it stays off
    # until asked for. An empty dict simply means no readout.
    rate_limits = {}
    try:
        rate_limits = json.loads((BUS / ".voice_rate_limits").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    note = {}
    try:
        note = json.loads((BUS / ".agent_note").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    # A PreToolUse hook (.claude/hooks/activity_log.py) appends here on
    # every tool call, so a face can show what is actually happening
    # instead of going idle-looking during a long silent stretch of work.
    activity = []
    try:
        activity = json.loads((BUS / ".agent_activity").read_text(encoding="utf-8"))
        if not isinstance(activity, list):
            activity = []
    except (OSError, ValueError):
        pass
    # Rolling conversation captions, written by backtalk: one entry per
    # spoken line, {"ts", "role", "text"}. Subtitles, and a scrollback for
    # when the room was noisy.
    transcript = []
    try:
        transcript = json.loads((BUS / ".voice_transcript").read_text(encoding="utf-8"))
        if not isinstance(transcript, list):
            transcript = []
    except (OSError, ValueError):
        pass
    # Absent unless local_llm.enabled — which model answered the turn
    # in flight, "local" or "cloud". See backtalk's signals.set_source.
    source = ""
    try:
        source = (BUS / ".voice_source").read_text().strip().lower()
    except OSError:
        pass
    # backtalk's own afplay thinking-cue volume and TTS output gain --
    # neither is a browser setting, both live on the bus so a face's
    # slider can read the current value and POST /volume to change it.
    thinking_volume = _read_volume(".thinking_volume", 0.35)
    voice_volume = _read_volume(".voice_volume", 1.0)
    # Silent mode: no TTS, no thinking cue, no push-to-talk -- a face's
    # chat box (POST /type) replaces all three. See backtalk's
    # signals.is_silent_mode()/get_typed_input().
    silent_mode = _read_silent_mode()
    # Which model tier backtalk reports live ("fast"|"deep"|"fable"|"haiku") --
    # written at startup and on every console model switch, so a face's
    # model selector shows real state. "" when backtalk predates this.
    model = ""
    try:
        model = (BUS / ".voice_model").read_text().strip()
    except OSError:
        pass
    return {"state": state, "level": level, "samples": samples, "model": model,
            "alert": alert, "loading": loading, "rate_limits": rate_limits,
            "source": source,
            "thinking_volume": thinking_volume, "voice_volume": voice_volume,
            "silent_mode": silent_mode,
            "system": read_sys_stats(), "calendar": read_cal_state(),
            "rosa": read_rosa_state(),
            "note": note, "activity": activity, "transcript": transcript}


def _read_volume(filename: str, default: float) -> float:
    try:
        v = float((BUS / filename).read_text().strip())
        return min(1.0, max(0.0, v))
    except (OSError, ValueError):
        return default


INBOX_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif",
             "image/webp": "webp"}
INBOX_MAX = 25 * 1024 * 1024


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            if path == "/inbox":
                self._inbox()
            elif path == "/volume":
                self._volume()
            elif path == "/mode":
                self._mode()
            elif path == "/type":
                self._type()
            elif path == "/calendar_selection":
                self._calendar_selection()
            elif path == "/rosa_submit":
                self._rosa_submit()
            elif path == "/rosa_delete":
                self._rosa_delete()
            elif path == "/rosa_clear":
                self._rosa_clear()
            elif path == "/choose_file":
                self._choose_file()
            elif path == "/mic_primed":
                self._mic_primed()
            else:
                self._send(b"not found", "text/plain", 404)
        except ConnectionError:
            pass
        except Exception as e:
            body = json.dumps({"error": str(e)}).encode()
            try:
                self._send(body, "application/json", 500)
            except ConnectionError:
                pass

    def _inbox(self):
        # A screen-snip pasted into any face lands here and is written
        # straight to the bus, so whatever is driving the session can just
        # read inbox/latest.<ext> without needing to be told the filename.
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > INBOX_MAX:
            self._send(json.dumps({"error": "bad length"}).encode(),
                       "application/json", 400)
            return
        data = self.rfile.read(length)
        ext = INBOX_EXT.get(self.headers.get("Content-Type", ""), "png")
        d = BUS / "inbox"
        d.mkdir(parents=True, exist_ok=True)
        name = f"paste-{time.strftime('%Y%m%d-%H%M%S')}.{ext}"
        (d / name).write_bytes(data)
        (d / f"latest.{ext}").write_bytes(data)
        for other in INBOX_EXT.values():
            if other != ext:
                (d / f"latest.{other}").unlink(missing_ok=True)
        self._send(json.dumps({"ok": True, "file": name}).encode(),
                   "application/json")

    VOLUME_FILES = {"thinking": ".thinking_volume", "voice": ".voice_volume"}

    def _volume(self):
        # A slider on a face's own UI lands here -- backtalk has no HTTP
        # server of its own, so the shared bus file is the only channel
        # back to it. Written as plain text, same as the other single-value
        # bus files (.voice_state, .voice_source), not JSON.
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length)) if length else {}
            kind = body.get("kind")
            value = float(body.get("value"))
        except (ValueError, TypeError):
            self._send(json.dumps({"error": "bad body"}).encode(),
                       "application/json", 400)
            return
        filename = self.VOLUME_FILES.get(kind)
        if not filename:
            self._send(json.dumps({"error": "bad kind"}).encode(),
                       "application/json", 400)
            return
        value = min(1.0, max(0.0, value))
        (BUS / filename).write_text(f"{value:.3f}")
        self._send(json.dumps({"ok": True, "kind": kind, "value": value}).encode(),
                   "application/json")

    def _mic_primed(self):
        # core.js's micPrime() calls this the instant it's done grabbing
        # and releasing the mic on page load. Exists so "Talk to Cipher"
        # (streamdeck_talk_to_cipher.sh) can wait for the REAL condition
        # -- mic priming actually finished, macOS's output-ducking window
        # for it actually closed -- instead of guessing a fixed sleep
        # long enough to outlast it. Confirmed live 2026-09-08: a fixed
        # delay isn't reliable once page-load itself is slow under system
        # load, and this exact mic-hold mechanism was already confirmed
        # (2026-09-06, a different symptom) to duck backtalk's own output
        # volume for as long as anything holds an open mic stream.
        (BUS / ".mic_primed").write_text(str(time.time()))
        self._send(json.dumps({"ok": True}).encode(), "application/json")

    def _mode(self):
        # Silent/Voice toggle -- backtalk's is_silent_mode() reads this
        # same file. No TTS, no thinking cue, no push-to-talk while "1".
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length)) if length else {}
            silent = bool(body.get("silent"))
        except (ValueError, TypeError):
            self._send(json.dumps({"error": "bad body"}).encode(),
                       "application/json", 400)
            return
        (BUS / ".silent_mode").write_text("1" if silent else "0")
        self._send(json.dumps({"ok": True, "silent": silent}).encode(),
                   "application/json")

    TYPE_MAX = 4000

    def _type(self):
        # Silent mode's chat box -- backtalk's get_typed_input() reads
        # and clears this same file, treating it as one first-class turn,
        # same as a spoken utterance. Not a queue: one pending message at
        # a time, matching how a chat box is actually used (send, wait
        # for the reply, send the next).
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > self.TYPE_MAX:
            self._send(json.dumps({"error": "bad length"}).encode(),
                       "application/json", 400)
            return
        try:
            body = json.loads(self.rfile.read(length))
            text = str(body.get("text", "")).strip()
        except (ValueError, TypeError):
            self._send(json.dumps({"error": "bad body"}).encode(),
                       "application/json", 400)
            return
        if not text:
            self._send(json.dumps({"error": "empty text"}).encode(),
                       "application/json", 400)
            return
        (BUS / ".typed_input").write_text(text)
        self._send(json.dumps({"ok": True}).encode(), "application/json")

    def _calendar_selection(self):
        # The picker pop-out POSTs the full checked set here on every
        # change. Written to the same bus file the poll loop reads, plus
        # an immediate re-query so the panel updates right away instead
        # of waiting up to CAL_POLL_S for the next background tick.
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length)) if length else {}
            pks = [int(x) for x in body.get("pks", [])]
        except (ValueError, TypeError):
            self._send(json.dumps({"error": "bad body"}).encode(),
                       "application/json", 400)
            return
        _cal_write_selection(pks)
        calendars, events, month_days = _cal_query(pks)
        global _cal_calendars, _cal_events, _cal_month_days
        with _cal_lock:
            _cal_calendars = calendars
            _cal_events = events
            _cal_month_days = month_days
        self._send(json.dumps({"ok": True, "selected": pks}).encode(),
                   "application/json")

    ROSA_SUBMIT_MAX = 4000

    def _rosa_submit(self):
        # Queues a job for backtalk's own _rosa_queue_loop to pick up --
        # this process never calls the model itself, it only writes the
        # request. At least one of file_path/text, plus a non-empty
        # instruction. file_path is read directly off disk by backtalk
        # (same local-machine trust level as everything else here, e.g.
        # the calendar widget reading Spark's db straight off disk).
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0 or length > self.ROSA_SUBMIT_MAX:
            self._send(json.dumps({"error": "bad length"}).encode(),
                       "application/json", 400)
            return
        try:
            body = json.loads(self.rfile.read(length))
            instruction = str(body.get("instruction", "")).strip()
            file_path = str(body.get("file_path", "")).strip()
            text = str(body.get("text", "")).strip()
        except (ValueError, TypeError):
            self._send(json.dumps({"error": "bad body"}).encode(),
                       "application/json", 400)
            return
        if not instruction or not (file_path or text):
            self._send(json.dumps(
                {"error": "instruction and (file_path or text) required"}
            ).encode(), "application/json", 400)
            return
        job = {"id": uuid.uuid4().hex[:8], "instruction": instruction,
               "file_path": file_path or None, "text": text or None,
               "status": "pending", "created_at": time.time(),
               "result_path": None, "error": None}
        jobs = _rosa_read_queue()
        jobs.append(job)
        _rosa_write_queue(jobs)
        self._send(json.dumps({"ok": True, "id": job["id"]}).encode(),
                   "application/json")

    def _rosa_delete(self):
        # Deletes one job's queue entry AND its output file on disk, if
        # it has one -- Kevin's explicit ask, 2026-09-07: a cleared entry
        # must never leave an orphaned .txt behind in ~/Documents/Rosa/.
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length)) if length else {}
            job_id = str(body.get("id", "")).strip()
        except (ValueError, TypeError):
            self._send(json.dumps({"error": "bad body"}).encode(),
                       "application/json", 400)
            return
        if not job_id:
            self._send(json.dumps({"error": "id required"}).encode(),
                       "application/json", 400)
            return
        found = rosa_delete_job(job_id)
        self._send(json.dumps({"ok": found}).encode(), "application/json")

    def _rosa_clear(self):
        # Clears every done/error job (what the outbox actually shows)
        # and deletes each one's output file from disk -- pending/running
        # jobs are left alone, "clear the outbox" should never touch work
        # still in flight.
        cleared = rosa_clear_outbox()
        self._send(json.dumps({"ok": True, "cleared": cleared}).encode(),
                   "application/json")

    def _choose_file(self):
        # A real native Finder "choose file" dialog -- the browser's own
        # <input type=file> deliberately never exposes an absolute path
        # (a universal browser security restriction, not something any
        # amount of JS can work around), but this server IS a plain local
        # Python process on Kevin's own Mac with full OS access, same
        # trust level as every other subprocess call in this file. Kevin's
        # ask, 2026-09-07: clicking the Rosa queue's file-path field
        # should pull up Finder, not require typing a path by hand.
        # Blocks this ONE request's thread while the dialog is open --
        # fine, ThreadingHTTPServer gives every request its own thread,
        # so /state polling elsewhere is untouched.
        try:
            result = subprocess.run(
                ["osascript", "-e",
                 'POSIX path of (choose file with prompt '
                 '"Choose a file for Rosa to process")'],
                capture_output=True, text=True, timeout=300)
        except Exception:
            self._send(json.dumps({"ok": False}).encode(), "application/json")
            return
        if result.returncode != 0:
            # Cancel button, or the dialog errored -- either way, no path.
            self._send(json.dumps({"ok": False}).encode(), "application/json")
            return
        self._send(json.dumps({"ok": True, "path": result.stdout.strip()}).encode(),
                   "application/json")

    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path == "/state":
                self._send(json.dumps(read_bus()).encode(),
                           "application/json")
            elif path == "/config":
                out = {"name": CFG["name"], "badge": CFG["badge"],
                       "face": CFG["face"],
                       "thinking_sound": bool(CFG["thinking_sound"]),
                       "local_name": CFG.get("local_name", ""),
                       "faces": list_faces()}
                self._send(json.dumps(out).encode(), "application/json")
            elif path == "/calendar_day":
                date_str = (parse_qs(urlparse(self.path).query).get("date")
                            or [""])[0]
                events = _cal_day_events(date_str) if date_str else []
                self._send(json.dumps({"date": date_str,
                                       "events": events}).encode(),
                           "application/json")
            elif path == "/rosa_result":
                job_id = (parse_qs(urlparse(self.path).query).get("id")
                          or [""])[0]
                text = _rosa_result_text(job_id) if job_id else None
                self._send(json.dumps({"id": job_id, "text": text}).encode(),
                           "application/json")
            else:
                self._static(path)
        except ConnectionError:
            # THE WHOLE FAMILY, not one member of it. A tab closed or
            # reloaded mid-response raises ConnectionResetError, which is a
            # SIBLING of BrokenPipeError rather than a subclass -- so
            # catching only BrokenPipeError sent it to the generic branch
            # below, which then wrote a 500 back down the socket that had
            # just died and raised a SECOND, uncaught error from inside
            # flush_headers(). One disconnect, two tracebacks. ConnectionError
            # is the common parent of Reset, Broken, Aborted and Refused.
            pass
        except Exception as e:
            body = json.dumps({"error": str(e)}).encode()
            try:
                self._send(body, "application/json", 500)
            except ConnectionError:
                # A real error AND the client already gone. There is nobody
                # left to tell; saying so twice helps no one.
                pass

    def _static(self, path):
        if path == "/":
            path = "/index.html"
        target = (HERE / path.lstrip("/")).resolve()
        if target != HERE and HERE not in target.parents:
            self._send(b"not found", "text/plain", 404)
            return
        if target.is_dir():
            target = target / "index.html"
        if not target.is_file():
            self._send(b"not found", "text/plain", 404)
            return
        ctype = mimetypes.guess_type(str(target))[0] or \
            "application/octet-stream"
        self._send(target.read_bytes(), ctype)

    def _send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def open_visualizer(url):
    """Open the visualizer. If ai-visualizer.json sets "open_app" (a full
    path to a standalone .app -- e.g. a Safari "Add to Dock" web app
    pinned to this same URL -- or a list of them, when more than one face
    has its own pinned app), launch the first one instead of a browser
    tab, since a dedicated app has a far smaller memory footprint than a
    Chrome tab (confirmed 2026-09-05: ~300MB isolated vs 800MB+ for one
    Chrome tab). Falls back to opening Chrome directly, then the OS
    default browser, if no "open_app" is configured or launching it
    fails."""
    mac_app = CFG.get("open_app")
    if mac_app:
        first = mac_app[0] if isinstance(mac_app, list) else mac_app
        try:
            subprocess.run(["open", "-a", first], check=True)
            return
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    try:
        subprocess.run(["open", "-a", "Google Chrome", url], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        webbrowser.open(url)


if __name__ == "__main__":
    threading.Thread(target=_sys_sample_loop, daemon=True).start()
    threading.Thread(target=_sys_disk_scan_loop, daemon=True).start()
    threading.Thread(target=_cal_poll_loop, daemon=True).start()
    mode = f"MOCK={MOCK}" if MOCK else f"bus: {BUS}"
    root = f"http://127.0.0.1:{PORT}/"
    # The browser opens on the configured face; the gallery stays at "/" for switching.
    face = CFG.get("face", "")
    url = f"{root}faces/{face}/" if face and (HERE / "faces" / face / "index.html").exists() else root
    # ALREADY RUNNING IS NOT AN ERROR, and treating it as one was the whole
    # bug. Closing the browser tab does not stop this server; it keeps going
    # headless. Relaunching then failed to bind, died before the line that
    # opens the browser, and took the traceback with it when the launcher
    # window closed. The end-user symptom was "I can hear my agent but the
    # face never shows up", with the face running perfectly the entire time.
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError as e:
        if e.errno not in (errno.EADDRINUSE, errno.EACCES):
            raise
        # Something holds the port. Ask it whether it is us before claiming
        # anything: a stranger on this port is a different problem and
        # deserves a different sentence.
        mine = False
        try:
            with urllib.request.urlopen(root + "state", timeout=2) as r:
                mine = r.status == 200
        except Exception:
            mine = False
        if mine:
            print(f"already running at {root}  opening it instead", flush=True)
            if not NO_OPEN:
                open_visualizer(url)
            sys.exit(0)
        print(f"port {PORT} is taken by something that is not this server.",
              flush=True)
        print("Close whatever is using it, or set a different \"port\" in "
              "ai-visualizer.json.", flush=True)
        sys.exit(1)
    srv.allow_reuse_address = True
    print(f"ai-visualizer on {root}  opening {url}  ({mode})  Ctrl-C stops", flush=True)

    if not NO_OPEN:
        threading.Timer(0.6, open_visualizer, args=(url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
