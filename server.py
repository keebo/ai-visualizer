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
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
import urllib.request
import errno
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
SYS_SAMPLE_S = 3.0
_sys_lock = threading.Lock()
_sys_stats = {"cpu_pct": None, "mem_pct": None, "disk_pct": None,
              "disk_io_mbs": None, "net_kbs": None}
_sys_prev_net = None   # (bytes_total, monotonic_ts), for a manual rate calc


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
    global _sys_prev_net
    iface = _sys_default_iface()
    while True:
        cpu = _sys_cpu_pct()
        mem = _sys_mem_pct()
        disk_pct = _sys_disk_pct()
        disk_io = _sys_disk_io_mbs()
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
        with _sys_lock:
            _sys_stats.update(cpu_pct=cpu, mem_pct=mem, disk_pct=disk_pct,
                              disk_io_mbs=disk_io, net_kbs=net_kbs)
        time.sleep(SYS_SAMPLE_S)


def read_sys_stats() -> dict:
    with _sys_lock:
        return dict(_sys_stats)


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
            "system": read_sys_stats(),
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
