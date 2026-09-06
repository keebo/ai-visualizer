/*
 * ai-visualizer: give your AI agent a face.
 * Copyright (C) 2026 Jared Rhodenizer
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */
/* ============================================================
   ai-visualizer core — the shared plumbing every face rides on.

   A face is one self-contained page in faces/<name>/index.html.
   It includes this script, calls AV.init(opts), then reads these
   fields every animation frame after calling AV.tick(dtMs):

     AV.state      "idle" | "listening" | "thinking" | "working" | "speaking"
     AV.stateElapsed  ms since AV.state last changed — a face can use
                   this to surface "still working, Ns" on a long thinking
     AV.level      0..1 raw voice loudness (speaking only)
     AV.env        0..1 smoothed speech envelope (attack/release eased,
                   adaptively normalized — use this for motion)
     AV.samples    Float32Array(64), 0..1 normalized waveform ring
     AV.alert      bool, optional attention signal
     AV.micLevel   0..1 your microphone (only if init({mic:true}))
     AV.name       display name from config ("JARVIS" by default)
     AV.label      the dotted chip label ("J.A.R.V.I.S.")
     AV.badge      optional handle from config ("" by default)
     AV.source     "" | "local" | "cloud" — which model answered the
                   turn in flight, only set when backtalk's
                   local_llm.enabled is on
     AV.localName  the local model's own name from config
                   (ai-visualizer.json's local_name), "" if unset
     AV.note       {title,text} or null — free text dropped on the bus
                   (.agent_note) by whatever is driving the session, for
                   a face to render as a HUD panel; null when empty
     AV.transcript [{ts,role,text}, ...], oldest first, the spoken
                   conversation as captions (.voice_transcript)
     AV.activity   [{ts,tool,detail}, ...], oldest first, kept by a
                   PreToolUse hook (.agent_activity) — a live "what is
                   Jarvis doing" feed a face can render as a terminal

   Modes:
     live   served by server.py — rides the real signal bus
     demo   ?demo=1, or the page opened as a plain file — a scripted
            voice-turn loop (idle, listening, thinking, working,
            speaking) with synthesized audio, so every face performs
            with no voice line installed
     shot   ?shot=<state>&t=ms — pins one state and runs the frame
            loop deterministically, then sets document.title to
            "ready" (screenshot/verification harness)

   The thinking sound: assets/thinking.wav plays while the state is
   "thinking" or "working", exactly like a voice line would play it.
   If the bus says the voice line is already playing its own
   (.voice_loading_pid), this player stays quiet — you never hear it
   twice. The speaker button (bottom left) toggles it; browsers may
   require one click on the page before audio is allowed.

   The paste inbox: Ctrl+V anywhere on the page with an image on the
   clipboard (a Win+Shift+S snip, say) POSTs it to /inbox, no click to
   open anything first. A toast confirms the save; server.py writes it
   to inbox/latest.<ext> under the bus dir for whatever is driving the
   session to pick up.
   ============================================================ */
"use strict";

const AV = (() => {
  const Q = new URLSearchParams(location.search);
  const SHOT = Q.get("shot");
  const SHOT_T = parseInt(Q.get("t") || "4000", 10);
  const DEMO = Q.get("demo") === "1" || location.protocol === "file:" || !!SHOT;

  // where core.js lives -> where assets/ lives (works over http and file://)
  const ROOT = new URL(".", document.currentScript.src);

  const A = {
    state: "idle", level: 0, env: 0, alert: false, micLevel: 0,
    stateElapsed: 0, note: null, activity: [], transcript: [],
    samples: new Float32Array(64),
    name: "JARVIS", label: "J.A.R.V.I.S.", badge: "",
    demo: DEMO, shot: SHOT, faces: [],
    _sndOn: true, _mic: false, _readyCbs: [], _ready: false,
    _lastSpeakingT: -1e9,
  };

  function dotted(name) {
    const up = String(name).toUpperCase();
    if (/^[A-Z0-9]{2,10}$/.test(up)) return up.split("").join(".") + ".";
    return up;
  }

  /* -------------------------------- config -------------------------------- */
  function applyConfig(cfg) {
    if (cfg.name) { A.name = String(cfg.name); A.label = dotted(A.name); }
    A.badge = String(cfg.badge || "");
    // Shown in place of a generic "LOCAL" tag when AV.source === "local"
    // — "" falls back to that generic tag, a face isn't required to use it.
    A.localName = String(cfg.local_name || "");
    if (cfg.thinking_sound === false) A._sndWant = false;
    A.faces = cfg.faces || [];
    A._ready = true;
    A._readyCbs.forEach(cb => cb(A));
    A._readyCbs = [];
  }

  A.ready = cb => { A._ready ? cb(A) : A._readyCbs.push(cb); };

  /* ------------------------------ bus polling ------------------------------ */
  let raw = { state: "idle", level: 0, samples: null, alert: false,
              loading: false };
  if (!DEMO) {
    setInterval(async () => {
      try {
        const r = await fetch("/state", { cache: "no-store" });
        raw = await r.json();
      } catch (e) { /* server gone: hold last state */ }
    }, 120);
  }

  /* ------------------------------ demo driver ------------------------------ */
  // A scripted voice turn: the face performs everything with no voice line.
  const SCRIPT = [["idle", 6000], ["listening", 3500], ["thinking", 4200],
                  ["working", 4200], ["speaking", 8500]];
  let demoT = 0, demoClock = 0;
  const PIN = SHOT || Q.get("state");   // ?state=speaking pins the demo
  function demoUpdate(dt) {
    demoClock += dt;
    let st = PIN || "idle";
    if (!PIN) {
      demoT = (demoT + dt) % SCRIPT.reduce((a, s) => a + s[1], 0);
      let t = demoT;
      for (const [name, len] of SCRIPT) {
        if (t < len) { st = name; break; }
        t -= len;
      }
    }
    const tt = demoClock / 1000;
    const speaking = st === "speaking";
    const cadence = speaking
      ? Math.max(0, Math.sin(tt * 2.1) * 0.6 + Math.sin(tt * 0.9) * 0.5)
      : 0;
    const samples = new Array(64);
    for (let i = 0; i < 64; i++) {
      // drifting per-sample color so the synthetic voice has a moving
      // spectrum, not a steady tone — spectrum-driven faces dance
      const m = 0.3 + 0.7 * Math.abs(Math.sin(i * 0.23 + tt * 1.7))
        * Math.abs(Math.sin(tt * 2.9 + i * 0.05));
      samples[i] = speaking
        ? (Math.sin(i * 0.55 + tt * 9) * 0.6 + Math.sin(i * 1.7 - tt * 13)
           * 0.4) * 9000 * (0.15 + 0.85 * cadence) * m
        : 0;
    }
    raw = { state: st, level: speaking ? Math.min(1, cadence) : 0,
            samples, alert: false, loading: false };
    if (st === "listening")
      A.micLevel = 0.25 + 0.55 * Math.abs(Math.sin(tt * 2.7))
        * Math.abs(Math.sin(tt * 0.61));
  }

  // A chart is present if it carries rows a face could actually draw —
  // a flat/pie series ("data"), at least one grouped-bar series with rows
  // ("bars"), or at least one named network ("networks", topology) —
  // independent of which face is looking at it.
  function noteChartPresent(c) {
    if (!c) return false;
    if (Array.isArray(c.data) && c.data.length) return true;
    if (Array.isArray(c.bars) && c.bars.some(b => Array.isArray(b.data) && b.data.length)) return true;
    if (Array.isArray(c.networks) && c.networks.length) return true;
    return false;
  }

  /* ----------------------- envelope + samples easing ----------------------- */
  let peak = 0.05, sPeak = 200;
  let stateSince = 0;
  function tick(dt) {
    if (DEMO) demoUpdate(dt);
    const st = raw.state || "idle";
    if (st !== A.state) {
      stateSince = 0;
      if (A._mic && !DEMO) {
        if (st === "listening") micStart();
        else if (A.state === "listening") micStop();
      }
    }
    A.state = st;
    stateSince += dt;
    A.stateElapsed = stateSince;
    A.alert = !!raw.alert;
    // Empty unless the voice line was told to publish usage. A face that
    // wants to draw it reads AV.rateLimits; every other face ignores it.
    A.rateLimits = raw.rate_limits || {};
    // "" unless local_llm.enabled — "local" or "cloud", which model
    // answered the turn in flight. A face that wants to flag a
    // local-model answer reads AV.source; every other face ignores it.
    A.source = raw.source || "";
    A.note = (raw.note && (raw.note.text || noteChartPresent(raw.note.chart))) ? raw.note : null;
    A.activity = Array.isArray(raw.activity) ? raw.activity : [];
    A.transcript = Array.isArray(raw.transcript) ? raw.transcript : [];
    A.level = raw.level || 0;
    if (A.state === "speaking") A._lastSpeakingT = performance.now();

    // adaptive envelope: normalize against a decaying peak, then ease
    // (attack 50ms, release 350ms) — motion code rides AV.env
    const dts = dt / 1000;
    peak = Math.max(A.level, 0.05, peak - 0.5 * peak * dts);
    const target = Math.min(1, A.level / peak);
    const tau = target > A.env ? 50 : 350;
    A.env += (target - A.env) * Math.min(1, dt / tau);

    // waveform ring: rectify, normalize against its own decaying peak,
    // blend toward the newest frame so the ring flows instead of flickers
    const s = raw.samples;
    A.rawSamples = s && s.length ? s : null;   // signed, int16-scale floats
    if (s && s.length) {
      let mx = 0;
      for (let i = 0; i < s.length; i++) mx = Math.max(mx, Math.abs(s[i]));
      sPeak = Math.max(mx, 200, sPeak * 0.98);
      const n = s.length;
      for (let i = 0; i < 64; i++) {
        const v = Math.abs(s[Math.min(n - 1, Math.round(i * (n - 1) / 63))])
          / sPeak;
        A.samples[i] = A.samples[i] * 0.45 + Math.min(1, v) * 0.55;
      }
    } else {
      for (let i = 0; i < 64; i++) A.samples[i] *= Math.max(0, 1 - dts * 6);
    }
    if (A.state !== "speaking" && !DEMO)
      for (let i = 0; i < 64; i++) A.samples[i] *= Math.max(0, 1 - dts * 6);

    if (A._mic && A._micAnalyser) micRead();
    soundUpdate();
  }

  /* --------------------------------- mic ----------------------------------
     Grabbed only while AV.state === "listening" (push-to-talk actually
     held), not for the life of the page -- macOS ducks other apps' output
     volume for as long as anything holds an open mic stream, and holding
     it continuously was quietly lowering the voice line's own spoken
     replies for the entire session (found 2026-09-06, reactor is the only
     face that sets mic:true). Started/stopped from tick()'s state-change
     edge below. */
  let micPeak = 0.02;
  function micRead() {
    const an = A._micAnalyser;
    const buf = A._micBuf;
    an.getFloatTimeDomainData(buf);
    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    const rms = Math.sqrt(sum / buf.length);
    micPeak = Math.max(rms, 0.02, micPeak * 0.999);
    A.micLevel = Math.min(1, rms / micPeak);
  }
  let kickWired = false;
  function ensureKick() {
    if (kickWired) return;
    kickWired = true;
    // resumes whichever mic AudioContext is current at the time, not the
    // one live when this listener was first attached -- attached once,
    // reused across every start/stop cycle instead of stacking a new pair
    // of listeners on every PTT press
    const kick = () => { const c = A._micCtx; if (c && c.state === "suspended") c.resume(); };
    addEventListener("click", kick); addEventListener("keydown", kick);
  }
  let micStarting = false;
  async function micStart() {
    if (micStarting || A._micStream) return;
    micStarting = true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (A.state !== "listening") { stream.getTracks().forEach(t => t.stop()); return; }
      const ctx = new AudioContext();
      const src = ctx.createMediaStreamSource(stream);
      const an = ctx.createAnalyser();
      an.fftSize = 512;
      src.connect(an);
      A._micStream = stream;
      A._micCtx = ctx;
      A._micAnalyser = an;
      A._micBuf = new Float32Array(an.fftSize);
      ensureKick();
    } catch (e) { /* no mic permission: level stays 0, faces degrade */ }
    finally { micStarting = false; }
  }
  function micStop() {
    try { A._micStream && A._micStream.getTracks().forEach(t => t.stop()); } catch (e) {}
    try { A._micCtx && A._micCtx.close(); } catch (e) {}
    A._micStream = null; A._micCtx = null; A._micAnalyser = null; A._micBuf = null;
    A.micLevel = 0;
  }

  /* ----------------------------- thinking sound ---------------------------- */
  let audio = null, sndBtn = null, playing = false;
  A._sndWant = true;
  function soundInit() {
    if (SHOT) return;
    try { A._sndOn = localStorage.getItem("av_sound") !== "0"; }
    catch (e) { A._sndOn = true; }
    audio = new Audio(new URL("assets/thinking.wav", ROOT).href);
    audio.volume = 0.35;
    sndBtn = document.createElement("div");
    // hidden until the mouse moves, so it never collides with a face's
    // chrome and never shows on camera or in an OBS source
    sndBtn.style.cssText =
      "position:fixed;left:64px;bottom:14px;z-index:50;cursor:pointer;" +
      "font:12px 'SF Mono',Menlo,Consolas,monospace;letter-spacing:.2em;" +
      "color:#5a6a72;opacity:0;transition:opacity .4s;user-select:none;" +
      "pointer-events:none";
    sndBtn.title = "thinking sound on/off";
    let hideT = null;
    addEventListener("mousemove", () => {
      sndBtn.style.opacity = ".65";
      sndBtn.style.pointerEvents = "auto";
      clearTimeout(hideT);
      hideT = setTimeout(() => {
        sndBtn.style.opacity = "0";
        sndBtn.style.pointerEvents = "none";
      }, 3000);
    });
    sndBtn.onclick = () => {
      A._sndOn = !A._sndOn;
      try { localStorage.setItem("av_sound", A._sndOn ? "1" : "0"); }
      catch (e) {}
      if (!A._sndOn) stopSound();
      paintBtn();
    };
    paintBtn();
    document.body.appendChild(sndBtn);
  }
  function paintBtn() {
    if (sndBtn) sndBtn.textContent = A._sndOn ? "SND ON" : "SND OFF";
  }
  function stopSound() {
    if (audio && playing) { audio.pause(); audio.currentTime = 0; }
    playing = false;
  }
  function soundUpdate() {
    if (!audio || !A._sndWant) return;
    // The backend re-asserts "speaking" every ~70-90ms while mouth.py is
    // actually writing audio to the device (its own self-heal). We only
    // poll /state every 120ms, so a "working"/"thinking" write that lands
    // mid-sentence (a tool call starting while a prior line is still
    // physically playing) can be caught before the self-heal overwrites
    // it. Hold off starting the sound until it's been a beat since we
    // last actually saw "speaking" — long enough to absorb that race and
    // a missed poll, short enough to never delay a real thinking/working.
    const graceOk = performance.now() - A._lastSpeakingT > 300;
    const want = A._sndOn && (A.state === "thinking" || A.state === "working")
      && !raw.loading && graceOk;
    if (want && !playing) {
      playing = true;
      audio.currentTime = 0;
      audio.play().catch(() => { playing = false; });
    } else if (!want && playing) {
      stopSound();
    }
  }

  /* ------------------------------ paste inbox ------------------------------ */
  // Paste an image anywhere on the page (Ctrl+V — a Win+Shift+S snip is
  // already on the clipboard) and it's POSTed straight to /inbox, no
  // click-to-open step. A small toast confirms it landed; the file itself
  // goes to the bus at inbox/latest.<ext> for whatever is driving the
  // session to read.
  let toastEl = null, toastT = null;
  function toast(msg, ok) {
    if (!toastEl) {
      toastEl = document.createElement("div");
      toastEl.style.cssText =
        "position:fixed;left:50%;bottom:54px;transform:translateX(-50%);" +
        "z-index:60;padding:10px 18px;border-radius:6px;border:1px solid;" +
        "font:14px 'SF Mono',Menlo,Consolas,monospace;letter-spacing:.03em;" +
        "background:rgba(10,14,18,.92);opacity:0;transition:opacity .25s;" +
        "pointer-events:none;white-space:nowrap";
      document.body.appendChild(toastEl);
    }
    toastEl.textContent = msg;
    toastEl.style.borderColor = ok ? "#3a9" : "#a53";
    toastEl.style.color = ok ? "#8fd" : "#f98";
    toastEl.style.opacity = "1";
    clearTimeout(toastT);
    toastT = setTimeout(() => { toastEl.style.opacity = "0"; }, 3200);
  }
  function pasteInit() {
    addEventListener("paste", async (e) => {
      const items = e.clipboardData && e.clipboardData.items;
      if (!items) return;
      let file = null;
      for (const it of items) if (it.type.startsWith("image/")) file = it.getAsFile();
      if (!file) return;
      e.preventDefault();
      try {
        const r = await fetch(new URL("inbox", ROOT).href, {
          method: "POST",
          headers: { "Content-Type": file.type || "image/png" },
          body: file,
        });
        const j = await r.json();
        toast(j.ok ? `saved ${j.file} — tell ${A.name || "Jarvis"} to look` : "save failed", !!j.ok);
      } catch (err) { toast("no server — can't save the paste", false); }
    });
  }

  /* ------------------------------ shot harness ----------------------------- */
  // Runs the face's frame() deterministically (a synchronous burst of t ms).
  // A headless browser resizes the window and finishes loading images AFTER
  // the first burst, so the burst re-runs on resize and on two late timers
  // (the last one flags "ready"), then keeps painting at frame pace so the
  // late capture always sees a fresh composite.
  A.shotRun = (frame) => {
    const burst = () => { for (let t = 0; t < SHOT_T; t += 16.6) frame(16.6); };
    burst();
    addEventListener("resize", burst);
    setTimeout(burst, 450);
    setTimeout(burst, 900);
    setTimeout(() => { burst(); document.title = "ready"; }, 3000);
    // fat 100ms steps: assets that finish loading after the last burst
    // still reach their steady state within a few paints
    const loop = () => { frame(100); requestAnimationFrame(loop); };
    requestAnimationFrame(loop);
  };

  /* ---------------------------------- init --------------------------------- */
  A.init = (opts = {}) => {
    A._mic = !!opts.mic;
    if (opts.sound !== false) soundInit(); else A._sndWant = false;
    pasteInit();
    if (DEMO) {
      applyConfig({ name: Q.get("name") || "JARVIS" });
    } else {
      fetch("/config", { cache: "no-store" })
        .then(r => r.json()).then(applyConfig)
        .catch(() => applyConfig({}));
    }
    return A;
  };

  A.tick = tick;

  /* ----------------------------- render helpers ---------------------------- */
  const U = {};
  U.dim = (c, f) => {
    f = Math.max(0, Math.min(1, f));
    return `rgb(${c[0] * f | 0},${c[1] * f | 0},${c[2] * f | 0})`;
  };
  U.rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

  // How long until a usage window resets, in the shortest honest unit.
  U.relTime = (ep) => {
    const d = ep - Date.now() / 1000;
    if (!(d > 0)) return "";
    if (d < 3600) return Math.round(d / 60) + "m";
    if (d < 86400) {
      const h = Math.floor(d / 3600), m = Math.round((d - h * 3600) / 60);
      return m < 60 ? `${h}h ${m}m` : `${h + 1}h 0m`;
    }
    return Math.round(d / 86400) + "d";
  };

  // The plan-usage windows, formatted ONCE for every face that draws them.
  // Lives here rather than in each face because four copies of one format
  // drift apart silently, and the first symptom is two faces disagreeing
  // about the same number.
  //
  // Returns [] when the voice line publishes no usage, so a face can call
  // it unconditionally and simply draw nothing when there is nothing to say.
  // A window that is KNOWN but has no percentage yet still returns a row:
  // hiding it entirely was the original bug, and a row that says "no number
  // yet" is information where a missing row is just confusing.
  U.usageRows = () => {
    const rl = A.rateLimits || {};
    const out = [];
    for (const [label, w] of [["5H", rl.five_hour], ["7D", rl.seven_day]]) {
      if (!w) continue;
      const known = w.utilization != null;
      const pct = known ? Math.round(w.utilization * 100) : null;
      const rel = w.resets_at ? U.relTime(w.resets_at) : "";
      // Same three-tier thresholds as the board face's usage readout:
      // green under 51%, amber 51-80%, red at 81%+ (unknown reads as amber).
      const level = !known ? "yellow" : pct >= 81 ? "red" : pct >= 51 ? "yellow" : "green";
      out.push({
        label, pct, known, level,
        hot: level === "red",
        text: (known ? pct + "%" : "\u2014") + (rel ? "  " + rel : "")
      });
    }
    return out;
  };
  U.mix = (c1, c2, t) => [c1[0] + (c2[0] - c1[0]) * t | 0,
                          c1[1] + (c2[1] - c1[1]) * t | 0,
                          c1[2] + (c2[2] - c1[2]) * t | 0];
  // soft additive glow sprite (canvas), cached by the caller
  U.makeGlow = (rgb, size) => {
    const c = document.createElement("canvas");
    c.width = c.height = size;
    const g = c.getContext("2d");
    const grd = g.createRadialGradient(size / 2, size / 2, 0,
                                       size / 2, size / 2, size / 2);
    grd.addColorStop(0, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},1)`);
    grd.addColorStop(.25, `rgba(${rgb[0]},${rgb[1]},${rgb[2]},.55)`);
    grd.addColorStop(1, "rgba(0,0,0,0)");
    g.fillStyle = grd;
    g.fillRect(0, 0, size, size);
    return c;
  };
  // the one-field bloom rule: draw everything luminous into one field
  // canvas, bloom the WHOLE field (two downscale taps), composite
  // additively — bloom applied per-element reads as pencil lines
  U.bloomBlit = (dst, field, w, h) => {
    if (!field._b4 || field._b4.width !== w >> 2) {
      field._b4 = document.createElement("canvas");
      field._b4.width = Math.max(1, w >> 2);
      field._b4.height = Math.max(1, h >> 2);
      field._b8 = document.createElement("canvas");
      field._b8.width = Math.max(1, w >> 3);
      field._b8.height = Math.max(1, h >> 3);
    }
    const g4 = field._b4.getContext("2d"), g8 = field._b8.getContext("2d");
    g4.clearRect(0, 0, field._b4.width, field._b4.height);
    g4.drawImage(field, 0, 0, field._b4.width, field._b4.height);
    g8.clearRect(0, 0, field._b8.width, field._b8.height);
    g8.drawImage(field, 0, 0, field._b8.width, field._b8.height);
    const prev = dst.globalCompositeOperation;
    dst.globalCompositeOperation = "lighter";
    dst.drawImage(field, 0, 0);
    dst.drawImage(field._b4, 0, 0, w, h);
    dst.drawImage(field._b8, 0, 0, w, h);
    dst.globalCompositeOperation = prev;
  };
  // text that resolves out of glyph noise, left to right
  U.Descrambler = class {
    constructor(text, perChar = 50, hold = null) {
      this.text = text; this.per = perChar; this.hold = hold;
      this.t = 0; this.done = false;
      this.chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#$%&";
    }
    render(dt) {
      this.t += dt;
      const n = this.t / this.per | 0;
      let out = "";
      for (let i = 0; i < this.text.length; i++) {
        const ch = this.text[i];
        out += (i < n || ch === " ") ? ch
          : this.chars[Math.random() * this.chars.length | 0];
      }
      if (this.hold != null && this.t > this.per * this.text.length + this.hold)
        this.done = true;
      return out;
    }
  };
  A.util = U;

  return A;
})();
