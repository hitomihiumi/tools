#!/usr/bin/env python3
"""The cameras' setup in one page: zones, floor calibration and the check, edited together and saved to barn.toml.

A small web server (in Docker: docker compose run --rm -p 8765:8765 herd studio ...) serving one page with a
still of every camera:

  zones        mask (the floor a camera counts), exclude (greyed before the detector), tracking_mask (where
               it follows cows beyond its mask) - drawn on the picture, the cows found coloured by the rule
               the barn uses (as herd.py zones page)
  calibration  two cameras side by side: the same spot of the floor clicked on both (as autocalib --pick);
               from 5 spots the floor mapping is fitted at once (autocalib's fit: lens bend from 7, spots
               that disagree left out) - a preview, nothing written yet
  check        any camera with another one's picture carried over the floor onto it, half-transparent with
               a slider (as autocalib --overlay), the other camera's mask / tracking_mask drawn where they
               fall on this one, the spots and how far apart they land; the zones and spots stay editable
               over it, the overlay follows each change

"Save to barn.toml" writes the zones, the calibrated cameras' floor_points / distortion / aspect and [mct]
(link on, link_dist, the calibrated pairs out of disjoint) into the config (barn.toml.bak kept) and the spots
into CALIB/points.json. A camera's floor can be forgotten there too (a calibration that was wrong).

The floor plan is in about cm when CALIB holds the cows' boxes autocalib cached (boxes_<cam>.json, from any
autocalib --videos run); without them its units are arbitrary and link_dist is a guess.

    python herd/herd.py studio --config barn.toml --videos rec_fixed [--calib calib] [--port 8765]
    python herd/herd.py studio --config barn.toml --image cam1=cam1.jpg --image cam2=cam2.jpg
"""

from __future__ import annotations

import argparse
import base64
import glob
import http.server
import json
import os
import re
import shutil
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import autocalib  # noqa: E402
from common import in_zone, load_config, recordings  # noqa: E402,F401

ZONES = ("mask", "exclude", "tracking_mask")
FLOOR = ("floor_points", "distortion", "aspect")
SHOW_W = 1280                       # the stills' width on the page (and the overlays')


# ------------------------------------------------------------------ barn.toml

def _key_span(block, key):
    """(start, end) of `key = value` in a [[cameras]] block - an array value to its matching bracket, with the
    rest of its line - or None."""
    m = re.search(rf"(?m)^{re.escape(key)}\s*=\s*", block)
    if not m:
        return None
    i = m.end()
    if i < len(block) and block[i] == "[":
        depth = 0
        while i < len(block):
            depth += {"[": 1, "]": -1}.get(block[i], 0)
            if depth == 0:
                break
            i += 1
    nl = block.find("\n", i)
    return m.start(), (len(block) if nl < 0 else nl + 1)


def set_keys(block, values):
    """A [[cameras]] block with each key of `values` set (a line `key = value`, where it was or at the end of
    the block) or, for None, removed."""
    for key, v in values.items():
        span = _key_span(block, key)
        line = None if v is None else f"{key} = {json.dumps(v)}\n"
        if span:
            block = block[:span[0]] + (line or "") + block[span[1]:]
        elif line:
            body = block.rstrip("\n")
            block = body + "\n" + line + "\n" * (2 if block.endswith("\n\n") else 1)
    return block


def _zone_value(polys):
    """The page's polygons -> what barn.toml holds: [] none, one polygon as [[u, v], ...], several as a list."""
    polys = [[[round(float(u), 4), round(float(v), 4)] for u, v in p] for p in polys if len(p) >= 3]
    return polys[0] if len(polys) == 1 else polys


def save_config(path, cams, floor, link_dist=None, shared=()):
    """barn.toml with the cameras' zones (cams: {id: {mask, exclude, tracking_mask, min_box_pct}}) and floors
    (floor: {id: {floor_points, distortion, aspect}, or None to forget it}); with link_dist, [mct] link on and
    the `shared` pairs out of disjoint. The old file -> barn.toml.bak."""
    text = open(path, encoding="utf-8").read()
    shutil.copy(path, path + ".bak")
    blocks = re.split(r"(?m)^(?=\[\[cameras\]\])", text)
    out = [blocks[0]]
    for b in blocks[1:]:
        m = re.search(r'(?m)^id\s*=\s*"([^"]+)"', b)
        cam = m.group(1) if m else None
        vals = {}
        if cam in cams:
            z = cams[cam]
            for k in ZONES:
                if k in z:
                    v = _zone_value(z[k])
                    vals[k] = v if v else (None if k != "mask" else [])
            if "min_box_pct" in z:
                vals["min_box_pct"] = float(z["min_box_pct"]) if z["min_box_pct"] else None
        if cam in floor:
            f = floor[cam]
            vals.update({k: (f[k] if f else None) for k in FLOOR})
        # the [[cameras]] blocks come last in barn.toml; a block ends where the next one starts
        out.append(set_keys(b, vals) if vals else b)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(out))
    if link_dist is not None:
        autocalib.apply_config(path, {}, link_dist, shared, backup=False)
    load_config(path)                                         # still a config


# ------------------------------------------------------------------ the server's state

class Studio:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.cfg = load_config(args.config)
        self.ids = [c["id"] for c in self.cfg["cameras"]]
        self.calib = args.calib or os.path.join(os.path.dirname(os.path.abspath(args.config)), "calib")
        os.makedirs(self.calib, exist_ok=True)
        self.videos = {}
        if args.videos:
            recs, _ = recordings(args.videos, self.ids, self.cfg["farm"]["timezone_offset_hours"])
            for cam, path, ts in sorted(recs, key=lambda r: r[2]):
                self.videos.setdefault(cam, []).append((path, ts))
        self.images = dict(v.split("=", 1) for v in args.image)
        self.frames, self.size, self.jpg, self.boxes, self.at = {}, {}, {}, {}, args.at
        self.live = self._detector()
        self.T = {}
        for cam in self.ids:
            f = os.path.join(self.calib, f"boxes_{cam}.json")
            if os.path.exists(f):
                try:
                    self.T[cam] = autocalib._tracks(json.load(open(f, encoding="utf-8"))["rows"])
                except (OSError, ValueError, KeyError):
                    pass
        self.load_frames(args.at)

    def _detector(self):
        if self.args.detector == "none":
            return None
        weights = self.args.detector or self.cfg["detector"]["weights"]
        if not (weights and os.path.isdir(weights)):
            print(f"[studio] [detector] weights = {weights!r}: none there - the page shows no cows", flush=True)
            return None
        try:
            import detector as det_mod
            return det_mod.Live(weights, threshold=self.cfg["detector"]["threshold"] or None,
                                tiles=self.cfg["detector"].get("tiles"))
        except Exception as e:                                # noqa: BLE001 - the page works without cows
            print(f"[studio] [detector] not loaded ({e}) - the page shows no cows", flush=True)
            return None

    def load_frames(self, at):
        """Every camera's picture: the recordings' moment `at` s in (all cameras at one time), else --image,
        else the one its spots were clicked on (CALIB/pick)."""
        import cv2
        self.at = at
        starts = [v[0][1] for v in self.videos.values() if v]
        moment = (max(starts) + at) if starts else None
        for cam in self.ids:
            img = None
            if cam in self.images:
                img = cv2.imread(self.images[cam])
            elif self.videos.get(cam):
                img = autocalib.frame_at(self.videos, cam, moment)
            if img is None:
                files = sorted(glob.glob(os.path.join(self.calib, "pick", f"{cam}_*.jpg")), key=os.path.getmtime)
                img = cv2.imread(files[-1]) if files else None
            if img is None:
                self.frames.pop(cam, None)
                continue
            h, w = img.shape[:2]
            self.size[cam] = (w, h)
            if w > SHOW_W:
                img = cv2.resize(img, (SHOW_W, round(h * SHOW_W / w)), interpolation=cv2.INTER_AREA)
            self.frames[cam] = img
            self.jpg[cam] = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
            self.boxes[cam] = []
            if self.live is not None:
                from PIL import Image
                found = self.live(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                self.boxes[cam] = [[round(float(v), 4) for v in d["bbox"]] for d in found]
        return moment

    def aspect(self, cam):
        w, h = self.size.get(cam, (16, 9))
        return w / h

    def state(self):
        self.cfg = load_config(self.args.config)
        points = {}
        pj = os.path.join(self.calib, "points.json")
        if os.path.exists(pj):
            try:
                points = json.load(open(pj, encoding="utf-8")).get("pairs", {})
            except (OSError, ValueError):
                points = {}

        def polys(m):
            if not m:
                return []
            return [m] if isinstance(m[0][0], (int, float)) else m
        cams = []
        for c in self.cfg["cameras"]:
            w, h = self.frames[c["id"]].shape[1::-1] if c["id"] in self.frames else (16, 9)
            cams.append({"id": c["id"], "w": w, "h": h, "still": c["id"] in self.frames,
                         "boxes": self.boxes.get(c["id"], []), "min_box_pct": c.get("min_box_pct") or 0,
                         **{k: polys(c.get(k)) for k in ZONES},
                         "calib": {k: c.get(k) for k in FLOOR} if c.get("floor_points") else None})
        m = self.cfg["mct"]
        return {"cams": cams, "points": points, "at": self.at, "videos": bool(self.videos),
                "boxes_cached": sorted(self.T), "link_dist": m.get("link_dist"),
                "neighbours": m.get("neighbours", []), "disjoint": m.get("disjoint", []),
                "config": os.path.abspath(self.args.config), "calib_dir": os.path.abspath(self.calib)}

    # -------------------------------------------------------------- calibration

    def fit(self, req):
        """Fit the pair req["pair"] and every pair with spots connected to it (one floor plan for them all):
        -> each pair's result, the cameras' new floors (a preview) and link_dist."""
        points = {k: np.array(v, float).reshape(-1, 4) for k, v in req["points"].items() if v}
        target = req["pair"]
        graph = {}
        for k in points:
            a, b = k.split(":")
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)
        a0 = target.split(":")[0]
        group, queue = {a0}, [a0]
        while queue:
            for y in graph.get(queue.pop(), ()):
                if y not in group:
                    group.add(y)
                    queue.append(y)
        aspect = {c: self.aspect(c) for c in self.ids}
        results, found, notes = {}, {}, []
        for k, P in points.items():
            a, b = k.split(":")
            if a not in group or b not in group:
                continue
            r, said = autocalib.fit_spots(a, b, P, self.size, aspect, self.T)
            notes += said
            results[k] = {"ok": bool(r and r.get("ok")), "notes": said,
                          "errors": (r or {}).get("spot_error_px", []), "left_out": (r or {}).get("spots_left_out", []),
                          "median": (r or {}).get("median_px"), "lens": [(r or {}).get("kA"), (r or {}).get("kB")]}
            if r and r.get("ok"):
                found[(a, b)] = r
        out = {"pairs": results, "notes": notes, "floor": {}, "link_dist": None}
        if not results.get(target, {}).get("ok"):
            return out
        # the other calibrated cameras keep their place on the plan: this group goes clear of them
        calibrated = {c: v for c, v in (req.get("floor") or {}).items() if v and c not in group}
        taken = [(min(p[2] for p in v["floor_points"]), max(p[2] for p in v["floor_points"]))
                 for v in calibrated.values()]
        calib, link, _ = autocalib.floor_plan(found, self.T, aspect, taken)
        missing = sorted(c for c in calib if c not in self.T)
        if missing:
            out["notes"].append(f"no boxes cached for {', '.join(missing)} in {self.calib} (boxes_<cam>.json from an "
                                "autocalib --videos run): the plan is not in cm and link_dist is a guess")
        out.update(floor=calib, link_dist=round(link))
        return out

    def overlay(self, req):
        """req: base, top, floor {cam: floor}, top_zones {mask, tracking_mask}, spots [[ub, vb, ut, vt]] ->
        the top camera's picture carried over the floor onto the base one (PNG, transparent off the shared
        floor), the top's zones and spots where they fall on the base picture."""
        import cv2
        base, top = req["base"], req["top"]
        if base not in self.frames or top not in self.frames:
            return {"error": "no picture"}
        cams = {c: {**(req["floor"].get(c) or {})} for c in (base, top)}
        fa = autocalib.camera_floor(cams[base], self.aspect(base)) if cams[base] else None
        fb = autocalib.camera_floor(cams[top], self.aspect(top)) if cams[top] else None
        if fa is None or fb is None:
            return {"error": "not calibrated: " + ", ".join(c for c, f in ((base, fa), (top, fb)) if f is None)}
        img_a, img_b = self.frames[base], self.frames[top]
        warped, valid = autocalib.other_onto(img_a, img_b, fa, fb)
        png = cv2.imencode(".png", np.dstack([warped, valid.astype(np.uint8) * 255]))[1].tobytes()
        zones = {}
        for k in ("mask", "tracking_mask"):
            zones[k] = []
            for poly in req.get("top_zones", {}).get(k) or []:
                if len(poly) >= 3:
                    pts = autocalib._poly_onto([tuple(p) for p in poly], fa, fb, 1.0, 1.0)
                    zones[k].append([p if p and abs(p[0]) < 4 and abs(p[1]) < 4 else None for p in pts])
        spots = []
        wa = self.size.get(base, (img_a.shape[1], 0))[0]
        S = np.array(req.get("spots") or [], float).reshape(-1, 4)
        if len(S):
            X, Y, ok_b = fb[0](S[:, 2], S[:, 3])
            u, v, ok_a = fa[1](X, Y)
            for (ub, vb, _, _), x, y, ok in zip(S, u, v, ok_a & ok_b):
                good = bool(ok and np.isfinite(x) and np.isfinite(y) and abs(x) < 4 and abs(y) < 4)
                spots.append([float(x), float(y), float(np.hypot((x - ub) * wa, (y - vb) * wa / self.aspect(base)))]
                             if good else None)
        return {"png": base64.b64encode(png).decode(), "share": float(valid.mean()), "zones": zones, "spots": spots}

    def save(self, req):
        floor = {c: v for c, v in (req.get("floor") or {}).items()}
        link = req.get("link_dist")
        shared = [tuple(k.split(":")) for k in req.get("calibrated_pairs") or []]
        save_config(self.args.config, req.get("cams") or {}, floor, link, shared)
        with open(os.path.join(self.calib, "points.json"), "w", encoding="utf-8") as fh:
            json.dump({"pairs": {k: v for k, v in (req.get("points") or {}).items() if v}}, fh, indent=1)
        said = [f"{self.args.config}: zones of " + (", ".join(req.get("cams") or {}) or "no camera changed")]
        if floor:
            said.append("floor " + ", ".join(f"{c} {'calibrated' if v else 'forgotten'}" for c, v in floor.items()))
        if link is not None:
            said.append(f"[mct] link on, link_dist {link}")
        said.append(f"the old file: {self.args.config}.bak; spots: {os.path.join(self.calib, 'points.json')}")
        print("[studio] saved - " + "; ".join(said), flush=True)
        return {"said": said, "state": self.state()}


def serve(studio, host, port):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, kind="application/json", code=200):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                return self._send(PAGE, "text/html; charset=utf-8")
            if path == "/api/state":
                with studio.lock:
                    return self._send(studio.state())
            m = re.fullmatch(r"/still/([^/]+)\.jpg", path)
            if m and m.group(1) in studio.jpg:
                return self._send(studio.jpg[m.group(1)], "image/jpeg")
            self._send({"error": "not found"}, code=404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
                with studio.lock:
                    if self.path == "/api/fit":
                        return self._send(studio.fit(req))
                    if self.path == "/api/overlay":
                        return self._send(studio.overlay(req))
                    if self.path == "/api/save":
                        return self._send(studio.save(req))
                    if self.path == "/api/moment":
                        studio.load_frames(float(req.get("at", studio.at)))
                        return self._send(studio.state())
                self._send({"error": "not found"}, code=404)
            except SystemExit as e:                           # load_config refuses a broken barn.toml
                self._send({"error": str(e)}, code=400)
            except Exception as e:                            # noqa: BLE001 - said on the page, the server lives on
                import traceback
                traceback.print_exc()
                self._send({"error": f"{type(e).__name__}: {e}"}, code=500)

    srv = http.server.ThreadingHTTPServer((host, port), Handler)
    print(f"[studio] open http://localhost:{port} in a browser (Ctrl+C stops it)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Camera studio</title>
<style>
:root{--bg:#f7f7f4;--card:#fff;--fg:#1d1d1b;--mute:#6b6b66;--line:#dddcd5;--accent:#2f6f4f;--warn:#b45309;--bad:#b91c1c}
@media (prefers-color-scheme: dark){:root{--bg:#161715;--card:#1f211e;--fg:#ecebe6;--mute:#9a9a93;--line:#34362f;--accent:#7fc29c;--warn:#f59e0b;--bad:#f87171}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 16px}
h1{font-size:17px;margin:0 12px 0 0;display:inline}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:5px 0}
button,select,input{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:3px 9px}
button{cursor:pointer}button.on{background:var(--accent);color:var(--bg);border-color:var(--accent)}
button.main{background:var(--accent);color:var(--bg);border-color:var(--accent);font-weight:600}
.mute{color:var(--mute)}.warn{color:var(--warn)}.bad{color:var(--bad)}.good{color:var(--accent)}
details{color:var(--mute);font-size:13px;max-width:1200px}details b{color:var(--fg)}
main{padding:8px 16px 40px}
.panes{display:flex;gap:10px}.pane{flex:1 1 0;min-width:0}
.view{overflow:auto;max-height:76vh;border:1px solid var(--line);border-radius:6px;background:#000}
canvas{display:block;touch-action:none;cursor:crosshair}
.panel{border:1px solid var(--line);border-radius:6px;background:var(--card);padding:8px 10px;margin:8px 0}
.panel h3{font-size:14px;margin:0 0 4px}
.notes div{font-size:13px}
@media (max-width:800px){.panes{flex-direction:column}}
</style></head><body>
<header>
<div class="bar"><h1>Camera studio</h1>
<button class="main" id="save">Save to barn.toml</button><button id="undo">Undo all (as in barn.toml)</button>
<span id="status" class="mute"></span></div>
<div class="bar"><span class="mute">view:</span><button data-n="1">one camera</button><button data-n="2" class="on">two cameras</button>
<span class="mute">&nbsp;draw:</span><button data-l="mask" class="on">mask</button><button data-l="exclude">exclude</button>
<button data-l="tracking_mask">tracking</button><button data-l="spots">spots</button>
<label><input type="checkbox" id="cows" checked> cows</label>
<span id="moment"></span></div>
<details><summary>How to use</summary>
<b>Zones</b> (layers mask / exclude / tracking): click round an area - a click adds a corner, a click on the first corner or
<b>Enter</b> closes it, <b>Esc</b> drops it; drag a corner to move it; click inside a shape, then <b>Del</b> removes it (a click
inside a shape selects it: to start a new shape inside another one, click there twice).
mask: the floor this camera counts (cameras over one floor: every patch in one mask only); exclude: greyed before the
detector; tracking: where it follows a cow beyond its mask, onto a neighbour's floor.
<b>Spots</b> (two cameras): click a spot of the floor on one picture, then the same spot on the other - the foot of a
post, a trough's corner where it meets the floor, a drain; not a cow's back or a top rail. Spread them over the shared
floor, not along one fence: 5 at least, 7-10 better (the lens bend is fitted too). From 5 the floor is fitted at once:
green spots agree, red are off, grey left out. Drag a spot to move it; click it, then Del removes it.
<b>Check</b>: "over it" lays the other camera's picture over this one through the floor (slider: how much);
where it is right the floor's lines and the feet of posts stay put as the slider moves - cows' backs and walls never
do. The other camera's mask (yellow) and tracking zone (blue, dashed) are drawn where they fall here; circles are the
spots clicked here, crosses where the other camera's clicks land. Nothing is written until <b>Save</b>.
</details>
</header>
<main>
<div class="panes" id="panes"></div>
<div class="panel" id="pairpanel"></div>
<div class="panel"><h3>Floors</h3><div id="floors"></div></div>
<div class="panel notes"><h3>Said</h3><div id="notes" class="mute">-</div></div>
</main>
<script>
const $ = id => document.getElementById(id);
const COL = {mask: "#3ee07a", exclude: "#ff4040", tracking_mask: "#4aa3ff"};
let S = null, layer = "mask", nPanes = 2, cur = null, sel = null, drag = null, pending = null;
let panes = [{cam: null, over: "", alpha: 0.5, zoom: 1, ov: null}, {cam: null, over: "", alpha: 0.5, zoom: 1, ov: null}];
let fits = {}, floorDirty = {}, linkDist = null, calibratedPairs = new Set(), imgs = {}, notes = [];

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const j = await r.json(); if (j.error) throw new Error(j.error); return j;
}
const cam = id => S.cams.find(c => c.id === id);
function say(lines, cls) { notes = (Array.isArray(lines) ? lines : [lines]).map(t => ({t, cls})); $("notes").innerHTML =
  notes.map(n => `<div class="${n.cls || ""}">${n.t.replace(/&/g, "&amp;").replace(/</g, "&lt;")}</div>`).join("") || "-"; }
function status(t, cls) { $("status").textContent = t; $("status").className = cls || "mute"; }

async function load(state) {
  S = state || await api("/api/state");
  fits = {}; floorDirty = {}; linkDist = null; calibratedPairs = new Set(); cur = null; sel = null; pending = null;
  zonesChanged = new Set();
  imgs = {};
  const t = Date.now();
  S.cams.forEach(c => { if (c.still) { const im = new Image(); im.onload = drawAll; im.src = `/still/${c.id}.jpg?${t}`; imgs[c.id] = im; } });
  const withPic = S.cams.filter(c => c.still).map(c => c.id);
  if (!panes[0].cam || !cam(panes[0].cam)) panes[0].cam = withPic[0] || S.cams[0].id;
  if (!panes[1].cam || !cam(panes[1].cam)) panes[1].cam = withPic.find(x => x !== panes[0].cam) || panes[0].cam;
  $("moment").innerHTML = S.videos ? `<span class="mute">&nbsp;moment</span> <input id="at" type="number" step="10" value="${S.at}" style="width:6em"> s <button id="go">show</button>` : "";
  if (S.videos) $("go").onclick = async () => { status("loading pictures..."); try { await load(await api("/api/moment", {at: +$("at").value})); status("pictures of " + $("at").value + " s"); } catch (e) { status(e.message, "bad"); } };
  build(); floors(); refreshOverlays(); pairPanel();
  if (!S.boxes_cached.length) say(`No boxes_<cam>.json in ${S.calib_dir}: a fit gives a floor plan not in cm and a guessed link_dist (run autocalib --videos once to cache the cows' boxes).`, "warn");
}

// ---------------------------------------------------------------- panes
function build() {
  const root = $("panes"); root.innerHTML = "";
  for (let i = 0; i < nPanes; i++) {
    const p = panes[i], d = document.createElement("div"); d.className = "pane";
    const opts = (sel_, none) => (none ? `<option value="">${none}</option>` : "") + S.cams.map(c => `<option value="${c.id}" ${c.id === sel_ ? "selected" : ""}>${c.id}${c.still ? "" : " (no picture)"}</option>`).join("");
    d.innerHTML = `<div class="bar"><select class="c">${opts(p.cam)}</select>
      <span class="mute">over it:</span><select class="o">${opts(p.over, "nothing")}</select>
      <input class="a" type="range" min="0" max="100" value="${Math.round(p.alpha * 100)}" title="the other camera's picture">
      <button class="zi">+</button><button class="zo">&minus;</button><span class="mute s"></span></div>
      <div class="view"><canvas></canvas></div>`;
    d.querySelector(".c").onchange = e => { p.cam = e.target.value; cur = null; sel = null; pending = null; p.ov = null; build(); refreshOverlays(); pairPanel(); };
    d.querySelector(".o").onchange = e => { p.over = e.target.value; p.ov = null; refreshOverlays(); pairPanel(); };
    d.querySelector(".a").oninput = e => { p.alpha = e.target.value / 100; draw(i); };
    d.querySelector(".zi").onclick = () => { p.zoom = Math.min(6, p.zoom * 1.5); size(i); };
    d.querySelector(".zo").onclick = () => { p.zoom = Math.max(1, p.zoom / 1.5); size(i); };
    root.appendChild(d); p.el = d; p.cv = d.querySelector("canvas"); p.ctx = p.cv.getContext("2d");
    hook(i);
  }
  requestAnimationFrame(() => { for (let i = 0; i < nPanes; i++) size(i); });
}
function size(i) {
  const p = panes[i], c = cam(p.cam), W = Math.round((p.el.querySelector(".view").clientWidth - 2) * p.zoom);
  p.cv.width = W; p.cv.height = Math.round(W * c.h / c.w); draw(i);
}
function drawAll() { for (let i = 0; i < nPanes; i++) if (panes[i].cv) draw(i); }

function pairKey(x, y) {             // [points key, x is its B side]
  if (S.points[x + ":" + y]) return [x + ":" + y, false];
  if (S.points[y + ":" + x]) return [y + ":" + x, true];
  const ix = S.cams.findIndex(c => c.id === x), iy = S.cams.findIndex(c => c.id === y);
  return ix <= iy ? [x + ":" + y, false] : [y + ":" + x, true];
}
function partner(i) {                // the camera whose spots this pane shows: the other pane's, else what lies over it
  if (layer === "spots" && nPanes === 2 && panes[1 - i].cam !== panes[i].cam) return panes[1 - i].cam;
  return panes[i].over || (nPanes === 2 && panes[1 - i].cam !== panes[i].cam ? panes[1 - i].cam : "");
}
function spotsOf(i) {                // [{u, v, k}] of this pane's camera in its pair, k the index in the pair's list
  const y = partner(i); if (!y) return [];
  const [key, flip] = pairKey(panes[i].cam, y), P = S.points[key] || [];
  return P.map((p, k) => ({u: flip ? p[2] : p[0], v: flip ? p[3] : p[1], k, key}));
}
function inPoly(x, y, p) { let ins = false; for (let i = 0, j = p.length - 1; i < p.length; j = i++) {
  const [x1, y1] = p[i], [x2, y2] = p[j]; if ((y1 > y) !== (y2 > y) && x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1) ins = !ins; } return ins; }
function verdict(c, b) {
  const cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2;
  if (c.exclude.some(p => inPoly(cx, cy, p))) return "excluded";
  if ((b[2] - b[0]) * (b[3] - b[1]) < (c.min_box_pct || 0) / 100) return "small";
  if (c.mask.length && !c.mask.some(p => inPoly(cx, cy, p))) return "other";
  return "kept";
}

function draw(i) {
  const p = panes[i], c = cam(p.cam), ctx = p.ctx, W = p.cv.width, H = p.cv.height, X = v => v * W, Y = v => v * H;
  ctx.fillStyle = "#222"; ctx.fillRect(0, 0, W, H);
  const im = imgs[c.id];
  if (im && im.complete && im.naturalWidth) ctx.drawImage(im, 0, 0, W, H);
  else { ctx.fillStyle = "#aaa"; ctx.font = "16px sans-serif"; ctx.fillText("no picture of " + c.id + ": --videos or --image", 20, 30); }
  const ov = p.ov;
  if (p.over && ov && ov.img && ov.img.complete) { ctx.globalAlpha = p.alpha; ctx.drawImage(ov.img, 0, 0, W, H); ctx.globalAlpha = 1; }
  const path = q => { ctx.beginPath(); q.forEach(([x, y], k) => k ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))); ctx.closePath(); };
  c.exclude.forEach(q => { path(q); ctx.fillStyle = "rgba(114,114,114,0.6)"; ctx.fill(); });
  ["mask", "tracking_mask", "exclude"].forEach(l => c[l].forEach((q, k) => {
    path(q); ctx.lineWidth = sel && sel.pane === i && sel.l === l && sel.i === k ? 4 : 2; ctx.strokeStyle = COL[l];
    ctx.setLineDash(l === "tracking_mask" ? [8, 5] : []); ctx.stroke(); ctx.setLineDash([]);
    if (l === layer) q.forEach(([x, y]) => { ctx.fillStyle = COL[l]; ctx.fillRect(X(x) - 4, Y(y) - 4, 8, 8); });
  }));
  if (p.over && ov && ov.zones) {     // the other camera's zones where they fall on this picture
    [["mask", "#ffd400", []], ["tracking_mask", "#4aa3ff", [3, 4]]].forEach(([k, col, dash]) => (ov.zones[k] || []).forEach(q => {
      ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.setLineDash(dash); ctx.beginPath(); let on = false;
      q.forEach(pt => { if (!pt) { on = false; return; } if (on) ctx.lineTo(X(pt[0]), Y(pt[1])); else { ctx.moveTo(X(pt[0]), Y(pt[1])); on = true; } });
      if (q[0] && q[q.length - 1]) ctx.lineTo(X(q[0][0]), Y(q[0][1]));
      ctx.stroke(); ctx.setLineDash([]); }));
  }
  let n = {kept: 0, excluded: 0, small: 0, other: 0};
  if ($("cows").checked) (c.boxes || []).forEach(b => {
    const v = verdict(c, b); n[v]++;
    ctx.lineWidth = 1.5; ctx.setLineDash(v === "other" ? [6, 4] : []);
    ctx.strokeStyle = {kept: "#3ee07a", excluded: "#ff4040", small: "#ffa020", other: "#ffffff"}[v];
    ctx.strokeRect(X(b[0]), Y(b[1]), X(b[2] - b[0]), Y(b[3] - b[1])); ctx.setLineDash([]);
  });
  // spots: numbered, coloured by the fit (green agree, red off, grey left out); crosses where the other camera's land
  const sp = spotsOf(i), y = partner(i), f = y ? fits[pairKey(p.cam, y)[0]] : null;
  sp.forEach(s => {
    let col = "#ffffff";
    if (f && f.errors && f.errors.length > s.k) {
      const e = f.errors[s.k], med = f.median || 0;
      col = f.left_out.includes(s.k + 1) ? "#9ca3af" : (e > Math.max(3 * med, 25) ? "#ef4444" : "#22c55e");
    }
    const selected = sel && sel.spot && sel.key === s.key && sel.k === s.k;
    ctx.strokeStyle = col; ctx.lineWidth = selected ? 4 : 2.5; ctx.beginPath(); ctx.arc(X(s.u), Y(s.v), 8, 0, 7); ctx.stroke();
    ctx.font = "bold 14px sans-serif"; ctx.lineWidth = 3; ctx.strokeStyle = "#000"; ctx.strokeText(s.k + 1, X(s.u) + 10, Y(s.v) - 8);
    ctx.fillStyle = col; ctx.fillText(s.k + 1, X(s.u) + 10, Y(s.v) - 8);
    const pr = p.over && p.over === y && ov && ov.spots ? ov.spots[s.k] : null;
    if (pr) {
      ctx.strokeStyle = "#ff3b3b"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(X(s.u), Y(s.v)); ctx.lineTo(X(pr[0]), Y(pr[1])); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(X(pr[0]) - 7, Y(pr[1])); ctx.lineTo(X(pr[0]) + 7, Y(pr[1])); ctx.moveTo(X(pr[0]), Y(pr[1]) - 7); ctx.lineTo(X(pr[0]), Y(pr[1]) + 7); ctx.stroke();
    }
  });
  if (pending && pending.pane === i) { ctx.setLineDash([4, 3]); ctx.strokeStyle = "#fff"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(pending.u), Y(pending.v), 9, 0, 7); ctx.stroke(); ctx.setLineDash([]); }
  if (cur && cur.pane === i) {
    ctx.strokeStyle = COL[layer]; ctx.lineWidth = 2; ctx.beginPath();
    cur.pts.forEach(([x, y_], k) => k ? ctx.lineTo(X(x), Y(y_)) : ctx.moveTo(X(x), Y(y_))); ctx.stroke();
    cur.pts.forEach(([x, y_], k) => { ctx.fillStyle = k ? COL[layer] : "#fff"; ctx.fillRect(X(x) - 4, Y(y_) - 4, 8, 8); });
  }
  const info = [];
  if ($("cows").checked && (c.boxes || []).length) info.push(`cows ${c.boxes.length}: kept ${n.kept}, excluded ${n.excluded}, small ${n.small}, another camera's ${n.other}`);
  if (p.over) info.push(ov ? (ov.error ? ov.error : `floor shared on ${Math.round(ov.share * 100)}%`) : "...");
  p.el.querySelector(".s").textContent = info.join(" | ");
}

// ---------------------------------------------------------------- editing
function pt(i, e) { const r = panes[i].cv.getBoundingClientRect();
  return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))]; }
function near(i, a, b) { const r = panes[i].cv.getBoundingClientRect(); return Math.abs(a[0] - b[0]) * r.width < 10 && Math.abs(a[1] - b[1]) * r.height < 10; }
function hook(i) {
  const cv = panes[i].cv;
  cv.addEventListener("pointerdown", e => {
    const p = pt(i, e), c = cam(panes[i].cam);
    const hit = spotsOf(i).find(s => near(i, [s.u, s.v], p));
    if (hit && (layer === "spots" || !cur)) {          // spots can be grabbed on any layer
      sel = {spot: true, key: hit.key, k: hit.k}; drag = {spot: hit, pane: i}; cv.setPointerCapture(e.pointerId); drawAll(); return;
    }
    if (layer === "spots") {
      if (nPanes < 2 || panes[0].cam === panes[1].cam) { say("Spots: two cameras side by side (view: two cameras), a spot clicked on each.", "warn"); return; }
      if (pending && pending.pane !== i) {
        const a = pending.pane === 0 ? [pending.u, pending.v] : p, b = pending.pane === 0 ? p : [pending.u, pending.v];
        const [key, flip] = pairKey(panes[0].cam, panes[1].cam);
        const row = flip ? [b[0], b[1], a[0], a[1]] : [a[0], a[1], b[0], b[1]];
        (S.points[key] = S.points[key] || []).push(row.map(v => +v.toFixed(5)));
        pending = null; changedSpots(key);
      } else { pending = {pane: i, u: p[0], v: p[1]}; drawAll(); }
      return;
    }
    if (cur && cur.pane === i) {
      if (cur.pts.length >= 3 && near(i, cur.pts[0], p)) { c[layer].push(cur.pts); cur = null; changedZones(c.id); return; }
      cur.pts.push(p); draw(i); return;
    }
    for (let k = 0; k < c[layer].length; k++) {
      const j = c[layer][k].findIndex(q => near(i, q, p));
      if (j >= 0) { sel = {pane: i, l: layer, i: k}; drag = {pane: i, k, j}; cv.setPointerCapture(e.pointerId); draw(i); return; }
    }
    const inside = c[layer].findIndex(q => inPoly(p[0], p[1], q));
    if (inside >= 0 && !(sel && sel.pane === i && sel.l === layer && sel.i === inside)) { sel = {pane: i, l: layer, i: inside}; draw(i); return; }
    sel = null; cur = {pane: i, pts: [p]}; draw(i);
  });
  cv.addEventListener("pointermove", e => {
    if (!drag || drag.pane !== i) return;
    const p = pt(i, e).map(v => +v.toFixed(5));
    if (drag.spot) {
      const row = S.points[drag.spot.key][drag.spot.k], flip = pairKey(panes[i].cam, partner(i))[1];
      if (flip) { row[2] = p[0]; row[3] = p[1]; } else { row[0] = p[0]; row[1] = p[1]; }
      drawAll();
    } else { cam(panes[i].cam)[layer][drag.k][drag.j] = p; draw(i); }
  });
  cv.addEventListener("pointerup", () => {
    if (!drag || drag.pane !== i) return;
    const d = drag; drag = null;
    if (d.spot) changedSpots(d.spot.key); else changedZones(panes[i].cam);
  });
}
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "Enter" && cur && cur.pts.length >= 3) { const c = cam(panes[cur.pane].cam); c[layer].push(cur.pts); cur = null; changedZones(c.id); }
  else if (e.key === "Escape") { cur = null; sel = null; pending = null; drawAll(); }
  else if ((e.key === "Delete" || e.key === "Backspace") && sel) {
    e.preventDefault();
    if (sel.spot) { S.points[sel.key].splice(sel.k, 1); const k = sel.key; sel = null; changedSpots(k); }
    else { const c = cam(panes[sel.pane].cam); c[sel.l].splice(sel.i, 1); sel = null; changedZones(c.id); }
  }
});
document.querySelectorAll("[data-l]").forEach(b => b.onclick = () => { layer = b.dataset.l; cur = null; sel = null; pending = null;
  document.querySelectorAll("[data-l]").forEach(x => x.classList.toggle("on", x === b)); drawAll(); pairPanel(); });
document.querySelectorAll("[data-n]").forEach(b => b.onclick = () => { nPanes = +b.dataset.n;
  document.querySelectorAll("[data-n]").forEach(x => x.classList.toggle("on", x === b)); cur = null; pending = null; build(); refreshOverlays(); pairPanel(); });
$("cows").onchange = drawAll;

let zoneTimer = null, fitTimer = null, ovTimer = null, zonesChanged = new Set();
function changedZones(id) { zonesChanged.add(id); drawAll(); status("changed - not saved yet", "warn"); refreshOverlays(); }
function changedSpots(key) {
  drawAll(); pairPanel(); status("changed - not saved yet", "warn");
  clearTimeout(fitTimer);
  if ((S.points[key] || []).length >= 5) fitTimer = setTimeout(() => fit(key), 350);
  else { delete fits[key]; refreshOverlays(); }
}

// ---------------------------------------------------------------- calibration and the overlay
function currentFloor() { const f = {}; S.cams.forEach(c => f[c.id] = c.calib); return f; }
async function fit(key) {
  try {
    const r = await api("/api/fit", {pair: key, points: S.points, floor: currentFloor()});
    Object.assign(fits, r.pairs);
    if (r.pairs[key] && r.pairs[key].ok) {
      for (const id in r.floor) { cam(id).calib = r.floor[id]; floorDirty[id] = true; }
      for (const k in r.pairs) if (r.pairs[k].ok) calibratedPairs.add(k);
      linkDist = Math.max(linkDist || 0, r.link_dist);
      say(r.notes.concat(["floor fitted (a preview): " + Object.keys(r.floor).join(", ") + ` - link_dist ${r.link_dist}; Save writes it`]));
    } else say(r.notes.length ? r.notes : ["not fitted"], "warn");
  } catch (e) { say(e.message, "bad"); }
  floors(); pairPanel(); drawAll(); refreshOverlays();
}
function refreshOverlays() {
  clearTimeout(ovTimer);
  ovTimer = setTimeout(async () => {
    for (let i = 0; i < nPanes; i++) {
      const p = panes[i];
      if (!p.over || p.over === p.cam) { p.ov = null; draw(i); continue; }
      const top = cam(p.over), [key, flip] = pairKey(p.cam, p.over);
      const spots = (S.points[key] || []).map(r => flip ? [r[2], r[3], r[0], r[1]] : r);
      try {
        const r = await api("/api/overlay", {base: p.cam, top: p.over, floor: currentFloor(),
          top_zones: {mask: top.mask, tracking_mask: top.tracking_mask}, spots});
        const im = new Image(); im.onload = () => draw(i); im.src = "data:image/png;base64," + r.png; r.img = im; p.ov = r;
      } catch (e) { p.ov = {error: e.message}; }
      draw(i);
    }
  }, 250);
}
function pairPanel() {
  const a = panes[0].cam, b = nPanes === 2 ? panes[1].cam : panes[0].over;
  if (!b || a === b) { $("pairpanel").innerHTML = `<h3>Spots</h3><div class="mute">two different cameras side by side (or one with another over it) to click or see their spots</div>`; return; }
  const [key] = pairKey(a, b), P = S.points[key] || [], f = fits[key];
  const st = !P.length ? "no spots yet" : P.length < 5 ? `${P.length} spots - 5 at least to fit` :
    f ? (f.ok ? `fitted: ${P.length - f.left_out.length} of ${P.length} agree within ${Math.round(f.median)} px` + (f.left_out.length ? `; left out: ${f.left_out.join(", ")}` : "") : "NOT fitted - see Said") : `${P.length} spots`;
  $("pairpanel").innerHTML = `<h3>Spots ${key.replace(":", " &harr; ")}</h3><div class="bar"><span class="${f && !f.ok ? "bad" : ""}">${st}</span>
    ${P.length >= 5 ? `<button id="refit">Fit now</button>` : ""}${P.length ? `<button id="clearpts">Clear the spots</button>` : ""}
    <span class="mute">${layer === "spots" ? (pending ? "now the same spot on the other picture" : "click a spot of the floor on either picture") : "layer spots to add some"}</span></div>`;
  if ($("refit")) $("refit").onclick = () => fit(key);
  if ($("clearpts")) $("clearpts").onclick = () => { if (confirm("Remove all spots of " + key + "?")) { S.points[key] = []; delete fits[key]; changedSpots(key); } };
}
function floors() {
  $("floors").innerHTML = S.cams.map(c => {
    const st = !c.calib ? `<span class="mute">not calibrated</span>` : floorDirty[c.id] ? `<span class="warn">fitted now - not saved</span>` : `<span class="good">calibrated (barn.toml)</span>`;
    const lens = c.calib ? ` <span class="mute">lens ${c.calib.distortion || 0}</span>` : "";
    return `<div class="bar"><b>${c.id}</b> ${st}${lens} ${c.calib ? `<button data-forget="${c.id}">forget this floor</button>` : ""}</div>`;
  }).join("") + (linkDist ? `<div class="mute">[mct] link_dist on Save: ${linkDist} (now ${S.link_dist ?? "-"})</div>` : "");
  document.querySelectorAll("[data-forget]").forEach(b => b.onclick = () => {
    const id = b.dataset.forget;
    if (!confirm(`Forget ${id}'s floor calibration? (written on Save)`)) return;
    cam(id).calib = null; floorDirty[id] = true;
    [...calibratedPairs].forEach(k => { if (k.split(":").includes(id)) calibratedPairs.delete(k); });
    status("changed - not saved yet", "warn"); floors(); refreshOverlays();
  });
}

$("save").onclick = async () => {
  const cams = {}, floor = {};
  S.cams.filter(c => zonesChanged.has(c.id)).forEach(c => { cams[c.id] = {mask: c.mask, exclude: c.exclude, tracking_mask: c.tracking_mask}; });
  for (const id in floorDirty) floor[id] = cam(id).calib;
  const pairs = [...calibratedPairs].filter(k => k.split(":").every(id => cam(id).calib));
  try {
    status("saving...");
    const r = await api("/api/save", {cams, floor, points: S.points, link_dist: pairs.length ? linkDist : null, calibrated_pairs: pairs});
    const keep = panes.map(p => ({...p}));
    await load(r.state); panes.forEach((p, i) => { p.over = keep[i].over; });
    build(); refreshOverlays(); say(r.said, "good"); status("saved to barn.toml", "good");
  } catch (e) { status("not saved: " + e.message, "bad"); }
};
$("undo").onclick = async () => { if (confirm("Drop every change since the last save?")) { await load(); status("as in barn.toml"); } };
window.addEventListener("beforeunload", e => { if ($("status").textContent.startsWith("changed")) { e.preventDefault(); e.returnValue = ""; } });
window.addEventListener("resize", () => { for (let i = 0; i < nPanes; i++) size(i); });
load().catch(e => status(e.message, "bad"));
</script></body></html>
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--videos", default=None, help="the recordings: every camera's picture at one moment (--at; the "
                   "page can move it)")
    p.add_argument("--image", action="append", default=[], help="cam1=picture.jpg (instead of, or besides, --videos)")
    p.add_argument("--calib", default=None, help="autocalib's folder (default: calib beside the config): its "
                   "points.json, pick pictures and cached boxes; the spots are saved there")
    p.add_argument("--at", type=float, default=60.0, help="the pictures this many s into the recordings")
    p.add_argument("--detector", default=None, help="a best/ folder (default: the config's); none: no cows drawn")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args(argv)
    studio = Studio(args)
    shown = sorted(studio.frames)
    print(f"[studio] {len(studio.ids)} cameras, pictures of {', '.join(shown) or 'none'}; spots and boxes from "
          f"{studio.calib}", flush=True)
    serve(studio, args.host, args.port)


if __name__ == "__main__":
    sys.exit(main())
