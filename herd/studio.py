#!/usr/bin/env python3
"""The cameras' setup in one page: zones, the floor plane and its check, edited together and saved to barn.toml.

A small web server (in Docker: docker compose run --rm -p 8765:8765 herd studio ...) serving one page
(studio.html) with a still of every camera, one, two or four at a time, or the plan:

  zones   mask (the floor a camera counts), exclude (greyed before the detector), tracking_mask (where it
          follows cows beyond its mask) - drawn on the picture, the cows found coloured by the rule the barn
          uses (as herd.py zones page)
  spots   the same spot of the floor, clicked on every camera that sees it (one spot, any number of
          cameras; picked on one, it is shown - or where it should be - on all). Every two cameras with 5
          spots in common are fitted at once (autocalib's fit: the lens bend from 7, spots that disagree
          left out) into one floor plan; each spot's error on each camera is shown
  plane   each calibrated camera's floor drawn on its picture as a grid (about 1 m when the plan is in
          cm) and moved as a whole: in the picture as the camera would turn (tilt, pan, roll), zoomed, its
          lens bend set; on the plan moved, turned and sized (studio.html: any plane, also one fitted
          from spots that no real camera gives exactly); a camera without a plane gets one to start from
  check   any camera with another one's picture carried over the floor onto it, half-transparent (as
          autocalib --overlay), the other's mask / tracking_mask drawn where they fall on this one
  plan    the floor from above: the cameras' pictures carried onto it, their zones and the spots

Nothing is written until "Save": the changed zones, the cameras' planes (floor_points / distortion /
aspect, or removed) and, after a fit, [mct] (link on, link_dist, the fitted pairs out of disjoint) into
the config (barn.toml.bak kept); the spots into CALIB/points.json (with the pairs autocalib --points reads).

The floor plan is in about cm when CALIB holds the cows' boxes autocalib cached (boxes_<cam>.json, from any
autocalib --videos run); without them its units are arbitrary and link_dist is a guess.

    python herd/herd.py studio --config barn.toml --videos rec_fixed [--calib calib] [--port 8765]
    python herd/herd.py studio --config barn.toml --image cam1=cam1.jpg --image cam2=cam2.jpg
"""

from __future__ import annotations

import argparse
import base64
import collections
import glob
import http.server
import itertools
import json
import os
import re
import shutil
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import autocalib  # noqa: E402
from common import load_config, recordings  # noqa: E402

ZONES = ("mask", "exclude", "tracking_mask")
FLOOR = ("floor_points", "distortion", "aspect")
SHOW_W = 1280                       # the stills' width on the page (and the overlays')
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "studio.html")


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
        out.append(set_keys(b, vals) if vals else b)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(out))
    if link_dist is not None:
        autocalib.apply_config(path, {}, link_dist, shared, backup=False)
    load_config(path)                                         # still a config


# ------------------------------------------------------------------ spots

def spots_from_file(data):
    """points.json -> [{id, at: {cam: [u, v]}}]: its spots (one spot of the floor, on any number of cameras),
    or, from autocalib's pick page, each pair's row as a spot on those two."""
    if data.get("spots"):
        return [{"id": int(s["id"]), "at": {c: [float(p[0]), float(p[1])] for c, p in s["at"].items()}}
                for s in data["spots"] if s.get("at")]
    out = []
    for key, rows in data.get("pairs", {}).items():
        a, b = key.split(":")
        for r in rows:
            out.append({"id": len(out) + 1, "at": {a: [float(r[0]), float(r[1])], b: [float(r[2]), float(r[3])]}})
    return out


def pairs_from_spots(spots, ids):
    """Every two cameras' spots in common -> {"a:b": ([[uA, vA, uB, vB], ...], [spot id, ...])}, a before b
    in the config's order (autocalib's points.json pairs)."""
    order = {c: i for i, c in enumerate(ids)}
    out = {}
    for s in spots:
        cams = sorted((c for c in s["at"] if c in order), key=order.get)
        for a, b in itertools.combinations(cams, 2):
            rows, sid = out.setdefault(f"{a}:{b}", ([], []))
            rows.append([*s["at"][a], *s["at"][b]])
            sid.append(s["id"])
    return out


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

    def _floor(self, cam, calib):
        return autocalib.camera_floor(calib, self.aspect(cam)) if calib and calib.get("floor_points") else None

    def state(self):
        self.cfg = load_config(self.args.config)
        spots = []
        pj = os.path.join(self.calib, "points.json")
        if os.path.exists(pj):
            try:
                spots = spots_from_file(json.load(open(pj, encoding="utf-8")))
            except (OSError, ValueError, KeyError, IndexError):
                spots = []

        def polys(m):
            if not m:
                return []
            return [m] if isinstance(m[0][0], (int, float)) else m
        cams = []
        for c in self.cfg["cameras"]:
            w, h = self.frames[c["id"]].shape[1::-1] if c["id"] in self.frames else (16, 9)
            cams.append({"id": c["id"], "w": w, "h": h, "still": c["id"] in self.frames,
                         "boxes": self.boxes.get(c["id"], []), "min_box_pct": c.get("min_box_pct") or 0,
                         "px_w": self.size.get(c["id"], (w, h))[0],
                         **{k: polys(c.get(k)) for k in ZONES},
                         "calib": {k: c.get(k) for k in FLOOR} if c.get("floor_points") else None})
        m = self.cfg["mct"]
        return {"cams": cams, "spots": spots, "at": self.at, "videos": bool(self.videos),
                "boxes_cached": sorted(self.T), "link_dist": m.get("link_dist"),
                "neighbours": m.get("neighbours", []), "disjoint": m.get("disjoint", []),
                "config": os.path.abspath(self.args.config), "calib_dir": os.path.abspath(self.calib)}

    # -------------------------------------------------------------- calibration

    def spot_errors(self, spots, floor):
        """Each spot on each camera: how far (px of the camera's picture) from where the cameras together put
        it - its floor point the mean of theirs, taken back onto each picture."""
        fs = {c: self._floor(c, v) for c, v in floor.items() if c in self.ids}
        out = {}
        for s in spots:
            cams = [c for c in s["at"] if fs.get(c)]
            pts = {}
            for c in cams:
                u, v = s["at"][c]
                X, Y, ok = fs[c][0](np.array([u]), np.array([v]))
                if ok[0] and np.isfinite(X[0]) and np.isfinite(Y[0]):
                    pts[c] = (float(X[0]), float(Y[0]))
            if len(pts) < 2:
                continue
            m = np.mean(list(pts.values()), 0)
            errs = {}
            for c in pts:
                u, v, ok = fs[c][1](np.array([m[0]]), np.array([m[1]]))
                if ok[0] and np.isfinite(u[0]) and np.isfinite(v[0]):
                    w = self.size.get(c, (1280, 720))[0]
                    du, dv = (u[0] - s["at"][c][0]) * w, (v[0] - s["at"][c][1]) * w / self.aspect(c)
                    errs[c] = round(float(np.hypot(du, dv)), 1)
            out[str(s["id"])] = errs
        return out

    def fit(self, req):
        """The spots -> every two cameras with 5 in common fitted (autocalib.fit_spots), each group of cameras
        they join (only those with a camera in req["changed"], all without it) on one floor plan, placed
        clear of the other calibrated cameras -> the pairs' results, the cameras' new planes (a preview),
        link_dist, each spot's error on each camera."""
        spots = req.get("spots") or []
        floor = dict(req.get("floor") or {})
        changed = set(req.get("changed") or self.ids)
        pairs = pairs_from_spots(spots, self.ids)
        aspect = {c: self.aspect(c) for c in self.ids}
        graph = collections.defaultdict(set)
        for k, (rows, _) in pairs.items():
            if len(rows) >= 5:
                a, b = k.split(":")
                graph[a].add(b)
                graph[b].add(a)
        groups, seen = [], set()
        for c in self.ids:
            if c in graph and c not in seen:
                g, queue = {c}, [c]
                while queue:
                    for y in graph[queue.pop()]:
                        if y not in g:
                            g.add(y)
                            queue.append(y)
                seen |= g
                groups.append(g)
        results, notes, new, links = {}, [], {}, []
        for k, (rows, sid) in pairs.items():
            if len(rows) < 5:
                results[k] = {"ok": False, "n": len(rows), "spots": sid,
                              "notes": [f"{k.replace(':', '-')}: {len(rows)} spots in common - 5 at least"]}
        for g in groups:
            if not g & changed:
                continue
            found = {}
            for k, (rows, sid) in pairs.items():
                a, b = k.split(":")
                if a not in g or b not in g or len(rows) < 5:
                    continue
                r, said = autocalib.fit_spots(a, b, np.array(rows, float), self.size, aspect, self.T)
                notes += said
                results[k] = {"ok": bool(r and r.get("ok")), "n": len(rows), "spots": sid, "notes": said,
                              "left_out": [sid[i - 1] for i in (r or {}).get("spots_left_out", [])],
                              "median": (r or {}).get("median_px"), "lens": [(r or {}).get("kA"), (r or {}).get("kB")]}
                if r and r.get("ok"):
                    found[(a, b)] = r
            if not found:
                continue
            fitted = {c for ab in found for c in ab}
            taken = [(min(p[2] for p in v["floor_points"]), max(p[2] for p in v["floor_points"]))
                     for c, v in floor.items() if v and v.get("floor_points") and c not in fitted]
            calib, link, _ = autocalib.floor_plan(found, self.T, aspect, taken)
            new.update(calib)
            links.append(link)
            missing = sorted(c for c in calib if c not in self.T)
            if missing:
                notes.append(f"no boxes cached for {', '.join(missing)} in {self.calib} (boxes_<cam>.json from an "
                             "autocalib --videos run): the plan is not in cm and link_dist is a guess")
        floor.update(new)
        return {"pairs": results, "notes": notes, "floor": new,
                "link_dist": round(max(links)) if links else None, "errors": self.spot_errors(spots, floor)}

    def overlay(self, req):
        """req: base, top, floor {cam: floor}, top_zones {mask, tracking_mask} -> the top camera's picture
        carried over the floor onto the base one (PNG, transparent off the shared floor) and the top's zones
        where they fall on the base picture."""
        import cv2
        base, top = req["base"], req["top"]
        if base not in self.frames or top not in self.frames:
            return {"error": "no picture"}
        fa = self._floor(base, (req.get("floor") or {}).get(base))
        fb = self._floor(top, (req.get("floor") or {}).get(top))
        if fa is None or fb is None:
            return {"error": "no plane: " + ", ".join(c for c, f in ((base, fa), (top, fb)) if f is None)}
        warped, valid = autocalib.other_onto(self.frames[base], self.frames[top], fa, fb)
        png = cv2.imencode(".png", np.dstack([warped, valid.astype(np.uint8) * 255]))[1].tobytes()
        zones = {}
        for k in ("mask", "tracking_mask"):
            zones[k] = []
            for poly in (req.get("top_zones") or {}).get(k) or []:
                if len(poly) >= 3:
                    pts = autocalib._poly_onto([tuple(p) for p in poly], fa, fb, 1.0, 1.0)
                    zones[k].append([p if p and abs(p[0]) < 4 and abs(p[1]) < 4 else None for p in pts])
        return {"png": base64.b64encode(png).decode(), "share": float(valid.mean()), "zones": zones}

    def plan(self, req):
        """The floor from above: the calibrated cameras of req["focus"]'s group (cameras whose floors lie
        together - groups that share no floor are kept far apart on the plan) each carried onto one picture
        of the plan -> its extent [x0, y0, x1, y1] in plan units, size, and each camera's PNG (transparent
        where it does not see)."""
        import cv2
        floor = req.get("floor") or {}
        fs = {c: self._floor(c, v) for c, v in floor.items() if c in self.frames}
        fs = {c: f for c, f in fs.items() if f}
        if not fs:
            return {"error": "no camera has a plane yet"}
        box = {}
        for c, f in fs.items():
            u, v = np.meshgrid(np.linspace(0.02, 0.98, 40), np.linspace(0.02, 0.98, 40))
            u, v = u.ravel(), v.ravel()
            zone = [p for k in ("mask", "tracking_mask") for p in ((req.get("zones") or {}).get(c, {}).get(k) or [])
                    if len(p) >= 3]
            if zone:
                keep = np.array([any(_inside(x, y, p) for p in zone) for x, y in zip(u, v)])
                if keep.sum() >= 20:
                    u, v = u[keep], v[keep]
            X, Y, ok = f[0](u, v)
            ok &= np.isfinite(X) & np.isfinite(Y)
            if ok.sum() >= 10:
                box[c] = (np.percentile(X[ok], 3), np.percentile(Y[ok], 3), np.percentile(X[ok], 97),
                          np.percentile(Y[ok], 97))
        if not box:
            return {"error": "no camera sees its floor"}
        focus = req.get("focus") if req.get("focus") in box else next(iter(box))
        cx = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        fx, fy = cx(box[focus])
        group = [c for c in self.ids if c in box and np.hypot(cx(box[c])[0] - fx, cx(box[c])[1] - fy)
                 < autocalib.GROUP_APART / 2]
        x0, y0 = min(box[c][0] for c in group), min(box[c][1] for c in group)
        x1, y1 = max(box[c][2] for c in group), max(box[c][3] for c in group)
        pad = 0.06 * max(x1 - x0, y1 - y0, 1e-6)
        x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        W = 1100
        H = int(np.clip(W * (y1 - y0) / max(x1 - x0, 1e-6), 200, 1400))
        W = int(np.clip(H * (x1 - x0) / max(y1 - y0, 1e-6), 200, 1400))
        X, Y = np.meshgrid(x0 + (np.arange(W) + 0.5) / W * (x1 - x0), y0 + (np.arange(H) + 0.5) / H * (y1 - y0))
        pics = {}
        for c in group:
            img = self.frames[c]
            h, w = img.shape[:2]
            u, v, ok = fs[c][1](X, Y)
            valid = ok & np.isfinite(u) & np.isfinite(v) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
            mx, my = (np.where(valid, u, -1) * w - 0.5).astype(np.float32), (np.where(valid, v, -1) * h - 0.5).astype(np.float32)
            warped = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            png = cv2.imencode(".png", np.dstack([warped, valid.astype(np.uint8) * 255]))[1].tobytes()
            pics[c] = base64.b64encode(png).decode()
        return {"extent": [float(x0), float(y0), float(x1), float(y1)], "w": W, "h": H, "cams": pics,
                "group": group, "apart": sorted(set(box) - set(group))}

    def save(self, req):
        floor = dict(req.get("floor") or {})
        link = req.get("link_dist")
        shared = [tuple(k.split(":")) for k in req.get("calibrated_pairs") or []]
        save_config(self.args.config, req.get("cams") or {}, floor, link, shared)
        spots = [s for s in (req.get("spots") or []) if s.get("at")]
        pairs = {k: rows for k, (rows, _) in pairs_from_spots(spots, self.ids).items()}
        with open(os.path.join(self.calib, "points.json"), "w", encoding="utf-8") as fh:
            json.dump({"spots": spots, "pairs": pairs}, fh, indent=1)
        said = [f"{self.args.config}: zones of " + (", ".join(req.get("cams") or {}) or "no camera changed")]
        if floor:
            said.append("plane " + ", ".join(f"{c} {'set' if v else 'removed'}" for c, v in floor.items()))
        if link is not None:
            said.append(f"[mct] link on, link_dist {link}")
        said.append(f"the old file: {self.args.config}.bak; {len(spots)} spots: "
                    f"{os.path.join(self.calib, 'points.json')}")
        print("[studio] saved - " + "; ".join(said), flush=True)
        return {"said": said, "state": self.state()}


def _inside(x, y, poly):
    ins, j = False, len(poly) - 1
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[j]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1:
            ins = not ins
        j = i
    return ins


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
                return self._send(open(PAGE, encoding="utf-8").read(), "text/html; charset=utf-8")
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
                    if self.path == "/api/plan":
                        return self._send(studio.plan(req))
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
