#!/usr/bin/env python3
"""The cameras' setup in one page: zones, the floor plane and its check, edited together and saved to barn.toml.

A small web server (in Docker: docker compose run --rm -p 8765:8765 herd studio ...) serving one page
(studio.html) with a still of every camera, one, two or four at a time, or the plan:

  zones   mask (the floor a camera counts), exclude (greyed before the detector), tracking_mask (where it
          follows cows beyond its mask) - drawn on the picture, the cows found coloured by the rule the barn
          uses (as herd.py zones page)
  floor   the floors themselves (a pen A, the next B, ...): their corners clicked round the edge, the same
          corners on every camera that sees them (corners are spots with a floor and a place along its
          edge), points along an edge where its corners are out of sight; measured sizes (a rectangle,
          or any corner's or spot's X, Y) put the plan on the real floor: a camera with 8 equations of it
          gets its plane alone (dlt_mixed: points and points on lines), the others follow through shared
          spots or are fitted in pairs and put on the corners either sees, then all are refined together
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
from common import homography, load_config, recordings, undistort  # noqa: E402

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
    """points.json -> [{id, at: {cam: [u, v]}, corner?, xy?}]: its spots (one spot of the floor, on any number of
    cameras; a corner of a floor's outline has its floor (1, 2, ... - A, B, ...) and its place along it,
    `corner`, and maybe its measured floor point, `xy`), or, from autocalib's pick page, each pair's row as a
    spot on those two."""
    if data.get("spots"):
        out = []
        for s in data["spots"]:
            if not s.get("at") and s.get("corner") is None:
                continue
            sp = {"id": int(s["id"]), "at": {c: [float(p[0]), float(p[1])] for c, p in (s.get("at") or {}).items()}}
            if s.get("corner") is not None:
                sp["corner"] = int(s["corner"])
                sp["floor"] = int(s.get("floor") or 1)
            if s.get("xy") is not None:
                sp["xy"] = [float(s["xy"][0]), float(s["xy"][1])]
            out.append(sp)
        return out
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


def _norm(P):
    """Hartley's normalisation of 2D points: centred, mean distance sqrt(2) -> 3x3."""
    P = np.asarray(P, float)
    c = P.mean(0)
    s = np.sqrt(2) / max(float(np.mean(np.hypot(*(P - c).T))), 1e-12)
    return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])


def dlt_mixed(p, X, e=(), L=()):
    """H (picture -> floor) from picture points p on floor points X, and picture points e on floor lines L
    (a, b, c: aX + bY + c = 0 - an edge of the floor, its corners measured): one DLT, Hartley-normalised.
    A point gives 2 equations, a point on a line 1; 8 are needed, and they must fix H (no 3 points on one
    line doing all the work) -> 3x3, or None."""
    p, X = np.asarray(p, float).reshape(-1, 2), np.asarray(X, float).reshape(-1, 2)
    e, L = np.asarray(e, float).reshape(-1, 2), np.asarray(L, float).reshape(-1, 3)
    if len(X) < 2:
        return None
    Ti, Tf = _norm(np.r_[p, e]), _norm(X)
    ap = lambda T, P: (np.c_[P, np.ones(len(P))] @ T.T)[:, :2]
    rows = []
    for (x, y), (U, V) in zip(ap(Ti, p), ap(Tf, X)):
        rows += [[x, y, 1, 0, 0, 0, -U * x, -U * y, -U], [0, 0, 0, x, y, 1, -V * x, -V * y, -V]]
    Ln = L @ np.linalg.inv(Tf)                    # lines go with the inverse: l' = l T^-1
    for (x, y), (a, b, c) in zip(ap(Ti, e), Ln):
        n = np.hypot(a, b) or 1.0
        a, b, c = a / n, b / n, c / n
        rows.append([a * x, a * y, a, b * x, b * y, b, c * x, c * y, c])
    if len(rows) < 8:
        return None
    _, S, Vt = np.linalg.svd(np.array(rows))
    if (S[-1] if len(S) < 9 else S[-2]) < 1e-6 * S[0]:
        return None                               # not fixed: the constraints leave a family of planes
    H = np.linalg.inv(Tf) @ Vt[-1].reshape(3, 3) @ Ti
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H


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
        spots, edges = [], []
        pj = os.path.join(self.calib, "points.json")
        if os.path.exists(pj):
            try:
                data = json.load(open(pj, encoding="utf-8"))
                spots = spots_from_file(data)
                edges = [{"id": int(e["id"]), "a": int(e["a"]), "b": int(e["b"]), "cam": e["cam"],
                          "p": [float(e["p"][0]), float(e["p"][1])]} for e in data.get("edges") or []]
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                spots, edges = [], []

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
        return {"cams": cams, "spots": spots, "edges": edges, "at": self.at, "videos": bool(self.videos),
                "boxes_cached": sorted(self.T), "link_dist": m.get("link_dist"),
                "neighbours": m.get("neighbours", []), "disjoint": m.get("disjoint", []),
                "config": os.path.abspath(self.args.config), "calib_dir": os.path.abspath(self.calib)}

    # -------------------------------------------------------------- calibration

    def spot_errors(self, spots, floor):
        """Each spot on each camera: how far (px of the camera's picture) from where it should be - its measured
        floor point (xy), else the mean of the cameras' - taken back onto each picture."""
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
            if len(pts) < (1 if s.get("xy") else 2):
                continue
            m = np.array(s["xy"], float) if s.get("xy") else np.mean(list(pts.values()), 0)
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
        """The floor and the spots -> the cameras' planes (a preview): first every camera onto the measured
        floor (onto_floor: its measured corners, its points on the floor's edges, and the spots it shares
        with a camera already on it); then every two cameras with 5 spots in common fitted to each other
        (autocalib.fit_spots), each group they join - one with no camera on the measured floor, and a
        camera of it in req["changed"] (all without it) - on one floor plan, placed clear of the other
        calibrated cameras and moved onto the measured corners it sees -> the pairs' results, the cameras'
        new planes, how each was found, link_dist, each spot's error on each camera."""
        spots = req.get("spots") or []
        edges = req.get("edges") or []
        floor = dict(req.get("floor") or {})
        changed = set(req.get("changed") or self.ids)
        aspect = {c: self.aspect(c) for c in self.ids}
        new, planes, floor_said = self.onto_floor(spots, edges, aspect)
        notes = [n for c, n in floor_said.items() if c in new]          # placed on the floor: said now
        pairs = pairs_from_spots(spots, self.ids)
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
        results, links = {}, []
        for k, (rows, sid) in pairs.items():
            if len(rows) < 5:
                results[k] = {"ok": False, "n": len(rows), "spots": sid,
                              "notes": [f"{k.replace(':', '-')}: {len(rows)} spots in common - 5 at least"]}
        for g in groups:
            if not g & changed or g & set(new):       # on the measured floor already: by it, not by pairs
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
                     for c, v in {**floor, **new}.items() if v and v.get("floor_points") and c not in fitted]
            calib, link, _ = autocalib.floor_plan(found, self.T, aspect, taken)
            calib, said, measured = self.onto_measured(calib, spots, aspect)
            notes += said
            new.update(calib)
            links.append(link)
            missing = sorted(c for c in calib if c not in self.T)
            if missing and measured < 3:
                notes.append(f"no boxes cached for {', '.join(missing)} in {self.calib} (boxes_<cam>.json from an "
                             "autocalib --videos run): the plan is not in cm and link_dist is a guess")
        notes += [n for c, n in floor_said.items() if c not in new]     # why not on the floor - those left without
        floor.update(new)
        if req.get("refine", True):
            moved, said = self.refine(floor, spots, edges, aspect)
            new.update(moved)
            floor.update(moved)
            notes += said
        return {"pairs": results, "notes": notes, "floor": new, "planes": planes,
                "link_dist": round(max(links)) if links else None, "errors": self.spot_errors(spots, floor)}

    def onto_floor(self, spots, edges, aspect):
        """Every camera that sees enough of the measured floor gets its plane on its own: its measured corners
        (2 equations each), its points on the floor's edges between two measured corners (1 each), and the
        spots it shares with cameras placed so far, at the floor point they give them (2 each) - 8 at least
        (fit_floor). Round by round, so that a camera seeing no measured corner follows one that does. ->
        ({camera: calib}, {camera: how}, {camera: what to say})."""
        measured = {sp["id"]: sp["xy"] for sp in spots if sp.get("xy")}
        if len(measured) < 2:
            return {}, {}, {}
        line = lambda a, b: np.cross([*measured[a], 1.0], [*measured[b], 1.0]) if a in measured and b in measured else None
        known, out, planes, said = dict(measured), {}, {}, {}
        for _ in range(len(self.ids)):
            grew = False
            for c in self.ids:
                if c in out:
                    continue
                pts = [[*sp["at"][c], *known[sp["id"]]] for sp in spots if c in sp["at"] and sp["id"] in known]
                lns = [[*e["p"], *L] for e in edges if e.get("cam") == c for L in [line(e.get("a"), e.get("b"))] if L is not None]
                n_eq = 2 * len(pts) + len(lns)
                if not pts and not lns:
                    continue
                if len(pts) < 2 or n_eq < 8:
                    said[c] = (f"{c}: {len(pts)} floor point(s) and {len(lns)} edge point(s) - {n_eq} of the 8 equations "
                               "a plane needs (a measured corner or a spot placed by another camera: 2, a point on a "
                               "floor's edge: 1; 2 points at least)")
                    continue
                cal, px, why = self.fit_floor(c, pts, lns, aspect[c])
                if cal is None:
                    said[c] = f"{c}: {why}"
                    continue
                out[c] = cal
                planes[c] = {"points": len(pts), "edge_points": len(lns), "rms": px, "lens": cal["distortion"]}
                said[c] = (f"{c}: its plane from the floor - {len(pts)} point(s), {len(lns)} on its edges"
                           + (f", within {px:.0f} px (rms)" if n_eq > 8 else " (8 equations: exact, nothing to check them against)")
                           + (f"; lens {cal['distortion']}" if cal["distortion"] else ""))
                grew = True
            fs = {c: autocalib.camera_floor(v, aspect[c]) for c, v in out.items()}
            for sp in spots:                     # the spots of the cameras placed: where they are on the floor
                if sp["id"] in measured:
                    continue
                P = []
                for c, (u, v) in sp["at"].items():
                    if c in fs:
                        X, Y, ok = fs[c][0](np.array([u]), np.array([v]))
                        if ok[0] and np.isfinite(X[0]) and np.isfinite(Y[0]):
                            P.append((float(X[0]), float(Y[0])))
                if P:
                    known[sp["id"]] = list(np.mean(P, 0))
            if not grew:
                break
        return out, planes, {c: said[c] for c in self.ids if c in said}

    def fit_floor(self, cam, pts, lns, ar):
        """One camera's plane from floor points [[u, v, X, Y], ...] and points on floor lines [[u, v, a, b, c],
        ...] (dlt_mixed); with 12 equations or more the lens bend too, when clearly closer -> (calib, rms px
        off, None) or (None, None, why not)."""
        P, E = np.array(pts, float).reshape(-1, 4), np.array(lns, float).reshape(-1, 5)
        w = self.size.get(cam, (1280, 720))[0]
        n_eq = 2 * len(P) + len(E)
        best = None
        for k in (autocalib.K_GRID_POINTS if n_eq >= 12 else (0.0,)):
            pu = np.c_[undistort(P[:, 0], P[:, 1], k, ar)]
            eu = np.c_[undistort(E[:, 0], E[:, 1], k, ar)] if len(E) else np.zeros((0, 2))
            H = dlt_mixed(pu, P[:, 2:4], eu, E[:, 2:5])
            if H is None:
                continue
            side = np.sign(np.r_[pu, eu] @ H[2, :2] + H[2, 2])
            if len(set(side)) > 1:
                continue                          # the floor folded over: not this lens
            Hi = np.linalg.inv(H)
            err = []
            for (u, v, X, Y) in P:
                q = Hi @ [X, Y, 1.0]
                ua, va = autocalib.distort(q[0] / q[2], q[1] / q[2], k, ar)
                err.append(np.hypot((ua - u) * w, (va - v) * w / ar))
            for (u, v, a, b, c), (x, y) in zip(E, eu):
                q = H @ [x, y, 1.0]
                q = q[:2] / q[2]
                n2 = a * a + b * b
                foot = q - (a * q[0] + b * q[1] + c) / n2 * np.array([a, b])
                r = Hi @ [foot[0], foot[1], 1.0]
                ua, va = autocalib.distort(r[0] / r[2], r[1] / r[2], k, ar)
                err.append(np.hypot((ua - u) * w, (va - v) * w / ar))
            err = np.array(err, float)
            if not np.all(np.isfinite(err)):
                continue
            rms = float(np.sqrt(np.mean(err ** 2)))
            if best is None or (rms < 0.8 * best[0] if best[1] == 0.0 else rms < best[0]):
                best = (rms, k, H, side[0])
        if best is None:
            return None, None, ("what it sees of the floor does not fix a plane - 2 far corners and the side edges leave "
                                "how deep the floor is open (a corner already lies on its edge: an edge then adds only "
                                "its direction, and a measured point on an edge only its place along it). Add measured points "
                                "(a post along each side at a known distance, or one inside the floor), spots shared with "
                                "a camera that sees the other corners, or set the plane roughly (tool Plane) and Refine")
        rms, k, H, side = best
        uv = np.r_[P[:, :2], E[:, :2]]
        lo, hi = np.clip(uv.min(0) - 0.02, 0.03, 0.97), np.clip(uv.max(0) + 0.02, 0.03, 0.97)
        fp = []
        for u in np.linspace(lo[0], hi[0], 3):
            for v in np.linspace(lo[1], hi[1], 3):
                a, b = undistort(u, v, k, ar)
                q = H @ [a, b, 1.0]
                if np.sign(q[2]) == side:
                    fp.append([round(float(u), 4), round(float(v), 4), round(float(q[0] / q[2]), 1), round(float(q[1] / q[2]), 1)])
        if len(fp) < 4:
            return None, None, "its floor points lie beyond the picture's horizon"
        return {"floor_points": fp, "distortion": k, "aspect": round(ar, 4)}, round(rms, 1), None

    def refine(self, floor, spots, edges, aspect):
        """Every camera's plane adjusted at once (least squares, robust to a bad click): the measured points
        and corners where each camera sees them, the points on the floor's edges on those edges, and each
        spot shared by cameras at one floor point for them all - in px of the pictures. What none of that
        fixes stays as it is (a weak hold on each plane): a plane set by hand (tool Plane) and refined keeps
        what was set where nothing says otherwise - a thin strip shared by two cameras is enough then.
        -> ({camera: calib} of those that moved, what to say)."""
        from scipy.optimize import least_squares
        cams = [c for c in self.ids if (floor.get(c) or {}).get("floor_points")]
        measured = {sp["id"]: sp["xy"] for sp in spots if sp.get("xy")}
        line = lambda a, b: np.cross([*measured[a], 1.0], [*measured[b], 1.0]) if a in measured and b in measured else None
        info = {}
        for c in cams:
            f = floor[c]
            k, ar = float(f.get("distortion") or 0.0), float(f.get("aspect") or self.aspect(c))
            P = np.array(f["floor_points"], float)
            uu, vv = undistort(P[:, 0], P[:, 1], k, ar)
            H0 = homography(np.c_[uu, vv], P[:, 2:4])
            g = np.c_[uu, vv]
            d_img = np.hypot(*(g[:, None] - g[None]).transpose(2, 0, 1))
            gf = np.c_[g, np.ones(len(g))] @ H0.T
            gf = gf[:, :2] / gf[:, 2:3]
            d_fl = np.hypot(*(gf[:, None] - gf[None]).transpose(2, 0, 1))
            scale = float(np.median(d_fl[d_img > 1e-6] / d_img[d_img > 1e-6]))          # floor units per picture unit
            info[c] = {"k": k, "ar": ar, "w": self.size.get(c, (1280, 720))[0], "H0": H0 / H0[2, 2], "g": g, "gf": gf,
                       "scale": scale, "P": P}
        ci = {c: i for i, c in enumerate(cams)}
        pts, shared, lns = [], [], []
        for sp in spots:
            seen = [c for c in sp["at"] if c in ci]
            if sp["id"] in measured:
                pts += [(ci[c], sp["at"][c], measured[sp["id"]]) for c in seen]
            elif len(seen) >= 2:
                shared.append((sp["id"], [(ci[c], sp["at"][c]) for c in seen]))
        for e in edges:
            L = line(e.get("a"), e.get("b"))
            if L is not None and e.get("cam") in ci:
                lns.append((ci[e["cam"]], e["p"], L))
        if not shared and not pts and not lns:
            return {}, []
        # the start: the planes as they are; each shared spot at the mean of the cameras' floor points
        x0 = [info[c]["H0"].ravel()[:8] for c in cams]
        spot0 = []
        for _, obs in shared:
            fl = []
            for i, (u, v) in obs:
                c = cams[i]
                a, b = undistort(u, v, info[c]["k"], info[c]["ar"])
                q = info[c]["H0"] @ [a, b, 1.0]
                fl.append(q[:2] / q[2])
            spot0.append(np.mean(fl, 0))
        x0 = np.concatenate([np.concatenate(x0)] + [np.array(s_) for s_ in spot0]) if spot0 else np.concatenate(x0)
        n8 = 8 * len(cams)

        def px(c, H, X, Y, u, v):                # the floor point (X, Y) on camera c's picture vs the click, px
            q = np.linalg.solve(H, [X, Y, 1.0])
            ua, va = autocalib.distort(q[0] / q[2], q[1] / q[2], info[c]["k"], info[c]["ar"])
            w, ar = info[c]["w"], info[c]["ar"]
            return [(ua - u) * w, (va - v) * w / ar]

        def residuals(x, hold=True):
            Hs = [np.append(x[8 * i: 8 * i + 8], 1.0).reshape(3, 3) for i in range(len(cams))]
            r = []
            for i, (u, v), (X, Y) in pts:
                r += px(cams[i], Hs[i], X, Y, u, v)
            for j, (_, obs) in enumerate(shared):
                X, Y = x[n8 + 2 * j: n8 + 2 * j + 2]
                for i, (u, v) in obs:
                    r += px(cams[i], Hs[i], X, Y, u, v)
            for i, (u, v), (a, b, c0) in lns:
                c = cams[i]
                ua_, va_ = undistort(u, v, info[c]["k"], info[c]["ar"])
                q = Hs[i] @ [ua_, va_, 1.0]
                q = q[:2] / q[2]
                foot = q - (a * q[0] + b * q[1] + c0) / (a * a + b * b) * np.array([a, b])
                d = px(c, Hs[i], foot[0], foot[1], u, v)
                r.append(float(np.hypot(*d)))
            if hold:                             # the weak hold: each plane's own grid stays where it was
                for i, c in enumerate(cams):
                    I = info[c]
                    q = np.c_[I["g"], np.ones(len(I["g"]))] @ Hs[i].T
                    r += list((((q[:, :2] / q[:, 2:3]) - I["gf"]) / I["scale"] * I["w"] * 0.05).ravel())
            return np.array(r, float)

        before = residuals(x0, hold=False)
        sol = least_squares(residuals, x0, loss="soft_l1", f_scale=8.0, max_nfev=400, x_scale="jac")
        after = residuals(sol.x, hold=False)
        rms = lambda r: float(np.sqrt(np.mean(np.square(r)))) if len(r) else 0.0
        out = {}
        for i, c in enumerate(cams):
            I, H = info[c], np.append(sol.x[8 * i: 8 * i + 8], 1.0).reshape(3, 3)
            q = np.c_[I["g"], np.ones(len(I["g"]))] @ H.T
            fl = q[:, :2] / q[:, 2:3]
            moved = float(np.max(np.hypot(*(fl - I["gf"]).T)) / I["scale"] * I["w"])
            if moved > 0.5 and np.all(np.sign(q[:, 2]) == np.sign(q[0, 2])):
                out[c] = {"floor_points": [[round(float(u), 4), round(float(v), 4), round(float(X), 1), round(float(Y), 1)]
                                           for (u, v), (X, Y) in zip(I["P"][:, :2], fl)],
                          "distortion": I["k"], "aspect": round(I["ar"], 4)}
        said = [f"refined together ({len(pts)} measured, {len(shared)} shared spots, {len(lns)} edge points): "
                f"{rms(before):.1f} -> {rms(after):.1f} px (rms); moved: {', '.join(out) or 'none'}"] if out else []
        return out, said

    def onto_measured(self, calib, spots, aspect):
        """A group of cameras on one plan -> the plan moved onto the floor's measured corners (xy) they see:
        4 or more a perspective correction (the plan then square and in the measured units), 3 an affine one,
        2 a turn, shift and scale. -> (calib, what to say, how many measured corners)."""
        fs = {c: autocalib.camera_floor(v, aspect[c]) for c, v in calib.items()}
        src, dst = [], []
        for sp in spots:
            if not sp.get("xy"):
                continue
            pts = []
            for c, (u, v) in sp["at"].items():
                if c in fs:
                    X, Y, ok = fs[c][0](np.array([u]), np.array([v]))
                    if ok[0] and np.isfinite(X[0]) and np.isfinite(Y[0]):
                        pts.append((float(X[0]), float(Y[0])))
            if pts:
                src.append(np.mean(pts, 0))
                dst.append(sp["xy"])
        n = len(src)
        if n < 2:
            return calib, [], n
        src, dst = np.array(src, float), np.array(dst, float)
        if n >= 4:
            T = homography(src, dst)
        elif n == 3:
            A = np.linalg.lstsq(np.c_[src, np.ones(3)], dst, rcond=None)[0].T
            T = np.vstack([A, [0, 0, 1]])
        else:
            a = complex(*(dst[1] - dst[0])) / complex(*(src[1] - src[0]))
            b = complex(*dst[0]) - a * complex(*src[0])
            T = np.array([[a.real, -a.imag, b.real], [a.imag, a.real, b.imag], [0, 0, 1.0]])
        out = {}
        for c, v in calib.items():
            P = np.array(v["floor_points"], float)
            q = np.c_[P[:, 2:4], np.ones(len(P))] @ T.T
            P[:, 2:4] = q[:, :2] / q[:, 2:3]
            out[c] = {**v, "floor_points": [[round(float(x), 4) for x in r[:2]] + [round(float(x), 1) for x in r[2:]]
                                            for r in P]}
        q = np.c_[src, np.ones(n)] @ T.T
        off = np.hypot(*(q[:, :2] / q[:, 2:3] - dst).T)
        how = {2: "turned, shifted and scaled", 3: "fitted (affine)"}.get(n, "fitted (perspective: square, in their units)")
        said = [f"{', '.join(sorted(calib))}: the plan {how} onto {n} measured corners"
                + (f", the farthest {off.max():.0f} off" if n > {2: 2, 3: 3}.get(n, 4) else "")]
        if n == 2:
            said.append("2 measured corners only turn, shift and scale the plan: 3 or more also fix its shape")
        return out, said, n

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
        spots = [s for s in (req.get("spots") or []) if s.get("at") or s.get("corner") is not None]
        pairs = {k: rows for k, (rows, _) in pairs_from_spots(spots, self.ids).items()}
        with open(os.path.join(self.calib, "points.json"), "w", encoding="utf-8") as fh:
            json.dump({"spots": spots, "edges": req.get("edges") or [], "pairs": pairs}, fh, indent=1)
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
