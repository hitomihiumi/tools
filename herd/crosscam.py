#!/usr/bin/env python3
"""Which cow on the neighbour camera is she - by where she stands, learnt from your labelled boxes.

The cameras do not move: what surrounds a cow in one camera's frame (the walls, the drinker, the posts)
says where on the floor she is, and where that place lies in the neighbour's frame is one mapping
between the two pictures - a homography of the floor, the lens taken out with the config's distortion.
Your labels give it with no measuring: a cow of yours on two cameras at one moment is a pair of points.

    python herd/herd.py crosscam --config barn.toml --boxes boxes --labels labels.json [--out D] [--apply]

  1. the pairs, for each two cameras ([mct] neighbours, else every two with a cow of yours on both at once)
  2. the mapping (RANSAC) and its check moment by moment: fitted without that moment, its cows on the
     first camera put onto the second and matched one to one by distance - the right cow how often, beside
     the chance of a guess. Fitted twice, on the boxes' bottom centres and on their centres
  3. OUT/crosscam.json; --apply (one of the two calibrated, floor_points; both: --onto names the one to
     keep): the other camera's floor_points onto its floor, by the better of the two points (floor_at on
     both cameras says which the run then maps), so [mct] link joins them - barn.toml.bak kept
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import homography, load_config, undistort  # noqa: E402

RANSAC_THR = 0.05     # a pair agrees within this (frame heights, the lens taken out)
GATE = 0.15           # a cow put further than this from every box of the other camera matches none
AT_ONCE_S = 1.0       # two cameras' frames this close are one moment
MIN_PAIRS = 6


def lens(cam, w, h):
    return cam.get("distortion") or 0.0, cam.get("aspect") or (w / h if h else 16 / 9)


def point(box, cam, w, h, where):
    """A box -> its point in the straightened frame, x scaled by the aspect (one unit = the frame's height)."""
    k, ar = lens(cam, w, h)
    u, v = (box[0] + box[2]) / 2, (box[3] if where == "bottom" else (box[1] + box[3]) / 2)
    a, b = undistort(u, v, k, ar) if k else (u, v)
    return (a * ar, b)


def moments(cfg, frames, labels, a, b):
    """[{"a": [(id, box, w, h)], "b": [...]}]: frames of a and b at one moment, your named boxes on each."""
    cams = {c["id"]: c for c in cfg["cameras"]}
    named = {}
    for file, lab in labels.items():
        f = frames.get(file)
        if not f or lab.get("status") != "done" or f["cam"] not in (a, b) or f["cam"] not in cams:
            continue
        bx = [(str(x[4]).strip(), x[:4], f.get("w", 1920), f.get("h", 1080)) for x in lab.get("boxes", [])
              if len(x) >= 5 and str(x[4]).strip() and str(x[4]).strip().lower() != "x"]
        named.setdefault(f["cam"], []).append((f["ts"], bx))
    out, used = [], set()
    for ta, ba in sorted(named.get(a, [])):
        best = min(((abs(ta - tb), i) for i, (tb, _) in enumerate(named.get(b, [])) if i not in used), default=None)
        if best and best[0] <= AT_ONCE_S:
            used.add(best[1])
            out.append({"ts": ta, "a": ba, "b": named[b][best[1]][1]})
    return out


def pairs_of(mo, cam_a, cam_b, where):
    """One moment -> (points on a, points on b) of your cows on both."""
    ib = {i: (box, w, h) for i, box, w, h in mo["b"]}
    pa, pb = [], []
    for i, box, w, h in mo["a"]:
        if i in ib:
            pa.append(point(box, cam_a, w, h, where))
            pb.append(point(ib[i][0], cam_b, ib[i][1], ib[i][2], where))
    return pa, pb


def apply_h(H, pts):
    p = np.c_[np.asarray(pts, float), np.ones(len(pts))] @ H.T
    w = np.where(np.abs(p[:, 2]) < 1e-12, 1e-12, p[:, 2])
    return p[:, :2] / w[:, None]


def ransac(src, dst, rng, iters=1000, thr=RANSAC_THR):
    """H (src -> dst) from the pairs most of which agree, refitted on them -> (H, inliers) or (None, None)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    n = len(src)
    if n < MIN_PAIRS:
        return None, None
    best = None
    for _ in range(iters):
        s = rng.sample(range(n), 4)
        try:
            H = homography(src[s], dst[s])
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(H)):
            continue
        inl = np.hypot(*(apply_h(H, src) - dst).T) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < MIN_PAIRS:
        return None, None
    H = homography(src[best], dst[best])
    inl = np.hypot(*(apply_h(H, src) - dst).T) < thr
    if inl.sum() >= MIN_PAIRS:
        H = homography(src[inl], dst[inl])
    return H, inl


def check(mos, cam_a, cam_b, where, seed=0, gate=GATE):
    """Each moment matched by the mapping fitted without it -> the counts and the misses by distance."""
    from scipy.optimize import linear_sum_assignment
    per = [pairs_of(mo, cam_a, cam_b, where) for mo in mos]
    n = collections.Counter()
    miss, chance, cows = [], [], collections.defaultdict(collections.Counter)
    for k, mo in enumerate(mos):
        src = [p for j, (pa, _) in enumerate(per) if j != k for p in pa]
        dst = [p for j, (_, pb) in enumerate(per) if j != k for p in pb]
        H, _ = ransac(src, dst, random.Random(seed + k))
        if H is None:
            n["not_fitted"] += 1
            continue
        A = [point(box, cam_a, w, h, where) for _, box, w, h in mo["a"]]
        B = [point(box, cam_b, w, h, where) for _, box, w, h in mo["b"]]
        if not A or not B:
            continue
        P = apply_h(H, A)
        D = np.hypot(P[:, None, 0] - np.asarray(B)[None, :, 0], P[:, None, 1] - np.asarray(B)[None, :, 1])
        r, c = linear_sum_assignment(D)
        got = {i: j for i, j in zip(r, c) if D[i, j] <= gate}
        idb = [i for i, _, _, _ in mo["b"]]
        for i, (cid, _, _, _) in enumerate(mo["a"]):
            if cid not in idb:
                continue
            n["cows"] += 1
            chance.append(1.0 / len(idb))
            miss.append(float(D[i, idb.index(cid)]))
            j = got.get(i)
            what = "none" if j is None else ("right" if idb[j] == cid else "wrong")
            n[what] += 1
            cows[cid][what] += 1
    return {"cows": n["cows"], "right": n["right"], "wrong": n["wrong"], "none": n["none"],
            "moments_not_fitted": n["not_fitted"], "chance": float(np.mean(chance)) if chance else 0.0,
            "median_miss": float(np.median(miss)) if miss else None,
            "by_cow": {c: dict(v) for c, v in sorted(cows.items())}}


def onto(cfg, ref, oth, H, mos, where):
    """oth's floor_points on ref's floor: points of oth where your cows stood, through H (oth -> ref) onto ref's
    frame and ref's calibration -> ([[u, v, X, Y]] of oth, as clicked, its lens). mos: oth's moments with ref
    ("a" oth, "b" ref). With where = "centre" the points are the boxes' centres, read on ref's floor as
    its centres are (floor_at = "centre" on both)."""
    from autocalib import camera_floor, distort
    cams = {c["id"]: c for c in cfg["cameras"]}
    fr = camera_floor(cams[ref])
    _, _, w, h = next(x for mo in mos for x in mo["a"])
    ko, aro = lens(cams[oth], w, h)
    _, _, wr, hr = next(x for mo in mos for x in mo["b"])
    kr, arr = lens(cams[ref], wr, hr)
    raw = np.array([((x[0] + x[2]) / 2, x[3] if where == "bottom" else (x[1] + x[3]) / 2)
                    for mo in mos for _, x, _, _ in mo["a"]])
    us = np.quantile(raw[:, 0], [0.1, 0.5, 0.9])
    vs = np.quantile(raw[:, 1], [0.1, 0.5, 0.9])
    out = []
    for v in vs:
        for u in us:
            uo, vo = undistort(u, v, ko, aro) if ko else (u, v)
            q = apply_h(H, [(uo * aro, vo)])[0]
            ur, vr = distort(q[0] / arr, q[1], kr, arr)
            X, Y, ok = fr[0](np.array([ur]), np.array([vr]))
            if bool(ok[0]) and np.isfinite(X[0]) and np.isfinite(Y[0]):
                out.append([round(float(u), 4), round(float(v), 4), round(float(X[0]), 1), round(float(Y[0]), 1)])
    return out, ko, aro


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--boxes", required=True, help="herd.py boxes' folder (frames.json)")
    p.add_argument("--labels", required=True, help="labels.json with your cow ids (one cow, one id on every camera)")
    p.add_argument("--out", default=None, help="where crosscam.json goes (default: next to --boxes)")
    p.add_argument("--cameras", nargs=2, action="append", default=None, metavar=("A", "B"),
                   help="these two (again for more pairs); default [mct] neighbours, else every two")
    p.add_argument("--apply", action="store_true",
                   help="the uncalibrated camera of each pair onto the calibrated one's floor in barn.toml")
    p.add_argument("--gate", type=float, default=GATE,
                   help="a cow put further than this (frame heights) from every box of the other camera matches none")
    p.add_argument("--onto", default=None,
                   help="with --apply, both of a pair calibrated: the camera the other is put onto (its own "
                        "floor_points replaced)")
    args = p.parse_args(argv)
    from box_dataset import read_labels
    cfg = load_config(args.config)
    cams = {c["id"]: c for c in cfg["cameras"]}
    frames = {f["file"]: f for f in json.load(open(os.path.join(args.boxes, "frames.json"), encoding="utf-8"))["frames"]}
    labels = read_labels(args.labels)
    pairs = args.cameras or cfg["mct"].get("neighbours") or list(itertools.combinations(sorted(cams), 2))
    report, calib, shared = {}, {}, []
    for a, b in pairs:
        if a not in cams or b not in cams:
            print(f"[crosscam] {a} / {b}: not in the config")
            continue
        mos = moments(cfg, frames, labels, a, b)
        npairs = sum(len(pairs_of(mo, cams[a], cams[b], "bottom")[0]) for mo in mos)
        if npairs < MIN_PAIRS + 4:
            print(f"[crosscam] {a} -> {b}: {npairs} of your cows on both at one moment - too few to learn from")
            continue
        r = {"moments": len(mos), "pairs": npairs}
        for where in ("bottom", "centre"):
            src = [q for mo in mos for q in pairs_of(mo, cams[a], cams[b], where)[0]]
            dst = [q for mo in mos for q in pairs_of(mo, cams[a], cams[b], where)[1]]
            H, inl = ransac(src, dst, random.Random(0))
            r[where] = dict(check(mos, cams[a], cams[b], where, gate=args.gate), inliers=int(inl.sum()) if inl is not None else 0,
                            H=H.tolist() if H is not None else None)
        best = max(("bottom", "centre"), key=lambda w: r[w]["right"])
        r["best"] = best
        report[f"{a}->{b}"] = r
        x = r[best]
        print(f"[crosscam] {a} -> {b}: {npairs} pairs of your cows at {len(mos)} moments")
        for where in ("bottom", "centre"):
            y = r[where]
            if not y["cows"]:
                print(f"  {where:6}: no mapping")
                continue
            print(f"  {where:6}: the right cow {y['right'] / y['cows']:.0%} (another {y['wrong'] / y['cows']:.0%}, none near "
                  f"{y['none'] / y['cows']:.0%}) of {y['cows']}, a guess {y['chance']:.0%}; median miss "
                  f"{y['median_miss']:.1%} of the frame height; {y['inliers']} of {npairs} pairs agree")
        bad = [(c, v) for c, v in x["by_cow"].items() if v.get("right", 0) < sum(v.values())]
        if bad:
            print("  cows it missed (" + best + "): " + ", ".join(
                f"{c} {v.get('right', 0)}/{sum(v.values())}" for c, v in sorted(bad, key=lambda kv: kv[1].get('right', 0) - sum(kv[1].values()))[:12]))
        if args.apply:
            fa, fb = bool(cams[a].get("floor_points")), bool(cams[b].get("floor_points"))
            ref = args.onto if args.onto in (a, b) else (a if fa and not fb else b if fb and not fa else None)
            if ref is None or not cams[ref].get("floor_points"):
                print("  --apply: " + (f"neither {a} nor {b} has floor_points - calibrate one (studio), then again"
                                        if not (fa or fb) else
                                        f"{a} and {b} both calibrated - --onto {a} (or {b}): the other is put onto it"))
                continue
            oth = b if ref == a else a
            mos_o = moments(cfg, frames, labels, oth, ref)
            src = [q for mo in mos_o for q in pairs_of(mo, cams[oth], cams[ref], best)[0]]
            dst = [q for mo in mos_o for q in pairs_of(mo, cams[oth], cams[ref], best)[1]]
            H, _ = ransac(src, dst, random.Random(0))
            if H is None:
                print("  --apply: no mapping")
                continue
            pts, k, ar = onto(cfg, ref, oth, H, mos_o, best)
            if len(pts) < 4:
                print(f"  --apply: {oth}'s points do not land on {ref}'s floor")
                continue
            calib[oth] = {"floor_points": pts, "distortion": k, "aspect": ar, "floor_at": best}
            calib.setdefault(ref, {})["floor_at"] = best
            shared.append((a, b))
            print(f"  --apply: {oth} onto {ref}'s floor ({len(pts)} points, by the boxes' {best}s"
                  + ("; its own calibration replaced" if cams[oth].get("floor_points") else "") + ")")
    out = args.out or os.path.dirname(os.path.abspath(os.path.join(args.boxes, "frames.json")))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "crosscam.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"-> {os.path.join(out, 'crosscam.json')}")
    if args.apply and calib:
        from autocalib import apply_config
        apply_config(args.config, calib, cfg["mct"].get("link_dist", 160), shared=shared)
        print(f"-> {args.config} (the old one: {args.config}.bak)")


if __name__ == "__main__":
    sys.exit(main())
