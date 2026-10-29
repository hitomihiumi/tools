#!/usr/bin/env python3
"""Which cow on the neighbour camera is she - by where she stands, learnt from your labelled boxes.

The cameras do not move: what surrounds a cow in one camera's frame (the walls, the drinker, the posts)
says where on the floor she is, and where that place lies in the neighbour's frame is one mapping
between the two pictures - a homography of the floor, the lens taken out with the config's distortion.
Your labels give it with no measuring: a cow of yours on two cameras at one moment is a pair of points.

    python herd/herd.py crosscam --config barn.toml --boxes boxes --labels labels.json [--out D] [--apply]

  1. the pairs, for each two cameras ([mct] neighbours, else every two with a cow of yours on both at once)
  2. each lens's distortion found from the pairs themselves (the one most pairs agree under; the config's
     may be off), then the mapping (RANSAC) and its check moment by moment: fitted without that moment, its
     cows on the first camera put onto the second and matched one to one by distance - the right cow how
     often, beside the chance of a guess; your cows that land on another of yours again and again (names
     swapped between the cameras?). Fitted twice, on the boxes' bottom centres and on their centres
  3. OUT/crosscam_<a>_<b>.jpg: where each of your cows was put on the second camera (hollow) against where
     you named her (filled), the line between them red when far
  4. OUT/crosscam.json; --apply (one of the two calibrated, floor_points; both: --onto names the one to
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
K_GRID = (0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6)    # lens distortions tried (common.undistort's k)


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
        named.setdefault(f["cam"], []).append((f["ts"], bx, file))
    out, used = [], set()
    for ta, ba, fa in sorted(named.get(a, []), key=lambda x: x[0]):
        best = min(((abs(ta - tb), i) for i, (tb, _, _) in enumerate(named.get(b, [])) if i not in used), default=None)
        if best and best[0] <= AT_ONCE_S:
            used.add(best[1])
            _, bb, fb = named[b][best[1]]
            out.append({"ts": ta, "a": ba, "b": bb, "fa": fa, "fb": fb})
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


def sym(H, A, B):
    """Each pair's miss, half on each camera's picture: a cow far from one camera is small on its picture and
    large on the other's, so a miss counted on one picture alone weighs her by where she happens to lie."""
    Hi = np.linalg.inv(H)
    return 0.5 * (np.hypot(*(apply_h(H, A) - B).T) + np.hypot(*(apply_h(Hi, B) - A).T))


def sym_matrix(H, A, B):
    """Every a against every b: the half-and-half miss."""
    Hi = np.linalg.inv(H)
    P, Q = apply_h(H, A), apply_h(Hi, B)
    return 0.5 * (np.hypot(P[:, None, 0] - B[None, :, 0], P[:, None, 1] - B[None, :, 1]) +
                  np.hypot(A[:, None, 0] - Q[None, :, 0], A[:, None, 1] - Q[None, :, 1]))


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
        if not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
            continue
        inl = sym(H, src, dst) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < MIN_PAIRS:
        return None, None
    H = homography(src[best], dst[best])
    inl = sym(H, src, dst) < thr
    if inl.sum() >= MIN_PAIRS:
        H = homography(src[inl], dst[inl])
    return H, inl


def fit_planes(src, dst, rng, most=3, iters=1000):
    """Up to `most` mappings, each the one most of the pairs left agree on (the bedding and the alley lie at
    other heights; a lens's bend the one plane cannot follow) -> {"Hs", "src", "lab"}: each pair's plane by
    its least miss (-1: none), what knn() reads a new point's plane from - the planes of the pairs nearest
    it on the first camera."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    left, Hs = np.arange(len(src)), []
    while len(Hs) < most and len(left) >= MIN_PAIRS * 2:
        H, inl = ransac(src[left], dst[left], rng, iters=iters)
        if H is None or inl.sum() < max(MIN_PAIRS * 2, 0.1 * len(src)):
            break
        Hs.append(H)
        left = left[~inl]
    if not Hs:
        return None
    E = np.stack([sym(H, src, dst) for H in Hs], 1)
    lab = np.where(E.min(1) < RANSAC_THR, E.argmin(1), -1)
    return {"Hs": Hs, "src": src, "lab": lab}


def knn(model, A, k=7):
    """The plane of each point of the first camera: most of its k nearest pairs' (one plane: 0)."""
    if len(model["Hs"]) == 1:
        return np.zeros(len(A), int)
    S, lab = model["src"][model["lab"] >= 0], model["lab"][model["lab"] >= 0]
    out = []
    for a in np.asarray(A, float):
        near = lab[np.argsort(np.hypot(*(S - a).T))[:k]]
        out.append(np.bincount(near, minlength=len(model["Hs"])).argmax() if len(near) else 0)
    return np.asarray(out, int)


def fit_model(src, dst, rng, planes=1, iters=1000):
    if planes == 1:
        H, inl = ransac(src, dst, rng, iters=iters)
        return None if H is None else {"Hs": [H], "src": np.asarray(src, float), "lab": np.zeros(len(src), int)}
    return fit_planes(src, dst, rng, planes, iters)


def costs(model, A, B):
    """Every a against every b, each a by its own plane's mapping."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    pl = knn(model, A)
    return np.stack([sym_matrix(model["Hs"][p], A[i:i + 1], B)[0] for i, p in enumerate(pl)])


def fit_lens(mos, cam_a, cam_b, where, keep=(), iters=300):
    """(k of a, k of b, pairs agreeing): the two lenses' distortion under which most pairs agree on one
    mapping. keep: cameras whose config value stays (a calibrated camera's floor_points depend on it)."""
    rng = random.Random(0)
    ka0, kb0 = cam_a.get("distortion") or 0.0, cam_b.get("distortion") or 0.0
    grid_a = [ka0] if cam_a["id"] in keep else sorted(set(K_GRID) | {ka0})
    grid_b = [kb0] if cam_b["id"] in keep else sorted(set(K_GRID) | {kb0})
    best = None
    for ka in grid_a:
        for kb in grid_b:
            ca, cb = dict(cam_a, distortion=ka), dict(cam_b, distortion=kb)
            src = [q for mo in mos for q in pairs_of(mo, ca, cb, where)[0]]
            dst = [q for mo in mos for q in pairs_of(mo, ca, cb, where)[1]]
            H, inl = ransac(src, dst, rng, iters=iters)
            if H is None:
                continue
            e = sym(H, np.asarray(src, float), np.asarray(dst, float))
            cost = float(np.minimum(e, RANSAC_THR) ** 2 @ np.ones(len(e))) / len(e)    # truncated: far pairs cap
            score = (-round(cost, 7), ka == ka0 and kb == kb0)
            if best is None or score > best[0]:
                best = (score, ka, kb, int(inl.sum()))
    if best is None:
        return ka0, kb0, 0
    return best[1], best[2], best[3]


def check(mos, cam_a, cam_b, where, seed=0, gate=GATE, planes=1, train=None):
    """Each moment matched by the mapping fitted without it (train given: fitted once, on another recording's
    moments) -> the counts and the misses by distance."""
    from scipy.optimize import linear_sum_assignment
    per = [pairs_of(mo, cam_a, cam_b, where) for mo in mos]
    n = collections.Counter()
    miss, chance, cows = [], [], collections.defaultdict(collections.Counter)
    lands = collections.defaultdict(collections.Counter)      # your cow -> your other cows she was put on
    partner = collections.defaultdict(collections.Counter)    # your cow -> what she was matched to, each moment
    fixed = None
    if train is not None:
        tr = [pairs_of(mo, cam_a, cam_b, where) for mo in train]
        fixed = fit_model([p for pa, _ in tr for p in pa], [p for _, pb in tr for p in pb], random.Random(seed), planes)
    for k, mo in enumerate(mos):
        src = [p for j, (pa, _) in enumerate(per) if j != k for p in pa]
        dst = [p for j, (_, pb) in enumerate(per) if j != k for p in pb]
        model = fixed if train is not None else fit_model(src, dst, random.Random(seed + k), planes, iters=500)
        if model is None:
            n["not_fitted"] += 1
            continue
        A = [point(box, cam_a, w, h, where) for _, box, w, h in mo["a"]]
        B = [point(box, cam_b, w, h, where) for _, box, w, h in mo["b"]]
        if not A or not B:
            continue
        D = costs(model, A, B)
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
            partner[cid][None if j is None else idb[j]] += 1
            if what == "wrong":
                lands[cid][idb[j]] += 1
    return {"cows": n["cows"], "right": n["right"], "wrong": n["wrong"], "none": n["none"],
            "moments_not_fitted": n["not_fitted"], "chance": float(np.mean(chance)) if chance else 0.0,
            "median_miss": float(np.median(miss)) if miss else None,
            "by_cow": {c: dict(v) for c, v in sorted(cows.items())}, **over_time(partner),
            "lands_on": {c: dict(v.most_common(3)) for c, v in sorted(lands.items())}}


def over_time(partner):
    """What the linker asks: over all the moments, is the cow she was matched to most often (her pair's
    tracks standing together longest) the right one? -> cows right / another / never matched."""
    right = wrong = never = 0
    for cid, cnt in partner.items():
        seen = [(k, c) for c, k in cnt.items() if c is not None]
        if not seen:
            never += 1
        elif max(seen)[1] == cid:
            right += 1
        else:
            wrong += 1
    return {"over_time": {"cows": len(partner), "right": right, "wrong": wrong, "never": never}}


def picture(path, boxes_dir, mos, cam_a, cam_b, model, where):
    """Side by side: the first camera's frame with your cows' points, the second's with where the mapping put
    each (hollow) and where you named her (filled), joined by a line - green near, orange, red far."""
    from PIL import Image, ImageDraw
    W = 960
    mid = mos[len(mos) // 2]

    def frame(file):
        try:
            im = Image.open(os.path.join(boxes_dir, file)).convert("RGB")
        except OSError:
            im = Image.new("RGB", (1920, 1080), (60, 60, 60))
        return im.resize((W, round(W * im.height / im.width)))
    A, B = frame(mid["fa"]), frame(mid["fb"])
    canvas = Image.new("RGB", (2 * W + 10, max(A.height, B.height) + 28), (24, 24, 24))
    canvas.paste(A, (0, 28))
    canvas.paste(B, (W + 10, 28))
    d = ImageDraw.Draw(canvas)
    d.text((6, 6), f"{cam_a['id']}: your cows ({where} of the box)", fill=(240, 240, 240))
    d.text((W + 16, 6), f"{cam_b['id']}: put here (hollow) / named here (filled)", fill=(240, 240, 240))

    def back(q, cam):                                   # straightened point -> pixel of the picture
        from autocalib import distort
        k, ar = lens(cam, 1920, 1080)
        u, v = distort(q[0] / ar, q[1], k, ar)
        return float(u), float(v)
    said = set()
    for mo in mos:
        pa, pb = pairs_of(mo, cam_a, cam_b, where)
        if not pa:
            continue
        ib = {i for i, _, _, _ in mo["b"]}
        ids = [i for i, _, _, _ in mo["a"] if i in ib]             # pairs_of's order
        pl = knn(model, pa)
        P = np.stack([apply_h(model["Hs"][p], [q])[0] for p, q in zip(pl, pa)])
        for cid, qa, qp, qb in zip(ids, pa, P, pb):
            e = float(np.hypot(*(qp - np.asarray(qb))))
            col = (60, 200, 90) if e < 0.03 else (240, 160, 40) if e < 0.08 else (230, 60, 60)
            ua, va = back(qa, cam_a)
            up, vp = back(qp, cam_b)
            ub, vb = back(qb, cam_b)
            if not all(np.isfinite([ua, va, up, vp, ub, vb])):
                continue
            x, y = ua * W, 28 + va * A.height
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=col)
            xp, yp, xb, yb = W + 10 + up * W, 28 + vp * B.height, W + 10 + ub * W, 28 + vb * B.height
            d.line([xp, yp, xb, yb], fill=col, width=1)
            d.ellipse([xp - 4, yp - 4, xp + 4, yp + 4], outline=col, width=2)
            d.ellipse([xb - 3, yb - 3, xb + 3, yb + 3], fill=col)
            if e >= 0.08 and cid not in said:                 # a far one: your name by her, once
                said.add(cid)
                d.text((xb + 5, yb - 6), cid, fill=(255, 255, 255))
                d.text((x + 5, y - 6), cid, fill=(255, 255, 255))
    canvas.save(path, quality=90)
    return path


def box_feats(box, w, h):
    """A box -> what the learnt mapping reads: where it is (to the third power: a lens's bend, the alley's other
    level), and how big and how tall (a standing cow's centre is higher than a lying one's)."""
    u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    bw, bh = box[2] - box[0], box[3] - box[1]
    return [1.0, u, v, u * u, u * v, v * v, u ** 3, u * u * v, u * v * v, v ** 3, bw, bh, bh * u, bh * v, bw * bh,
            bh / max(bw, 1e-3) * 0.1]


def centre(box, w, h):
    ar = w / h if h else 16 / 9
    return ((box[0] + box[2]) / 2 * ar, (box[1] + box[3]) / 2)


def fit_poly(F, T, lam=1e-4, rounds=3):
    """Ridge least squares F -> T (each pair's other box's centre), refitted without the pairs it misses by
    far more than most (a name wrong on one camera must not bend the mapping)."""
    F, T = np.asarray(F, float), np.asarray(T, float)
    keep = np.ones(len(F), bool)
    W = None
    for _ in range(rounds):
        A = F[keep]
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ T[keep])
        e = np.hypot(*(F @ W - T).T)
        keep = e <= max(3 * np.median(e[keep]), RANSAC_THR)
        if keep.sum() < F.shape[1] + 4:
            break
    return W


class Local:
    """Where a box lands on the other picture from the pairs nearest it on its own (k of them, an affine map of
    where and how tall fitted on those, nearer ones weighing more): a step between the bedding and the alley
    stays a step, which one smooth mapping would blur."""

    def __init__(self, F, T, k=12):
        self.F, self.T, self.k = np.asarray(F, float), np.asarray(T, float), k
        self.P = self.F[:, [1, 2, 11]] * [1.0, 1.0, 0.5]          # u, v, height: what "near" is measured on

    def map(self, Fq):
        Fq = np.asarray(Fq, float)
        out = []
        for f in Fq:
            q = f[[1, 2, 11]] * [1.0, 1.0, 0.5]
            d = np.sqrt(((self.P - q) ** 2).sum(1))
            idx = np.argsort(d)[:self.k]
            w = 1.0 / (d[idx] + 0.02)
            A = self.F[idx][:, [0, 1, 2, 11]]
            Aw = A * w[:, None]
            X = np.linalg.lstsq(Aw.T @ A + 1e-4 * np.eye(4), Aw.T @ self.T[idx], rcond=None)[0]
            out.append(f[[0, 1, 2, 11]] @ X)
        return np.asarray(out)


def learn(mos, kind="poly"):
    """The learnt mapping both ways from moments' pairs (named the same on both) -> (a -> b, b -> a) or None."""
    Fa, Fb, Ta, Tb = [], [], [], []
    for mo in mos:
        ib = {i: (box, w, h) for i, box, w, h in mo["b"]}
        for i, box, w, h in mo["a"]:
            if i in ib:
                bb, wb, hb = ib[i]
                Fa.append(box_feats(box, w, h))
                Tb.append(centre(bb, wb, hb))
                Fb.append(box_feats(bb, wb, hb))
                Ta.append(centre(box, w, h))
    if not Fa or len(Fa) < 3 * len(Fa[0]):                     # too few pairs for its 16 numbers a way
        return None
    if kind == "local":
        return Local(Fa, Tb), Local(Fb, Ta)
    return fit_poly(Fa, Tb), fit_poly(Fb, Ta)


def learnt_costs(model, mo):
    """Every a against every b at one moment: each put on the other's picture by the learnt mapping, the miss
    half on each picture."""
    Wab, Wba = model
    Fa = np.array([box_feats(box, w, h) for _, box, w, h in mo["a"]])
    Fb = np.array([box_feats(box, w, h) for _, box, w, h in mo["b"]])
    Ca = np.array([centre(box, w, h) for _, box, w, h in mo["a"]])
    Cb = np.array([centre(box, w, h) for _, box, w, h in mo["b"]])
    P, Q = (Wab.map(Fa), Wba.map(Fb)) if isinstance(Wab, Local) else (Fa @ Wab, Fb @ Wba)
    return 0.5 * (np.hypot(P[:, None, 0] - Cb[None, :, 0], P[:, None, 1] - Cb[None, :, 1]) +
                  np.hypot(Ca[:, None, 0] - Q[None, :, 0], Ca[:, None, 1] - Q[None, :, 1]))


def check_learnt(mos, gate=GATE, train=None, kind="poly"):
    """The learnt mapping matched moment by moment: fitted without that moment (or, train given, on another
    recording's moments only) -> the same counts as check()."""
    from scipy.optimize import linear_sum_assignment
    n, cows = collections.Counter(), collections.defaultdict(collections.Counter)
    partner = collections.defaultdict(collections.Counter)
    chance, fixed = [], (learn(train, kind) if train is not None else None)
    for k, mo in enumerate(mos):
        model = fixed if train is not None else learn(mos[:k] + mos[k + 1:], kind)
        if model is None or not mo["a"] or not mo["b"]:
            n["not_fitted"] += model is None
            continue
        D = learnt_costs(model, mo)
        r, c = linear_sum_assignment(D)
        got = {i: j for i, j in zip(r, c) if D[i, j] <= gate}
        idb = [i for i, _, _, _ in mo["b"]]
        for i, (cid, _, _, _) in enumerate(mo["a"]):
            if cid not in idb:
                continue
            n["cows"] += 1
            chance.append(1.0 / len(idb))
            j = got.get(i)
            what = "none" if j is None else ("right" if idb[j] == cid else "wrong")
            n[what] += 1
            cows[cid][what] += 1
            partner[cid][None if j is None else idb[j]] += 1
    return {"cows": n["cows"], "right": n["right"], "wrong": n["wrong"], "none": n["none"],
            "moments_not_fitted": n["not_fitted"], "chance": float(np.mean(chance)) if chance else 0.0,
            "by_cow": {c: dict(v) for c, v in sorted(cows.items())}, **over_time(partner)}


def say(name, y):
    if not y["cows"]:
        return f"  {name}: no mapping"
    t = y.get("over_time")
    return (f"  {name}: the right cow {y['right'] / y['cows']:.0%} (another {y['wrong'] / y['cows']:.0%}, none near "
            f"{y['none'] / y['cows']:.0%}) of {y['cows']}" +
            (f"; over all the moments the right partner for {t['right']} of {t['cows']} cows (another "
             f"{t['wrong']}, none {t['never']})" if t else ""))


def onto(cams, ref, oth, H, mos, where):
    """oth's floor_points on ref's floor: points of oth where your cows stood, through H (oth -> ref) onto ref's
    frame and ref's calibration -> ([[u, v, X, Y]] of oth, as clicked, its lens). mos: oth's moments with ref
    ("a" oth, "b" ref). With where = "centre" the points are the boxes' centres, read on ref's floor as
    its centres are (floor_at = "centre" on both)."""
    from autocalib import camera_floor, distort
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
    p.add_argument("--also", nargs=2, default=None, metavar=("BOXES", "LABELS"),
                   help="another recording's labelled boxes: the learnt mapping fitted on one and checked on the "
                        "other, both ways - whether it holds beyond the minutes it was learnt on")
    p.add_argument("--planes", type=int, default=3,
                   help="also try this many floor planes, each pair on the one it fits (1: one plane only)")
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
        print(f"[crosscam] {a} -> {b}: {npairs} pairs of your cows at {len(mos)} moments")
        fitted = {}
        for where in ("bottom", "centre"):
            ka, kb, agree = fit_lens(mos, cams[a], cams[b], where)
            ca, cb = dict(cams[a], distortion=ka), dict(cams[b], distortion=kb)
            fitted[where] = (ca, cb)
            src = [q for mo in mos for q in pairs_of(mo, ca, cb, where)[0]]
            dst = [q for mo in mos for q in pairs_of(mo, ca, cb, where)[1]]
            H, inl = ransac(src, dst, random.Random(0))
            r[where] = dict(check(mos, ca, cb, where, gate=args.gate), inliers=int(inl.sum()) if inl is not None else 0,
                            H=H.tolist() if H is not None else None, lens={a: ka, b: kb})
            many = fit_planes(src, dst, random.Random(0), args.planes)
            if args.planes > 1 and many is not None and len(many["Hs"]) > 1:
                r[where + "_planes"] = dict(check(mos, ca, cb, where, gate=args.gate, planes=args.planes),
                                            planes=len(many["Hs"]), Hs=[h.tolist() for h in many["Hs"]],
                                            share=[float(np.mean(many["lab"] == i)) for i in range(len(many["Hs"]))])
                fitted[where + "_planes"] = (ca, cb, many)
        best = max(("bottom", "centre"), key=lambda w: r[w]["right"])
        r["best"] = best
        report[f"{a}->{b}"] = r
        x = r[best]
        k0 = {c: cams[c].get("distortion") or 0.0 for c in (a, b)}
        for where in ("bottom", "centre"):
            y = r[where]
            if not y["cows"]:
                print(f"  {where:6}: no mapping")
                continue
            z = r.get(where + "_planes")
            ls = ", ".join(f"{c} {y['lens'][c]:g}" + ("" if y["lens"][c] == k0[c] else f" (config {k0[c]:g})")
                           for c in (a, b))
            print(f"  {where:6}: the right cow {y['right'] / y['cows']:.0%} (another {y['wrong'] / y['cows']:.0%}, none near "
                  f"{y['none'] / y['cows']:.0%}) of {y['cows']}, a guess {y['chance']:.0%}; median miss "
                  f"{y['median_miss']:.1%} of the frame height; {y['inliers']} of {npairs} pairs agree; lens {ls}; "
                  f"over all the moments the right partner for {y['over_time']['right']} of {y['over_time']['cows']} "
                  f"cows (another {y['over_time']['wrong']})")
            if z and z["cows"]:
                print(f"  {'':6}  in {z['planes']} planes (" + ", ".join(f"{x:.0%}" for x in z["share"]) +
                      f" of the pairs): the right cow {z['right'] / z['cows']:.0%} (another {z['wrong'] / z['cows']:.0%}, "
                      f"none near {z['none'] / z['cows']:.0%}); over all the moments the right partner for "
                      f"{z['over_time']['right']} of {z['over_time']['cows']} cows (another {z['over_time']['wrong']})")
        bad = [(c, v) for c, v in x["by_cow"].items() if v.get("right", 0) < sum(v.values())]
        if bad:
            print("  cows it missed (" + best + "): " + ", ".join(
                f"{c} {v.get('right', 0)}/{sum(v.values())}" for c, v in sorted(bad, key=lambda kv: kv[1].get('right', 0) - sum(kv[1].values()))[:12]))
        swapped = [(c, o, k, sum(x["by_cow"][c].values())) for c, v in x["lands_on"].items() for o, k in v.items()
                   if k >= max(3, 0.3 * sum(x["by_cow"][c].values()))]
        if swapped:
            print("  put again and again where another cow of yours is (one cow under two names, or two swapped "
                  "between the cameras? boxes.html / ids.html): " + ", ".join(
                      f"your {c} on {b} where your {o} is ({k} of {m})" for c, o, k, m in swapped[:10]))
        out_dir = args.out or os.path.dirname(os.path.abspath(os.path.join(args.boxes, "frames.json")))
        os.makedirs(out_dir, exist_ok=True)
        if x.get("H"):
            many = r.get(best + "_planes")
            if many and many["right"] > x["right"]:                # the picture of the better of the two
                ca, cb, model = fitted[best + "_planes"]
            else:
                (ca, cb), model = fitted[best], {"Hs": [np.asarray(x["H"])]}
            pic = picture(os.path.join(out_dir, f"crosscam_{a}_{b}.jpg"), args.boxes, mos, ca, cb, model, best)
            print(f"  where each was put: {pic}")
        # the learnt mapping: no plane, the pairs themselves (where, how big, how tall the box)
        r["learnt"] = check_learnt(mos, gate=args.gate)
        r["learnt_local"] = check_learnt(mos, gate=args.gate, kind="local")
        print(say("learnt from the boxes, one smooth mapping (where, size, height)", r["learnt"]))
        print(say("learnt from the boxes, from the nearest pairs (steps kept)", r["learnt_local"]))
        if args.also:
            fr2 = {f["file"]: f for f in json.load(open(os.path.join(args.also[0], "frames.json"), encoding="utf-8"))["frames"]}
            mos2 = moments(cfg, fr2, read_labels(args.also[1]), a, b)
            if mos2:
                ca, cb = fitted[best][:2]
                for nm, pl in (("one plane", 1), (f"{args.planes} planes", args.planes)):
                    x1 = check(mos2, ca, cb, best, gate=args.gate, planes=pl, train=mos)
                    x2 = check(mos, ca, cb, best, gate=args.gate, planes=pl, train=mos2)
                    r[f"{best}_{pl}_other_recording"], r[f"{best}_{pl}_from_other_recording"] = x1, x2
                    print(say(f"{nm} ({best}s): learnt here, checked on the other recording", x1))
                    print(say(f"{nm} ({best}s): learnt on the other, checked here", x2))
                for kind, nm in (("poly", "smooth"), ("local", "nearest pairs")):
                    x1 = check_learnt(mos2, gate=args.gate, train=mos, kind=kind)
                    x2 = check_learnt(mos, gate=args.gate, train=mos2, kind=kind)
                    sfx = "" if kind == "poly" else "_local"
                    r["learnt_other_recording" + sfx], r["learnt_from_other_recording" + sfx] = x1, x2
                    print(say(f"{nm}: learnt here, checked on the other recording ({len(mos2)} moments)", x1))
                    print(say(f"{nm}: learnt on the other, checked here", x2))
            else:
                print(f"  --also: no moment with your cows on both {a} and {b} there")
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
            ko, _, _ = fit_lens(mos_o, cams[oth], cams[ref], best, keep=(ref,))   # ref's lens is its calibration's
            use = dict(cams, **{oth: dict(cams[oth], distortion=ko)})
            src = [q for mo in mos_o for q in pairs_of(mo, use[oth], use[ref], best)[0]]
            dst = [q for mo in mos_o for q in pairs_of(mo, use[oth], use[ref], best)[1]]
            H, _ = ransac(src, dst, random.Random(0))
            if H is None:
                print("  --apply: no mapping")
                continue
            pts, k, ar = onto(use, ref, oth, H, mos_o, best)
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
