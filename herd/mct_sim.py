#!/usr/bin/env python3
"""A barn made up, where the truth is known: the cross-camera joining rules (mct.TrackletLinker) compared on it.

MmCows needs a pod and the barn's own labels a person's hours; here the cows, the cameras and the floor
mapping's errors are drawn, so a rule is tried in minutes on any computer, and on the barn's case: cameras
facing each other over a shared strip, cows lying side by side in rows of stalls for long stretches, each
camera's floor mapping bent by tens of cm and wrong in a part of the seconds (a cow's two sides, a wrong
posture), the look weak across the two views. The defaults put a cow's true partner nearest on the other
camera in ~50-70% of the seconds (the barn's cam1/cam2: ~70%).

    python herd/mct_sim.py                       # the main rules, 30 min, 3 seeds, one process a core
    python herd/mct_sim.py --minutes 45 --cows 40 --bend 80 --wrong 0.3 --seeds 0 1 --rules "share (default)" llr

Per rule: IDF1 over both cameras (the reports' number), a cow seen by both at once on one track, track
purity, wrong joins (two cows on one track at one second), and the share of seconds a cow's true partner
was the nearest on the other camera. For the llr rules also the night's step (mct.cow_merges): each camera
names its tracks by their cow (a perfect gallery of its own side), and the cows whose two names are made
one by place - at the config's cow_merge_llr / cow_merge_margin and a sweep of both.

The pen: X 0-2400 cm, Y 0-1200 cm. Camera a sees X 0-1500, camera b (facing it) X 900-2400; a counts
X < 1200, b the rest (masks share no floor), both track over the whole of their picture.
"""

from __future__ import annotations

import argparse
import collections
import copy
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import DEFAULTS  # noqa: E402

PEN = (2400.0, 1200.0)
VIEW = {"a": (0.0, 1500.0), "b": (900.0, 2400.0)}
SPLIT = 1200.0                     # the masks' border
BOX = (0.10, 0.12)                 # a cow's box, frame widths / heights


def to_frame(cam, x, y):
    """Floor cm -> 0-1 of the camera's frame (b faces a: both axes turned)."""
    x0, x1 = VIEW[cam]
    u, v = (x - x0) / (x1 - x0), y / PEN[1]
    return (u, v) if cam == "a" else (1 - u, 1 - v)


def floor_points(cam):
    return [[u, v, x, y] for x, y in ((VIEW[cam][0], 0), (VIEW[cam][1], 0), (VIEW[cam][1], PEN[1]), (VIEW[cam][0], PEN[1]))
            for u, v in [to_frame(cam, x, y)]]


def masks():
    ua, _ = to_frame("a", SPLIT, 0)
    ub, _ = to_frame("b", SPLIT, 0)
    return {"a": [[0, 0], [ua, 0], [ua, 1], [0, 1]], "b": [[0, 0], [ub, 0], [ub, 1], [0, 1]]}


def cows_path(n, seconds, rng, lying_share=0.6, speed_x=1.0):
    """[cow][second] -> (x, y, lying): cows lying in rows of stalls (1.2 m apart) for long stretches, standing,
    walking from place to place at 0.5-0.9 m/s."""
    stalls = [(x, y) for y in (150.0, 1050.0) for x in np.arange(100.0, PEN[0] - 50, 120.0)]
    free = set(range(len(stalls)))
    pos = np.zeros((n, seconds, 3))
    for c in range(n):
        s, t = rng.integers(len(stalls)), 0
        x, y = rng.uniform(100, PEN[0] - 100), rng.uniform(300, 900)
        stall = None
        while t < seconds:
            if rng.random() < lying_share and free:
                stall = rng.choice(sorted(free))
                free.discard(stall)
                tx, ty = stalls[stall]
                lie = True
                dur = int(rng.uniform(600, 2400))
            else:
                tx, ty = rng.uniform(100, PEN[0] - 100), rng.uniform(300, 900)
                lie = False
                dur = int(rng.uniform(60, 400))
            speed = rng.uniform(50, 90) * speed_x
            while t < seconds and np.hypot(tx - x, ty - y) > 1:
                step = min(speed, np.hypot(tx - x, ty - y))
                a = np.arctan2(ty - y, tx - x)
                x, y = x + step * np.cos(a), y + step * np.sin(a)
                pos[c, t] = (x, y, 0)
                t += 1
            for _ in range(dur):
                if t >= seconds:
                    break
                pos[c, t] = (x, y, lie)
                t += 1
            if stall is not None:
                free.add(stall)
                stall = None
        del s
    return pos


def bend_field(cam, bend, rng):
    """A camera's floor mapping off by up to `bend` cm, smoothly and unevenly over the floor."""
    ph = rng.uniform(0, 2 * np.pi, 4)
    return lambda x, y: (bend * np.sin(x / 400 + ph[0]) * np.cos(y / 500 + ph[1]),
                         bend * np.cos(x / 350 + ph[2]) * np.sin(y / 450 + ph[3]))


def simulate(n=40, minutes=30, look=0.35, seed=0, miss=0.05, weak=0.0, speed=1.0, lag=0.0):
    """-> dets {cam: {ts: [(box, emb, p_lying, score)]}}, gt {cam: {ts: [(cow, box)]}}, the cameras' configs, the
    paths. The boxes sit where the cows are (a few cm of jitter): within a camera boxes are reliable; the floor
    mapping's errors are put in by noisy_floor. weak > 0: a lying cow fades for the detector - each second that
    chance of 30-300 s in which her box comes at score 0.3 (under the threshold, 0.5) in 70% of the seconds and
    not at all in the rest; a missed box is then low half the time as well. lag: camera b's frames stamped that
    many s late (its stream's delay): at its second t it shows the cows where they were at t - lag."""
    rng = np.random.default_rng(seed)
    T = int(minutes * 60)
    pos = cows_path(n, T, rng, speed_x=speed)
    dim = 32
    unit = lambda v: v / np.linalg.norm(v)
    cow_vec = rng.normal(size=(n, dim))
    look_of = {cam: np.stack([unit(look * unit(cow_vec[c]) + (1 - look) * unit(rng.normal(size=dim))) for c in range(n)])
               for cam in VIEW}             # a cow's look from one side: a little of her, mostly the view
    dets = {cam: {} for cam in VIEW}
    gt = {cam: {} for cam in VIEW}
    faded = {(cam, c): -1 for cam in VIEW for c in range(n)}     # until when the detector is unsure of her
    for t in range(T):
        for cam in VIEW:
            x0, x1 = VIEW[cam]
            ds, gs = [], []
            for c in range(n):
                x, y, lie = pos[c, t]
                if cam == "b" and lag:
                    k = t - lag
                    i0 = int(np.clip(np.floor(k), 0, T - 1))
                    i1, w = min(i0 + 1, T - 1), float(np.clip(k - i0, 0, 1))
                    x, y = (1 - w) * pos[c, i0, :2] + w * pos[c, i1, :2]
                    lie = pos[c, i0, 2]
                if not x0 + 30 <= x <= x1 - 30:
                    continue
                score = 0.9
                if weak and lie and faded[(cam, c)] < t and rng.random() < weak:
                    faded[(cam, c)] = t + int(rng.uniform(30, 300))
                if weak and lie and faded[(cam, c)] >= t:
                    if rng.random() < 0.3:
                        continue
                    score = 0.3
                elif rng.random() < miss:
                    if not weak or rng.random() < 0.5:
                        continue
                    score = 0.3
                u, v = to_frame(cam, x + rng.normal(0, 5), y + rng.normal(0, 5))
                w, h = BOX
                box = [u - w / 2, v - h, u + w / 2, v]
                emb = unit(look_of[cam][c] + rng.normal(0, 0.15, dim))
                ds.append((box, emb.tolist(), 0.9 if lie else 0.1, score))
                gs.append((c, box))
            dets[cam][t] = ds
            gt[cam][t] = gs
    m = masks()
    cams = [{"id": cam, "mask": m[cam], "floor_points": floor_points(cam)} for cam in VIEW]
    return dets, gt, cams, pos


def noisy_floor(f, cam, bend, wrong, jitter, seed):
    """A camera's floor mapping f as the barn's is: bent by up to `bend` cm (smooth, uneven), `jitter` cm of noise
    a second, and in `wrong` of the seconds 0.8-2 m off (a cow's other side, a wrong height, half hidden) -
    the same for one box whichever height is asked."""
    field = bend_field(cam, bend, np.random.default_rng(seed * 7 + (cam == "b")))

    def g(box, p_lying=0.5):
        x, y = f(box, p_lying)
        r = np.random.default_rng(zlib.crc32(repr((seed, cam) + tuple(round(v, 6) for v in box)).encode()))
        dx, dy = field(x, y)
        ex, ey = r.normal(0, jitter, 2)
        if r.random() < wrong:
            a, d = r.uniform(0, 2 * np.pi), r.uniform(80, 200)
            ex, ey = ex + d * np.cos(a), ey + d * np.sin(a)
        return (x + dx + ex, y + dy + ey)
    return g


def nearest_right(gt, floors):
    """The share of seconds a cow seen by both cameras has her own box as the other camera's nearest on the floor."""
    right = total = 0
    for t in gt["a"]:
        A = [(c, floors["a"](b)) for c, b in gt["a"][t]]
        B = [(c, floors["b"](b)) for c, b in gt["b"].get(t, [])]
        cb = {c for c, _ in B}
        for c, p in A:
            if c not in cb:
                continue
            d = [(np.hypot(p[0] - q[0], p[1] - q[1]), c2) for c2, q in B]
            total += 1
            right += min(d)[1] == c
    return right / max(1, total)


def floors_of(cams, bend, wrong, jitter, seed):
    from common import floor_mapper
    return {c["id"]: noisy_floor(floor_mapper(c, DEFAULTS["mct"]), c["id"], bend, wrong, jitter, seed) for c in cams}


def run(dets, gt, cams, floors, over, label_every=5):
    """One rule over the whole recording, as the reports see it (every track under the one it was joined to),
    scored on every label_every-th second against the true cows."""
    import mct
    import mct_eval
    cfg = copy.deepcopy(DEFAULTS)
    cfg["cameras"] = copy.deepcopy(cams)
    over = dict(over)
    cfg["tracker"].update(over.pop("_tracker", {}))
    cfg["mct"].update(reacquire_sim=0.9, **over)
    low_on = cfg["tracker"]["low_score"] > 0
    co = mct.CoSeen(cfg["identity"]["cannot_link_min_s"], cfg["identity"]["cannot_link_max_iou"])
    lk = mct.TrackletLinker(cfg)
    lk.floor.update(floors)
    got, links = {}, set()
    for ts in sorted({t for per in dets.values() for t in per}):
        per, where = {}, {}
        for c in dets:
            # the detector at its threshold (0.5); with low_score its unsure boxes go in too, marked
            keep = [j for j, d in enumerate(dets[c].get(ts, [])) if low_on or d[3] >= 0.5]
            per[c] = [{"box": dets[c][ts][j][0], "emb": dets[c][ts][j][1], "p_lying": dets[c][ts][j][2],
                       "low": dets[c][ts][j][3] < 0.5} for j in keep]
            where[c] = keep
        for c, ids in lk.update_all(float(ts), per).items():
            for j, g in zip(where[c], ids):
                if g is not None:
                    got[(c, ts, j)] = g
            kept = [(g, d["box"]) for g, d in zip(ids, per[c]) if g is not None]
            links.update(co.add([g for g, _ in kept], [b for _, b in kept]))
    raw = dict(got)
    got = {k: lk.root(g) for k, g in got.items()}
    links = {(min(lk.root(a), lk.root(b)), max(lk.root(a), lk.root(b))) for a, b in links if lk.root(a) != lk.root(b)}
    lab = {cam: {t: v for t, v in per.items() if t % label_every == 0} for cam, per in gt.items()}
    gt_map = {(cam, t, j): c for cam, per in lab.items() for t, cows in per.items() for j, (c, _) in enumerate(cows)}
    n = sum(len(v) for per in lab.values() for v in per.values())
    r, _ = mct_eval.score_run(got, links, gt_map, lab, dets, n, n)
    if lk.ev:
        r["names"] = names_check(lk, raw, gt)
    r["clocks"] = {f"{a}-{b}": v for (a, b), v in lk.clocks().items()}
    return r, dict(lk.stats)


def names_check(lk, raw, gt):
    """Each camera names its tracks by their cow, as a gallery of that one side would (the name: cow + camera);
    mct.cow_merges, from the linker's evidence, then puts one cow's two names together. -> how many cows seen by
    both cameras got one name, and how many names went to another cow."""
    import mct
    votes = collections.defaultdict(collections.Counter)
    for (cam, t, j), g in raw.items():
        votes[g][f"{gt[cam][t][j][0]}{cam}"] += 1
    track_cow = {g: v.most_common(1)[0][0] for g, v in votes.items()}
    evidence = {(a, b): v for a, per in lk.ev.items() for b, v in per.items() if a < b}
    apart = set()
    for cam, per in gt.items():
        for t, cows in per.items():
            if t % 5 == 0:
                ns = sorted({f"{c}{cam}" for c, _ in cows})
                apart |= {(x, y) for i, x in enumerate(ns) for y in ns[i + 1:]}
    size = collections.Counter(track_cow.values())
    both = {n[:-1] for n in track_cow.values() if n.endswith("a")} & {n[:-1] for n in track_cow.values() if n.endswith("b")}
    out = {"cows_on_both": len(both)}
    for min_llr, margin in SWEEP:
        group = dict(mct.cow_merges(evidence, track_cow, apart, min_llr, margin, size=size))
        top = lambda x: top(group[x]) if x in group else x
        right = sum(top(f"{c}a") == top(f"{c}b") for c in both)
        wrong = sum(top(old)[:-1] != old[:-1] for old in group)
        out[f"{min_llr:g}/{margin:g}"] = (right, wrong)
    d = f"{DEFAULTS['mct']['cow_merge_llr']:g}/{DEFAULTS['mct']['cow_merge_margin']:g}"
    out["one_name"], out["wrong_merges"] = out[d] if d in out else (None, None)
    return out


SWEEP = [(60.0, 30.0), (100.0, 50.0), (150.0, 75.0), (200.0, 100.0), (300.0, 150.0)]


RULES = [
    ("share (default)", {"mode": "link"}),
    ("vote2", {"mode": "link", "link_vote": 2.0, "link_margin": 0.0}),
    ("vote2+look300", {"mode": "link", "link_vote": 2.0, "link_margin": 0.0, "link_look_cm": 300.0}),
    ("llr", {"mode": "link", "link_rule": "llr"}),
    ("llr+look", {"mode": "link", "link_rule": "llr", "link_look_cm": 300.0}),
    ("llr+self-calib", {"mode": "link", "link_rule": "llr", "link_learn_offset": True}),
    ("llr+look+self-calib", {"mode": "link", "link_rule": "llr", "link_look_cm": 300.0, "link_learn_offset": True}),
    # the camera's own tracker ([tracker]): Kalman, ByteTrack's second round for unsure boxes (--weak > 0)
    ("llr+kalman", {"mode": "link", "link_rule": "llr", "_tracker": {"kalman": True}}),
    ("llr+low", {"mode": "link", "link_rule": "llr", "_tracker": {"low_score": 0.2}}),
    ("llr+kalman+low", {"mode": "link", "link_rule": "llr", "_tracker": {"kalman": True, "low_score": 0.2}}),
    # a library's tracker per camera instead of this one (libtrack.py: Roboflow's trackers), unsure boxes handed in
    ("llr+bytetrack", {"mode": "link", "link_rule": "llr", "_tracker": {"kind": "bytetrack", "low_score": 0.2}}),
    ("llr+botsort", {"mode": "link", "link_rule": "llr", "_tracker": {"kind": "botsort", "low_score": 0.2}}),
    ("llr+ocsort", {"mode": "link", "link_rule": "llr", "_tracker": {"kind": "ocsort", "low_score": 0.2}}),
    ("llr+bytetrack-300s", {"mode": "link", "link_rule": "llr",
                            "_tracker": {"kind": "bytetrack", "low_score": 0.2, "lib_lost_s": 300.0}}),
]


DEFAULT_RULES = ["share (default)", "vote2", "llr", "llr+self-calib"]


def one(job):
    """(seed, rule name, args) -> (seed, name, result, stats, the share of seconds the true partner was nearest)."""
    import contextlib
    import io
    seed, name, a = job
    dets, gt, cams, _ = simulate(a["cows"], a["minutes"], look=a["look"], seed=seed, weak=a["weak"], speed=a["speed"],
                                 lag=a["lag"])
    floors = floors_of(cams, a["bend"], a["wrong"], a["jitter"], seed)
    with contextlib.redirect_stdout(io.StringIO()):          # the linker's notes, once a place
        r, stats = run(dets, gt, cams, floors, dict(RULES)[name])
    return seed, name, r, stats, nearest_right(gt, floors)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cows", type=int, default=40)
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--bend", type=float, default=60.0, help="cm the floor mapping is off by, at most (smooth)")
    ap.add_argument("--wrong", type=float, default=0.15, help="share of seconds a camera's floor point lands 0.8-2 m off")
    ap.add_argument("--jitter", type=float, default=20.0, help="cm of noise on every floor point")
    ap.add_argument("--look", type=float, default=0.35, help="how much of a cow's look is hers on both sides (0-1)")
    ap.add_argument("--weak", type=float, default=0.0,
                    help="each second, the chance a lying cow fades for the detector for 30-300 s (e.g. 0.003)")
    ap.add_argument("--speed", type=float, default=1.0, help="walking cows this many times faster (0.5-0.9 m/s x)")
    ap.add_argument("--lag", type=float, default=0.0, help="camera b's frames stamped this many s late (its stream's "
                    "delay, no time_offset_s); the linker should measure it")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rules", nargs="*", default=DEFAULT_RULES, help=f"of: {', '.join(n for n, _ in RULES)}")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = ap.parse_args(argv)
    import concurrent.futures as cf
    a = {k: getattr(args, k) for k in ("cows", "minutes", "look", "bend", "wrong", "jitter", "weak", "speed", "lag")}
    jobs = [(seed, name, a) for seed in args.seeds for name in args.rules]
    res = collections.defaultdict(list)
    with cf.ProcessPoolExecutor(max(1, args.jobs)) as ex:
        for seed, name, r, stats, near in ex.map(one, jobs):
            res[name].append(r)
            print(f"[sim] seed {seed} (true partner nearest in {near:.0%} of the seconds) {name:22s} "
                  f"IDF1 {r['idf1_global']:.3f} (camera {r['idf1_camera']:.3f})  one track on both {r['cross_cam_same_track'] or 0:.1%}  "
                  f"purity {r['track_purity']:.1%}  wrong {r['false_merges']}  joins {stats.get('linked', 0)} "
                  f"refused {stats.get('link_refused', 0)} unlinked {stats.get('unlinked', 0)}"
                  + (f"  names: {r['names']['one_name']}/{r['names']['cows_on_both']} cows one name, "
                     f"{r['names']['wrong_merges']} wrong" if "names" in r else "")
                  + "".join(f"  clocks {k}: b {v[0]:+.1f} s late ({v[1]} walks, {v[2]:.0f} cm)" for k, v in r.get("clocks", {}).items()),
                  flush=True)
    print(f"\n{args.cows} cows, {args.minutes:g} min, floor bent {args.bend:g} cm, {args.wrong:.0%} of seconds wrong, "
          f"seeds {args.seeds}\n")
    print("| rule | IDF1 both cameras | IDF1 within a camera | cow on both at once, one track | track purity "
          "| wrong (two cows, one track) | cows with one name (cow_merges) |")
    print("|---|---|---|---|---|---|---|")
    for name in args.rules:
        rs = res[name]
        mean = lambda k: float(np.mean([r[k] or 0 for r in rs]))
        nm = [r["names"] for r in rs if "names" in r]
        names = (f"{sum(x['one_name'] for x in nm)}/{sum(x['cows_on_both'] for x in nm)}, "
                 f"{sum(x['wrong_merges'] for x in nm)} wrong") if nm else "-"
        print(f"| {name} | {mean('idf1_global'):.3f} | {mean('idf1_camera'):.3f} | {mean('cross_cam_same_track'):.1%} | "
              f"{mean('track_purity'):.1%} | "
              f"{mean('false_merges'):.0f} | {names} |")
    named = [(name, r["names"]) for name in args.rules for r in res[name] if "names" in r]
    if named:
        print("\ncow_merges at other cow_merge_llr / cow_merge_margin: cows with one name, wrong merges (all seeds)")
        for name in dict(named):
            nm = [x for n, x in named if n == name]
            print(f"  {name:22s} " + "   ".join(f"{k}: {sum(x[k][0] for x in nm)}/{sum(x['cows_on_both'] for x in nm)}, "
                                                f"{sum(x[k][1] for x in nm)} wrong" for k in (f"{a:g}/{b:g}" for a, b in SWEEP)))


if __name__ == "__main__":
    main()
