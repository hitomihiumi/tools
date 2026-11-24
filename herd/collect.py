#!/usr/bin/env python3
"""The collector: every camera's tracks of a stretch of time grouped into cows at once, then named.

The live linker ([mct] link) joins two cameras' tracks pair by pair as the seconds come; with a pen's four corner
cameras, a floor known to ~1 m and cows 75 cm apart at the feed rail, it joins half its pairs wrong and misses
many - and a cow's bursts named by look alone (the gallery) are right only 30-70% of the time there. The
collector looks at the whole stretch:

  1. tracks of at least min_len seconds (the short ones are named on their own at the end);
  2. two cameras' tracks seen at one second: one cow that second when each is the other's nearest on the floor
     (common.floor_mapper: proj_mat at the cow's body height, or floor_points) within `near`, the next one
     `margin` further; two cows when they are more than `far` apart - summed over every second both were seen.
     Seen far apart in cannot_share of cannot_min seconds or more: never one cow;
  3. one camera's tracks seen at once (over_s): never one cow; one beginning within seq_gap s of another's end,
     seq_dist cm from it: likely her (seq_w);
  4. two tracks whose bursts' names agree (or not): name_w; two cameras' tracks that look alike (the encoder's own
     vectors, burst_fp.frames, when the store keeps them): look_w;
  5. groups merged, the strongest sum first, never across a "never" - until no sum reaches merge_min;
  6. names: each group's bursts' votes (confirmed), the most voted group-name first, a name never to two groups one
     camera saw at the same time (one cow, one name, at a time) - the 2/3 majority of report.track_map left
     too many groups without one.

On the 09 Nov barn (4 corner cameras, 13 cows, 30 min, the labelled boxes): IDF1 over all cameras 46.1 -> 52.5
(the second half, once the gallery knew the cows: 54.4 -> 65.0); on the five other runs of that recording +2 to +9
over all cameras. The store gets the groups as track_links and the names as track_names (report.track_map reads
them first): every report follows.

    python herd/herd.py collect --config barn.toml --store herd.sqlite [--out copy.sqlite] [key=value ...]
"""

from __future__ import annotations

import argparse
import collections
import itertools
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import floor_mapper, load_config  # noqa: E402

PARAMS = dict(min_len=15, near=120.0, far=300.0, margin=50.0, cannot_share=0.5, cannot_min=5, neg_w=1.0,
              seq_gap=30.0, seq_dist=120.0, seq_w=4.0, name_w=1.0, merge_min=1.0, over_s=3, name_overlap_s=5,
              look_w=3.0)


def load(store, cfg):
    """-> rows {track: {second: (box, floor xy, p_lying)}}, cam_of {track: camera}, votes {track: Counter(cow)}."""
    fm = {c["id"]: floor_mapper(c, cfg["mct"]) for c in cfg["cameras"]}
    rows, cam_of = collections.defaultdict(dict), {}
    for cam, tr, ts, x1, y1, x2, y2, post, pl in store.query(
            "SELECT cam, track, ts, x1, y1, x2, y2, posture, p_lying FROM seconds"):
        if cam not in fm:
            continue
        p = (1.0 if post == "lying" else 0.0) if pl is None else pl
        b = (x1, y1, x2, y2)
        xy = fm[cam](b, p) if fm[cam] else None
        rows[tr][int(round(ts))] = (b, None if xy is None else np.asarray(xy, float), p)
        cam_of[tr] = cam
    votes = collections.defaultdict(collections.Counter)
    for tr, cow, n in store.query("SELECT track, cow, COUNT(*) FROM bursts WHERE state='confirmed' AND cow IS NOT NULL "
                                  "GROUP BY track, cow"):
        votes[tr][cow] += n
    return rows, cam_of, votes


def looks(store, cam_of):
    """{track: its look} - the mean of its bursts' encoder vectors (burst_fp.frames: [store] keep_fingerprints), the
    camera's own mean taken off, unit length; {} when the store keeps none."""
    if not hasattr(store, "burst_fps"):
        return {}
    per = collections.defaultdict(list)
    for _, t, _, _, _, fr in store.burst_fps(frames=True):
        if fr is not None and t in cam_of:
            per[t].append(fr)
    if not per:
        return {}
    cam_mean = {}
    for c in set(cam_of[t] for t in per):
        cam_mean[c] = np.mean([v for t, vs in per.items() if cam_of[t] == c for v in vs], 0)
    out = {}
    for t, vs in per.items():
        v = np.mean(vs, 0) - cam_mean[cam_of[t]]
        out[t] = v / max(float(np.linalg.norm(v)), 1e-9)
    return out


def add_looks(A, long, cam_of, look, w):
    """Two cameras' tracks that look alike (their looks' cosine, as standard scores over all such pairs): w each.
    The 09 Nov barn: DINOv2's vectors tell a cow on another camera 27-33% of the time (chance 15%) - too little to
    name by, a little evidence for the groups (IDF1 48.3 -> 50.2). One camera's own tracks: none (neighbours at the
    feed rail share their crops)."""
    ts = [t for t in long if t in look]
    pairs = [(a, b) for i, a in enumerate(ts) for b in ts[i + 1:] if cam_of[a] != cam_of[b]]
    if not pairs or not w:
        return
    sims = np.array([float(look[a] @ look[b]) for a, b in pairs])
    z = (sims - sims.mean()) / max(float(sims.std()), 1e-9)
    for (a, b), v in zip(pairs, z):
        A[(a, b) if a < b else (b, a)] += w * float(v)


def affinities(rows, cam_of, votes, p):
    """-> (the long tracks, {(a, b): evidence of one cow, a < b}, {(a, b): never one cow})."""
    long = sorted(t for t in rows if len(rows[t]) >= p["min_len"])
    span = {t: (min(rows[t]), max(rows[t])) for t in long}
    A, never = collections.Counter(), set()
    at = collections.defaultdict(lambda: collections.defaultdict(list))      # second -> camera -> [(track, xy)]
    for t in long:
        for s, (_, xy, _) in rows[t].items():
            if xy is not None:
                at[s][cam_of[t]].append((t, xy))
    shared, far = collections.Counter(), collections.Counter()
    for s, per in at.items():
        for ca, cb in itertools.combinations(sorted(per), 2):
            ta, tb = per[ca], per[cb]
            D = np.linalg.norm(np.array([x for _, x in ta])[:, None, :] - np.array([y for _, y in tb])[None, :, :], axis=2)
            for i, (a, _) in enumerate(ta):
                for j, (b, _) in enumerate(tb):
                    key = (a, b) if a < b else (b, a)
                    shared[key] += 1
                    d = D[i, j]
                    if d > p["far"]:
                        A[key] -= p["neg_w"]
                        far[key] += 1
                    elif d <= p["near"] and D[i].argmin() == j and D[:, j].argmin() == i:
                        row = np.sort(D[i]); col = np.sort(D[:, j])
                        second = min(row[1] if len(row) > 1 else np.inf, col[1] if len(col) > 1 else np.inf)
                        if second - d >= p["margin"]:
                            A[key] += 1
    for key, n in shared.items():
        if n >= p["cannot_min"] and far[key] >= p["cannot_share"] * n:
            never.add(key)
    for a, b in itertools.combinations(long, 2):
        key = (a, b)
        (a0, a1), (b0, b1) = span[a], span[b]
        if cam_of[a] == cam_of[b]:
            if min(a1, b1) - max(a0, b0) >= p["over_s"]:
                never.add(key)                                 # one camera saw both at once: two cows
            else:
                first, nxt = (a, b) if a1 <= b0 else (b, a)
                gap = span[nxt][0] - span[first][1]
                xa, xb = rows[first][span[first][1]][1], rows[nxt][span[nxt][0]][1]
                if 0 <= gap <= p["seq_gap"] and xa is not None and xb is not None \
                        and np.linalg.norm(xa - xb) <= p["seq_dist"]:
                    A[key] += p["seq_w"] * (1 - gap / p["seq_gap"] / 2)
        va, vb = votes.get(a), votes.get(b)
        na, nb = (sum(va.values()) if va else 0), (sum(vb.values()) if vb else 0)
        if na > 0 and nb > 0 and p["name_w"]:
            agree = sum(min(va[c] / na, vb[c] / nb) for c in va)          # 0 .. 1: the same name
            A[key] += p["name_w"] * (2 * agree - 1) * min(na, nb, 5)
    return long, A, never


def cluster(long, A, never, p):
    """Groups merged by the strongest summed evidence first, never across a pair that is never one cow."""
    idx = {t: i for i, t in enumerate(long)}
    n = len(long)
    M, X = np.zeros((n, n)), np.zeros((n, n), bool)
    for (a, b), v in A.items():
        if a in idx and b in idx:
            M[idx[a], idx[b]] += v
            M[idx[b], idx[a]] += v
    for a, b in never:
        if a in idx and b in idx:
            X[idx[a], idx[b]] = X[idx[b], idx[a]] = True
    members = {i: [i] for i in range(n)}
    alive = np.ones(n, bool)
    while n > 1:
        S = np.where(X | ~alive[:, None] | ~alive[None, :], -np.inf, M)
        np.fill_diagonal(S, -np.inf)
        i, j = np.unravel_index(np.argmax(S), S.shape)
        if S[i, j] < p["merge_min"]:
            break
        members[i] += members.pop(j)
        alive[j] = False
        M[i, :] += M[j, :]
        M[:, i] += M[:, j]
        X[i, :] |= X[j, :]
        X[:, i] |= X[:, j]
    return [[long[k] for k in m] for m in members.values()]


def name(groups, rows, cam_of, votes, others, p):
    """{track: cow}: the most voted (group, name) first; a name never to two groups one camera saw at once."""
    units = [list(g) for g in groups] + [[t] for t in others]
    v = [sum((votes.get(t, collections.Counter()) for t in u), collections.Counter()) for u in units]
    sp = [[(cam_of[t], min(rows[t]), max(rows[t])) for t in u] for u in units]

    def clash(a, b):
        return any(ca == cb and min(a1, b1) - max(a0, b0) >= p["name_overlap_s"]
                   for ca, a0, a1 in sp[a] for cb, b0, b1 in sp[b])
    cand = sorted(((c, k, cow) for k, vv in enumerate(v) for cow, c in vv.items()), key=lambda x: (-x[0], x[1], x[2]))
    got, holders = {}, collections.defaultdict(list)
    for c, k, cow in cand:
        if k in got or any(clash(k, o) for o in holders[cow]):
            continue
        got[k] = cow
        holders[cow].append(k)
    return {t: got[k] for k, u in enumerate(units) if k in got for t in u}


def collect(store, cfg, **params):
    """-> (groups [[track, ...]], names {track: cow}) for the whole store."""
    p = dict(PARAMS, **params)
    rows, cam_of, votes = load(store, cfg)
    long, A, never = affinities(rows, cam_of, votes, p)
    add_looks(A, long, cam_of, looks(store, cam_of), p["look_w"])
    groups = cluster(long, A, never, p)
    keep = set(long)
    return groups, name(groups, rows, cam_of, votes, [t for t in rows if t not in keep], p)


def write(store, groups, names, ts=0.0):
    """The groups as the store's track_links (each group joined to its first track; the old joins gone) and the names
    as track_names (report.track_map reads them first)."""
    store.execute("DELETE FROM track_links")
    for g in groups:
        for t in g[1:]:
            store.link_tracks(ts, t, g[0])
    store.put_track_names(names, ts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--store", required=True, help="a run's or a trial's herd.sqlite")
    ap.add_argument("--out", default=None, help="write to a copy here (default: the store itself)")
    ap.add_argument("params", nargs="*", help=f"key=value of {', '.join(PARAMS)}")
    args = ap.parse_args(argv)
    from store import Store
    params = {}
    for kv in args.params:
        k, _, v = kv.partition("=")
        if k not in PARAMS:
            ap.error(f"{k}: not one of {', '.join(PARAMS)}")
        params[k] = float(v)
    path = args.store
    if args.out:
        shutil.copy(args.store, args.out)
        path = args.out
    store = Store(path)
    groups, names = collect(store, load_config(args.config), **params)
    write(store, groups, names)
    tracks = sum(len(g) for g in groups)
    print(f"[collect] {tracks} tracks in {len(groups)} groups, {len(set(names.values()))} names on {len(names)} tracks "
          f"-> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
