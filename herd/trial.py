#!/usr/bin/env python3
"""A dress rehearsal on recordings: the barn's whole system with its own config - zones, detector, tracker,
posture and activity, the gallery - on the files, in a store of its own, and what it made drawn back.

A real barn's gallery fills over its first hours (a cow needs new_cow_min_bursts bursts before she gets an
id, and new cows are made at night); a quarter of an hour of recordings would end with every cow NaN.
So the recordings are played twice:

  1. the first pass - a burst every --burst-every s - and then the night's maintenance at its end: the cows
     seen often enough (--min-bursts) get ids (new-<day>-NN)
  2. the second pass with that gallery, as the next day would: who is who
  3. show-ids for every camera (OUT/show_<cam>/index.html, ids.mp4) and OUT/trial.json: per camera the
     seconds counted, the tracks, the share of boxes given a cow, posture and activity; the cows made

    python herd/herd.py trial --config barn.toml --videos /srv/herd/rec --out /srv/herd/trial

With the boxes you labelled with cow ids on frames of the same recordings (herd.py boxes: frames.json and
the labels.json the page, or the ids gallery, saved) the second pass is scored against them - IDF1, the
share of your boxes that got the right cow (the system's names matched one to one to yours), and IDP /
IDR beside it; per camera too, and which of its cows took two of yours (merged) or the other way (split):

    python herd/herd.py trial ... --boxes /srv/herd/boxes --labels /srv/herd/labels.json
    python herd/herd.py trial --score-only --config barn.toml --out /srv/herd/trial --boxes ... --labels ...
"""

from __future__ import annotations

import argparse
import collections
import copy
import datetime as dt
import itertools
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import iou, load_config, recordings, sessions  # noqa: E402


def play(cfg, groups, store_path, models):
    import pipeline
    for k, g in enumerate(groups, 1):
        c = copy.deepcopy(cfg)
        cams = {cam["id"]: cam for cam in c["cameras"]}
        c["cameras"] = [dict(cams[cam], url=path, start_ts=ts) for cam, path, ts in g]
        c["store"]["path"] = store_path
        start = min(ts for _, _, ts in g)
        print(f"[trial]   session {k}/{len(groups)}: {', '.join(cam for cam, _, _ in g)}", flush=True)
        models = models or pipeline.Models(c)
        status = {}
        pipeline.run(c, start, models=models, status=status)
        if status.get("interrupted"):
            sys.exit("[trial] stopped (Ctrl+C)")
    return models


def summary(store_path, cfg, gallery_folder):
    import report
    from store import Store
    store = Store(store_path)
    tm = report.track_map(store)
    out = {}
    for cam, n, tracks in store.query("SELECT cam, COUNT(*), COUNT(DISTINCT track) FROM seconds GROUP BY cam"):
        rows = store.query("SELECT track, posture, activity FROM seconds WHERE cam = ?", (cam,))
        named = sum(1 for t, _, _ in rows if tm.get(t))
        states = dict(store.query("SELECT state, COUNT(*) FROM bursts WHERE cam = ? GROUP BY state", (cam,)))
        out[cam] = {"cow_seconds": n, "tracks": tracks, "with_a_cow": named / max(1, n),
                    "cows": len({tm[t] for t, _, _ in rows if tm.get(t)}),
                    "posture": dict(collections.Counter(p for _, p, _ in rows)),
                    "activity": dict(collections.Counter(a for _, _, a in rows)), "bursts": states}
    meta = os.path.join(gallery_folder, "gallery.json")
    cows = len(json.load(open(meta)).get("cows", {})) if os.path.exists(meta) else 0
    return out, cows, impossible(store, tm, cfg), joined(store)


def joined(store):
    """Tracks of one camera joined to another camera's ([mct]: by the floor, or by look in a strip they
    share) -> {"cam1-cam2": tracks joined}."""
    roots = store.track_roots() if hasattr(store, "track_roots") else {}
    cam_of = dict(store.query("SELECT track, MIN(cam) FROM seconds GROUP BY track"))
    out = collections.Counter()
    for track, root in roots.items():
        a, b = cam_of.get(track), cam_of.get(root)
        if a and b and a != b:
            out["-".join(sorted((a, b)))] += 1
    return dict(out)


def impossible(store, tm, cfg):
    """What cannot be: one cow at the same second on two tracks of one camera, or on two cameras that share
    no floor ([mct] disjoint) - two cows given one id. -> counts, per cow and over all, of named cow-seconds."""
    disjoint = {frozenset(p) for p in cfg["mct"].get("disjoint", [])}
    at = collections.defaultdict(lambda: collections.defaultdict(set))       # second -> cow -> {(cam, track)}
    named = 0
    for cam, track, ts in store.query("SELECT cam, track, ts FROM seconds"):
        cow = tm.get(track)
        if cow:
            at[int(ts)][cow].add((cam, track))
            named += 1
    twice, apart, counted2, per_cow = 0, 0, 0, collections.Counter()
    for cows in at.values():
        for cow, seen in cows.items():
            cams = collections.Counter(c for c, _ in seen)
            bad = sum(n - 1 for n in cams.values())
            far = sum(1 for a, b in itertools.combinations(sorted(cams), 2) if frozenset((a, b)) in disjoint)
            both = len(cams) - 1 - far if len(cams) > 1 else 0     # cameras sharing floor: masks share none
            twice, apart, counted2 = twice + bad, apart + far, counted2 + max(0, both)
            if bad or far or both > 0:
                per_cow[cow] += 1
    return {"named_cow_seconds": named, "one_cow_twice_on_one_camera": twice,
            "one_cow_on_cameras_apart": apart, "one_cow_counted_by_two_cameras": counted2,
            "cows_seen_where_they_cannot_be": dict(per_cow.most_common())}


def idf1(co, n_true, n_pred):
    """co Counter((your id, its cow)) over the boxes matched -> IDF1, IDP, IDR, the one-to-one mapping."""
    from scipy.optimize import linear_sum_assignment
    if not co:
        return {"idf1": 0.0, "idp": 0.0, "idr": 0.0, "mapping": {}}
    ts = sorted({t for t, _ in co})
    ps = sorted({p for _, p in co})
    M = np.zeros((len(ts), len(ps)))
    for (t, p), n in co.items():
        M[ts.index(t), ps.index(p)] = n
    r, c = linear_sum_assignment(-M)
    tp = float(M[r, c].sum())
    return {"idf1": 2 * tp / max(1, n_true + n_pred), "idp": tp / max(1, n_pred), "idr": tp / max(1, n_true),
            "mapping": {ts[i]: ps[j] for i, j in zip(r, c) if M[i, j] > 0}}


def score(store_path, cfg, boxes_dir, labels_path, max_dt=0.6, min_iou=0.3, page_dir=None, cameras=None):
    """Your labelled boxes (status done, with an id) against the second pass at the same camera and second.

    Only boxes this camera counts (its mask, zone) are scored - one in another camera's part of an overlap
    is that camera's. Your box and its box matched by overlap (min_iou, one to one); the system's cow of a
    box is its track's (report.track_map). A box of yours it did not find, or found without a cow, counts
    against IDR; a cow it gave where you labelled none counts against IDP."""
    import report
    from box_dataset import read_labels
    from common import in_zone
    from store import Store
    store = Store(store_path)
    tm = report.track_map(store)
    frames = {f["file"]: f for f in json.load(open(os.path.join(boxes_dir, "frames.json"), encoding="utf-8"))["frames"]}
    cams = {c["id"]: c for c in cfg["cameras"] if not cameras or c["id"] in cameras}
    labels = read_labels(labels_path)
    shift = clock_shift(store, [frames[k] for k, v in labels.items() if v.get("status") == "done" and k in frames
                                and frames[k]["cam"] in cams])
    co, cam_co = collections.Counter(), collections.defaultdict(collections.Counter)
    n = collections.Counter()                      # (cam, what) -> boxes
    seen_as = collections.defaultdict(list)        # your id -> [(cam, ts, its cow)]: one cow on two cameras
    view = []                                      # every frame played: your boxes and its, for boxes.html
    for file, lab in labels.items():
        f = frames.get(file)
        if lab.get("status") != "done" or not f or f["cam"] not in cams:
            continue
        f = dict(f, ts=f["ts"] + shift)
        cam = f["cam"]
        named = [(b[:4], str(b[4]).strip()) for b in lab.get("boxes", []) if len(b) >= 5 and str(b[4]).strip()
                 and str(b[4]).strip().lower() != "x"]
        truth = [t for t in named if in_zone(t[0], cams[cam], f.get("w", 1920), f.get("h", 1080))]
        rows = store.query("SELECT track, ts, x1, y1, x2, y2 FROM seconds WHERE cam = ? AND ts BETWEEN ? AND ?",
                           (cam, f["ts"] - max_dt, f["ts"] + max_dt))
        if not rows:
            n[(cam, "frames_not_played")] += 1
            continue
        n[(cam, "frames")] += 1
        sec = min({r[1] for r in rows}, key=lambda t: abs(t - f["ts"]))
        pred = [(list(r[2:6]), tm.get(r[0])) for r in rows if r[1] == sec]
        view.append({"cam": cam, "ts": f["ts"], "file": file, "w": f.get("w", 1920), "h": f.get("h", 1080),
                     "mine": [[*map(float, b), i, (b, i) in truth] for b, i in named],   # [.., scored: in its mask]
                     "its": [[*map(float, r[2:6]), tm.get(r[0]), r[0]] for r in rows if r[1] == sec],
                     "hit": match([b for b, _ in named], [b for b, _ in pred], min_iou)})
        n[(cam, "your_boxes")] += len(truth)
        n[(cam, "its_named_boxes")] += sum(1 for _, cow in pred if cow)
        hit = match([b for b, _ in truth], [b for b, _ in pred], min_iou)
        n[(cam, "found")] += len(hit)
        n[(cam, "found_with_a_cow")] += sum(1 for _, j in hit if pred[j][1])
        for i, j in hit:
            if pred[j][1]:
                co[(truth[i][1], pred[j][1])] += 1
                cam_co[cam][(truth[i][1], pred[j][1])] += 1
                seen_as[truth[i][1]].append((cam, f["ts"], pred[j][1]))
    allc = sorted({c for c, _ in n})
    tot = lambda what: sum(n[(c, what)] for c in allc)
    out = {"all": idf1(co, tot("your_boxes"), tot("its_named_boxes")), "cameras": {}}
    out["all"].update({k: tot(k) for k in ("frames", "frames_not_played", "your_boxes", "found", "found_with_a_cow",
                                           "its_named_boxes")})
    for c in allc:
        r = idf1(cam_co[c], n[(c, "your_boxes")], n[(c, "its_named_boxes")])
        r.update({k: n[(c, k)] for k in ("frames", "frames_not_played", "your_boxes", "found", "found_with_a_cow",
                                         "its_named_boxes")})
        r.pop("mapping")
        out["cameras"][c] = r
    by_pred, by_true = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    for (t, p), k in co.items():
        by_pred[p][t] += k
        by_true[t][p] += k
    big = lambda cnt: [x for x, k in cnt.items() if k >= max(3, 0.2 * sum(cnt.values()))]
    out["merged"] = {p: dict(cnt.most_common()) for p, cnt in by_pred.items() if len(big(cnt)) > 1}
    out["split"] = {t: dict(cnt.most_common()) for t, cnt in by_true.items() if len(big(cnt)) > 1}
    out["your_cows"], out["its_cows"] = len(by_true), len(by_pred)
    out["cross_camera"] = cross_camera(seen_as)
    out["clock_shift_s"] = shift
    if page_dir and out["merged"]:
        out["same_cow_page"] = same_cow_page(page_dir, boxes_dir, labels, frames, out["merged"])
    if page_dir and view:
        out["boxes_page"] = boxes_page(page_dir, boxes_dir, view, out["all"]["mapping"],
                                       cfg["farm"].get("timezone_offset_hours", 0))
    return out


def match(mine, its, min_iou):
    """Your boxes against its at one moment, one to one by overlap -> [(your index, its index)]."""
    from scipy.optimize import linear_sum_assignment
    if not mine or not its:
        return []
    M = np.array([[iou(a, b) for b in its] for a in mine])
    r, c = linear_sum_assignment(-M)
    return [(int(i), int(j)) for i, j in zip(r, c) if M[i, j] >= min_iou]


def boxes_page(out, boxes_dir, view, mapping, tz_hours=0):
    """OUT/boxes.html: every labelled moment with all its cameras side by side - your boxes with your ids and
    the cow the system gave each (right / another cow / not found / no cow) - and any box, or any cow of
    yours, as crops from every camera and moment. The frames are read where they are (boxes/frames/...)."""
    rel = lambda file: os.path.relpath(os.path.join(boxes_dir, file), out).replace(os.sep, "/")
    frames = [dict(v, src=rel(v["file"])) for v in sorted(view, key=lambda v: (v["ts"], v["cam"]))]
    data = {"frames": frames, "mapping": mapping, "tz": tz_hours}
    path = os.path.join(out, "boxes.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(BOXES_PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    return path


def cross_camera(seen_as, min_boxes=3, at_once_s=1.0):
    """Your cows scored on two cameras or more: did the system give each the same cow on all of them? By cow (its
    cow most often on each camera, from min_boxes of that camera's boxes) and by moment (your cow in two
    cameras' frames within at_once_s: the same cow of its on both)."""
    cows, same = {}, 0
    for t, items in sorted(seen_as.items()):
        per = collections.defaultdict(collections.Counter)
        for cam, _, p in items:
            per[cam][p] += 1
        top = {cam: cnt.most_common(1)[0] + (sum(cnt.values()),) for cam, cnt in per.items()
               if sum(cnt.values()) >= min_boxes}
        if len(top) < 2:
            continue
        one = len({p for p, _, _ in top.values()}) == 1
        same += one
        cows[t] = {"same": one, "cameras": {cam: {"its": p, "boxes": k, "of": m} for cam, (p, k, m) in sorted(top.items())}}
    pairs = agree = 0
    for t, items in seen_as.items():
        for (ca, ta, pa), (cb, tb, pb) in itertools.combinations(items, 2):
            if ca != cb and abs(ta - tb) <= at_once_s:
                pairs += 1
                agree += pa == pb
    return {"cows_on_two_cameras": len(cows), "same_cow_on_all": same, "cows": cows,
            "moments": pairs, "moments_same_cow": agree}


def same_cow_page(out, boxes_dir, labels, frames, merged, per_cam=2):
    """OUT/same_cow.html: for each of its cows that took several of your ids, your ids side by side (crops
    from your frames, each camera) - one click says they are one cow (or not); Save gives labels.json with
    those ids made one (the cow a second camera sees under another of your names)."""
    from PIL import Image
    folder = os.path.join(out, "same_cow")
    os.makedirs(folder, exist_ok=True)
    want = {i for ids in merged.values() for i in ids}
    seen = collections.defaultdict(list)                       # your id -> [(cam, file, box)]
    for file, lab in labels.items():
        f = frames.get(file)
        if lab.get("status") != "done" or not f:
            continue
        for b in lab.get("boxes", []):
            if len(b) >= 5 and str(b[4]).strip() in want:
                seen[str(b[4]).strip()].append((f["cam"], file, b[:4], f["ts"]))
    crops = {}
    for cid, items in seen.items():
        out_c = []
        for cam in sorted({c for c, _, _, _ in items}):
            mine = sorted((x for x in items if x[0] == cam), key=lambda x: x[3])
            for k in sorted({int(round(i)) for i in np.linspace(0, len(mine) - 1, min(per_cam, len(mine)))}):
                _, file, b, _ = mine[k]
                if not os.path.exists(os.path.join(boxes_dir, file)):   # frames.json alone (no frames/): no crop
                    continue
                with Image.open(os.path.join(boxes_dir, file)) as im:
                    w, h = im.size
                    pad = 0.08
                    c = im.convert("RGB").crop((int(max(0, (b[0] - pad * (b[2] - b[0])) * w)),
                                                int(max(0, (b[1] - pad * (b[3] - b[1])) * h)),
                                                int(min(w, (b[2] + pad * (b[2] - b[0])) * w)),
                                                int(min(h, (b[3] + pad * (b[3] - b[1])) * h))))
                    c.thumbnail((200, 200))
                    name = f"{len(crops) * 10 + len(out_c)}_{cam}.jpg"
                    c.save(os.path.join(folder, name), quality=88)
                out_c.append({"src": f"same_cow/{name}", "cam": cam})
        crops[cid] = {"crops": out_c, "cams": sorted({c for c, _, _, _ in items}), "boxes": len(items)}
    cards = [{"cow": cow, "ids": [{"id": i, "n": n, **crops.get(i, {"crops": [], "cams": [], "boxes": 0})}
                                  for i, n in ids.items()]}
             for cow, ids in sorted(merged.items(), key=lambda kv: -sum(kv[1].values()))]
    data = {"cards": cards, "labels": {"frames": labels}}
    path = os.path.join(out, "same_cow.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(SAME_COW.replace("__DATA__", json.dumps(data).replace("</", "<\\/")))
    return path


SAME_COW = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>One cow, two names?</title>
<style>
:root { --bg: #f6f6f4; --fg: #1d1d1b; --muted: #6b6b66; --line: #d9d9d4; --card: #fff; --accent: #c2410c;
  --yes: #15803d; --no: #b91c1c; }
@media (prefers-color-scheme: dark) { :root { --bg: #161615; --fg: #ecece8; --muted: #9a9a94; --line: #34342f;
  --card: #1f1f1d; --accent: #fb923c; --yes: #4ade80; --no: #f87171; } }
body { margin: 0; background: var(--bg); color: var(--fg); font: 15px/1.45 system-ui, sans-serif; }
header { position: sticky; top: 0; z-index: 2; background: var(--bg); border-bottom: 1px solid var(--line);
  padding: 10px 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
h1 { font-size: 17px; margin: 0 8px 0 0; }
button { font: inherit; padding: 5px 12px; border: 1px solid var(--line); border-radius: 6px; background: var(--card);
  color: var(--fg); cursor: pointer; }
button.main { background: var(--accent); border-color: var(--accent); color: #fff; }
button.on.yes { background: var(--yes); border-color: var(--yes); color: #fff; }
button.on.no { background: var(--no); border-color: var(--no); color: #fff; }
.help { padding: 8px 16px; color: var(--muted); max-width: 1000px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; margin: 12px 16px; padding: 10px 12px; }
.card h2 { font-size: 15px; margin: 0 0 8px; color: var(--muted); font-weight: normal; }
.row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; padding: 6px 0; border-top: 1px solid var(--line); }
.row:first-of-type { border-top: 0; }
.name { min-width: 120px; }
.name b { font-size: 18px; }
.name small { display: block; color: var(--muted); }
.crops { display: flex; gap: 6px; flex-wrap: wrap; }
.crops figure { margin: 0; text-align: center; font-size: 12px; color: var(--muted); }
.crops img { height: 120px; border-radius: 4px; display: block; }
.ask { margin-left: auto; display: flex; gap: 6px; }
</style></head><body>
<header><h1>One cow, two names?</h1><button class="main" id="save">Save labels.json</button>
<span id="state" style="color: var(--muted)"></span></header>
<div class="help">The system gave one cow to boxes you named differently. Often it is one cow that two cameras
see, named once on each - then say <b>same cow</b> and the second name becomes the first. If they are two cows,
say <b>different</b>: the system was wrong there. Save gives labels.json with the names made one; score again
with it (and use it for training).</div>
<main id="cards"></main>
<script>
const DATA = __DATA__;
const KEY = "same-cow:" + DATA.cards.map(c => c.cow).join(",");
let ans = {};
try { ans = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { ans = {}; }
const keep = () => { try { localStorage.setItem(KEY, JSON.stringify(ans)); } catch (e) {} };
function render() {
  const main = document.getElementById("cards");
  main.innerHTML = "";
  let asked = 0, done = 0;
  DATA.cards.forEach(c => {
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<h2>The system's ${c.cow}</h2>`;
    const first = c.ids[0].id;
    c.ids.forEach((x, k) => {
      const row = document.createElement("div");
      row.className = "row";
      const figs = x.crops.map(p => `<figure><img src="${p.src}" alt="${x.id}"><figcaption>${p.cam}</figcaption></figure>`).join("");
      row.innerHTML = `<div class="name"><b>${x.id}</b><small>${x.n} boxes with it here; ${(x.cams || []).join(", ")}</small></div>
        <div class="crops">${figs}</div>`;
      if (k > 0) {
        asked++;
        const key = c.cow + "|" + x.id, a = ans[key];
        if (a) done++;
        const ask = document.createElement("div");
        ask.className = "ask";
        ask.innerHTML = `<button class="yes ${a === "same" ? "on" : ""}">same cow as ${first}</button>
          <button class="no ${a === "diff" ? "on" : ""}">different</button>`;
        ask.querySelector(".yes").onclick = () => { ans[key] = "same"; keep(); render(); };
        ask.querySelector(".no").onclick = () => { ans[key] = "diff"; keep(); render(); };
        row.appendChild(ask);
      }
      card.appendChild(row);
    });
    main.appendChild(card);
  });
  document.getElementById("state").textContent = done + " of " + asked + " answered";
}
function renames() {
  const parent = {};
  const root = x => { while (parent[x] && parent[x] !== x) x = parent[x]; return x; };
  DATA.cards.forEach(c => c.ids.slice(1).forEach(x => {
    if (ans[c.cow + "|" + x.id] === "same") {
      const a = root(c.ids[0].id), b = root(x.id);
      if (a !== b) parent[b] = a;
    }
  }));
  const out = {};
  Object.keys(parent).forEach(k => { const r = root(k); if (r !== k) out[k] = r; });
  return out;
}
document.getElementById("save").onclick = () => {
  const ren = renames(), lab = JSON.parse(JSON.stringify(DATA.labels));
  Object.values(lab.frames).forEach(f => (f.boxes || []).forEach(b => {
    if (b.length >= 5 && ren[String(b[4]).trim()] !== undefined) b[4] = ren[String(b[4]).trim()];
  }));
  const blob = new Blob([JSON.stringify(lab)], {type: "application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "labels.json"; a.click();
};
render();
</script></body></html>
"""


def clock_shift(store, frames):
    """Whole hours to add to the frames' clock to land in this run: frames cut with another time zone
    (timezone_offset_hours changed since) are as many hours off. 0 when they land as they are."""
    span = {cam: (a, b) for cam, a, b in store.query("SELECT cam, MIN(ts), MAX(ts) FROM seconds GROUP BY cam")}
    hits = lambda off: sum(1 for f in frames if f["cam"] in span and span[f["cam"]][0] - 1 <= f["ts"] + off
                           <= span[f["cam"]][1] + 1)
    best = max(range(-14, 15), key=lambda h: (hits(h * 3600), -abs(h)))
    if hits(best * 3600) <= hits(0):
        if frames and not hits(0):
            fr = sorted(f["ts"] for f in frames)
            print(f"[score] none of your {len(frames)} frames is in this run: they are from "
                  f"{dt.datetime.utcfromtimestamp(fr[0]):%Y-%m-%d %H:%M} to {dt.datetime.utcfromtimestamp(fr[-1]):%H:%M} "
                  f"UTC, the run from {dt.datetime.utcfromtimestamp(min(a for a, _ in span.values())):%Y-%m-%d %H:%M}"
                  f" to {dt.datetime.utcfromtimestamp(max(b for _, b in span.values())):%H:%M} UTC", flush=True)
        return 0.0
    print(f"[score] your frames' clock is {best:+d} h off this run's: cut with another timezone_offset_hours than "
          f"the config has now - taken into account ({hits(best * 3600)} of {len(frames)} frames in the run)", flush=True)
    return best * 3600.0


def print_score(sc):
    a = sc["all"]
    if sc.get("clock_shift_s"):
        print(f"[score] (your frames' clock moved {sc['clock_shift_s'] / 3600:+.0f} h to this run's)")
    print(f"\n[score] against your labels: {a['your_boxes']} boxes of {sc['your_cows']} cows on {a['frames']} frames"
          + (f" ({a['frames_not_played']} frames not in this run)" if a["frames_not_played"] else ""))
    print(f"  found {a['found'] / max(1, a['your_boxes']):.0%} of your boxes, {a['found_with_a_cow'] / max(1, a['your_boxes']):.0%}"
          f" with a cow; IDF1 {a['idf1']:.1%} (IDP {a['idp']:.1%}: of its cows given, the right one; IDR "
          f"{a['idr']:.1%}: of your boxes, the right cow)")
    for c, r in sorted(sc["cameras"].items()):
        print(f"  {c}: IDF1 {r['idf1']:.1%} (IDP {r['idp']:.1%}, IDR {r['idr']:.1%}) on {r['your_boxes']} boxes")
    x = sc.get("cross_camera")
    if x and x["cows_on_two_cameras"]:
        print(f"  your cows scored on two cameras or more: {x['cows_on_two_cameras']}; the same cow of its on all of "
              f"them: {x['same_cow_on_all']}" + (f"; at one moment on two cameras: the same cow {x['moments_same_cow']} "
                                                f"of {x['moments']} times" if x["moments"] else ""))
        for t, r in x["cows"].items():
            if not r["same"]:
                print(f"    your {t}: " + ", ".join(f"{c} {v['its']} ({v['boxes']}/{v['of']})" for c, v in r["cameras"].items()))
    elif x is not None:
        print("  none of your cows scored on two cameras (each camera counts its own floor: --overlap lets the "
              "masks share it)")
    print(f"  its cows: {sc['its_cows']} for your {sc['your_cows']}; one of its cows on several of yours (merged): "
          f"{len(sc['merged'])}; one of yours under several of its (split): {len(sc['split'])}")
    if sc.get("boxes_page"):
        print(f"  every moment, all cameras, box by box: {sc['boxes_page']}")
    if sc.get("same_cow_page"):
        print(f"  one cow two names? {sc['same_cow_page']} - your names side by side: say which are one cow, save "
              "labels.json, score again with it")


BOXES_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Boxes, every camera</title>
<style>
:root { --bg: #f6f6f4; --fg: #1d1d1b; --muted: #6b6b66; --line: #d9d9d4; --card: #fff; --accent: #c2410c;
  --right: #15803d; --wrong: #dc2626; --missed: #d97706; --nocow: #7c3aed; --extra: #64748b; }
@media (prefers-color-scheme: dark) { :root { --bg: #161615; --fg: #ecece8; --muted: #9a9a94; --line: #34342f;
  --card: #1f1f1d; --accent: #fb923c; --right: #4ade80; --wrong: #f87171; --missed: #fbbf24; --nocow: #a78bfa;
  --extra: #94a3b8; } }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 14px/1.4 system-ui, sans-serif; }
header { position: sticky; top: 0; z-index: 5; background: var(--bg); border-bottom: 1px solid var(--line);
  padding: 8px 16px; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
h1 { font-size: 16px; margin: 0 6px 0 0; }
button, select { font: inherit; padding: 4px 10px; border: 1px solid var(--line); border-radius: 6px;
  background: var(--card); color: var(--fg); cursor: pointer; }
button.on { background: var(--accent); border-color: var(--accent); color: #fff; }
label { display: inline-flex; gap: 4px; align-items: center; cursor: pointer; }
#when { min-width: 150px; text-align: center; font-variant-numeric: tabular-nums; }
.legend { display: flex; gap: 10px; flex-wrap: wrap; color: var(--muted); font-size: 12px; padding: 6px 16px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
#grid { display: grid; gap: 10px; padding: 10px 16px; grid-template-columns: repeat(var(--cols, 2), minmax(0, 1fr)); }
@media (max-width: 760px) { #grid { grid-template-columns: minmax(0, 1fr); } }
.panel { background: var(--card); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.ph { padding: 4px 8px; display: flex; justify-content: space-between; color: var(--muted); font-size: 12px; }
.ph b { color: var(--fg); }
.frame { position: relative; background: var(--line); }
.frame img { display: block; width: 100%; height: 100%; object-fit: fill; }
.none { padding: 40px 10px; text-align: center; color: var(--muted); }
.box { position: absolute; border: 2px solid var(--c); border-radius: 2px; cursor: pointer; }
.box.out { border-style: dashed; opacity: .55; }
.box.dim { opacity: .18; }
.box.sel { box-shadow: 0 0 0 3px var(--fg); z-index: 3; }
.box span { position: absolute; left: -2px; top: -18px; white-space: nowrap; background: var(--c); color: var(--bg);
  font-size: 11px; line-height: 16px; padding: 0 4px; border-radius: 3px; }
.box.extra span { top: auto; bottom: -18px; }
.right { --c: var(--right); } .wrong { --c: var(--wrong); } .missed { --c: var(--missed); }
.nocow { --c: var(--nocow); } .extra { --c: var(--extra); } .box.extra { border-width: 1px; border-style: dotted; }
#insp { margin: 0 16px 24px; background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }
#insp:empty { display: none; }
.top { display: flex; gap: 14px; align-items: flex-start; flex-wrap: wrap; }
.facts { line-height: 1.7; }
.facts b { font-size: 16px; }
.facts .note { color: var(--muted); }
h3 { font-size: 13px; color: var(--muted); font-weight: normal; margin: 12px 0 4px; }
.strip { display: flex; gap: 6px; overflow-x: auto; padding-bottom: 6px; }
.strip figure { margin: 0; flex: none; text-align: center; font-size: 11px; color: var(--muted); cursor: pointer; }
.crop { border: 3px solid var(--c, var(--line)); border-radius: 4px; background-color: var(--line);
  background-repeat: no-repeat; }
.shot { position: relative; background: var(--line); border-radius: 4px; overflow: hidden; flex: none; }
.shot img { display: block; width: 100%; height: 100%; object-fit: fill; }
.shot .mark { position: absolute; border: 3px solid var(--c, var(--accent)); border-radius: 2px;
  box-shadow: 0 0 0 9999px rgba(0, 0, 0, .38); }
.strip figure.here .crop, .strip figure.here .shot { outline: 3px solid var(--fg); outline-offset: 1px; }
.cam { font-weight: 600; color: var(--fg); font-size: 12px; margin-top: 4px; }
</style></head><body>
<header><h1>Boxes</h1>
  <button id="prev" title="Left arrow">&#9664;</button><span id="when"></span><button id="next" title="Right arrow">&#9654;</button>
  <span id="cams"></span>
  <select id="cow"><option value="">all your cows</option></select>
  <label><input type="checkbox" id="errs"> moments with a mistake (E)</label>
  <label><input type="checkbox" id="extra" checked> its boxes you did not name</label>
  <label><input type="checkbox" id="zoom"> close-ups (Z)</label>
</header>
<div class="legend">
  <span><i style="background:var(--right)"></i>the right cow</span><span><i style="background:var(--wrong)"></i>another cow</span>
  <span><i style="background:var(--missed)"></i>not found</span><span><i style="background:var(--nocow)"></i>found, no cow</span>
  <span><i style="background:var(--extra)"></i>its box, not yours</span><span>dashed: outside this camera's mask (not scored)</span>
  <span>Click a box: it from every camera and moment.</span>
</div>
<div id="grid"></div>
<section id="insp"></section>
<script>
const DATA = __DATA__;
const F = DATA.frames, MAP = DATA.mapping;
const CAMS = [...new Set(F.map(f => f.cam))].sort();
const moments = [];                                  // frames of the cameras within 1.5 s: one moment
F.forEach((f, i) => {
  const last = moments[moments.length - 1];
  if (last && f.ts - last.ts <= 1.5 && !(f.cam in last.at)) last.at[f.cam] = i;
  else moments.push({ ts: f.ts, at: { [f.cam]: i } });
});
const hitOf = f => { const m = {}, t = {}; f.hit.forEach(([i, j]) => { m[i] = j; t[j] = i; }); return [m, t]; };
function status(f, k) {                              // your box k of frame f
  const [m] = hitOf(f), j = m[k], id = f.mine[k][4];
  if (j === undefined) return "missed";
  const c = f.its[j][4];
  if (!c) return "nocow";
  return c === MAP[id] ? "right" : "wrong";
}
const pad = n => String(n).padStart(2, "0");
function clock(ts) {
  const d = new Date((ts + DATA.tz * 3600) * 1000);
  return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
}
let shown = new Set(CAMS), cow = "", errs = false, extra = true, zoom = false, sel = null, m = 0;
try { const s = JSON.parse(localStorage.getItem("boxes-cams") || "null"); if (s) shown = new Set(s.filter(c => CAMS.includes(c))); } catch (e) {}
if (!shown.size) shown = new Set(CAMS);
const save = () => { try { localStorage.setItem("boxes-cams", JSON.stringify([...shown])); } catch (e) {} };
const ids = [...new Set(F.flatMap(f => f.mine.map(b => b[4])))].sort((a, b) => (+a - +b) || a.localeCompare(b));
const sel_ = document.getElementById("cow");
ids.forEach(i => sel_.insertAdjacentHTML("beforeend", `<option value="${i}">your cow ${i}${MAP[i] ? " (its " + MAP[i] + ")" : ""}</option>`));
function wanted(mo) {
  const fs = Object.entries(mo.at).filter(([c]) => shown.has(c)).map(([, i]) => F[i]);
  if (cow && !fs.some(f => f.mine.some(b => b[4] === cow))) return false;
  if (errs && !fs.some(f => f.mine.some((b, k) => b[5] && (!cow || b[4] === cow) && status(f, k) !== "right"))) return false;
  return fs.length > 0;
}
const list = () => moments.map((mo, i) => i).filter(i => wanted(moments[i]));
function go(d) {
  const l = list();
  if (!l.length) return;
  let p = l.indexOf(m);
  if (p < 0) p = l.findIndex(i => i > m), p = p < 0 ? l.length - 1 : p, d = 0;
  m = l[Math.max(0, Math.min(l.length - 1, p + d))];
  render();
}
function crop(f, b, H) {
  const bw = Math.max(1e-3, b[2] - b[0]), bh = Math.max(1e-3, b[3] - b[1]);
  let W = H * (bw * f.w) / (bh * f.h);
  if (W > 2 * H) W = 2 * H;
  const gw = W / bw, gh = gw * f.h / f.w, h = bh * gh;
  return `<div class="crop" style="width:${W.toFixed(0)}px;height:${h.toFixed(0)}px;background-image:url('${f.src}');` +
    `background-size:${gw.toFixed(1)}px ${gh.toFixed(1)}px;background-position:${(-b[0] * gw).toFixed(1)}px ${(-b[1] * gh).toFixed(1)}px"></div>`;
}
function boxEl(f, fi, b, kind, k, st) {
  const [x1, y1, x2, y2] = b;
  const isSel = sel && sel.fi === fi && sel.kind === kind && sel.k === k;
  const dim = cow && !(kind === "mine" ? b[4] === cow : false) && !isSel;
  let lab;
  if (kind === "mine") {
    const [mm] = hitOf(f), j = mm[k], c = j === undefined ? null : f.its[j][4];
    lab = st === "right" ? `${b[4]} = ${c}` : st === "wrong" ? `${b[4]}: ${c}` : st === "missed" ? `${b[4]}: not found` : `${b[4]}: no cow`;
  } else lab = b[4] || "no cow";
  return `<div class="box ${st} ${kind === "mine" && !b[5] ? "out" : ""} ${dim ? "dim" : ""} ${isSel ? "sel" : ""}"
    data-fi="${fi}" data-kind="${kind}" data-k="${k}"
    style="left:${x1 * 100}%;top:${y1 * 100}%;width:${(x2 - x1) * 100}%;height:${(y2 - y1) * 100}%"><span>${lab}</span></div>`;
}
function render() {
  const mo = moments[m];
  document.getElementById("when").textContent = mo ? `${clock(mo.ts)}  (${list().indexOf(m) + 1}/${list().length})` : "";
  const cams = CAMS.filter(c => shown.has(c));
  const grid = document.getElementById("grid");
  grid.style.setProperty("--cols", cams.length > 2 ? 2 : cams.length);
  grid.innerHTML = cams.map(c => {
    const fi = mo ? mo.at[c] : undefined;
    if (fi === undefined) return `<div class="panel"><div class="ph"><b>${c}</b></div><div class="none">no labelled frame at this moment</div></div>`;
    const f = F[fi], [, taken] = hitOf(f);
    let html = f.mine.map((b, k) => boxEl(f, fi, b, "mine", k, status(f, k))).join("");
    if (extra) html += f.its.map((b, j) => taken[j] === undefined ? boxEl(f, fi, b, "its", j, "extra") : "").join("");
    const bad = f.mine.filter((b, k) => b[5] && status(f, k) !== "right").length;
    return `<div class="panel"><div class="ph"><b>${c}</b><span>${clock(f.ts)} · ${f.mine.length} yours, ${bad} not right · ${f.file}</span></div>
      <div class="frame" style="aspect-ratio:${f.w}/${f.h}"><img src="${f.src}" alt="${c}" onerror="this.style.visibility='hidden'">${html}</div></div>`;
  }).join("");
  document.getElementById("cams").innerHTML = CAMS.map(c => `<button data-cam="${c}" class="${shown.has(c) ? "on" : ""}">${c}</button>`).join(" ");
  inspect();
}
function shot(f, b, W, st) {                        // the whole frame, the box marked, the rest darker
  return `<div class="shot ${st || ""}" style="width:${W.toFixed(0)}px;aspect-ratio:${f.w}/${f.h}">` +
    `<img src="${f.src}" alt="${f.cam}" loading="lazy" onerror="this.style.visibility='hidden'">` +
    `<div class="mark" style="left:${b[0] * 100}%;top:${b[1] * 100}%;width:${(b[2] - b[0]) * 100}%;height:${(b[3] - b[1]) * 100}%"></div></div>`;
}
const view = (f, b, big, st) => zoom ? crop(f, b, big ? 220 : 110)
  : shot(f, b, big ? Math.min(760, window.innerWidth - 64) : Math.min(300, window.innerWidth - 64), st);
function strip(items) {
  const by = {};
  items.forEach(x => (by[x.f.cam] = by[x.f.cam] || []).push(x));
  return Object.keys(by).sort().map(c => `<div class="cam">${c} · ${by[c].length}</div><div class="strip">` +
    by[c].map(x => `<figure class="${x.here ? "here" : ""} ${x.st}" data-fi="${x.fi}" data-kind="${x.kind}" data-k="${x.k}">` +
      `${view(x.f, x.b, false, x.st)}<figcaption>${clock(x.f.ts)}<br>${x.cap}</figcaption></figure>`).join("") + `</div>`).join("");
}
function inspect() {
  const box = document.getElementById("insp");
  if (!sel) { box.innerHTML = ""; return; }
  const f = F[sel.fi], b = (sel.kind === "mine" ? f.mine : f.its)[sel.k];
  if (!b) { box.innerHTML = ""; return; }
  let facts, items = [], title;
  if (sel.kind === "mine") {
    const st = status(f, sel.k), [mm] = hitOf(f), j = mm[sel.k], c = j === undefined ? null : f.its[j][4];
    const words = { right: "the right cow", wrong: "another cow", missed: "not found", nocow: "found, no cow" }[st];
    facts = `<b>your ${b[4]}</b> → its <b>${c || "—"}</b> <span class="${st}" style="color:var(--c)">${words}</span><br>
      ${MAP[b[4]] ? `<span class="note">your ${b[4]} is its ${MAP[b[4]]} overall</span><br>` : ""}
      ${f.cam} · ${clock(f.ts)}${j !== undefined ? ` · track ${f.its[j][5]}` : ""}${b[5] ? "" : `<br><span class="note">outside ${f.cam}'s mask: not scored</span>`}`;
    title = `Your cow ${b[4]}, every camera and moment (its cow under each)`;
    F.forEach((g, fi) => g.mine.forEach((x, k) => {
      if (x[4] !== b[4]) return;
      const s = status(g, k), [gm] = hitOf(g), gj = gm[k];
      items.push({ f: g, fi, b: x, kind: "mine", k, st: s, here: fi === sel.fi && k === sel.k,
                   cap: gj === undefined ? "not found" : (g.its[gj][4] || "no cow") });
    }));
  } else {
    facts = `<b>its ${b[4] || "box without a cow"}</b> <span class="note">(no box of yours here)</span><br>${f.cam} · ${clock(f.ts)} · track ${b[5]}`;
    title = b[4] ? `Its cow ${b[4]}, every camera and moment (your name under each)` : "";
    if (b[4]) F.forEach((g, fi) => { const [, gt] = hitOf(g); g.its.forEach((x, j) => {
      if (x[4] !== b[4]) return;
      const yk = gt[j];
      items.push({ f: g, fi, b: x, kind: yk === undefined ? "its" : "mine", k: yk === undefined ? j : yk,
                   st: yk === undefined ? "extra" : status(g, yk), here: fi === sel.fi && j === sel.k,
                   cap: yk === undefined ? "not yours" : "your " + g.mine[yk][4] });
    }); });
  }
  const st0 = sel.kind === "mine" ? status(f, sel.k) : "extra";
  box.innerHTML = `<div class="top">${view(f, b, true, st0)}<div class="facts">${facts}</div></div>` +
    (title ? `<h3>${title}</h3>` + strip(items) : "");
}
document.addEventListener("click", e => {
  const t = e.target.closest("[data-fi]");
  if (t) {
    sel = { fi: +t.dataset.fi, kind: t.dataset.kind, k: +t.dataset.k };
    const mi = moments.findIndex(mo => Object.values(mo.at).includes(sel.fi));
    if (mi >= 0) m = mi;
    if (!shown.has(F[sel.fi].cam)) { shown.add(F[sel.fi].cam); save(); }
    render();
    return;
  }
  const c = e.target.closest("[data-cam]");
  if (c) {
    shown.has(c.dataset.cam) ? shown.delete(c.dataset.cam) : shown.add(c.dataset.cam);
    if (!shown.size) shown.add(c.dataset.cam);
    save(); render();
  }
});
document.getElementById("prev").onclick = () => go(-1);
document.getElementById("next").onclick = () => go(1);
sel_.onchange = () => { cow = sel_.value; sel = null; go(0); };
document.getElementById("errs").onchange = e => { errs = e.target.checked; go(0); };
document.getElementById("extra").onchange = e => { extra = e.target.checked; render(); };
document.getElementById("zoom").onchange = e => { zoom = e.target.checked; inspect(); };
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT") return;
  if (e.key === "ArrowLeft") go(-1);
  else if (e.key === "ArrowRight") go(1);
  else if (e.key === "Escape") { sel = null; render(); }
  else if (e.key === "z" || e.key === "Z") { const x = document.getElementById("zoom"); x.checked = !x.checked; zoom = x.checked; inspect(); }
  else if (e.key === "e" || e.key === "E") { const x = document.getElementById("errs"); x.checked = !x.checked; errs = x.checked; go(0); }
});
render();
</script></body></html>
"""


def overlap(cfg):
    """--overlap: each camera's mask its tracking_mask (the masks share floor), and the two rules that make a cow
    counted by two cameras at once two cows - [identity] counted_apart_s (never one cow in the gallery) and [mct]
    unlink_s (a cross-camera join undone) - off."""
    grown = []
    for cam in cfg["cameras"]:
        if cam.get("tracking_mask"):
            cam["mask"] = cam["tracking_mask"]
            grown.append(cam["id"])
    cfg["identity"]["counted_apart_s"] = 0
    cfg["mct"]["unlink_s"] = 0
    print(f"[trial] --overlap: {', '.join(grown) or 'no camera'} count their whole tracking_mask; a cow counted by "
          "two cameras at once may be one (identity.counted_apart_s 0, mct.unlink_s 0) - an experiment, its cows "
          "on shared floor are counted twice", flush=True)
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--videos", default=None, help="recordings named <camera>..<date and time>")
    p.add_argument("--out", required=True)
    p.add_argument("--burst-every", type=float, default=20.0, help="the first pass: a burst every this many s")
    p.add_argument("--min-bursts", type=int, default=6, help="bursts a cow needs in the first pass to get an id")
    p.add_argument("--show-minutes", type=float, default=10.0, help="minutes of every camera drawn (show-ids)")
    p.add_argument("--gap", type=float, default=300.0)
    p.add_argument("--boxes", default=None, help="your labelled frames (herd.py boxes: the folder with frames.json)")
    p.add_argument("--labels", default=None, help="their labels.json with cow ids: the second pass scored (IDF1)")
    p.add_argument("--score-only", action="store_true", help="score the trial already in --out, play nothing")
    p.add_argument("--cameras", nargs="*", default=None, help="score these cameras only (one id across them)")
    p.add_argument("--overlap", action="store_true",
                   help="an experiment: every camera counts its whole tracking_mask (masks share floor) and a cow "
                        "counted by two cameras at once may be one cow - does one cow get one id on both? "
                        "Never for a barn: shared floor counts its cows twice")
    args = p.parse_args(argv)
    if args.score_only:
        if not (args.boxes and args.labels):
            p.error("--score-only needs --boxes and --labels")
        cfg = overlap(load_config(args.config)) if args.overlap else load_config(args.config)
        out = os.path.abspath(args.out.rstrip("\\"))
        sc = score(os.path.join(out, "pass2", "herd.sqlite"), cfg, args.boxes, args.labels, page_dir=out,
                   cameras=args.cameras)
        name = "score.json" if not args.cameras else f"score_{'_'.join(args.cameras)}.json"
        with open(os.path.join(out, name), "w", encoding="utf-8") as fh:
            json.dump(sc, fh, indent=2)
        print_score(sc)
        print(f"-> {os.path.join(out, name)}")
        return
    if not args.videos:
        p.error("--videos is needed")
    import gallery as gallery_mod
    import show_ids
    from store import Store
    cfg = overlap(load_config(args.config)) if args.overlap else load_config(args.config)
    tz = dt.timezone(dt.timedelta(hours=cfg["farm"]["timezone_offset_hours"]))
    recs, bad = recordings(args.videos, [c["id"] for c in cfg["cameras"]], cfg["farm"]["timezone_offset_hours"])
    if not recs:
        sys.exit(f"no recordings under {args.videos} named <camera>..<YYYY-MM-DD_HH-MM-SS>")
    groups = sessions(recs, args.gap)
    out = os.path.abspath(args.out.rstrip("\\"))
    for d in ("pass1", "pass2"):
        shutil.rmtree(os.path.join(out, d), ignore_errors=True)
        os.makedirs(os.path.join(out, d))
    if cfg["sampling"]["fps"] != 1.0:
        print(f"[trial] !! [sampling] fps = {cfg['sampling']['fps']:g}: every camera writes that many rows a "
              f"second (posture, identity, the gallery - {cfg['sampling']['fps']:g}x the work, the counts in "
              "cow-seconds no more seconds). To follow walking cows more often set track_fps instead "
              "(the tracker only) and keep fps = 1", flush=True)
    cfg["training_cache"]["enabled"] = False
    cfg["identity"]["new_cow_min_bursts"] = args.min_bursts
    first = copy.deepcopy(cfg)
    first["sampling"]["burst_every_s"] = args.burst_every
    print(f"[trial] 1/3 the first pass ({len(recs)} recordings, a burst every {args.burst_every:g} s): the gallery "
          f"collects the cows", flush=True)
    models = play(first, groups, os.path.join(out, "pass1", "herd.sqlite"), None)
    # the night after it: the cows seen often enough get ids
    abstain = getattr(models, "id_abstain", None)
    folder = os.path.join(out, "pass1", "gallery")
    from reid import apply_thresholds
    from common import DEFAULTS
    apply_thresholds(first, getattr(models, "id_thresholds", {}) or {}, DEFAULTS, None)
    g = gallery_mod.Gallery(folder, first, abstain)
    last = max(ts for _, _, ts in recs) + 3600
    res = g.nightly(last, Store(os.path.join(out, "pass1", "herd.sqlite")).cannot_link_pairs())
    print(f"[trial] the night: {len(res['enrolled'])} cows enrolled ({res['pool']} bursts left unknown"
          + (f"; {res['held_back_herd_full']} more held back: max_cows reached" if res.get("held_back_herd_full") else "")
          + ")", flush=True)
    shutil.copytree(folder, os.path.join(out, "pass2", "gallery"))
    print("[trial] 2/3 the second pass, with those cows: who is who", flush=True)
    play(cfg, groups, os.path.join(out, "pass2", "herd.sqlite"), models)
    print("[trial] 3/3 drawn back", flush=True)
    store2 = os.path.join(out, "pass2", "herd.sqlite")
    shown = {}
    for cam, path, ts in recs:
        if cam in shown:
            continue
        dest = os.path.join(out, f"show_{cam}")
        try:
            show_ids.main(["--config", args.config, "--store", store2, "--cam", cam, "--video", path,
                           "--start", dt.datetime.fromtimestamp(ts, tz).isoformat(), "--minutes", str(args.show_minutes),
                           "--out", dest])
            shown[cam] = os.path.join(dest, "index.html")
        except SystemExit as e:
            print(f"[trial] {cam}: nothing drawn ({e})", flush=True)
    per_cam, cows, odd, links = summary(store2, cfg, os.path.join(out, "pass2", "gallery"))
    report = {"cows_enrolled": res["enrolled"], "cows_in_gallery": cows, "cameras": per_cam, "impossible": odd,
              "joined_across_cameras": links,
              "held_back_herd_full": res.get("held_back_herd_full", 0), "show": shown}
    with open(os.path.join(out, "trial.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[trial] {len(res['enrolled'])} cows made from the first pass")
    for cam, r in sorted(per_cam.items()):
        b = r["bursts"]
        print(f"  {cam}: {r['cow_seconds']} cow-seconds on {r['tracks']} tracks; {r['with_a_cow']:.0%} of them given a "
              f"cow ({r['cows']} cows); bursts confirmed {b.get('confirmed', 0)}, tentative {b.get('tentative', 0)}, "
              f"unknown {b.get('unknown', 0)}")
    n = max(1, odd["named_cow_seconds"])
    bad = odd["one_cow_twice_on_one_camera"] + odd["one_cow_on_cameras_apart"] + odd["one_cow_counted_by_two_cameras"]
    print(f"  what cannot be: one cow twice on one camera at once {odd['one_cow_twice_on_one_camera']} s, on two "
          f"cameras that share no floor at once {odd['one_cow_on_cameras_apart']} s, counted by two cameras at once "
          f"{odd['one_cow_counted_by_two_cameras']} s ({bad / n:.1%} of the named cow-seconds) - two cows given one "
          f"id (or masks that overlap); {len(odd['cows_seen_where_they_cannot_be'])} cows so")
    if cfg["mct"].get("enabled"):
        print("  tracks joined across cameras (one cow, one track): " +
              (", ".join(f"{k} {v}" for k, v in sorted(links.items())) or "none"))
    if res.get("held_back_herd_full"):
        print(f"  !! [identity] max_cows = {cfg['identity'].get('max_cows')}: the gallery was full, "
              f"{res['held_back_herd_full']} more cows got no id - set it to the cows these cameras see (0: no "
              "limit)")
    if args.boxes and args.labels:
        sc = score(store2, cfg, args.boxes, args.labels, page_dir=out, cameras=args.cameras)
        with open(os.path.join(out, "score.json"), "w", encoding="utf-8") as fh:
            json.dump(sc, fh, indent=2)
        print_score(sc)
    print(f"-> {os.path.join(out, 'trial.json')}; the videos: " + ", ".join(shown.values()))


if __name__ == "__main__":
    sys.exit(main())
