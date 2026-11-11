#!/usr/bin/env python3
"""Which frame width ([sampling] frame_width): quality against cost, Full HD down to 800 px.

The detector scales every frame to its own input size, so its whole-frame pass costs the GPU about the same at any
width (its tiles do not: they keep the frame's pixels). What the width changes is the rest: the video decode and
resize on the CPU (25 frames a second a camera), the 10 s ring buffer in RAM, and the crops of far cows - the
posture, activity and who-is-who read from them. So, at each width, on the barn's labelled frames (herd.py boxes:
frames.json + labels.json, cut at Full HD):

  detector   your boxes found (IoU 0.5) and its boxes right, by how far the cow is (her box's height)
  posture /  on your boxes that carry them (the box page, keys 1-6): the frame heads on the crop of your box
  activity   at that width - right share, lying and feeding found
  identity   the frame fingerprint (the identity model) of every named box: is her nearest box on another frame
             the same cow - over all, and only boxes of other cameras
  cost       ms a frame: resize from Full HD, detector, a crop, its encoding; with --video also the decode of
             that recording re-encoded at the width (ffmpeg) - a camera sending that size; RAM of the ring buffer

and an estimate of the cameras one machine keeps up with (GPU, CPU and RAM, the smallest): a rough sum of the costs,
not a run - check the widths you like with herd.py stress --videos (the barn's clips at that width).

    python herd/herd.py resolution --config barn.toml --boxes /data/boxes_first10 --labels /data/labels.json \\
        --widths 1920,1600,1280,1024,960,800 --video /data/rec/cam1_2026-10-23_17-36-30.mp4 --out /data/resolution
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402,F401  (puts cowbench on the path)
from common import ACTIVITIES, POSTURES, iou, load_config  # noqa: E402

SIZES = (("far", 0.0, 0.10), ("mid", 0.10, 0.25), ("near", 0.25, 9.0))      # a cow by her box's height (frame)


def labelled_frames(boxes_dir, labels_path, max_frames=0):
    """The frames marked done: [{file, cam, ts, path, boxes: [(box, cow id or "", {posture, activity})]}]."""
    from box_dataset import box_attributes, in_grey, read_labels
    index = json.load(open(os.path.join(boxes_dir, "frames.json"), encoding="utf-8"))
    labels = read_labels(labels_path)
    out = []
    for f in index["frames"]:
        lab = labels.get(f["file"])
        if not lab or lab.get("status") != "done" or (f.get("cut") and lab.get("cut") != f["cut"]):
            continue
        bs = [(list(b[:4]), str(b[4]).strip() if len(b) > 4 and str(b[4]).strip().lower() != "x" else "", box_attributes(b))
              for b in lab.get("boxes", []) if b[2] - b[0] > 0.002 and b[3] - b[1] > 0.002 and not in_grey(b, f.get("exclude"))]
        out.append(dict(f, path=os.path.join(boxes_dir, f["file"]), boxes=bs))
    if max_frames and len(out) > max_frames:                  # spread over cameras and time
        step = len(out) / max_frames
        out = [out[int(i * step)] for i in range(max_frames)]
    return out


def at_width(img, width):
    import cv2
    h, w = img.shape[:2]
    if w <= width:
        return img
    return cv2.resize(img, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def match(truth, found, min_iou=0.5):
    from scipy.optimize import linear_sum_assignment
    if not truth or not found:
        return []
    M = np.array([[iou(a, b) for b in found] for a in truth])
    r, c = linear_sum_assignment(-M)
    return [(int(i), int(j)) for i, j in zip(r, c) if M[i, j] >= min_iou]


def size_of(box):
    h = box[3] - box[1]
    return next(n for n, lo, hi in SIZES if lo <= h < hi)


def timed(fn, *a, **kw):
    t = time.perf_counter()
    r = fn(*a, **kw)
    return r, (time.perf_counter() - t) * 1000


def one_width(models, frames, width, cams, batch=8):
    """Everything measured at one width -> a dict."""
    import cv2
    from pipeline import crops_of
    det_n = collections.Counter()
    post, act, fps_ = collections.Counter(), collections.Counter(), []      # (yours, its); (cow, cam, frame, vector)
    ms = collections.defaultdict(list)
    imgs = []
    for f in frames:
        img = cv2.cvtColor(cv2.imread(f["path"]), cv2.COLOR_BGR2RGB)
        small, t = timed(at_width, img, width)
        ms["resize_from_full_hd"].append(t)
        imgs.append(small)
    # the detector, as the cameras run it (their own tiles), a few frames a pass
    for i in range(0, len(frames), batch):
        chunk = list(range(i, min(i + batch, len(frames))))
        by_tiles = collections.defaultdict(list)
        for k in chunk:
            by_tiles[cams.get(frames[k]["cam"], {}).get("tiles")].append(k)
        for tiles, ks in by_tiles.items():
            kw = {} if tiles is None else {"tiles": bool(tiles)}
            got, t = timed(models.detect_many, [imgs[k] for k in ks], urgent=True, kind="resolution", **kw)
            ms["detector_frame"].append(t / len(ks))
            for k, found in zip(ks, got):
                truth = [b for b, _, _ in frames[k]["boxes"]]
                hit = match(truth, found)
                det_n["boxes"] += len(found)
                det_n["right"] += len(hit)
                for j, b in enumerate(truth):
                    det_n[("yours", size_of(b))] += 1
                hit_t = {i for i, _ in hit}
                for j, b in enumerate(truth):
                    if j in hit_t:
                        det_n[("found", size_of(b))] += 1
    # your boxes' crops at this width: posture, activity, the fingerprint
    for k, f in enumerate(frames):
        if not f["boxes"]:
            continue
        boxes = [b for b, _, _ in f["boxes"]]
        crops, t = timed(crops_of, [(imgs[k], b) for b in boxes], models.crop_size, models.margin)
        ms["crop"].append(t / len(boxes))
        feats, t = timed(models.encode, crops, urgent=True, kind="resolution")
        ms["encode_crop"].append(t / len(boxes))
        (pp, pa, ids), t = timed(models.frame_heads, feats, boxes, with_id=True)
        ms["heads_box"].append(t / len(boxes))
        for (b, cow, att), p, a, e in zip(f["boxes"], pp, pa, ids):
            if att.get("posture"):
                post[(att["posture"], POSTURES[int(np.argmax(p))])] += 1
            if att.get("activity"):
                mine = "none" if att["activity"] == "ruminating" else att["activity"]
                act[(mine, ACTIVITIES[int(np.argmax(a))])] += 1
            if cow:
                e = np.asarray(e, np.float32)
                fps_.append((cow, f["cam"], k, e / (np.linalg.norm(e) + 1e-12), size_of(b)))
    out = {"width": width}
    yours = sum(v for k, v in det_n.items() if isinstance(k, tuple) and k[0] == "yours")
    found = sum(v for k, v in det_n.items() if isinstance(k, tuple) and k[0] == "found")
    rec, prec = found / max(1, yours), det_n["right"] / max(1, det_n["boxes"])
    out["detector"] = {"your_boxes": yours, "found": found, "recall": rec, "precision": prec,
                       "f1": 2 * rec * prec / max(1e-9, rec + prec),
                       "recall_by_size": {n: det_n[("found", n)] / det_n[("yours", n)] for n, _, _ in SIZES if det_n[("yours", n)]},
                       "boxes_by_size": {n: det_n[("yours", n)] for n, _, _ in SIZES}}
    share = lambda c, keep=lambda y: True: (sum(v for (y, i), v in c.items() if y == i and keep(y)) /
                                            max(1, sum(v for (y, i), v in c.items() if keep(y))))
    out["posture"] = {"boxes": sum(post.values()), "right": share(post), "lying_found": share(post, lambda y: y == "lying"),
                      "standing_found": share(post, lambda y: y == "standing")}
    out["activity"] = {"boxes": sum(act.values()), "right": share(act), "feeding_found": share(act, lambda y: y == "feeding")}
    out["identity"] = identity_top1(fps_)
    out["ms"] = {k: float(np.median(v)) for k, v in ms.items() if v}
    return out


def crop_side(policy, img, box, margin, full=224, low=112, step=14):
    """A crop's side in pixels for the encoder: policy "224" / "168" / ... that size for every cow; "native" the box's
    own pixels (its square crop at this frame width) rounded to the encoder's patch (14), within low..full - a far
    cow is not blown up to 224, a near one is shrunk to it as ever."""
    if policy != "native":
        return int(policy)
    from common import square_crop_box
    h, w = img.shape[:2]
    x1, y1, x2, y2 = square_crop_box(box, w, h, margin)
    return int(min(full, max(low, round((x2 - x1) / step) * step)))


def crop_policies(models, frames, imgs, policies):
    """Your boxes' crops at each crop-size policy: posture, activity, the fingerprint's same-cow nearest, and the
    encoder's ms a crop (crops of one size encoded together, as a batch of the pipeline would be)."""
    from common import crop
    out = {}
    for pol in policies:
        post, act, fps_, sides = collections.Counter(), collections.Counter(), [], []
        enc_ms, n_crops = 0.0, 0
        for k, f in enumerate(frames):
            if not f["boxes"]:
                continue
            boxes = [b for b, _, _ in f["boxes"]]
            size = [crop_side(pol, imgs[k], b, models.margin, models.crop_size) for b in boxes]
            sides += size
            feats = [None] * len(boxes)
            for sz in sorted(set(size)):
                idx = [i for i, z in enumerate(size) if z == sz]
                cs = [crop(imgs[k], boxes[i], sz, models.margin) for i in idx]
                fe, t = timed(models.encode, cs, urgent=True, kind="resolution")
                enc_ms += t
                n_crops += len(idx)
                for i, v in zip(idx, fe):
                    feats[i] = v
            try:
                import torch
                F = torch.stack(feats)
            except (ImportError, TypeError):
                F = np.stack([np.asarray(v) for v in feats])
            pp, pa, ids = models.frame_heads(F, boxes, with_id=True)
            for (b, cow, att), p, a, e in zip(f["boxes"], pp, pa, ids):
                if att.get("posture"):
                    post[(att["posture"], POSTURES[int(np.argmax(p))])] += 1
                if att.get("activity"):
                    mine = "none" if att["activity"] == "ruminating" else att["activity"]
                    act[(mine, ACTIVITIES[int(np.argmax(a))])] += 1
                if cow:
                    e = np.asarray(e, np.float32)
                    fps_.append((cow, f["cam"], k, e / (np.linalg.norm(e) + 1e-12), size_of(b)))
        share = lambda c, keep=lambda y: True: (sum(v for (y, i), v in c.items() if y == i and keep(y)) /
                                                max(1, sum(v for (y, i), v in c.items() if keep(y))))
        side = np.array(sides, float) if sides else np.zeros(1)
        out[pol] = {"posture_right": share(post), "lying_found": share(post, lambda y: y == "lying"),
                    "activity_right": share(act), "feeding_found": share(act, lambda y: y == "feeding"),
                    "identity": identity_top1(fps_), "encode_ms_a_crop": enc_ms / max(1, n_crops),
                    "patches_a_crop": float(np.mean((side / 14) ** 2)), "side_median": float(np.median(side)),
                    "sides": {int(z): int(c) for z, c in zip(*np.unique(side, return_counts=True))}}
    full = out.get(str(models.crop_size)) or next(iter(out.values()))
    for r in out.values():
        r["patches_vs_full"] = r["patches_a_crop"] / max(1e-9, full["patches_a_crop"])
    return out


def identity_top1(items):
    """Each named box against every named box of another frame: is the nearest one the same cow (all cameras /
    other cameras only), over all and by how far she is."""
    if len(items) < 3:
        return {}
    cows = [c for c, *_ in items]
    cams = [m for _, m, *_ in items]
    frames = np.array([k for _, _, k, _, _ in items])
    E = np.stack([e for *_, e, _ in items])
    S = E @ E.T
    S[frames[:, None] == frames[None, :]] = -2                       # not a box of the same frame (nor herself)
    other = np.array(cams)[:, None] != np.array(cams)[None, :]
    res = collections.defaultdict(list)
    for q in range(len(items)):
        if cows.count(cows[q]) < 2:
            continue
        j = int(np.argmax(S[q]))
        res["all"].append(cows[j] == cows[q])
        res["all_" + items[q][4]].append(cows[j] == cows[q])
        if any(cows[i] == cows[q] and other[q, i] for i in range(len(items))):
            s = np.where(other[q], S[q], -2)
            res["other_camera"].append(cows[int(np.argmax(s))] == cows[q])
    out = {k: float(np.mean(v)) for k, v in res.items() if v}
    out["questions"] = len(res["all"])
    out["cows"] = len(set(cows))
    return out


def decode_ms(video, width, seconds=20.0):
    """ms to decode a frame of the recording re-encoded at this width (a camera sending that size), and at its own."""
    import cv2
    if not shutil.which("ffmpeg"):
        return None
    tmp = tempfile.mkdtemp()
    try:
        out = os.path.join(tmp, f"w{width}.mp4")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-t", str(seconds), "-i", video, "-vf", f"scale={width}:-2",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an", out], check=True)
        cap = cv2.VideoCapture(out)
        n, t = 0, time.perf_counter()
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        cap.release()
        return (time.perf_counter() - t) * 1000 / max(1, n)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def burst_ms(models, frames):
    """One cow's burst on the GPU (its ~175 crops encoded, the temporal model): ms."""
    from pipeline import crops_of
    import cv2
    f = next((f for f in frames if f["boxes"]), None)
    if f is None:
        return None
    img = cv2.cvtColor(cv2.imread(f["path"]), cv2.COLOR_BGR2RGB)
    b = f["boxes"][0][0]
    n = 175
    crops = crops_of([(img, b)] * n, models.crop_size, models.margin)
    models.encode(crops[:8])                                        # warm
    t = time.perf_counter()
    feats = models.encode(crops)
    models.burst(feats, np.arange(n) / 25.0, np.ones(n, bool), [b] * n)
    return (time.perf_counter() - t) * 1000


def capacity(r, cfg, cows, burst_cow_ms, cores, ram_gb, aspect):
    """Cameras one machine keeps up with at this width, by GPU, CPU and RAM (rough sums of the costs)."""
    s = cfg["sampling"]
    fps, every, bfps, bs = s["fps"], s["burst_every_s"], s["burst_detect_fps"], s["burst_seconds"]
    ms = r["ms"]
    det, enc, heads, crop = ms.get("detector_frame", 0), ms.get("encode_crop", 0), ms.get("heads_box", 0), ms.get("crop", 0)
    crops_burst = s["burst_fps"] * bs * cows
    gpu = fps * (det + cows * (enc + heads)) + (bfps * bs * det + (burst_cow_ms or crops_burst / cows * enc) * cows) / every
    cpu_crops = fps * cows * crop + crops_burst * crop / every
    out = {"cows_a_frame": cows, "gpu_ms_a_camera_second": gpu, "cpu_crops_ms_a_camera_second": cpu_crops}
    out["cameras_by_gpu"] = int(0.8 * 1000 / gpu) if gpu else None
    for name, dec in (("stream_at_width", r.get("decode_ms")), ("full_hd_stream_resized", r.get("decode_full_hd_ms"))):
        if dec is None:
            continue
        per = 25 * (dec + (ms.get("resize_from_full_hd", 0) if name == "full_hd_stream_resized" else 0)) + cpu_crops
        out[f"cpu_ms_a_camera_second_{name}"] = per
        out[f"cameras_by_cpu_{name}"] = int(0.7 * cores * 1000 / per)
    h = int(round(r["width"] / aspect))
    ring = r["width"] * h * 3 * 25 * 10
    out["ring_buffer_mb"] = ring / 2 ** 20
    if ram_gb:
        out["cameras_by_ram"] = int(0.7 * ram_gb * 2 ** 30 / (ring + 300 * 2 ** 20))   # + ~300 MB a camera besides
    limits = [v for k, v in out.items() if k.startswith("cameras_by_") and v is not None]
    out["cameras"] = min(limits) if limits else None
    return out


def ram_gb():
    try:
        return int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0]) / 2 ** 20
    except (OSError, IndexError, ValueError):
        return None


def cv2_rgb(path):
    import cv2
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def crop_table(crop_res):
    rows = ["| frame width | crop size | posture right (lying found) | activity right | the same cow nearest (far cows) | "
            "encoder work (patches vs 224) | ms a crop | crop sides |", "|---|---|---|---|---|---|---|---|"]
    for w, pols in crop_res.items():
        for p, r in pols.items():
            i = r["identity"]
            rows.append(f"| {w} | {p} | {r['posture_right']:.0%} ({r['lying_found']:.0%}) | {r['activity_right']:.0%} | "
                        f"{i.get('all', 0):.1%} ({i.get('all_far', 0):.1%}) | {r['patches_vs_full']:.0%} | "
                        f"{r['encode_ms_a_crop']:.2f} | " + ", ".join(f"{z}: {c}" for z, c in r["sides"].items()) + " |")
    return "\n".join(rows)


def table(res):
    rows = ["| width | detector: found / right (far, mid, near found) | posture right (lying found) | activity right "
            "(feeding found) | the same cow nearest (other camera) | cameras (GPU / CPU / RAM) |",
            "|---|---|---|---|---|---|"]
    pc = lambda x: "-" if x is None else f"{x:.0%}"
    for r in res:
        d, p, a, i, c = r["detector"], r["posture"], r["activity"], r["identity"], r["capacity"]
        bs = d["recall_by_size"]
        cpu = c.get("cameras_by_cpu_stream_at_width", c.get("cameras_by_cpu_full_hd_stream_resized"))
        rows.append(f"| {r['width']} | {pc(d['recall'])} / {pc(d['precision'])} ({pc(bs.get('far'))}, {pc(bs.get('mid'))}, "
                    f"{pc(bs.get('near'))}) | {pc(p['right'] if p['boxes'] else None)} ({pc(p['lying_found'] if p['boxes'] else None)}) | "
                    f"{pc(a['right'] if a['boxes'] else None)} ({pc(a['feeding_found'] if a['boxes'] else None)}) | "
                    f"{pc(i.get('all'))} ({pc(i.get('other_camera'))}) | {c.get('cameras_by_gpu')} / {cpu if cpu is not None else '-'} / "
                    f"{c.get('cameras_by_ram', '-')} |")
    return "\n".join(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--boxes", required=True, help="herd.py boxes' folder (frames.json, cut at Full HD)")
    p.add_argument("--labels", required=True, help="labels.json (frames marked done; ids, posture, activity)")
    p.add_argument("--widths", default="1920,1600,1280,1024,960,800")
    p.add_argument("--video", default=None, help="a recording of a camera: its decode timed at each width (ffmpeg)")
    p.add_argument("--frames", type=int, default=0, help="at most this many labelled frames (0: all)")
    p.add_argument("--crops", default="224,native,168,112",
                   help="crop-size policies to compare: a side in px for every cow, or native (the box's own pixels, "
                        "112-224)")
    p.add_argument("--crop-widths", default="1920,1280", help="the frame widths the crop policies are compared at")
    p.add_argument("--cores", type=int, default=os.cpu_count() or 1, help="CPU cores for the cameras")
    p.add_argument("--out", default=None, help="where resolution.json / .md go (default: next to --boxes)")
    args = p.parse_args(argv)
    import pipeline
    cfg = load_config(args.config)
    widths = sorted({int(w) for w in args.widths.split(",") if w.strip()}, reverse=True)
    frames = labelled_frames(args.boxes, args.labels, args.frames)
    if not frames:
        sys.exit(f"no frames marked done in {args.labels}")
    src_w = max(f.get("w", 1920) for f in frames)
    if widths[0] > src_w:
        print(f"[resolution] the frames were cut {src_w} px wide: wider than that is the same picture", flush=True)
    aspect = np.median([f.get("w", 1920) / f.get("h", 1080) for f in frames])
    cams = {c["id"]: c for c in cfg["cameras"]}
    cows = float(np.mean([len(f["boxes"]) for f in frames]))
    print(f"[resolution] {len(frames)} labelled frames, {sum(len(f['boxes']) for f in frames)} of your boxes "
          f"({cows:.1f} a frame); widths {widths}", flush=True)
    models = pipeline.Models(cfg)
    one_width(models, frames[:2], widths[0], cams)                   # warm the GPU
    b_ms = burst_ms(models, frames)
    full_dec = decode_ms(args.video, src_w) if args.video else None
    gb, res = ram_gb(), []
    for w in widths:
        t = time.time()
        r = one_width(models, frames, w, cams)
        if args.video:
            r["decode_ms"] = decode_ms(args.video, w)
            r["decode_full_hd_ms"] = full_dec
        r["capacity"] = capacity(r, cfg, cows, b_ms, args.cores, gb, aspect)
        res.append(r)
        d, i = r["detector"], r["identity"]
        print(f"[resolution] {w} px ({time.time() - t:.0f} s): detector found {d['recall']:.1%}, right {d['precision']:.1%}; "
              f"posture {r['posture']['right']:.1%} of {r['posture']['boxes']}; the same cow nearest {i.get('all', 0):.1%} "
              f"(other camera {i.get('other_camera', 0):.1%}); about {r['capacity']['cameras']} cameras", flush=True)
    crop_res = {}
    pols = [x.strip() for x in args.crops.split(",") if x.strip()]
    for w in sorted({int(x) for x in args.crop_widths.split(",") if x.strip()}, reverse=True):
        if not pols:
            break
        imgs = [at_width(cv2_rgb(f["path"]), w) for f in frames]
        t = time.time()
        crop_res[w] = crop_policies(models, frames, imgs, pols)
        print(f"[resolution] crops at {w} px ({time.time() - t:.0f} s): " + "; ".join(
            f"{p}: posture {r['posture_right']:.1%} (lying {r['lying_found']:.0%}), same cow {r['identity'].get('all', 0):.1%}, "
            f"{r['patches_vs_full']:.0%} of the patches, {r['encode_ms_a_crop']:.2f} ms a crop" for p, r in crop_res[w].items()),
            flush=True)
    out_dir = args.out or os.path.dirname(os.path.abspath(os.path.join(args.boxes, "frames.json")))
    os.makedirs(out_dir, exist_ok=True)
    report = {"frames": len(frames), "your_boxes": sum(len(f["boxes"]) for f in frames), "burst_cow_ms": b_ms,
              "decode_full_hd_ms": full_dec, "cores": args.cores, "ram_available_gb": gb, "widths": res,
              "crops": {str(w): r for w, r in crop_res.items()}}
    with open(os.path.join(out_dir, "resolution.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)
    md = table(res) + ("\n\n" + crop_table(crop_res) if crop_res else "")
    with open(os.path.join(out_dir, "resolution.md"), "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    print("\n" + md)
    print(f"\n(cameras: a rough sum of the measured costs at {cfg['sampling']['fps']:g} tick/s, a "
          f"{cfg['sampling']['burst_seconds']:g} s burst every {cfg['sampling']['burst_every_s']:g} s, {cows:.1f} cows a "
          f"frame; CPU {args.cores} cores{'' if args.video else ' - no --video: decode not timed'}. Check the ones you "
          f"like: herd.py stress --videos <clips at that width> --cameras N)")
    print(f"-> {os.path.join(out_dir, 'resolution.json')}, resolution.md")


if __name__ == "__main__":
    sys.exit(main())
