"""The barn, live: IP cameras (RTSP) or recorded files -> SQLite (store.py).

Per camera, every frame goes into a ring buffer of the last few seconds;
nothing is analysed at 25 fps except bursts:

  every 1/fps s (1 s)   detector (RT-DETRv2) -> camera mask -> tracker -> crops
                        -> frame encoder -> posture, activity  -> table seconds
  every burst_every_s   the last burst_seconds of the buffer (7 s, 25 fps):
  (60 s, staggered      detector at burst_detect_fps, linked into one box chain
   over the cameras)    per cow, interpolated to every frame -> 175 crops
                        -> frame encoder -> temporal model -> fingerprint,
                        quality, rumination, posture, activity, lameness
                        (+ the rhythm of the crops, motion.py: chewing)
                        -> ruminating yes / no (heads.json cut-off)
                        -> gallery: confirmed / tentative / unknown  -> table bursts
  nightly (02:00)       gallery: retire, rebuild prototypes, enroll new cows
  Tuesday 10:00         change-over starts (gallery.migration_due)

Cameras see the barn from above and partly overlap: each camera's mask (in
config) covers only the floor it owns, so a cow is counted by one camera.
With [mct] enabled each camera tracks in its wider tracking_mask and one
GlobalTracker (mct.py) gives a cow one track across cameras by her frame id;
seconds and bursts are still written only by the camera whose mask owns her.

    python pipeline.py --config barn.toml                     # live, RTSP urls in config
    python pipeline.py --config barn.toml --start 2026-10-07T06:00:00   # files, as recorded at that time
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import datetime as dt
import json
import math
import os
import sys
import threading
import time

import numpy as np

from common import (ACTIVITIES, DEFAULTS, POSTURES, BoxKalman, blank_excluded, crop, device, in_mask, in_zone,
                    interpolate_box, iou, load_config, same_place)
from motion import motion_features


# ------------------------------------------------------------------ tracker

class Tracker:
    """IoU tracker for 1 fps: cows move little in a second, seen from above.

    The detector loses lying cows for seconds to minutes: her box back at the same place (common.same_place)
    within still_s is her, and within still_long_s when she had not moved for still_for_s before - one
    that walked is not waited for that long (another cow may lie down where she was)."""

    def __init__(self, cam, min_iou=0.3, max_age_s=5, still_s=0.0, still_iou=0.5, still_contain=0.8,
                 still_for_s=30.0, still_moved=0.3, still_long_s=0.0, kalman=False, low_iou=0.5):
        self.cam, self.min_iou, self.max_age = cam, min_iou, max_age_s
        self.kalman, self.low_iou = kalman, low_iou
        self.still_s, self.still_iou, self.still_contain = still_s, still_iou, still_contain
        self.still_long = still_long_s
        self.still_for, self.still_moved = still_for_s, still_moved
        self.tracks = {}            # id -> {"box", "last", "anchor", "since"}
        self.history = collections.defaultdict(lambda: collections.deque(maxlen=120))
        self.n, self.prev = 0, None

    def _held(self, t, ts):
        """Still waited for at ts, lost at the same place."""
        age = ts - t["last"]
        return age <= self.still_s or (age <= self.still_long and t["last"] - t["since"] >= self.still_for)

    def _seen(self, tid, ts, box):
        prev = self.tracks.get(tid)
        anchor, since = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), ts
        if prev is not None and math.hypot(prev["anchor"][0] - anchor[0], prev["anchor"][1] - anchor[1]) \
                <= self.still_moved * max(box[3] - box[1], 1e-6):
            anchor, since = prev["anchor"], prev["since"]
        kf = None
        if self.kalman:
            kf = prev.get("kf") if prev is not None and ts - prev["last"] <= self.max_age else None
            if kf is None:
                kf = BoxKalman(box, ts)
            else:
                kf.update(ts, box)
        self.tracks[tid] = {"box": box, "last": ts, "anchor": anchor, "since": since, "kf": kf}

    def _expected(self, t, ts):
        return t["kf"].predict(ts) if t.get("kf") is not None else t["box"]

    def update(self, ts, boxes, low=()):
        """boxes -> [(track, box)]; low: boxes under the detector's threshold, each only continuing a track no sure
        box took (seen a moment ago where it is expected, IoU low_iou, or lying still at her place) - ByteTrack's
        second round; a low box matching none is dropped."""
        live = lambda t: self.prev is None or self.prev - t["last"] <= self.max_age   # alive at the last second
        near = lambda t, b: max(iou(self._expected(t, ts), b), iou(t["box"], b))   # a cow may stop short of the guess
        pairs = sorted(((near(t, b), tid, j) for tid, t in self.tracks.items()
                        if live(t) for j, b in enumerate(boxes)), reverse=True)
        used_t, used_b, out = set(), set(), []
        for v, tid, j in pairs:
            if v < self.min_iou or tid in used_t or j in used_b:
                continue
            used_t.add(tid), used_b.add(j)
            self._seen(tid, ts, boxes[j])
            out.append((tid, boxes[j]))
        # lying still, lost a while: back at the same place
        held = sorted(((iou(t["box"], b), tid, j) for tid, t in self.tracks.items()
                       if tid not in used_t and not live(t) and self._held(t, ts)
                       for j, b in enumerate(boxes) if j not in used_b), reverse=True)
        for v, tid, j in held:
            if tid in used_t or j in used_b or not same_place(self.tracks[tid]["box"], boxes[j], self.still_iou,
                                                               self.still_contain):
                continue
            used_t.add(tid), used_b.add(j)
            self._seen(tid, ts, boxes[j])
            out.append((tid, boxes[j]))
        for j, b in enumerate(boxes):
            if j not in used_b:
                self.n += 1
                tid = f"{self.cam}-{int(ts)}-{self.n}"
                self._seen(tid, ts, b)
                out.append((tid, b))
        if low:
            cand = sorted(((near(t, b) if live(t) else 0.0, tid, j)
                           for tid, t in self.tracks.items() if tid not in used_t and (live(t) or self._held(t, ts))
                           for j, b in enumerate(low)), reverse=True)
            used_l = set()
            for v, tid, j in cand:
                if tid in used_t or j in used_l:
                    continue
                t = self.tracks[tid]
                if v < self.low_iou and not (not live(t) and self._held(t, ts)
                                             and same_place(t["box"], low[j], self.still_iou, self.still_contain)):
                    continue
                used_t.add(tid), used_l.add(j)
                self._seen(tid, ts, low[j])
                out.append((tid, low[j]))
        self.tracks = {k: v for k, v in self.tracks.items()
                       if ts - v["last"] <= self.max_age or self._held(v, ts)}
        for tid, b in out:
            self.history[tid].append((ts, b))
        self.prev = ts
        return out

    def track_at(self, ts, box):
        """The track whose box near time ts overlaps box most (for burst chains)."""
        best, score = None, 0.0
        for tid, h in self.history.items():
            near = min(h, key=lambda x: abs(x[0] - ts), default=None)
            if near and abs(near[0] - ts) <= 2.0:
                v = iou(near[1], box)
                if v > score:
                    best, score = tid, v
        return best if score >= self.min_iou else None


# ------------------------------------------------------------------- models

class GpuLock:
    """One GPU, many camera threads. The once-a-second work (urgent) goes before
    a waiting burst batch, so a big burst never holds up the 1 fps ticks for
    longer than one batch. Counts the time it is held: the GPU's busy share."""

    def __init__(self):
        self.cv = threading.Condition()
        self.busy, self.urgent_waiting = False, 0
        self.held = collections.Counter()           # kind -> seconds held

    @contextlib.contextmanager
    def __call__(self, urgent=False, kind="other"):
        with self.cv:
            self.urgent_waiting += urgent
            while self.busy or (not urgent and self.urgent_waiting):
                self.cv.wait()
            self.urgent_waiting -= urgent
            self.busy = True
        t = time.perf_counter()
        try:
            yield
        finally:
            dt_ = time.perf_counter() - t
            with self.cv:
                self.held[kind] += dt_
                self.busy = False
                self.cv.notify_all()


class Models:
    """Detector, frame encoder and herd model, shared by all cameras (one GPU)."""

    def __init__(self, cfg, detector=None):
        import torch
        from model import FrameEncoder, HerdModel
        self.gpu = GpuLock()
        self.dev = device()
        ck_path = cfg["model"]["checkpoint"]
        for what, path, needs in (("[model] checkpoint", ck_path, None),
                                  ("[detector] weights", cfg["detector"]["weights"], "det_train_meta.json")):
            if detector is not None and needs:
                continue
            if not path or not os.path.exists(os.path.join(path, needs) if needs else path):
                raise SystemExit(f'{what} = "{path}": no such {"best/ folder" if needs else "file"} - set it in the '
                                 f'config (in Docker: `models` lists what the image has, `init` fills it in)')
        self.model, ck = HerdModel.load(ck_path, map_location=self.dev)
        self.model.to(self.dev).eval()
        feat = ck.get("features", {})
        self.features_meta = {k: feat.get(k, d) for k, d in (("encoder", cfg["model"]["encoder"]),
                              ("grid", cfg["model"]["grid"]), ("crop", cfg["model"]["crop"]), ("margin", 0.1),
                              ("dim", None))}
        self.crop_size = feat.get("crop", cfg["model"]["crop"])
        self.margin = feat.get("margin", 0.1)          # the context the model was trained with
        self.encoder = FrameEncoder(feat.get("encoder", cfg["model"]["encoder"]),
                                    feat.get("grid", cfg["model"]["grid"])).to(self.dev).eval()
        if detector is not None:
            self.detector = detector
        else:
            import detector as det_mod
            self.detector = det_mod.Live(cfg["detector"]["weights"], cfg["detector"]["threshold"],
                                         tiles=cfg["detector"].get("tiles"))
        # [tracker] low_score: the detector keeps boxes down to it; those under its own threshold go to the tracker
        # apart (ByteTrack's second round), never to the bursts
        self.det_high = 0.0
        low = cfg["tracker"].get("low_score") or 0.0
        thr = getattr(self.detector, "threshold", None)
        if low and thr is not None and low < thr:
            self.det_high, self.detector.threshold = thr, low
        self.lameness = cfg["lameness"]["enabled"]
        self.motion_dim = self.model.motion_dim
        # rumination.py's model, when the config names one: it reads the same frame vectors
        # (it must have been trained on this encoder's) and decides rumination in place of the head
        self.rum_model = None
        rp = cfg["model"].get("rumination")
        if rp:
            from model import TemporalModel
            rk = torch.load(rp, map_location=self.dev, weights_only=False)
            want = {k: rk["features"].get(k) for k in ("encoder", "grid", "crop", "margin")}
            have = {k: self.features_meta.get(k) for k in want}
            if want != have:
                raise SystemExit(f"{rp} was trained on frame vectors {want}, the herd model's are {have}")
            self.rum_model = TemporalModel(rk["in_dim"], **rk["kw"]).to(self.dev).eval()
            self.rum_model.load_state_dict(rk["state"])
            self.rum_model_thr = float(rk["threshold"])
            self.motion_dim = max(self.motion_dim, rk["kw"].get("motion_dim", 0))
        # reid.py's identity model, when the config names one: the fingerprint (bursts, and one frame a
        # second for mct.py) comes from it - taught on cows with ids - and the gallery takes its abstain.json
        self.id_model, self.id_abstain, self.id_thresholds = None, None, {}
        ip = cfg["model"].get("identity")
        if ip:
            from reid import identity_thresholds, load_identity
            self.id_model, self.id_abstain = load_identity(ip, self.features_meta, self.dev)
            self.id_thresholds = identity_thresholds(ip)   # its own similarity thresholds (reid.calibrate)
        # The cut-off calibrated with the model: "ruminating" per burst, summed
        # into minutes by the reports (the one that keeps the share true).
        try:
            h = json.load(open(os.path.join(os.path.dirname(os.path.abspath(ck_path)), "heads.json"), encoding="utf-8"))
            self.rum_thr = float(h.get("rumination_threshold_time", h["rumination_threshold"]))
        except (OSError, KeyError, ValueError):
            self.rum_thr = float(ck.get("rum_threshold") or 0.5)
        self.torch = torch

    def _sync(self):
        # the lock must cover the GPU work itself, not just queueing it
        if self.dev.type == "cuda":
            self.torch.cuda.synchronize()

    def detect(self, img, urgent=False, kind="detect"):
        return self.detect_many([img], urgent=urgent, kind=kind)[0]

    def detect_split(self, img, urgent=False, kind="detect"):
        """(sure boxes, low boxes): low ones under the detector's threshold, down to [tracker] low_score."""
        return self.detect_many([img], urgent=urgent, kind=kind, split=True)[0]

    def detect_many(self, imgs, batch=8, urgent=False, kind="detect", split=False):
        from PIL import Image
        out = []
        many = getattr(self.detector, "many", None)
        step = batch if many else 1
        for i in range(0, len(imgs), step):
            pils = [Image.fromarray(x) for x in imgs[i:i + step]]
            with self.gpu(urgent, kind):
                res = many(pils) if many else [self.detector(pils[0])]
            for r in res:
                sure = [c["bbox"] for c in r if c.get("det_score", 1.0) >= self.det_high]
                out.append((sure, [c["bbox"] for c in r if c.get("det_score", 1.0) < self.det_high]) if split else sure)
        return out

    def encode(self, crops, batch=256, urgent=False, kind="encode"):
        torch = self.torch
        out = []
        for i in range(0, len(crops), batch):
            x = torch.from_numpy(np.ascontiguousarray(np.stack(crops[i:i + batch])))
            with self.gpu(urgent, kind), torch.no_grad():
                x = x.to(self.dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dev.type == "cuda"):
                    out.append(self.encoder(x).float())
                self._sync()
        return torch.cat(out) if out else torch.zeros((0, self.encoder.out_dim), device=self.dev)

    def frame_heads(self, feats, boxes, urgent=True, kind="heads", with_id=False):
        """Posture and activity per box; with_id also the frame id (model.frame_fingerprint)."""
        from model import box_pos
        with self.gpu(urgent, kind), self.torch.no_grad():
            pos = self.torch.tensor([box_pos(b) for b in boxes], device=self.dev).float()
            o = self.model.frame(feats, pos)
            pp, pa = o["posture"].softmax(-1).cpu().numpy(), o["activity"].softmax(-1).cpu().numpy()
            if not with_id:
                return pp, pa
            if self.id_model is not None:
                x = feats.float().unsqueeze(1)
                o = self.id_model(x, x.new_zeros(len(x), 1), self.torch.ones(len(x), 1, dtype=self.torch.bool,
                                                                               device=self.dev), pos)
                return pp, pa, o["fingerprint"].float().cpu().numpy()
            return pp, pa, self.model.frame_fingerprint(feats.float(), pos).float().cpu().numpy()

    def burst(self, feats, times, valid, boxes, motion=None, urgent=False, kind="temporal"):
        from model import box_pos
        torch = self.torch
        with self.gpu(urgent, kind), torch.no_grad():
            pos = torch.tensor([box_pos(b) for b in boxes], device=self.dev).float().unsqueeze(0)
            mot = None if motion is None else torch.as_tensor(motion, device=self.dev).float().unsqueeze(0)
            o = self.model.temporal(feats.unsqueeze(0), torch.as_tensor(times, device=self.dev).unsqueeze(0).float(),
                                    torch.as_tensor(valid, device=self.dev).unsqueeze(0), pos, mot)
            out = {"fingerprint": o["fingerprint"][0].cpu().numpy(),
                   "quality_max": float(o["weights"][0].max()),
                   "quality_mean": float(o["quality"][0][torch.as_tensor(valid, device=self.dev)].mean()),
                   "rumination": float(o["rumination"][0].sigmoid()),
                   "posture": POSTURES[int(o["posture"][0].argmax())],
                   "activity": ACTIVITIES[int(o["activity"][0].argmax())],
                   "lameness": float(o["lameness"][0]) if self.lameness else None}
            if self.rum_model is not None:
                r = self.rum_model(feats.unsqueeze(0).float(), torch.as_tensor(times, device=self.dev).unsqueeze(0).float(),
                                   torch.as_tensor(valid, device=self.dev).unsqueeze(0),
                                   motion=mot if self.rum_model.motion_dim else None)
                out["rumination"] = float(r["rumination"][0].sigmoid())
            if self.id_model is not None:
                i = self.id_model(feats.unsqueeze(0).float(), torch.as_tensor(times, device=self.dev).unsqueeze(0).float(),
                                  torch.as_tensor(valid, device=self.dev).unsqueeze(0), pos)
                out["fingerprint"] = i["fingerprint"][0].cpu().numpy()
                out["quality_max"] = float(i["weights"][0].max())
                out["quality_mean"] = float(i["quality"][0][torch.as_tensor(valid, device=self.dev)].mean())
        if self.rum_model is not None:
            out["ruminating"] = bool(out["rumination"] >= self.rum_model_thr)
            return out
        # a cow that feeds or drinks is not ruminating, whatever the head says
        out["ruminating"] = bool(out["rumination"] >= self.rum_thr and out["activity"] == "none")
        return out


# ------------------------------------------------------------------- camera

def link_chains(stamped, iou_thr):
    """A burst's detections [(t, [boxes])] in time order -> chains [{t: box}]: each box goes to the
    chain whose last box overlaps it most (IoU >= iou_thr), the rest start chains of their own."""
    chains = []
    for ts, boxes in stamped:
        used = set()
        for ch in chains:
            last_t = max(ch)
            cands = [(iou(ch[last_t], b), j) for j, b in enumerate(boxes) if j not in used]
            if cands:
                v, j = max(cands)
                if v >= iou_thr:
                    ch[ts] = boxes[j]
                    used.add(j)
        chains += [{ts: b} for j, b in enumerate(boxes) if j not in used]
    return chains


_CROP_POOL = None


def cpu_pool():
    global _CROP_POOL
    if _CROP_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _CROP_POOL = ThreadPoolExecutor(max(1, min(8, (os.cpu_count() or 2) - 1)), thread_name_prefix="crop")
    return _CROP_POOL


def crops_of(pairs, size, margin):
    """[(frame, box)] -> crops, in threads: PIL's resize lets go of the GIL, and a
    burst is ~175 crops a cow - serial, the slowest part of it. Same pixels."""
    cpu_pool()
    if len(pairs) < 8:
        return [crop(img, b, size, margin) for img, b in pairs]
    return list(_CROP_POOL.map(lambda p: crop(p[0], p[1], size, margin), pairs, chunksize=16))


class Camera:
    """One camera: frames into a ring buffer (push, the reader's thread); the
    1 fps path (step_seconds) and the bursts (step_burst) each in a thread of
    their own when live, so a burst never stops the seconds."""

    def __init__(self, cam, index, n_cams, cfg, models, gallery, store, start_ts):
        s = cfg["sampling"]
        self.cam, self.cfg, self.models, self.gallery, self.store = cam, cfg, models, gallery, store
        self.id = cam["id"]
        self.mask = cam.get("mask", [])
        self.fps_out = s["fps"]
        self.burst_s, self.burst_every, self.burst_det_fps = s["burst_seconds"], s["burst_every_s"], s["burst_detect_fps"]
        self.width = s["frame_width"]
        self.buffer = collections.deque()
        self.buffer_s = self.burst_s + 3
        self.buf_lock, self.trk_lock = threading.Lock(), threading.Lock()
        tk = cfg["tracker"]
        self.tracker = Tracker(self.id, tk["iou"], tk["max_age_s"], tk.get("still_s", 0.0), tk.get("still_iou", 0.5),
                               tk.get("still_contain", 0.8), tk.get("still_for_s", 30.0), tk.get("still_moved", 0.3),
                               tk.get("still_long_s", 0.0), tk.get("kalman", False), tk.get("low_iou", 0.5))
        self.next_second = start_ts
        # track_fps above fps: the detector and the tracker also on frames between the seconds (a cow that
        # walks moves more than her box in a second at 1 fps, and her track breaks); only the seconds
        # are written, posture and identity once a second as before
        tf = s.get("track_fps") or self.fps_out
        self.track_dt = 1.0 / tf if tf > self.fps_out else None
        self.next_track = start_ts + (self.track_dt or 0.0)
        # Cameras burst in turn, not all at once: one GPU, an even load.
        self.next_burst = start_ts + self.burst_every * index / max(1, n_cams)
        self.stats = collections.Counter()
        self.timings = None          # a list to record every tick and burst into (stress.py)
        self.cache = None            # barn_dataset.TrainingCache, when [training_cache] enabled
        self.glob = None             # mct.GlobalTracker, when [mct] enabled: one track per cow across cameras

    def push(self, ts, frame_rgb):
        h, w = frame_rgb.shape[:2]
        if w > self.width:
            import cv2
            frame_rgb = cv2.resize(frame_rgb, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA)
        with self.buf_lock:
            self.buffer.append((ts, frame_rgb))
            while self.buffer and ts - self.buffer[0][0] > self.buffer_s:
                self.buffer.popleft()

    def blank(self, img):
        """The far zone greyed out before the detector (camera `exclude`)."""
        return blank_excluded(img, self.cam.get("exclude")) if self.cam.get("blank_excluded", True) else img

    def zone(self, boxes, img):
        """Boxes in the camera's working zone; the rest (far, too small, another
        camera's floor) are dropped here and never tracked or identified."""
        h, w = img.shape[:2]
        keep = [b for b in boxes if in_zone(b, self.cam, w, h, tracking=self.glob is not None)]
        self.stats["boxes_outside_zone"] += len(boxes) - len(keep)
        return keep

    def frames(self):
        with self.buf_lock:
            return list(self.buffer)

    def _time(self, kind, ts, **kw):
        if self.timings is not None:
            self.timings.append({"cam": self.id, "kind": kind, "ts": ts, **kw})

    def step(self, now):
        """Files: one thread does both, on the recording's clock."""
        self.step_seconds(now)
        self.step_burst(now)

    def step_seconds(self, now):
        while True:
            if self.track_dt and self.next_track < self.next_second - 1e-6:    # a frame between the seconds
                if now < self.next_track:
                    return
                t = self.next_track
                self.next_track += self.track_dt
                frame = min(self.frames(), key=lambda x: abs(x[0] - t), default=None)
                if frame and abs(frame[0] - t) <= self.track_dt / 2:
                    self.track_only(t, frame[1])
                continue
            if now < self.next_second:
                return
            t = self.next_second
            self.next_second += 1.0 / self.fps_out
            if self.track_dt and abs(self.next_track - t) < 1e-6:             # the second does this step
                self.next_track += self.track_dt
            frame = min(self.frames(), key=lambda x: abs(x[0] - t), default=None)
            if frame and abs(frame[0] - t) <= 0.5:
                self.second(t, frame[1])
            else:                                         # fell behind past the buffer, or no stream
                self.stats["seconds_missed"] += 1
                self._time("second", t, missed=True)

    def step_burst(self, now):
        if now < self.next_burst + self.burst_s:
            return
        t0 = self.next_burst
        self.next_burst += self.burst_every
        frames = [f for f in self.frames() if t0 <= f[0] < t0 + self.burst_s]
        if len(frames) >= 10 and frames[-1][0] - frames[0][0] >= 0.8 * self.burst_s:
            self.burst(t0, frames)
        else:
            self.stats["bursts_missed"] += 1
            self._time("burst", t0, missed=True, frames=len(frames))

    def _detect(self, img, kind):
        """(sure boxes, low boxes) in the working zone; low ones only with [tracker] low_score, and not for the
        one-canvas floor tracker ([mct] mode fused), which has no second round."""
        low_on = (self.cfg["tracker"].get("low_score") or 0) > 0 and hasattr(self.models, "detect_split") and \
            not (self.glob is not None and self.cfg["mct"].get("mode") == "fused")
        if not low_on:
            return self.zone(self.models.detect(self.blank(img), urgent=True, kind=kind), img), []
        sure, low = self.models.detect_split(self.blank(img), urgent=True, kind=kind)
        return self.zone(sure, img), self.zone(low, img)

    def track_only(self, ts, img):
        """A frame between the seconds: boxes for the tracker, nothing written."""
        start = time.time()
        boxes, low = self._detect(img, "track_detect")
        if self.glob is not None:
            if hasattr(self.glob, "track_only"):
                self.glob.track_only(self.id, ts, [{"box": b, "emb": None} for b in boxes] +
                                     [{"box": b, "emb": None, "low": True} for b in low])
        else:
            with self.trk_lock:
                self.tracker.update(ts, boxes, low)
        self.stats["track_frames"] += 1
        self._time("track", ts, start=start, end=time.time(), cows=len(boxes))

    # one frame a second
    def second(self, ts, img):
        start = time.time()
        boxes, low = self._detect(img, "second_detect")
        if self.glob is not None:
            return self._second_global(ts, img, boxes, start, low)
        with self.trk_lock:
            tracked = self.tracker.update(ts, boxes, low)
        self.stats["low_boxes_kept"] += len(tracked) - len(boxes)
        if len(tracked) > 1 and hasattr(self.gallery, "together"):
            self.gallery.together([tid for tid, _ in tracked], ts, [b for _, b in tracked])   # two cows, with evidence
        if tracked and hasattr(self.gallery, "counted"):
            self.gallery.counted(self.id, ts, [tid for tid, _ in tracked])                # in its zone: counted here
        if tracked:
            size = self.models.crop_size
            feats = self.models.encode(crops_of([(img, b) for _, b in tracked], size, self.models.margin),
                                       urgent=True, kind="second_encode")
            pp, pa = self.models.frame_heads(feats, [b for _, b in tracked])
            rows = [(self.id, tid, ts, *b, POSTURES[int(p.argmax())], ACTIVITIES[int(a.argmax())],
                     float(p[POSTURES.index("lying")])) for (tid, b), p, a in zip(tracked, pp, pa)]
            self.store.seconds(rows)
        self.stats["seconds"] += 1
        self.stats["cow_seconds"] += len(tracked)
        self._time("second", ts, start=start, end=time.time(), cows=len(tracked))

    def _second_global(self, ts, img, boxes, start, low=()):
        """The second with mct.py: every box in the tracking zone is encoded first, its frame id
        carries the cow from camera to camera; only the boxes this camera owns (its mask) are
        counted, so a cow in an overlap is one row, under one track, whichever camera sees her.
        Low boxes ([tracker] low_score) count only where they continued a track."""
        rows, owned_n = [], 0
        n_sure, boxes = len(boxes), list(boxes) + list(low)
        if boxes:
            feats = self.models.encode(crops_of([(img, b) for b in boxes], self.models.crop_size, self.models.margin),
                                       urgent=True, kind="second_encode")
            pp, pa, ids = self.models.frame_heads(feats, boxes, with_id=True)
            lying = POSTURES.index("lying")
            gids = self.glob.update(self.id, ts, [{"box": b, "emb": e, "p_lying": float(p[lying]), "low": k >= n_sure}
                                                  for k, (b, e, p) in enumerate(zip(boxes, ids, pp))])
            if len(boxes) > n_sure:
                keep = [k for k, g in enumerate(gids) if g is not None]
                self.stats["low_boxes_kept"] += sum(k >= n_sure for k in keep)
                gids, boxes = [gids[k] for k in keep], [boxes[k] for k in keep]
                pp, pa = pp[keep], pa[keep]
            if len(gids) > 1 and hasattr(self.gallery, "together"):
                self.gallery.together(gids, ts, boxes)              # two cows, with evidence (mct.CoSeen)
            for gid, b, p, a in zip(gids, boxes, pp, pa):
                if in_mask(b, self.mask):
                    rows.append((self.id, gid, ts, *b, POSTURES[int(p.argmax())], ACTIVITIES[int(a.argmax())],
                                 float(p[POSTURES.index("lying")])))
            owned_n = len(rows)
            if rows:
                self.store.seconds(rows)
                if hasattr(self.gallery, "counted"):
                    self.gallery.counted(self.id, ts, [r[1] for r in rows])
        self.stats["seconds"] += 1
        self.stats["cow_seconds"] += owned_n
        self.stats["cow_seconds_tracked_not_owned"] += len(boxes) - owned_n
        self._time("second", ts, start=start, end=time.time(), cows=owned_n)

    def _track_of(self, ts, box):
        if self.glob is not None:
            return self.glob.track_at(self.id, ts, box)
        with self.trk_lock:
            return self.tracker.track_at(ts, box)

    # 7 s at 25 fps
    def burst(self, t0, frames):
        start = time.time()
        part = collections.Counter()
        step = max(1, int(round(len(frames) / (self.burst_s * self.burst_det_fps))))
        sampled = frames[::step]
        t = time.perf_counter()
        detections = self.models.detect_many([self.blank(img) for _, img in sampled], kind="burst_detect")
        part["detect"] += time.perf_counter() - t
        chains = link_chains([(ts, self.zone(found, img)) for (ts, img), found in zip(sampled, detections)],
                             self.cfg["tracker"]["iou"])
        size = self.models.crop_size
        cows = crops_n = rum_yes = 0
        rum_p = []
        done = []                                   # the burst's cows, decided together below
        for ch in chains:
            if len(ch) < max(2, 0.5 * len(sampled)):      # seen in too little of the burst
                continue
            times = sorted(ch)
            mid = times[len(times) // 2]
            if self.glob is not None and not in_mask(ch[mid], self.mask):
                continue                                  # tracked here, counted by the camera that owns it
            tid = self._track_of(mid, ch[mid]) or f"{self.id}-burst-{int(t0)}-{times[0]:.0f}"
            valid = np.array([times[0] - 0.5 <= ts <= times[-1] + 0.5 for ts, _ in frames])
            boxes = [interpolate_box(ch, ts) for ts, _ in frames]
            t = time.perf_counter()
            crops = crops_of([(img, b) for (_, img), b in zip(frames, boxes)], size, self.models.margin)
            part["crop"] += time.perf_counter() - t
            rel = [ts - t0 for ts, _ in frames]
            # the rhythm on the CPU while the GPU encodes the same crops
            fut = (cpu_pool().submit(motion_features, np.stack(crops), rel, valid)
                   if self.models.motion_dim else None)
            t = time.perf_counter()
            feats = self.models.encode(crops, kind="burst_encode")
            part["encode"] += time.perf_counter() - t
            t = time.perf_counter()
            motion = None
            if fut is not None:
                m, ok = fut.result()
                motion = m if ok else None
            part["motion_wait"] += time.perf_counter() - t
            t = time.perf_counter()
            out = self.models.burst(feats, rel, valid, boxes, motion)
            part["temporal"] += time.perf_counter() - t
            area = float(np.mean([(b[2] - b[0]) * (b[3] - b[1]) for b in ch.values()]))
            # kept until the burst's decision: only what the training cache needs, not every crop
            keep_feats = feats.float().cpu().numpy() if self.cache is not None else None
            done.append((tid, out, area, keep_feats, rel, valid, motion, boxes, crops[len(crops) // 2]))
            cows += 1
            crops_n += len(crops)
            rum_p.append(round(out["rumination"], 3))
            rum_yes += out["ruminating"]
        # the burst's cows are decided together: they were all there at once, so no cow twice
        t = time.perf_counter()
        if len(done) > 1 and hasattr(self.gallery, "together"):
            # seven seconds side by side is evidence enough, unless the two chains are one cow's
            self.gallery.together([d[0] for d in done], t0, [d[7][len(d[7]) // 2] for d in done],
                                  weight=self.cfg["identity"].get("cannot_link_min_s", 5))
        decisions = (self.gallery.decide_batch([(d[1], d[2], d[0]) for d in done], t0)
                     if hasattr(self.gallery, "decide_batch") else
                     [self.gallery.decide(d[1], d[2], t0, d[0]) for d in done])
        for (tid, out, area, feats, rel, valid, motion, boxes, mid_crop), decision in zip(done, decisions):
            self.store.burst((self.id, tid, t0, decision["state"], decision["cow"], decision["sim"],
                              decision["margin"], decision["p"], out["rumination"], out["posture"],
                              out["activity"], out["lameness"], out["quality_max"], int(valid.sum()),
                              int(out["ruminating"])))
            if self.cache is not None:
                from model import box_pos
                self.cache.add(self.id, tid, t0, feats, rel, valid, motion,
                               np.mean([box_pos(b) for b in boxes], 0), mid_crop,
                               {"state": decision["state"], "cow": decision["cow"], "sim": decision["sim"],
                                "p": decision["p"], "area": area})
            if decision["state"] == "confirmed" and hasattr(self.gallery, "face"):
                self.gallery.face(decision["cow"], self.id, t0, mid_crop)
            self.stats[f"id_{decision['state']}"] += 1
        part["gallery"] += time.perf_counter() - t
        self.stats["bursts"] += 1
        self._time("burst", t0, start=start, end=time.time(), ready=t0 + self.burst_s, frames=len(frames),
                   detected=len(sampled), chains=len(chains), cows=cows, crops=crops_n,
                   ruminating=rum_yes, rumination_p=rum_p,
                   parts={k: round(v, 4) for k, v in part.items()})


def merge_by_place(gallery, store, cfg, now):
    """One cow under two names (each camera's gallery naming her by the side it sees) made one by where she stood:
    [mct] link_rule "llr" keeps every two cameras' tracks' evidence of being one cow (store.track_evidence); summed
    per pair of names over history_days (mct.cow_merges) - three zones:
      sure     cow_merge_llr, cow_merge_margin ahead of every rival, never two names seen at the same time: the
               gallery merges them, the store aliases them (the reports follow)
      unsure   cow_review_llr, cow_review_margin: merge_reviews, pending - a person looks at both names' faces
               (herd.py merges page) and says same or different (herd.py merges apply)
      the rest nothing
    A person's "same" is aliased at once and merged in the gallery here; "different" is never proposed again.
    -> [(old, into)] merged this night; nothing unless [mct] is on in mode link with that rule."""
    m = cfg["mct"]
    if not (m.get("enabled") and m.get("mode") == "link" and m.get("link_rule") == "llr"
            and m.get("cow_merge_llr", 0) > 0 and hasattr(store, "evidence")):
        return []
    from mct import cow_merges
    from report import track_map
    done = []
    reviewed = store.reviews() if hasattr(store, "reviews") else []
    for r in reviewed:                                       # a person said "same": the gallery follows tonight
        if r["state"] == "same":
            if gallery.merge(r["old"], r["into"]):
                done.append((r["old"], r["into"]))
            store.decide_review(r["old"], r["into"], "applied", now)
    since = now - cfg["identity"]["history_days"] * 86400
    evidence = store.evidence(since)
    if not evidence:
        return done
    tm = track_map(store)                                    # track -> cow, joined tracks and aliases followed
    aliases = store.aliases()
    canon = lambda c: aliases.get(c, c)
    apart = {(tm[a], tm[b]) for a, b in store.cannot_link_pairs(since)
             if a in tm and b in tm and tm[a] != tm[b]}
    apart |= {(canon(r["old"]), canon(r["into"])) for r in reviewed if r["state"] == "different"}
    size = collections.Counter(tm.values())
    unsure = []
    for old, into in cow_merges(evidence, tm, apart, m["cow_merge_llr"], m.get("cow_merge_margin", 100.0), size,
                                reviews=unsure, review_llr=m.get("cow_review_llr", 60.0),
                                review_margin=m.get("cow_review_margin", 30.0)):
        if gallery.merge(old, into):
            store.alias_cow(old, into, now)
            done.append((old, into))
    unsure = [u for u in unsure if u[0] in gallery.cows and u[1] in gallery.cows]
    if unsure and hasattr(store, "put_reviews"):
        store.put_reviews(unsure, now)
    if done or unsure:
        store.event(now, "merged-by-place", json.dumps({"merged": done, "to_review": [u[:2] for u in unsure]}))
    return done


class SharedGallery:
    """The gallery behind a lock, with the nightly job and the weekly change-over.

    It also keeps who-was-where-when: tracks seen at the same moment are two cows (cannot_link,
    from each camera's tracker and from the chains of one burst), and a cow confirmed on one
    track is that track's for claim_s - another track seen at the same time cannot be the same
    cow unless its burst is clearly surer (claim_steal_margin); otherwise that cow is ruled out
    for it and the next one is tried."""

    def __init__(self, gallery, store, cfg, start_ts):
        self.g, self.store, self.cfg = gallery, store, cfg
        self.lock = threading.Lock()
        self.next_nightly = self._next_hour(start_ts, cfg["farm"]["nightly_hour"])
        self.cannot_link = set(store.cannot_link_pairs(start_ts - cfg["identity"]["history_days"] * 86400)) \
            if hasattr(store, "cannot_link_pairs") else set()
        self._pending_links, self._last_flush = [], start_ts
        from mct import CoSeen
        self.co_seen = CoSeen(cfg["identity"].get("cannot_link_min_s", 5), cfg["identity"].get("cannot_link_max_iou", 0.3))
        self.claims = {}                         # cow -> (track, ts, p or sim)
        self._counted, self._counted_n, self._counted_pruned = {}, {}, start_ts   # counted(): second -> cam -> tracks
        self._masked = {c["id"] for c in cfg.get("cameras", []) if c.get("mask")}
        self._faces = {}                         # (cow, cam) -> ts of the last face kept

    def _next_hour(self, ts, hour):
        tz = dt.timezone(dt.timedelta(hours=self.cfg["farm"]["timezone_offset_hours"]))
        t = dt.datetime.fromtimestamp(ts, tz).replace(hour=hour, minute=0, second=0, microsecond=0)
        if t.timestamp() <= ts:
            t += dt.timedelta(days=1)
        return t.timestamp()

    def together(self, tracks, ts, boxes=None, weight=1):
        """Tracks seen at the same moment by one camera. With their boxes: two different cows once
        seen so in cannot_link_min_s moments with boxes apart (mct.CoSeen; weight: moments this
        sighting counts for). Without boxes: every pair at once."""
        if boxes is not None:
            pairs = self.co_seen.add(tracks, boxes, weight)
        else:
            tracks = sorted(set(tracks))
            pairs = [(a, b) for i, a in enumerate(tracks) for b in tracks[i + 1:]]
        with self.lock:
            for k in pairs:
                if k not in self.cannot_link:
                    self.cannot_link.add(k)
                    self._pending_links.append(k)
            if self._pending_links and (len(self._pending_links) >= 500 or ts - self._last_flush >= 60):
                self.store.add_cannot_link(self._pending_links, ts)
                self._pending_links, self._last_flush = [], ts

    def counted(self, cam, ts, tracks):
        """The tracks a camera counts this second (in its mask). The masks share no floor - each patch of it is
        one camera's - so a cow is counted by one camera at a time: two tracks counted by two cameras at the
        same second for counted_apart_s seconds are two cows (cannot_link), the night never makes one of them.
        (Seconds, not one: a cow on the line between two masks may be counted by both a moment.)"""
        need = self.cfg["identity"].get("counted_apart_s", 10)
        if not need or not tracks or cam not in self._masked:   # no mask: it counts the whole picture,
            return                                              # floor it shares with whoever sees it
        sec = int(ts)
        with self.lock:
            at = self._counted.setdefault(sec, {})
            at[cam] = set(tracks)
            new = []
            for other, theirs in at.items():
                if other == cam:
                    continue
                for a in tracks:
                    for b in theirs:
                        if a == b:
                            continue
                        k = (min(a, b), max(a, b))
                        n, _ = self._counted_n.get(k, (0, ts))
                        self._counted_n[k] = (n + 1, ts)
                        if n + 1 >= need and k not in self.cannot_link:
                            new.append(k)
            for k in new:
                self.cannot_link.add(k)
                self._pending_links.append(k)
            for old in [s_ for s_ in self._counted if s_ < sec - 5]:
                del self._counted[old]
            if ts - self._counted_pruned > 600:              # pairs not seen together for an hour: forgotten
                self._counted_n = {k: v for k, v in self._counted_n.items() if ts - v[1] <= 3600}
                self._counted_pruned = ts

    def separate(self, pairs, ts):
        """Pairs of tracks seen at the same moment (cameras with no floor in common): two cows each."""
        with self.lock:
            for a, b in pairs:
                k = (min(a, b), max(a, b))
                if a != b and k not in self.cannot_link:
                    self.cannot_link.add(k)
                    self._pending_links.append(k)

    def apart(self, a, b):
        return a != b and (min(a, b), max(a, b)) in self.cannot_link

    def _maintain(self, ts):
        if self.g.migration_due(ts):
            self.store.event(ts, "change-over", "weekly cow change-over started")
        if ts >= self.next_nightly:
            merged = merge_by_place(self.g, self.store, self.cfg, ts)
            res = self.g.nightly(ts, self.cannot_link)
            if merged:
                res["merged_by_place"] = merged
            self.store.event(ts, "nightly", json.dumps(res))
            self.next_nightly = self._next_hour(ts, self.cfg["farm"]["nightly_hour"])
        hold = self.cfg["identity"].get("claim_s", 120.0)
        self.claims = {c: v for c, v in self.claims.items() if ts - v[1] <= hold}

    def decide_batch(self, items, ts):
        """items [(out, area, track)] seen at the same moment (one burst) -> a decision each."""
        steal = self.cfg["identity"].get("claim_steal_margin", 0.08)
        conf = lambda d: d["p"] if d.get("p") is not None else (d.get("sim") or 0.0)
        with self.lock:
            self._maintain(ts)
            q = [{"fingerprint": o["fingerprint"], "quality_max": o["quality_max"], "quality_mean": o["quality_mean"],
                  "area": a} for o, a, _ in items]
            ds = self.g.match_batch(q)
            for k, (o, a, tr) in enumerate(items):
                d = ds[k]
                if d["state"] != "confirmed":
                    continue
                cow = d["cow"]
                held = self.claims.get(cow)
                if held and self.apart(tr, held[0]) and conf(d) <= held[2] + steal:
                    # that cow is on a track seen at the same time, with a burst at least as sure:
                    # not her - the next cow, with every cow held elsewhere ruled out
                    ex = {c for c, (t2, _, _) in self.claims.items() if self.apart(tr, t2)}
                    ex |= {x["cow"] for j, x in enumerate(ds) if j != k and x.get("cow")}
                    d = ds[k] = self.g.match(o["fingerprint"], o["quality_max"], o["quality_mean"], a, exclude=ex)
                    self.store.event(ts, "claim-kept", json.dumps({"cow": cow, "held_by": held[0], "track": tr,
                                                                   "now": d.get("cow"), "state": d["state"]}))
                if d["state"] == "confirmed":
                    self.claims[d["cow"]] = (tr, ts, conf(d))
            for (o, a, tr), d in zip(items, ds):
                self.g.observe(d, o["fingerprint"], ts, tr)
            return ds

    def decide(self, out, area, ts, track):
        return self.decide_batch([(out, area, track)], ts)[0]

    def face(self, cow, cam, ts, crop_rgb):
        """A confirmed burst's middle crop kept as the cow's face on this camera (gallery folder faces/<cow>/), at
        most one each [identity] face_every_s, the last face_keep: what a person compares when asked whether two
        names are one cow (herd.py merges page)."""
        every, keep = self.cfg["identity"].get("face_every_s", 600.0), self.cfg["identity"].get("face_keep", 6)
        folder = getattr(self.g, "folder", None)
        if not keep or folder is None or crop_rgb is None:
            return
        with self.lock:
            last = self._faces.get((cow, cam))
            if last is not None and ts - last < every:
                return
            self._faces[(cow, cam)] = ts
        from PIL import Image
        d = os.path.join(folder, "faces", cow)
        os.makedirs(d, exist_ok=True)
        Image.fromarray(np.asarray(crop_rgb, np.uint8)).resize((160, 160)).save(
            os.path.join(d, f"{cam}_{int(ts)}.jpg"), quality=85)
        mine = sorted(f for f in os.listdir(d) if f.startswith(f"{cam}_") and f.endswith(".jpg"))
        for f in mine[:-keep]:
            os.remove(os.path.join(d, f))

    def save(self):
        with self.lock:
            if self._pending_links:                      # the last minute's two-cows, before the night reads them
                self.store.add_cannot_link(self._pending_links, self._last_flush)
                self._pending_links = []
            self.g.save()


# -------------------------------------------------------------------- input

def frames_from(url, start_ts=None, offset=0.0):
    """(ts, rgb) from a file (its own clock, from start_ts) or RTSP (wall clock when the frame is read, + offset: the
    camera's time_offset_s - its frames arrive that much late, so a negative one)."""
    import cv2
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {url}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not 1.0 <= fps <= 120.0:                # a container's time base, not a frame rate (some .mkv)
        fps = 25.0
    live = start_ts is None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    i, t, bad = 0, -1.0, 0
    while True:
        ok, f = cap.read()
        if not ok:
            # a damaged stretch of a recording (a recorder's file): skip it, unless this is its end
            bad += 1
            if live or bad > 100 or (n and cap.get(cv2.CAP_PROP_POS_FRAMES) >= n - 1):
                break
            continue
        bad = 0
        if not live:
            # the file's own clock (its frames' time stamps) - right when the rate it states is not
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            t = ms / 1000.0 if ms / 1000.0 > t else t + 1.0 / fps
        yield (time.time() + offset if live else start_ts + t), cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()


class Lockstep:
    """Recordings of several cameras played together: no camera more than ahead_s ahead of the others still
    playing, on the barn's clock. Live cameras keep together by the wall clock; files played as fast as each
    can go drift apart by minutes - and what spans cameras (the [mct] tracker, its pruning by time, a cow on
    two cameras at the same second, the gallery's claims) needs the same moment from all of them. A camera
    whose recording begins later holds no one up before it begins; one that is done, none at all."""

    def __init__(self, starts, ahead_s=1.0):
        self.pos = dict(starts)                 # cam -> where it is (its start until it begins)
        self.done, self.ahead = set(), ahead_s
        self.cv = threading.Condition()

    def wait(self, cam, ts):
        with self.cv:
            self.pos[cam] = ts
            self.cv.notify_all()
            while any(o != cam and o not in self.done and t < ts - self.ahead for o, t in self.pos.items()):
                self.cv.wait(timeout=1.0)

    def finish(self, cam):
        with self.cv:
            self.done.add(cam)
            self.cv.notify_all()


def run_file(camera, url, start_ts, every_s=60.0, lockstep=None):
    """A recording on its own clock, as fast as it can go (with lockstep: as the slowest camera with it);
    where it is, every every_s of recording."""
    t0, said = time.time(), start_ts
    try:
        for ts, img in frames_from(url, start_ts):
            if lockstep is not None:
                lockstep.wait(camera.id, ts)
            camera.push(ts, img)
            camera.step(ts)
            if ts - said >= every_s:
                said = ts
                done = ts - start_ts
                print(f"[{camera.id}] {int(done // 60)}:{int(done % 60):02d} of the recording "
                      f"({done / max(1e-6, time.time() - t0):.1f}x real time)", flush=True)
    finally:
        if lockstep is not None:
            lockstep.finish(camera.id)
    print(f"[{camera.id}] the recording is done", flush=True)


def run_live(camera, source, stop):
    """Three threads per camera: the reader fills the buffer at the camera's
    rate, the 1 fps path and the bursts run on the wall clock beside it, so a
    slow burst neither drops frames nor holds up the seconds. source() gives
    (ts, rgb) frames: frames_from(url) for RTSP, or a paced file (stress.py)."""
    def reader():
        while not stop.is_set():
            try:
                for ts, img in source():
                    camera.push(ts, img)
                    if stop.is_set():
                        return
            except RuntimeError as e:
                print(f"[{camera.id}] {e}; reconnecting in 5 s", flush=True)
            stop.wait(5)

    def loop(step):
        while not stop.is_set():
            try:
                step(time.time())
            except Exception as e:                   # one bad frame must not stop the camera
                import traceback
                traceback.print_exc()
                print(f"[{camera.id}] {e.__class__.__name__}: {e}", flush=True)
            stop.wait(0.05)

    threads = [threading.Thread(target=f, daemon=True, name=f"{camera.id}-{n}")
               for n, f in (("reader", reader), ("seconds", lambda: loop(camera.step_seconds)),
                            ("bursts", lambda: loop(camera.step_burst)))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()


def run(cfg, start=None, models=None, max_wall_s=None, status=None):
    """status (a dict): "interrupted" is set when Ctrl+C stopped it before the files ended."""
    import gallery as gallery_mod
    from store import Store
    store = Store(cfg["store"]["path"])
    ab_path = os.path.join(os.path.dirname(os.path.abspath(cfg["model"]["checkpoint"])), "abstain.json")
    abstain = json.load(open(ab_path, encoding="utf-8")) if os.path.exists(ab_path) else None
    folder = os.path.join(os.path.dirname(os.path.abspath(cfg["store"]["path"])), "gallery")
    start_ts = start if start is not None else time.time()
    models = models or Models(cfg)
    if getattr(models, "id_model", None) is not None:
        abstain = models.id_abstain          # the NaN model fitted on the identity model's fingerprints
        from reid import apply_thresholds
        # each model's similarities sit elsewhere: its calibrated thresholds, where barn.toml left the default
        if not apply_thresholds(cfg, getattr(models, "id_thresholds", {}) or {}, DEFAULTS,
                                lambda x: print(f"[run] {x}", flush=True)) \
                and cfg["mct"]["reacquire_sim"] == DEFAULTS["mct"]["reacquire_sim"]:
            # an identity model without a calibration (an older reid.py): MmCows' best for the first one
            cfg["mct"]["reacquire_sim"] = 0.9
            print("[run] [mct] reacquire_sim 0.9 with the identity model (no thresholds.json)", flush=True)
    shared = SharedGallery(gallery_mod.Gallery(folder, cfg, abstain), store, cfg, start_ts)
    cams = [Camera(c, i, len(cfg["cameras"]), cfg, models, shared, store, start_ts)
            for i, c in enumerate(cfg["cameras"])]
    if cfg["mct"]["enabled"]:
        from mct import FloorTracker, FusedSeconds, GlobalTracker, TrackletLinker
        mode = cfg["mct"].get("mode")
        if mode != "link" and cfg["mct"].get("link_rule", "near") != "near":
            print(f"[run] [mct] !! link_rule = \"{cfg['mct']['link_rule']}\" but mode = \"{mode}\": the link_* settings "
                  "and the night's merge by place work only with mode = \"link\"", flush=True)
        by_id = {c["id"]: c for c in cfg["cameras"]}
        for a, b in cfg["mct"].get("neighbours", []):
            if a in by_id and b in by_id and not any(by_id[x].get("tracking_mask") for x in (a, b)):
                print(f"[run] [mct] {a}-{b}: neither tracks beyond its mask (tracking_mask) - the masks share no "
                      "floor, so no cow is seen by both at once and their tracks are never joined; give one of "
                      "them a tracking_mask reaching into the other's floor (herd.py zones page)", flush=True)
        if mode in ("fused", "link"):               # all cameras at once, a second at a time
            t = (FloorTracker if mode == "fused" else TrackletLinker)(cfg, store, shared.separate)
            glob = FusedSeconds(t, [c["id"] for c in cfg["cameras"]])
        else:
            glob = GlobalTracker(cfg, store, shared.separate)
        for cam in cams:
            cam.glob = glob
    if cfg["training_cache"]["enabled"]:
        from barn_dataset import TrainingCache
        meta = dict(models.features_meta, dim=models.features_meta.get("dim") or models.encoder.out_dim)
        cache = TrainingCache(cfg, meta)
        for cam in cams:
            cam.cache = cache
        print(f"[run] keeping every {cache.every_n}th burst for training in {cache.folder}", flush=True)
    stop = threading.Event()
    threads = []
    # files: each from its own start (start_ts, when its recording began later than the others'), together;
    # a camera's time_offset_s moves its frames onto the barn's one clock (its stream's delay, its clock's error)
    off = lambda c: float(c.get("time_offset_s") or 0.0)
    lock = Lockstep({c["id"]: c.get("start_ts", start_ts) + off(c) for c in cfg["cameras"]}) \
        if start is not None else None
    for cam, c in zip(cams, cfg["cameras"]):
        target = (lambda cam=cam, c=c: run_file(cam, c["url"], c.get("start_ts", start_ts) + off(c), lockstep=lock)) \
            if start is not None else \
                 (lambda cam=cam, c=c: run_live(cam, lambda: frames_from(c["url"], offset=off(c)), stop))
        th = threading.Thread(target=target, daemon=True)
        th.start()
        threads.append(th)
    t0 = time.time()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(1)
            if max_wall_s and time.time() - t0 > max_wall_s:
                stop.set()
                break
    except KeyboardInterrupt:
        stop.set()
        if status is not None:
            status["interrupted"] = True
    linker = getattr(getattr(cams[0], "glob", None), "ft", None) if cams else None
    if hasattr(linker, "flush_evidence") and getattr(linker, "last_ts", None) is not None:
        linker.flush_evidence(linker.last_ts)             # the last minutes' evidence, before the night reads it
        if linker.clocks():                               # kept for later: herd.py trial, a person reading events
            store.event(linker.last_ts, "clocks", json.dumps({f"{a}-{b}": v for (a, b), v in linker.clocks().items()}))
    shared.save()
    st = dict(getattr(cams[0].glob, "stats", None) or {}) if cams and getattr(cams[0], "glob", None) else {}
    if cfg["mct"].get("enabled") and cfg["mct"].get("mode") == "link":
        near = list(getattr(getattr(cams[0].glob, "ft", None), "nearest", []) or []) if cams else []
        if near:
            q = np.percentile(near, [10, 50, 90])
            print(f"[run] [mct] two cameras' cows each other's nearest on the floor {len(near)} times: "
                  f"{q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f} apart (10th / 50th / 90th percentile), "
                  f"{np.mean(np.asarray(near) <= cfg['mct'].get('link_dist', 120)):.0%} within link_dist "
                  f"{cfg['mct'].get('link_dist', 120):g}", flush=True)
        elif cams:
            print("[run] [mct] two cameras' cows never seen at one second on one floor: no camera pair to join "
                  "(floor_points on both, [mct] disjoint, tracking_mask)", flush=True)
        by_id = {c["id"]: c for c in cfg["cameras"]}
        for (ca, cb), (lag, n, sharp) in sorted((linker.clocks() if hasattr(linker, "clocks") else {}).items()):
            now = float(by_id.get(cb, {}).get("time_offset_s") or 0.0)
            say = (f"time_offset_s = {now - lag:+.1f} on {cb} in barn.toml (now {now:+.1f})" if abs(lag) >= 0.5
                   else "in step")
            print(f"[run] [mct] clocks: {cb}'s frames come {lag:+.1f} s late against {ca}'s ({n} times a cow walked "
                  f"on both; there her paths lie {sharp:.0f} cm nearer than a second either side): {say}", flush=True)
        print(f"[run] [mct] tracks joined across cameras {st.get('linked', 0)}, taken apart again "
              f"{st.get('unlinked', 0)} (a joined cow counted by both cameras at once for unlink_s: masks that "
              f"overlap?), joins refused {st.get('link_refused', 0)}; seconds two cows were each other's nearest "
              f"but another cow within link_margin {st.get('near_but_next_cow_close', 0)}", flush=True)
    return {c.id: dict(c.stats) for c in cams}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--start", default=None, help="files: wall-clock time the recordings start (ISO)")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    start = dt.datetime.fromisoformat(args.start).timestamp() if args.start else None
    stats = run(cfg, start)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    sys.exit(main())
