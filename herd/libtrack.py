"""A camera's tracker from a library instead of the system's own: ByteTrack, BoT-SORT, OC-SORT or SORT from Roboflow's
`trackers` (Apache-2.0; numpy, scipy, OpenCV - no torch). A test: does a ready tracker follow the barn's cows better
than pipeline.Tracker / mct.GlobalTracker (IoU, Kalman, unsure boxes, a cow lying still waited for at her place)?

    [tracker]
    kind = "bytetrack"      # "own" (the default) | "bytetrack" | "botsort" | "ocsort" | "sort"
    lib_lost_s = 30         # a lost track kept this long (s) - the library's lost_track_buffer
    lib_min_frames = 1      # updates before a track gets its id (the library's minimum_consecutive_frames)
    lib_min_iou = 0         # the least overlap to stay one track (0: the library's own, 0.1)
    low_score = 0.1         # boxes under [detector] threshold handed in too: ByteTrack / BoT-SORT's second round

Where it goes: pipeline.Camera's tracker (no [mct]) and TrackletLinker's camera trackers ([mct] mode = "link");
not [mct] mode "handover" / "fused", which track with GlobalTracker / FloorTracker. What the libraries do not have:
the cow lying still for minutes waited for at her place (only lib_lost_s, the Kalman guess drifting), the frame id
(BoT-SORT here has no ReID model; camera motion compensation is off - the cameras do not move), the late
reacquire. Boxes go in as the detector's confidence is unknown here: a sure box 0.9, a low one ([tracker]
low_score) 0.3.

pip install "supervision>=0.26.1" "pydeprecate<0.14" rich && pip install --no-deps trackers==2.6.1

mct_sim.py, 30 min, 3 seeds, joining llr - IDF1 over both cameras / within a camera / track purity:
    own tracker            0.683 / 0.772 / 91.8%     own + kalman   0.752 / 0.849 / 92.7%
    bytetrack              0.553 / 0.623 / 67.2%     (lib_lost_s 300: 0.549; lib_min_iou 0.3: 0.470)
    botsort                0.484 / 0.547 / 67.1%
    ocsort                 0.486 / 0.551 / 67.6%
  lying cows fading for the detector (--weak 0.003):
    own + kalman + low     0.672 / 0.756 / 85.4%
    bytetrack              0.499 / 0.578 / 62.9%     botsort 0.311 / 0.356     ocsort 0.298 / 0.343
The libraries' tracks mix cows a third of the time (purity ~67%) whatever their buffer or overlap threshold.
"""

from __future__ import annotations

import collections
import inspect

import numpy as np

from common import iou

KINDS = {"bytetrack": "ByteTrackTracker", "botsort": "BoTSORTTracker", "ocsort": "OCSORTTracker", "sort": "SORTTracker"}
SURE, UNSURE = 0.9, 0.3


def make(kind, tcfg, rate=1.0):
    """The library's tracker of that kind, on a timestamp clock at the rate it is updated (frames a second: the
    seconds and [sampling] track_fps). Not 30: a second would then be 30 frames missed, every track lost and
    started anew (4 minutes of 30 cows: 193 tracks at 30, 39 at 1)."""
    try:
        import trackers
    except ImportError as e:
        raise SystemExit(f"[tracker] kind = \"{kind}\" needs Roboflow's trackers: pip install \"supervision>=0.26.1\" "
                         f"\"pydeprecate<0.14\" rich && pip install --no-deps trackers==2.6.1 ({e})")
    if kind not in KINDS:
        raise SystemExit(f"[tracker] kind = \"{kind}\": one of own, {', '.join(KINDS)}")
    cls = getattr(trackers, KINDS[kind])
    want = {"lost_track_buffer": int(round(tcfg.get("lib_lost_s", 30.0) * 30)),   # in 30 fps frames
            "frame_rate": float(rate),                                            # updates a second
            "minimum_consecutive_frames": int(tcfg.get("lib_min_frames", 1)),
            "track_activation_threshold": 0.5, "high_conf_det_threshold": 0.5,
            "enable_cmc": False}                                                  # fixed cameras
    if tcfg.get("lib_min_iou"):                       # the boxes' least overlap to be one track (library default 0.1)
        want.update(minimum_iou_threshold=tcfg["lib_min_iou"], minimum_iou_threshold_first_assoc=tcfg["lib_min_iou"])
    params = inspect.signature(cls.__init__).parameters
    return cls(**{k: v for k, v in want.items() if k in params})


class LibTracker:
    """One camera: pipeline.Tracker's interface (update(ts, boxes, low) -> [(track, box)], track_at)."""

    def __init__(self, cam, cfg, kind=None):
        t = cfg["tracker"]
        self.cam, self.kind = cam, kind or t.get("kind")
        self.t = make(self.kind, t, max(cfg["sampling"].get("track_fps", 1.0), cfg["sampling"].get("fps", 1.0)))
        self.min_iou = t.get("iou", 0.3)
        self.history = collections.defaultdict(lambda: collections.deque(maxlen=120))
        self.last = None
        self.stats = collections.Counter()

    def ids(self, ts, boxes, scores):
        """A track id per box, or None (a low box no track took, a track not yet confirmed)."""
        import supervision as sv
        n = len(boxes)
        if self.last is not None and ts <= self.last:          # the library wants time to move on
            ts = self.last + 1e-3
        self.last = ts
        det = sv.Detections(xyxy=np.asarray(boxes, float).reshape(-1, 4), confidence=np.asarray(scores, float),
                            data={"i": np.arange(n)})
        res = self.t.update(det, timestamp=float(ts))
        out = [None] * n
        if len(res) and res.tracker_id is not None:
            for i, tid in zip(res.data["i"], res.tracker_id):
                if tid >= 0:
                    out[int(i)] = f"{self.cam}-{self.kind}-{int(tid)}"
        for tid, b in zip(out, boxes):
            if tid is not None:
                self.history[tid].append((ts, list(b)))
        self.stats["boxes"] += n
        self.stats["dropped"] += out.count(None)
        return out

    def update(self, ts, boxes, low=()):
        allb = list(boxes) + list(low)
        got = self.ids(ts, allb, [SURE] * len(boxes) + [UNSURE] * len(low))
        return [(tid, b) for tid, b in zip(got, allb) if tid is not None]

    def track_at(self, ts, box):
        """The track whose box near time ts overlaps box most (for burst chains), as pipeline.Tracker."""
        best, score = None, 0.0
        for tid, h in self.history.items():
            near = min(h, key=lambda x: abs(x[0] - ts), default=None)
            if near and abs(near[0] - ts) <= 2.0:
                v = iou(near[1], box)
                if v > score:
                    best, score = tid, v
        return best if score >= self.min_iou else None


class LibCamTrackers:
    """Every camera: mct.GlobalTracker's interface as TrackletLinker uses it (update(cam, ts, dets) -> [gid or None],
    track_at(cam, ts, box), late_links, stats)."""

    def __init__(self, cfg, kind=None):
        self.cfg, self.kind = cfg, kind or cfg["tracker"].get("kind")
        self.per = {}
        self.late_links = []
        self.alias = {}

    def _cam(self, cam):
        if cam not in self.per:
            self.per[cam] = LibTracker(cam, self.cfg, self.kind)
        return self.per[cam]

    def update(self, cam, ts, dets):
        return self._cam(cam).ids(ts, [d["box"] for d in dets], [UNSURE if d.get("low") else SURE for d in dets])

    def track_only(self, cam, ts, dets):
        self.update(cam, ts, dets)

    def track_at(self, cam, ts, box):
        return self.per[cam].track_at(ts, box) if cam in self.per else None

    def root(self, g):
        return g

    @property
    def stats(self):
        out = collections.Counter()
        for t in self.per.values():
            out.update(t.stats)
        return out
