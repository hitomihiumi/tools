"""Who-was-where-when in identification: cows seen at the same moment are different cows."""
import _paths  # noqa: F401
import collections
import copy
import itertools
import json
import os
import tempfile

import numpy as np

from common import DEFAULTS


def _cfg(**identity):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["identity"].update(identity)
    return cfg


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def _gallery(d, cows, **identity):
    import gallery as g
    gal = g.Gallery(d, _cfg(**identity))
    for cid, v in cows.items():
        gal.cows[cid] = {"prototypes": _unit(v)[None], "created": 0.0, "last_seen": 0.0}
    return gal


def _near(v, rng, noise=0.05):
    return _unit(np.asarray(v, np.float32) + rng.normal(0, noise, len(v)))


def test_match_batch_gives_no_cow_twice():
    rng = np.random.default_rng(0)
    a, b = np.eye(16)[0], np.eye(16)[0] * 0.8 + np.eye(16)[1] * 0.6
    with tempfile.TemporaryDirectory() as d:
        gal = _gallery(d, {"A": a, "B": b})
        items = [{"fingerprint": _near(a, rng, 0.01)}, {"fingerprint": _near(a, rng, 0.2)}]
        ds = gal.match_batch(items)
        cows = [x["cow"] for x in ds if x["cow"]]
        assert len(cows) == len(set(cows))                         # never one cow for two bursts at once
        assert ds[0]["cow"] == "A"
        assert gal.match(items[1]["fingerprint"], exclude={"A"})["cow"] in (None, "B")


def test_claims_hold_a_cow_against_a_track_seen_at_the_same_time():
    import pipeline
    from store import Store
    a, b = np.eye(16)[0], np.eye(16)[2]
    with tempfile.TemporaryDirectory() as d:
        store = Store(os.path.join(d, "s.sqlite"))
        cfg = _cfg()
        shared = pipeline.SharedGallery(_gallery(d, {"A": a, "B": b}), store, cfg, 1000.0)
        out = lambda v: {"fingerprint": v, "quality_max": 0.5, "quality_mean": 0.5}
        d1 = shared.decide(out(_unit(a)), 0.05, 1000.0, "cam1-1")
        assert d1["state"] == "confirmed" and d1["cow"] == "A"
        shared.together(["cam1-1", "cam1-2"], 1001.0)               # both tracks in one frame: two cows
        # a less sure burst of the other track: A is held, so not A
        d2 = shared.decide(out(_unit(a * 0.9 + b * 0.25)), 0.05, 1010.0, "cam1-2")
        assert d2["cow"] != "A"
        # a track never seen with cam1-1 may still be A (it may be the same cow, later, elsewhere)
        d3 = shared.decide(out(_unit(a)), 0.05, 1020.0, "cam2-7")
        assert d3["cow"] == "A"
        assert store.query("SELECT COUNT(*) FROM events WHERE kind='claim-kept'")[0][0] >= 1


def test_enroll_never_merges_cows_seen_together():
    rng = np.random.default_rng(1)
    x = _unit(rng.normal(size=32))
    y = _unit(x + 0.55 * _unit(rng.normal(size=32)))               # a look-alike, cos ~0.85
    with tempfile.TemporaryDirectory() as d:
        gal = _gallery(d, {}, new_cow_min_bursts=4)
        for k in range(6):
            gal.pool.append({"ts": k, "track": "t-x1", "fp": _near(x, rng, 0.02)})
            gal.pool.append({"ts": 100 + k, "track": "t-x2", "fp": _near(x, rng, 0.02)})
            gal.pool.append({"ts": k, "track": "t-y", "fp": _near(y, rng, 0.02)})
        made = gal.enroll(200.0, cannot_link={("t-x1", "t-y")})
        assert len(made) == 2                                      # x's two tracks one cow, y another
        sizes = sorted(len(gal.history[c]) for c in made)
        assert sizes == [6, 12]


def test_enroll_holds_back_when_the_herd_is_full():
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as d:
        gal = _gallery(d, {"A": np.eye(8)[0]}, new_cow_min_bursts=3, max_cows=1)
        for k in range(5):
            gal.pool.append({"ts": k, "track": "t1", "fp": _near(np.eye(8)[3], rng, 0.02)})
        assert gal.enroll(10.0) == [] and gal.held_back == 1 and len(gal.pool) == 5


def test_report_follows_cow_aliases():
    import report
    from store import Store
    with tempfile.TemporaryDirectory() as d:
        store = Store(os.path.join(d, "s.sqlite"))
        for k, cow in enumerate(["new-1", "new-1", "A"]):
            store.burst(("cam1", "trk", 100.0 + k, "confirmed", cow, 0.9, 0.2, 0.9, 0.1, "lying", "none",
                         None, 0.5, 100))
        assert report.track_map(store) == {"trk": "new-1"}
        store.alias_cow("new-1", "A", 200.0)
        assert report.track_map(store) == {"trk": "A"}
        store.unalias_cow("new-1")
        assert report.track_map(store) == {"trk": "new-1"}


# ------------------------------------------------------------------ stage B: one track across cameras

def _mct_cfg(cams, **mct):
    cfg = _cfg()
    cfg["cameras"] = [dict({"id": c, "mask": []}, **extra) for c, extra in cams]
    cfg["mct"].update(mct)
    return cfg


def _box(cx, cy=0.5, s=0.1):
    return [cx - s, cy - s, cx + s, cy + s]


def test_overlap_hands_a_cow_on_and_keeps_two_apart():
    import mct
    rng = np.random.default_rng(3)
    A, B = _unit(rng.normal(size=64)), _unit(rng.normal(size=64))
    gt = mct.GlobalTracker(_mct_cfg([("cam1", {}), ("cam2", {})], neighbours=[["cam1", "cam2"]]))
    g = None
    for k in range(5):                                       # A walks right in cam1
        g1 = gt.update("cam1", float(k), [{"box": _box(0.5 + 0.08 * k), "emb": _near(A, rng)}])[0]
        assert g is None or g1 == g
        g = g1
    # the overlap: cam2 sees her at its left edge while cam1 still has her, and B beside her
    out = gt.update("cam2", 4.0, [{"box": _box(0.12), "emb": _near(A, rng)}, {"box": _box(0.5), "emb": _near(B, rng)}])
    assert out[0] == g and out[1] != g
    for k in range(5, 8):                                    # cam1 has lost her; cam2 follows by IoU
        out = gt.update("cam2", float(k), [{"box": _box(0.12 + 0.05 * (k - 4)), "emb": _near(A, rng)}])
        assert out[0] == g
    assert gt.stats["overlap"] == 1


def test_handover_refuses_a_look_alike_pair():
    import mct
    rng = np.random.default_rng(4)
    A = _unit(rng.normal(size=64))
    A2 = _unit(A + 0.02 * rng.normal(size=64))             # two cows the frame id cannot tell apart
    gt = mct.GlobalTracker(_mct_cfg([("cam1", {}), ("cam2", {})], neighbours=[["cam1", "cam2"]]))
    ga, gb = gt.update("cam1", 0.0, [{"box": _box(0.8, 0.3), "emb": A}, {"box": _box(0.8, 0.7), "emb": A2}])
    out = gt.update("cam2", 1.0, [{"box": _box(0.1, 0.5), "emb": _near(A, rng, 0.01)}])
    assert out[0] not in (ga, gb) and gt.stats["overlap_ambiguous"] == 1


def test_portal_bridges_a_blind_passage_in_its_time_window():
    import mct
    rng = np.random.default_rng(5)
    A = _unit(rng.normal(size=64))
    cfg = _mct_cfg([("cam1", {}), ("cam3", {})],
                   portals=[{"from": "cam1", "to": "cam3", "min_s": 5, "max_s": 30, "exit": [[0.7, 0], [1, 0], [1, 1], [0.7, 1]]}])
    gt = mct.GlobalTracker(cfg)
    g = gt.update("cam1", 0.0, [{"box": _box(0.85), "emb": A}])[0]
    assert gt.update("cam3", 2.0, [{"box": _box(0.2), "emb": _near(A, rng)}])[0] != g      # too soon
    gt2 = mct.GlobalTracker(cfg)
    g = gt2.update("cam1", 0.0, [{"box": _box(0.85), "emb": A}])[0]
    assert gt2.update("cam3", 12.0, [{"box": _box(0.2), "emb": _near(A, rng)}])[0] == g
    gt3 = mct.GlobalTracker(cfg)
    g = gt3.update("cam1", 0.0, [{"box": _box(0.85), "emb": A}])[0]
    assert gt3.update("cam3", 45.0, [{"box": _box(0.2), "emb": _near(A, rng)}])[0] != g    # too late


def test_reacquire_after_a_short_hide():
    import mct
    rng = np.random.default_rng(6)
    A = _unit(rng.normal(size=64))
    gt = mct.GlobalTracker(_mct_cfg([("cam1", {})]))
    g = gt.update("cam1", 0.0, [{"box": _box(0.5), "emb": A}])[0]
    gt.update("cam1", 1.0, [])
    # back 7 s later (past max_age_s), a little further on: the same cow by her frame id
    assert gt.update("cam1", 8.0, [{"box": _box(0.58), "emb": _near(A, rng, 0.02)}])[0] == g
    assert gt.stats["reacquired"] == 1


def test_disjoint_cameras_add_cannot_link():
    import mct
    rng = np.random.default_rng(7)
    got = []
    gt = mct.GlobalTracker(_mct_cfg([("cam1", {}), ("cam4", {})], disjoint=[["cam1", "cam4"]]),
                           apart=lambda pairs, ts: got.extend(pairs))
    a = gt.update("cam1", 10.0, [{"box": _box(0.5), "emb": _unit(rng.normal(size=8))}])[0]
    b = gt.update("cam4", 10.2, [{"box": _box(0.5), "emb": _unit(rng.normal(size=8))}])[0]
    assert (b, a) in got or (a, b) in got


class _FakeModels:
    """What Camera._second_global asks of the models: crops, encoder and frame heads, ids set by the test."""
    crop_size, margin = 16, 0.1

    def __init__(self):
        self.ids = {}

    def detect(self, img, **kw):
        return self.boxes

    def encode(self, crops, **kw):
        return None

    def frame_heads(self, feats, boxes, with_id=False, **kw):
        n = len(boxes)
        pp, pa = np.tile([0.9, 0.1], (n, 1)), np.tile([0.1, 0.1, 0.8], (n, 1))
        return pp, pa, [self.ids[round((b[0] + b[2]) / 2, 2)] for b in boxes]


def test_overlap_counts_every_second_once():
    """Two cameras over one floor, each owning a half (mask) and tracking all of it (tracking_mask):
    one cow walking across is one track and one row a second, whichever camera owns her."""
    import mct
    import pipeline
    from store import Store
    rng = np.random.default_rng(8)
    A = _unit(rng.normal(size=64))
    left, right = [[0, 0], [0.5, 0], [0.5, 1], [0, 1]], [[0.5, 0], [1, 0], [1, 1], [0.5, 1]]
    whole = [[0, 0], [1, 0], [1, 1], [0, 1]]
    # both cameras see the same floor in the same frame coordinates (the simplest overlap)
    cams = [("cam1", {"mask": left, "tracking_mask": whole}), ("cam2", {"mask": right, "tracking_mask": whole})]
    cfg = _mct_cfg(cams, enabled=True, neighbours=[["cam1", "cam2"]])
    cfg["cameras"] = [dict(c, min_box_pct=0.0, min_box_side_px=0) for c in cfg["cameras"]]
    with tempfile.TemporaryDirectory() as d:
        store = Store(os.path.join(d, "s.sqlite"))
        shared = pipeline.SharedGallery(_gallery(d, {}), store, cfg, 0.0)
        models = _FakeModels()
        glob = mct.GlobalTracker(cfg, store, shared.separate)
        cams_ = [pipeline.Camera(c, i, 2, cfg, models, shared, store, 0.0) for i, c in enumerate(cfg["cameras"])]
        for c in cams_:
            c.glob = glob
        img = np.zeros((64, 64, 3), np.uint8)
        for k in range(10):
            cx = round(0.15 + 0.08 * k, 2)
            models.boxes = [_box(cx)]
            models.ids = {cx: _near(A, rng, 0.03)}
            for c in cams_:
                c.second(float(k), img)
        rows = store.query("SELECT cam, track, ts FROM seconds ORDER BY ts")
        assert [r[2] for r in rows] == [float(k) for k in range(10)]            # one row a second
        assert len({r[1] for r in rows}) == 1                                   # one track
        assert {r[0] for r in rows} == {"cam1", "cam2"}                          # owned by each in turn


def test_frame_id_training_points_frames_at_their_burst():
    import json as js
    import model as mdl
    import train
    rng = np.random.default_rng(9)
    dim, T = 24, 30
    with tempfile.TemporaryDirectory() as d:
        feat, run = os.path.join(d, "features"), os.path.join(d, "run")
        tr = os.path.join(feat, "train")
        os.makedirs(tr), os.makedirs(run)
        js.dump({"dim": dim, "encoder": "x", "grid": 2, "crop": 224, "margin": 0.1}, open(os.path.join(tr, "meta.json"), "w"))
        rows = []
        for clip in range(10):
            cows = rng.normal(size=(4, dim)) * 2
            bf = np.stack([c + rng.normal(size=(T, dim)) * 0.7 for c in cows]).astype(np.float16)
            bt = np.tile(np.arange(T, dtype=np.float32) / 25, (4, 1))
            np.savez(os.path.join(tr, f"{clip}.npz"), burst_feats=bf, burst_times=bt, key_feats=np.zeros((0, dim), np.float16))
            rows += [{"kind": "burst", "clip": str(clip), "i": i, "posture": 0, "activity": 2, "rumination": 0,
                      "area": 0.05, "uids": []} for i in range(4)]
            rows.append({"kind": "clip_done", "clip": str(clip)})
        with open(os.path.join(tr, "index.jsonl"), "w") as fh:
            fh.writelines(js.dumps(r) + "\n" for r in rows)
        import torch
        torch.manual_seed(0)
        m = mdl.HerdModel(dim, d=32, layers=1, heads=2)
        m.save(os.path.join(run, "model.pt"), {"crop_s": 0.5, "epoch": 1, "features": {"dim": dim},
                                                 "dev_clips": ["0", "1"], "rum_threshold": 0.5})
        train.main(["frame-id", "--features", feat, "--out", run, "--epochs", "8", "--P", "8", "--seed", "0"])
        ev = js.load(open(os.path.join(run, "eval_frame_id.json")))
        assert ev["frame_to_burst_top1_clip"] >= 0.8, ev
        m2, ck = mdl.HerdModel.load(os.path.join(run, "model.pt"))
        assert m2.frame_id and ck["dev_clips"] == ["0", "1"] and "frame_id" in ck
        assert os.path.exists(os.path.join(run, "model.pre_frame_id.pt"))
        old, _ = mdl.HerdModel.load(os.path.join(run, "model.pre_frame_id.pt"))
        assert not old.frame_id


def test_mct_eval_scores_one_track_across_cameras():
    """Four cows under two cameras (the second sees the pen mirrored), labelled every 15 s: the per-camera
    IoU tracker cannot join the cameras, mct must; cows hidden for a few seconds come back by frame id."""
    import json as js
    import mct_eval
    rng = np.random.default_rng(10)
    cows = {c: _unit(rng.normal(size=32)) for c in (1, 2, 3, 4)}
    with tempfile.TemporaryDirectory() as d:
        js.dump({"cams": ["cam_1", "cam_2"], "start_ts": 0, "end_ts": 300, "step": 1}, open(os.path.join(d, "meta.json"), "w"))
        os.makedirs(os.path.join(d, "emb"))
        for cam in ("cam_1", "cam_2"):
            os.makedirs(os.path.join(d, "labels", cam))
            T, N, B, F = [], [], [], []
            for ts in range(300):
                rows = []
                for c, v in cows.items():
                    x = 0.15 + 0.2 * (c - 1) + 0.03 * np.sin(ts / 40 + c)
                    y = 0.5 + 0.2 * np.cos(ts / 50 + c)
                    if cam == "cam_2":
                        x = 1 - x
                    box = [x - 0.06, y - 0.08, x + 0.06, y + 0.08]
                    rows.append((c, box))
                    hidden = cam == "cam_1" and c == 2 and 100 <= ts < 107
                    if not hidden:
                        B.append(box), F.append(_near(v, rng, 0.05))
                T.append(ts), N.append(sum(1 for c, _ in rows if not (cam == "cam_1" and c == 2 and 100 <= ts < 107)))
                if ts % 15 == 0:
                    with open(os.path.join(d, "labels", cam, f"{1690304400 + ts}_12-00-00.txt"), "w") as fh:
                        for c, b in rows:
                            fh.write(f"{c} {(b[0] + b[2]) / 2} {(b[1] + b[3]) / 2} {b[2] - b[0]} {b[3] - b[1]}\n")
            np.savez(os.path.join(d, "emb", f"{cam}.npz"), ts=np.array(T) + 1690304400, n=np.array(N),
                     boxes=np.array(B, np.float32), cls=np.array(F, np.float16), fid=np.array(F, np.float16))
        mct_eval.main(["score", "--data", d, "--min-bursts", "3", "--appearance", "--sweep", "0.6",
                       "--enroll", "fid/mct"])
        r = js.load(open(os.path.join(d, "mct_eval.json")))
        assert r["iou"]["cross_cam_same_track"] == 0.0 and r["iou"]["idf1_global"] < 0.7
        m = r["fid/mct"]
        assert m["cross_cam_same_track"] > 0.95 and m["idf1_global"] > 0.95, m
        assert m["false_merges"] == 0 and m["false_cannot_link"] == 0
        assert r["fid/cam"]["tracks_per_cow"] < r["iou"]["tracks_per_cow"]      # the hidden cow comes back
        e = m["enroll"]["with_cannot_link@0.55"]
        assert e["cows_made"] == 4 and e["purity"] == 1.0


def test_frame_fingerprint_without_a_head_is_the_burst_of_one_frame():
    import torch
    import model as mdl
    torch.manual_seed(0)
    m = mdl.HerdModel(24, d=32, layers=1, heads=2).eval()
    x = torch.randn(5, 24)
    with torch.no_grad():
        z = m.frame_fingerprint(x)
        ref = m.temporal(x[2:3].unsqueeze(1), torch.zeros(1, 1), torch.ones(1, 1, dtype=torch.bool))["fingerprint"]
    assert z.shape == (5, mdl.ID_DIM) and torch.allclose(z.norm(dim=-1), torch.ones(5), atol=1e-5)
    assert torch.allclose(z[2], ref[0], atol=1e-5)


def _write_proj(d, cam, P):
    os.makedirs(os.path.join(d, "proj_mat"), exist_ok=True)
    with open(os.path.join(d, "proj_mat", f"proj_mat_cam{cam[-1]}.xml"), "w") as fh:
        fh.write("<opencv_storage><projection_matrix><rows>3</rows><cols>4</cols><dt>d</dt><data>"
                 + " ".join(map(str, np.ravel(P))) + "</data></projection_matrix></opencv_storage>")


def test_geo_hands_cows_over_by_place_alone():
    """Two cameras from opposite sides, frame ids that say nothing (random): passed on by where the cows
    lie on the floor (the projection matrices, body height from p(lying)) - and drawn by show."""
    import json as js
    import mct_eval
    rng = np.random.default_rng(11)
    W, H = mct_eval.PROJ_SIZE
    # world cm -> pixels of a 4480x2800 frame; the second camera sees the pen mirrored
    P = {"cam_1": np.array([[3.0, 0.0, -1.0, 500.0], [0.0, 2.0, -2.0, 800.0], [0, 0, 0, 1.0]]),
         "cam_2": np.array([[-3.0, 0.0, -1.0, 3900.0], [0.0, -2.0, -2.0, 2200.0], [0, 0, 0, 1.0]])}
    cows = {c: (150.0 + 250 * c, 100.0 + 120 * (c % 3)) for c in range(1, 5)}
    with tempfile.TemporaryDirectory() as d:
        js.dump({"cams": ["cam_1", "cam_2"], "start_ts": 1690304400, "end_ts": 1690304400 + 120, "step": 1},
                open(os.path.join(d, "meta.json"), "w"))
        os.makedirs(os.path.join(d, "emb"))
        for cam in P:
            _write_proj(d, cam, P[cam])
            os.makedirs(os.path.join(d, "labels", cam))
            T, N, B, F, L = [], [], [], [], []
            for k in range(120):
                ts = 1690304400 + k
                rows = []
                for c, (x, y) in cows.items():
                    u, v, w = P[cam] @ [x + 0.3 * k, y, 55.0, 1.0]            # lying: centre 55 cm up
                    cx, cy = u / w / W, v / w / H
                    rows.append((c, [cx - 0.03, cy - 0.04, cx + 0.03, cy + 0.04]))
                for c, b in rows:
                    B.append(b), F.append(_unit(rng.normal(size=16))), L.append(1.0)
                T.append(ts), N.append(len(rows))
                if k % 15 == 0:
                    with open(os.path.join(d, "labels", cam, f"{ts}_12-00-00.txt"), "w") as fh:
                        fh.writelines(f"{c} {(b[0] + b[2]) / 2} {(b[1] + b[3]) / 2} {b[2] - b[0]} {b[3] - b[1]}\n"
                                      for c, b in rows)
            np.savez(os.path.join(d, "emb", f"{cam}.npz"), ts=np.array(T), n=np.array(N), boxes=np.array(B, np.float32),
                     cls=np.array(F, np.float16), fid=np.array(F, np.float16), lying=np.array(L, np.float32))
        mct_eval.main(["score", "--data", d, "--min-bursts", "1", "--dist-sweep"])
        r = js.load(open(os.path.join(d, "mct_eval.json")))
        assert r["data"]["floor_points"]["same_cow_median_cm"] < 5
        g = r["fid/geo"]
        assert g["cross_cam_same_track"] == 1.0 and g["false_merges"] == 0 and g["tracks_per_cow"] == 1.0, g
        assert r["fid/still"]["cross_cam_same_track"] == 0.0
        f = r["fid/fused"]                                  # one canvas: each cow one point, one track
        assert f["cross_cam_same_track"] == 1.0 and f["false_merges"] == 0 and f["tracks_per_cow"] == 1.0, f
        assert f["tracker"]["fused"] >= 4 * 100
        k = r["fid/link-final"]                              # each camera's tracks joined by floor evidence
        assert k["cross_cam_same_track"] == 1.0 and k["false_merges"] == 0 and k["tracks_per_cow"] == 1.0, k
        assert r["fid/link"]["tracker"]["linked"] == 4
        mct_eval.main(["show", "--data", d, "--minutes", "0.5", "--still-every", "10"])
        out = os.path.join(d, "show", "fid_geo")
        assert os.path.getsize(os.path.join(out, "ids.mp4")) > 1000 and len(os.listdir(os.path.join(out, "stills"))) == 3
        # embeddings of an identity model with its thresholds: show knows the variants score made with them
        for cam in P:
            z = dict(np.load(os.path.join(d, "emb", f"{cam}.npz")))
            z["stamp"] = js.dumps({"thresholds": {"reacquire_sim": 0.9, "late_sim": 0.95}})
            np.savez(os.path.join(d, "emb", f"{cam}.npz"), **z)
        mct_eval.main(["show", "--data", d, "--minutes", "0.2", "--still-every", "10", "--variant", "fid/link-final+late"])
        assert os.path.exists(os.path.join(d, "show", "fid_link-final+late", "ids.mp4"))


def test_still_cow_kept_and_cannot_link_needs_evidence():
    import mct
    gt = mct.GlobalTracker(_mct_cfg([("cam1", {})]))
    g = gt.update("cam1", 0.0, [{"box": _box(0.5), "emb": None}])[0]
    # unseen for 40 s (another cow lay in front), back at the same place: the same cow, no look needed
    assert gt.update("cam1", 40.0, [{"box": _box(0.505), "emb": None}])[0] == g
    co = mct.CoSeen(min_s=3, max_iou=0.3)
    a, b, dup = _box(0.2), _box(0.7), _box(0.21)
    assert co.add(["t1", "t2", "t3"], [a, b, dup]) == []               # t1 / t3: two boxes on one cow
    co.add(["t1", "t2", "t3"], [a, b, dup])
    assert co.add(["t1", "t2", "t3"], [a, b, dup]) == [("t1", "t2"), ("t2", "t3")]


def test_floor_mapper_from_floor_points():
    from common import floor_mapper
    pts = [[0.1, 0.9, 0, 0], [0.9, 0.9, 800, 0], [0.9, 0.1, 800, 1200], [0.1, 0.1, 0, 1200]]
    f = floor_mapper({"floor_points": pts}, DEFAULTS["mct"])
    x, y = f([0.45, 0.4, 0.55, 0.5])                                    # bottom centre (0.5, 0.5)
    assert abs(x - 400) < 1e-6 and abs(y - 600) < 1e-6
    assert floor_mapper({}, DEFAULTS["mct"]) is None


def test_show_ids_draws_the_store_on_the_recording():
    import datetime as dt
    import cv2
    import show_ids
    from store import Store
    with tempfile.TemporaryDirectory() as d:
        video = os.path.join(d, "cam1.mp4")
        w = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 5, (320, 240))
        for _ in range(5 * 12):
            w.write(np.full((240, 320, 3), 90, np.uint8))
        w.release()
        start = dt.datetime(2026, 10, 7, 6, 0, 0).timestamp()
        cfg_path = os.path.join(d, "barn.toml")
        store_path = os.path.join(d, "herd.sqlite")
        open(cfg_path, "w").write(f'[store]\npath = "{store_path}"\n')
        store = Store(store_path)
        store.seconds([("cam1", "trk-a", start + k, 0.1, 0.2, 0.4, 0.6, "lying", "none", 0.9) for k in range(10)] +
                      [("cam1", "trk-b", start + k, 0.5, 0.2, 0.8, 0.6, "standing", "feeding", 0.1) for k in range(10)])
        store.burst(("cam1", "trk-a", start + 2, "confirmed", "A12", 0.9, 0.2, 0.95, 0.1, "lying", "none", None, 0.5, 100))
        out = os.path.join(d, "show")
        show_ids.main(["--config", cfg_path, "--cam", "cam1", "--video", video, "--start", "2026-10-07T06:00:00",
                       "--minutes", "0.2", "--out", out, "--still-every", "5"])
        assert os.path.getsize(os.path.join(out, "ids.mp4")) > 1000
        assert len(os.listdir(os.path.join(out, "stills"))) == 3                  # seconds 0, 5, 10
        assert "50%" in open(os.path.join(out, "index.html")).read()              # trk-a has a cow, trk-b is NaN


def test_floor_tracker_fuses_cameras_and_leaves_phantoms_aside():
    import mct
    cfg = _cfg()
    cfg["cameras"] = [{"id": c, "mask": [], "floor_points": pts} for c, pts in (
        ("a", [[0, 0, 0, 0], [1, 0, 1000, 0], [1, 1, 1000, 1000], [0, 1, 0, 1000]]),
        ("b", [[0, 0, 1000, 1000], [1, 0, 0, 1000], [1, 1, 0, 0], [0, 1, 1000, 0]]))]   # b sees it turned round
    cfg["mct"].update(floor_area=[[0, 0], [1000, 0], [1000, 1000], [0, 1000]])
    ft = mct.FloorTracker(cfg)
    box = lambda cx, by: [cx - 0.05, by - 0.1, cx + 0.05, by]            # the floor point is the bottom centre
    g = None
    for k in range(10):
        x = 0.3 + 0.01 * k
        out = ft.update_all(float(k), {"a": [{"box": box(x, 0.5)}, {"box": box(0.5, 1.4)}],   # 2nd: off the floor
                                       "b": [{"box": box(1 - x, 0.5)}]})
        assert out["a"][0] == out["b"][0]                                 # one cow, both cameras
        assert g is None or out["a"][0] == g
        g = out["a"][0]
        assert out["a"][1] != g and out["a"][1].startswith("a-off")
    assert ft.stats["fused"] == 10 and ft.stats["off_floor"] == 10


def test_fused_seconds_waits_for_every_camera():
    import threading
    import mct
    cfg = _cfg()
    cfg["cameras"] = [{"id": c, "mask": [], "floor_points": [[0, 0, 0, 0], [1, 0, 1000, 0], [1, 1, 1000, 1000],
                                                             [0, 1, 0, 1000]]} for c in ("a", "b")]
    fs = mct.FusedSeconds(mct.FloorTracker(cfg), ["a", "b"], wait_s=2.0)
    box = [0.45, 0.4, 0.55, 0.5]
    fs.update("a", 0.0, [{"box": box}])
    fs.update("b", 0.0, [{"box": box}])                  # both alive from now on
    got = {}
    th = [threading.Thread(target=lambda c=c: got.__setitem__(c, fs.update(c, 1.0, [{"box": box}]))) for c in "ab"]
    [t.start() for t in th]
    [t.join() for t in th]
    assert got["a"] == got["b"] and fs.stats["fused"] >= 1          # one cow: one point, one track


def test_report_follows_joined_tracks():
    import report
    from store import Store
    with tempfile.TemporaryDirectory() as d:
        store = Store(os.path.join(d, "s.sqlite"))
        for k in range(3):
            store.burst(("cam1", "t-a", 100.0 + k, "confirmed", "A", 0.9, 0.2, 0.9, 0.1, "lying", "none", None, 0.5, 100))
        store.link_tracks(150.0, "t-b", "t-a")              # cam2's track: the same cow
        store.link_tracks(160.0, "t-c", "t-b")
        assert store.track_roots() == {"t-b": "t-a", "t-c": "t-a"}
        assert report.track_map(store) == {"t-a": "A", "t-b": "A", "t-c": "A"}


def test_linker_refuses_a_chain_that_joins_cows_apart():
    """A=B and B=C must not make A=C when A and C stood apart (link_contra)."""
    import mct
    pts = [[0, 0, 0, 0], [1, 0, 1000, 0], [1, 1, 1000, 1000], [0, 1, 0, 1000]]
    cfg = _cfg()
    cfg["cameras"] = [{"id": c, "mask": [], "floor_points": pts} for c in ("a", "b", "c")]
    cfg["mct"].update(link_contra=0.3, link_min_s=3, link_share=0.3, link_dist=120.0)
    lk = mct.TrackletLinker(cfg)
    box = lambda x: [x - 0.03, 0.45, x + 0.03, 0.5]
    for k in range(20):
        # a sees a cow at x=0.2, c a cow at x=0.4 the whole time; b's one track walks from a's cow to c's
        bx = 0.25 + 0.1 * k / 19
        out = lk.update_all(float(k), {"a": [{"box": box(0.2)}], "b": [{"box": box(bx)}], "c": [{"box": box(0.4)}]})
    assert out["a"][0] != out["c"][0]
    assert lk.stats["linked"] >= 1 and lk.stats["link_contradicted"] >= 1


def test_linker_by_vote_joins_the_pair_most_often_near():
    """Two cows side by side, still; camera b's floor mapping is off: half the seconds its cows land 3 m away,
    now and then swapped. Each pair is near under half the time (never link_share's 60%), but clearly more
    often than the swapped one - joined by vote, each to the right one."""
    import mct
    pts = [[0, 0, 0, 0], [1, 0, 1000, 0], [1, 1, 1000, 1000], [0, 1, 0, 1000]]
    box = lambda x: [x - 0.03, 0.45, x + 0.03, 0.5]
    results = {}
    for vote in (0.0, 2.0):
        cfg = _cfg()
        cfg["cameras"] = [{"id": c, "mask": [], "floor_points": pts} for c in ("a", "b")]
        cfg["mct"].update(link_min_s=3, link_dist=120.0, link_margin=0.0, link_vote=vote, link_vote_share=0.15)
        lk = mct.TrackletLinker(cfg)
        true_b, now = lk.floor["b"], {"k": 0}

        def bent(b, pl=0.5):                               # where b says its cow is: off as the second has it
            x, y = true_b(b, pl)
            k = now["k"]
            x += 300.0 if k % 2 == 0 else 0.0
            if k % 10 == 1:                                # swapped: each lands on the other cow's place
                x += 100.0 if b[0] < 0.3 else -100.0
            return x, y
        lk.floor["b"] = bent
        for k in range(60):
            now["k"] = k
            out = lk.update_all(float(k), {"a": [{"box": box(0.3)}, {"box": box(0.4)}],
                                           "b": [{"box": box(0.3)}, {"box": box(0.4)}]})
        results[vote] = out
    assert results[0.0]["a"] != results[0.0]["b"]                  # 60% of the time: never
    assert results[2.0]["a"] == results[2.0]["b"]                  # by vote: each to the right one


def test_linker_vote_rivals_only_on_the_partner_camera():
    """Three cameras over one cow: her tracks on b and c are not each other's rivals for a's track - all three
    joined by vote."""
    import mct
    pts = [[0, 0, 0, 0], [1, 0, 1000, 0], [1, 1, 1000, 1000], [0, 1, 0, 1000]]
    box = lambda x: [x - 0.03, 0.45, x + 0.03, 0.5]
    cfg = _cfg()
    cfg["cameras"] = [{"id": c, "mask": [], "floor_points": pts} for c in ("a", "b", "c")]
    cfg["mct"].update(link_min_s=3, link_dist=120.0, link_margin=0.0, link_vote=2.0)
    lk = mct.TrackletLinker(cfg)
    for k in range(10):
        out = lk.update_all(float(k), {c: [{"box": box(0.3)}] for c in ("a", "b", "c")})
    assert out["a"] == out["b"] == out["c"]


def test_linker_place_and_look_together():
    """Two cows lying 10 cm apart, camera b's floor 80 cm off at random: by place alone each is b's nearest to
    either about as often (no vote won); with the look in the same number (link_look_cm) to the right one."""
    import mct
    pts = [[0, 0, 0, 0], [1, 0, 1000, 0], [1, 1, 1000, 1000], [0, 1, 0, 1000]]
    box = lambda x: [x - 0.03, 0.45, x + 0.03, 0.5]
    e1, e2 = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    out = {}
    for look in (0.0, 300.0):
        rng = np.random.default_rng(0)
        cfg = _cfg()
        cfg["cameras"] = [{"id": c, "mask": [], "floor_points": pts} for c in ("a", "b")]
        cfg["mct"].update(link_min_s=3, link_dist=120.0, link_margin=0.0, link_vote=2.0, link_look_cm=look)
        lk = mct.TrackletLinker(cfg)
        true_b = lk.floor["b"]
        lk.floor["b"] = lambda b, pl=0.5: (lambda p: (p[0] + rng.normal(0, 80.0), p[1]))(true_b(b, pl))
        for k in range(60):
            res = lk.update_all(float(k), {"a": [{"box": box(0.30), "emb": e1}, {"box": box(0.31), "emb": e2}],
                                           "b": [{"box": box(0.30), "emb": e1}, {"box": box(0.31), "emb": e2}]})
        out[look] = res
    assert out[0.0]["a"] != out[0.0]["b"]
    assert out[300.0]["a"] == out[300.0]["b"]


def test_linker_leaves_disjoint_cameras_alone():
    """Two cameras in [mct] disjoint whose floor plans happen to coincide: their cows are never joined."""
    import mct
    pts = [[0, 0, 0, 0], [1, 0, 1000, 0], [1, 1, 1000, 1000], [0, 1, 0, 1000]]
    box = lambda x: [x - 0.03, 0.45, x + 0.03, 0.5]
    for disjoint, joined in (([], True), ([["a", "b"]], False)):
        cfg = _cfg()
        cfg["cameras"] = [{"id": c, "mask": [], "floor_points": pts} for c in ("a", "b")]
        cfg["mct"].update(link_min_s=3, link_share=0.3, link_dist=120.0, disjoint=disjoint)
        lk = mct.TrackletLinker(cfg)
        for k in range(10):
            out = lk.update_all(float(k), {"a": [{"box": box(0.3)}], "b": [{"box": box(0.3)}]})
        assert (out["a"][0] == out["b"][0]) == joined, disjoint


def test_linker_says_cows_on_disjoint_cameras_are_two():
    """Link mode: cows seen at the same second by two cameras that share no floor are two cows - handed to
    the gallery (cannot_link), so the night does not make one cow of them."""
    import mct
    got = []
    cfg = _cfg()
    cfg["cameras"] = [{"id": c, "mask": []} for c in ("a", "b", "c")]
    cfg["mct"].update(disjoint=[["a", "c"]])
    lk = mct.TrackletLinker(cfg, None, lambda pairs, ts: got.extend(pairs))
    box = lambda x: [x - 0.03, 0.45, x + 0.03, 0.5]
    for k in range(3):
        out = lk.update_all(float(k), {"a": [{"box": box(0.2)}, {"box": box(0.6)}], "b": [{"box": box(0.2)}],
                                       "c": [{"box": box(0.4)}]})
    pairs = {frozenset(p) for p in got}
    assert pairs == {frozenset((out["a"][0], out["c"][0])), frozenset((out["a"][1], out["c"][0]))}, pairs


def test_late_reacquire_by_the_tracks_own_fingerprint():
    """Two cows hidden 15 s (past reacquire_s), back near where they were: one frame is too noisy to
    take them back, a few seconds of each are not - the new tracks continue the old ones; a third cow
    that looks like neither stays new. The linker follows the same late links."""
    import mct
    rng = np.random.default_rng(11)
    cows = {c: _unit(rng.normal(size=64)) for c in "ABC"}
    frame = lambda c: _unit(cows[c] + 0.15 * rng.normal(size=64))          # one frame: cos ~0.6 to her
    where = {"A": _box(0.3), "B": _box(0.7), "C": _box(0.5, 0.2)}
    cfg = _mct_cfg([("cam1", {})], still_s=0.0, reacquire_sim=0.9, late_sim=0.5)
    gt = mct.GlobalTracker(cfg)
    old = None
    for k in range(6):
        old = gt.update("cam1", float(k), [{"box": where[c], "emb": frame(c)} for c in "AB"])
    new = None
    for k in range(21, 26):                                                 # back after 15 s, C new
        new = gt.update("cam1", float(k), [{"box": where[c], "emb": frame(c)} for c in "ABC"])
    assert new[0] == old[0] and new[1] == old[1] and new[2] not in old
    assert gt.stats["late_reacquired"] == 2 and gt.stats["reacquired"] == 0
    assert set(gt.alias.values()) == set(old)

    # the linker: the same late links joined (one camera)
    lk = mct.TrackletLinker(cfg)
    rng = np.random.default_rng(11)
    cows = {c: _unit(rng.normal(size=64)) for c in "ABC"}
    first = None
    for k in range(6):
        out = lk.update_all(float(k), {"cam1": [{"box": where[c], "emb": frame(c)} for c in "AB"]})["cam1"]
        first = first or out
    for k in range(21, 26):
        out = lk.update_all(float(k), {"cam1": [{"box": where[c], "emb": frame(c)} for c in "ABC"]})["cam1"]
    assert out[:2] == first and lk.stats["linked"] == 2


def test_mct_eval_uses_the_identity_models_thresholds():
    import types
    import mct_eval
    args = types.SimpleNamespace(reacquire_sweep=[0.9], apart=500.0, dist_sweep=[], floor_area=[], fuse_sweep=[],
                                 link_sweep=[], appearance=False, sweep=[], dino=False)
    plain = {n: o for n, _, _, o in mct_eval.variants(args, True)}
    assert plain["fid/link-final"] == {"reacquire_sim": 0.9} and "fid/still+late" not in plain
    cal = {"reacquire_sim": 0.97, "late_sim": 0.98}
    v = {n: o for n, _, _, o in mct_eval.variants(args, True, cal)}
    # calibrated: no fixed 0.9 (cmd_score lays the thresholds under every fid tracker); the late reacquire,
    # off by default, tried at its calibrated threshold
    assert v["fid/link-final"] == {} and v["fid/still+late"] == {"late_sim": 0.98}
    assert v["fid/link-final+late"] == {"late_sim": 0.98} and v["fid/still~0.9"] == {"reacquire_sim": 0.9}
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "emb"))
        np.savez(os.path.join(d, "emb", "cam_1.npz"), ts=np.zeros(0), stamp=json.dumps({"thresholds": cal}))
        assert mct_eval.load_thresholds(d, ["cam_1"]) == cal
        assert mct_eval.load_thresholds(d, ["cam_9"]) == {}


def _lost_and_back(update):
    """One camera, three cows; the detector's gaps as in the barn. -> {name: [track per second seen]}."""
    lying, walker, hidden = _box(0.2), _box(0.5), _box(0.8)
    seen = collections.defaultdict(list)
    for t in range(400):
        boxes = {}
        if t < 60 or t >= 180:                       # lying still, then not detected for 2 minutes
            boxes["lying"] = lying
        if t < 50:                                   # walks off at 40 s: gone by 50 s
            boxes["walker"] = _box(0.5 + 0.01 * max(0, t - 40))
        if t >= 250:                                 # another cow lies down where the walker was
            boxes["newcomer"] = walker
        if t < 100 or t >= 120:                      # half hidden by her neighbour 20 s: a smaller box inside
            boxes["hidden"] = hidden if t < 100 or t >= 140 else [hidden[0] + 0.05, hidden[1] + 0.06, hidden[2] - 0.02,
                                                                   hidden[3]]
        names = list(boxes)
        for n, tid in zip(names, update(float(t), [boxes[n] for n in names])):
            seen[n].append(tid)
    return seen


def test_lying_cow_lost_minutes_kept_walker_not():
    """The detector loses a lying cow for 2 minutes: the same track when she is back; a cow that walked
    off is not waited for long - another lying down there later is a new track; a half hidden cow (her
    box smaller, inside) keeps hers. Both trackers: the camera's own and [mct]'s."""
    import mct
    import pipeline
    tk = DEFAULTS["tracker"]
    simple = pipeline.Tracker("cam1", tk["iou"], tk["max_age_s"], tk["still_s"], tk["still_iou"], tk["still_contain"],
                              tk["still_for_s"], tk["still_moved"], tk["still_long_s"])
    gt = mct.GlobalTracker(_mct_cfg([("cam1", {})]))
    def by_box(ts, bs):                                  # the camera's tracker answers in its own order
        got = {tuple(b): tid for tid, b in simple.update(ts, bs)}
        return [got[tuple(b)] for b in bs]
    for name, update in (("simple", by_box),
                         ("mct", lambda ts, bs: gt.update("cam1", ts, [{"box": b, "emb": None} for b in bs]))):
        seen = _lost_and_back(update)
        assert len(set(seen["lying"])) == 1, (name, set(seen["lying"]))
        assert len(set(seen["hidden"])) == 1, (name, set(seen["hidden"]))
        assert not set(seen["newcomer"]) & set(seen["walker"]), name


def test_recordings_played_together():
    """Three recordings played in threads at their own speeds: none gets more than a second (and a frame)
    ahead of another still playing; one beginning later holds no one up before it begins; one done, none."""
    import threading
    import time as time_
    import pipeline
    starts = {"fast": 1000.0, "slow": 1000.0, "late": 1060.0}
    ends = {"fast": 1200.0, "slow": 1100.0, "late": 1200.0}
    lock = pipeline.Lockstep(starts)
    pos, spread = {}, []
    guard = threading.Lock()

    def play(cam, pause):
        try:
            ts = starts[cam]
            while ts < ends[cam]:
                lock.wait(cam, ts)
                with guard:
                    pos[cam] = ts
                    live = [t for c, t in pos.items() if c not in lock.done]
                    if len(pos) == 3 and min(live) >= starts["late"] + 1:   # all three playing
                        spread.append(max(live) - min(live))
                time_.sleep(pause)
                ts += 0.5
        finally:
            lock.finish(cam)
    th = [threading.Thread(target=play, args=(c, p)) for c, p in (("fast", 0.0), ("slow", 0.002), ("late", 0.0))]
    for t in th:
        t.start()
    for t in th:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in th)
    assert pos == {"fast": 1199.5, "slow": 1099.5, "late": 1199.5}
    assert spread and max(spread) <= 1.5, max(spread)


def test_trial_counts_what_cannot_be():
    """One id on two tracks of one camera at the same second, or on two cameras that share no floor: counted."""
    import trial
    from store import Store
    with tempfile.TemporaryDirectory() as d:
        s = Store(os.path.join(d, "s.sqlite"))
        rows = []
        for t in range(10):
            rows += [("cam1", "t1", 100.0 + t, 0.1, 0.1, 0.2, 0.2, "lying", "none", 0.9),
                     ("cam1", "t2", 100.0 + t, 0.5, 0.1, 0.6, 0.2, "lying", "none", 0.9),
                     ("cam3", "t3", 100.0 + t, 0.5, 0.1, 0.6, 0.2, "lying", "none", 0.9)]
        s.seconds(rows)
        tm = {"t1": "A", "t2": "A", "t3": "B"}                       # t1, t2: one camera, one second, one id
        cfg = _cfg()
        cfg["mct"]["disjoint"] = [["cam1", "cam3"]]
        odd = trial.impossible(s, tm, cfg)
        assert odd["one_cow_twice_on_one_camera"] == 10 and odd["one_cow_on_cameras_apart"] == 0
        odd = trial.impossible(s, {"t1": "A", "t2": "C", "t3": "A"}, cfg)   # cam1 and cam3 at once
        assert odd["one_cow_on_cameras_apart"] == 10 and odd["cows_seen_where_they_cannot_be"] == {"A": 10}


def test_trial_scored_against_your_ids():
    """Two of your cows given one of its cows: IDF1 one half, the merge named; a box outside the camera's
    mask is not scored; a frame of a moment the trial did not play is said so."""
    import trial
    from store import Store
    with tempfile.TemporaryDirectory() as d:
        st = os.path.join(d, "s.sqlite")
        s = Store(st)
        rows = []
        for t in range(100, 121):
            rows += [("cam1", "t1", float(t), 0.1, 0.4, 0.3, 0.6, "lying", "none", 0.9),
                     ("cam1", "t2", float(t), 0.5, 0.4, 0.7, 0.6, "lying", "none", 0.9)]
        s.seconds(rows)
        for tr in ("t1", "t2"):
            for k in range(3):
                s.burst(("cam1", tr, 100.0 + k, "confirmed", "P1", 0.9, 0.2, 0.9, 0.1, "lying", "none", None, 0.5, 100))
        boxes = os.path.join(d, "boxes")
        os.makedirs(boxes)
        frames = [{"file": f"cam1/{t}.jpg", "cam": "cam1", "ts": float(t), "w": 1920, "h": 1080}
                  for t in (100, 105, 110, 115, 500)]
        json.dump({"frames": frames}, open(os.path.join(boxes, "frames.json"), "w"))
        lab = {"boxes": [[0.1, 0.4, 0.3, 0.6, "4"], [0.5, 0.4, 0.7, 0.6, "7"], [0.9, 0.1, 0.95, 0.2, "9"]],
               "status": "done"}
        labels = os.path.join(d, "labels.json")
        json.dump({"frames": {f["file"]: lab for f in frames}}, open(labels, "w"))
        cfg = _cfg()
        cfg["cameras"] = [{"id": "cam1", "mask": [[0, 0.3], [0.8, 0.3], [0.8, 0.8], [0, 0.8]]}]
        from PIL import Image
        os.makedirs(os.path.join(boxes, "cam1"))
        for f in frames:
            Image.new("RGB", (320, 180), (90, 70, 50)).save(os.path.join(boxes, f["file"]))
        sc = trial.score(st, cfg, boxes, labels, page_dir=d)
        a = sc["all"]
        assert a["frames"] == 4 and a["frames_not_played"] == 1 and a["your_boxes"] == 8 and a["found"] == 8
        assert abs(a["idf1"] - 0.5) < 1e-9 and list(sc["merged"]) == ["P1"] and sc["your_cows"] == 2
        # the page: your two names of its one cow side by side, the labels to save back
        html = open(sc["same_cow_page"], encoding="utf-8").read()
        data = json.loads(html.split("const DATA = ", 1)[1].split(";\n", 1)[0])
        assert [x["id"] for x in data["cards"][0]["ids"]] == ["4", "7"] and len(data["labels"]["frames"]) == 5
        assert all(os.path.exists(os.path.join(d, c["src"])) for x in data["cards"][0]["ids"] for c in x["crops"])
        # every moment box by box: the frames read where they are, the one outside the mask kept (not scored)
        page = open(sc["boxes_page"], encoding="utf-8").read()
        bd = json.loads(page.split("const DATA = ", 1)[1].split(";\n", 1)[0])
        assert len(bd["frames"]) == 4 and bd["mapping"] == sc["all"]["mapping"]
        f0 = bd["frames"][0]
        assert os.path.exists(os.path.join(d, f0["src"])) and [b[5] for b in f0["mine"]] == [True, True, False]
        assert sorted(map(tuple, f0["hit"])) == [(0, 0), (1, 1)]


def test_trial_says_why_a_track_has_no_cow():
    import report
    import trial
    from store import Store
    with tempfile.TemporaryDirectory() as d:
        s = Store(os.path.join(d, "s.sqlite"))
        s.seconds([("cam1", t, 100.0 + k, 0.1, 0.1, 0.2, 0.2, "lying", "none", 0.9)
                   for t in ("named", "short", "unsure", "split") for k in range(5)])
        b = lambda tr, st, cow, ts: s.burst(("cam1", tr, ts, st, cow, 0.9, 0.2, 0.9, 0.1, "lying", "none", None, 0.5, 100))
        for k in range(3):
            b("named", "confirmed", "P1", 100.0 + k)
            b("unsure", "tentative", "P2", 100.0 + k)
        b("split", "confirmed", "P1", 100.0)
        b("split", "confirmed", "P3", 101.0)
        why = trial.unnamed(s, report.track_map(s))
        assert set(why) == {"short", "unsure", "split"}
        assert "no burst" in why["short"] and "none sure" in why["unsure"] and "split" in why["split"]


def test_trial_cross_camera_and_overlap():
    """Your cow on two cameras: the same cow of its on both or not, by cow and by moment; --overlap grows each
    mask to its tracking_mask and turns off the two rules that make two cameras' counts two cows."""
    import copy as _copy
    import trial
    from common import DEFAULTS
    seen = {"4": [("cam1", 100.0, "P1"), ("cam1", 105.0, "P1"), ("cam1", 110.0, "P1"),
                  ("cam2", 100.2, "P1"), ("cam2", 105.0, "P1"), ("cam2", 110.0, "P1")],
            "7": [("cam1", 100.0, "P2"), ("cam1", 105.0, "P2"), ("cam1", 110.0, "P2"),
                  ("cam2", 100.0, "P3"), ("cam2", 105.0, "P3"), ("cam2", 120.0, "P3")],
            "9": [("cam1", 100.0, "P4")] * 5}
    x = trial.cross_camera(seen)
    assert x["cows_on_two_cameras"] == 2 and x["same_cow_on_all"] == 1
    assert x["cows"]["4"]["same"] and not x["cows"]["7"]["same"] and "9" not in x["cows"]
    assert x["moments"] == 5 and x["moments_same_cow"] == 3            # 4 at 100/105/110, 7 at 100/105
    cfg = _copy.deepcopy(DEFAULTS)
    cfg["cameras"] = [{"id": "cam1", "mask": [[0, 0], [0.5, 0], [0.5, 1]], "tracking_mask": [[0, 0], [1, 0], [1, 1]]},
                      {"id": "cam2", "mask": [[0.5, 0], [1, 0], [1, 1]]}]
    trial.overlap(cfg)
    assert cfg["cameras"][0]["mask"] == [[0, 0], [1, 0], [1, 1]] and cfg["cameras"][1]["mask"] == [[0.5, 0], [1, 0], [1, 1]]
    assert cfg["identity"]["counted_apart_s"] == 0 and cfg["mct"]["unlink_s"] == 0


def test_trial_score_finds_frames_cut_with_another_time_zone():
    """Frames cut when timezone_offset_hours was one more: an hour off the run - found and scored."""
    import trial
    from store import Store
    with tempfile.TemporaryDirectory() as d:
        st = os.path.join(d, "s.sqlite")
        s = Store(st)
        s.seconds([("cam1", "t1", float(t), 0.1, 0.4, 0.3, 0.6, "lying", "none", 0.9) for t in range(5000, 5100)])
        for k in range(3):
            s.burst(("cam1", "t1", 5000.0 + k, "confirmed", "P1", 0.9, 0.2, 0.9, 0.1, "lying", "none", None, 0.5, 100))
        boxes = os.path.join(d, "boxes")
        os.makedirs(boxes)
        frames = [{"file": f"f{t}.jpg", "cam": "cam1", "ts": float(t) - 3600, "w": 1920, "h": 1080}
                  for t in range(5010, 5090, 10)]
        json.dump({"frames": frames}, open(os.path.join(boxes, "frames.json"), "w"))
        labels = os.path.join(d, "labels.json")
        json.dump({"frames": {f["file"]: {"status": "done", "boxes": [[0.1, 0.4, 0.3, 0.6, "4"]]} for f in frames}},
                  open(labels, "w"))
        cfg = _cfg()
        cfg["cameras"] = [{"id": "cam1", "mask": []}]
        sc = trial.score(st, cfg, boxes, labels)
        assert sc["clock_shift_s"] == 3600 and sc["all"]["frames"] == 8 and sc["all"]["idf1"] == 1.0, sc["all"]
        assert trial.score(st, cfg, boxes, labels, cameras=["cam2"])["all"]["frames"] == 0     # cam1 left out


def test_tracks_counted_by_two_cameras_at_once_are_two_cows():
    """Masks share no floor: a cow is counted by one camera at a time. Two tracks counted by cam1 and cam2 at
    the same second for counted_apart_s seconds are two cows; a moment on the line between masks is not;
    one track (joined across the cameras) is never apart from itself. Kept in the store by save()."""
    import pipeline
    from store import Store
    with tempfile.TemporaryDirectory() as d:
        st = Store(os.path.join(d, "s.sqlite"))
        cfg = _cfg(counted_apart_s=10)
        cfg["cameras"] = [{"id": "cam1", "mask": [[0, 0], [1, 0], [1, 1]]}, {"id": "cam2", "mask": [[0, 0], [1, 0], [1, 1]]}]
        g = pipeline.SharedGallery(type("G", (), {"save": lambda self: None, "migration_due": lambda self, ts: False})(),
                                   st, cfg, 0.0)
        g.next_nightly = 1e12
        for t in range(12):
            g.counted("cam1", 100.0 + t, ["a", "joined"])
            g.counted("cam2", 100.0 + t, ["b", "joined"])
        for t in range(3):                                    # a moment only
            g.counted("cam1", 200.0 + t, ["c"])
            g.counted("cam2", 200.0 + t, ["e"])
        assert g.apart("a", "b") and g.apart("a", "joined") and not g.apart("joined", "joined")
        assert not g.apart("c", "e")
        g.save()
        assert ("a", "b") in set(st.cannot_link_pairs())


def test_neighbours_without_floor_joined_by_look_in_their_strip():
    """cam4 sees, beyond its mask (its tracking_mask), the strip along the fence that cam3 counts. No floor
    mapping between them. A cow there - cam3's, seen by cam4 too - becomes one track: her frame ids
    clearly nearest to one of cam3's cows over look_min_s seconds; two look-alikes are left apart."""
    import mct
    rng = np.random.default_rng(4)
    A, B, C = (_unit(rng.normal(size=64)) for _ in range(3))
    near = lambda v: _unit(v + 0.1 * rng.normal(size=64))           # one frame: cos ~0.6 to her own
    cfg = _cfg()
    cfg["cameras"] = [{"id": "cam3", "mask": [[0, 0], [1, 0], [1, 1], [0, 1]]},
                      {"id": "cam4", "mask": [[0.3, 0], [1, 0], [1, 1], [0.3, 1]],
                       "tracking_mask": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
    cfg["mct"].update(neighbours=[["cam3", "cam4"]], look_min_s=10, look_min_sim=0.5, look_margin=0.05)
    lk = mct.TrackletLinker(cfg)
    box = lambda x, y=0.5: [x - 0.05, y - 0.05, x + 0.05, y + 0.05]
    for k in range(15):
        out = lk.update_all(float(k), {
            "cam3": [{"box": box(0.2), "emb": near(A)}, {"box": box(0.6), "emb": near(B)}],
            "cam4": [{"box": box(0.1), "emb": near(A)},              # in cam4's strip: cam3's cow A
                     {"box": box(0.7), "emb": near(C)}]})            # cam4's own cow
    assert out["cam4"][0] == out["cam3"][0] and out["cam4"][1] not in out["cam3"]
    assert lk.stats["joined_by_look"] == 1

    # two of cam3's cows alike: the one in the strip is not given to either
    lk = mct.TrackletLinker(cfg)
    for k in range(15):
        out = lk.update_all(float(k), {"cam3": [{"box": box(0.2), "emb": near(A)}, {"box": box(0.6), "emb": near(A)}],
                                       "cam4": [{"box": box(0.1), "emb": near(A)}]})
    assert out["cam4"][0] not in out["cam3"] and lk.stats["joined_by_look"] == 0

    # cam4's own cow (once inside its mask) that walks into the strip is not a visitor: never joined
    lk = mct.TrackletLinker(cfg)
    for k in range(70):
        x = max(0.1, 0.5 - 0.02 * max(0, k - 10))              # walks from cam4's floor into the strip
        out = lk.update_all(float(k), {"cam3": [{"box": box(0.2), "emb": near(A)}],
                                       "cam4": [{"box": box(x), "emb": near(A)}]})
    assert out["cam4"][0] != out["cam3"][0] and lk.stats["joined_by_look"] == 0


class _WalkModels:
    """The detector reads the cow's place off the frame (its first pixel); posture and activity fixed."""
    crop_size, margin = 16, 0.1

    def detect(self, img, **kw):
        x = img[0, 0, 0] / 255.0
        return [[x - 0.05, 0.4, x + 0.05, 0.6]]

    def encode(self, crops, **kw):
        return None

    def frame_heads(self, feats, boxes, with_id=False, **kw):
        n = len(boxes)
        pp, pa = np.tile([0.1, 0.9], (n, 1)), np.tile([0.1, 0.1, 0.8], (n, 1))
        return (pp, pa, [None] * n) if with_id else (pp, pa)


def test_walking_cow_keeps_her_track_with_track_fps():
    """A cow walking 1.5 box widths a second: tracked once a second her boxes never overlap - a new track
    every second; with track_fps 4 the frames between keep her on one track (still one row a second)."""
    import mct
    import pipeline
    from store import Store
    tracks = {}
    for link, fps in itertools.product((False, True), (1.0, 4.0)):
        cfg = _cfg()
        cfg["sampling"]["track_fps"] = fps
        cfg["cameras"] = [{"id": "cam1", "mask": []}]
        cfg["mct"].update(enabled=link, mode="link")
        with tempfile.TemporaryDirectory() as d:
            store = Store(os.path.join(d, "s.sqlite"))
            shared = pipeline.SharedGallery(_gallery(d, {}), store, cfg, 0.0)
            cam = pipeline.Camera(cfg["cameras"][0], 0, 1, cfg, _WalkModels(), shared, store, 0.0)
            if link:                                              # [mct] link: the per-second rendezvous
                cam.glob = mct.FusedSeconds(mct.TrackletLinker(cfg, store, shared.separate), ["cam1"])
            for i in range(41):                                   # 8 frames a second, 5 s
                t = i / 8
                img = np.zeros((64, 64, 3), np.uint8)
                img[0, 0, 0] = int(round((0.1 + 0.15 * t) * 255))
                cam.push(t, img)
                cam.step_seconds(t)
            rows = store.query("SELECT track, ts FROM seconds ORDER BY ts")
            assert [r[1] for r in rows] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]       # one row a second either way
            tracks[(link, fps)] = len({r[0] for r in rows})
            assert cam.stats["track_frames"] == (0 if fps == 1.0 else 15)
    assert tracks == {(False, 1.0): 6, (False, 4.0): 1, (True, 1.0): 6, (True, 4.0): 1}, tracks


def test_join_undone_when_both_cameras_count_the_cow():
    """A join by look, then cam4's track goes on to one of cam4's own cows (inside its mask) while cam3 still
    counts its cow: counted by both cameras at once for unlink_s seconds - two cows: undone, and the
    store's link too; they are never joined again."""
    import mct
    from store import Store
    rng = np.random.default_rng(4)
    A, B = (_unit(rng.normal(size=64)) for _ in range(2))
    near = lambda v: _unit(v + 0.1 * rng.normal(size=64))
    cfg = _cfg()
    cfg["cameras"] = [{"id": "cam3", "mask": [[0, 0], [1, 0], [1, 1], [0, 1]]},
                      {"id": "cam4", "mask": [[0.3, 0], [1, 0], [1, 1], [0.3, 1]],
                       "tracking_mask": [[0, 0], [1, 0], [1, 1], [0, 1]]}]
    cfg["mct"].update(neighbours=[["cam3", "cam4"]], look_min_s=10, look_min_sim=0.5, look_margin=0.05, unlink_s=10)
    box = lambda x, y=0.5: [x - 0.05, y - 0.05, x + 0.05, y + 0.05]
    with tempfile.TemporaryDirectory() as d:
        st = Store(os.path.join(d, "s.sqlite"))
        lk = mct.TrackletLinker(cfg, st)
        for k in range(15):                                   # cam3's cow A seen by cam4 in its strip
            out = lk.update_all(float(k), {"cam3": [{"box": box(0.2), "emb": near(A)}],
                                           "cam4": [{"box": box(0.1), "emb": near(A)}]})
        assert out["cam4"][0] == out["cam3"][0] and st.track_roots()
        for k in range(15, 60):                               # cam4's track walks into its own floor
            x = min(0.6, 0.1 + 0.02 * (k - 14))
            out = lk.update_all(float(k), {"cam3": [{"box": box(0.2), "emb": near(A)}],
                                           "cam4": [{"box": box(x), "emb": near(B)}]})
        assert out["cam4"][0] != out["cam3"][0] and lk.stats["unlinked"] == 1
        assert not st.track_roots()                                          # the link gone from the store


def test_calibrated_cows_counted_by_two_cameras_near_stay_two():
    """Two calibrated cameras, a cow counted in each one's mask at nearly one floor place (the barn: two cows
    lying side by side, each camera seeing the nearer): joined by the floor at first, then undone as two
    cows - each camera counts its own; the place said once."""
    import contextlib
    import io
    import mct
    cfg = _cfg()
    pts = [[0, 0, 0, 0], [1, 0, 1000, 0], [1, 1, 1000, 1000], [0, 1, 0, 1000]]
    cfg["cameras"] = [{"id": "cam1", "mask": [[0, 0], [0.6, 0], [0.6, 1], [0, 1]], "floor_points": pts},
                      {"id": "cam2", "mask": [[0.4, 0], [1, 0], [1, 1], [0.4, 1]], "floor_points": pts}]
    cfg["mct"].update(neighbours=[["cam1", "cam2"]], link_dist=80, unlink_s=10)
    box = [0.45, 0.4, 0.55, 0.5]
    lk = mct.TrackletLinker(cfg)
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        for k in range(40):
            out = lk.update_all(float(k), {"cam1": [{"box": box}], "cam2": [{"box": [v + 0.03 for v in box]}]})
    assert out["cam1"][0] != out["cam2"][0] and lk.stats["linked"] == 1 and lk.stats["unlinked"] == 1
    assert lk.stats["counted_by_two_near"] > 0 and said.getvalue().count("nearly one floor place") == 1


def test_linker_llr_joins_through_wrong_seconds_and_keeps_neighbours_apart():
    """Three cows lying 1.2 m apart in a row, camera b's floor 60 cm off along the row and in 30% of the seconds
    1-2 m anywhere: counted second by second a neighbour is often the nearest; summed as evidence, each cow is
    joined to her own track and no two cows to one."""
    import mct
    pts = [[0, 0, 0, 0], [1, 0, 1000, 0], [1, 1, 1000, 1000], [0, 1, 0, 1000]]
    box = lambda x: [x - 0.03, 0.45, x + 0.03, 0.5]
    rng = np.random.default_rng(0)
    cfg = _cfg()
    cfg["cameras"] = [{"id": c, "mask": [], "floor_points": pts} for c in ("a", "b")]
    cfg["mct"].update(link_rule="llr", link_dist=160.0, link_min_s=5)
    lk = mct.TrackletLinker(cfg)
    true_b = lk.floor["b"]

    def bent(b, pl=0.5):
        x, y = true_b(b, pl)
        if rng.random() < 0.3:
            a = rng.uniform(0, 2 * np.pi)
            return x + 150 * np.cos(a), y + 150 * np.sin(a)
        return x + 60 + rng.normal(0, 20), y + rng.normal(0, 20)
    lk.floor["b"] = bent
    xs = (0.40, 0.52, 0.64)
    for k in range(120):
        out = lk.update_all(float(k), {c: [{"box": box(x)} for x in xs] for c in ("a", "b")})
    assert out["a"] == out["b"]
    assert len(set(out["a"])) == 3


def test_linker_llr_learns_the_bend_between_two_cameras():
    """Camera b's floor 90 cm off everywhere: once the first cows are joined the offset is learnt from them."""
    import mct
    pts = [[0, 0, 0, 0], [1, 0, 1000, 0], [1, 1, 1000, 1000], [0, 1, 0, 1000]]
    box = lambda x, y: [x - 0.03, y - 0.05, x + 0.03, y]
    cfg = _cfg()
    cfg["cameras"] = [{"id": c, "mask": [], "floor_points": pts} for c in ("a", "b")]
    cfg["mct"].update(link_rule="llr", link_dist=160.0, link_learn_offset=True, link_offset_min=10)
    lk = mct.TrackletLinker(cfg)
    true_b = lk.floor["b"]
    lk.floor["b"] = lambda b, pl=0.5: (lambda p: (p[0] + 90.0, p[1]))(true_b(b, pl))
    cows = [(0.1 + 0.2 * i, 0.3 + 0.4 * (i % 2)) for i in range(5)]
    for k in range(60):
        out = lk.update_all(float(k), {c: [{"box": box(x, y)} for x, y in cows] for c in ("a", "b")})
    assert out["a"] == out["b"]
    off = lk._offset("a", "b", np.array([[500.0, 500.0]]))[0]
    assert abs(off[0] - 90.0) < 5 and abs(off[1]) < 5


def test_cow_merges_puts_one_cows_two_names_together():
    """Each camera names a cow by her side (xa / xb): their tracks' evidence summed over pairs of names joins xa
    with xb, not with the neighbour ya her tracks were also near now and then; names seen together never."""
    import mct
    track_cow = {}
    ev = {}
    for c in "xyz":
        for k in range(4):                                   # four short visits to the overlap, each too weak alone
            ta, tb = f"{c}a{k}", f"{c}b{k}"
            track_cow[ta], track_cow[tb] = f"{c}a", f"{c}b"
            ev[(ta, tb)] = 20.0
    ev[("xa0", "yb0")] = 25.0                                # a neighbour, near once
    apart = [("xa", "ya"), ("xa", "za"), ("ya", "za"), ("xb", "yb"), ("xb", "zb"), ("yb", "zb")]
    got = dict(mct.cow_merges(ev, track_cow, apart, min_llr=60.0, margin=30.0))
    top = lambda n: top(got[n]) if n in got else n
    assert all(top(f"{c}a") == top(f"{c}b") for c in "xyz")
    assert len({top(f"{c}a") for c in "xyz"}) == 3
    kept = dict(mct.cow_merges(ev, track_cow, apart + [("xa", "xb")], min_llr=60.0, margin=30.0))
    top = lambda n: top(kept[n]) if n in kept else n
    assert top("xa") != top("xb") and top("ya") == top("yb")


def test_merge_by_place_merges_the_gallery_and_the_reports_follow():
    """The night: cow A (camera a's name) and cow B (camera b's) are one by their tracks' evidence - the gallery
    keeps one, the store aliases the other, the report counts her tracks on both cameras as one cow."""
    import pipeline
    import report
    from store import Store
    with tempfile.TemporaryDirectory() as d:
        store = Store(os.path.join(d, "s.sqlite"))
        for k in range(3):
            store.burst(("a", "ta", 100.0 + k, "confirmed", "A", 0.9, 0.2, 0.9, 0.1, "lying", "none", None, 0.5, 100))
            store.burst(("b", "tb", 100.0 + k, "confirmed", "B", 0.9, 0.2, 0.9, 0.1, "lying", "none", None, 0.5, 100))
        store.put_evidence([("ta", "tb", 250.0, 300, 400.0)])
        gal = _gallery(os.path.join(d, "g"), {"A": [1, 0, 0], "B": [0, 1, 0]})
        cfg = _cfg()
        cfg["mct"].update(enabled=True, mode="link", link_rule="llr")
        assert pipeline.merge_by_place(gal, store, cfg, 1000.0) == [("B", "A")]
        assert set(gal.cows) == {"A"} and len(gal.cows["A"]["prototypes"]) == 2
        assert report.track_map(store) == {"ta": "A", "tb": "A"}
        cfg["mct"]["link_rule"] = "near"                     # the old rule keeps no evidence: nothing merged
        assert pipeline.merge_by_place(gal, store, cfg, 1000.0) == []


# ------------------------------------------------------------------ the camera's tracker: Kalman, unsure boxes

def test_box_kalman_follows_a_walking_box():
    """A box moving 0.08 a second: after a few seconds the filter expects it where it will be, not where it was."""
    from common import BoxKalman, iou
    box = lambda x: [x, 0.4, x + 0.1, 0.5]
    kf = BoxKalman(box(0.1), 0.0)
    for k in range(1, 6):
        kf.update(float(k), box(0.1 + 0.08 * k))
    assert iou(kf.predict(6.0), box(0.1 + 0.08 * 6)) > 0.8
    assert iou(box(0.1 + 0.08 * 5), box(0.1 + 0.08 * 6)) < 0.5


def test_tracker_keeps_a_fast_walker_with_kalman():
    """A cow setting off and walking up to 0.08 of the frame a second with a box 0.1 wide: at 1 fps her boxes soon
    overlap too little for IoU alone (a new track every second); with the Kalman filter one track."""
    import mct
    import pipeline
    box = lambda x: [x, 0.4, x + 0.1, 0.5]
    xs = np.cumsum([0.05, 0.02, 0.035, 0.05, 0.065, 0.075, 0.08, 0.08, 0.08, 0.08])
    for kalman, ok in ((False, lambda n: n >= 5), (True, lambda n: n == 1)):
        cfg = _mct_cfg([("a", {})])
        cfg["tracker"]["kalman"] = kalman
        g = mct.GlobalTracker(cfg)
        t = pipeline.Tracker("a", 0.3, 5, kalman=kalman)
        gids, tids = set(), set()
        for k, x in enumerate(xs):
            gids |= set(g.update("a", float(k), [{"box": box(x)}]))
            tids |= {tid for tid, _ in t.update(float(k), [box(x)])}
        assert ok(len(gids)) and ok(len(tids)), (kalman, gids, tids)


def test_unsure_boxes_only_continue_a_track():
    """ByteTrack's second round: a cow's box under the threshold keeps her track; an unsure box where no track was
    starts none (None / left out)."""
    import mct
    import pipeline
    box = lambda x: [x, 0.4, x + 0.1, 0.5]
    g = mct.GlobalTracker(_mct_cfg([("a", {})]))
    t = pipeline.Tracker("a", 0.3, 5)
    first = g.update("a", 0.0, [{"box": box(0.2)}])[0]
    (tid, _), = t.update(0.0, [box(0.2)])
    for k in range(1, 4):                                  # the detector unsure of her for three seconds
        out = g.update("a", float(k), [{"box": box(0.2), "low": True}, {"box": box(0.7), "low": True}])
        assert out == [first, None]
        got = t.update(float(k), [], low=[box(0.2), box(0.7)])
        assert [x for x, _ in got] == [tid]
    assert g.update("a", 30.0, [{"box": box(0.5), "low": True}]) == [None]
    lk = mct.TrackletLinker(_mct_cfg([("a", {})]))
    lk.update_all(0.0, {"a": [{"box": box(0.2)}]})
    assert lk.update_all(1.0, {"a": [{"box": box(0.2), "low": True}, {"box": box(0.7), "low": True}]})["a"][1] is None


def test_models_split_the_detectors_boxes_at_its_threshold():
    """[tracker] low_score: the detector keeps boxes down to it; detect() still gives only the sure ones."""
    import pipeline

    class Det:
        threshold = 0.5

        def __call__(self, img):
            return [{"bbox": [0, 0, 0.1, 0.1], "det_score": 0.9}, {"bbox": [0.5, 0.5, 0.6, 0.6], "det_score": 0.3},
                    {"bbox": [0.7, 0.7, 0.8, 0.8], "det_score": 0.1}][:2 + (self.threshold < 0.1)]
    m = pipeline.Models.__new__(pipeline.Models)
    m.detector, m.det_high = Det(), 0.0
    m.gpu = lambda urgent, kind: __import__("contextlib").nullcontext()
    m.detector.threshold, m.det_high = 0.2, 0.5
    img = np.zeros((4, 4, 3), np.uint8)
    assert m.detect(img) == [[0, 0, 0.1, 0.1]]
    assert m.detect_split(img) == ([[0, 0, 0.1, 0.1]], [[0.5, 0.5, 0.6, 0.6]])


# ------------------------------------------------------------------ the night unsure: a person asked

def test_cow_merges_three_zones():
    """Sure pairs merged; pairs with some evidence but not enough go to a person; the rest nothing."""
    import mct
    track_cow = {f"{c}{k}": c for c in ("xa", "xb", "ya", "yb", "za", "zb") for k in range(2)}
    ev = {("xa0", "xb0"): 300.0, ("ya0", "yb0"): 90.0, ("za0", "zb0"): 20.0}
    apart = [("xa", "ya"), ("xa", "za"), ("ya", "za"), ("xb", "yb"), ("xb", "zb"), ("yb", "zb")]
    reviews = []
    got = mct.cow_merges(ev, track_cow, apart, reviews=reviews)
    assert got == [("xb", "xa")]
    assert [r[:2] for r in reviews] == [("yb", "ya")] and reviews[0][2] == 90.0


def test_unsure_pair_asked_then_merged_or_never_again():
    """The night stores an unsure pair; a person's "same" aliases it at once and the next night merges the gallery;
    "different" keeps it from being proposed again; the page shows both names' faces."""
    import merges
    import pipeline
    import report
    from store import Store
    with tempfile.TemporaryDirectory() as d:
        store = Store(os.path.join(d, "herd.sqlite"))
        for k in range(3):
            for cam, tr, cow in (("a", "ta", "A"), ("b", "tb", "B"), ("a", "tc", "C"), ("b", "td", "D")):
                store.burst((cam, tr, 100.0 + k, "confirmed", cow, 0.9, 0.2, 0.9, 0.1, "lying", "none", None, 0.5, 100))
        store.put_evidence([("ta", "tb", 120.0, 300, 400.0), ("tc", "td", 100.0, 300, 400.0)])
        store.add_cannot_link([("ta", "tc"), ("tb", "td")], 100.0)
        gal = _gallery(os.path.join(d, "gallery"), {c: v for c, v in zip("ABCD", np.eye(4))})
        cfg = _cfg()
        cfg["mct"].update(enabled=True, mode="link", link_rule="llr")
        cfg["store"]["path"] = os.path.join(d, "herd.sqlite")
        shared = pipeline.SharedGallery(gal, store, cfg, 0.0)
        for cow, cam in (("A", "a"), ("B", "b"), ("A", "a")):
            shared.face(cow, cam, 1000.0, np.full((224, 224, 3), 128, np.uint8))
        assert len(os.listdir(os.path.join(d, "gallery", "faces", "A"))) == 1      # one each face_every_s
        assert pipeline.merge_by_place(gal, store, cfg, 1000.0) == []
        assert {(r["old"], r["into"]) for r in store.reviews("pending")} == {("B", "A"), ("D", "C")}
        page = os.path.join(d, "merges.html")
        merges.main(["page", "--store", store.db.execute("PRAGMA database_list").fetchone()[2], "--out", page])
        html = open(page, encoding="utf-8").read()
        assert "gallery/faces/A/a_1000.jpg" in html and "gallery/faces/B/b_1000.jpg" in html
        dec = os.path.join(d, "decisions.json")
        json.dump({"decisions": [{"old": "B", "into": "A", "answer": "same"},
                                 {"old": "D", "into": "C", "answer": "different"}]}, open(dec, "w"))
        merges.main(["apply", "--store", os.path.join(d, "herd.sqlite"), "--decisions", dec])
        assert report.track_map(store)["tb"] == "A" and set(gal.cows) == set("ABCD")   # the reports at once
        assert pipeline.merge_by_place(gal, store, cfg, 2000.0) == [("B", "A")]      # the gallery the next night
        assert set(gal.cows) == {"A", "C", "D"}
        assert not store.reviews("pending")
        store.put_evidence([("tc", "td", 1000.0, 3000, 2500.0)])                    # however sure it looks now
        assert pipeline.merge_by_place(gal, store, cfg, 3000.0) == [] and "D" in gal.cows


# ------------------------------------------------------------------ the cameras' clocks

def test_linker_measures_how_late_a_cameras_frames_come():
    """Cows walking on two cameras whose floor plans agree, camera b's frames 1.2 s late: her two tracks are joined
    and the linker says b is 1.2 s late (what time_offset_s = -1.2 on b puts right)."""
    import mct
    pts = [[0, 0, 0, 0], [1, 0, 2000, 0], [1, 1, 2000, 1000], [0, 1, 0, 1000]]
    box = lambda x, y: [x / 2000 - 0.03, y / 1000 - 0.08, x / 2000 + 0.03, y / 1000]
    cfg = _cfg()
    cfg["cameras"] = [{"id": c, "mask": [], "floor_points": pts} for c in ("a", "b")]
    cfg["mct"].update(link_rule="llr", link_dist=160.0)
    for late in (1.2, -0.7, 0.0):
        lk = mct.TrackletLinker(cfg)
        where = lambda t, y0: (200 + 60 * max(0.0, t - 30), y0)         # 30 s still, then walking 60 cm/s
        for k in range(150):
            out = lk.update_all(float(k), {"a": [{"box": box(*where(k, y))} for y in (300, 700)],
                                           "b": [{"box": box(*where(k - late, y))} for y in (300, 700)]})
        assert out["a"] == out["b"]
        (lag, n, sharp), = lk.clocks(min_n=1).values()
        assert abs(lag - late) <= 0.15 and sharp > 20, (late, lag, n, sharp)


def test_neighbours_on_floor_plans_of_their_own_joined_by_look():
    """cam3 and cam4 each calibrated on a plane of its own, placed far apart on the plan (the barn's config): by place
    they never meet, so their strip is joined by look, as for neighbours with no floor mapping."""
    import mct
    plan = lambda x0: [[0, 0, x0, 0], [1, 0, x0 + 1000, 0], [1, 1, x0 + 1000, 1000], [0, 1, x0, 1000]]
    cfg = _cfg()
    cfg["cameras"] = [{"id": "cam3", "mask": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]], "floor_points": plan(200000)},
                      {"id": "cam4", "mask": [[0.5, 0], [1, 0], [1, 1], [0.5, 1]], "floor_points": plan(400000)},
                      {"id": "cam5", "mask": [], "floor_points": plan(400500)}]
    cfg["mct"].update(neighbours=[["cam3", "cam4"], ["cam4", "cam5"]])
    assert mct.TrackletLinker(cfg).by_look == [("cam3", "cam4")]


def test_trial_says_how_often_one_cow_on_two_cameras_got_one_id():
    """Your cow in both cameras' frames at one moment: on one joined track, the same cow, two cows, or no name on one
    of them - counted per moment, per camera pair and per cow of yours."""
    import trial
    seen = {"4": [("cam1", 100.0, "g1", "P1"), ("cam2", 100.2, "g1", "P1"),            # one track, one id
                  ("cam1", 105.0, "g1", "P1"), ("cam2", 105.0, "g7", None)],           # her cam2 track: no name yet
            "7": [("cam1", 100.0, "g2", "P2"), ("cam2", 100.0, "g3", "P3"),            # two tracks, two cows
                  ("cam1", 110.0, "g2", "P2"), ("cam2", 130.0, "g3", "P2")]}           # not at one moment
    m = trial.cross_camera_moments(seen)
    assert (m["moments"], m["one_track"], m["same_cow"], m["two_cows"], m["no_name"]) == (3, 1, 1, 1, 1)
    assert m["cows"]["7"] == {"moments": 1, "one_track": 0, "same_cow": 0, "two_cows": 1, "no_name": 0}
    assert m["camera_pairs"]["cam1-cam2"]["moments"] == 3


def test_trial_relink_joins_a_recorded_trial_anew():
    """trial --relink: the second pass's tracks as they were, joined across cameras again with the config's floor -
    a copy in OUT/relink, the old joins cleared, the original store untouched."""
    import trial
    from store import Store
    pts = [[0, 0, 0, 0], [1, 0, 1000, 0], [1, 1, 1000, 1000], [0, 1, 0, 1000]]
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "pass2"))
        st = Store(os.path.join(d, "pass2", "herd.sqlite"))
        rows = []
        for t in range(100, 220):
            for cam, tr, x in (("cam1", "a1", 0.3), ("cam1", "a2", 0.6), ("cam2", "b1", 0.3), ("cam2", "b2", 0.6)):
                rows.append((cam, tr, float(t), x - 0.03, 0.45, x + 0.03, 0.5, "lying", "none", 0.9))
        st.seconds(rows)
        st.link_tracks(100.0, "b2", "a1")                    # a wrong join of the trial: cleared
        cfg = _cfg()
        cfg["cameras"] = [{"id": c, "mask": [], "floor_points": pts} for c in ("cam1", "cam2")]
        cfg["mct"].update(link_rule="llr")
        dst = trial.relink(cfg, d)
        roots = Store(dst).track_roots()
        assert {frozenset((t, r)) for t, r in roots.items()} == {frozenset(("a1", "b1")), frozenset(("a2", "b2"))}
        assert Store(os.path.join(d, "pass2", "herd.sqlite")).track_roots() == {"b2": "a1"}


def test_joined_tracks_two_cameras_two_names_one_cow():
    """cam1 named her A (7 bursts), cam2 B (10): joined, the two-thirds rule over the group gave no name at all.
    Each camera clear on its own: the group takes B (more bursts), the night sees A and B as one cow. A camera
    split between two names still gives none."""
    import report
    from store import Store
    with tempfile.TemporaryDirectory() as d:
        store = Store(os.path.join(d, "s.sqlite"))
        for cam, tr, cow, n in (("cam1", "ta", "A", 7), ("cam2", "tb", "B", 10), ("cam1", "tc", "C", 3),
                                ("cam1", "tc", "D", 3), ("cam2", "td", "C", 5)):
            for k in range(n):
                store.burst((cam, tr, 100.0 + k, "confirmed", cow, 0.9, 0.2, 0.9, 0.1, "lying", "none", None, 0.5, 100))
        store.link_tracks(150.0, "tb", "ta")
        store.link_tracks(150.0, "td", "tc")
        assert report.track_map(store) == {"ta": "B", "tb": "B"}
        assert report.track_map(store, joins=False) == {"ta": "A", "tb": "B", "td": "C"}


def test_boxes_page_shows_one_cow_joined_across_cameras():
    """boxes.html: your cow on both cameras at one moment - its two tracks joined: the group's number on both
    (joined), the page's data says which root each box is under."""
    import trial
    from store import Store
    with tempfile.TemporaryDirectory() as d:
        st = Store(os.path.join(d, "s.sqlite"))
        rows = []
        for t in range(100, 121):
            rows += [("cam1", "a1", float(t), 0.1, 0.4, 0.3, 0.6, "lying", "none", 0.9),
                     ("cam2", "b1", float(t), 0.5, 0.4, 0.7, 0.6, "lying", "none", 0.9),
                     ("cam2", "b2", float(t), 0.1, 0.1, 0.2, 0.2, "lying", "none", 0.9)]
        st.seconds(rows)
        for tr, cam, cow in (("a1", "cam1", "P1"), ("b1", "cam2", "P1"), ("b2", "cam2", "P2")):
            for k in range(3):
                st.burst((cam, tr, 100.0 + k, "confirmed", cow, 0.9, 0.2, 0.9, 0.1, "lying", "none", None, 0.5, 100))
        st.link_tracks(105.0, "b1", "a1")
        boxes = os.path.join(d, "boxes")
        os.makedirs(boxes)
        frames = [{"file": f"{c}/{t}.jpg", "cam": c, "ts": float(t), "w": 1920, "h": 1080} for c in ("cam1", "cam2")
                  for t in (100, 110)]
        json.dump({"frames": frames}, open(os.path.join(boxes, "frames.json"), "w"))
        lab = {"cam1": [[0.1, 0.4, 0.3, 0.6, "4"]], "cam2": [[0.5, 0.4, 0.7, 0.6, "4"], [0.1, 0.1, 0.2, 0.2, "9"]]}
        json.dump({"frames": {f["file"]: {"boxes": lab[f["cam"]], "status": "done"} for f in frames}},
                  open(os.path.join(d, "labels.json"), "w"))
        cfg = _cfg()
        cfg["cameras"] = [{"id": "cam1", "mask": []}, {"id": "cam2", "mask": []}]
        sc = trial.score(os.path.join(d, "s.sqlite"), cfg, boxes, os.path.join(d, "labels.json"), page_dir=d)
        assert sc["cross_camera_moments"]["one_track"] == 2
        bd = json.loads(open(sc["boxes_page"], encoding="utf-8").read().split("const DATA = ", 1)[1].split(";\n", 1)[0])
        assert bd["joined"] == {"a1": 1}
        cam2 = next(f for f in bd["frames"] if f["cam"] == "cam2")
        assert sorted(b[6] for b in cam2["its"]) == ["a1", "b2"]


def test_floor_point_uncertainty_through_the_cameras_mapping():
    """link_px (Sternig et al.): a far box's point is metres of floor along the camera's view, a near one's
    tens of cm - so 1.5 m off far away is still one cow, while 1.5 m off right under the camera is not."""
    import mct
    import mct_sim
    mct_sim.perspective(True)
    try:
        cams = [{"id": c, "mask": [], "floor_points": mct_sim.floor_points(c), "floor_at": "centre"} for c in ("a", "b")]
        far_a, near_a = mct_sim.cow_box("a", 1300.0, 600.0), mct_sim.cow_box("a", 150.0, 600.0)
    finally:
        mct_sim.perspective(False)
    cfg = _cfg()
    cfg["cameras"] = cams
    cfg["mct"].update(link_rule="llr", link_sigma=30.0, link_px=8.0)
    lk = mct.TrackletLinker(cfg)
    (pf,), (pn,) = lk._points("a", [{"box": far_a}]), lk._points("a", [{"box": near_a}])
    ev_far, ev_near = np.linalg.eigvalsh(pf[3]), np.linalg.eigvalsh(pn[3])
    assert np.sqrt(ev_far[-1]) > 2.5 * np.sqrt(ev_near[-1])         # far: much broader
    assert np.sqrt(ev_far[-1]) > 1.8 * np.sqrt(ev_far[0])            # and long along the view (x, the barn's length)
    exact = lambda p, dx: (((p[0][0] + dx, p[0][1]), (p[1][0] + dx, p[1][1]), p[2], np.eye(2) * lk.sigma ** 2 / 2))
    llr = lambda p, dx: float(lk._llr_cov("a", "b", [p], [exact(p, dx)])[0, 0])
    assert llr(pf, 150.0) > 0 > llr(pn, 150.0)
    flat = mct.TrackletLinker(dict(cfg, mct=dict(cfg["mct"], link_px=0.0)))
    assert float(flat._llr(np.array([[150.0]]))[0, 0]) < 0          # one sigma for all: far 1.5 m counts against


def test_fused_seconds_meet_by_the_recordings_clock_however_slow_a_camera():
    """Files: a camera a second ahead (pipeline.Lockstep allows one) met the other one second off, every second -
    by the wall clock (wait_s) the barn's trial joined no track across cameras. By the recordings' clock every
    second meets, a camera whose recording ends early or begins late holds no one up."""
    import threading
    import time
    import mct
    import pipeline

    class Rec:
        stats = {}

        def __init__(self):
            self.calls = []

        def update_all(self, ts, per):
            self.calls.append((ts, tuple(sorted(per))))
            return {c: [None] * len(d) for c, d in per.items()}

    def play(cam, fs, lock, start, seconds, busy):
        for k in range(seconds * 10):                      # 10 frames a second
            ts = start + k / 10
            lock.wait(cam, ts)
            if k % 10 == 0:
                time.sleep(busy.get(ts, 0.03))             # the second's work, a burst now and then
                fs.update(cam, ts, [{"box": [0, 0, 1, 1]}])
        lock.finish(cam)

    def run():
        rec = Rec()
        fs = mct.FusedSeconds(rec, ["a", "b", "c"], wait_s=0.02)
        lock = pipeline.Lockstep({"a": 100.0, "b": 100.0, "c": 108.0})
        fs.files(lock, stuck_s=20.0)
        th = [threading.Thread(target=play, args=("a", fs, lock, 100.0, 12, {103.0: 0.3})),
              threading.Thread(target=play, args=("b", fs, lock, 100.0, 9, {105.0: 0.2})),     # ends at 109
              threading.Thread(target=play, args=("c", fs, lock, 108.0, 4, {}))]               # begins at 108
        for t in th:
            t.start()
        for t in th:
            t.join(timeout=30)
        assert not any(t.is_alive() for t in th)
        return rec.calls
    calls = run()
    assert len(calls) == 12                                # each second once, every camera of it together
    got = dict(calls)
    assert all(got[float(t)] == ("a", "b") for t in range(100, 108))
    assert got[108.0] == ("a", "b", "c") and got[110.0] == ("a", "c")


def test_inside_low_a_piece_of_a_cow_takes_no_track():
    """[tracker] inside_low: a box inside a bigger one (a piece of a crowded cow) starts no track and takes no
    neighbour's; a cow half hidden behind another keeps the track that expects her."""
    import mct
    cfg = _cfg()
    cfg["mct"]["enabled"] = False
    cfg["tracker"]["inside_low"] = 0.8
    gt = mct.GlobalTracker(cfg)
    near, behind = [0.5, 0.3, 0.9, 0.9], [0.55, 0.35, 0.7, 0.6]           # behind: 100% inside near's box
    her = lambda x: {"box": [x, 0.33, x + 0.17, 0.6], "emb": None}         # a cow walking behind the near one
    a, b = gt.update("c1", 0.0, [{"box": near, "emb": None}, her(0.30)])
    assert a and b
    # a piece of the near cow shows up: no track of its own
    a2, b2, piece = gt.update("c1", 1.0, [{"box": near, "emb": None}, her(0.35),
                                         {"box": [0.6, 0.4, 0.75, 0.55], "emb": None}])
    assert (a2, b2) == (a, b) and piece is None
    # she walks on behind the near cow (at 0.5 wholly inside its box): her track expects her, she keeps it
    for t, x in ((2.0, 0.40), (3.0, 0.45), (4.0, 0.50)):
        got = gt.update("c1", t, [{"box": near, "emb": None}, her(x)])
        assert got == [a, b], (t, got)
    # off (0): the piece would have started a track
    cfg["tracker"]["inside_low"] = 0.0
    g0 = mct.GlobalTracker(cfg)
    g0.update("c1", 0.0, [{"box": near, "emb": None}])
    assert g0.update("c1", 1.0, [{"box": near, "emb": None}, {"box": behind, "emb": None}])[1] is not None
