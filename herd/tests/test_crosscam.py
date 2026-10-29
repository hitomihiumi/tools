import json
import os
import random
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _scene(d, n_cows=14, n_moments=24, noise=0.004, seed=1, k2=0.0, alley=False):
    """Two cameras over one 20 x 10 m floor, facing each other; lying cows that mostly stay, a few that
    walk; your ids on both cameras' frames at the same moments. k2: cam2's lens bends (the config says 0)."""
    from autocalib import distort
    from common import homography
    rng = np.random.default_rng(seed)
    floor = [(0, 0), (2000, 0), (2000, 1000), (0, 1000)]
    Ha = homography(floor, [(0.1, 0.35), (0.9, 0.35), (1.0, 0.95), (0.0, 0.95)])       # floor -> cam1 frame
    Hb = homography(floor, [(1.0, 0.95), (0.0, 0.95), (0.1, 0.35), (0.9, 0.35)])       # cam2: the other side

    def proj(H, X, Y):
        q = H @ [X, Y, 1.0]
        return q[0] / q[2], q[1] / q[2]
    pos = rng.uniform([150, 100], [1850, 900], size=(n_cows, 2))
    frames, labels = [], {}
    for m in range(n_moments):
        ts = 1000.0 + 10 * m
        pos[: n_cows // 4] += rng.normal(0, 60, size=(n_cows // 4, 2))               # a few walk
        pos = np.clip(pos, [100, 80], [1900, 920])
        for cam, H in (("cam1", Ha), ("cam2", Hb)):
            file = f"frames/{cam}/{m}.jpg"
            boxes = []
            for i, (X, Y) in enumerate(pos):
                u, v = proj(H, X, Y + (500 if alley and cam == "cam2" and X > 1100 else 0))   # another level
                if cam == "cam2" and k2:
                    u, v = (float(x) for x in distort(u, v, k2, 1920 / 1080))
                u, v = u + rng.normal(0, noise), v + rng.normal(0, noise)
                if 0.02 < u < 0.98 and 0.05 < v < 0.99:
                    boxes.append([u - 0.05, v - 0.12, u + 0.05, v, str(i + 1)])
            frames.append({"file": file, "cam": cam, "ts": ts + (0.3 if cam == "cam2" else 0.0), "w": 1920, "h": 1080})
            labels[file] = {"status": "done", "boxes": boxes}
    os.makedirs(os.path.join(d, "boxes"))
    json.dump({"frames": frames}, open(os.path.join(d, "boxes", "frames.json"), "w"))
    json.dump({"frames": labels}, open(os.path.join(d, "labels.json"), "w"))
    inv = np.linalg.inv(Ha)
    cal = [[u, v, *proj(inv, u, v)] for u in (0.2, 0.5, 0.8) for v in (0.5, 0.7, 0.9)]
    toml = ("[mct]\nenabled = true\nlink_dist = 160\nneighbours = [[\"cam1\", \"cam2\"]]\n\n"
            "[[cameras]]\nid = \"cam1\"\nurl = \"\"\nfloor_points = " + json.dumps([[round(x, 5) for x in r] for r in cal]) +
            "\n\n[[cameras]]\nid = \"cam2\"\nurl = \"\"\n")
    open(os.path.join(d, "barn.toml"), "w").write(toml)
    return pos


def test_crosscam_learns_where_she_stands_and_puts_the_other_camera_on_the_floor():
    import crosscam
    from common import floor_mapper, load_config
    with tempfile.TemporaryDirectory() as d:
        pos = _scene(d)
        random.seed(0)
        crosscam.main(["--config", os.path.join(d, "barn.toml"), "--boxes", os.path.join(d, "boxes"),
                       "--labels", os.path.join(d, "labels.json"), "--apply"])
        r = json.load(open(os.path.join(d, "boxes", "crosscam.json")))["cam1->cam2"]
        b = r["bottom"]
        assert b["cows"] > 100 and b["right"] / b["cows"] > 0.9 and b["chance"] < 0.2, b
        assert b["median_miss"] < 0.02
        # cam2 now on cam1's floor: the last moment's cows of cam2 land where they are
        cfg = load_config(os.path.join(d, "barn.toml"))
        cam2 = next(c for c in cfg["cameras"] if c["id"] == "cam2")
        assert len(cam2["floor_points"]) >= 4 and os.path.exists(os.path.join(d, "barn.toml.bak"))
        f = floor_mapper(cam2, cfg["mct"])
        lab = json.load(open(os.path.join(d, "labels.json")))["frames"]
        last = max((k for k in lab if k.startswith("frames/cam2/")), key=lambda k: int(k.split("/")[-1][:-4]))
        err = [np.hypot(*(np.array(f(bx[:4])) - pos[int(bx[4]) - 1])) for bx in lab[last]["boxes"]]
        assert np.median(err) < 40, np.median(err)                     # cm, on a 20 x 10 m floor


def test_crosscam_finds_the_lens_and_names_swapped_between_cameras(capsys):
    """cam2's lens bends more than the config says: found from the pairs; two cows whose names are swapped
    on cam2 are said; the picture of where each was put is drawn; --apply writes the lens found."""
    import crosscam
    from common import floor_mapper, load_config
    with tempfile.TemporaryDirectory() as d:
        pos = _scene(d, k2=-0.3)
        path = os.path.join(d, "labels.json")
        lab = json.load(open(path))
        for k, v in lab["frames"].items():
            if k.startswith("frames/cam2/"):
                for b in v["boxes"]:
                    b[4] = {"3": "5", "5": "3"}.get(b[4], b[4])
        json.dump(lab, open(path, "w"))
        crosscam.main(["--config", os.path.join(d, "barn.toml"), "--boxes", os.path.join(d, "boxes"),
                       "--labels", path, "--apply"])
        out = capsys.readouterr().out
        r = json.load(open(os.path.join(d, "boxes", "crosscam.json")))["cam1->cam2"]
        b = r["bottom"]
        # a homography takes up part of a lens's bend: found near it, not the config's 0
        assert abs(b["lens"]["cam2"] + 0.3) <= 0.1 + 1e-9 and abs(b["lens"]["cam1"]) <= 0.1 + 1e-9, b["lens"]
        assert b["right"] / b["cows"] > 0.8, b
        assert "your 3 on cam2 where your 5 is" in out and "your 5 on cam2 where your 3 is" in out, out
        assert os.path.exists(os.path.join(d, "boxes", "crosscam_cam1_cam2.jpg"))
        cfg = load_config(os.path.join(d, "barn.toml"))
        cam2 = next(c for c in cfg["cameras"] if c["id"] == "cam2")
        assert abs(cam2["distortion"] + 0.3) <= 0.1 + 1e-9
        f = floor_mapper(cam2, cfg["mct"])
        last = max((k for k in lab["frames"] if k.startswith("frames/cam2/")), key=lambda k: int(k.split("/")[-1][:-4]))
        err = [np.hypot(*(np.array(f(bx[:4])) - pos[int(bx[4]) - 1])) for bx in lab["frames"][last]["boxes"]
               if bx[4] not in ("3", "5")]
        assert np.median(err) < 40, np.median(err)


def test_crosscam_two_floor_levels_want_two_planes(capsys):
    """Half the pen at another level (seen by cam2 shifted): one plane fits one half, two fit both."""
    import crosscam
    with tempfile.TemporaryDirectory() as d:
        _scene(d, n_cows=30, alley=True, seed=3)
        crosscam.main(["--config", os.path.join(d, "barn.toml"), "--boxes", os.path.join(d, "boxes"),
                       "--labels", os.path.join(d, "labels.json"), "--planes", "2", "--gate", "0.06"])
        r = json.load(open(os.path.join(d, "boxes", "crosscam.json")))["cam1->cam2"]
        one, two = r["bottom"], r["bottom_planes"]
        assert two["planes"] == 2 and two["right"] / two["cows"] > 0.9, two
        assert two["right"] > one["right"] + 0.1 * one["cows"], (one, two)
        assert "in 2 planes" in capsys.readouterr().out


def test_crosscam_learnt_mapping_holds_on_another_recording(capsys):
    """Two floor levels and a bent lens on cam2: the learnt mapping (no plane) matches the cows, fitted moment
    by moment and fitted on one recording, checked on another with the cows elsewhere."""
    import crosscam
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as d2:
        _scene(d, n_cows=24, alley=True, k2=-0.3, seed=5)
        _scene(d2, n_cows=24, alley=True, k2=-0.3, seed=9)
        crosscam.main(["--config", os.path.join(d, "barn.toml"), "--boxes", os.path.join(d, "boxes"),
                       "--labels", os.path.join(d, "labels.json"), "--planes", "1", "--gate", "0.06",
                       "--also", os.path.join(d2, "boxes"), os.path.join(d2, "labels.json")])
        r = json.load(open(os.path.join(d, "boxes", "crosscam.json")))["cam1->cam2"]
        rate = lambda y: y["right"] / y["cows"]
        assert rate(r["learnt_local"]) > 0.9 and r["learnt_local"]["right"] > r["bottom"]["right"], r["learnt_local"]
        assert rate(r["learnt"]) > rate(r["bottom"])
        for k in ("learnt_other_recording", "learnt_from_other_recording"):       # the cows elsewhere: honest
            assert r[k + "_local"]["cows"] > 100 and rate(r[k + "_local"]) > 0.7, (k, r[k + "_local"])
            assert rate(r[k + "_local"]) > rate(r[k]) and r[k + "_local"]["wrong"] / r[k + "_local"]["cows"] < 0.05
        out = capsys.readouterr().out
        assert "learnt from the boxes" in out and "checked on the other" in out


def test_crosscam_onto_a_camera_when_both_are_calibrated():
    """Both calibrated: --apply alone leaves them; --onto cam1 puts cam2 on cam1's floor, floor_at on both."""
    import crosscam
    from common import floor_mapper, load_config
    with tempfile.TemporaryDirectory() as d:
        pos = _scene(d)
        path = os.path.join(d, "barn.toml")
        text = open(path).read().rstrip("\n") + "\nfloor_points = [[0.1, 0.5, 0, 0], [0.9, 0.5, 100, 0], [0.9, 0.9, 100, 100], [0.1, 0.9, 0, 100]]\n"
        open(path, "w").write(text)
        args = ["--config", path, "--boxes", os.path.join(d, "boxes"), "--labels", os.path.join(d, "labels.json"), "--apply"]
        crosscam.main(args)
        assert not os.path.exists(path + ".bak")                         # both calibrated: untouched
        crosscam.main(args + ["--onto", "cam1"])
        best = json.load(open(os.path.join(d, "boxes", "crosscam.json")))["cam1->cam2"]["best"]
        cfg = load_config(path)
        cams = {c["id"]: c for c in cfg["cameras"]}
        assert cams["cam1"]["floor_at"] == cams["cam2"]["floor_at"] == best
        assert len(cams["cam2"]["floor_points"]) >= 4 and cams["cam2"]["floor_points"][0][2] != 0
        lab = json.load(open(os.path.join(d, "labels.json")))["frames"]
        last = max((k for k in lab if k.startswith("frames/cam2/")), key=lambda k: int(k.split("/")[-1][:-4]))
        f1, f2 = floor_mapper(cams["cam1"], cfg["mct"]), floor_mapper(cams["cam2"], cfg["mct"])
        last1 = last.replace("cam2", "cam1")
        b1 = {bx[4]: bx[:4] for bx in lab[last1]["boxes"]}
        err = [np.hypot(*(np.array(f2(bx[:4])) - np.array(f1(b1[bx[4]])))) for bx in lab[last]["boxes"] if bx[4] in b1]
        assert len(err) > 5 and np.median(err) < 40, np.median(err)    # one cow, one place on both


def test_floor_at_centre_reads_the_box_middle():
    from common import DEFAULTS, floor_mapper
    cam = {"floor_points": [[0, 0, 0, 0], [1, 0, 100, 0], [1, 1, 100, 100], [0, 1, 0, 100]]}
    assert np.allclose(floor_mapper(cam, DEFAULTS["mct"])([0.2, 0.2, 0.4, 0.6]), (30, 60))
    cam["floor_at"] = "centre"
    assert np.allclose(floor_mapper(cam, DEFAULTS["mct"])([0.2, 0.2, 0.4, 0.6]), (30, 40))


def test_crosscam_without_labels_on_both_says_so(capsys):
    import crosscam
    with tempfile.TemporaryDirectory() as d:
        _scene(d, n_moments=2)
        lab = json.load(open(os.path.join(d, "labels.json")))
        for k in lab["frames"]:
            if "cam2" in k:
                lab["frames"][k]["boxes"] = [b[:4] + ["x"] for b in lab["frames"][k]["boxes"]]
        json.dump(lab, open(os.path.join(d, "labels.json"), "w"))
        crosscam.main(["--config", os.path.join(d, "barn.toml"), "--boxes", os.path.join(d, "boxes"),
                       "--labels", os.path.join(d, "labels.json")])
        assert "too few" in capsys.readouterr().out
