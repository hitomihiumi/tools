"""studio.py: barn.toml edited in place, the fit and the overlay behind the page, the server."""
import _paths  # noqa: F401
import json
import os
import tempfile
import threading
import urllib.request

import numpy as np

CFG = '''[farm]
timezone_offset_hours = 3

[mct]
enabled = false   # off for now
disjoint = [["cam1", "cam2"], ["cam1", "cam3"]]

[[cameras]]
id = "cam1"
url = "rtsp://x"
mask = [[0, 0], [0.6, 0], [0.6, 1], [0, 1]]
floor_points = [
  [0.1, 0.5, 0, 0], [0.9, 0.5, 800, 0],
  [0.9, 0.9, 800, 600], [0.1, 0.9, 0, 600],
]
distortion = -0.2
min_box_pct = 0.5

[[cameras]]
id = "cam2"
url = "rtsp://y"
'''


def test_set_keys_replaces_multiline_arrays_and_keeps_the_rest():
    import studio
    block = CFG.split("[[cameras]]")[1]
    out = studio.set_keys(block, {"floor_points": None, "distortion": None, "mask": [[0, 0], [1, 0], [1, 1]],
                                  "exclude": [[0, 0], [0.1, 0], [0.1, 0.1]]})
    assert "floor_points" not in out and "distortion" not in out and "800" not in out
    assert 'url = "rtsp://x"' in out and "min_box_pct = 0.5" in out
    assert "mask = [[0, 0], [1, 0], [1, 1]]" in out and "exclude = [[0, 0], [0.1, 0], [0.1, 0.1]]" in out


def test_save_config_zones_floors_and_mct():
    import studio
    from common import load_config
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "barn.toml")
        open(path, "w").write(CFG)
        fp = [[0.1, 0.2, 10.0, 20.0], [0.9, 0.2, 810.0, 20.0], [0.9, 0.9, 810.0, 620.0], [0.1, 0.9, 10.0, 620.0]]
        studio.save_config(path, {"cam2": {"mask": [[[0, 0], [1, 0], [1, 1]]], "exclude": [], "tracking_mask": []}},
                           {"cam1": None, "cam2": {"floor_points": fp, "distortion": -0.3, "aspect": 1.7778}},
                           link_dist=90, shared=[("cam1", "cam2")])
        cfg = load_config(path)
        c1, c2 = cfg["cameras"]
        assert not c1.get("floor_points") and not c1.get("distortion") and c1["min_box_pct"] == 0.5
        assert c1["mask"] == [[0, 0], [0.6, 0], [0.6, 1], [0, 1]] and c1["url"] == "rtsp://x"
        assert c2["floor_points"] == fp and c2["distortion"] == -0.3 and c2["mask"] == [[0, 0], [1, 0], [1, 1]]
        assert cfg["mct"]["enabled"] and cfg["mct"]["mode"] == "link" and cfg["mct"]["link_dist"] == 90
        assert cfg["mct"]["disjoint"] == [["cam1", "cam3"]]
        assert open(path + ".bak").read() == CFG


def _scene(d):
    """Two cameras over one floor (a pattern on it), their pictures, and spots clicked on both."""
    import cv2
    rng = np.random.default_rng(1)
    floor = cv2.resize(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8), (800, 600), interpolation=cv2.INTER_NEAREST)
    # floor cm -> picture px for each camera: a perspective view, the second turned and shifted
    Ha = np.array([[1.2, 0.3, 100], [0.0, 0.9, 250], [0.0, 0.0006, 1.0]])
    Hb = np.array([[-1.1, 0.2, 1100], [0.1, -0.8, 650], [0.0, -0.0004, 1.0]])
    pics, spots = {}, []
    for cam, H in (("cam1", Ha), ("cam2", Hb)):
        img = cv2.warpPerspective(floor, H, (1280, 720))
        cv2.imwrite(os.path.join(d, f"{cam}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        pics[cam] = H
    for X, Y in ((100, 100), (700, 120), (650, 500), (150, 480), (400, 300), (300, 200), (550, 380), (420, 520)):
        row = []
        for H in (Ha, Hb):
            q = H @ [X, Y, 1.0]
            row += [q[0] / q[2] / 1280, q[1] / q[2] / 720]
        spots.append(row)
    open(os.path.join(d, "barn.toml"), "w").write(CFG.replace('''floor_points = [
  [0.1, 0.5, 0, 0], [0.9, 0.5, 800, 0],
  [0.9, 0.9, 800, 600], [0.1, 0.9, 0, 600],
]
distortion = -0.2
''', ""))
    os.makedirs(os.path.join(d, "calib"))
    json.dump({"pairs": {"cam1:cam2": spots}}, open(os.path.join(d, "calib", "points.json"), "w"))
    return spots


def _studio(d):
    import argparse
    import studio
    args = argparse.Namespace(config=os.path.join(d, "barn.toml"), videos=None, calib=None, at=0.0, detector="none",
                              image=[f"cam1={os.path.join(d, 'cam1.jpg')}", f"cam2={os.path.join(d, 'cam2.jpg')}"])
    return studio.Studio(args)


def test_fit_and_overlay_line_the_floor_up():
    with tempfile.TemporaryDirectory() as d:
        spots = _scene(d)
        st = _studio(d)
        state = st.state()
        assert [c["still"] for c in state["cams"]] == [True, True]
        # autocalib's pairs -> spots, one per row, on both cameras
        assert [s["at"] for s in state["spots"]] == [{"cam1": r[:2], "cam2": r[2:]} for r in spots]
        r = st.fit({"spots": state["spots"], "floor": {}, "changed": ["cam1"]})
        assert r["pairs"]["cam1:cam2"]["ok"] and r["pairs"]["cam1:cam2"]["median"] < 2
        assert set(r["floor"]) == {"cam1", "cam2"} and r["link_dist"] > 0
        assert all(max(e.values()) < 3 for e in r["errors"].values()) and len(r["errors"]) == len(spots)
        o = st.overlay({"base": "cam1", "top": "cam2", "floor": r["floor"],
                        "top_zones": {"mask": [[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8]]]}})
        assert o["share"] > 0.2 and len(o["zones"]["mask"]) == 1 and o["png"]
        p = st.plan({"floor": r["floor"], "focus": "cam1"})
        assert p["group"] == ["cam1", "cam2"] and set(p["cams"]) == {"cam1", "cam2"} and p["w"] >= 200
        # the warped picture sits on this one where both see the floor
        import base64
        import cv2
        top = cv2.imdecode(np.frombuffer(base64.b64decode(o["png"]), np.uint8), cv2.IMREAD_UNCHANGED)
        base = st.frames["cam1"]
        ok = top[..., 3] > 0
        ok[:, :4] = ok[:, -4:] = False
        assert np.abs(top[..., :3][ok].astype(int) - base[ok]).mean() < 25
        # too few spots: no floor, said so
        r = st.fit({"spots": state["spots"][:4], "floor": {}})
        assert not r["pairs"]["cam1:cam2"]["ok"] and not r["floor"] and "5 at least" in r["pairs"]["cam1:cam2"]["notes"][0]


def test_server_serves_the_page_and_saves():
    import studio
    with tempfile.TemporaryDirectory() as d:
        spots = _scene(d)
        st = _studio(d)
        srv = studio.http.server.ThreadingHTTPServer(("127.0.0.1", 0), None)
        port = srv.server_address[1]
        srv.server_close()
        th = threading.Thread(target=studio.serve, args=(st, "127.0.0.1", port), daemon=True)
        th.start()
        base = f"http://127.0.0.1:{port}"
        for _ in range(50):
            try:
                page = urllib.request.urlopen(base + "/").read().decode()
                break
            except OSError:
                import time
                time.sleep(0.1)
        assert "Camera studio" in page and "function homography" in page
        assert urllib.request.urlopen(base + "/still/cam1.jpg").read()[:2] == b"\xff\xd8"

        def post(path, body):
            req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req))
        sp = json.load(urllib.request.urlopen(base + "/api/state"))["spots"]
        sp[0]["at"]["cam3"] = [0.5, 0.5]                     # a spot on a third camera (not in the config: ignored)
        r = post("/api/fit", {"spots": sp, "floor": {}})
        s = post("/api/save", {"cams": {"cam2": {"exclude": [[[0, 0], [0.2, 0], [0.2, 0.2]]]}}, "floor": r["floor"],
                               "spots": sp[:-1], "link_dist": r["link_dist"], "calibrated_pairs": ["cam1:cam2"]})
        cams = {c["id"]: c for c in s["state"]["cams"]}
        assert cams["cam1"]["calib"] and cams["cam2"]["calib"] and cams["cam2"]["exclude"] == [[[0, 0], [0.2, 0], [0.2, 0.2]]]
        assert len(s["state"]["spots"]) == len(spots) - 1 and s["state"]["spots"][0]["at"]["cam3"] == [0.5, 0.5]
        saved = json.load(open(os.path.join(d, "calib", "points.json")))
        assert len(saved["pairs"]["cam1:cam2"]) == len(spots) - 1             # what autocalib --points reads


def _floor_scene(d):
    """The two cameras of _scene, the floor 800 x 600 with its 4 corners (cam2 sees 3 of them) and spots."""
    _scene(d)
    Ha = np.array([[1.2, 0.3, 100], [0.0, 0.9, 250], [0.0, 0.0006, 1.0]])
    Hb = np.array([[-1.1, 0.2, 1100], [0.1, -0.8, 650], [0.0, -0.0004, 1.0]])
    seen = lambda H, X, Y: (lambda q: (q[0] / q[2] / 1280, q[1] / q[2] / 720))(H @ [X, Y, 1.0])
    spots = []
    for k, (X, Y) in enumerate(((0, 0), (800, 0), (800, 600), (0, 600)), 1):
        at = {}
        for cam, H in (("cam1", Ha), ("cam2", Hb)):
            u, v = seen(H, X, Y)
            if 0 <= u <= 1 and 0 <= v <= 1:
                at[cam] = [u, v]
        spots.append({"id": k, "corner": k, "xy": [X, Y], "at": at})
    for k, (X, Y) in enumerate(((300, 200), (550, 380), (420, 520), (150, 300)), 5):
        spots.append({"id": k, "at": {c: list(seen(H, X, Y)) for c, H in (("cam1", Ha), ("cam2", Hb))}})
    return spots, Ha, Hb


def _off(calib, ar, H, pts):
    """How far (floor units) the plane puts true floor points, seen through the camera's H."""
    import autocalib
    f = autocalib.camera_floor(calib, ar)
    out = []
    for X, Y in pts:
        q = H @ [X, Y, 1.0]
        Xf, Yf, ok = f[0](np.array([q[0] / q[2] / 1280]), np.array([q[1] / q[2] / 720]))
        out.append(float(np.hypot(Xf[0] - X, Yf[0] - Y)) if ok[0] else np.inf)
    return max(out)


def test_floor_corners_measured_put_the_plan_on_the_floor():
    with tempfile.TemporaryDirectory() as d:
        spots, Ha, Hb = _floor_scene(d)
        assert len(spots[1]["at"]) == 1 and len(spots[0]["at"]) == 2      # F2 only on cam1
        st = _studio(d)
        probe = [(100, 100), (700, 500), (400, 300)]
        # one camera alone: its 4 measured corners
        r = st.fit({"spots": [s for s in spots if s.get("corner")], "floor": {}, "changed": ["cam1"]})
        assert set(r["floor"]) == {"cam1"} and r["planes"]["cam1"]["points"] == 4
        assert _off(r["floor"]["cam1"], 1280 / 720, Ha, probe) < 2
        # both: fitted together by 3 corners + 4 spots, the plan put on the 4 measured corners
        r = st.fit({"spots": spots, "floor": {}})
        assert set(r["floor"]) == {"cam1", "cam2"} and r["planes"]["cam2"]["points"] >= 4     # cam2 through cam1
        assert _off(r["floor"]["cam1"], 1280 / 720, Ha, probe) < 3 and _off(r["floor"]["cam2"], 1280 / 720, Hb, probe) < 3
        assert all(max(e.values()) < 3 for e in r["errors"].values())
        # 2 measured corners: turned, shifted and scaled only - and said so
        two = [dict(s) for s in spots]
        for s in two[2:4]:
            s.pop("xy")
        r = st.fit({"spots": two, "floor": {}})
        assert any("only turn, shift and scale" in n for n in r["notes"])
        # the corners kept through a save (with the pairs autocalib reads)
        st.save({"cams": {}, "floor": {}, "spots": spots + [{"id": 99, "corner": 5, "at": {}}]})
        back = st.state()["spots"]
        assert [s.get("corner") for s in back][:4] == [1, 2, 3, 4] and back[0]["xy"] == [0.0, 0.0]
        assert any(s["id"] == 99 and s["corner"] == 5 and not s["at"] for s in back)


def test_a_camera_that_sees_two_corners_and_the_edges_and_one_that_sees_none():
    """The barn's cameras: one sees only its pen's 2 far corners (the near ones are under it) and the two side
    edges - 6 equations, not 8 (a corner lies on its edge already): how deep the floor is stays open, said so;
    one measured point along a side (a post) closes it. Another sees no measured corner at all, only spots it
    shares with the first - placed through it; with only a thin strip of 3 spots, its plane set roughly by
    hand is refined onto the floor."""
    with tempfile.TemporaryDirectory() as d:
        _scene(d)
        st = _studio(d)
        Hc = np.array([[1.0, 0.25, 140], [0.0, -0.9, 900], [0.0, 0.0007, 1.0]])     # cam1: A3, A4 and the sides
        Hd = np.array([[-0.9, 0.1, 1000], [0.05, 0.7, 120], [0.0, 0.0004, 1.0]])    # cam2: its own view
        uv = lambda H, X, Y: (lambda q: [q[0] / q[2] / 1280, q[1] / q[2] / 720])(H @ [X, Y, 1.0])
        corners = [{"id": k, "floor": 1, "corner": k, "xy": list(xy), "at": {}}
                   for k, xy in enumerate(((0, 0), (800, 0), (800, 600), (0, 600)), 1)]
        for c in corners[2:]:
            c["at"]["cam1"] = uv(Hc, *c["xy"])
        assert all(0 < x < 1 for c in corners[2:] for x in c["at"]["cam1"])
        assert not all(0 < x < 1 for x in uv(Hc, 0, 0))                          # A1, A2: not in the picture
        edges = [{"id": 100 + i, "a": a, "b": b, "cam": "cam1", "p": uv(Hc, X, Y)}
                 for i, (a, b, X, Y) in enumerate(((4, 1, 0, 300), (4, 1, 0, 150), (2, 3, 800, 300), (2, 3, 800, 150)))]
        probe = [(100, 200), (700, 500), (400, 400)]
        r = st.fit({"spots": corners, "edges": edges, "floor": {}})
        assert not r["floor"] and any("how deep the floor is open" in n for n in r["notes"])
        post = {"id": 50, "xy": [0, 250], "at": {"cam1": uv(Hc, 0, 250)}}           # a post 3.5 m from A4 along the side
        r = st.fit({"spots": corners + [post], "edges": edges, "floor": {}})          # on its edge: 1 equation more, 7
        assert not r["floor"]
        post = [post, {"id": 51, "xy": [800, 250], "at": {"cam1": uv(Hc, 800, 250)}}]  # and one on the other side: 8
        r = st.fit({"spots": corners + post, "edges": edges, "floor": {}})
        assert set(r["floor"]) == {"cam1"} and r["planes"]["cam1"]["edge_points"] == 4
        assert _off(r["floor"]["cam1"], 1280 / 720, Hc, probe) < 2
        # one edge point less and no post: 7 equations - said, no plane
        r = st.fit({"spots": corners, "edges": edges[:3], "floor": {}})
        assert not r["floor"] and any("7 of the 8" in n for n in r["notes"])
        # cam2 sees no measured corner, only 4 spots it shares with cam1: placed through cam1
        strip = [{"id": 10 + i, "at": {"cam1": uv(Hc, X, Y), "cam2": uv(Hd, X, Y)}}
                 for i, (X, Y) in enumerate(((150, 380), (650, 380), (300, 470), (520, 450)))]
        assert all(0 < x < 1 for sp in strip for c in sp["at"].values() for x in c)
        r = st.fit({"spots": corners + post + strip, "edges": edges, "floor": {}})
        assert set(r["floor"]) == {"cam1", "cam2"} and r["planes"]["cam2"]["points"] == 4
        assert _off(r["floor"]["cam2"], 1280 / 720, Hd, [(200, 400), (600, 440)]) < 3
        # a thin strip of 3 spots only (6 equations): cam2's plane set roughly by hand (40 off), then refined
        thin = [{"id": 20 + i, "at": {"cam1": uv(Hc, X, 420), "cam2": uv(Hd, X, 420)}} for i, X in enumerate((150, 400, 650))]
        rough_pts = [[u, v] for u in (0.2, 0.5, 0.8) for v in (0.3, 0.55, 0.8)]
        Hdi = np.linalg.inv(Hd)
        rough = []
        for u, v in rough_pts:
            q = Hdi @ [u * 1280, v * 720, 1.0]
            rough.append([u, v, q[0] / q[2] + 40, q[1] / q[2] - 15])
        start = {"cam2": {"floor_points": rough, "distortion": 0.0, "aspect": 1280 / 720}}
        before = _off(start["cam2"], 1280 / 720, Hd, [(200, 420), (600, 420)])
        r = st.fit({"spots": corners + post + thin, "edges": edges, "floor": start})
        assert "cam2" in r["floor"] and any("refined together" in n for n in r["notes"])
        after = _off(r["floor"]["cam2"], 1280 / 720, Hd, [(200, 420), (600, 420)])
        assert before > 30 and after < 0.3 * before, (before, after)
        # the edges kept through a save
        st.save({"cams": {}, "floor": {}, "spots": corners + strip, "edges": edges})
        assert len(st.state()["edges"]) == 4 and st.state()["spots"][0]["floor"] == 1
