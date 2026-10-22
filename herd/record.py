#!/usr/bin/env python3
"""Record the barn's cameras to files named as the dataset tools read them: every camera of the config
at once, for --minutes, into OUT/<camera>_<local start time>.mp4 (ffmpeg, the stream copied as it is).

    python herd/herd.py record --config barn.toml --out /srv/herd/rec --minutes 60
    python herd/herd.py record --config barn.toml --out /srv/herd/rec --minutes 30 --cameras cam1 cam2
    python herd/herd.py record --config barn.toml --out /srv/herd/rec --minutes 60 \
        --url cam1=rtsp://user:pass@10.0.0.11:554/stream1 cam2=rtsp://...     (the config's urls left as they are)

Each camera is written to a Matroska file first (.mkv: one cut short - Ctrl+C, the computer asleep, the
network gone - still reads up to where it stopped), then made a plain .mp4 that every player plays (the
.mkv is kept when that fails, or with --keep-mkv). An HEVC (H.265) stream is re-encoded to H.264 on the
way: the FFmpeg inside OpenCV, which the tools read with, decodes some cameras' HEVC poorly (the barn's:
496 of 926 frames, where ffmpeg itself decodes them all); --no-h264 copies it as it is. The rate written
is the camera's own (the spacing of its time stamps), steady, as the tools expect: every frame that arrived
keeps its place, a frame that did not is the one before held. The barn's cameras stamp 25 a second and
~60% arrive when all four record at once: the camera drops the rest when the connection cannot carry them
(said after the recording: what share arrived, the gaps). --check says where the picture stands still. Every file is checked at the end: how many seconds
and frames it holds, and that its last seconds read. --check DIR checks recordings already made.

Then: herd.py barn video / herd.py boxes frames --videos OUT (SETUP.md sections 11-12).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import load_config  # noqa: E402

# the decoder's complaints while checking (a frame read without its reference after a jump) are not news
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")


def command(url, seconds, path):
    """The stream copied as it is into path (.mkv): over TCP, time stamps made where the camera's are
    missing, damaged packets dropped (a camera's stream has them; HEVC 'too short RTP packet')."""
    rtsp = ["-rtsp_transport", "tcp"] if url.startswith("rtsp") else []
    return ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *rtsp, "-fflags", "+genpts+discardcorrupt",
            "-i", url, "-t", str(seconds), "-map", "0:v:0", "-c", "copy", "-an", path]


def codec(path):
    """The video stream's codec name (ffprobe), or what OpenCV's fourcc says."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                              "-of", "csv=p=0", path], capture_output=True, text=True, timeout=60).stdout.strip()
        if out:
            return out.splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    import cv2
    cap = cv2.VideoCapture(path)
    cc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    cap.release()
    four = "".join(chr((cc >> 8 * i) & 0xFF) for i in range(4)).lower()
    return "hevc" if four in ("hev1", "hvc1", "hevc", "h265") else four


def delivered(path):
    """What the camera's own clock says of the stream as it came (the recording before any re-encoding):
    -> (its frames a second, the share of them that arrived, the gaps of 0.5 s or more, the longest gap)
    or None. Inside the stream a camera stamps frames 1/rate apart; a jump in the stamps is frames that
    never arrived (a connection too slow for it - the camera drops them)."""
    try:
        out = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-map", "0:v:0", "-c", "copy",
                              "-f", "framecrc", "-"], capture_output=True, text=True, timeout=600).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    tb, pts = None, []
    for line in out.splitlines():
        if line.startswith("#tb"):
            tb = line.split(":", 1)[1].strip()
        elif line and not line.startswith("#"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) > 2:
                pts.append(int(parts[2]))
    if not tb or len(pts) < 10:
        return None
    num, den = (int(x) for x in tb.split("/"))
    import numpy as np
    t = np.sort(np.array(pts, float) * num / den)
    d = np.diff(t)
    d = d[d > 0]
    if not len(d):
        return None
    step = float(np.median(d))                       # the camera's own frame time
    span = t[-1] - t[0] + step
    rate = 1 / step
    return rate, min(1.0, len(t) / (span * rate)), int((d >= 0.5).sum()), float(d.max())


def real_rate(path):
    """The camera's own frame rate (the spacing of its time stamps): every frame that arrived keeps its
    place, the ones that did not are the previous frame held - the gaps as they were. Without stamps:
    frames over seconds."""
    got = delivered(path)
    if got:
        return float(max(1, round(got[0])))
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                              "stream=nb_read_packets:format=duration", "-of", "json", path],
                             capture_output=True, text=True, timeout=600).stdout
        import json
        j = json.loads(out or "{}")
        n = int(((j.get("streams") or [{}])[0]).get("nb_read_packets") or 0)
        secs = float((j.get("format") or {}).get("duration") or 0)
        if n and secs > 1 and 1 <= n / secs <= 60:
            return float(max(1, round(n / secs)))
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    import fix_videos
    return fix_videos.stated_fps(path) or 25.0


def to_h264(src, dst, crf=20):
    """The recording re-encoded to H.264 at the rate the camera really sent, steady (what the tools expect)."""
    import fix_videos
    return fix_videos.command(src, dst, real_rate(src), crf=crf)


def to_mp4(src, dst):
    """The finished .mkv copied into a plain .mp4 (index at the front: every player plays it)."""
    r = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src, "-map", "0:v:0", "-c", "copy",
                        "-movflags", "+faststart", dst])
    return r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0


def decoded_share(path):
    """Every frame read in order, as the tools read a recording -> (frames that decode, reads that failed,
    the picture's stalls in s, repeated frames). A stream that lost packets on the way has frames without
    their slices or references; a stretch where the picture stands still - frames between their times, or
    repeated frames filling a steady rate - is a gap in what the camera sent."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(path)
    good = bad = run = same = 0
    last_t, prev, stalls, still_from = None, None, [], None
    while run < 200:
        ok, f = cap.read()
        if not ok:
            bad, run = bad + 1, run + 1
            continue
        good, run = good + 1, 0
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        small = cv2.resize(f, (160, 90), interpolation=cv2.INTER_AREA).astype(np.int16)
        repeat = prev is not None and float(np.abs(small - prev).mean()) < 0.08     # a copy, not a still scene
        if repeat:
            same += 1
            still_from = last_t if still_from is None else still_from
        else:
            start = still_from if still_from is not None else last_t
            if start is not None and t > start:
                stalls.append(t - start)                 # how long the picture stood (one frame's time if not)
            still_from = None
        last_t, prev = t, small
    cap.release()
    bad -= run                                       # the reads past the end
    return good, max(0, bad), stalls, same


def check(path, full=False):
    """How long the file is, how many frames it holds, and whether its last seconds read -> (text, ok).
    full: every frame decoded in order too (minutes for an hour of HEVC)."""
    import cv2
    info = {}
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                              "stream=codec_name,nb_read_packets,r_frame_rate:format=duration", "-of", "json", path],
                             capture_output=True, text=True, timeout=600).stdout
        import json
        j = json.loads(out or "{}")
        st = (j.get("streams") or [{}])[0]
        info = {"codec": st.get("codec_name", "?"), "frames": int(st.get("nb_read_packets") or 0),
                "seconds": float((j.get("format") or {}).get("duration") or 0)}
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    cap = cv2.VideoCapture(path)
    if not info.get("frames"):                       # no ffprobe: what OpenCV says
        n, rate = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0, cap.get(cv2.CAP_PROP_FPS) or 0
        if n > 0 and 1 <= rate <= 120:
            info = {"codec": info.get("codec", "?"), "frames": int(n), "seconds": n / rate}
    ok_start = cap.read()[0]
    secs = info.get("seconds") or 0
    ok_end = False
    if ok_start and secs > 3:
        # the last 20 s read in order: a camera's key frames may be seconds apart, and frames read right
        # after a jump have nothing to decode from - so not a jump to the last second
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, secs - 20) * 1000)
        last, fails = 0.0, 0
        while fails < 200:
            if cap.read()[0]:
                fails = 0
                last = max(last, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            else:
                fails += 1
        ok_end = last >= secs - 3
    cap.release()
    fps = info.get("frames", 0) / secs if secs else 0
    ok = ok_start and (ok_end or secs <= 3) and info.get("frames", 0) > 0
    text = (f"{secs / 60:.1f} min, {info.get('frames', 0)} frames ({fps:.1f} a second), {info.get('codec', '?')}; "
            + ("reads to the end" if ok else "!! " + ("does not read at all" if not ok_start else
                                                    "its end does not read")))
    if full and ok_start:
        good, bad, stalls, same = decoded_share(path)
        # the frames the file holds, not the reads: OpenCV skips a frame it cannot decode without saying so
        share = good / max(1, info.get("frames", 0), good + bad)
        new = good - same
        text += f"; {good} of them decode ({share:.0%}), {new} different ({new / max(1, secs):.1f} a second)"
        if stalls:
            g = sorted(stalls)
            long_ = [x for x in g if x >= 0.5]
            text += (f"; the picture stands still {len(long_)} times for 0.5 s or more "
                     f"({sum(long_):.0f} s in all, the longest {g[-1]:.1f} s) - gaps in what the camera sent")
        if share < 0.95:
            text += " !! frames that do not decode"
            ok = False
    return text, ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--minutes", type=float, default=60.0)
    p.add_argument("--cameras", nargs="*", default=None, help="only these camera ids")
    p.add_argument("--url", nargs="*", default=[], metavar="CAM=URL",
                   help="the stream of a camera, in place of the config's url (and a camera the config lacks)")
    p.add_argument("--keep-mkv", action="store_true", help="keep the .mkv beside the .mp4")
    p.add_argument("--no-h264", action="store_true", help="an HEVC stream copied as it is (no re-encoding)")
    p.add_argument("--crf", type=int, default=20, help="H.264 quality when re-encoding (lower = better, bigger)")
    p.add_argument("--check", default=None, metavar="DIR", help="only check the recordings in DIR")
    args = p.parse_args(argv)
    if args.check:
        from common import VIDEO_EXT
        files = sorted(f for f in os.listdir(args.check) if f.lower().endswith(VIDEO_EXT))
        bad = 0
        for f in files:
            if f.lower().endswith(".mkv"):                     # as it came from the camera
                got = delivered(os.path.join(args.check, f))
                if got:
                    print(f"[record] {f}: the camera sends {got[0]:.0f} frames a second; {got[1]:.0%} of them "
                          f"arrived ({got[2]} gaps of 0.5 s or more, the longest {got[3]:.1f} s)", flush=True)
                continue
            text, ok = check(os.path.join(args.check, f), full=True)
            bad += not ok
            print(f"[record] {f}: {text}", flush=True)
        if bad:
            print("[record] the ones marked !! : herd.py fix-videos makes them readable", flush=True)
        return 1 if bad else 0
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is not installed (the Docker image has it; on a server: apt install ffmpeg)")
    cfg = load_config(args.config)
    urls = {}
    for u in args.url:
        cam, sep, url = u.partition("=")
        if not sep or not url:
            sys.exit(f"--url {u}: CAM=URL, e.g. cam1=rtsp://user:pass@10.0.0.11:554/stream1")
        urls[cam] = url
    known = {c["id"] for c in cfg["cameras"]}
    cams = [dict(c, url=urls.get(c["id"], c.get("url", ""))) for c in cfg["cameras"]]
    cams += [{"id": c, "url": u} for c, u in urls.items() if c not in known]
    if urls and not args.cameras:                          # the streams named: those cameras
        cams = [c for c in cams if c["id"] in urls]
    cams = [c for c in cams if not args.cameras or c["id"] in args.cameras]
    if not cams:
        sys.exit(f"no camera {args.cameras} in {args.config}")
    empty = [c["id"] for c in cams if not c.get("url")]
    if empty:
        sys.exit(f"{', '.join(empty)}: no url (in the config, or --url {empty[0]}=rtsp://...)")
    os.makedirs(args.out, exist_ok=True)
    tz = dt.timezone(dt.timedelta(hours=cfg["farm"]["timezone_offset_hours"]))
    start = dt.datetime.now(tz)
    seconds = int(args.minutes * 60)
    procs = []
    for c in cams:
        path = os.path.join(args.out, f"{c['id']}_{start:%Y-%m-%d_%H-%M-%S}.mkv")
        procs.append((c["id"], path, subprocess.Popen(command(c["url"], seconds, path))))
    print(f"[record] {len(procs)} cameras for {args.minutes:g} min from {start:%Y-%m-%d %H:%M:%S} -> {args.out}",
          flush=True)
    t0 = time.time()
    bad = 0
    try:
        for _, _, pr in procs:
            pr.wait()
    except KeyboardInterrupt:                        # stopped early: what is recorded is kept (the .mkv reads)
        print("[record] stopped - keeping what was recorded", flush=True)
        for _, _, pr in procs:
            pr.wait()
    for cam, path, pr in procs:
        size = os.path.getsize(path) / 2 ** 20 if os.path.exists(path) else 0
        if not size:
            bad += 1
            print(f"[record] {cam}: nothing recorded  !! ffmpeg exit {pr.returncode} - check its url", flush=True)
            continue
        got = delivered(path)
        if got:
            rate, share, gaps, longest = got
            print(f"[record] {cam}: the camera sends {rate:.0f} frames a second; {share:.0%} of them arrived "
                  f"({gaps} gaps of 0.5 s or more, the longest {longest:.1f} s)"
                  + ("  !! frames lost on the way: the connection to the cameras is too slow for them all - "
                     "try one camera alone (--cameras cam1), a cable, or the cameras' second (lower) stream"
                     if share < 0.9 else ""), flush=True)
        final = path
        mp4 = path[:-4] + ".mp4"
        if not args.no_h264 and codec(path) == "hevc":
            print(f"[record] {cam}: HEVC -> H.264 (what the tools read well) ...", flush=True)
            made = subprocess.run(to_h264(path, mp4, args.crf), capture_output=True).returncode == 0 \
                and os.path.exists(mp4) and os.path.getsize(mp4) > 0
        else:
            made = to_mp4(path, mp4)
        if made:
            final = mp4
            if not args.keep_mkv:
                os.remove(path)
        text, ok = check(final)
        bad += not ok
        print(f"[record] {cam}: {final} {os.path.getsize(final) / 2 ** 20:.0f} MB, {text}", flush=True)
    print(f"[record] done in {(time.time() - t0) / 60:.1f} min")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
