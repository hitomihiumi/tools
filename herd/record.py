#!/usr/bin/env python3
"""Record the barn's cameras to files named as the dataset tools read them: every camera of the config
at once, for --minutes, into OUT/<camera>_<local start time>.mkv (Matroska: a file cut short - Ctrl+C, the
computer asleep, the network gone - still reads up to where it stopped).

    python herd/herd.py record --config barn.toml --out /srv/herd/rec --minutes 60
    python herd/herd.py record --config barn.toml --out /srv/herd/rec --minutes 60 --fps 5 [--gpu]
    python herd/herd.py record --config barn.toml --out /srv/herd/rec --minutes 30 --cameras cam1 cam2
    python herd/herd.py record --config barn.toml --out /srv/herd/rec --minutes 60 \
        --url cam1=rtsp://user:pass@10.0.0.11:554/stream1 cam2=rtsp://...     (the config's urls left as they are)

While recording, each camera's stream is copied as it comes into OUT/<name>.mkv.part (ffmpeg); then it
becomes OUT/<name>.mkv:

  --fps N   re-encoded to H.264 at N frames a second, steady: at each 1/N s the frame the camera sent last
            (a frame that did not arrive: the one before held). The tools take 1 frame a second, the
            tracker [sampling] track_fps more - 5 is plenty, and a fraction of the size and of the time to
            encode
  HEVC      (H.265, without --fps) re-encoded to H.264 at the camera's own rate (the spacing of its time
            stamps), steady: the FFmpeg inside OpenCV, which the tools read with, decodes some cameras'
            HEVC poorly (the barn's: 496 of 926 frames, where ffmpeg itself decodes them all); --no-h264
            keeps it as it came
  H.264     (without --fps) kept as it came
  --gpu     the re-encoding on the NVIDIA card (decoded with NVDEC, encoded with NVENC); the CPU when
            the card cannot (said at the start)

The length goes by the computer's clock: a live stream's frames are stamped as they arrive (a camera's own
stamps can jump ahead - the barn's did, and its recordings ended seconds in), and a stream that stops
early (the camera or the network dropped it; 10 s without data) is recorded on into a new file named by
its own start, which the tools read after the first. At the end each camera's minutes recorded are said
against the minutes asked (!! when short).

--keep-raw keeps the stream as it came beside it (OUT/<name>.mkv.raw, Matroska; not read by the tools).
After the recording each camera's share of frames that arrived is said: the barn's cameras stamp 25 a
second and ~60% arrive when all four record at once over the internet (the camera drops the rest when the
connection cannot carry them). Every file is checked at the end: how many seconds and frames it holds,
and that its last seconds read. --check DIR checks recordings already made (every frame decoded, where
the picture stands still).

Then: herd.py barn video / herd.py boxes frames --videos OUT (SETUP.md sections 11-12).
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import load_config  # noqa: E402

# the decoder's complaints while checking (a frame read without its reference after a jump) are not news
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")


RAW = (".part", ".raw")          # the stream as it came (Matroska), not a recording the tools read


def _fmt(path):
    """-f matroska for the files named .part / .raw (ffmpeg goes by the name otherwise)."""
    return ["-f", "matroska"] if path.endswith(RAW) else []


LIVE = ("rtsp://", "rtsps://", "rtmp://", "http://", "https://", "udp://", "rtp://", "srt://")


def command(url, seconds, path):
    """The stream copied as it is into path (Matroska) for `seconds`: over TCP (a connection silent for 10 s
    given up), damaged packets dropped (a camera's stream has them; HEVC 'too short RTP packet'). A live
    stream's frames are stamped with the time they arrive, not the camera's own stamps: those can jump
    ahead (the barn's did), and -t, which goes by the stamps, ended such recordings seconds in, the frame
    after the jump never written; by arrival the length is the clock's, and every camera on one clock.
    A file (or an .sdp) plays at its own pace, as a camera would send it."""
    rtsp = ["-rtsp_transport", "tcp", "-timeout", "10000000"] if url.startswith("rtsp") else []
    sdp = ["-protocol_whitelist", "file,udp,rtp"] if url.lower().endswith(".sdp") else []
    live = url.startswith(LIVE) or bool(sdp)
    clock = ["-use_wallclock_as_timestamps", "1"] if live else ["-re"]
    return ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *rtsp, *sdp, *clock,
            "-fflags", "+genpts+discardcorrupt", "-i", url, "-t", str(seconds), "-map", "0:v:0", "-c", "copy", "-an",
            "-f", "matroska", path]


def encode(src, dst, fps, crf=20, gpu=False):
    """The stream as it came -> H.264 at fps frames a second, steady (at each 1/fps s the frame sent last), in
    Matroska. gpu: decoded on the NVIDIA card (NVDEC - the heavy part: every frame of the camera's stream)
    and encoded there (NVENC, -cq at the crf's quality); else libx264 on the CPU."""
    head = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-err_detect", "ignore_err",
            "-fflags", "+genpts+discardcorrupt"]
    if gpu:
        head += ["-hwaccel", "cuda"]                     # frames come back to memory for the filters
        enc = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    else:
        enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf)]
    return head + [*_fmt(src), "-i", src, "-map", "0:v:0", "-an", "-vf", f"setpts=PTS-STARTPTS,fps={fps:g}",
                   *enc, "-pix_fmt", "yuv420p", "-f", "matroska", dst]


def nvenc_ready():
    """Whether ffmpeg encodes on the NVIDIA card here -> (yes, why not). In Docker the container needs the
    driver's video part (NVIDIA_DRIVER_CAPABILITIES with video: the image sets it; compose.yaml gives the
    card)."""
    try:
        r = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                            "testsrc=size=640x360:rate=5", "-t", "1", "-c:v", "h264_nvenc", "-f", "null", "-"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)
    if r.returncode == 0:
        return True, ""
    lines = [x.split("] ", 1)[-1] for x in (r.stderr or "").strip().splitlines()]
    return False, (lines or ["ffmpeg failed"])[0]          # the first: the cause (Cannot load libcuda.so.1, ...)


def codec(path):
    """The video stream's codec name (ffprobe), or what OpenCV's fourcc says."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", *_fmt(path), "-select_streams", "v:0", "-show_entries",
                              "stream=codec_name", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=60).stdout.strip()
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


RATES = (1, 2, 5, 6, 8, 10, 12, 12.5, 15, 20, 24, 25, 30, 50, 60)      # what cameras send


def delivered(path):
    """The stream as it came (the recording before any re-encoding, its frames stamped as they arrived):
    -> (the camera's frames a second, the share of them that arrived, the gaps of 0.5 s or more, the
    longest gap) or None. A camera sends frames 1/rate apart; a gap is frames that never arrived (a
    connection too slow for them - the camera drops them)."""
    try:
        out = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", *_fmt(path), "-i", path, "-map", "0:v:0", "-c",
                              "copy", "-f", "framecrc", "-"], capture_output=True, text=True, timeout=600).stdout
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
    step = float(np.median(d))                       # the camera's frame time, as they arrive
    rate = 1 / step
    near = min(RATES, key=lambda r: abs(np.log(r * step)))
    rate = float(near) if abs(np.log(near * step)) < 0.25 else rate
    span = t[-1] - t[0] + 1 / rate
    return rate, min(1.0, len(t) / (span * rate)), int((d >= 0.5).sum()), float(d.max())


def real_rate(path):
    """The camera's frame rate (the spacing of its frames): every frame that arrived keeps its place, the
    ones that did not are the previous frame held - the gaps as they were. Without stamps: frames over
    seconds."""
    got = delivered(path)
    if got:
        return float(max(1, round(got[0])))
    try:
        out = subprocess.run(["ffprobe", "-v", "error", *_fmt(path), "-select_streams", "v:0", "-count_packets",
                              "-show_entries", "stream=nb_read_packets:format=duration", "-of", "json", path],
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
    size = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
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
    text = (f"{secs / 60:.1f} min, {size}, {info.get('frames', 0)} frames ({fps:.1f} a second), "
            f"{info.get('codec', '?')}; "
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


def keep_recording(cam, url, out, tz, end, stop, segs, lock):
    """One camera until `end` (the computer's clock): ffmpeg into <cam>_<start>.mkv.part; a stream that stops
    before then (the camera or the network dropped it, 10 s without data) is recorded again into a new file
    - the tools read a camera's files one after the other. ffmpeg still running at the end (+15 s) or on
    Ctrl+C is stopped as by Ctrl+C: its file is closed properly. Three starts in a row that record nothing:
    the camera given up."""
    import math
    failed = 0
    while not stop.is_set():
        left = end - time.time()
        if left < 2:
            return
        base = os.path.join(out, f"{cam}_{dt.datetime.now(tz):%Y-%m-%d_%H-%M-%S}")
        part = base + ".mkv.part"
        pr = subprocess.Popen(command(url, int(math.ceil(left)), part))
        t0 = time.time()
        with lock:
            segs.append((cam, base, pr))
        while pr.poll() is None:
            if stop.wait(0.5) or time.time() > end + 15:
                pr.send_signal(signal.SIGINT)
                try:
                    pr.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pr.kill()
                    pr.wait()
                break
        if stop.is_set() or end - time.time() < 2:
            return
        size = os.path.getsize(part) if os.path.exists(part) else 0
        failed = 0 if size else failed + 1
        if failed >= 3:
            print(f"[record] {cam}: 3 starts in a row recorded nothing (ffmpeg exit {pr.returncode}) - given up; "
                  "check its url", flush=True)
            return
        print(f"[record] {cam}: the stream stopped after {time.time() - t0:.0f} s (ffmpeg exit {pr.returncode}) - "
              "recording on into a new file", flush=True)
        stop.wait(2)


def finish(cam, base, pr, args):
    """A camera's .mkv.part -> <name>.mkv (re-encoded at --fps, or HEVC to H.264, or kept as it came) ->
    its length in s, or None when nothing was recorded or it does not read."""
    part, final, raw = base + ".mkv.part", base + ".mkv", base + ".mkv.raw"
    size = os.path.getsize(part) / 2 ** 20 if os.path.exists(part) else 0
    if not size:
        if os.path.exists(part):
            os.remove(part)
        print(f"[record] {cam}: nothing recorded into {os.path.basename(final)}  !! ffmpeg exit {pr.returncode}",
              flush=True)
        return None
    got = delivered(part)
    if got:
        rate, share, gaps, longest = got
        print(f"[record] {cam}: the camera sends {rate:g} frames a second; {share:.0%} of them arrived "
              f"({gaps} gaps of 0.5 s or more, the longest {longest:.1f} s)"
              + ("  !! frames lost on the way: the connection to the cameras is too slow for them all - "
                 "try one camera alone (--cameras cam1), a cable, or the cameras' second (lower) stream"
                 if share < 0.9 else ""), flush=True)
        if args.fps and args.fps > rate * share + 0.5:
            print(f"[record] {cam}: --fps {args.fps:g} is more than the ~{rate * share:.0f} frames a second that "
                  "arrived: many frames are the one before held", flush=True)
    hevc = codec(part) == "hevc"
    if args.fps or (hevc and not args.no_h264):
        fps = args.fps or real_rate(part)
        print(f"[record] {cam}: {'HEVC' if hevc else 'H.264'} -> H.264 at {fps:g} frames a second"
              f"{' on the NVIDIA card' if args.gpu else ''} ...", flush=True)
        tmp = base + ".mkv.encoding"
        r = subprocess.run(encode(part, tmp, fps, args.crf, gpu=args.gpu), capture_output=True, text=True)
        made = r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0
        if not made and args.gpu:                         # the card refused this one: the CPU does it
            print(f"[record] {cam}: on the NVIDIA card it failed ({(r.stderr or '').strip().splitlines()[-1:]}) - "
                  "on the CPU instead ...", flush=True)
            r = subprocess.run(encode(part, tmp, fps, args.crf), capture_output=True, text=True)
            made = r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0
        if made:
            os.replace(tmp, final)
            if args.keep_raw:
                os.replace(part, raw)
            else:
                os.remove(part)
        else:
            if os.path.exists(tmp):
                os.remove(tmp)
            print(f"[record] {cam}: re-encoding failed ({(r.stderr or '').strip().splitlines()[-1:]}) - the stream "
                  "kept as it came", flush=True)
            os.replace(part, final)
    else:
        os.replace(part, final)                           # the stream as it came
    text, ok = check(final)
    print(f"[record] {cam}: {final} {os.path.getsize(final) / 2 ** 20:.0f} MB, {text}", flush=True)
    return length(final) if ok else None


def length(path):
    """The file's length in s (ffprobe)."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=120).stdout.strip()
        return float(out.splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return 0.0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--minutes", type=float, default=60.0)
    p.add_argument("--cameras", nargs="*", default=None, help="only these camera ids")
    p.add_argument("--url", nargs="*", default=[], metavar="CAM=URL",
                   help="the stream of a camera, in place of the config's url (and a camera the config lacks)")
    p.add_argument("--fps", type=float, default=None, help="the frames a second written: re-encoded to H.264 at "
                   "this rate, steady (default: the camera's own, the stream kept as it came unless HEVC)")
    p.add_argument("--keep-raw", "--keep-mkv", dest="keep_raw", action="store_true",
                   help="keep the stream as it came beside the recording (<name>.mkv.raw)")
    p.add_argument("--no-h264", action="store_true", help="an HEVC stream kept as it came (no re-encoding)")
    p.add_argument("--crf", type=int, default=20, help="H.264 quality when re-encoding (lower = better, bigger)")
    p.add_argument("--gpu", action="store_true", help="re-encode on the NVIDIA card (NVDEC + NVENC: many times "
                   "faster, the CPU left free); the CPU when the card cannot")
    p.add_argument("--check", default=None, metavar="DIR", help="only check the recordings in DIR")
    args = p.parse_args(argv)
    if args.fps is not None and not 0 < args.fps <= 60:
        p.error("--fps: between 0 and 60 frames a second")
    if args.fps and args.no_h264:
        p.error("--fps re-encodes (to H.264): not with --no-h264")
    if args.check:
        from common import VIDEO_EXT
        files = sorted(f for f in os.listdir(args.check) if f.lower().endswith(VIDEO_EXT + RAW))
        bad = 0
        for f in files:
            if f.endswith(RAW):                                # as it came from the camera
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
    if args.gpu:                                       # said now, not after an hour of recording
        ok, why = nvenc_ready()
        if not ok:
            print(f"[record] --gpu: ffmpeg cannot encode on the NVIDIA card here ({why}) - re-encoding on the CPU. "
                  "In Docker: a current NVIDIA driver, the image built again (NVIDIA_DRIVER_CAPABILITIES with "
                  "video), compose.yaml's GPU", flush=True)
            args.gpu = False
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
    end = time.time() + seconds
    stop, lock, segs = threading.Event(), threading.Lock(), []
    threads = [threading.Thread(target=keep_recording, args=(c["id"], c["url"], args.out, tz, end, stop, segs, lock),
                                daemon=True) for c in cams]
    print(f"[record] {len(cams)} cameras for {args.minutes:g} min from {start:%Y-%m-%d %H:%M:%S} -> {args.out}"
          + (f", written at {args.fps:g} frames a second" if args.fps else ""), flush=True)
    t0 = time.time()
    for th in threads:
        th.start()
    try:
        while any(th.is_alive() for th in threads):
            for th in threads:
                th.join(timeout=1)
    except KeyboardInterrupt:                        # stopped early: what is recorded is kept (Matroska reads)
        print("[record] stopped - keeping what was recorded", flush=True)
        stop.set()
        for th in threads:
            th.join()
    bad = 0
    got = {c["id"]: 0.0 for c in cams}
    for cam, base, pr in sorted(segs, key=lambda x: (x[0], x[1])):
        secs = finish(cam, base, pr, args)
        bad += secs is None
        got[cam] += secs or 0.0
    want = min(seconds, time.time() - t0) if stop.is_set() else seconds
    for cam, secs in got.items():
        n = sum(1 for c, _, _ in segs if c == cam)
        short = secs < 0.95 * want
        bad += short
        print(f"[record] {cam}: {secs / 60:.1f} of {want / 60:.1f} min recorded" + (f" in {n} files" if n > 1 else "")
              + ("  !! short - the stream stopped or did not start (see above)" if short else ""), flush=True)
    print(f"[record] done in {(time.time() - t0) / 60:.1f} min")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
