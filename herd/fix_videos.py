#!/usr/bin/env python3
"""Recordings whose files are damaged (a recorder's broken metadata, time stamps, corrupt stretches) made
readable: each video encoded again by ffmpeg into a clean .mp4 - a steady frame rate, time stamps from
zero, broken frames dropped - under the same name, in another folder. They play in a player but the
tools read them wrong (one camera's clock standing still, frames missing, decoder errors).

    python herd/herd.py fix-videos --videos /srv/herd/rec --out /srv/herd/rec_fixed
    python herd/herd.py fix-videos --videos rec --out rec_fixed --fps 25 --copy    # only re-wrap, no encode
    python herd/herd.py fix-videos --videos rec --out rec_first10 --cameras cam1 cam2 --first 10
    python herd/herd.py fix-videos --videos rec --out rec_1400 --cameras cam1 cam2 --first 10 --from 14:00

--first N / --last N: only the first / last N minutes, the same stretch of time on every camera (it starts
where the camera that started last started / ends where the camera that stopped first stopped); each file
named by its new start (the time in its name moved on), so the tools put the cameras side by side as
before. --cameras: only these cameras' files.

Then use --videos /srv/herd/rec_fixed with barn video / boxes frames / zones page.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import json
import re
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import _STAMP, VIDEO_EXT  # noqa: E402


def stated_fps(path):
    """The video's frame rate as ffprobe reads it, or None when it makes no sense (a time base, 0)."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=avg_frame_rate,r_frame_rate", "-of", "json", path],
                             capture_output=True, text=True, timeout=60).stdout
        s = json.loads(out)["streams"][0]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError):
        return None
    for key in ("avg_frame_rate", "r_frame_rate"):
        num, _, den = s.get(key, "0/0").partition("/")
        try:
            fps = float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            continue
        if 1.0 <= fps <= 60.0:
            return round(fps, 3)
    return None


def command(src, dst, fps, copy=False, crf=20, cut=None):
    """cut (start s, length s) of the file's own time line, its first frame 0."""
    head = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-err_detect", "ignore_err", "-fflags", "+genpts+discardcorrupt", "-i", src, "-map", "0:v:0", "-an"]
    if copy:                                     # the stream as it is, in a new container with new time stamps
        part = ["-ss", f"{cut[0]:.3f}", "-t", f"{cut[1]:.3f}"] if cut else []   # from a keyframe on: not exact
        return head + part + ["-c:v", "copy", "-fps_mode", "passthrough", "-movflags", "+faststart", dst]
    vf = f"setpts=PTS-STARTPTS,fps={fps}"
    if cut:                                      # on the clean time line, frame-exact
        vf += f",trim=start={cut[0]:.3f}:duration={cut[1]:.3f},setpts=PTS-STARTPTS"
    return head + ["-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst]


def duration(path):
    """The file's length in s (ffprobe, else as the tools read it)."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
                             capture_output=True, text=True, timeout=120).stdout
        d = float(json.loads(out)["format"]["duration"])
        if d > 0:
            return d
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        pass
    return check(path)[1]


def stamp(name):
    """The local time in a file's name -> (datetime, the match), or (None, None)."""
    m = _STAMP.search(name)
    if not m:
        return None, None
    try:
        return dt.datetime(*map(int, m.groups())), m
    except ValueError:
        return None, None


def restamp(name, shift_s):
    """The name with its time moved on by shift_s (the same digits' layout: 2026-10-20_06-00-00)."""
    t, m = stamp(name)
    if t is None:
        return name
    new = (t + dt.timedelta(seconds=round(shift_s))).strftime("%Y-%m-%d_%H-%M-%S")
    return name[:m.start()] + new + name[m.end():]


def stretch(files, minutes, first=False, start_at=None):
    """[(src, dst, camera)] -> ([(src, dst renamed, (start, length))], the window, what each camera covers):
    the last `minutes` before the first camera stopped (first: the first after the last camera started),
    the same stretch of time for all. start_at "HH:MM[:SS]" (or "YYYY-MM-DD HH:MM[:SS]"): `minutes` from then,
    in the clock of the names (the day the recordings start on, when no date is given)."""
    spans = []
    for src, dst, cam in files:
        t, _ = stamp(os.path.basename(src))
        if t is None:
            sys.exit(f"[fix] --first / --last: no date and time in the name of {src} (e.g. cam1_2026-10-20_06-00-00.ts)")
        d = duration(src)
        spans.append((src, dst, cam, t.replace(tzinfo=dt.timezone.utc).timestamp(), d))   # the name's clock
    ends, starts = {}, {}
    for _, _, cam, t0, d in spans:
        ends[cam] = max(ends.get(cam, 0.0), t0 + d)
        starts[cam] = min(starts.get(cam, t0), t0)
    if start_at:
        day = dt.datetime.fromtimestamp(max(starts.values()), dt.timezone.utc)
        parts = start_at.strip().replace("T", " ").split()
        try:
            hms = [int(x) for x in parts[-1].replace("-", ":").split(":")] + [0, 0]
            d = dt.date.fromisoformat(parts[0]) if len(parts) > 1 else day.date()
        except ValueError:
            sys.exit(f"[fix] --from {start_at}: want HH:MM, HH:MM:SS or YYYY-MM-DD HH:MM")
        start = dt.datetime(d.year, d.month, d.day, hms[0], hms[1], hms[2], tzinfo=dt.timezone.utc).timestamp()
        end = start + minutes * 60.0
    elif first:
        start = max(starts.values())
        end = start + minutes * 60.0
    else:
        end = min(ends.values())
        start = end - minutes * 60.0
    out, cover = [], {c: 0.0 for c in ends}
    for src, dst, cam, t0, d in spans:
        a, b = max(0.0, start - t0), min(d, end - t0)
        if b - a < 1.0:
            continue
        cover[cam] += b - a
        out.append((src, os.path.join(os.path.dirname(dst), restamp(os.path.basename(dst), a)), (a, b - a)))
    return out, (start, end), cover


def camera_of(rel, cams):
    """The camera id a file's path names: the id standing alone (cam1 in cam1_2026-..., not in cam10), any
    case, and for an id ending in a number also its usual spellings (Cam_1, camera1, CAM-01)."""
    for c in sorted(cams, key=len, reverse=True):
        if re.search(rf"(?i)(?<![a-z0-9]){re.escape(c)}(?![a-z0-9])", rel):
            return c
        m = re.fullmatch(r"([A-Za-z]+)[_ -]?0*(\d+)", c)
        if m and re.search(rf"(?i)(?<![a-z0-9]){re.escape(m.group(1))}[a-z]*[_ -]?0*{m.group(2)}(?![0-9])", rel):
            return c
    return None


def check(path):
    """(frames, seconds) as the tools will read the file."""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return n, (n / fps if fps else 0.0)


def one(src, dst, fps, copy, crf, cut=None):
    t0 = time.time()
    rate = fps or stated_fps(src) or 25.0
    part = dst[:-4] + ".part.mp4"
    r = subprocess.run(command(src, part, rate, copy, crf, cut), capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(part) or os.path.getsize(part) == 0:
        if os.path.exists(part):
            os.remove(part)
        return f"!! {os.path.basename(src)}: ffmpeg failed ({(r.stderr or '').strip().splitlines()[-1:]})"
    os.replace(part, dst)
    n, secs = check(dst)
    errors = len([x for x in (r.stderr or "").splitlines() if x.strip()])
    return (f"{os.path.basename(src)} -> {os.path.basename(dst)}: {n} frames, {int(secs // 60)}:{int(secs % 60):02d} "
            f"at {rate:g} fps{' (copied)' if copy else ''}; {errors} damaged spots skipped; "
            f"{(time.time() - t0) / 60:.1f} min")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", required=True, help="the folder of recordings (sub-folders too)")
    p.add_argument("--out", required=True, help="where the clean .mp4 go (the same names and sub-folders)")
    p.add_argument("--fps", type=float, default=None, help="the frame rate to write (default: the file's, else 25)")
    p.add_argument("--copy", action="store_true", help="no encoding: the stream re-wrapped with new time stamps "
                   "(fast; enough when only the container is broken)")
    p.add_argument("--crf", type=int, default=20, help="x264 quality (lower = better, bigger; 18-23)")
    p.add_argument("--jobs", type=int, default=2, help="files at once")
    p.add_argument("--force", action="store_true", help="convert again files already in --out")
    p.add_argument("--cameras", nargs="*", default=None, help="only these cameras' files (cam1 cam2: the id in the path)")
    p.add_argument("--first", type=float, default=None,
                   help="only the first this many minutes, the same stretch on every camera; files named by their "
                        "new start")
    p.add_argument("--from", dest="start_at", default=None,
                   help="with --first N: N minutes from this time (HH:MM[:SS], the clock of the names; or "
                        "YYYY-MM-DD HH:MM) instead of from the start")
    p.add_argument("--last", type=float, default=None,
                   help="only the last this many minutes, the same stretch on every camera; files named by their "
                        "new start")
    args = p.parse_args(argv)
    if args.first and args.last:
        p.error("--first or --last, not both")
    minutes = args.first or args.last
    if args.start_at and not args.first:
        p.error("--from needs --first N: N minutes from then")
    if minutes and args.copy:                    # a copied stream starts at a keyframe: seconds off, names wrong
        p.error("--first / --last cut frame-exact only when the video is encoded again: leave --copy out")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is not installed (the Docker image has it; on a server: apt install ffmpeg)")
    src_dir, out_dir = os.path.abspath(args.videos.rstrip("\\")), os.path.abspath(args.out.rstrip("\\"))
    todo, seen, other = [], [], []
    for d, _, fs in os.walk(src_dir):
        if os.path.abspath(d).startswith(out_dir):
            continue
        for f in sorted(fs):
            if ".part." in f:
                continue
            if not f.lower().endswith(VIDEO_EXT):
                other.append(f)
                continue
            src = os.path.join(d, f)
            seen.append(os.path.relpath(src, src_dir))
            cam = camera_of(os.path.relpath(src, src_dir), args.cameras) if args.cameras else None
            if args.cameras and cam is None:
                continue
            dst = os.path.join(out_dir, os.path.relpath(d, src_dir), os.path.splitext(f)[0] + ".mp4")
            todo.append((src, dst, cam or "all"))
    cuts = {}
    if minutes and todo:
        if not args.cameras:                      # every camera on its own: the id before the date in the name
            todo = [(s, d, re.split(r"[_ .-]", os.path.basename(s))[0]) for s, d, _ in todo]
        kept, (w0, w1), cover = stretch(todo, minutes, first=bool(args.first), start_at=args.start_at)
        fmt = lambda t: dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        what = (f"{minutes:g} min from {args.start_at}" if args.start_at else
                f"the {'first' if args.first else 'last'} {minutes:g} min")
        print(f"[fix] {what}: {fmt(w0)} - {fmt(w1)} (the times in the names), the same on every "
              "camera", flush=True)
        for c, secs in sorted(cover.items()):
            print(f"[fix]   {c}: {secs / 60:.1f} min of it" + ("" if secs >= 0.95 * (w1 - w0) else
                  "  !! its recording does not cover the whole stretch"), flush=True)
        cuts = {dst: cut for _, dst, cut in kept}
        todo = [(src, dst) for src, dst, _ in kept]
    else:
        todo = [(src, dst) for src, dst, _ in todo]
    ready = []
    for src, dst in todo:
        if os.path.exists(dst) and not args.force:
            print(f"[fix] {os.path.relpath(dst, out_dir)}: there already", flush=True)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        ready.append((src, dst))
    todo = ready
    if not todo:
        if not seen:
            print(f"[fix] no video under {src_dir} ({', '.join(VIDEO_EXT)})" +
                  (f"; there are {len(other)} other files, e.g. {', '.join(sorted(other)[:5])}" if other else
                   " - is the folder in docker/data/ (it is /data in the container)?"))
        elif args.cameras and not any(camera_of(x, args.cameras) for x in seen):
            print(f"[fix] {len(seen)} videos under {src_dir}, none of them named by a camera of "
                  f"{' '.join(args.cameras)}, e.g. {', '.join(seen[:5])} - rename them (cam1_2026-10-27_12-00-00.ts) "
                  "or leave --cameras out")
        else:
            print(f"[fix] nothing to convert under {src_dir} (all of it in {out_dir} already: --force does it again)")
        return 0
    print(f"[fix] {len(todo)} recordings -> {out_dir} ({'re-wrapped' if args.copy else 'encoded again, H.264'}, "
          f"{args.jobs} at once)", flush=True)
    bad = 0
    with cf.ThreadPoolExecutor(max(1, args.jobs)) as ex:
        for msg in ex.map(lambda t: one(t[0], t[1], args.fps, args.copy, args.crf, cuts.get(t[1])), todo):
            bad += msg.startswith("!!")
            print(f"[fix] {msg}", flush=True)
    print(f"[fix] done -> use --videos {out_dir}" + (f"; {bad} failed" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
