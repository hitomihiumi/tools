"""Loading CBVD-5 annotations.

The dataset ships its annotations in AVA format, which is multi-label: one
physical bounding box appears as several CSV rows, one per action id. The
dataset card describes "behavior categories 0-4", but labelmap.txt uses 1-5
and the two axes are independent:

    posture   1 stand | 2 lying down          - exactly one per box, always
    activity  3 foraging | 4 drinking water | 5 rumination  - zero or one

Flattening that into a 5-way single-label problem is wrong: a model that
answers "lying" for a box annotated (lying, rumination) has not made a
mistake, and scoring it as one inflates the error rate.
"""

from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass

LABEL_NAMES = {
    1: "stand",
    2: "lying down",
    3: "foraging",
    4: "drinking water",
    5: "rumination",
}

# Names as the model is asked to produce them. Kept separate from the dataset's
# own labelmap wording so the prompt can use plain English without silently
# redefining the ground truth.
POSTURE_OF = {1: "standing", 2: "lying"}
ACTIVITY_OF = {3: "feeding", 4: "drinking", 5: "ruminating"}
NO_ACTIVITY = "none"

POSTURES = tuple(POSTURE_OF.values())
ACTIVITIES = tuple(ACTIVITY_OF.values()) + (NO_ACTIVITY,)


@dataclass(frozen=True)
class Box:
    video_id: str
    timestamp: int
    x1: float
    y1: float
    x2: float
    y2: float
    entity_id: str
    labels: tuple  # sorted action ids

    @property
    def uid(self) -> str:
        # The AVA entity id column is 1 for every row in this dataset, so it
        # cannot separate the ~20 cows sharing a keyframe. The box itself is
        # what makes an example unique.
        digest = hashlib.sha1(
            f"{self.x1:.6f},{self.y1:.6f},{self.x2:.6f},{self.y2:.6f}".encode()
        ).hexdigest()[:8]
        return f"{self.video_id}_{self.timestamp:05d}_{digest}"

    @property
    def posture(self):
        for lid in self.labels:
            if lid in POSTURE_OF:
                return POSTURE_OF[lid]
        return None

    @property
    def activity(self) -> str:
        for lid in self.labels:
            if lid in ACTIVITY_OF:
                return ACTIVITY_OF[lid]
        return NO_ACTIVITY

    @property
    def xyxy(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)


def load_boxes(csv_path: str):
    """Group AVA rows back into boxes.

    Coordinates are grouped as their original strings, not as floats: they are
    the join key, and re-parsing them invites a float that does not round-trip
    to split one box into two.
    """
    grouped: dict = {}
    order: list = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if not row or row[0].startswith("#"):
                continue
            video_id, ts, x1, y1, x2, y2, action, entity_id = row[:8]
            key = (video_id, int(ts), x1, y1, x2, y2, entity_id)
            if key not in grouped:
                grouped[key] = set()
                order.append(key)
            # A set, because ava_test_v2.1.csv is ava_val duplicated row for row.
            grouped[key].add(int(action))

    boxes = []
    for key in order:
        video_id, ts, x1, y1, x2, y2, entity_id = key
        boxes.append(
            Box(
                video_id=video_id,
                timestamp=ts,
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                entity_id=entity_id,
                labels=tuple(sorted(grouped[key])),
            )
        )
    return boxes


def partition(boxes):
    """Split into (usable, rejected) by whether the label set is well formed.

    val contains one box annotated both stand AND lying down. There is no
    correct answer to grade a model against there, so it is dropped - counted
    and reported, never silently.
    """
    usable, rejected = [], []
    for box in boxes:
        postures = [l for l in box.labels if l in POSTURE_OF]
        activities = [l for l in box.labels if l in ACTIVITY_OF]
        if len(postures) == 1 and len(activities) <= 1:
            usable.append(box)
        else:
            reason = (
                f"{len(postures)} posture labels, {len(activities)} activity labels: "
                + ", ".join(LABEL_NAMES.get(l, str(l)) for l in box.labels)
            )
            rejected.append((box, reason))
    return usable, rejected


_FRAME_DIRS = (
    os.path.join("labelframes", "labelframes"),
    os.path.join("labelframes_add", "labelframes"),
    "minilabelframes",  # 256x256 - a last resort, too small for a VLM
)


_VIDEO_DIRS = (
    os.path.join("videos", "videos"),
    os.path.join("videos_add", "videos"),
)


def video_path(root: str, video_id: str) -> str:
    for sub in _VIDEO_DIRS:
        candidate = os.path.join(root, sub, f"{video_id}.mp4")
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"no clip {video_id}.mp4 in {', '.join(_VIDEO_DIRS)} under {root}"
    )


def frame_path(root: str, box: Box) -> str:
    name = f"{box.video_id}_{box.timestamp:05d}.jpg"
    for sub in _FRAME_DIRS:
        candidate = os.path.join(root, sub, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"no keyframe for video {box.video_id} at t={box.timestamp}s "
        f"(looked for {name} in {', '.join(_FRAME_DIRS)} under {root})"
    )
