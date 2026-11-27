#!/usr/bin/env python3
"""Lightweight multi-frame tracker + 14-class probability EMA voter for real-time video inference.

Speeds up embedded two-stage inference by 2x-4x:
- Once a stationary chess piece accumulates >= lock_frames high-confidence observations,
  its 14-class probability distribution is locked and reused without re-running Stage 2!
- Eliminates single-frame glare flickers (e.g., transient red_elephant <-> red_cannon flips).
"""
from __future__ import annotations

import numpy as np


def _iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    ab = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(aa + ab - inter, 1e-9)


class PieceTemporalTracker:
    def __init__(self, iou_thr: float = 0.40, ema_alpha: float = 0.45, lock_frames: int = 4, lock_conf: float = 0.80, max_missed: int = 6):
        self.iou_thr = iou_thr
        self.ema_alpha = ema_alpha
        self.lock_frames = lock_frames
        self.lock_conf = lock_conf
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: dict[int, dict] = {}

    def lookup_locked(self, box: list[float]) -> tuple[np.ndarray, float] | None:
        best_id, best_iou = None, self.iou_thr
        for tid, tr in self.tracks.items():
            ov = _iou(box, tr["xyxy"])
            if ov > best_iou:
                best_iou, best_id = ov, tid
        if best_id is None:
            return None
        tr = self.tracks[best_id]
        if tr["hits"] >= self.lock_frames and tr["conf"] >= self.lock_conf:
            return tr["probs"].copy(), float(tr["conf"])
        return None

    def update(self, detections: list[dict]) -> list[dict]:
        matched_tids = set()
        outputs = []
        for det in detections:
            box = det["xyxy"]
            best_tid, best_iou = None, self.iou_thr
            for tid, tr in self.tracks.items():
                if tid in matched_tids:
                    continue
                ov = _iou(box, tr["xyxy"])
                if ov > best_iou:
                    best_iou, best_tid = ov, tid
            if best_tid is not None:
                tr = self.tracks[best_tid]
                tr["xyxy"] = box
                tr["probs"] = (1.0 - self.ema_alpha) * tr["probs"] + self.ema_alpha * det["probs"]
                tr["probs"] /= max(1e-6, float(tr["probs"].sum()))
                tr["conf"] = max(float(det["conf"]), 0.7 * tr["conf"] + 0.3 * float(det["conf"]))
                tr["hits"] += 1
                tr["missed"] = 0
                matched_tids.add(best_tid)
                tid = best_tid
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "xyxy": box,
                    "probs": det["probs"].copy(),
                    "conf": float(det["conf"]),
                    "hits": 1,
                    "missed": 0,
                }
                matched_tids.add(tid)

            tr = self.tracks[tid]
            outputs.append({
                **det,
                "track_id": tid,
                "probs": tr["probs"].copy(),
                "conf": tr["conf"],
            })

        stale = []
        for tid, tr in self.tracks.items():
            if tid not in matched_tids:
                tr["missed"] += 1
                if tr["missed"] > self.max_missed:
                    stale.append(tid)
        for tid in stale:
            del self.tracks[tid]
        return outputs
