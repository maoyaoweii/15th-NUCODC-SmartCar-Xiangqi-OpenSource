#!/usr/bin/env python3
"""Pure ONNX Runtime Two-Stage Inference for 15th NUCODC SmartCar Xiangqi Competition (V5.2).

Zero PyTorch / Ultralytics dependency! Supports:
- Single image (`--image` or `--source`), folder batch, video file, or live camera (`--source 0`)
- Smart-car IoA/IoU deduplication (eliminates nested duplicate car boxes)
- Adaptive car-overlap Stage 2 confidence gate (eliminates chassis red-wire FPs while preserving car-edge pieces)
- Dual output: 14-class exact chess glyph + 10-class competition score tier (10/5/3/2/1 pts) + live score tally
- Optional distance & ground-coordinate estimation (`--calib calib.json`: monocular 67mm width or ground homography)
- Optional multi-frame temporal voting & Stage 2 lock caching (`--temporal`) for 30-60 FPS real-time video
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except Exception:
    HAS_PIL = False

from video_tracker import PieceTemporalTracker

STAGE1_NAMES = ["chess_piece", "smart_car"]
STAGE2_NAMES = [
    "red_rook", "red_horse", "red_elephant", "red_advisor", "red_general", "red_cannon", "red_pawn",
    "black_rook", "black_horse", "black_elephant", "black_advisor", "black_general", "black_cannon", "black_pawn",
]
ZH_NAMES = {
    "smart_car": "智能车",
    "red_rook": "红車", "red_horse": "红馬", "red_elephant": "红相", "red_advisor": "红仕",
    "red_general": "红帥", "red_cannon": "红炮", "red_pawn": "红兵",
    "black_rook": "白車", "black_horse": "白馬", "black_elephant": "白象", "black_advisor": "白士",
    "black_general": "白將", "black_cannon": "白炮", "black_pawn": "白卒",
}
SCORE_BUCKETS = [
    ("red_10", [4], 10, "红10分"),
    ("red_5", [0], 5, "红5分"),
    ("red_3", [1, 5], 3, "红3分"),
    ("red_2", [2, 3], 2, "红2分"),
    ("red_1", [6], 1, "红1分"),
    ("black_10", [11], 10, "白10分"),
    ("black_5", [7], 5, "白5分"),
    ("black_3", [8, 12], 3, "白3分"),
    ("black_2", [9, 10], 2, "白2分"),
    ("black_1", [13], 1, "白1分"),
]


def imread_unicode(path: str | Path) -> np.ndarray:
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img


def imwrite_unicode(path: str | Path, img: np.ndarray, quality: int = 94) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower() or ".jpg"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {p}")
    p.write_bytes(buf.tobytes())


def letterbox(img: np.ndarray, new_shape: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (new_shape - nw) // 2, (new_shape - nh) // 2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    canvas[dh:dh + nh, dw:dw + nw] = resized
    return canvas, r, (dw, dh)


def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / np.maximum(areas[i] + areas[order[1:]] - inter, 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def dedup_containment(items: list[dict], conf_key: str, ioa_thr: float, iou_thr: float) -> list[dict]:
    items = sorted(items, key=lambda x: -x[conf_key])
    kept = []
    for it in items:
        x1, y1, x2, y2 = it["xyxy"]
        a1 = max(1.0, (x2 - x1) * (y2 - y1))
        drop = False
        for k in kept:
            kx1, ky1, kx2, ky2 = k["xyxy"]
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            a2 = max(1.0, (kx2 - kx1) * (ky2 - ky1))
            if inter / min(a1, a2) > ioa_thr or inter / max(a1 + a2 - inter, 1e-9) > iou_thr:
                drop = True
                break
        if not drop:
            kept.append(it)
    return kept


def max_ioa_with_cars(box: list[float], cars: list[dict]) -> float:
    x1, y1, x2, y2 = box
    pa = max(1.0, (x2 - x1) * (y2 - y1))
    best = 0.0
    for c in cars:
        cx1, cy1, cx2, cy2 = c["xyxy"]
        inter = max(0.0, min(x2, cx2) - max(x1, cx1)) * max(0.0, min(y2, cy2) - max(y1, cy1))
        best = max(best, inter / pa)
    return best


class DistanceEstimator:
    """Optional monocular 67mm-width and/or ground-plane homography distance estimator."""

    def __init__(self, calib_path: str | Path | None = None):
        self.cfg = None
        self.H = None
        if calib_path:
            self.cfg = json.loads(Path(calib_path).read_text(encoding="utf-8"))
            if "homography" in self.cfg:
                self.H = np.array(self.cfg["homography"], dtype=np.float64)

    def estimate(self, box: list[float], img_w: int) -> dict | None:
        if not self.cfg:
            return None
        x1, y1, x2, y2 = box
        w_px = max(1.0, x2 - x1)
        out: dict = {}

        # Scale focal constant K if runtime resolution differs from calibration width
        if "K_cm_px" in self.cfg:
            calib_w = float(self.cfg.get("calib_img_width", img_w))
            k_scaled = float(self.cfg["K_cm_px"]) * (img_w / max(1.0, calib_w))
            out["mono_dist_cm"] = round(k_scaled / w_px, 1)

        if self.H is not None:
            # Ground-contact point = bottom-center of bounding box (cx, y2)
            calib_w = float(self.cfg.get("calib_img_width", img_w))
            s = calib_w / max(1.0, img_w)
            pt = np.array([[[((x1 + x2) * 0.5) * s, y2 * s]]], dtype=np.float64)
            world = cv2.perspectiveTransform(pt, self.H)[0, 0]
            x_mm, y_mm = float(world[0]), float(world[1])
            dist_cm = math.hypot(x_mm, y_mm) / 10.0
            out["world_x_mm"] = round(x_mm, 1)
            out["world_y_mm"] = round(y_mm, 1)
            out["dist_cm"] = round(dist_cm, 1)
            # Physical width consistency check when both H and K are calibrated
            if "K_cm_px" in self.cfg and dist_cm > 5.0:
                k_scaled = float(self.cfg["K_cm_px"]) * (img_w / max(1.0, calib_w))
                expected_w = k_scaled / dist_cm
                out["width_ratio"] = round(w_px / max(1.0, expected_w), 2)
        elif "mono_dist_cm" in out:
            out["dist_cm"] = out["mono_dist_cm"]
        return out or None


class YoloOnnxDetector:
    def __init__(self, onnx_path: str | Path, providers: list[str] | None = None):
        ort.set_default_logger_severity(3)
        avail = ort.get_available_providers()
        if providers is None:
            providers = [p for p in ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"] if p in avail]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers or ["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.imgsz = int(inp.shape[2])

    def predict(self, img: np.ndarray, conf_thr: float = 0.20, iou_thr: float = 0.55) -> list[tuple[int, float, list[float]]]:
        ih, iw = img.shape[:2]
        lb, r, (dw, dh) = letterbox(img, self.imgsz)
        blob = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        out = self.session.run(None, {self.input_name: blob[None, ...]})[0][0]  # (4+nc, N)
        out = out.T
        scores_all = out[:, 4:]
        cids = np.argmax(scores_all, axis=1)
        confs = scores_all[np.arange(len(out)), cids]
        mask = confs >= conf_thr
        if not np.any(mask):
            return []
        out, cids, confs = out[mask], cids[mask], confs[mask]
        cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

        results = []
        for cid in np.unique(cids):
            idx = np.where(cids == cid)[0]
            keep = nms_boxes(boxes[idx], confs[idx], iou_thr)
            for k in keep:
                b = boxes[idx[k]]
                x1 = max(0.0, min(float(iw), (b[0] - dw) / r))
                y1 = max(0.0, min(float(ih), (b[1] - dh) / r))
                x2 = max(0.0, min(float(iw), (b[2] - dw) / r))
                y2 = max(0.0, min(float(ih), (b[3] - dh) / r))
                if x2 > x1 and y2 > y1:
                    results.append((int(cid), float(confs[idx[k]]), [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]))
        return results


def predict_s2_probs(s2: YoloOnnxDetector, crop: np.ndarray) -> tuple[np.ndarray, float]:
    probs = np.zeros(14, dtype=np.float32)
    dets = s2.predict(crop, conf_thr=0.05, iou_thr=0.50)
    if not dets:
        return probs, 0.0
    max_c = 0.0
    for cid, cf, _ in dets:
        if 0 <= cid < 14:
            probs[cid] += cf
            if cf > max_c:
                max_c = cf
    tot = float(probs.sum())
    if tot > 1e-6:
        probs /= tot
    return probs, max_c


def run_twostage_frame(
    s1: YoloOnnxDetector,
    s2: YoloOnnxDetector,
    img: np.ndarray,
    car_conf: float = 0.34,
    piece_s1_conf: float = 0.22,
    iou1: float = 0.55,
    s2_floor_conf: float = 0.45,
    s2_car_conf: float = 0.56,
    fast_single_crop: bool = False,
    estimator: DistanceEstimator | None = None,
    tracker: PieceTemporalTracker | None = None,
) -> dict:
    ih, iw = img.shape[:2]
    dets = s1.predict(img, conf_thr=min(car_conf, piece_s1_conf), iou_thr=iou1)

    raw_cars, raw_pieces = [], []
    for cid, cf, box in dets:
        x1, y1, x2, y2 = box
        if cid == 1 and cf >= car_conf:
            raw_cars.append({"class": "smart_car", "zh": "智能车", "conf": round(cf, 4), "xyxy": box})
        elif cid == 0 and cf >= piece_s1_conf and (x2 - x1) >= 18 and (y2 - y1) >= 16:
            raw_pieces.append({"s1_conf": round(cf, 4), "xyxy": box})

    cars = dedup_containment(raw_cars, "conf", ioa_thr=0.45, iou_thr=0.30)
    raw_pieces = dedup_containment(raw_pieces, "s1_conf", ioa_thr=0.65, iou_thr=0.50)

    pads = [(0.15, 1.0)] if fast_single_crop else [(0.12, 1.0), (0.22, 0.70)]
    kept_pieces = []
    rejected = []

    for pc in raw_pieces:
        box = pc["xyxy"]
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1

        # Reuse cached Stage 2 lock in video mode if track is already stably locked
        cached = tracker.lookup_locked(box) if tracker else None
        if cached is not None:
            probs_norm, max_s2 = cached
        else:
            probs = np.zeros(14, dtype=np.float32)
            max_s2 = 0.0
            for pad, w in pads:
                px1, py1 = max(0, int(x1 - bw * pad)), max(0, int(y1 - bh * pad))
                px2, py2 = min(iw, int(x2 + bw * pad)), min(ih, int(y2 + bh * pad))
                crop = img[py1:py2, px1:px2]
                if crop.size == 0 or min(crop.shape[:2]) < 8:
                    continue
                pr, mc = predict_s2_probs(s2, crop)
                probs += pr * w
                if mc > max_s2:
                    max_s2 = mc
            tot = float(probs.sum())
            probs_norm = probs / tot if tot > 1e-6 else probs

        ioa_car = max_ioa_with_cars(box, cars)
        gate = s2_car_conf if ioa_car >= 0.50 else s2_floor_conf
        if float(probs_norm.sum()) <= 1e-6 or max_s2 < gate:
            rejected.append({"xyxy": box, "s1_conf": pc["s1_conf"], "s2_conf": round(max_s2, 4), "ioa_car": round(ioa_car, 2)})
            continue

        dist_info = estimator.estimate(box, iw) if estimator else None
        # Optional physical width sanity check (only when both H and K are calibrated)
        if dist_info and "width_ratio" in dist_info and not (0.45 <= dist_info["width_ratio"] <= 2.2) and max_s2 < 0.85:
            rejected.append({"xyxy": box, "s1_conf": pc["s1_conf"], "s2_conf": round(max_s2, 4), "reason": "width_mismatch"})
            continue

        kept_pieces.append({"xyxy": box, "s1_conf": pc["s1_conf"], "probs": probs_norm, "conf": max_s2, "dist": dist_info})

    if tracker:
        kept_pieces = tracker.update(kept_pieces)

    final_pieces = []
    red_score, black_score = 0, 0
    for kp in kept_pieces:
        probs_norm = kp["probs"]
        cls_id = int(np.argmax(probs_norm))
        cname = STAGE2_NAMES[cls_id]
        b_probs = [float(sum(probs_norm[c] for c in cids)) for _, cids, _, _ in SCORE_BUCKETS]
        best_b = int(np.argmax(b_probs))
        b_key, _, pts, b_zh = SCORE_BUCKETS[best_b]
        if b_key.startswith("red_"):
            red_score += pts
        else:
            black_score += pts
        item = {
            "class": cname,
            "zh": ZH_NAMES[cname],
            "conf": round(float(kp["conf"]), 4),
            "s1_conf": kp["s1_conf"],
            "score_bucket": b_key,
            "score_zh": b_zh,
            "score_pts": pts,
            "score_prob": round(b_probs[best_b], 4),
            "xyxy": [int(round(v)) for v in kp["xyxy"]],
        }
        if kp.get("track_id") is not None:
            item["track_id"] = kp["track_id"]
        if kp.get("dist"):
            item.update(kp["dist"])
        final_pieces.append(item)

    for c in cars:
        c["xyxy"] = [int(round(v)) for v in c["xyxy"]]
        if estimator:
            cd = estimator.estimate(c["xyxy"], iw)
            if cd and "world_x_mm" in cd:
                c["world_x_mm"] = cd["world_x_mm"]
                c["world_y_mm"] = cd["world_y_mm"]
                c["dist_cm"] = cd["dist_cm"]

    return {
        "width": iw,
        "height": ih,
        "cars": cars,
        "pieces": final_pieces,
        "filtered_out": rejected,
        "red_score": red_score,
        "black_score": black_score,
    }


def _load_cjk_font(size: int):
    if not HAS_PIL:
        return None
    for cand in [
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]:
        if Path(cand).exists():
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_visualization(img: np.ndarray, res: dict, score_only: bool = False) -> np.ndarray:
    vis = img.copy()
    if HAS_PIL:
        pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        f_sz = max(16, int(min(pil.size) * 0.022))
        font = _load_cjk_font(f_sz)
        bfont = _load_cjk_font(max(18, int(min(pil.size) * 0.026)))

        for c in res["cars"]:
            x1, y1, x2, y2 = c["xyxy"]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 180, 0), width=3)
            extra = f" {c['dist_cm']}cm" if "dist_cm" in c else ""
            txt = f"智能车 {c['conf']:.2f}{extra}"
            tb = draw.textbbox((x1, max(0, y1 - f_sz - 6)), txt, font=font)
            draw.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=(255, 180, 0))
            draw.text((x1, max(0, y1 - f_sz - 6)), txt, fill=(0, 0, 0), font=font)

        for p in res["pieces"]:
            x1, y1, x2, y2 = p["xyxy"]
            is_red = p["class"].startswith("red_")
            col = (235, 45, 45) if is_red else (0, 215, 245)
            bg = (185, 25, 25) if is_red else (0, 120, 165)
            draw.rectangle([x1, y1, x2, y2], outline=col, width=2)
            label = p["score_zh"] if score_only else f"{p['zh']}·{p['score_pts']}分"
            dist_s = f" {p['dist_cm']}cm" if "dist_cm" in p else ""
            txt = f"{label} {p['conf']:.2f}{dist_s}"
            tb = draw.textbbox((x1, max(0, y1 - f_sz - 4)), txt, font=font)
            draw.rectangle([tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1], fill=bg)
            draw.text((x1, max(0, y1 - f_sz - 4)), txt, fill=(255, 255, 255), font=font)

        banner = f"V5.2 ONNX | 车:{len(res['cars'])} 棋:{len(res['pieces'])} 拒:{len(res['filtered_out'])} | 红{res['red_score']}分 白{res['black_score']}分"
        draw.rectangle([0, 0, pil.width, f_sz + 18], fill=(15, 23, 42))
        draw.text((10, 6), banner, fill=(255, 255, 255), font=bfont)
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    # Pure OpenCV fallback (no Pillow)
    for c in res["cars"]:
        x1, y1, x2, y2 = c["xyxy"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(vis, f"car {c['conf']:.2f}", (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2)
    for p in res["pieces"]:
        x1, y1, x2, y2 = p["xyxy"]
        col = (45, 45, 235) if p["class"].startswith("red_") else (245, 215, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        lbl = f"{p['score_bucket'] if score_only else p['class']} {p['conf']:.2f}"
        cv2.putText(vis, lbl, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 2)
    return vis


def main() -> None:
    ap = argparse.ArgumentParser(description="V5.2 Pure ONNX Runtime Two-Stage Inference (15th NUCODC SmartCar Xiangqi).")
    ap.add_argument("--stage1", required=True, help="Path to stage1_v8*_v52_*.onnx")
    ap.add_argument("--stage2", required=True, help="Path to stage2_v8*_v52_*.onnx")
    ap.add_argument("--image", default=None, help="Single input image path (backward-compatible alias of --source)")
    ap.add_argument("--source", default=None, help="Image path, folder path, video path, or camera index (e.g. 0)")
    ap.add_argument("--out", default="output_pred.jpg", help="Output image/folder/video path")
    ap.add_argument("--json-out", default=None, help="Optional JSON output path")
    ap.add_argument("--calib", default=None, help="Optional calibration JSON from calibrate_camera.py")
    ap.add_argument("--car-conf", type=float, default=0.34)
    ap.add_argument("--conf1", type=float, default=0.22, help="Stage 1 chess_piece threshold")
    ap.add_argument("--iou1", type=float, default=0.55)
    ap.add_argument("--s2-floor-conf", type=float, default=0.45, help="Stage 2 min conf on open floor")
    ap.add_argument("--s2-car-conf", type=float, default=0.56, help="Stage 2 min conf inside smart_car bbox (suppresses wires)")
    ap.add_argument("--fast-single-crop", action="store_true", help="Use single 15%% crop for 2x faster Stage 2 on embedded boards")
    ap.add_argument("--score-only", action="store_true", help="Render 10-class competition score tiers instead of 14 glyphs")
    ap.add_argument("--temporal", action="store_true", help="Enable multi-frame voting & Stage 2 lock cache for video/camera")
    args = ap.parse_args()

    src_arg = args.source or args.image
    if not src_arg:
        ap.error("Please provide --image or --source")

    s1 = YoloOnnxDetector(args.stage1)
    s2 = YoloOnnxDetector(args.stage2)
    estimator = DistanceEstimator(args.calib)

    # Camera / Video mode
    is_cam = src_arg.isdigit()
    src_path = Path(src_arg)
    is_video = is_cam or (src_path.is_file() and src_path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv"))
    if is_video:
        cap = cv2.VideoCapture(int(src_arg) if is_cam else str(src_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {src_arg}")
        tracker = PieceTemporalTracker() if args.temporal else None
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = None
        if not is_cam and args.out:
            out_v = Path(args.out)
            out_v.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(out_v), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        f_idx = 0
        t0 = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = run_twostage_frame(
                s1, s2, frame, args.car_conf, args.conf1, args.iou1,
                args.s2_floor_conf, args.s2_car_conf, args.fast_single_crop, estimator, tracker,
            )
            vis = draw_visualization(frame, res, args.score_only)
            if writer:
                writer.write(vis)
            elif is_cam:
                cv2.imshow("15th NUCODC Xiangqi V5.2 ONNX", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            f_idx += 1
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        dt = max(1e-6, time.perf_counter() - t0)
        print(f"Processed {f_idx} frames in {dt:.2f}s ({f_idx / dt:.1f} FPS)")
        return

    # Single image or folder batch mode
    imgs = [src_path] if src_path.is_file() else sorted(p for p in src_path.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    all_res = {}
    out_p = Path(args.out)
    for img_p in imgs:
        t0 = time.perf_counter()
        im = imread_unicode(img_p)
        res = run_twostage_frame(
            s1, s2, im, args.car_conf, args.conf1, args.iou1,
            args.s2_floor_conf, args.s2_car_conf, args.fast_single_crop, estimator, None,
        )
        ms = (time.perf_counter() - t0) * 1000.0
        all_res[img_p.name] = res
        vis = draw_visualization(im, res, args.score_only)
        save_to = out_p if (src_path.is_file() and out_p.suffix) else (out_p / f"{img_p.stem}_v52.jpg")
        imwrite_unicode(save_to, vis)
        print(f"[{ms:5.1f} ms] {img_p.name}: cars={len(res['cars'])} pieces={len(res['pieces'])} rej={len(res['filtered_out'])} | 红{res['red_score']}分 白{res['black_score']}分 -> {save_to}")
        for p in res["pieces"]:
            ds = f" dist={p['dist_cm']}cm" if "dist_cm" in p else ""
            print(f"    {p['zh']:4s} ({p['score_zh']:4s}) conf={p['conf']:.2f}{ds} xyxy={p['xyxy']}")

    if args.json_out:
        def _json_default(o):
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(f"Not JSON serializable: {type(o)}")
        Path(args.json_out).write_text(json.dumps(all_res, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        print(f"Saved JSON -> {args.json_out}")


if __name__ == "__main__":
    main()
