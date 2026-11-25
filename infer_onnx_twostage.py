#!/usr/bin/env python3
"""Pure ONNX Runtime Two-Stage Inference for 15th NUCODC SmartCar Xiangqi Competition.
No PyTorch / Ultralytics required on the competition car!
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

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


class YoloOnnxDetector:
    def __init__(self, onnx_path: str | Path, providers: list[str] | None = None):
        self.session = ort.InferenceSession(str(onnx_path), providers=providers or ["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.imgsz = int(inp.shape[2])

    def predict(self, img: np.ndarray, conf_thr: float = 0.25, iou_thr: float = 0.50):
        ih, iw = img.shape[:2]
        lb, r, (dw, dh) = letterbox(img, self.imgsz)
        blob = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        out = self.session.run(None, {self.input_name: blob[None, ...]})[0][0]  # (4+nc, N)
        out = out.T  # (N, 4+nc)
        scores_all = out[:, 4:]
        cids = np.argmax(scores_all, axis=1)
        confs = scores_all[np.arange(len(out)), cids]
        mask = confs >= conf_thr
        if not np.any(mask):
            return []
        out, cids, confs = out[mask], cids[mask], confs[mask]
        cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        # Class-wise NMS
        results = []
        for cid in np.unique(cids):
            idx = np.where(cids == cid)[0]
            keep = nms_boxes(boxes[idx], confs[idx], iou_thr)
            for k in keep:
                b = boxes[idx[k]]
                x1 = max(0, min(iw, int(round((b[0] - dw) / r))))
                y1 = max(0, min(ih, int(round((b[1] - dh) / r))))
                x2 = max(0, min(iw, int(round((b[2] - dw) / r))))
                y2 = max(0, min(ih, int(round((b[3] - dh) / r))))
                if x2 > x1 and y2 > y1:
                    results.append((int(cid), float(confs[idx[k]]), [x1, y1, x2, y2]))
        return results


def run_twostage(s1: YoloOnnxDetector, s2: YoloOnnxDetector, img: np.ndarray, conf1: float = 0.22, iou1: float = 0.60, conf2: float = 0.25):
    ih, iw = img.shape[:2]
    dets = s1.predict(img, conf_thr=conf1, iou_thr=iou1)
    cars = [(sc, box) for cid, sc, box in dets if cid == 1]
    pieces = [(sc, box) for cid, sc, box in dets if cid == 0]
    outputs = [{"class": "smart_car", "zh": "智能车", "conf": round(sc, 3), "xyxy": box} for sc, box in cars]

    for sc, (x1, y1, x2, y2) in pieces:
        # Filter obvious false-positive detections deep inside the car's center PCB/roof core
        pcx, pcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        in_core = any(
            csc >= 0.45 and (cx1 + 0.25 * (cx2 - cx1)) <= pcx <= (cx2 - 0.25 * (cx2 - cx1))
            and (cy1 + 0.25 * (cy2 - cy1)) <= pcy <= (cy2 - 0.25 * (cy2 - cy1))
            for csc, (cx1, cy1, cx2, cy2) in cars
        )
        if in_core and sc < 0.72:
            continue
        pw, ph = x2 - x1, y2 - y1
        mx, my = int(round(pw * 0.06)), int(round(ph * 0.06))
        crop = img[max(0, y1 - my):min(ih, y2 + my), max(0, x1 - mx):min(iw, x2 + mx)]
        sub = s2.predict(crop, conf_thr=conf2, iou_thr=0.45)
        if not sub:
            continue
        ch, cw = crop.shape[:2]
        # Prefer box closest to crop center to avoid neighbor-piece stealing
        best = max(sub, key=lambda d: d[1] - 1.5 * ((((d[2][0] + d[2][2]) / 2 - cw / 2) / max(cw, 1)) ** 2 + (((d[2][1] + d[2][3]) / 2 - ch / 2) / max(ch, 1)) ** 2))
        cname = STAGE2_NAMES[best[0]]
        outputs.append({"class": cname, "zh": ZH_NAMES[cname], "conf": round(best[1], 3), "xyxy": [x1, y1, x2, y2]})
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Two-stage ONNX Runtime inference for 15th NUCODC SmartCar Xiangqi.")
    ap.add_argument("--stage1", required=True, help="Path to stage1_*.onnx")
    ap.add_argument("--stage2", required=True, help="Path to stage2_*.onnx")
    ap.add_argument("--image", required=True, help="Input image path")
    ap.add_argument("--out", default="output_pred.jpg", help="Output visualization path")
    ap.add_argument("--conf1", type=float, default=0.22)
    ap.add_argument("--iou1", type=float, default=0.60)
    ap.add_argument("--conf2", type=float, default=0.25)
    args = ap.parse_args()

    img = cv2.imdecode(np.fromfile(args.image, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert img is not None, f"Failed to load image: {args.image}"
    s1 = YoloOnnxDetector(args.stage1)
    s2 = YoloOnnxDetector(args.stage2)
    preds = run_twostage(s1, s2, img, args.conf1, args.iou1, args.conf2)

    vis = img.copy()
    for p in preds:
        x1, y1, x2, y2 = p["xyxy"]
        col = (0, 165, 255) if p["class"] == "smart_car" else ((50, 50, 240) if p["class"].startswith("red_") else (240, 220, 40))
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        cv2.putText(vis, f"{p['class']} {p['conf']:.2f}", (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    cv2.imwrite(args.out, vis)
    print(f"Detected {len(preds)} objects -> saved to {args.out}")
    for p in preds:
        print(f"  {p['zh']:4s} ({p['class']:14s}) conf={p['conf']:.2f} xyxy={p['xyxy']}")


if __name__ == "__main__":
    main()
