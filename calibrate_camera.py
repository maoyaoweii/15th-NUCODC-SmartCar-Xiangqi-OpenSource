#!/usr/bin/env python3
"""Zero-PyTorch Camera Calibration Tool for 15th NUCODC SmartCar Xiangqi (67 mm standard pieces).

Supports two calibration modes (or both combined in one JSON):
1. Monocular Horizontal-Width Ranging (`--mode monocular`):
   Place one standard 67 mm chess piece at known distance Z0 (cm) directly ahead,
   measure its horizontal bounding-box width w0 (px) -> saves K_cm_px = Z0 * w0.
   Runtime distance: Z = K_cm_px / w_px (scales automatically if image width changes).
2. Ground-Plane Homography (`--mode homography`):
   Provide >= 4 arena floor correspondences `u,v=X_mm,Y_mm` (e.g., relative to robot front center)
   -> computes 3x3 perspective matrix H and reports reprojection error in mm!
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def calibrate_monocular(z0_cm: float, w0_px: float, img_width: int, piece_diam_mm: float = 67.0) -> dict:
    k_cm_px = z0_cm * w0_px
    fx_px = k_cm_px / (piece_diam_mm / 10.0)
    return {
        "mode": "monocular",
        "piece_diameter_mm": piece_diam_mm,
        "calib_img_width": img_width,
        "z0_cm": z0_cm,
        "w0_px": w0_px,
        "K_cm_px": round(k_cm_px, 2),
        "estimated_fx_px": round(fx_px, 2),
    }


def calibrate_homography(pairs: list[str], img_width: int) -> dict:
    if len(pairs) < 4:
        raise ValueError("Homography calibration requires at least 4 point pairs: u,v=X_mm,Y_mm")
    img_pts, world_pts = [], []
    for item in pairs:
        left, right = item.split("=")
        u, v = [float(x) for x in left.split(",")]
        x_mm, y_mm = [float(x) for x in right.split(",")]
        img_pts.append([u, v])
        world_pts.append([x_mm, y_mm])
    src = np.array(img_pts, dtype=np.float64)
    dst = np.array(world_pts, dtype=np.float64)
    H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC if len(pairs) > 4 else 0)
    if H is None:
        raise RuntimeError("cv2.findHomography failed; ensure points are non-collinear")
    proj = cv2.perspectiveTransform(src[:, None, :], H)[:, 0, :]
    errs = np.linalg.norm(proj - dst, axis=1)
    return {
        "mode": "homography",
        "calib_img_width": img_width,
        "homography": H.tolist(),
        "mean_reproj_err_mm": round(float(np.mean(errs)), 2),
        "max_reproj_err_mm": round(float(np.max(errs)), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate camera distance / ground-plane homography for 67mm Xiangqi.")
    ap.add_argument("--mode", choices=["monocular", "homography", "both"], required=True)
    ap.add_argument("--img-width", type=int, required=True, help="Calibration image width in pixels (e.g. 1280)")
    ap.add_argument("--z0-cm", type=float, help="Known piece distance in cm (for monocular / both)")
    ap.add_argument("--w0-px", type=float, help="Measured piece horizontal pixel width w=x2-x1 (for monocular / both)")
    ap.add_argument("--piece-mm", type=float, default=67.0, help="Standard piece diameter in mm (15th NUCODC = 67.0)")
    ap.add_argument("--points", nargs="*", help=">=4 ground pairs 'u,v=X_mm,Y_mm' (for homography / both)")
    ap.add_argument("--out", default="calib.json", help="Output calibration JSON file")
    args = ap.parse_args()

    cfg: dict = {}
    if args.mode in ("monocular", "both"):
        if args.z0_cm is None or args.w0_px is None:
            ap.error("--z0-cm and --w0-px are required for monocular mode")
        cfg.update(calibrate_monocular(args.z0_cm, args.w0_px, args.img_width, args.piece_mm))
    if args.mode in ("homography", "both"):
        if not args.points or len(args.points) < 4:
            ap.error("--points requires at least 4 'u,v=X_mm,Y_mm' entries")
        hcfg = calibrate_homography(args.points, args.img_width)
        cfg.update(hcfg)
        if args.mode == "both":
            cfg["mode"] = "both"

    out_p = Path(args.out)
    out_p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Calibration saved -> {out_p}")
    print(json.dumps(cfg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
