#!/usr/bin/env python3
"""Global World Model & 4-ToF + IMU + Vision Sensor Fusion Engine
for the 15th NUCODC SmartCar Xiangqi Competition (3000 x 2000 mm arena).

Zero PyTorch dependency (pure NumPy). Solves:
1. Self-localization `(x_mm, y_mm, theta_rad)` via 4-way high-mounted ToF (150-240mm height)
   ray-casting against the 3000x2000mm 300mm-walled arena + automatic opponent-occlusion rejection.
2. Rigid-body projection of camera-detected pieces & opponent car (from `infer_onnx_twostage.py`
   + `calibrate_camera.py`) into persistent global arena coordinates.
3. Field-of-View (FOV) visibility cleanup: deletes remembered pieces that vanish inside the
   current camera view cone (e.g. pushed away by opponent), while retaining behind-robot memory.
4. Goal-zone safety classification (Zone A Wall Vault / Zone B Buffer / Zone C Edge Risk)
   and real-time Red/White score estimation.
5. Tactical target recommendation for both 120s Regular Match and 60s Golden-Goal Sudden Death.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Official 15th NUCODC Track 1 Arena Geometry (mm), origin at arena center:
# X in [-1500, +1500] (Red/Team-A goal on left [-1500, -1100], White/Team-B goal on right [+1100, +1500])
# Y in [-1000, +1000] (Goals span Y in [-300, +300])
ARENA_HALF_X = 1500.0
ARENA_HALF_Y = 1000.0
GOAL_DEPTH_MM = 400.0
GOAL_HALF_W_MM = 300.0
PIECE_DIAM_MM = 67.0

SCORE_BY_CLASS = {
    "red_general": 10, "black_general": 10,
    "red_rook": 5, "black_rook": 5,
    "red_horse": 3, "red_cannon": 3, "black_horse": 3, "black_cannon": 3,
    "red_elephant": 2, "red_advisor": 2, "black_elephant": 2, "black_advisor": 2,
    "red_pawn": 1, "black_pawn": 1,
}


def ray_distance_to_arena_wall(x: float, y: float, angle_rad: float) -> float:
    """Exact distance from (x, y) inside [-1500,1500]x[-1000,1000] along ray `angle_rad` to 300mm boundary wall."""
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)
    ts = []
    if abs(dx) > 1e-9:
        t_pos = (ARENA_HALF_X - x) / dx
        t_neg = (-ARENA_HALF_X - x) / dx
        for t in (t_pos, t_neg):
            if t > 0:
                ts.append(t)
    if abs(dy) > 1e-9:
        t_pos = (ARENA_HALF_Y - y) / dy
        t_neg = (-ARENA_HALF_Y - y) / dy
        for t in (t_pos, t_neg):
            if t > 0:
                ts.append(t)
    return min(ts) if ts else 0.0


class ArenaRayCastLocalizer:
    """Fuses wheel odometry + IMU yaw + 4-way high-mounted ToF (Front/Back/Left/Right) with occlusion rejection."""

    def __init__(
        self,
        init_x_mm: float = -1350.0,
        init_y_mm: float = -850.0,
        init_theta_rad: float = 0.0,
        sensor_offsets_mm: tuple[float, float, float, float] = (120.0, 120.0, 110.0, 110.0),
        max_range_mm: float = 3600.0,
        outlier_gate_mm: float = 160.0,
    ):
        self.x = float(init_x_mm)
        self.y = float(init_y_mm)
        self.theta = float(init_theta_rad)
        # Distance from robot center to Front, Back, Left, Right ToF emitters
        self.offsets = sensor_offsets_mm
        self.max_range_mm = max_range_mm
        self.outlier_gate_mm = outlier_gate_mm

    def predict_odometry(self, dx_body_mm: float, dy_body_mm: float, imu_theta_rad: float) -> None:
        """Predict pose from wheel encoders in robot frame + calibrated IMU absolute yaw."""
        self.theta = float(imu_theta_rad)
        c, s = math.cos(self.theta), math.sin(self.theta)
        self.x = float(np.clip(self.x + c * dx_body_mm - s * dy_body_mm, -ARENA_HALF_X + 80, ARENA_HALF_X - 80))
        self.y = float(np.clip(self.y + s * dx_body_mm + c * dy_body_mm, -ARENA_HALF_Y + 80, ARENA_HALF_Y - 80))

    def update_4tof(
        self,
        d_front_mm: float | None,
        d_back_mm: float | None,
        d_left_mm: float | None,
        d_right_mm: float | None,
        gain: float = 0.55,
    ) -> dict:
        """Correct (x, y) against 3000x2000mm walls while rejecting rays occluded by the opponent car.

        Every ToF ray hitting an opponent car returns a distance SHORTER than the wall ray prediction.
        We solve a fast gated Gauss-Newton step using only consistent non-occluded rays!
        """
        # Relative beam angles: Front(0), Back(pi), Left(+pi/2), Right(-pi/2)
        meas_list = [
            ("F", 0.0, self.offsets[0], d_front_mm),
            ("B", math.pi, self.offsets[1], d_back_mm),
            ("L", math.pi / 2.0, self.offsets[2], d_left_mm),
            ("R", -math.pi / 2.0, self.offsets[3], d_right_mm),
        ]

        valid_rays = []
        rejected_rays = []
        for tag, rel_ang, off_mm, d_raw in meas_list:
            if d_raw is None or d_raw <= 20.0 or d_raw >= self.max_range_mm:
                rejected_rays.append((tag, "out_of_range"))
                continue
            beam_ang = self.theta + rel_ang
            d_meas_center = d_raw + off_mm
            d_pred = ray_distance_to_arena_wall(self.x, self.y, beam_ang)
            residual = d_meas_center - d_pred  # Negative residual => obstacle/opponent closer than wall!
            if abs(residual) > self.outlier_gate_mm:
                reason = "opponent_occluded" if residual < -self.outlier_gate_mm else "glare_outlier"
                rejected_rays.append((tag, reason))
                continue
            valid_rays.append((tag, beam_ang, d_meas_center))

        # Iterative Gauss-Newton correction on (x, y) using valid wall rays (3 micro-steps)
        if len(valid_rays) >= 2:
            eps = 2.0
            cx, cy = self.x, self.y
            for _ in range(3):
                J = []
                r_vec = []
                for _, ang, d_meas in valid_rays:
                    d0 = ray_distance_to_arena_wall(cx, cy, ang)
                    jx = (ray_distance_to_arena_wall(cx + eps, cy, ang) - d0) / eps
                    jy = (ray_distance_to_arena_wall(cx, cy + eps, ang) - d0) / eps
                    J.append([jx, jy])
                    r_vec.append(d_meas - d0)
                Jm = np.array(J, dtype=np.float64)
                rv = np.array(r_vec, dtype=np.float64)
                H_mat = Jm.T @ Jm + np.eye(2) * 1e-4
                delta = np.linalg.solve(H_mat, Jm.T @ rv)
                delta = np.clip(delta, -80.0, 80.0)
                cx = float(np.clip(cx + gain * delta[0], -ARENA_HALF_X + 80, ARENA_HALF_X - 80))
                cy = float(np.clip(cy + gain * delta[1], -ARENA_HALF_Y + 80, ARENA_HALF_Y - 80))
            self.x, self.y = cx, cy

        return {
            "x_mm": round(self.x, 1),
            "y_mm": round(self.y, 1),
            "theta_deg": round(math.degrees(self.theta), 2),
            "used_rays": [v[0] for v in valid_rays],
            "rejected_rays": rejected_rays,
        }


@dataclass
class TrackedWorldPiece:
    piece_id: int
    x_mm: float
    y_mm: float
    cls_name: str
    score_pts: int
    conf: float
    last_seen_t: float
    hits: int = 1
    fov_misses: int = 0
    class_votes: dict[str, float] = field(default_factory=dict)


def classify_goal_zone(x_mm: float, y_mm: float, team: str) -> str:
    """Classify piece position relative to team's 400x600mm goal:
    - GOAL_A_SAFE : Deep wall vault (within 140mm of back wall, |y| <= 240mm)
    - GOAL_B_MID  : Mid buffer zone (140..280mm from back wall, |y| <= 240mm)
    - GOAL_C_EDGE : Risky boundary zone (inside goal but <120mm from front line or <60mm from side lines)
    - FIELD       : Outside team goal
    """
    depth = (x_mm - (-ARENA_HALF_X)) if team == "red" else (ARENA_HALF_X - x_mm)
    # Full 67mm projection must stay strictly inside [0, 400] x [-300, +300] (radius 33.5mm)
    rad = PIECE_DIAM_MM * 0.5
    if not (rad <= depth <= GOAL_DEPTH_MM - rad and abs(y_mm) <= GOAL_HALF_W_MM - rad):
        return "FIELD"
    if depth <= 140.0 and abs(y_mm) <= 240.0:
        return "GOAL_A_SAFE"
    if depth <= 280.0 and abs(y_mm) <= 240.0:
        return "GOAL_B_MID"
    return "GOAL_C_EDGE"


class XiangqiWorldModel:
    """Persistent global world map fusing self-pose + V5.2 vision detections."""

    def __init__(self, my_team: str = "red", cam_fov_deg: float = 78.0, cam_range_mm: float = 1800.0):
        assert my_team in ("red", "black"), "my_team must be 'red' (Team A) or 'black' (Team B)"
        self.my_team = my_team
        self.opp_team = "black" if my_team == "red" else "red"
        self.cam_half_fov = math.radians(cam_fov_deg * 0.5)
        self.cam_range_mm = cam_range_mm
        self.pieces: dict[int, TrackedWorldPiece] = {}
        self.opponent_car: dict | None = None
        self._next_id = 1

    def body_to_world(self, pose_x: float, pose_y: float, pose_theta: float, rel_x_mm: float, rel_y_mm: float) -> tuple[float, float]:
        """Convert camera/robot relative coords (x_forward=rel_y_mm, y_left=-rel_x_mm) to global arena coords.

        Note: `calibrate_camera.py` uses X_mm right, Y_mm forward; we convert cleanly to robot (fwd, left).
        """
        fwd = rel_y_mm
        left = -rel_x_mm
        c, s = math.cos(pose_theta), math.sin(pose_theta)
        wx = pose_x + c * fwd - s * left
        wy = pose_y + s * fwd + c * left
        return float(np.clip(wx, -ARENA_HALF_X, ARENA_HALF_X)), float(np.clip(wy, -ARENA_HALF_Y, ARENA_HALF_Y))

    def _in_camera_fov(self, pose_x: float, pose_y: float, pose_theta: float, wx: float, wy: float) -> bool:
        dx, dy = wx - pose_x, wy - pose_y
        dist = math.hypot(dx, dy)
        if dist < 180.0 or dist > self.cam_range_mm:
            return False
        ang = math.atan2(dy, dx)
        diff = (ang - pose_theta + math.pi) % (2.0 * math.pi) - math.pi
        return abs(diff) <= self.cam_half_fov * 0.85

    def ingest_vision_frame(self, pose_x: float, pose_y: float, pose_theta: float, frame_res: dict, now_t: float) -> None:
        """Fuse one frame of `infer_onnx_twostage.py` detections (with `--calib`) into global map."""
        matched_ids = set()
        for det in frame_res.get("pieces", []):
            if "world_x_mm" not in det or "world_y_mm" not in det:
                continue
            wx, wy = self.body_to_world(pose_x, pose_y, pose_theta, det["world_x_mm"], det["world_y_mm"])
            cname = det["class"]
            conf = float(det["conf"])
            # Nearest-neighbor match within 75 mm (slightly larger than 1 piece diameter)
            best_id, best_d = None, 75.0
            for pid, tp in self.pieces.items():
                if pid in matched_ids:
                    continue
                d = math.hypot(wx - tp.x_mm, wy - tp.y_mm)
                if d < best_d:
                    best_d, best_id = d, pid
            if best_id is not None:
                tp = self.pieces[best_id]
                tp.x_mm = 0.6 * tp.x_mm + 0.4 * wx
                tp.y_mm = 0.6 * tp.y_mm + 0.4 * wy
                tp.class_votes[cname] = tp.class_votes.get(cname, 0.0) + conf
                tp.cls_name = max(tp.class_votes.items(), key=lambda kv: kv[1])[0]
                tp.score_pts = SCORE_BY_CLASS.get(tp.cls_name, 1)
                tp.conf = max(tp.conf * 0.7 + conf * 0.3, conf)
                tp.last_seen_t = now_t
                tp.hits += 1
                tp.fov_misses = 0
                matched_ids.add(best_id)
            else:
                pid = self._next_id
                self._next_id += 1
                self.pieces[pid] = TrackedWorldPiece(
                    piece_id=pid,
                    x_mm=wx,
                    y_mm=wy,
                    cls_name=cname,
                    score_pts=SCORE_BY_CLASS.get(cname, 1),
                    conf=conf,
                    last_seen_t=now_t,
                    class_votes={cname: conf},
                )
                matched_ids.add(pid)

        # Update opponent car position
        cars = frame_res.get("cars", [])
        if cars and "world_x_mm" in cars[0]:
            ox, oy = self.body_to_world(pose_x, pose_y, pose_theta, cars[0]["world_x_mm"], cars[0]["world_y_mm"])
            self.opponent_car = {"x_mm": round(ox, 1), "y_mm": round(oy, 1), "last_seen_t": now_t}

        # FOV visibility cleanup: if a piece SHOULD be visible right in front of us but wasn't seen 3x, prune it
        to_delete = []
        for pid, tp in self.pieces.items():
            if pid in matched_ids:
                continue
            if self._in_camera_fov(pose_x, pose_y, pose_theta, tp.x_mm, tp.y_mm):
                tp.fov_misses += 1
                if tp.fov_misses >= 3:
                    to_delete.append(pid)
            elif (now_t - tp.last_seen_t) > 35.0:
                to_delete.append(pid)
        for pid in to_delete:
            del self.pieces[pid]

    def scoreboard(self) -> dict:
        my_score, opp_score = 0, 0
        zone_counts = {"GOAL_A_SAFE": 0, "GOAL_B_MID": 0, "GOAL_C_EDGE": 0}
        for tp in self.pieces.values():
            team = "red" if tp.cls_name.startswith("red_") else "black"
            z = classify_goal_zone(tp.x_mm, tp.y_mm, team)
            if z != "FIELD":
                if team == self.my_team:
                    my_score += tp.score_pts
                    zone_counts[z] += 1
                else:
                    opp_score += tp.score_pts
        return {"my_score": my_score, "opp_score": opp_score, "my_goal_zones": zone_counts}

    def recommend_next_action(self, pose_x: float, pose_y: float, time_left_s: float, golden_goal: bool = False) -> dict:
        """Select highest-utility tactical target for 120s Regular Match or 60s Golden-Goal Sudden Death."""
        my_goal_x = -ARENA_HALF_X + 75.0 if self.my_team == "red" else ARENA_HALF_X - 75.0  # Zone A Vault center!
        sb = self.scoreboard()

        # 1. Endgame raid check (last 25s when trailing/tied and opponent has high-value pieces in their goal)
        if not golden_goal and time_left_s <= 25.0 and sb["my_score"] <= sb["opp_score"]:
            opp_goal_pieces = [
                tp for tp in self.pieces.values()
                if tp.cls_name.startswith(self.opp_team + "_") and classify_goal_zone(tp.x_mm, tp.y_mm, self.opp_team) != "FIELD"
            ]
            raid_val = sum(tp.score_pts for tp in opp_goal_pieces)
            if raid_val >= 8:
                best_opp = max(opp_goal_pieces, key=lambda p: p.score_pts)
                return {
                    "action": "RAID_OPPONENT_GOAL",
                    "target_piece_id": best_opp.piece_id,
                    "target_class": best_opp.cls_name,
                    "expected_swing_pts": raid_val,
                    "target_xy_mm": [round(best_opp.x_mm, 1), round(best_opp.y_mm, 1)],
                }

        # 2. Score candidate own-team pieces outside my safe goal
        candidates = []
        for tp in self.pieces.values():
            if not tp.cls_name.startswith(self.my_team + "_"):
                continue
            if classify_goal_zone(tp.x_mm, tp.y_mm, self.my_team) in ("GOAL_A_SAFE", "GOAL_B_MID"):
                continue
            d_approach = math.hypot(tp.x_mm - pose_x, tp.y_mm - pose_y)
            d_push = math.hypot(my_goal_x - tp.x_mm, 0.0 - tp.y_mm)
            est_time_s = (d_approach + d_push) / 450.0 + 1.8  # Assume ~450 mm/s cruise + turn overhead
            # Golden goal: all pieces have equal value 1.0 -> strictly pick fastest goal!
            value = 1.0 if golden_goal else float(tp.score_pts)
            utility = (value * tp.conf) / max(0.5, est_time_s)
            candidates.append((utility, est_time_s, tp))

        if not candidates:
            return {"action": "PATROL_SCAN", "waypoint_xy_mm": [0.0, 0.0]}

        candidates.sort(key=lambda t: -t[0])
        best_u, est_t, bp = candidates[0]
        # Assign 3-lane drop slot in Zone A Wall Vault (y = -160 / 0 / +160 mm) based on existing fill
        slot_y = [-160.0, 0.0, 160.0][sb["my_goal_zones"]["GOAL_A_SAFE"] % 3]
        return {
            "action": "GOLDEN_BLITZ" if golden_goal else "HARVEST_HIGH_VALUE",
            "target_piece_id": bp.piece_id,
            "target_class": bp.cls_name,
            "score_pts": bp.score_pts,
            "utility": round(best_u, 3),
            "est_time_s": round(est_t, 1),
            "target_xy_mm": [round(bp.x_mm, 1), round(bp.y_mm, 1)],
            "drop_slot_xy_mm": [round(my_goal_x, 1), slot_y],
        }


def _self_test() -> None:
    print("=== 1. 4-ToF + IMU Ray-Cast Localization (with Opponent Occlusion on Left ToF) ===")
    true_x, true_y, true_theta = -420.0, 210.0, math.radians(28.0)
    loc = ArenaRayCastLocalizer(init_x_mm=-470.0, init_y_mm=175.0, init_theta_rad=true_theta)
    # Generate synthetic 4-ToF readings from true pose, then CORRUPT Left ToF (opponent car 350mm away!)
    dF = ray_distance_to_arena_wall(true_x, true_y, true_theta) - loc.offsets[0]
    dB = ray_distance_to_arena_wall(true_x, true_y, true_theta + math.pi) - loc.offsets[1]
    dL_occluded = 350.0  # Opponent car blocks Left ToF!
    dR = ray_distance_to_arena_wall(true_x, true_y, true_theta - math.pi / 2) - loc.offsets[3]
    for _ in range(4):
        st = loc.update_4tof(dF, dB, dL_occluded, dR)
    err = math.hypot(loc.x - true_x, loc.y - true_y)
    print(f"  True=({true_x}, {true_y}) -> Est=({st['x_mm']}, {st['y_mm']}) | err={err:.2f} mm | used={st['used_rays']} rej={st['rejected_rays']}")
    assert err < 5.0 and any(r[0] == "L" and r[1] == "opponent_occluded" for r in st["rejected_rays"])

    print("=== 2. Global World Map Projection, Scoreboard & Tactical Recommendations ===")
    wm = XiangqiWorldModel(my_team="red")
    # Frame 1: Robot at (-500, 0, 0 deg) sees Red General (10pt) at 600mm ahead & Red Pawn (1pt) at 300mm ahead
    fake_frame = {
        "cars": [{"world_x_mm": 400.0, "world_y_mm": 1200.0}],
        "pieces": [
            {"class": "red_general", "conf": 0.92, "world_x_mm": 50.0, "world_y_mm": 600.0},
            {"class": "red_pawn", "conf": 0.90, "world_x_mm": -30.0, "world_y_mm": 300.0},
            {"class": "black_general", "conf": 0.91, "world_x_mm": 0.0, "world_y_mm": 1900.0},  # Inside Black goal!
        ],
    }
    wm.ingest_vision_frame(-500.0, 0.0, 0.0, fake_frame, now_t=10.0)
    act_reg = wm.recommend_next_action(-500.0, 0.0, time_left_s=95.0, golden_goal=False)
    act_gg = wm.recommend_next_action(-500.0, 0.0, time_left_s=55.0, golden_goal=True)
    act_end = wm.recommend_next_action(-500.0, 0.0, time_left_s=18.0, golden_goal=False)
    print("  Regular Match (95s left) :", act_reg)
    print("  Golden Goal (55s left)   :", act_gg)
    print("  Endgame Raid (18s left)  :", act_end)
    assert act_reg["target_class"] == "red_general"
    assert act_gg["target_class"] == "red_pawn"
    assert act_end["action"] == "RAID_OPPONENT_GOAL"
    print("ALL WORLD MODEL SELF-TESTS PASSED!")


if __name__ == "__main__":
    _self_test()
