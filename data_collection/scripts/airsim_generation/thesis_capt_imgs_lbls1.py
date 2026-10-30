"""
AirSim multi-object capture with robust per-object centering:
- Captures all targets (by prefix) and all views (az/rad/elev).
- Keeps focus object centered by aiming at a per-object "visual center"
- Labels ALL visible objects in every image using segmentation -> YOLO boxes.
"""

import os, math, csv
from datetime import datetime
from pathlib import Path
import numpy as np
import airsim

SCENE       = "S0"
OUT_ROOT    = os.environ.get(
    "AIRSIM_CAPTURE_ROOT",
    str(Path(__file__).resolve().parents[2] / "raw_data" / "airsim_captures"),
)
VEHICLE     = "Rig"
CAMERA      = "0"
CAM_FOV_DEG = 65.0

TARGET_PREFIXES = [
    "SM_tent","SM_tank","SM_tower","SM_container","SM_whitevan",
    "SM_suv","SM_male","SM_rock","SM_barrel","SM_tree",
]

CLASS_BY_PREFIX = {
    "SM_tent":      0,
    "SM_tank":      1,
    "SM_tower":     2,
    "SM_container": 3,
    "SM_whitevan":  4,
    "SM_suv":       5,
    "SM_male":      6,
    "SM_rock":      7,
    "SM_barrel":    8,
    "SM_tree":      9,
}

ELEVATIONS = [
    {"name": "low",  "z_off": -2.0},
    {"name": "mid",  "z_off": -12.0},
    {"name": "high", "z_off": -22.0},
]

RADII = [
    {"name": "near", "radius": 10.0},
    {"name": "mid",  "radius": 16.0},
    {"name": "far",  "radius": 22.0},
]

AZ_LIST = list(range(0, 360, 45))  # 0,45,...,315

Z_NUDGE_UP = 0.6  # NED: negative z is up

CLASS_R_SCALE = { 0: 1.0, 1: 0.9, 2: 1.2, 3: 1.0 }
MIN_BBOX_FRAC = 0.005  # reject tiny specks

# For per-object visual-center calibration (object isolated)
FOCUS_CAL_AZ    = list(range(0, 360, 30))         # 12 angles
FOCUS_CAL_RADII = [6.0, 10.0, 16.0, 22.0, 30.0]   # near..far
FOCUS_CAL_ZOFF  = [-2.0, -8.0, -12.0, -18.0, -22.0]

# For rgb->instance_id calibration (color-coded only)
RGB_CAL_AZ    = list(range(0, 360, 45))
RGB_CAL_RADII = [8.0, 12.0, 18.0, 26.0]
RGB_CAL_ZOFF  = [-6.0, -12.0, -18.0]


# -----------------------
# UTILS
# -----------------------
def q_from_deg(pitch=0.0, roll=0.0, yaw=0.0):
    return airsim.to_quaternion(math.radians(pitch), math.radians(roll), math.radians(yaw))

def look_at_quat(src, dst):
    dx, dy, dz = dst.x_val - src.x_val, dst.y_val - src.y_val, dst.z_val - src.z_val
    yaw   = math.degrees(math.atan2(dy, dx))
    pitch = -math.degrees(math.atan2(dz, math.hypot(dx, dy)))  # NED: up is negative
    return q_from_deg(pitch=pitch, yaw=yaw), yaw, pitch

def finite_pose(p):
    try:
        v = p.position
        return all(map(math.isfinite, (v.x_val, v.y_val, v.z_val)))
    except Exception:
        return False

def list_targets_by_prefixes(client, prefixes):
    prefixes = [p.lower() for p in prefixes]
    names = client.simListSceneObjects(".*")
    picked, seen = [], set()
    for n in names:
        ln = n.lower()
        if any(ln.startswith(p) for p in prefixes) and n not in seen:
            pose = client.simGetObjectPose(n)
            if finite_pose(pose):
                picked.append((n, pose))
                seen.add(n)
    picked.sort(key=lambda t: t[0].lower())
    return picked

def class_id_from_object_name(name):
    ln = name.lower()
    for pref, cid in CLASS_BY_PREFIX.items():
        if ln.startswith(pref.lower()):
            return cid
    return None

def set_all_to_background(client):
    client.simSetSegmentationObjectID(".*", 0, True)
    client.simSetSegmentationObjectID("Landscape.*", 0, True)
    client.simSetSegmentationObjectID("SkySphere.*", 0, True)
    client.simSetSegmentationObjectID("BP_Sky.*", 0, True)

def set_actor_seg_id(client, actor_name, seg_id: int):
    client.simSetSegmentationObjectID(f"{actor_name}.*", seg_id, True)
    client.simSetSegmentationObjectID(f".*{actor_name}.*", seg_id, True)

def get_seg_image_uint8(seg_resp):
    if seg_resp.width == 0 or seg_resp.height == 0:
        return None
    img = np.frombuffer(seg_resp.image_data_uint8, dtype=np.uint8)
    return img.reshape(seg_resp.height, seg_resp.width, 3)

def dominant_nonzero_color(seg_img):
    flat = seg_img.reshape(-1, 3)
    nz = flat[np.any(flat != 0, axis=1)]
    if nz.size == 0:
        return None
    uniq, counts = np.unique(nz, axis=0, return_counts=True)
    return tuple(uniq[np.argmax(counts)].tolist())

def try_decode_grayscale_ids(seg_img):
    if np.array_equal(seg_img[...,0], seg_img[...,1]) and np.array_equal(seg_img[...,0], seg_img[...,2]):
        return seg_img[...,0].astype(np.int32)
    return None

def decode_instance_id_map(seg_img, rgb_to_instance_id):
    gray = try_decode_grayscale_ids(seg_img)
    if gray is not None:
        return gray

    H, W, _ = seg_img.shape
    id_map = np.zeros((H, W), dtype=np.int32)

    flat = seg_img.reshape(-1, 3)
    uniq = np.unique(flat, axis=0)
    for rgb in uniq:
        rgb_t = tuple(rgb.tolist())
        if rgb_t == (0,0,0):
            continue
        inst = rgb_to_instance_id.get(rgb_t, 0)
        if inst == 0:
            continue
        mask = (seg_img[...,0]==rgb[0]) & (seg_img[...,1]==rgb[1]) & (seg_img[...,2]==rgb[2])
        id_map[mask] = inst
    return id_map

def bboxes_from_id_map(id_map):
    H, W = id_map.shape
    inst_ids = np.unique(id_map)
    inst_ids = inst_ids[inst_ids != 0]

    out = {}
    for inst in inst_ids:
        ys, xs = np.where(id_map == inst)
        if xs.size == 0:
            continue
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        x_c = (x_min + x_max) / 2.0 / W
        y_c = (y_min + y_max) / 2.0 / H
        w_n = (x_max - x_min + 1) / W
        h_n = (y_max - y_min + 1) / H

        x_c = float(np.clip(x_c, 0.0, 1.0))
        y_c = float(np.clip(y_c, 0.0, 1.0))
        w_n = float(np.clip(w_n, 0.0, 1.0))
        h_n = float(np.clip(h_n, 0.0, 1.0))

        out[int(inst)] = (x_c, y_c, w_n, h_n)
    return out

def center_correction_from_bbox(x_min, y_min, x_max, y_max, W, H, fov_deg):
    bx = 0.5 * (x_min + x_max)
    by = 0.5 * (y_min + y_max)

    dx = bx - (W / 2.0)
    dy = by - (H / 2.0)

    nx = dx / (W / 2.0)
    ny = dy / (H / 2.0)

    hfov = math.radians(fov_deg)
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * (H / W))

    delta_yaw   = math.degrees(math.atan(nx * math.tan(hfov / 2.0)))
    delta_pitch = -math.degrees(math.atan(ny * math.tan(vfov / 2.0)))
    return delta_yaw, delta_pitch

def bbox_pixels_for_instance(id_map, inst_id):
    ys, xs = np.where(id_map == inst_id)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def refine_lookat_with_segmentation(client, cam_pos, target_point, focus_inst_id,
                                   rgb_to_instance_id, fov_deg,
                                   iters=2, tol_px=6, max_step_deg=8.0):
    """
    Keep camera position fixed; refine ONLY yaw/pitch so focus_inst bbox center -> image center.
    Returns (q_cam, yaw_deg, pitch_deg, success_bool)
    """
    # start from geometric look-at
    q_cam, yaw, pitch = look_at_quat(cam_pos, target_point)

    for _ in range(iters):
        client.simSetVehiclePose(airsim.Pose(cam_pos, q_from_deg()), True, vehicle_name=VEHICLE)
        client.simSetCameraPose(CAMERA, airsim.Pose(airsim.Vector3r(), q_cam), vehicle_name=VEHICLE)

        # IMPORTANT: flush 1 frame (AirSim/UE update latency)
        _ = client.simGetImages([airsim.ImageRequest(CAMERA, airsim.ImageType.Segmentation, False, False)],
                                vehicle_name=VEHICLE)[0]
        seg = client.simGetImages([airsim.ImageRequest(CAMERA, airsim.ImageType.Segmentation, False, False)],
                                  vehicle_name=VEHICLE)[0]

        seg_img = get_seg_image_uint8(seg)
        if seg_img is None:
            return q_cam, yaw, pitch, False

        id_map = decode_instance_id_map(seg_img, rgb_to_instance_id)
        H, W = id_map.shape

        bb = bbox_pixels_for_instance(id_map, focus_inst_id)
        if bb is None:
            # focus not visible in this view (occluded / too small / out of frame)
            return q_cam, yaw, pitch, False

        x_min, y_min, x_max, y_max = bb
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)

        dx = cx - (W / 2.0)
        dy = cy - (H / 2.0)

        if abs(dx) <= tol_px and abs(dy) <= tol_px:
            return q_cam, yaw, pitch, True

        dyaw, dpitch = center_correction_from_bbox(x_min, y_min, x_max, y_max, W, H, fov_deg)

        # clamp to avoid crazy jumps
        dyaw   = float(np.clip(dyaw,   -max_step_deg, max_step_deg))
        dpitch = float(np.clip(dpitch, -max_step_deg, max_step_deg))

        yaw   += dyaw
        pitch += dpitch
        q_cam = q_from_deg(pitch=pitch, yaw=yaw)

    return q_cam, yaw, pitch, True


def seg_bbox_pixels_nonblack(seg_img_uint8):
    fg = np.any(seg_img_uint8 != 0, axis=2)
    ys, xs = np.where(fg)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def assign_unique_instance_ids(client, targets, start_id=1):
    instance_id_by_name = {}
    class_id_by_instance = {}

    seg_id = start_id
    for name, _pose in targets:
        instance_id_by_name[name] = seg_id
        cid = class_id_from_object_name(name)
        class_id_by_instance[seg_id] = cid
        seg_id += 1

    set_all_to_background(client)
    for name, _pose in targets:
        set_actor_seg_id(client, name, instance_id_by_name[name])

    return instance_id_by_name, class_id_by_instance


# -----------------------
# (A) RGB->instance_id calibration (FC: only if color-coded
# -----------------------
def calibrate_rgb_to_instance_id(client, targets, instance_id_by_name):
    rgb_to_instance_id = {}

    # check if grayscale IDs
    test = client.simGetImages(
        [airsim.ImageRequest(CAMERA, airsim.ImageType.Segmentation, False, False)],
        vehicle_name=VEHICLE
    )[0]
    seg_img = get_seg_image_uint8(test)
    if seg_img is not None and try_decode_grayscale_ids(seg_img) is not None:
        print("Segmentation appears grayscale-coded (R=G=B=ID). Skipping RGB calibration.")
        return rgb_to_instance_id

    print("Segmentation appears color-coded. Calibrating rgb -> instance_id (one-time)...")

    for name, pose in targets:
        inst_id = instance_id_by_name[name]
        root = pose.position
        vc_guess = airsim.Vector3r(root.x_val, root.y_val, root.z_val - Z_NUDGE_UP)

        # isolate only this actor at its true instance id
        set_all_to_background(client)
        set_actor_seg_id(client, name, inst_id)

        found = None
        for z_off in RGB_CAL_ZOFF:
            if found is not None: break
            for r in RGB_CAL_RADII:
                if found is not None: break
                for az in RGB_CAL_AZ:
                    ang = math.radians(az % 360)
                    cam = airsim.Vector3r(
                        root.x_val + r * math.cos(ang),
                        root.y_val + r * math.sin(ang),
                        root.z_val + z_off
                    )
                    q_cam, _, _ = look_at_quat(cam, vc_guess)
                    client.simSetVehiclePose(airsim.Pose(cam, q_from_deg()), True, vehicle_name=VEHICLE)
                    client.simSetCameraPose(CAMERA, airsim.Pose(airsim.Vector3r(), q_cam), vehicle_name=VEHICLE)
                    seg_img = get_seg_stable(client)

                    if seg_img is None:
                        continue
                    dom = dominant_nonzero_color(seg_img) if seg_img is not None else None
                    if dom is not None:
                        found = dom
                        break

        if found is None:
            print(f"Calibration: no nonzero seg color found for {name}. (may be culled/hidden/tiny)")
        else:
            rgb_to_instance_id[found] = inst_id

    # restore full assignment
    set_all_to_background(client)
    for nm, _ in targets:
        set_actor_seg_id(client, nm, instance_id_by_name[nm])

    print(f"Calibrated {len(rgb_to_instance_id)} colors (out of {len(targets)} actors).")
    return rgb_to_instance_id


# -----------------------
# (B) Per-object "visual center" calibration (object isolated)
# -----------------------
def compute_focus_offset_once(client, actor_name, root_pos: airsim.Vector3r):
    """
    Returns world-space offset vector (Vector3r) to aim at instead of root_pos.
    Uses isolated segmentation pixels to find bbox center and compute ray correction.

    If fails, returns fallback (0,0,-Z_NUDGE_UP).
    """
    # isolate the actor with a fixed seg id (any nonzero)
    set_all_to_background(client)
    client.simSetSegmentationObjectID(f".*{actor_name}.*", 200, True)

    root = root_pos
    vc_guess = airsim.Vector3r(root.x_val, root.y_val, root.z_val - Z_NUDGE_UP)

    for z_off in FOCUS_CAL_ZOFF:
        for r in FOCUS_CAL_RADII:
            for az in FOCUS_CAL_AZ:
                ang = math.radians(az % 360)
                cam = airsim.Vector3r(
                    root.x_val + r * math.cos(ang),
                    root.y_val + r * math.sin(ang),
                    root.z_val + z_off
                )

                # aim at guess
                q_cam, _, _ = look_at_quat(cam, vc_guess)
                client.simSetVehiclePose(airsim.Pose(cam, q_from_deg()), True, vehicle_name=VEHICLE)
                client.simSetCameraPose(CAMERA, airsim.Pose(airsim.Vector3r(), q_cam), vehicle_name=VEHICLE)

                seg = client.simGetImages(
                    [airsim.ImageRequest(CAMERA, airsim.ImageType.Segmentation, False, False)],
                    vehicle_name=VEHICLE
                )[0]
                seg_img = get_seg_stable(client)
                if seg_img is None:
                    continue

                bb = seg_bbox_pixels_nonblack(seg_img)
                if bb is None:
                    continue

                x_min, y_min, x_max, y_max = bb
                H, W, _ = seg_img.shape

                # compute angular correction that would move bbox center to image center
                dyaw_deg, dpitch_deg = center_correction_from_bbox(x_min, y_min, x_max, y_max, W, H, CAM_FOV_DEG)

                # build forward/right/up purely from geometry (stable)
                cam_np = np.array([cam.x_val, cam.y_val, cam.z_val], dtype=np.float64)
                guess_np = np.array([vc_guess.x_val, vc_guess.y_val, vc_guess.z_val], dtype=np.float64)
                fwd = guess_np - cam_np
                nf = np.linalg.norm(fwd)
                if nf < 1e-9:
                    continue
                fwd = fwd / nf

                world_up = np.array([0.0, 0.0, -1.0], dtype=np.float64)  # NED up
                right = np.cross(fwd, world_up)
                nr = np.linalg.norm(right)
                if nr < 1e-9:
                    continue
                right = right / nr

                # rotate forward by yaw around world_up, then pitch around right
                def rodrigues(v, axis, ang):
                    axis = axis / (np.linalg.norm(axis) + 1e-12)
                    return (v * math.cos(ang) +
                            np.cross(axis, v) * math.sin(ang) +
                            axis * np.dot(axis, v) * (1 - math.cos(ang)))

                fwd1 = rodrigues(fwd, world_up, math.radians(dyaw_deg))
                fwd2 = rodrigues(fwd1, right,    math.radians(dpitch_deg))

                depth = np.linalg.norm(guess_np - cam_np)
                corrected_point = cam_np + depth * fwd2

                root_np = np.array([root.x_val, root.y_val, root.z_val], dtype=np.float64)
                offset = corrected_point - root_np

                # restore full segmentation IDs later in main
                return airsim.Vector3r(float(offset[0]), float(offset[1]), float(offset[2]))

    return airsim.Vector3r(0.0, 0.0, -Z_NUDGE_UP)
import time

def get_seg_stable(client, tries=6, sleep_s=0.03):
    """
    After camera/vehicle pose changes, the first seg frame can be stale/empty.
    This retries a few times until we see any non-black pixels.
    """
    last = None
    for _ in range(tries):
        seg = client.simGetImages(
            [airsim.ImageRequest(CAMERA, airsim.ImageType.Segmentation, False, False)],
            vehicle_name=VEHICLE
        )[0]
        seg_img = get_seg_image_uint8(seg)
        last = seg_img
        if seg_img is None:
            time.sleep(sleep_s)
            continue
        if np.any(seg_img != 0):   # any non-black pixel
            return seg_img
        time.sleep(sleep_s)
    return last  # could still be empty, but we tried

def compute_focus_point_from_ue_bounds(client, actor_name: str, root_pose: airsim.Pose,
                                      max_dist_m=4.5, min_points=2):
    """
    Compute a robust focus point using UE metadata exposed via AirSim:
    - Collect poses of scene objects that look like components of this actor
    - Keep only those within max_dist_m of the actor root (filters out accidental matches)
    - Build an AABB over those points and return its center
    Fallback: root position (nudged up)
    """
    root = root_pose.position
    root_np = np.array([root.x_val, root.y_val, root.z_val], dtype=np.float64)

    # Prefer strict prefix match first (usually safest)
    candidates = []
    for pat in (f"{actor_name}.*", f".*{actor_name}.*"):
        try:
            candidates = client.simListSceneObjects(pat)
        except Exception:
            candidates = []
        if candidates:
            break

    pts = []
    for cn in candidates:
        p = client.simGetObjectPose(cn)
        if not finite_pose(p):
            continue
        v = p.position
        npv = np.array([v.x_val, v.y_val, v.z_val], dtype=np.float64)

        # distance filter: keep only nearby components of the actor
        if np.linalg.norm(npv - root_np) <= max_dist_m:
            pts.append(npv)

    # not enough reliable UE component points -> fallback to root
    if len(pts) < min_points:
        return airsim.Vector3r(root.x_val, root.y_val, root.z_val - Z_NUDGE_UP), False

    pts = np.asarray(pts, dtype=np.float64)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    ctr = 0.5 * (mn + mx)

    return airsim.Vector3r(float(ctr[0]), float(ctr[1]), float(ctr[2])), True

def try_get_mesh_center_world(client, object_name):
    """
    Try to compute a better aim point using Unreal mesh geometry.
    Returns Vector3r world center OR None if unavailable.
    """
    try:
        resp = client.simGetMeshPositionVertexBuffers(object_name)
    except Exception:
        return None

    if resp is None:
        return None

    # AirSim can return a list OR a single response depending on version
    resps = resp if isinstance(resp, list) else [resp]
    if len(resps) == 0:
        return None

    # Pick the first buffer that actually has vertices
    best = None
    for r in resps:
        if hasattr(r, "vertices") and r.vertices and len(r.vertices) >= 3:
            best = r
            break
    if best is None:
        return None

    v = np.array(best.vertices, dtype=np.float64).reshape(-1, 3)
    vmin = v.min(axis=0)
    vmax = v.max(axis=0)
    center_local = 0.5 * (vmin + vmax)

    # best.position is typically the mesh position in world coordinates
    if hasattr(best, "position"):
        wp = best.position
        return airsim.Vector3r(
            float(wp.x_val + center_local[0]),
            float(wp.y_val + center_local[1]),
            float(wp.z_val + center_local[2]),
        )

    return None


# -----------------------
# MAIN
# -----------------------
def main():
    stamp   = f"{SCENE}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = os.path.join(OUT_ROOT, stamp)
    img_dir = os.path.join(out_dir, "images")
    lbl_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    manifest_csv = os.path.join(out_dir, "manifest.csv")

    print("Saving images to:", img_dir)
    print("Writing manifest:", manifest_csv)

    client = airsim.VehicleClient()
    client.confirmConnection()

    try:
        client.simSetCameraFov(CAMERA, CAM_FOV_DEG, vehicle_name=VEHICLE)
    except Exception:
        pass

    targets = list_targets_by_prefixes(client, TARGET_PREFIXES)
    if not targets:
        print(f"No scene objects found for prefixes: {TARGET_PREFIXES}")
        return

    # Assign unique IDs for ALL objects once
    instance_id_by_name, class_id_by_instance = assign_unique_instance_ids(client, targets, start_id=1)
    print("Sanity check: does segmentation show ANY objects?")

    seg_img = get_seg_stable(client)   # uses the helper we added
    if seg_img is None:
        print("Segmentation image is None")
    else:
        non_black = int(np.sum(np.any(seg_img != 0, axis=2)))
        print(f"Non-black pixels in segmentation: {non_black}")

        # Optional: print unique colors (limit to first few)
        uniq = np.unique(seg_img.reshape(-1, 3), axis=0)
        print("Unique seg colors (first 10):", uniq[:10])
    
    # --- Focus policy (one-time) ---
    print("ℹ Building focus points (mesh -> UE bounds -> seg offset -> root)...")

    focus_point_by_name = {}
    focus_mode_by_name = {}
    focus_offset_by_name = {}

    # 1) Mesh center (best when available)
    for name, pose in targets:
        mesh_center = try_get_mesh_center_world(client, name)
        if mesh_center is not None:
            focus_point_by_name[name] = mesh_center
            focus_mode_by_name[name] = "MESH"
        else:
            focus_point_by_name[name] = None
            focus_mode_by_name[name] = "NONE"

    # 2) UE bounds (fallback if mesh missing)
    for name, pose in targets:
        if focus_point_by_name[name] is not None:
            continue
        fp, ok = compute_focus_point_from_ue_bounds(
            client, name, pose,
            max_dist_m=4.5,     # IMPORTANT for vans/long actors
            min_points=2
        )
        if ok:
            focus_point_by_name[name] = fp
            focus_mode_by_name[name] = "UE_BOUNDS"

    # 3) Segmentation-derived visual-center offset (fallback)
    # NOTE: this function modifies segmentation IDs internally (isolating the object),
    # so we compute offsets now, then restore the full seg assignment afterwards.
    for name, pose in targets:
        if focus_point_by_name[name] is not None:
            continue
        focus_offset_by_name[name] = compute_focus_offset_once(client, name, pose.position)
        focus_mode_by_name[name] = "SEG_OFFSET"

    # Restore full segmentation assignment BEFORE rgb calibration
    set_all_to_background(client)
    for nm, _ in targets:
        set_actor_seg_id(client, nm, instance_id_by_name[nm])

    # Now do rgb->instance calibration (needs full scene IDs)
    rgb_to_instance_id = calibrate_rgb_to_instance_id(client, targets, instance_id_by_name)

    # Optional: quick log for debugging (first ~10)
    for i, (nm, _) in enumerate(targets[:10]):
        print(f"{nm}: focus={focus_mode_by_name.get(nm,'?')}")


    # Restore full segmentation assignment (ALL objects visible in seg)
    set_all_to_background(client)
    for nm, _ in targets:
        set_actor_seg_id(client, nm, instance_id_by_name[nm])

    fieldnames = [
        "scene_id", "episode_id", "focus_object", "file_name", "view_idx",
        "cam_x", "cam_y", "cam_z", "cam_yaw", "cam_pitch",
        "tgt_x", "tgt_y", "tgt_z",
        "azimuth_deg", "azimuth_bin",
        "radius_m", "radius_name", "radius_bin",
        "elevation_z_off", "elevation_name", "elevation_bin",
    ]

    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for focus_name, pose in targets:
            root = pose.position
            episode_id = f"{SCENE}_{focus_name}"
            focus_cid = class_id_from_object_name(focus_name)
            r_scale = CLASS_R_SCALE.get(focus_cid, 1.0)

            # Choose final aim point (vc)
            if focus_mode_by_name.get(focus_name) in ("MESH", "UE_BOUNDS"):
                vc = focus_point_by_name[focus_name]
            elif focus_mode_by_name.get(focus_name) == "SEG_OFFSET":
                off = focus_offset_by_name.get(focus_name, airsim.Vector3r(0.0, 0.0, -Z_NUDGE_UP))
                vc = airsim.Vector3r(root.x_val + off.x_val,
                                     root.y_val + off.y_val,
                                     root.z_val + off.z_val)
            else:
                vc = airsim.Vector3r(root.x_val, root.y_val, root.z_val - Z_NUDGE_UP)



            view_idx = 0
            for elev_idx, elev in enumerate(ELEVATIONS):
                cz = root.z_val + elev["z_off"]

                for rad_idx, rad in enumerate(RADII):
                    r = rad["radius"] * r_scale

                    for az_bin, az in enumerate(AZ_LIST):
                        ang = math.radians(az % 360)
                        cam = airsim.Vector3r(
                            root.x_val + r * math.cos(ang),
                            root.y_val + r * math.sin(ang),
                            cz
                        )

                        # Always look at visual center point
                        focus_inst = instance_id_by_name[focus_name]

                        q_cam, yaw, pitch, ok = refine_lookat_with_segmentation(
                            client=client,
                            cam_pos=cam,
                            target_point=vc,              # can still be your pivot/visual-center guess
                            focus_inst_id=focus_inst,
                            rgb_to_instance_id=rgb_to_instance_id,
                            fov_deg=CAM_FOV_DEG,
                            iters=2
                        )

                        client.simSetVehiclePose(airsim.Pose(cam, q_from_deg()), True, vehicle_name=VEHICLE)
                        client.simSetCameraPose(CAMERA, airsim.Pose(airsim.Vector3r(), q_cam), vehicle_name=VEHICLE)

                        # capture RGB + Seg in one call
                        rgb_resp, seg_resp = client.simGetImages([
                            airsim.ImageRequest(CAMERA, airsim.ImageType.Scene, False, True),
                            airsim.ImageRequest(CAMERA, airsim.ImageType.Segmentation, False, False),
                        ], vehicle_name=VEHICLE)

                        fname = (
                            f"{SCENE}-{focus_name}"
                            f"-el{elev['name']}"
                            f"-rad{rad['name']}"
                            f"-az{int(az):03d}.png"
                        )

                        airsim.write_file(os.path.join(img_dir, fname), rgb_resp.image_data_uint8)

                        # labels for ALL objects present in seg
                        seg_img = get_seg_image_uint8(seg_resp)
                        yolo_lines = []
                        if seg_img is not None:
                            id_map = decode_instance_id_map(seg_img, rgb_to_instance_id)
                            bboxes = bboxes_from_id_map(id_map)

                            for inst_id, (x_c, y_c, w_n, h_n) in bboxes.items():
                                cid = class_id_by_instance.get(inst_id, None)
                                if cid is None:
                                    continue
                                if w_n > MIN_BBOX_FRAC and h_n > MIN_BBOX_FRAC:
                                    yolo_lines.append(f"{cid} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

                        label_path = os.path.join(lbl_dir, fname.replace(".png", ".txt"))
                        with open(label_path, "w", encoding="utf-8") as lf:
                            if yolo_lines:
                                lf.write("\n".join(yolo_lines))

                        w.writerow({
                            "scene_id": SCENE,
                            "episode_id": episode_id,
                            "focus_object": focus_name,
                            "file_name": fname,
                            "view_idx": view_idx,
                            "cam_x": f"{cam.x_val:.6f}",
                            "cam_y": f"{cam.y_val:.6f}",
                            "cam_z": f"{cam.z_val:.6f}",
                            "cam_yaw": f"{yaw:.6f}",
                            "cam_pitch": f"{pitch:.6f}",
                            "tgt_x": f"{vc.x_val:.6f}",
                            "tgt_y": f"{vc.y_val:.6f}",
                            "tgt_z": f"{vc.z_val:.6f}",
                            "azimuth_deg": int(az),
                            "azimuth_bin": az_bin,
                            "radius_m": r,
                            "radius_name": rad["name"],
                            "radius_bin": rad_idx,
                            "elevation_z_off": elev["z_off"],
                            "elevation_name": elev["name"],
                            "elevation_bin": elev_idx,
                        })

                        view_idx += 1

    print("   Done.")
    print("   Images  :", img_dir)
    print("   Labels  :", lbl_dir)
    print("   Manifest:", manifest_csv)


if __name__ == "__main__":
    main()
