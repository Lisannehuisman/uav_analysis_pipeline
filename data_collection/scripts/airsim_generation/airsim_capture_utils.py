import re, os, cv2, json, math, time
import numpy as np, pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm
import airsim

# --- find all scene objects whose name starts with any prefix ---

def list_targets_by_prefixes(client, prefixes):
    """Return [(name, pose), ...]; match if prefix appears anywhere (case-insensitive)."""
    names = client.simListSceneObjects(".*")
    picked = []
    pats = [re.compile(re.escape(p), re.IGNORECASE) for p in prefixes]
    for n in names:
        if any(p.search(n) for p in pats):
            pose = client.simGetObjectPose(n)
            picked.append((n, pose))
    picked.sort(key=lambda t: t[0].lower())
    return picked

def class_id_from_object_name(name, classmap: dict):
    """Return class id if any key from classmap appears in name (case-insensitive)."""
    ln = name.lower()
    # Prefer exact startswith; fallback to substring
    for pref, cid in classmap.items():
        if ln.startswith(pref.lower()):
            return cid
    for pref, cid in classmap.items():
        if pref.lower() in ln:
            return cid
    return None

# set all segmentation IDs to background (0)
def set_all_to_background(client):
    client.simSetSegmentationObjectID(".*", 0, True)
    client.simSetSegmentationObjectID("Landscape.*", 0, True)
    client.simSetSegmentationObjectID("SkySphere.*", 0, True)
    client.simSetSegmentationObjectID("BP_Sky.*", 0, True)



# ---------- math ----------
def yaw_deg_from_vec(dx, dy):
    return math.degrees(math.atan2(dy, dx))

def look_at_yaw_pitch(from_xyz, to_xyz):
    dx = to_xyz[0] - from_xyz[0]
    dy = to_xyz[1] - from_xyz[1]
    dz = to_xyz[2] - from_xyz[2]   # positive if target is above camera (Z-up convention here)
    yaw = math.degrees(math.atan2(dy, dx))
    dist_xy = math.hypot(dx, dy)
    # AirSim (to_quaternion) expects pitch<0 to look DOWN.
    pitch = math.degrees(math.atan2(dz, dist_xy))   # <-- no minus
    return yaw, pitch

def pol2cart(radius, az_deg):
    rad = math.radians(az_deg)
    return radius*math.cos(rad), radius*math.sin(rad)

# ---------- AirSim ----------
def connect(preferred_vehicle: str | None = None):
    client = airsim.VehicleClient()
    client.confirmConnection()
    # Try to list vehicles (API name differs by AirSim versions)
    vehicles = []
    try:
        vehicles = client.listVehicles()
    except Exception:
        try:
            vehicles = client.simListVehicles()
        except Exception:
            vehicles = []

    # pick vehicle
    vehicle = None
    if preferred_vehicle and preferred_vehicle in vehicles:
        vehicle = preferred_vehicle
    elif vehicles:
        vehicle = vehicles[0]  # first available
    else:
        # No vehicles listed. In CV mode this can still work with empty vehicle_name.
        vehicle = ""  # fall back to empty (AirSim accepts this in many sim* APIs)

    # IMPORTANT: do NOT call enableApiControl in ComputerVision mode
    return client, vehicle

def set_pose(client, vehicle, xyz, pitch_deg, yaw_deg, z_is_up=True):
    pitch = math.radians(pitch_deg); yaw = math.radians(yaw_deg); roll = 0.0
    quat = airsim.to_quaternion(pitch, roll, yaw)
    pos = airsim.Vector3r(xyz[0], xyz[1], -xyz[2] if z_is_up else xyz[2])  # NED
    # pass vehicle even if it's "", AirSim accepts this in CV mode
    client.simSetVehiclePose(airsim.Pose(pos, quat), True, vehicle_name=vehicle)

def grab_rgba_and_seg(client, camera="0", vehicle=""):
    res = client.simGetImages([
        airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False),
        airsim.ImageRequest(camera, airsim.ImageType.Segmentation, False, False)
    ], vehicle_name=vehicle)
    if len(res) != 2: raise RuntimeError("simGetImages returned !=2")
    # Scene
    img1d = np.frombuffer(res[0].image_data_uint8, dtype=np.uint8)
    scene = img1d.reshape(res[0].height, res[0].width, 3)
    # Segmentation: use red channel
    seg1d = np.frombuffer(res[1].image_data_uint8, dtype=np.uint8)
    seg = seg1d.reshape(res[1].height, res[1].width, 3)[:, :, 0]
    return scene, seg

# ---------- seg->YOLO ----------
def bboxes_from_seg(seg_img: np.ndarray, id_whitelist: List[int], min_area_px=20):
    """
    Return list of tuples: (sid, x, y, w, h, area) in PIXELS.
    Uses connected components so multiple instances per class id are supported.
    """
    h, w = seg_img.shape[:2]
    boxes = []
    for sid in id_whitelist:
        mask = (seg_img == sid).astype(np.uint8)
        if mask.sum() < min_area_px: 
            continue
        n, labels = cv2.connectedComponents(mask, connectivity=8)
        for i in range(1, n):
            m = (labels == i).astype(np.uint8)
            if m.sum() < min_area_px: 
                continue
            ys, xs = np.where(m)
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            boxes.append((sid, x0, y0, x1 - x0 + 1, y1 - y0 + 1, int(m.sum())))
    return boxes

def yolo_line_from_px_bbox(px_bbox, img_w, img_h):
    x, y, w, h = px_bbox
    cx = (x + w/2) / img_w
    cy = (y + h/2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

# ---------- seg IDs ----------
def assign_seg_ids_by_prefix(client, classmap: dict, reset_all=False):
    if reset_all:
        client.simSetSegmentationObjectID(".*", 0, True)
    for pref, cid in classmap.items():
        # apply to any object whose name contains the token
        client.simSetSegmentationObjectID(f".*{re.escape(pref)}.*", cid, True)


# ---------- manifest ----------
def init_manifest():
    cols = [
        "env_id","scene_id","img_path","label_path","width","height",
        "drone_id","cam_id","azimuth_deg","elev_m","radius_m","pitch_deg","yaw_deg","fov_deg",
        "time_of_day","weather","sun_elev_deg","sun_az_deg",
        "jpeg_q","noise_level",
        "n_objects","class_ids","instance_ids","per_box_truncated","per_box_area_px"
    ]
    return pd.DataFrame(columns=cols)

def append_manifest(df, row_dict):
    return pd.concat([df, pd.DataFrame([row_dict])], ignore_index=True)
