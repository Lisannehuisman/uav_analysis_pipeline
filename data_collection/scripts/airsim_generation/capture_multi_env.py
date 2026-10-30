# capture_multi_anchor.py
# Multi-object capture like S0, but loop over *all* actors as anchors.
# For each anchor: orbit its true pose; capture RGB+labels for ALL objects in view.

import os, json, argparse, math, re
import numpy as np
import airsim

# ---------- helpers--------------------------------------

def ensure_dirs(*paths):
    for p in paths: os.makedirs(p, exist_ok=True)

def connect(preferred_vehicle="Rig"):
    client = airsim.VehicleClient(); client.confirmConnection()
    try:
        vehicles = client.listVehicles()
    except Exception:
        try: vehicles = client.simListVehicles()
        except Exception: vehicles = []
    vehicle = preferred_vehicle if preferred_vehicle in vehicles else (vehicles[0] if vehicles else "")
    return client, vehicle

def pol2cart(radius, az_deg):
    r = math.radians(az_deg)
    return radius * math.cos(r), radius * math.sin(r)

def look_at_yaw_pitch(from_xyz, to_xyz):
    dx = to_xyz[0] - from_xyz[0]
    dy = to_xyz[1] - from_xyz[1]
    dz = to_xyz[2] - from_xyz[2]          # positive if target above camera (z-up math)
    yaw   = math.degrees(math.atan2(dy, dx))
    dist  = math.hypot(dx, dy)
    pitch = math.degrees(math.atan2(dz, dist))  # negative when target below cam -> looks down in AirSim
    return max(-180, min(180, yaw)), max(-89, min(89, pitch))

def set_pose(client, vehicle, xyz, pitch_deg, yaw_deg, z_is_up=True):
    quat = airsim.to_quaternion(math.radians(pitch_deg), 0.0, math.radians(yaw_deg))
    pos  = airsim.Vector3r(xyz[0], xyz[1], -xyz[2] if z_is_up else xyz[2])  # NED
    client.simSetVehiclePose(airsim.Pose(pos, quat), True, vehicle_name=vehicle)

def grab_rgb_seg(client, camera="0", vehicle=""):
    req = [
        airsim.ImageRequest(camera, airsim.ImageType.Scene, False, False),
        airsim.ImageRequest(camera, airsim.ImageType.Segmentation, False, False),
    ]
    res = client.simGetImages(req, vehicle_name=vehicle)
    rgb = np.frombuffer(res[0].image_data_uint8, dtype=np.uint8).reshape(res[0].height, res[0].width, 3)
    seg = np.frombuffer(res[1].image_data_uint8, dtype=np.uint8).reshape(res[1].height, res[1].width, 3)[:, :, 0]
    return rgb, seg

def init_manifest():
    import pandas as pd
    cols = [
        "env_id","scene_id","anchor_name","img_path","label_path","width","height",
        "drone_id","cam_id","azimuth_deg","elev_m","radius_m","pitch_deg","yaw_deg","fov_deg",
        "time_of_day","weather","sun_elev_deg","sun_az_deg",
        "jpeg_q","noise_level",
        "n_objects","class_ids","instance_ids","per_box_truncated","per_box_area_px"
    ]
    return pd.DataFrame(columns=cols)

def append_manifest(df, row):
    import pandas as pd
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

def assign_seg_ids_by_prefix(client, classmap):
    """Set segmentation IDs for *all* objects by startswith(prefix)."""
    client.simSetSegmentationObjectID(".*", 0, True)
    names = client.simListSceneObjects(".*")
    for pref, cid in classmap.items():
        pat = f"^{re.escape(pref)}.*"
        # Apply to any object that *starts with* this prefix
        client.simSetSegmentationObjectID(pat, int(cid), True)

def discover_all_targets(client, prefixes):
    """Return list of (name, pose) for all actors whose name starts with any prefix."""
    names = client.simListSceneObjects(".*")
    out = []
    for n in sorted(names, key=str.lower):
        ln = n.lower()
        if any(ln.startswith(p.lower()) for p in prefixes):
            out.append((n, client.simGetObjectPose(n)))
    return out

def bboxes_from_seg_multi(seg_img, valid_ids, min_area_px=25):
    """Return boxes as list of (class_id, x, y, w, h, area)."""
    import cv2
    boxes = []
    H, W = seg_img.shape[:2]
    for cid in valid_ids:
        mask = (seg_img == int(cid)).astype(np.uint8)
        if mask.sum() < min_area_px: continue
        n, labels = cv2.connectedComponents(mask, connectivity=8)
        for i in range(1, n):
            m = (labels == i).astype(np.uint8)
            if m.sum() < min_area_px: continue
            ys, xs = np.where(m)
            x0, y0, x1, y1 = xs.min(), ys.min(), xs.max(), ys.max()
            boxes.append((int(cid), x0, y0, x1-x0+1, y1-y0+1, int(m.sum())))
    return boxes

def yolo_line_from_px_bbox(px_bbox, img_w, img_h):
    x, y, w, h = px_bbox
    cx = (x + w/2) / img_w
    cy = (y + h/2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh

# ---------- main -------------------------------------------------------------

def main(args):
    # ---- identical viewpoint grid to S0 ----
    heights_m = [5, 14, 20]
    az_list   = list(range(0, 360, 60))
    radii_m   = [5, 16]

    cam_id      = "0"
    fov_deg     = 70.0
    time_of_day = "noon"
    weather     = "clear"
    sun_elev_deg, sun_az_deg = 50.0, 120.0

    out_img = os.path.join(args.out, "images")
    out_lbl = os.path.join(args.out, "labels")
    ensure_dirs(out_img, out_lbl)

    client, vehicle = connect(preferred_vehicle="Rig")
    print("Connected.")
    try:
        print("Vehicles in sim:", client.listVehicles(), "| Using:", vehicle if vehicle else "(none)")
    except Exception:
        pass

    # Load classmap and set segmentation IDs for all objects
    with open(args.classmap, "r") as f:
        classmap = json.load(f)
    assign_seg_ids_by_prefix(client, classmap)
    id_whitelist = [int(v) for v in classmap.values()]
    prefixes = list(classmap.keys())

    # Discover all actors
    targets = discover_all_targets(client, prefixes)
    print("Anchors discovered (by prefix):")
    from collections import defaultdict
    grp = defaultdict(list)
    for n,_ in targets:
        key = next((p for p in classmap.keys() if n.lower().startswith(p.lower())), "OTHER")
        grp[key].append(n)
    for k,v in grp.items():
        print(f"  {k}: {len(v)} -> {v}")
    print("TOTAL anchors:", len(targets))
    
    
    if not targets:
        raise RuntimeError("No actors found for your prefixes.")
    print(f"Anchors discovered: {len(targets)}")

    manifest = init_manifest()
    scene_idx = 0

    for anchor_name, pose in targets:
        root = pose.position
        anchor_xyz = (root.x_val, root.y_val, -root.z_val)
        print(f"\n[{args.env_id}] Anchor START: {anchor_name} at {anchor_xyz}")

        for r in radii_m:
            for h in heights_m:
                for az in az_list:
                    dx, dy = pol2cart(r, az)
                    cam_xyz = (anchor_xyz[0]+dx, anchor_xyz[1]+dy, h)
                    yaw_deg, pitch_deg = look_at_yaw_pitch(cam_xyz, anchor_xyz)
                    set_pose(client, vehicle, cam_xyz, pitch_deg, yaw_deg)
                    airsim.time.sleep(0.03)

                    rgb, seg = grab_rgb_seg(client, camera=cam_id, vehicle=vehicle)
                    H, W = rgb.shape[:2]

                    # labels for ALL visible classes
                    boxes = bboxes_from_seg_multi(seg, id_whitelist, min_area_px=25)

                    lines, cls_ids, inst_ids, areas, trunc = [], [], [], [], []
                    for idx, (sid, x, y, w, hh, area) in enumerate(boxes):
                        cxn, cyn, nw, nh = yolo_line_from_px_bbox((x, y, w, hh), W, H)
                        if nw <= 0.005 or nh <= 0.005:
                            continue
                        lines.append(f"{sid} {cxn:.6f} {cyn:.6f} {nw:.6f} {nh:.6f}")
                        cls_ids.append(int(sid)); inst_ids.append(idx); areas.append(int(area))
                        trunc.append(bool(x<=1 or y<=1 or (x+w)>=W-2 or (y+hh)>=H-2))

                    import cv2
                    fname = f"{args.env_id}_scene{scene_idx:04d}_{anchor_name}_H{heights_m.index(h)+1}_AZ{az:03d}_R{'near' if r==radii_m[0] else 'far'}.png"
                    ipath = os.path.join(out_img, fname)
                    lpath = os.path.join(out_lbl, fname.replace(".png",".txt"))
                    cv2.imwrite(ipath, rgb)
                    with open(lpath, "w") as f:
                        f.write("\n".join(lines))

                    row = dict(
                        env_id=args.env_id, scene_id=f"{args.env_id}_{scene_idx:04d}",
                        anchor_name=anchor_name,
                        img_path=ipath, label_path=lpath, width=W, height=H,
                        drone_id=(vehicle or "Rig"), cam_id=cam_id,
                        azimuth_deg=az, elev_m=h, radius_m=r,
                        pitch_deg=pitch_deg, yaw_deg=yaw_deg, fov_deg=fov_deg,
                        time_of_day=time_of_day, weather=weather,
                        sun_elev_deg=sun_elev_deg, sun_az_deg=sun_az_deg,
                        jpeg_q=100, noise_level=0,
                        n_objects=len(lines),
                        class_ids=json.dumps(cls_ids),
                        instance_ids=json.dumps(inst_ids),
                        per_box_truncated=json.dumps(trunc),
                        per_box_area_px=json.dumps(areas)
                    )
                    manifest = append_manifest(manifest, row)
                    scene_idx += 1

        print(f"[{args.env_id}] Anchor DONE: {anchor_name}")

    import pandas as pd
    manifest.to_csv(os.path.join(args.out, "manifest.csv"), index=False)
    print(f"\n[{args.env_id}] Done: {args.out}")

# ---------- CLI --------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--env_id", required=True, choices=["S1","S2","S3","S4"])
    ap.add_argument("--classmap", required=True)
    args = ap.parse_args()
    main(args)
