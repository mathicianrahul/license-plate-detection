"""
=============================================================
COMPLETE 13-PHASE ANPR PIPELINE — FIXED PLATE DETECTION
=============================================================
Key fix: fast-alpr now runs on the FULL FRAME (not crop).
  → Plates mapped to vehicles by bounding-box containment.
  → Thresholds lowered so real detections pass the gate.
  → OCR correction simplified to avoid mangling valid text.
=============================================================
"""

import cv2, re, os, csv, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from ultralytics import YOLO
from fast_alpr import ALPR

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
BASE_DIR        = "D:/Number plate Task"
TEST_VIDEOS_DIR = f"{BASE_DIR}/test_videos"
OUTPUT_DIR      = f"{BASE_DIR}/op 2"
OUTPUT_CSV      = f"{OUTPUT_DIR}/results.csv"

TRAINED_MODEL        = f"{BASE_DIR}/runs/train_anpr/weights/best.pt"
FALLBACK_MODEL       = f"{BASE_DIR}/yolov8n.pt"
VEHICLE_CLASSES_COCO = [2, 3, 5, 7]   # car, motorcycle, bus, truck

CONF_THRESHOLD     = 0.25   # vehicle detection min confidence
MIN_FRAMES_FOR_OCR = 3      # min frame readings before voting
MIN_CONFIDENCE     = 20     # ← lowered: plates ≥20% pass the gate
MAX_INTERP_GAP     = 8      # max frames to interpolate

# Indian plate: XX00XX0000  (spaces stripped before matching)
PLATE_RE_INDIAN = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$')
PLATE_RE_GLOBAL = re.compile(r'^[A-Z0-9]{4,12}$')

CSV_FIELDS = ["Video","Bike ID","Plate Number","Speed",
              "Confidence","Detection Quality","Frame"]

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════
# PHASE 0 — MODEL SETUP
# ═══════════════════════════════════════════════════════════
def setup_models():
    print("\n🔵 PHASE 0 — Model Setup")
    if os.path.exists(TRAINED_MODEL):
        print(f"   ✅ Trained model: {TRAINED_MODEL}")
        vehicle_model  = YOLO(TRAINED_MODEL)
        detect_classes = [0]
    else:
        print(f"   ⚠️  COCO fallback: {FALLBACK_MODEL}")
        vehicle_model  = YOLO(FALLBACK_MODEL)
        detect_classes = VEHICLE_CLASSES_COCO
    print("   ✅ Vehicle detector ready")

    print("   Loading fast-alpr ONNX...")
    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-xs-v2-global-model",
    )
    print("   ✅ fast-alpr ready")
    return vehicle_model, detect_classes, alpr


# ═══════════════════════════════════════════════════════════
# PHASE 6 — OCR CORRECTION  (simplified — no mangling)
# ═══════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    """Strip spaces/dashes and uppercase."""
    return re.sub(r'[^A-Z0-9]', '', text.upper())


def correct_ocr(text: str) -> str:
    """
    Light position-aware correction only when text looks like
    a full Indian plate (≥8 chars). Otherwise just clean.
    """
    t = clean_text(text)
    if len(t) < 8:
        return t
    out = list(t)
    for i, ch in enumerate(out):
        if i in (2, 3, 6, 7, 8, 9):          # digit positions
            out[i] = {'O':'0','I':'1','B':'8',
                      'S':'5','Z':'2','Q':'0'}.get(ch, ch)
        elif i in (0, 1, 4, 5):               # letter positions
            out[i] = {'0':'O','1':'I','8':'B'}.get(ch, ch)
    return "".join(out)


# ═══════════════════════════════════════════════════════════
# PHASE 7 — VALIDATION + VOTING
# ═══════════════════════════════════════════════════════════
def is_valid(text: str) -> bool:
    if not text or len(text) < 4:
        return False
    t = clean_text(text)
    return bool(PLATE_RE_INDIAN.match(t) or PLATE_RE_GLOBAL.match(t))


def vote_plate(readings: list):
    """
    readings: [(text, conf), ...]
    Returns (best_text, avg_ocr_conf, consistency_ratio)
    """
    cleaned = [(clean_text(t), c) for t, c in readings]
    valid   = [(t, c) for t, c in cleaned if is_valid(t)]
    if not valid:
        return None, 0.0, 0.0

    scores = defaultdict(float)
    counts = defaultdict(int)
    for t, c in valid:
        scores[t] += c
        counts[t] += 1

    total = len(readings)
    # Try strict 70%, relax to 25% if needed
    for thresh in (0.70, 0.50, 0.25):
        pool = {t: c for t, c in counts.items()
                if c / max(total, 1) >= thresh}
        if pool:
            break
    if not pool:
        return None, 0.0, 0.0

    best = max(pool, key=lambda t: scores[t])
    return best, scores[best] / counts[best], counts[best] / max(total, 1)


# ═══════════════════════════════════════════════════════════
# PHASE 8 — DETECTION QUALITY
# ═══════════════════════════════════════════════════════════
def classify_quality(plate_img, pbox, fshape) -> str:
    if plate_img is None or plate_img.size == 0:
        return "Occluded"
    fh, fw = fshape[:2]
    x1, y1, x2, y2 = [int(v) for v in pbox]
    if x1 <= 2 or y1 <= 2 or x2 >= fw-2 or y2 >= fh-2:
        return "Occluded"
    gray = (cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            if len(plate_img.shape) == 3 else plate_img)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    area = (x2-x1)*(y2-y1)
    if area  < 300:  return "Partial"
    if blur  < 50:   return "Blurred"
    if blur >= 100:  return "Clear"
    return "Partial"


# ═══════════════════════════════════════════════════════════
# PHASE 9 — CONFIDENCE + SPEED
# ═══════════════════════════════════════════════════════════
def compute_conf(det, ocr, cons) -> int:
    adj_ocr = min(ocr * 1.5, 1.0)
    adj_det = min(det * 1.2, 1.0)
    return round(min((adj_det*0.4 + adj_ocr*0.4 + cons*0.2) * 100, 100))


# ═══════════════════════════════════════════════════════════
# PHASE 13 — ANTI-HALLUCINATION GATE
# ═══════════════════════════════════════════════════════════
def safe_output(plate, conf: int) -> str:
    if not plate or not is_valid(plate):
        return "Not Detected"
    if conf < MIN_CONFIDENCE:
        return "Not Detected"
    return plate


# ═══════════════════════════════════════════════════════════
# PHASE 11 — OVERLAY  (matches reference screenshots)
# ═══════════════════════════════════════════════════════════
def draw_overlay(frame, vbox, bid, plate_text, conf_pct,
                 speed, quality, pbox_abs=None):
    x1, y1, x2, y2 = [int(v) for v in vbox]
    fh, fw = frame.shape[:2]

    # Vehicle box — blue
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 130, 0), 2)

    # ── Tight GREEN box around plate ──────────────────────
    if pbox_abs:
        px1, py1, px2, py2 = [int(v) for v in pbox_abs]
        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 60), 2)
        if plate_text and plate_text != "Not Detected":
            # Small white text inside the green plate box
            cv2.putText(frame, plate_text,
                        (px1+3, py2-4),
                        FONT, 0.5, (255,255,255), 1, cv2.LINE_AA)

    # ── Large label ABOVE vehicle ─────────────────────────
    if plate_text and plate_text != "Not Detected":
        big   = f"{plate_text}  {conf_pct:.0f}%"
        color = (255, 255, 255)
    else:
        big   = f"ID:{bid}  No Plate"
        color = (80, 100, 255)

    scale, thick = 1.1, 2
    (tw, th), bl = cv2.getTextSize(big, FONT_BOLD, scale, thick)
    pad = 8
    bx1 = max(0, x1)
    bx2 = min(fw, bx1 + tw + pad*2)
    by2 = max(th + pad*2, y1)
    by1 = max(0, by2 - th - pad*2)

    bg = frame.copy()
    cv2.rectangle(bg, (bx1, by1), (bx2, by2), (10,10,10), -1)
    cv2.addWeighted(bg, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, big, (bx1+pad, by2-bl-pad//2),
                FONT_BOLD, scale, color, thick, cv2.LINE_AA)

    # HUD strip
    strip_y = min(y2+18, fh-4)
    cv2.putText(frame, f"ID:{bid}  {quality}  {speed}",
                (x1+4, strip_y), FONT, 0.44, (0,220,255), 1, cv2.LINE_AA)
    return frame


# ═══════════════════════════════════════════════════════════
# HELPER — does plate centroid fall inside vehicle box?
# ═══════════════════════════════════════════════════════════
def plate_in_vehicle(pbox, vbox, margin=30) -> bool:
    px1, py1, px2, py2 = pbox
    vx1, vy1, vx2, vy2 = vbox
    pcx = (px1+px2)/2
    pcy = (py1+py2)/2
    return (vx1-margin <= pcx <= vx2+margin and
            vy1-margin <= pcy <= vy2+margin)


# ═══════════════════════════════════════════════════════════
# 2-PASS VIDEO PROCESSOR  (Phases 3–13)
# ═══════════════════════════════════════════════════════════
def process_video(video_path: str, vehicle_model: YOLO,
                  detect_classes: list, alpr: ALPR, all_summary: list):
    name    = Path(video_path).stem
    outpath = f"{OUTPUT_DIR}/output_{name}.mp4"
    print(f"\n🎬 Processing: {name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"   ❌ Cannot open: {video_path}"); return

    fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── PASS 1: Detect + OCR every frame ─────────────────
    print(f"   [1/2] Pass 1 — Detecting plates ({tot} frames)...")
    raw_vbox    = defaultdict(dict)   # bid → {fidx: (x1,y1,x2,y2)}
    raw_pbox    = defaultdict(dict)   # bid → {fidx: (px1,py1,px2,py2)}
    plate_reads = defaultdict(list)   # bid → [(text, conf)]
    det_confs   = defaultdict(list)
    qual_reads  = defaultdict(list)
    speed_reads = defaultdict(list)
    prev_xy     = {}
    fidx        = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        fidx += 1
        if fidx % 100 == 0:
            pct = fidx / max(tot,1) * 100
            print(f"      {fidx}/{tot}  ({pct:.0f}%)")

        # ── PHASE 4: vehicle detect + track ───────────────
        res = vehicle_model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            classes=detect_classes, conf=CONF_THRESHOLD, verbose=False
        )
        if res[0].boxes.id is None:
            continue

        boxes  = res[0].boxes.xyxy.cpu().numpy().astype(int)
        ids    = res[0].boxes.id.cpu().numpy().astype(int)
        vconfs = res[0].boxes.conf.cpu().numpy()

        vehicle_map = {}   # bid → (x1,y1,x2,y2) this frame
        for vbox, bid, dc in zip(boxes, ids, vconfs):
            x1, y1, x2, y2 = vbox
            raw_vbox[bid][fidx] = (x1, y1, x2, y2)
            vehicle_map[bid]    = ((x1, y1, x2, y2), float(dc))

            # Speed
            cx, cy = (x1+x2)//2, (y1+y2)//2
            if bid in prev_xy:
                px, py = prev_xy[bid]
                d = ((cx-px)**2 + (cy-py)**2)**0.5
                if d >= 1:
                    kmh = (d/12.0) * fps * 3.6
                    if 1 < kmh < 200:
                        speed_reads[bid].append(round(kmh))
            prev_xy[bid] = (cx, cy)

        # ── PHASE 5+6: Run fast-alpr on FULL FRAME ────────
        # This gives the ONNX model maximum resolution / context
        try:
            alpr_results = alpr.predict(frame)
        except Exception:
            alpr_results = []

        for r in alpr_results:
            if not r.ocr or not r.ocr.text:
                continue
            bb   = r.detection.bounding_box
            px1  = max(0,  int(bb.x1))
            py1  = max(0,  int(bb.y1))
            px2  = min(fw, int(bb.x2))
            py2  = min(fh, int(bb.y2))
            if px2 <= px1 or py2 <= py1:
                continue
            pbox = (px1, py1, px2, py2)

            # Associate plate with the vehicle whose box contains it
            matched_bid = None
            matched_dc  = 0.0
            for bid, (vbox, dc) in vehicle_map.items():
                if plate_in_vehicle(pbox, vbox):
                    matched_bid = bid
                    matched_dc  = dc
                    break

            # If no vehicle matched, find nearest vehicle centroid
            if matched_bid is None and vehicle_map:
                pcx = (px1+px2)/2; pcy = (py1+py2)/2
                best_dist = float('inf')
                for bid, (vbox, dc) in vehicle_map.items():
                    vcx = (vbox[0]+vbox[2])/2
                    vcy = (vbox[1]+vbox[3])/2
                    dist = (pcx-vcx)**2 + (pcy-vcy)**2
                    if dist < best_dist:
                        best_dist  = dist
                        matched_bid = bid
                        matched_dc  = dc

            if matched_bid is None:
                continue

            # Store
            raw_pbox[matched_bid][fidx] = pbox
            plate_img = frame[py1:py2, px1:px2]
            qual_reads[matched_bid].append(
                classify_quality(plate_img, pbox, frame.shape))

            txt   = correct_ocr(r.ocr.text)
            c_val = r.ocr.confidence or 0
            oconf = float(c_val[0] if isinstance(c_val, list) else c_val)
            plate_reads[matched_bid].append((txt, oconf))
            det_confs[matched_bid].append(matched_dc)

    cap.release()

    # ── Interpolate gaps → zero flicker ───────────────────
    interp_vbox = defaultdict(dict)
    interp_pbox = defaultdict(dict)
    for bid, bd in raw_vbox.items():
        frames = sorted(bd.keys())
        for i, fi in enumerate(frames):
            interp_vbox[bid][fi] = bd[fi]
            if bid in raw_pbox and fi in raw_pbox[bid]:
                interp_pbox[bid][fi] = raw_pbox[bid][fi]
            if i < len(frames)-1:
                f1, f2 = frames[i], frames[i+1]
                gap = f2 - f1
                if 1 < gap <= MAX_INTERP_GAP:
                    b1 = np.array(bd[f1], dtype=float)
                    b2 = np.array(bd[f2], dtype=float)
                    for s in range(1, gap):
                        t = s/gap
                        interp_vbox[bid][f1+s] = tuple((b1+t*(b2-b1)).astype(int))
                        if (bid in raw_pbox and
                                f1 in raw_pbox[bid] and f2 in raw_pbox[bid]):
                            p1 = np.array(raw_pbox[bid][f1], dtype=float)
                            p2 = np.array(raw_pbox[bid][f2], dtype=float)
                            interp_pbox[bid][f1+s] = tuple((p1+t*(p2-p1)).astype(int))

    # ── Phases 7, 8, 9, 13: compute final result per track ─
    final_stats = {}
    for bid in raw_vbox:
        vp, aocr, cons = vote_plate(plate_reads[bid])
        adet   = (sum(det_confs[bid])/max(len(det_confs[bid]),1)
                  if det_confs[bid] else 0.0)
        cscore = compute_conf(adet, aocr, cons)
        plate  = safe_output(vp, cscore)

        qc   = Counter(qual_reads[bid])
        fq   = qc.most_common(1)[0][0] if qual_reads[bid] else "Partial"
        sv   = speed_reads[bid]
        fspd = f"{round(sum(sv)/len(sv))} km/h" if sv else "N/A"

        final_stats[bid] = {
            "Video"            : name,
            "Bike ID"          : int(bid),
            "Plate Number"     : plate,
            "Speed"            : fspd,
            "Confidence"       : cscore,
            "Detection Quality": fq,
            "Frame"            : min(raw_vbox[bid].keys()),
        }
        # Phase 10 — live CSV
        if plate != "Not Detected":
            try:
                with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(final_stats[bid])
            except PermissionError:
                pass # Ignore if file locked

    # ── PASS 2: Render annotated video ────────────────────
    print(f"   [2/2] Pass 2 — Rendering annotated video...")
    cap2   = cv2.VideoCapture(video_path)
    writer = cv2.VideoWriter(
        outpath, cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
    fidx = 0
    while True:
        ret, frame = cap2.read()
        if not ret: break
        fidx += 1
        for bid in interp_vbox:
            if fidx in interp_vbox[bid]:
                stats = final_stats.get(bid, {})
                frame = draw_overlay(
                    frame,
                    vbox      = interp_vbox[bid][fidx],
                    bid       = bid,
                    plate_text= stats.get("Plate Number","Not Detected"),
                    conf_pct  = stats.get("Confidence",0),
                    speed     = stats.get("Speed","N/A"),
                    quality   = stats.get("Detection Quality","Partial"),
                    pbox_abs  = interp_pbox[bid].get(fidx),
                )
        writer.write(frame)
    cap2.release()
    writer.release()

    # Phase 12 — strict print
    print(f"\n  ─── Results: {name} ───")
    for bid, r in final_stats.items():
        all_summary.append(r)
        print(f"Bike ID: {r['Bike ID']}")
        print(f"Plate Number: {r['Plate Number']}")
        print(f"Speed: {r['Speed']}")
        print(f"Confidence: {r['Confidence']}%")
        print(f"Detection Quality: {r['Detection Quality']}")
        print("-"*40)
    print(f"   ✅ Saved → {outpath}")


# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("  13-PHASE ANPR PIPELINE — FULL-FRAME PLATE DETECTION")
    print("="*60)

    vehicle_model, detect_classes, alpr = setup_models()

    # Init CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
    print(f"\n📄 CSV → {OUTPUT_CSV}")

    videos = (list(Path(TEST_VIDEOS_DIR).glob("*.mp4")) +
              list(Path(TEST_VIDEOS_DIR).glob("*.avi")) +
              list(Path(TEST_VIDEOS_DIR).glob("*.MP4")))
    if not videos:
        print(f"\n❌ No videos in {TEST_VIDEOS_DIR}"); exit(1)
    print(f"\n🔵 PHASE 3 — {len(videos)} video(s) found")

    all_summary = []
    for v in sorted(videos):
        process_video(str(v), vehicle_model, detect_classes, alpr, all_summary)

    if all_summary:
        df = (pd.DataFrame(all_summary)
              .sort_values("Confidence", ascending=False)
              .drop_duplicates(subset=["Video","Bike ID"], keep="first"))
        df["Confidence"] = df["Confidence"].astype(str) + "%"
        df.to_csv(OUTPUT_CSV, index=False)
        print("\n"+"="*60+"\n  FINAL RESULTS\n"+"="*60)
        cols = ["Bike ID","Plate Number","Speed","Confidence","Detection Quality"]
        print(df[cols].to_string(index=False))
        print(f"\n✅ CSV → {OUTPUT_CSV}")
    else:
        print("\n⚠️  No detections.")

    print(f"\n🎉 DONE\n   Videos → {OUTPUT_DIR}/output_*.mp4\n   CSV → {OUTPUT_CSV}")