# predict_with_display_v2.py — Phase 3 with batched inference + timing
import os, json, csv, time
import cv2, numpy as np
import tensorflow as tf
from datetime import datetime

# ─── CONFIG ────────────────────────────────────────────────────────────────
MODEL_PATH    = "best_face_model.h5"
LABELS_PATH   = "label_list.json"
PHOTO_DIR     = r"C:\Users\user\OneDrive\Documents\Desktop\Face Recognition"
OUTPUT_DIR    = "predicted_faces"
CSV_LOG       = "attendance.csv"
CSV_DEBUG     = "attendance_debug.csv"
CONF_THRESH   = 0.20         # only names ≥20% confidence
IMAGE_SIZE    = (224, 224)   # match your training size
SHOW_GUI      = True

# ─── PREP WORK ─────────────────────────────────────────────────────────────
# load model & labels
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH) as f:
    class_names = json.load(f)

# face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# make sure output dirs exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "Unknown"), exist_ok=True)

# logs
results, debug_results = [], []

# ─── PREDICTION FUNCTION ──────────────────────────────────────────────────
def predict_file(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Could not read {img_path}")
        return

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)

    if len(faces) == 0:
        return  # no faces to process

    # prepare a batch of face crops
    crops = []
    coords = []
    for (x, y, w, h) in faces:
        crop = img[y:y+h, x:x+w]
        crop_resized = cv2.resize(crop, IMAGE_SIZE)
        crops.append(crop_resized.astype("float32") / 255.0)
        coords.append((x, y, w, h))

    batch = np.stack(crops, axis=0)

    # time the model call
    t0 = time.time()
    preds = model.predict(batch, verbose=0)
    t1 = time.time()

    for (x,y,w,h), probs in zip(coords, preds):
        idx  = int(np.argmax(probs))
        conf = float(probs[idx])
        name = class_names[idx] if conf >= CONF_THRESH else "Unknown"
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # console output
        print(f"[{os.path.basename(img_path)}] → {name} ({conf*100:.1f}%)"
              f"  ➞ {1000*(t1-t0)/len(crops):.1f} ms/face")

        entry = {
            "Timestamp":  ts,
            "Image":      os.path.basename(img_path),
            "Name":       name,
            "Confidence": round(conf*100, 2)
        }

        # attendance: only first time each person appears
        if name != "Unknown" and name not in {r["Name"] for r in results}:
            results.append(entry)
        debug_results.append(entry)

        # save the crop
        subdir = name
        os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)
        out_name = f"{name}_{int(conf*100)}_{x}_{y}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, subdir, out_name), img[y:y+h, x:x+w])

        # annotate the image
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(
            img,
            f"{name} ({conf*100:.1f}%)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

    # show full image with overlays
    if SHOW_GUI:
        cv2.imshow("Prediction", img)
        cv2.waitKey(500)
        cv2.destroyAllWindows()

# ─── WALK DIRECTORY & PREDICT ──────────────────────────────────────────────
for root, _, files in os.walk(PHOTO_DIR):
    for fname in files:
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            predict_file(os.path.join(root, fname))

# ─── WRITE CSV LOGS ───────────────────────────────────────────────────────
def write_csv(path, data):
    with open(path, "w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["Timestamp","Image","Name","Confidence"])
        writer.writeheader()
        writer.writerows(data)

write_csv(CSV_LOG, results)
write_csv(CSV_DEBUG, debug_results)

print(f"\n✅ Attendance saved → {CSV_LOG}")
print(f"🐞 Debug log saved → {CSV_DEBUG}")
