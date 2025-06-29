# evaluate.py — phase 4: metrics & plots on hold-out test set
import os, json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─── CONFIG ────────────────────────────────────────────
MODEL_PATH  = "best_face_model.h5"
LABELS_PATH = "label_list.json"
TEST_DIR    = r"C:\Users\user\OneDrive\Documents\Desktop\Face Recognition\test"
IMAGE_SIZE  = (224, 224)
BATCH_SIZE  = 16
OUT_SUM     = "evaluation_summary.txt"

# ─── LOAD ──────────────────────────────────────────────
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH) as f:
    class_names = json.load(f)

test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    TEST_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
    class_mode="sparse", shuffle=False
)

y_true = test_gen.classes
y_prob = model.predict(test_gen, verbose=1)
y_pred = np.argmax(y_prob, axis=1)

# ─── REPORT & CM ───────────────────────────────────────
report = classification_report(
    y_true, y_pred, target_names=class_names, digits=4
)
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# ─── ROC CURVES ────────────────────────────────────────
y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
plt.figure(figsize=(8,6))
for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve(y_true_bin[:,i], y_prob[:,i])
    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc(fpr,tpr):.2f})")
plt.plot([0,1],[0,1],"k--")
plt.legend(loc="lower right"); plt.title("ROC Curves")
plt.savefig("roc_curve.png", dpi=150)
plt.close()

# ─── SAVE SUMMARY ──────────────────────────────────────
with open(OUT_SUM, "w") as f:
    f.write("CLASSIFICATION REPORT\n\n" + report)
    f.write("\n\nConfusion Matrix: confusion_matrix.png")
    f.write("\nROC Curve:       roc_curve.png\n")

print(f"📄 Eval summary → {OUT_SUM}")
