import tarfile
_orig = tarfile.TarFile.extractall
def extractall(self, path="", members=None, *, numeric_owner=False, filter=None):
    # ignore the filter kwarg, forward the rest
    return _orig(self, path=path, members=members, numeric_owner=numeric_owner)
tarfile.TarFile.extractall = extractall

from sklearn.datasets import fetch_lfw_people
# … rest of your code …
# pretrain_lfw_sklearn.py
import numpy as np
import tensorflow as tf
from sklearn.datasets        import fetch_lfw_people
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers       import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models       import Model
from tensorflow.keras.optimizers   import Adam
from tensorflow.keras.callbacks    import ModelCheckpoint, EarlyStopping

# ─── CONFIG ────────────────────────────────────────────────────────────────
IMG_SIZE     = (224,224)
BATCH_SIZE   = 32
EPOCHS       = 15
BACKBONE_OUT = "face_backbone_weights.h5"

# ─── LOAD & PREP LFW ───────────────────────────────────────────────────────
lfw = fetch_lfw_people(min_faces_per_person=20, resize=0.5, color=True)
X, y = lfw.images, lfw.target
X = tf.image.resize(X, IMG_SIZE).numpy()/255.0

Xtr, Xva, ytr, yva = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ─── BUILD & TRAIN BACKBONE ───────────────────────────────────────────────
backbone = MobileNetV2((*IMG_SIZE,3), include_top=False, weights="imagenet")
x = GlobalAveragePooling2D()(backbone.output)
x = Dropout(0.3)(x)
out = Dense(len(lfw.target_names), activation="softmax")(x)
model = Model(backbone.input, out)
model.compile(Adam(1e-4), "sparse_categorical_crossentropy", ["accuracy"])

cb = [
  ModelCheckpoint("lfw_full_model.h5", monitor="val_accuracy", save_best_only=True),
  EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
]
model.fit(Xtr, ytr, validation_data=(Xva, yva),
          epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb)

# ─── DUMP THE BACKBONE WEIGHTS ─────────────────────────────────────────────
backbone.save_weights(BACKBONE_OUT)
print(f"✅ Saved backbone → {BACKBONE_OUT}")
