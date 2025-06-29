# train.py — Phase 2: train (from scratch) or fine-tune (with TL backbone)
import os, json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models               import Model, Sequential
from tensorflow.keras.layers               import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    GlobalAveragePooling2D
)
from tensorflow.keras.applications          import MobileNetV2
from tensorflow.keras.optimizers            import Adam
from tensorflow.keras.callbacks             import ModelCheckpoint, EarlyStopping

# ─── CONFIG ────────────────────────────────────────────────────────────────
BASE_DIR           = r"C:\Users\user\OneDrive\Documents\Desktop\Face Recognition"
TRAIN_DIR          = os.path.join(BASE_DIR, "train")
VAL_DIR            = os.path.join(BASE_DIR, "valid")
IMAGE_SIZE         = (100, 100)     # for scratch CNN
TL_SIZE            = (224, 224)     # for MobileNetV2
BATCH_SIZE         = 16
EPOCHS             = 30
USE_TL             = True           # ← set False to use the small CNN
BACKBONE_WEIGHTS   = "face_backbone_weights.h5"
MODEL_OUT          = "best_face_model.h5"
LABEL_JSON         = "label_list.json"

# ─── DATA AUGMENTATION ─────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8,1.2],
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1/255)

# choose the right input size
target_size = TL_SIZE if USE_TL else IMAGE_SIZE

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=target_size,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=target_size,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    shuffle=False
)

# ─── SAVE LABEL ORDER ──────────────────────────────────────────────────────
labels = sorted(train_gen.class_indices, key=lambda k: train_gen.class_indices[k])
with open(LABEL_JSON, "w") as f:
    json.dump(labels, f)
print(f"Labels ({len(labels)}): {labels}")

# ─── BUILD MODEL ───────────────────────────────────────────────────────────
if USE_TL:
    # 1) load a fresh MobileNetV2 backbone
    backbone = MobileNetV2(
        input_shape=(*TL_SIZE,3),
        include_top=False,
        weights=None            # we’ll load our own face‐pretrained weights
    )
    # 2) load the weights you saved in Stage 1, freeze the backbone
    backbone.load_weights(BACKBONE_WEIGHTS, by_name=True)
    backbone.trainable = False

    # 3) attach a new head for your N classes
    x = GlobalAveragePooling2D()(backbone.output)
    x = Dropout(0.3)(x)
    output = Dense(len(labels), activation="softmax")(x)
    model = Model(backbone.input, output)
else:
    # Build a small CNN from scratch
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(*IMAGE_SIZE, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(len(labels), activation="softmax")
    ])

# ─── COMPILE ───────────────────────────────────────────────────────────────
model.compile(
    optimizer=Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ─── CALLBACKS & TRAIN ─────────────────────────────────────────────────────
checkpoint = ModelCheckpoint(
    MODEL_OUT,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
earlystop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, earlystop]
)

print(f"✅ Saved model → {MODEL_OUT}")
print(f"📑 Labels list → {LABEL_JSON}")
