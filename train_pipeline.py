# train_pipeline.py
import argparse, os, json, numpy as np, tensorflow as tf

# ─ common configs ─────────────────────────────────────────────
IMG_SIZE    = (224,224)
B1, E1      = 32, 15      # batch & epochs for LFW
B2, E2      = 16, 10      # batch & epochs for your 10 classes

def stage1():
    from sklearn.datasets        import fetch_lfw_people
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers       import GlobalAveragePooling2D, Dropout, Dense
    from tensorflow.keras.models       import Model
    from tensorflow.keras.optimizers   import Adam
    from tensorflow.keras.callbacks    import ModelCheckpoint, EarlyStopping

    print("▶ Loading LFW faces…")
    lfw = fetch_lfw_people(min_faces_per_person=20, resize=0.5, color=True)
    X, y = lfw.images, lfw.target; n_id = len(lfw.target_names)
    X = tf.image.resize(X, IMG_SIZE).numpy()/255.0
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    backbone = MobileNetV2((*IMG_SIZE,3), include_top=False, weights="imagenet")
    x = GlobalAveragePooling2D()(backbone.output)
    x = Dropout(0.3)(x)
    out = Dense(n_id, activation="softmax")(x)
    model = Model(backbone.input, out)
    model.compile(Adam(1e-4), "sparse_categorical_crossentropy", ["accuracy"])

    cb = [
      ModelCheckpoint("lfw_full_model.h5", monitor="val_accuracy", save_best_only=True),
      EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    ]
    model.fit(Xtr, ytr, validation_data=(Xva,yva), epochs=E1, batch_size=B1, callbacks=cb)

    backbone.save_weights("face_backbone_weights.h5")
    print("✅ Stage 1 done: face_backbone_weights.h5 saved")

def stage2(base_dir):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications       import MobileNetV2
    from tensorflow.keras.layers            import GlobalAveragePooling2D, Dropout, Dense
    from tensorflow.keras.models            import Model
    from tensorflow.keras.optimizers        import Adam
    from tensorflow.keras.callbacks         import ModelCheckpoint, EarlyStopping

    train_gen = ImageDataGenerator(
      rescale=1/255, rotation_range=15, width_shift_range=0.1,
      height_shift_range=0.1, brightness_range=[0.8,1.2], horizontal_flip=True
    ).flow_from_directory(
      os.path.join(base_dir,"train"), IMG_SIZE, batch_size=B2, class_mode="sparse"
    )
    val_gen = ImageDataGenerator(rescale=1/255).flow_from_directory(
      os.path.join(base_dir,"valid"), IMG_SIZE, batch_size=B2, class_mode="sparse"
    )

    labels = sorted(train_gen.class_indices, key=lambda k: train_gen.class_indices[k])
    with open("label_list.json","w") as f: json.dump(labels, f)

    backbone = MobileNetV2((*IMG_SIZE,3), include_top=False, weights=None)
    backbone.load_weights("face_backbone_weights.h5", by_name=True)
    backbone.trainable = False

    x = GlobalAveragePooling2D()(backbone.output)
    x = Dropout(0.3)(x)
    out = Dense(len(labels), activation="softmax")(x)
    model = Model(backbone.input, out)
    model.compile(Adam(1e-4), "sparse_categorical_crossentropy", ["accuracy"])

    cp = ModelCheckpoint("best_face_finetuned.h5", monitor="val_accuracy", save_best_only=True)
    es = EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)
    model.fit(train_gen, validation_data=val_gen, epochs=E2, callbacks=[cp,es])

    print("✅ Stage 2 done: best_face_finetuned.h5 + label_list.json saved")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["1","2","all"])
    p.add_argument("--base_dir", default=".", help="Path to train/valid folders")
    args = p.parse_args()

    if args.stage in ("1","all"): stage1()
    if args.stage in ("2","all"): stage2(args.base_dir)
