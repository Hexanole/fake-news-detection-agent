import argparse
import json
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import BASE_DIR

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 16

tf.random.set_seed(SEED)


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable : {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["image_path", "label"])
    df = df[df["label"].isin(["fake", "real"])].copy()

    df["image_path"] = df["image_path"].astype(str)
    df["y"] = df["label"].map({"real": 0, "fake": 1}).astype("float32")

    df = df[df["image_path"].apply(lambda p: Path(p).exists())].copy()

    if df.empty:
        raise ValueError("Aucune image valide trouvée.")

    return df


def load_and_preprocess_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32)

    # Prétraitement MobileNetV2 : pixels vers intervalle adapté au modèle
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    return image, label


def make_dataset(paths, labels, batch_size: int, shuffle: bool = False):
    paths = tf.constant(list(paths), dtype=tf.string)
    labels = tf.constant(labels, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)

    ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds


def build_model() -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.10),
        ],
        name="augmentation",
    )

    x = augmentation(inputs)

    try:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
        print("MobileNetV2 chargé avec poids ImageNet ✅")
    except Exception as e:
        print(f"Impossible de charger les poids ImageNet : {e}")
        print("MobileNetV2 sera entraîné sans poids pré-entraînés.")
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights=None,
        )

    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(0.30, name="dropout")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense")(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="fake_probability"
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="image_fake_news_model")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="Entraîner modèle image fake/real")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)

    args = parser.parse_args()

    df = load_csv(Path(args.csv))

    X = df["image_path"].tolist()
    y = df["y"].values

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    train_ds = make_dataset(X_train, y_train, args.batch_size, shuffle=True)
    val_ds = make_dataset(X_val, y_val, args.batch_size, shuffle=False)

    model = build_model()
    print(model.summary())

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "image_fake_news_model.keras"
    model.save(model_path)

    report_path = BASE_DIR / "outputs" / "history" / "image_model_training_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "csv": str(args.csv),
                "samples": int(len(df)),
                "train_samples": int(len(X_train)),
                "val_samples": int(len(X_val)),
                "history": {
                    k: [float(v) for v in values]
                    for k, values in history.history.items()
                },
                "model_path": str(model_path),
            },
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("\n✅ Modèle image entraîné et sauvegardé")
    print(f"Modèle : {model_path}")
    print(f"Rapport : {report_path}")


if __name__ == "__main__":
    main()