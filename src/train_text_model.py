import argparse
import json
from pathlib import Path

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import BASE_DIR

SEED = 42
tf.random.set_seed(SEED)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {"text", "label"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    df = df[df["label"].isin(["fake", "real"])].copy()

    if df.empty:
        raise ValueError("Dataset vide après filtrage fake/real.")

    df["y"] = df["label"].map({"real": 0, "fake": 1}).astype("float32")

    return df


def build_model(max_tokens: int = 3000) -> tf.keras.Model:
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="multi_hot",
        pad_to_max_tokens=True,
        name="text_vectorization",
    )
    inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text")

    x = vectorizer(inputs)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(0.20, name="dropout_1")(x)
    x = tf.keras.layers.Dense(32, activation="relu", name="dense_2")(x)
    x = tf.keras.layers.Dropout(0.10, name="dropout_2")(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="fake_probability"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="text_fake_news_tfidf_model"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )

    return model


def make_dataset(texts, labels, batch_size: int, shuffle: bool = False):
    texts_tensor = tf.constant(list(texts), dtype=tf.string)
    labels_tensor = tf.constant(labels, dtype=tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((texts_tensor, labels_tensor))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(texts), seed=SEED)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def main():
    parser = argparse.ArgumentParser(description="Train TensorFlow TF-IDF text fake news model")

    parser.add_argument(
        "--csv",
        type=str,
        default=str(BASE_DIR / "data" / "raw" / "train_text.csv"),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    args = parser.parse_args()

    df = load_dataset(Path(args.csv))

    X = df["text"].tolist()
    y = df["y"].values

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=SEED,
        stratify=y_train_full,
    )

    model = build_model()

    vectorizer = model.get_layer("text_vectorization")
    vectorizer.adapt(
        tf.data.Dataset.from_tensor_slices(
            tf.constant(X_train, dtype=tf.string)
        ).batch(32)
    )

    print(model.summary())

    train_ds = make_dataset(X_train, y_train, args.batch_size, shuffle=True)
    val_ds = make_dataset(X_val, y_val, args.batch_size)
    test_ds = make_dataset(X_test, y_test, args.batch_size)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
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

    evaluation = model.evaluate(test_ds, verbose=0)
    metrics = dict(zip(model.metrics_names, [float(v) for v in evaluation]))

    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "text_fake_news_model.keras"
    model.save(model_path)

    label_map = {
        "real": 0,
        "fake": 1,
        "threshold": 0.5,
        "model_path": str(model_path),
        "model_type": "tfidf_dense",
    }

    label_map_path = models_dir / "text_label_map.json"

    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4, ensure_ascii=False)

    training_report_path = BASE_DIR / "outputs" / "history" / "text_model_training_report.json"
    training_report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(training_report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": str(args.csv),
                "samples": int(len(df)),
                "train_samples": int(len(X_train)),
                "val_samples": int(len(X_val)),
                "test_samples": int(len(X_test)),
                "metrics": metrics,
                "history": {
                    k: [float(x) for x in v]
                    for k, v in history.history.items()
                },
                "model_path": str(model_path),
            },
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("\n✅ Modèle TF-IDF TensorFlow entraîné et sauvegardé")
    print(f"Modèle : {model_path}")
    print(f"Métriques test : {metrics}")


if __name__ == "__main__":
    main()