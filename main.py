import argparse
import contextlib
import io
import logging
import os
import warnings
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

with contextlib.redirect_stderr(io.StringIO()):
    import tensorflow as tf

from utils import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    LABEL_OFFSET,
    build_console_report,
    format_confusion_matrix,
    format_metrics_block,
    build_sample_prediction_lines,
    clean_text,
    decode_labels,
    encode_labels,
    ensure_directory,
    load_dataset,
    save_json,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings(
    "ignore",
    message=r".*joblib will operate in serial mode.*",
    category=UserWarning,
)


def build_model(train_texts, max_tokens=20000, sequence_length=150, embedding_dim=128):
    """Build a Keras text classifier and adapt the vectorizer on training text only."""
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length,
        standardize=None,
    )
    vectorizer.adapt(train_texts)

    inputs = tf.keras.Input(shape=(), dtype=tf.string, name="review_text")
    x = vectorizer(inputs)
    x = tf.keras.layers.Embedding(max_tokens, embedding_dim, mask_zero=True)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(5, activation="softmax", name="score_class")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="amazon_review_classifier")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def split_dataset(texts, labels):
    """Create train/validation/test splits with stratification when possible."""
    _, counts = np.unique(labels, return_counts=True)
    stratify_labels = labels if len(np.unique(labels)) > 1 and np.min(counts) >= 2 else None

    X_train, X_temp, y_train, y_temp = train_test_split(
        texts,
        labels,
        test_size=0.3,
        random_state=42,
        stratify=stratify_labels,
    )

    _, temp_counts = np.unique(y_temp, return_counts=True)
    stratify_temp = y_temp if len(np.unique(y_temp)) > 1 and np.min(temp_counts) >= 2 else None
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=stratify_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_tf_dataset(texts, labels, batch_size, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((texts.values, labels))
    if training:
        dataset = dataset.shuffle(buffer_size=len(texts), seed=42)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_classifier(args):
    data_path = Path(args.data)
    output_dir = Path(args.save_dir)
    ensure_directory(output_dir)

    df = load_dataset(
        filepath=data_path,
        text_column=args.text_column,
        score_column=args.score_column,
        num_rows=args.rows,
    )
    df["clean_text"] = df[args.text_column].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].copy()

    if df.empty:
        raise ValueError("No usable text rows remained after preprocessing.")

    labels = encode_labels(df[args.score_column].to_numpy())
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df["clean_text"], labels)

    model = build_model(
        train_texts=X_train.values,
        max_tokens=args.max_tokens,
        sequence_length=args.sequence_length,
        embedding_dim=args.embedding_dim,
    )

    train_ds = create_tf_dataset(X_train, y_train, args.batch_size, training=True)
    val_ds = create_tf_dataset(X_val, y_val, args.batch_size)
    test_ds = create_tf_dataset(X_test, y_test, args.batch_size)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=3,
            restore_best_weights=True,
        )
    ]

    logging.info("Starting training on %s rows", len(X_train))
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=0,
    )

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    probabilities = model.predict(test_ds, verbose=0)
    predicted_indices = np.argmax(probabilities, axis=1)

    true_labels = decode_labels(y_test)
    predicted_labels = decode_labels(predicted_indices)

    train_accuracy = float(history.history["accuracy"][-1])
    val_accuracy = float(history.history["val_accuracy"][-1])
    report = classification_report(true_labels, predicted_labels, digits=4, zero_division=0)
    class_labels = [1, 2, 3, 4, 5]
    matrix = confusion_matrix(true_labels, predicted_labels, labels=class_labels)

    model_path = output_dir / "model.keras"
    model.save(model_path)

    metrics = {
        "train_accuracy": train_accuracy,
        "validation_accuracy": val_accuracy,
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "label_offset": LABEL_OFFSET,
        "classes": [1, 2, 3, 4, 5],
        "text_column": args.text_column,
        "score_column": args.score_column,
    }
    save_json(output_dir / "metadata.json", metrics)
    (output_dir / "classification_report.txt").write_text(report + "\n", encoding="utf-8")
    pd.DataFrame(
        matrix,
        index=class_labels,
        columns=class_labels,
    ).to_csv(output_dir / "confusion_matrix.csv")

    sample_indices = list(range(min(args.sample_count, len(X_test))))
    sample_lines = build_sample_prediction_lines(
        texts=X_test.iloc[sample_indices].tolist(),
        true_labels=true_labels[sample_indices],
        predicted_labels=predicted_labels[sample_indices],
        probabilities=probabilities[sample_indices],
    )
    (output_dir / "sample_predictions.txt").write_text(
        "\n\n".join(sample_lines) + "\n",
        encoding="utf-8",
    )
    metrics_block = format_metrics_block(
        [
            ("Training accuracy", f"{train_accuracy:.4f}"),
            ("Validation accuracy", f"{val_accuracy:.4f}"),
            ("Test accuracy", f"{test_accuracy:.4f}"),
            ("Training rows", str(len(X_train))),
            ("Validation rows", str(len(X_val))),
            ("Test rows", str(len(X_test))),
        ]
    )
    matrix_text = format_confusion_matrix(matrix, class_labels)
    console_report = build_console_report(metrics_block, report, matrix_text, sample_lines)
    (output_dir / "training_console_report.txt").write_text(console_report, encoding="utf-8")

    print()
    print(console_report)
    logging.info("Saved model to %s", model_path)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a deep learning classifier for Amazon review scores.",
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to the CSV dataset.")
    parser.add_argument("--text-column", default="Text", help="Name of the review text column.")
    parser.add_argument("--score-column", default="Score", help="Name of the score column.")
    parser.add_argument("--rows", type=int, default=None, help="Optional limit on dataset rows.")
    parser.add_argument("--save-dir", default=str(DEFAULT_MODEL_DIR), help="Directory for saved artifacts.")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--max-tokens", type=int, default=20000, help="Vocabulary size for TextVectorization.")
    parser.add_argument("--sequence-length", type=int, default=150, help="Token sequence length.")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension size.")
    parser.add_argument("--sample-count", type=int, default=5, help="Number of sample predictions to print/save.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        train_classifier(args)
    except Exception as exc:
        logging.error("Training failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
