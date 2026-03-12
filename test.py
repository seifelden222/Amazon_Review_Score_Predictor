import argparse
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

from utils import (
    DEFAULT_MODEL_DIR,
    build_sample_prediction_lines,
    clean_text,
    load_json,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


DEFAULT_REVIEWS = [
    ("This product is perfect in every way.", 5),
    ("Absolutely terrible quality and I regret buying it.", 1),
    ("It is okay for the price, nothing special.", 3),
    ("Very solid purchase, I would buy it again.", 4),
    ("Stopped working after two days, complete waste of money.", 1),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using the saved Keras classifier.")
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Directory containing model.keras and metadata.json.",
    )
    parser.add_argument(
        "--text",
        action="append",
        dest="texts",
        help="Review text to classify. Use multiple times for multiple reviews.",
    )
    return parser.parse_args()


def load_saved_model(model_dir):
    model_path = Path(model_dir) / "model.keras"
    metadata_path = Path(model_dir) / "metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found at {model_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

    model = tf.keras.models.load_model(model_path)
    metadata = load_json(metadata_path)
    return model, metadata


def run_inference(model_dir, review_texts):
    model, metadata = load_saved_model(model_dir)
    cleaned_reviews = [clean_text(text) for text in review_texts]
    probabilities = model.predict(np.array(cleaned_reviews, dtype=object), verbose=0)
    predicted_indices = np.argmax(probabilities, axis=1)
    predicted_labels = (predicted_indices + 1).tolist()

    sample_lines = build_sample_prediction_lines(
        texts=review_texts,
        true_labels=["N/A"] * len(review_texts),
        predicted_labels=predicted_labels,
        probabilities=probabilities,
    )

    logging.info(
        "Loaded model trained on text column '%s' and score column '%s'.",
        metadata["text_column"],
        metadata["score_column"],
    )
    logging.info("Predictions:\n%s", "\n\n".join(sample_lines))
    return sample_lines


def main():
    args = parse_args()
    if args.texts:
        review_texts = args.texts
    else:
        review_texts = [text for text, _ in DEFAULT_REVIEWS]

    try:
        run_inference(args.model_dir, review_texts)
    except Exception as exc:
        logging.error("Inference failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
