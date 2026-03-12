import json
import re
import string
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_PATH = Path("Data") / "Amazon Customer Reviews.csv"
DEFAULT_MODEL_DIR = Path("artifacts") / "review_score_classifier"
LABEL_OFFSET = 1


def ensure_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_text(text):
    """Shared text cleaning for both training and inference."""
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_dataset(filepath, text_column, score_column, num_rows=None):
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path, nrows=num_rows)

    missing = [column for column in [text_column, score_column] if column not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available columns: {list(df.columns)}"
        )

    df = df[[text_column, score_column]].dropna().copy()

    if df.empty:
        raise ValueError("Dataset is empty after dropping rows with missing values.")

    if not df[score_column].between(1, 5).all():
        raise ValueError("Score column must contain integer labels from 1 to 5.")

    return df


def encode_labels(labels):
    return np.asarray(labels, dtype=np.int32) - LABEL_OFFSET


def decode_labels(labels):
    return np.asarray(labels, dtype=np.int32) + LABEL_OFFSET


def save_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_sample_prediction_lines(texts, true_labels, predicted_labels, probabilities):
    lines = []
    for text, true_label, predicted_label, probs in zip(
        texts,
        true_labels,
        predicted_labels,
        probabilities,
    ):
        prob_list = [round(float(value), 4) for value in probs]
        lines.append(
            "\n".join(
                [
                    f"Text: {text}",
                    f"True label: {true_label}",
                    f"Predicted label: {predicted_label}",
                    f"Class probabilities: {prob_list}",
                ]
            )
        )
    return lines


def format_confusion_matrix(matrix, labels):
    header = "True\\Pred".ljust(10) + "".join(f"{label:>8}" for label in labels)
    rows = [header]
    for label, row in zip(labels, matrix):
        rows.append(f"{str(label).ljust(10)}" + "".join(f"{value:>8}" for value in row))
    return "\n".join(rows)


def format_metrics_block(metrics):
    lines = []
    for label, value in metrics:
        lines.append(f"{label:<22}: {value}")
    return "\n".join(lines)


def build_console_report(metrics_block, report, matrix_text, sample_lines):
    sections = [
        "=" * 72,
        "AMAZON REVIEW SCORE CLASSIFICATION REPORT",
        "=" * 72,
        "",
        "[Accuracy Summary]",
        metrics_block,
        "",
        "[Classification Report]",
        report.strip(),
        "",
        "[Confusion Matrix]",
        matrix_text,
        "",
        "[Sample Predictions]",
    ]

    for index, sample in enumerate(sample_lines, start=1):
        wrapped_lines = []
        for line in sample.splitlines():
            if line.startswith("Text: "):
                prefix = "Text: "
                content = line[len(prefix):]
                wrapped = textwrap.fill(
                    content,
                    width=66,
                    initial_indent=prefix,
                    subsequent_indent=" " * len(prefix),
                )
                wrapped_lines.append(wrapped)
            else:
                wrapped_lines.append(line)
        sections.append(f"Sample {index}")
        sections.append("\n".join(wrapped_lines))
        sections.append("-" * 72)

    return "\n".join(sections).rstrip("-\n") + "\n"
