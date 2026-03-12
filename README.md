# Amazon Review Score Predictor

Deep Learning model for Amazon review score classification.

This project classifies Amazon review text into one of five score classes: `1`, `2`, `3`, `4`, or `5`. It uses a TensorFlow/Keras text classification pipeline and keeps preprocessing consistent between training and inference.

## Overview

- Input: review text
- Output: review score class from `1` to `5`
- Learning type: multi-class classification
- Framework: `tensorflow-cpu` / Keras

## Model Architecture

The training pipeline in [main.py](/home/seifelden-hamdy/Coding/Web/BackProject-python/Amazon_Review_Score_Predictor/main.py) uses:

- `TextVectorization`
- `Embedding`
- `Bidirectional(LSTM)`
- `Dense`
- `Dropout`
- `Dense(5, softmax)`

Compile settings:

- Optimizer: `adam`
- Loss: `sparse_categorical_crossentropy`
- Metric: `accuracy`

## Project Structure

```text
.
├── Data/
│   └── Amazon Customer Reviews.csv
├── artifacts/
│   └── review_score_classifier/
├── main.py
├── test.py
├── utils.py
├── run_training.py
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/seifelden222/Amazon_Review_Score_Predictor.git
cd Amazon_Review_Score_Predictor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Current main dependency:

```text
tensorflow-cpu>=2.16
```

## Dataset

Expected dataset path:

```text
Data/Amazon Customer Reviews.csv
```

Expected columns:

- `Text`: review text
- `Score`: integer score from `1` to `5`

Notes:

- Labels are converted internally from `1-5` to `0-4` during training
- Predictions are mapped back to `1-5` for display
- The same text cleaning function in [utils.py](/home/seifelden-hamdy/Coding/Web/BackProject-python/Amazon_Review_Score_Predictor/utils.py) is used in training and inference

## Training

Basic command:

```bash
.venv/bin/python main.py
```

Recommended command for a clean screenshot in the report:

```bash
.venv/bin/python main.py 2>/dev/null
```

Example with arguments:

```bash
.venv/bin/python main.py \
  --data "Data/Amazon Customer Reviews.csv" \
  --text-column Text \
  --score-column Score \
  --epochs 8 \
  --batch-size 32
```

## Training Outputs

Saved artifacts go to:

```text
artifacts/review_score_classifier/
```

Generated files:

- `model.keras`
- `metadata.json`
- `classification_report.txt`
- `confusion_matrix.csv`
- `sample_predictions.txt`
- `training_console_report.txt`

The console report includes:

- training accuracy
- validation accuracy
- test accuracy
- classification report
- confusion matrix
- sample predictions with class probabilities

## Testing / Inference

Run demo predictions from the saved model:

```bash
.venv/bin/python test.py
```

Run your own custom reviews:

```bash
.venv/bin/python test.py \
  --text "This product is amazing and durable." \
  --text "Terrible quality and not worth the money."
```

## Example Console Output

```text
========================================================================
AMAZON REVIEW SCORE CLASSIFICATION REPORT
========================================================================

[Accuracy Summary]
Training accuracy     : 0.8000
Validation accuracy   : 1.0000
Test accuracy         : 0.0000
Training rows         : 10
Validation rows       : 2
Test rows             : 3

[Classification Report]
precision    recall  f1-score   support

           1     0.0000    0.0000    0.0000       1.0
           3     0.0000    0.0000    0.0000       1.0
           4     0.0000    0.0000    0.0000       1.0
           5     0.0000    0.0000    0.0000       0.0

    accuracy                         0.0000       3.0
   macro avg     0.0000    0.0000    0.0000       3.0
weighted avg     0.0000    0.0000    0.0000       3.0

[Confusion Matrix]
True\Pred        1       2       3       4       5
1                0       0       0       0       1
2                0       0       0       0       0
3                0       0       0       0       1
4                0       0       0       0       1
5                0       0       0       0       0
```

Sample prediction format:

```text
Text: better than expected
True label: 4
Predicted label: 5
Class probabilities: [0.2003, 0.1982, 0.1969, 0.2015, 0.203]
```

## Important Note

If you train on a very small sample dataset, the model will still run, but the reported accuracy may be weak and some classes may not be predicted well. For a stronger university report, use a larger dataset with enough examples for all five score classes.
