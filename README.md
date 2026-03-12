# Amazon Review Score Predictor

Deep Learning model for Amazon review score classification.

The project classifies review text into one of five score classes: `1`, `2`, `3`, `4`, or `5`. It uses a TensorFlow/Keras text classification pipeline with a shared preprocessing function for training and inference.

## Installation

```bash
git clone https://github.com/seifelden222/Amazon_Review_Score_Predictor.git
cd Amazon_Review_Score_Predictor
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

- Put the dataset file at `Data/Amazon Customer Reviews.csv`
- Expected columns:
  - `Text`: review text
  - `Score`: review score from `1` to `5`

You can also pass a custom dataset path and custom column names through CLI arguments.

## Training

```bash
python main.py
```

Example with options:

```bash
python main.py --data "Data/Amazon Customer Reviews.csv" --text-column Text --score-column Score --epochs 10 --batch-size 32
```

Saved artifacts go to `artifacts/review_score_classifier/` by default:

- `model.keras`
- `metadata.json`
- `classification_report.txt`
- `confusion_matrix.csv`
- `sample_predictions.txt`

## Testing / Inference

Run demo predictions using the saved model:

```bash
python test.py
```

Run prediction on your own reviews:

```bash
python test.py --text "This product is amazing and durable." --text "Terrible quality and not worth the money."
```

## Model Architecture

The classifier uses:

- `TextVectorization`
- `Embedding`
- `Bidirectional(LSTM)`
- `Dense + Dropout`
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

## Example Training Output

```text
Training accuracy: 0.9125
Validation accuracy: 0.8500
Test accuracy: 0.8333

Classification report:
              precision    recall  f1-score   support
1                1.0000    1.0000    1.0000         2
2                0.5000    1.0000    0.6667         1
3                1.0000    0.5000    0.6667         2
4                1.0000    1.0000    1.0000         2
5                1.0000    1.0000    1.0000         2
```

Sample prediction format:

```text
Text: This product is perfect in every way.
True label: 5
Predicted label: 5
Class probabilities: [0.0012, 0.0031, 0.0104, 0.0815, 0.9038]
```

## Notes

- Labels are converted internally from `1-5` to `0-4` for training, then mapped back to `1-5` for reports and predictions.
- The same `clean_text()` preprocessing function is used in both training and inference to avoid training-serving skew.
- The saved Keras model contains the vectorization layer, so inference uses the same vocabulary and text pipeline learned during training.
