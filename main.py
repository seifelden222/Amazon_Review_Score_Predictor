import pandas as pd
import numpy as np
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import nltk
import logging
import argparse
import os
import create_dummy_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

class ScorePredictor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and numbers
        text = re.sub(rf'[{re.escape(string.punctuation)}\d+]', '', text)

        # Tokenize and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in word_tokenize(text)
                  if token not in self.stop_words and len(token) > 2]

        return ' '.join(tokens)

    def load_and_preprocess_data(self, filepath, text_column, score_column, num_rows=None):
        """Load and preprocess data"""
        df = pd.read_csv(filepath, nrows=num_rows)
        texts = df[text_column].apply(self.preprocess_text)
        scores = df[score_column]
        return texts, scores

    def train_and_evaluate(self, filepath, text_column, score_column,
                           model_save_path=None, num_rows=None):
        """
        Complete training pipeline with TF-IDF
        """
        # Load and preprocess data
        logging.info(f"Loading {num_rows if num_rows else 'all'} rows...")
        texts, scores = self.load_and_preprocess_data(
            filepath, text_column, score_column, num_rows)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, scores, test_size=0.2, random_state=42)

        # TF-IDF Vectorization
        logging.info("Vectorizing ...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Model training
        logging.info("Training Random Forest...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_vec, y_train)

        # Evaluation
        y_pred = self.model.predict(X_test_vec)
        self._print_evaluation_metrics(y_test, y_pred)

        # Save model
        if model_save_path:
            self._save_model(model_save_path)
            logging.info(f"Model saved to {model_save_path}")

    def _print_evaluation_metrics(self, y_true, y_pred):
        """Print evaluation metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        logging.info(f"Mean Squared Error (MSE): {mse:.4f}")
        logging.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        logging.info(f"R-squared (R²): {r2:.4f}")

    def _save_model(self, filepath):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer,
            'preprocessor': {
                'lemmatizer': self.lemmatizer,
                'stop_words': self.stop_words
            }
        }, filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ScorePredictor')
    parser.add_argument('--data', '-d', help='Path to CSV data file for training')
    parser.add_argument('--text-column', default='Text', help='Name of text column in CSV')
    parser.add_argument('--score-column', default='Score', help='Name of score column in CSV')
    # parser.add_argument('--save', '-s', default='model.joblib', help='Path to save trained model')
    parser.add_argument('--rows', '-r', type=int, default=30000, help='Number of rows to read from CSV')
    parser.add_argument('--create-dummy', action='store_true', help='Create and save a dummy model')

    args = parser.parse_args()

    trainer = ScorePredictor()

    # If requested, create dummy model and exit
    if args.create_dummy:
        logging.info('Creating dummy model via main...')
        create_dummy_model.create_and_save_dummy_model(path=args.save)
        raise SystemExit(0)

    data_path = args.data or 'Data/Amazon Customer Reviews.csv'

    # If data is missing, create a dummy model so `main.py` still runs
    if not os.path.exists(data_path):
        logging.warning('Data file not found at %s. Creating dummy model instead.', data_path)
        create_dummy_model.create_and_save_dummy_model(path=args.save)
        raise SystemExit(0)

    try:
        trainer.train_and_evaluate(
            filepath=data_path,
            text_column=args.text_column,
            score_column=args.score_column,
            model_save_path=args.save,
            num_rows=args.rows
        )
    except FileNotFoundError:
        logging.error('Error: File not found at %s', data_path)
    except Exception as e:
        logging.error('Training error: %s', str(e))
