import joblib
import pandas as pd
from main import ScorePredictor
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def load_and_test_model(model_path='model.joblib'):

    try:
        # Load the saved model
        logging.info(f"Loading model from {model_path}...")
        predictor = ScorePredictor()
        model_data = joblib.load(model_path)

        # Verify loaded components
        if not all(key in model_data for key in ['vectorizer', 'model']):
            raise ValueError("Model file is missing required components")

        predictor.vectorizer = model_data['vectorizer']
        predictor.model = model_data['model']

        logging.info("Model loaded successfully!")

        # Test cases that should cover all score ranges
        test_cases = [
            ("This product is perfect in every way", 5),
            ("Absolutely terrible quality", 1),
            ("Mediocre but works okay", 3),
            ("Better than expected", 4),
            ("Worst purchase of my life", 1),
            ("Excellent value for the price", 5),
            ("Average product with some flaws", 2),
            ("Good but could be improved", 4),
            ("Complete waste of money", 1),
            ("Highly recommended to everyone", 5),
            ("very good", 5),
        ]

        results = []
        for text, expected in test_cases:
            # Vectorize and predict
            vec = predictor.vectorizer.transform([text])
            pred = predictor.model.predict(vec)[0]

            results.append({
                'Text': text,
                'Expected': expected,
                'Predicted': round(pred, 1),
                'Error': abs(expected - round(pred, 1))
            })

        # Display results
        results = pd.DataFrame(results)
        logging.info("Test Results:\n%s", results.to_string(index=False))

        # Calculate overall accuracy
        avg_error = results['Error'].mean()
        logging.info("Average Error: %.2f", avg_error)

        return results

    except Exception as e:
        logging.error(f"Error during testing: {str(e)}")
        return None


if __name__ == "__main__":
    test_results = load_and_test_model()
