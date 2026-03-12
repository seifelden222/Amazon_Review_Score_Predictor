from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import joblib

def create_and_save_dummy_model(path='model.joblib'):
    # Small synthetic dataset
    texts = [
        "This product is excellent and works perfectly",
        "Terrible item, broke after one use",
        "Average quality, not bad",
        "Highly recommend, very satisfied",
        "Do not buy, waste of money",
        "Good value for the price",
        "Poor build quality",
        "Exceeded my expectations",
        "Not as described",
        "Fantastic product"
    ]
    # Corresponding scores (1-5)
    scores = [5, 1, 3, 5, 1, 4, 2, 5, 2, 5]

    vect = TfidfVectorizer(max_features=5000)
    X = vect.fit_transform(texts)

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, scores)

    joblib.dump({'vectorizer': vect, 'model': model}, path)
    print(f"Dummy model saved to {path}")


if __name__ == '__main__':
    create_and_save_dummy_model()
