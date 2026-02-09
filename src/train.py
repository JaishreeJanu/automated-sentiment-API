import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import os

def train():
    # 1. Load processed data
    train_df = pd.read_csv('data/preprocessed/train.csv')
    test_df = pd.read_csv('data/preprocessed/test.csv')

    X_train, y_train = train_df['review'], train_df['sentiment']
    X_test, y_test = test_df['review'], test_df['sentiment']

    # 2. Start MLflow Tracking
    # This will create a local 'mlruns' folder
    with mlflow.start_run():
        params = {"C": 1.0, "max_iter": 100}
        mlflow.log_params(params)

        # 3. Create Pipeline (Vectorization + Model)
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', LogisticRegression(**params))
        ])

        # 4. Train
        pipeline.fit(X_train, y_train)

        # 5. Evaluate
        predictions = pipeline.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        # 6. Log Metrics & Model
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(pipeline, "sentiment_model")
        
        print(f"Model trained with accuracy: {acc}")

        # 7. Save model locally for FastAPI
        os.makedirs('models', exist_ok=True)
        import joblib
        joblib.dump(pipeline, 'models/model.pkl')

if __name__ == "__main__":
    train()