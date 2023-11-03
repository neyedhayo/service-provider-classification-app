import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_preprocessing import extract_phone_prefixes

class ServiceProviderClassifier():
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.model = Pipeline([
            ('one_hot_encoder', OneHotEncoder()),
            ('classifier', LogisticRegression())
        ])

    def train(self, X, y):
        y_encode = self.label_encoder.fit_transform(y)
        self.model.fit(X, y_encode)

    def predict(self, X):
        y_pred_encode = self.model.predict(X)
        y_pred = self.label_encoder.inverse_transform(y_pred_encode)
        return y_pred
    
    def save_model(self, file_path = '../model/service_provider_classifier.joblib'):
        joblib.dump((self.model, self.label_encoder), file_path)

    def load_model(self, file_path = '../model/service_provider_classifier.joblib'):
        self.model, self.label_encoder = joblib.load(file_path)


def train_model():
    # data
    datapath = "data/processed/cleaned_data.csv"
    mobile_df = pd.read_csv(datapath)
    
    # Extract phone prefixes and prepare data
    X = extract_phone_prefixes(mobile_df)
    y = mobile_df['Service_Provider']

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    classifier = ServiceProviderClassifier()
    classifier.train(X_train, y_train)
    classifier.save_model('service_provider_classifier.joblib')

    # Evaluate the model
    # y_pred = classifier.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # print("Model trained and saved successfully.")

if __name__ == "__main__":
    train_model()