import pickle
# import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from src.data_preprocessing import prefix_extractor, extract_prefixes


BASE_DIR = Path().resolve()
MODEL_DIR = BASE_DIR/'model'
MODEL_PATH = MODEL_DIR/'model.pk1'
LABEL_ENCODER_PATH = MODEL_DIR/'label_encoder.pk1'
PREPROCESSING_PATH = MODEL_DIR/'preprocessing.pk1'
# MODEL_PATH = MODEL_DIR/'model.joblib'
# LABEL_ENCODER_PATH = MODEL_DIR/'label_encoder.joblib'
# PREPROCESSING_PATH = MODEL_DIR/'preprocessing.joblib'

print(BASE_DIR)

# function to load the preprocessing steps
def load_preprocessing():
    with open(PREPROCESSING_PATH, 'rb') as f:
        preprocessed = pickle.load(f)
        # preprocessed = joblib.load(f)
    return preprocessed


# function declaration to read the stored model
# def load_model():
#     # Loading the trained model
#     model = pickle.load(open(MODEL_PATH, 'rb'))
#     return model

# OR

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        # model = joblib.load(f)
    return model


# function to load the label encoder
# def load_label_encoder():
#     # load the saved le
#     label_encoder = pickle.load(open(LABEL_ENCODER_PATH, 'rb'))
#     return label_encoder

# OR

def load_label_encoder():
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
        # label_encoder = joblib.load(f)
    return label_encoder

