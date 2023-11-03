import pandas as pd
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# PREPROCESSING STEPS

# custom transformer to extract prefixes from phone number
# class PrefixExtractor(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         X = pd.DataFrame(X)
#         X['Phone_Prefixes'] = X['Telephone_Number'].str[:4]
#         return X[['Phone_Prefixes']]

# OR
# using a functiontransfomer
def extract_prefixes(X):
    X = pd.DataFrame(X)
    X['Phone_Prefixes'] = X['Telephone_Number'].str[:4]
    return X[['Phone_Prefixes']]

prefix_extractor = FunctionTransformer(extract_prefixes)


# preprocessor to extract prefixes and encode them
preprocessor = ColumnTransformer(
    transformers = [
        ('Phone_Prefixes', OneHotEncoder(), ['Phone_Prefixes'])
    ],
    remainder = 'passthrough'
)

# combining the prefix extractor and preprocessor in a pipeline
# pipeline = Pipeline([
#     ('prefix_extractor', prefix_extractor),
#     ('preprocessor', preprocessor)
# ])

# catching column not in dataframe error
# if "Phone_Prefixes" in dataframe.columns:
#     continue
# else:
#     print("Error: 'Phone_Prefixes not found in DataFrame'")
