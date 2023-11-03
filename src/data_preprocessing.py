import pandas as pd

# PREPROCESSING STEPS

def extract_phone_prefixes(dataframe):
    dataframe = dataframe.copy()
    dataframe['Phone_Prefixes'] = dataframe['Telephone_Number'].str[:4]
    return dataframe[['Phone_Prefixes']]