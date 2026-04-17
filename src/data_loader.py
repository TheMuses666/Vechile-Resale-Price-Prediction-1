import pandas as pd
import pickle

def read_files():
    with open('rf_model.pkl','rb') as f:
        model = pickle.load(f)

    with open('feature_columns.pkl', 'rb') as f:
        feature_columns = pickle.load(f)

    return model, feature_columns


def load_data(path='data/raw/data.csv'):
    df = pd.read_csv(path)
    return df
