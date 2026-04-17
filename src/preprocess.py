import pandas as pd
import numpy as np
from datetime import datetime

def preprocess():
    return


def handle_missing_value(df,X_train,X_test):
    df = df.copy()
    X_train = X_train.copy()
    X_test = X_test.copy()

    # Fill the Numeric Columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        X_train[col] = X_train[col].fillna(X_train[col].median())
        X_test[col] = X_test[col].fillna(X_train[col].median())

    # Fill the Categorical Columns
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        X_train[col] = X_train[col].fillna(X_train[col].mode())
        X_test[col] = X_test[col].fillna(X_train[col].median())
    
    return df, X_train, X_test

def feature_engineering(df):
    # Create new feature instead of year of registration
    current_year = datetime.now().year
    df = df.copy()

    df['car_age'] = current_year - df['year_of_registration']
    df = df.drop('year_of_registration', axis=1)

    # Drop the unnecessary column

    df = df.drop(['public_reference'], axis=1)
    df = df.drop(['reg_code'], axis=1)

    if 'price' in df.columns:
        df['price'] = np.log1p(df['price'])

    return df

def fix_outlier(X_train,X_test,cols):
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in cols:
        q1 = X_train[col].quantile(0.25)
        q3 = X_train[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        X_train[col] = X_train[col].clip(upper=upper_bound,lower=lower_bound)
        X_test[col] = X_test[col].clip(upper=upper_bound,lower=lower_bound)

    return X_train, X_test


def encoding_binary(df):
    df = df.copy()
    mappings = {
        'vehicle_condition': {'USED': 1, 'NEW': 0},
        'crossover_car_and_van': {True: 1, False: 0}
    }

    for col, mapping in mappings.items():
        if col in df.columns:

            if df[col].dtype == 'object':
                df[col] = df[col].str.upper()

            df[col] = df[col].map(mapping).fillna(0)

    return df
