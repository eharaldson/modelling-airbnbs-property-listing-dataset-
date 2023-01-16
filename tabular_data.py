from ast import literal_eval

import pandas as pd
import numpy as np
from sklearn import preprocessing
import os

def remove_rows_with_missing_ratings(df_null):
    df_null = df_null.iloc[:,:-1]
    df = df_null.dropna(subset=['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating'])
    return df.copy()

def description_func(x):
    ls = literal_eval(x)
    ls.remove('About this space')
    if '' in ls:
        ls.remove('')
        return ''.join(ls)
    else:
        return ''.join(ls)

def combine_description_strings(df):
    df = df.copy().dropna(subset={'Description'})
    df['Description'] = df['Description'].apply(description_func)
    return df.copy()

def set_default_feature_value(df):
    df[['guests', 'beds', 'bathrooms', 'bedrooms']] = df[['guests', 'beds', 'bathrooms', 'bedrooms']].fillna(1)
    return df.copy()

def clean_tabular_data(df):
    df = remove_rows_with_missing_ratings(df)
    df = combine_description_strings(df)
    df = set_default_feature_value(df)
    df = df.reset_index(drop=True)
    return df

def to_one_hot(labels, max_labels: int = None):
    if max_labels is None:
        max_labels = np.max(labels) + 1
    return np.eye(max_labels)[labels]

def load_airbnb(label_name='Price_Night'):
    cwd = os.getcwd()
    df = pd.read_csv(os.path.join(cwd, 'data/tabular_data/clean_listing.csv'), index_col='ID')
    labels = df[[label_name]]
    features = df.drop(label_name, axis=1, inplace=False).select_dtypes('number')
    if label_name == 'bedrooms':
        le = preprocessing.LabelEncoder()
        category_column = le.fit_transform(df['Category'])
        ohe_column = to_one_hot(category_column)[:,1:]
        features['category1'] = ohe_column[:,0]
        features['category2'] = ohe_column[:,1]
        features['category3'] = ohe_column[:,2]
        features['category4'] = ohe_column[:,3]
    return (features, labels)

if __name__ == "__main__":
    cwd = os.getcwd()
    df = pd.read_csv(os.path.join(cwd, 'data/tabular_data/listing.csv'))

    df.info()

    print(df.head())