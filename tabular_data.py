import pandas as pd
import numpy as np
from ast import literal_eval

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

if __name__ == "__main__":

    df_clean = pd.read_csv(r'./airbnb-property-listings/tabular_data/clean_listing.csv')
    print(df_clean.head())
    df = pd.read_csv(r'./airbnb-property-listings/tabular_data/listing.csv')

    df = clean_tabular_data(df)

    df.to_csv(r'./airbnb-property-listings/tabular_data/clean_listing.csv', index=False)