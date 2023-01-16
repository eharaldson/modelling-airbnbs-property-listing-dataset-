from tabular_data import load_airbnb
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing

class AirbnbNightlyPriceImageDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.X, self.y = load_airbnb()
        scaler = preprocessing.StandardScaler()
        scaler.fit(self.X)
        self.X = scaler.transform(self.X)

        self.y = self.y.to_numpy()

    def __getitem__(self, index) -> tuple[torch.tensor, torch.tensor]:
        return (torch.tensor(self.X[index]).float(), torch.tensor(self.y[index]).float())

    def __len__(self) -> int:
        return len(self.y)

class AirbnbCategoryImageDataset(Dataset):

    def __init__(self):
        super().__init__()
        X, y = load_airbnb(label_name='bedrooms')
        self._scale_data(X)
        self._encode_labels(y)

    def _encode_labels(self, y):
        le = preprocessing.LabelEncoder()
        self.y = le.fit_transform(np.ravel(y))

    def _scale_data(self, X):
        categories = X[['category1', 'category2', 'category3', 'category4']]
        X.drop(['category1', 'category2', 'category3', 'category4'], axis=1, inplace=True)
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
        self.X = np.concatenate((X, categories), axis=1)

    def __getitem__(self, index) -> tuple[torch.tensor, torch.tensor]:
        return (torch.tensor(self.X[index]).float(), torch.tensor(self.y[index]))

    def __len__(self) -> int:
        return len(self.y)
