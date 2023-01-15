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
        X, self.y = load_airbnb(label_name='bedrooms')
        categories = X[['category1', 'category2', 'category3', 'category4']]
        X.drop(['category1', 'category2', 'category3', 'category4'], axis=1, inplace=True)
        scaler = preprocessing.StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        self.X = np.concatenate((X, categories), axis=1)

    def __getitem__(self, index) -> tuple[torch.tensor, torch.tensor]:
        return (torch.tensor(self.X[index]).float(), torch.tensor(self.y.iloc[index]))

    def __len__(self) -> int:
        return len(self.y)
