from tabular_data import load_airbnb
from torch.utils.data import Dataset
import torch
import numpy as np

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
        self.X, self.y = load_airbnb(label_name='Category')
        scaler = preprocessing.StandardScaler()
        scaler.fit(self.X)
        self.X = scaler.transform(self.X)

        le = preprocessing.LabelEncoder()
        le.fit(np.ravel(self.y))
        self.y = le.transform(np.ravel(self.y))

    def __getitem__(self, index) -> tuple[torch.tensor, torch.tensor]:
        return (torch.tensor(self.X[index]).float(), torch.tensor(self.y[index]).long())

    def __len__(self) -> int:
        return len(self.y)
