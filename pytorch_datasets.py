from tabular_data import load_airbnb
from torch.utils.data import Dataset
import torch

from sklearn import preprocessing

class AirbnbNightlyPriceImageDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.X, self.y = load_airbnb()
        scaler = preprocessing.StandardScaler()
        scaler.fit(self.X)
        self.X = scaler.transform(self.X)

    def __getitem__(self, index) -> tuple[torch.tensor, torch.tensor]:
        return (torch.tensor(self.X[index]).float(), torch.tensor(self.y.iloc[index]).float())

    def __len__(self) -> int:
        return len(self.y)

