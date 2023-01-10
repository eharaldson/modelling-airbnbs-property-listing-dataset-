from tabular_data import load_airbnb
from torch.utils.data import Dataset
import torch

class AirbnbNightlyPriceImageDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.X, self.y = load_airbnb()

    def __getitem__(self, index) -> tuple[torch.tensor, torch.tensor]:
        return (torch.tensor(self.X.iloc[index]), torch.tensor(self.y.iloc[index]))

    def __len__(self) -> int:
        return len(self.y)

