import torch
from torch.utils.data import Dataset

class EGO_VID(Dataset):  
    def __init__(self, file_path):  
        self.data = torch.load(file_path)
        self.length = len(self.data)
  
    def __len__(self):  
        return self.length  
  
    def __getitem__(self, idx):  
        file_name,label = self.data[idx]

        return file_name,label