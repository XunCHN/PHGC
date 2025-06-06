import torch
import os
from torch.utils.data import Dataset

  
class graphDataset(Dataset):  
    def __init__(self, task_type, args):  

        if task_type == 'train':
            self.data = torch.load(os.path.join(args.data_path,"train.pt"))
        else:
            self.data = torch.load(os.path.join(args.data_path, task_type +".pt"))

        self.length = len(self.data)
  
    def __len__(self):  
        return self.length  
  
    def __getitem__(self, idx):  
        
        file_name, label, vid_feat, text_graph, hypotheses = self.data[idx]

        return file_name, vid_feat, text_graph, hypotheses, torch.tensor(float(label))
    
