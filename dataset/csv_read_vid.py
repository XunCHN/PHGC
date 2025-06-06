from torch.utils.data import Dataset


# Set your own raw video path
VID_PATH = ""

class CSV_VID(Dataset):  
    def __init__(self,train = True):  

        if train == True:
            file_name = "data_split/CSV_NL/train_split.txt"
        else:
            file_name = "data_split/CSV_NL/train_split.txt"

        with open(file_name, 'r', encoding='utf-8') as file:  
            self.data = file.readlines()

        self.length = len(self.data)
        self.train  =train
  
    def __len__(self):  
        return self.length  
  
    def __getitem__(self, idx):  
        name, number = self.data[idx].split()
        if self.train == True:
            addr = VID_PATH + number +"/" + name + ".mp4"
        else:
            addr = VID_PATH + number +"/" + name

        return addr, number
    