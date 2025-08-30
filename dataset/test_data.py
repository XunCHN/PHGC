import torch
import os
from torch.utils.data import Dataset, DataLoader
import json
  
class REAL(Dataset):  
    def __init__(self,domain,test_type): 




        lab_vid_feats = "../data/lab/vid_feats/lab_real_vid_feats.pt"
        kit_vid_feats = "../data/kit/vid_feats/kitchen_real_vid_feats.pt"
        daily_vid_feats = "../data/daily/vid_feats/daily_real_vid_feats.pt"

        if test_type =='nt':
            lab_name_file = "../data/lab/test_lab_nt.json"
            kit_name_file = "../data/kit/test_kit_nt.json"
            daily_name_file = "../data/daily/test_nt.json"
            
            lab_des_file = "../data/lab/test_nt_des.json"
            kit_des_file = "../data/kit/test_nt_des.json"
            daily_des_file = "../data/daily/test_nt_des.json"
        
        elif test_type == 'ns':
            
            lab_name_file = "../data/lab/test_lab_ns.json"
            kit_name_file = "../data/kit/test_kit_ns.json"
            daily_name_file = "../data/daily/test_ns.json"
            
            lab_des_file = "../data/lab/test_ns_des.json"
            kit_des_file = "../data/kit/test_ns_des.json"
            daily_des_file = "../data/daily/test_ns_des.json"
        
        elif test_type == 'os':
            
            lab_name_file = "../data/lab/test_lab_os.json"
            kit_name_file = "../data/kit/test_kit_os.json"
            daily_name_file = "../data/daily/test_os.json"
            
            lab_des_file = "../data/lab/test_os_des.json"
            kit_des_file = "../data/kit/test_os_des.json"
            daily_des_file = "../data/daily/test_os_des.json"
        
        else:
            raise ValueError('None exist test type!')
        



        if domain == 'lab':
            self.vid_feats = torch.load(lab_vid_feats)
            with open (lab_name_file,'r') as file:
                self.name = json.load(file)
            with open (lab_des_file,'r') as file:
                self.des = json.load(file)


        elif domain == 'kit':
            self.vid_feats = torch.load(kit_vid_feats)
            with open (kit_name_file,'r') as file:
                self.name = json.load(file)
            with open (kit_des_file,'r') as file:
                self.des = json.load(file) 


        elif domain == 'daily':
            self.vid_feats = torch.load(daily_vid_feats)
            with open (daily_name_file,'r') as file:
                self.name = json.load(file)
            with open (daily_des_file,'r') as file:
                self.des = json.load(file)







    def __len__(self):  
        return len(self.name) 
  
    def __getitem__(self, idx):  

        add = self.name[idx][0]
        label = self.name[idx][1]
        des = self.des[self.name[idx][2]]
        vid_feat = self.vid_feats[add]


        return add,torch.tensor(float(label)),des,vid_feat
    
