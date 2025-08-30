import torch
import os
from torch.utils.data import Dataset, DataLoader
import json

class SIM(Dataset):  
    def __init__(self, domain="all",use_real=False):  
        self.domain = domain
        

        if domain == "all" or domain == "lab":
            lab_vid_feats = torch.load("../data/vid_feats/lab_sim_vid_feats.pt")
            
            with open("../data/lab/train_sim.json", 'r') as file:
                lab_name = json.load(file)
            
            with open("../data/lab/train_sim_des.json", 'r') as file:
                lab_des = json.load(file)

            lab_real_feats= torch.load("../data/vid_feats/lab_real_vid_feats.pt")
            with open("../data_da/lab/train_real.json", 'r') as file:
                lab_name_real = json.load(file)
            with open("../data_da/lab/train_real_des.json", 'r') as file:
                lab_des_real = json.load(file)
        else:
            lab_vid_feats = {}
            lab_name = []
            lab_des = {}
            lab_real_feats = {}
            lab_name_real = []
            lab_des_real = {}
            
        if domain == "all" or domain == "kit":
            kit_vid_feats = torch.load("../data/kit/vid_feats/kitchen_sim_vid_feats.pt")
            with open("../data/kit/train_sim.json", 'r') as file:
                kit_name = json.load(file)
            with open("../data/kit/train_sim_des.json", 'r') as file:
                kit_des = json.load(file)

            kit_real_feats= torch.load("../data/kit/vid_feats/kitchen_real_vid_feats.pt")
            with open("../data_da/kit/train_real.json", 'r') as file:
                kit_name_real = json.load(file)
            with open("../data_da/kit/train_real_des.json", 'r') as file:
                kit_des_real = json.load(file)
        else:
            kit_vid_feats = {}
            kit_name = []
            kit_des = {}
            kit_real_feats = {}
            kit_name_real = []
            kit_des_real = {}
            
        if domain == "all" or domain == "daily":
            daily_vid_feats = torch.load("../data/ADL/vid_feats/daily_sim_vid_feats.pt")
            with open("../data/daily/train_sim.json", 'r') as file:
                daily_name = json.load(file)
            with open("../data/daily/train_sim_des.json", 'r') as file:
                daily_des = json.load(file)

            daily_real_feats= torch.load("../data/ADL/vid_feats/daily_real_vid_feats.pt")
            with open("../data/daily/train_real.json", 'r') as file:
                daily_name_real = json.load(file)
            with open("../data/daily/train_real_des.json", 'r') as file:
                daily_des_real = json.load(file)
        else:
            daily_vid_feats = {}
            daily_name = []
            daily_des = {}
            daily_real_feats = {}
            daily_name_real = []
            daily_des_real = {}
        

        self.all_feats = lab_vid_feats | kit_vid_feats | daily_vid_feats
        self.all_name = lab_name + kit_name + daily_name
        self.all_des = lab_des | kit_des | daily_des
        self.all_feats_real = lab_real_feats | kit_real_feats | daily_real_feats
        self.all_name_real = lab_name_real + kit_name_real + daily_name_real
        self.all_des_real = lab_des_real | kit_des_real | daily_des_real
    
        print(f"Loaded {domain} dataset with {len(self.all_name)} samples")


    def __len__(self):  
        return len(self.all_name)
  
    def __getitem__(self, idx, use_real=False):  

        add = self.all_name[idx][0]
        label = self.all_name[idx][1]
        des = self.all_des[self.all_name[idx][2]]
        vid_feat = self.all_feats[add]
        

        add_real = self.all_name_real[idx][0]
        des_real = self.all_des_real[self.all_name_real[idx][2]]
        vid_feat_real = self.all_feats_real[add_real]
        return add, torch.tensor(float(label)), des, vid_feat,add_real,des_real, vid_feat_real

    
