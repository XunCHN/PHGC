import os
from utils.feat_extraction import *
import math
import torch
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import cv2
from torchvision import transforms
from EgoTV.args import Arguments 
from dataset.csv_read_vid import CSV_VID
from dataset.egotv_read_vid import EGO_VID
from torch.utils.data import DataLoader

def transform_image(video_frame, type='rgb'):
    if type == 'rgb':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif type == 'flow':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif type == 'mvit':
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        ])
    elif type == 'coca':
        transform = transforms.Compose([
            transforms.ToPILImage()
        ])
    return transform(video_frame)


def process_batch(data_batch):

    video_features_batch = []  # transforms + visual model features
    frames_per_segment = args.fp_seg


    for filepath in data_batch:


        type = 'rgb'

        video_frames = []
        video = cv2.VideoCapture(filepath)
        success = video.grab()
        sample_rate = args.sample_rate
        fno = 0
        while success:
            if fno % sample_rate == 0:
                _, img = video.retrieve()
                video_frames.append(transform_image(img, type=type))
            success = video.grab()
            fno += 1

        video_frames = torch.stack(video_frames).cuda()  # [t, c, h, w]

        video_frames = extract_video_features(video_frames, model=visual_model,
                                              feature_extractor=args.visual_feature_extractor,
                                              feat_size=vid_feat_size).reshape(1, -1, vid_feat_size) 


        b, t, _ = video_frames.shape
        num_segments = math.ceil(t / frames_per_segment)
        to_pad = num_segments * frames_per_segment - t
        # zero-padding to match the number of frames per segment
        video_frames = torch.cat((video_frames, torch.zeros(b, to_pad, vid_feat_size).cuda()), dim=1)

        # [num_segments, frames_per_segment, 512]
        video_frames = video_frames.reshape(b * num_segments, frames_per_segment, vid_feat_size)

        video_features_batch.append(video_frames.cpu())


    
    return data_batch,video_features_batch


def iterate(dataloader, validation=False):
    for data_batch,_ in tqdm(dataloader):
        yield process_batch(data_batch)


if __name__=='__main__':

    dist_url = "env://"  # default
    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank)
    # this will make all .cuda() calls work properly
    torch.cuda.set_device(local_rank)
    # synchronizes all the threads to reach this point before moving on
    dist.barrier()



    dataset = "CSV"
    # dataset = "EgoTV"


    args = Arguments()

    if dataset == 'EgoTV':
        all_set = EGO_VID("")

    elif dataset =='CSV':
        train =CSV_VID(train = True)
        test = CSV_VID(train = False)
        all_set = train + test # use one file to save all video feats, including train set and test set

    data_loader = DataLoader(all_set, batch_size = args.batch_size,shuffle = False)
                        
    
    visual_model, vid_feat_size, transform = initiate_visual_module(args.visual_feature_extractor)
    visual_model.cuda()
    visual_model = DDP(visual_model, device_ids=[local_rank])



    dic = {}

    for data_batch, video_feats in tqdm(iterate(data_loader), desc='Train'):
        for file_name,vid_feat in zip(data_batch,video_feats):
            dic[file_name] = vid_feat



    torch.save(dic,os.join(args.data_path,"vid_feats.pt"))

