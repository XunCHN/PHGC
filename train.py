import os

from utils.nesy_arguments import Arguments
from utils.dataset_utils import *
from utils.distributed_utils import *
from utils.feature_extraction import *
from utils.rnn import RNNEncoder
from dataset.train_data import SIM
from dataset.test_data import REAL
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torchmetrics import MetricCollection, Accuracy, F1Score, ConfusionMatrix
import torch.distributed as dist
from torch.utils.data import random_split,Sampler,Subset,DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from model import SR


def custom_collate(batch):
    """
    Custom collate function to handle DiGraph objects.
    """
    add,label,des,vid_feat = zip(*batch)
    
    return add,label,des,vid_feat
def train_custom_collate(batch):
    """
    Custom collate function to handle DiGraph objects.
    """
    add,label,des,vid_feat,name_real,des_real,vid_feat_real = zip(*batch)
    
    return add,label,des,vid_feat,name_real,des_real,vid_feat_real
def test_model(domain,test_type):

    model.eval()

    count = 0
    count_pos = 0
    count_neg = 0
    test_set = REAL(domain,test_type)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,num_workers=args.num_workers, shuffle=False,collate_fn=custom_collate)

    
    with torch.no_grad():
        print("Testing on {} domain".format(domain))
        for (file_names, labels, hypotheses, vid_feats)  in tqdm(test_loader):
            labels = torch.tensor(labels).cuda()
            labels = labels.float()

            dp_outputs = []
            map_outputs = []
            valid_indices = []  
            for i in range(len(file_names)):
                try:
                    map_pred, _,= model(file_names[i], hypotheses[i], vid_feats[i],None,None,None,None,None,None,None,None)
                    map_outputs.append(map_pred)
                    valid_indices.append(i)
                except Exception as e:
                    if epoch == 1:
                        untrained_log_file.write(f"File: {file_names[i]}\n")
                        untrained_log_file.write(f"Label: {labels[i].item()}\n")
                        untrained_log_file.write(f"Error: {str(e)}\n")
                        untrained_log_file.write(f"Video feature length: {len(vid_feats[i])}\n\n")
                        untrained_log_file.flush()
                    
                    if labels[i] == 1:
                        count_pos += 1
                    else:
                        count_neg += 1
                    continue
            if not valid_indices:
                continue
                

            valid_labels = labels[valid_indices]
            map_preds = torch.stack(map_outputs).view(-1)
            test_metrics_map.update(preds=map_preds, target=valid_labels)
            test_metrics_map1.update(preds=1-map_preds, target=1-valid_labels)
            

           

            

        dist.barrier()

        test_acc_map, test_f1_map = list(test_metrics_map.compute().values())
        test_acc_map1, test_f1_map1 = list(test_metrics_map1.compute().values())

        test_acc_map=(test_acc_map + test_acc_map1) / 2
        test_f1_map=(test_f1_map + test_f1_map1) / 2

        dist.barrier()
        if is_main_process():

            print('Test Acc_map: {} | Test F1_map: {}'.format(test_acc_map, test_f1_map))

            log_file.write( test_type +': Test Acc_map: ' + str(test_acc_map.item()) + ' | Test F1_map: ' + str(test_f1_map.item()) + "\n")

            log_file.flush()
        print(count)
    return test_acc_map, test_f1_map



def train_epoch(model, train_loader,epoch, best_acc,sim_prototypes,real_prototypes,video_ass,total_epochs):

    model.train()
    train_loss = []
    cdel=[]
    cal=[]
    count_pos = 0
    count_neg = 0

    
    print("Training epoch: {}".format(epoch))
    for (file_names, labels, hypotheses, vid_feats,real_names,real_hypothese,realv_feats) in (tqdm(train_loader)):
        labels = torch.tensor(labels).cuda()
        labels=labels.float().cuda()
        map_outputs = []
        valid_indices = [] 
        batch_domain_losses = []

        for i in range(len(file_names)):
            try:
                map_pred, domain_loss= model(
                        file_names[i], 
                        hypotheses[i], 
                        vid_feats[i],
                        real_names[i],
                        real_hypothese[i],
                        realv_feats[i],
                        sim_prototypes,
                        real_prototypes,
                        video_ass,
                        epoch,
                        total_epochs
                    )
                map_outputs.append(map_pred)
                valid_indices.append(i)
                batch_domain_losses.append(domain_loss)
            except Exception as e:
                if epoch == 1:
                    untrained_log_file.write(f"File: {file_names[i]}\n")
                    untrained_log_file.write(f"Label: {labels[i].item()}\n")
                    untrained_log_file.write(f"Error: {str(e)}\n")
                    untrained_log_file.write(f"Video feature length: {len(vid_feats[i])}\n\n")
                    untrained_log_file.flush()
                
                if labels[i] == 1:
                    count_pos += 1
                else:
                    count_neg += 1
                continue
            
        if not valid_indices:
            continue
            
        valid_labels = labels[valid_indices]
        map_preds = torch.stack(map_outputs).view(-1)
    
        
        domain_loss = torch.stack(batch_domain_losses).mean()
        
        map_loss = bce_loss(map_preds, valid_labels)+0.1*domain_loss
        
        optimizer_map.zero_grad()
        
        map_loss.backward()
        
        optimizer_map.step()

        valid_labels = valid_labels.type(torch.int)
        
        
        
     

    acc, f1 = list(train_metrics.compute().values())
    print('Train Loss: {}'.format(np.array(train_loss).mean()))
  
    print('pos:%d' %count_pos)
    print('neg:%d' %count_neg)

    if is_main_process():

        print('Epoch: {} | Train Acc: {} | Train F1: {} '.format(epoch, acc, f1))
        print('pos: %d  neg: %d' %(count_pos, count_neg))

        
        log_file.write('Epoch: ' + str(epoch) + ' | Train Acc: ' + str(acc.item()) +' | Train F1: ' + str(f1.item()) +"\n")
        log_file.write('Train Loss: {}\n'.format(np.array(train_loss).mean()))
        log_file.write('pos: %d  neg: %d' %(count_pos, count_neg))
        log_file.write("\n")

        if acc > torch.tensor(best_acc):
            best_acc = acc.item()
            print('============== Saving best model(s) ================')
            torch.save(model.module.state_dict(), model_ckpt_path_train)
            optimizer={'optimizer_dp': optimizer_dp.state_dict(),
                      'optimizer_map': optimizer_map.state_dict()}
            torch.save(optimizer, optimizer_ckpt_path)
            print('Model saved to {}'.format(model_ckpt_path_train))
            print('Optimizer state saved to {}'.format(optimizer_ckpt_path))

        log_file.flush()
        



    return best_acc

if __name__ == '__main__':

    dist_url = "env://"
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(
        backend="nccl",
        init_method=dist_url,
        world_size=world_size,
        rank=rank)
    torch.cuda.set_device(local_rank)

    dist.barrier()
    args = Arguments()



    
    logger_filename = 'log_files/log_{}_{}.txt'.format(args.dataset,args.run_id)
    log_file = open(logger_filename, "a")
    log_file.write(str(args) + '\n')

    untrained_filename = 'untrained_files/untrained_{}_{}.txt'.format(args.dataset,args.run_id)
    untrained_log_file = open(untrained_filename, "w")

    model_ckpt_path_train = "ckpts/weight_{}_{}.pth".format(args.dataset,args.run_id)
    optimizer_ckpt_path = "ckpts/optimizer_{}_{}.pth".format(args.dataset,args.run_id)

    # dataset = SIM(args.dataset)
    # train_loader= DataLoader(dataset, batch_size=args.batch_size, shuffle = True, collate_fn=train_custom_collate)


    
    visual_model, vid_feat_size, transform = initiate_visual_module(args.visual_feature_extractor)
    visual_model.cuda()
    visual_model = DDP(visual_model, device_ids=[local_rank])

    if not args.finetune:
        visual_model.eval()
    else:
        visual_model_ckpt_path = os.path.join(os.getcwd(), "{}.pth".format(args.visual_feature_extractor))

    if args.visual_feature_extractor == args.text_feature_extractor:
        _, tokenizer_text, text_feat_size = initiate_text_module(args.text_feature_extractor)
        text_model = visual_model  # for clip/coca model
    else:
        text_model, tokenizer_text, text_feat_size = initiate_text_module(
            feature_extractor=args.text_feature_extractor)
        text_model.cuda()
        text_model = DDP(text_model, device_ids=[local_rank])
    text_model.eval()

    hsize = 150
    model = SR(vid_embed_size=vid_feat_size,
                     hsize=hsize,
                     rnn_enc=RNNEncoder,
                     text_model=text_model,
                     text_feature_extractor=args.text_feature_extractor,
                     tokenizer=tokenizer_text,
                     context_encoder=args.context_encoder)

    model.load_state_dict(torch.load('../ckpts/kit_model.pth'))
    model.cuda()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    all_params = list(model.parameters())
    optimizer_dp = optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay) 
    optimizer_map = optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)


    bce_loss = nn.BCELoss()
   
    train_metrics = MetricCollection([Accuracy(threshold=0.5, dist_sync_on_step=True, task='binary'),
                                F1Score(threshold=0.5, dist_sync_on_step=True, task='binary')]).cuda()

    test_metrics_map = MetricCollection([Accuracy(threshold=0.5, dist_sync_on_step=True, task='binary'),
                                F1Score(threshold=0.5, dist_sync_on_step=True, task='binary')]).cuda()
    test_metrics_map1 = MetricCollection([Accuracy(threshold=0.5, dist_sync_on_step=True, task='binary'),
                                F1Score(threshold=0.5, dist_sync_on_step=True, task='binary')]).cuda()
    

    test_metrics_map = test_metrics_map.clone(prefix='test_map')

    test_metrics_map1 = test_metrics_map1.clone(prefix='test_map')

    
    train_metrics = train_metrics.clone(prefix='train')
    
    
    sim_prototypes = torch.load('sim_anchors.pt').float().cuda()
    real_prototypes = torch.load('real_anchors.pt').float().cuda()
    video_ass=torch.load('video_cluster_gt.pt')
    sim_prototypes = sim_prototypes.cuda()
    real_prototypes = real_prototypes.cuda()
    best_acc = 0.
    test_best_f1 = 0.
    for epoch in range(1, args.epochs+1):

        f1s = []
        # for domain in ['csv','kit','adl']:
        for domain in [args.dataset]:
            print("Testing on {} domain".format(domain))
            log_file.write("Testing on {} domain".format(domain) + '\n')
            for test_type in ['nt','ns','os']:
                log_file.write("testing on {}".format(test_type) + '\n')
                test_acc,test_f1 = test_model(domain,test_type)
                test_metrics_map.reset()
                test_metrics_map1.reset()
                f1s.append(test_f1)
            log_file.write("\n")
        log_file.write("\n\n")

        print("\n")

        train_metrics.reset()

        

    cleanup()
    log_file.close()
    untrained_log_file.close()
    print('Done!')





