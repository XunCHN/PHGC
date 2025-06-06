import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Accuracy, F1Score
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import numpy as np
from dataset.dataset import graphDataset
from model.PHGC import PHGC
from utils.feat_extraction import *
from utils.rnn import RNNEncoder
from EgoTV.args import Arguments

def custom_collate(batch):
    """
    Custom collate function to handle DiGraph objects in batch.
    Returns:
        file_name, vid_feat, text_graph, hypotheses, label
    """
    file_name, vid_feat, text_graph, hypotheses, label = zip(*batch)
    return file_name, vid_feat, text_graph, hypotheses, label

def test_model(test_type):
    """
    Function to test the model on a given dataset type.
    Args:
        test_type (str): Type of test data to use ('nt', 'ns', 'nsc', 'abs')

    Returns:
        test_acc (float): Accuracy of the model on the test set
        test_f1 (float): F1 score of the model on the test set
    """
    # Load the test dataset and prepare the DataLoader
    test_set = graphDataset(test_type,args)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, collate_fn=custom_collate)

    with torch.no_grad():
        for file_name, vid_feat, text_graph, hypotheses, label in tqdm(test_loader):
            dp_preds, map_preds, labels = model(file_name, vid_feat, text_graph, hypotheses, label)
            labels = labels.type(torch.int)
            test_metrics.update(preds=dp_preds, target=labels)
            test_metrics.update(preds=map_preds, target=labels)

        # Compute metrics after the entire test dataset has been processed
        test_acc, test_f1 = list(test_metrics.compute().values())
        print(f'Test Acc: {test_acc} | Test F1: {test_f1}')
        log_file.write(f'{test_type}: Test Acc: {test_acc.item()} | Test F1: {test_f1.item()}\n')
        log_file.flush()

    return test_acc, test_f1

def train_epoch(model, train_loader, epoch, previous_best_acc):
    """
    Function to train the model for one epoch.
    Args:
        model (nn.Module): The model to be trained
        train_loader (DataLoader): The DataLoader for the training data
        epoch (int): The current epoch
        previous_best_acc (float): The previous best accuracy

    Returns:
        previous_best_acc (float): The updated best accuracy
    """
    model.train()
    train_loss = []
    count = 0

    for file_name, vid_feat, text_graph, hypotheses, label in tqdm(train_loader):
        try:
            dp_preds, map_preds, labels = model(file_name, vid_feat, text_graph, hypotheses, label)

            # Calculate losses
            dp_loss = bce_loss(dp_preds, labels)
            map_loss = bce_loss(map_preds, labels)

            # Zero the gradients for the optimizers
            optimizer_dp.zero_grad()
            optimizer_map.zero_grad()

            # Backpropagate the losses
            dp_loss.backward(retain_graph=True)
            map_loss.backward()

            # Update model parameters
            optimizer_dp.step()
            optimizer_map.step()

            train_loss.append((dp_loss.item(), map_loss.item()))
            labels = labels.type(torch.int)
            train_metrics.update(preds=dp_preds, target=labels)
            break

        except:
            print(f'Error processing {file_name}')
            count += 1
            continue

    acc, f1 = list(train_metrics.compute().values())
    print(f'Untrained num: {count}')
    print(f'Train Loss: {np.array(train_loss).mean()}')
    print(f'Epoch: {epoch} | Train Acc: {acc} | Train F1: {f1}')

    # Log training metrics
    log_file.write(f'Epoch: {epoch} | Train Acc: {acc.item()} | Train F1: {f1.item()}\n')
    log_file.write(f'Train Loss: {np.array(train_loss).mean()}\n')
    
    # Save the model if the accuracy is the best so far
    if acc > torch.tensor(previous_best_acc):
        previous_best_acc = acc.item()
        print('============== Saving best model ===============')
        torch.save(model.module.state_dict(), model_ckpt_path_train)
        log_file.flush()

    return previous_best_acc



if __name__ == '__main__':

    # Initialize distributed training
    dist_url = "env://"
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    dist.init_process_group(backend="nccl", init_method=dist_url, world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)

    # Parse arguments
    args = Arguments()

    # Setup logging
    logger_path = os.path.join("logs", args.log_name)
    log_file = open(logger_path, "w")
    log_file.write(str(args) + '\n')

    # Model checkpoint path
    model_ckpt_path_train = os.path.join("ckpt", args.ckpt_name)

    # Prepare datasets and DataLoader for training
    train_set = graphDataset('train', args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, collate_fn=custom_collate)

    # Initialize text model
    text_model, tokenizer_text, text_feat_size = initiate_text_module(feature_extractor=args.text_feature_extractor)
    text_model.cuda()
    text_model = DDP(text_model, device_ids=[local_rank])
    text_model.eval()

    # Initialize PHGC model
    hsize = 150
    model = PHGC(vid_embed_size=512, hsize=hsize, rnn_enc=RNNEncoder, text_model=text_model)
    model.cuda()
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Optimizers
    all_params = list(model.parameters())
    optimizer_dp = optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_map = optim.Adam(all_params, lr=args.lr, weight_decay=args.weight_decay)

    # Loss function
    bce_loss = nn.BCELoss()

    # Metrics
    metrics = MetricCollection([Accuracy(threshold=0.5, dist_sync_on_step=True, task='binary'),
                                F1Score(threshold=0.5, dist_sync_on_step=True, task='binary')]).cuda()
    test_metrics = MetricCollection([Accuracy(threshold=0.5, dist_sync_on_step=True, task='binary'),
                                     F1Score(threshold=0.5, dist_sync_on_step=True, task='binary')]).cuda()
    train_metrics = metrics.clone(prefix='train_')
    test_metrics = test_metrics.clone(prefix='test_')

    # Training loop
    best_acc = 0.
    for epoch in range(1, args.epochs + 1):

        # Train for one epoch
        best_acc = train_epoch(model, train_loader, epoch, previous_best_acc=best_acc)

        # Test the model on different test types
        for test_type in ['nt', 'ns', 'nsc', 'abs']:
            test_acc, test_f1 = test_model(test_type)
            test_metrics.reset()

        log_file.write("\n")
        train_metrics.reset()

    log_file.close()
    print('Training complete!')
