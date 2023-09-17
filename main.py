import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR
import yaml
import os
import timm

# 分散トレーニングの初期化
def setup(rank, world_size):
    dist.init_process_group(
        "nccl",
        init_method="tcp://localhost:12345",
        rank=rank,
        world_size=world_size
        )

def main():
    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    model = timm.create_model(config['model'], pretrained=False, num_classes=config['num_class']).to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    if config['optimizer'] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters())
    elif config['optimizer'] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['0.9'])
    
    scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])


    
    traindir = os.path.join(config['data_path'], 'train')
    valdir = os.path.join(config['data_path'], 'val')
    normalize = transforms.Normalize(mean=config['mean'],std=config['std'])
    
    train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                transforms.RandomResizedCrop(config['image_size']),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize]))

    val_dataset = datasets.ImageFolder(
                valdir,
                transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(config['image_size']),
                transforms.ToTensor(),
                normalize,]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)  

    train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=config['batch_size'], 
                num_workers=config['num_workers'], 
                pin_memory=True, 
                sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=config['batch_size'], 
                shuffle=False,
                num_workers=config['num_workers'], 
                pin_memory=True, 
                sampler=val_sampler)

    print(f"Data loaded: there are {len(train_dataset)} train images.")
    print(f"Data loaded: there are {len(val_dataset)} val images.")


    for epoch in range(0,config['epoch']):
        train_sampler.set_epoch(epoch)

        train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, config)

def train_one_epoch():

def val():
    


if __name__ == '__main__':
    main()