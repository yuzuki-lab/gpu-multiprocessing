import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision
from torchvision import models
from torchvision import datasets, transforms
from tqdm import tqdm
import scipy
import pandas as pd
import yaml
import numpy as np
import os
import timm
import random
import wandb
import warnings

def torch_fix_seed(config):
    seed = config['seed']
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True


def main():
    warnings.simplefilter('ignore')

    with open('config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="flowers",
        name='center_crop2',
        

        # track hyperparameters and run metadata
        config={
        "architecture": config['model'],
        "dataset": "flowers",
        "epochs": config['epoch'],
        })

    device = 'cuda'
    torch_fix_seed(config)

    # world_size = int(os.environ["WORLD_SIZE"])
    # ngpus_per_node = torch.cuda.device_count()
    # world_size = ngpus_per_node * world_size

    
    model = timm.create_model(config['model'], pretrained=True, num_classes=config['num_class']).to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    if config['optimizer'] == "adamw":
        optimizer = torch.optim.AdamW(model.parameters())
    elif config['optimizer'] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['0.9'])
    
    scheduler = StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])

    normalize = transforms.Normalize(mean=config['mean'],std=config['std'])

    matdata = scipy.io.loadmat("/home/yishido/DATA/flower/imagelabels.mat")
    labels = matdata['labels'][0]
    images = ['image_{:05}.jpg'.format(i + 1) for i in range(len(labels))]
    image_label_df = pd.DataFrame({'image': images, 'label': labels})

    class CustomDataset(Dataset):
        def __init__(self, dataframe, transform=None):
            self.dataframe = dataframe
            self.transform = transform

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            img_name = f'/home/yishido/DATA/flower/flowers/{self.dataframe.iloc[idx, 0]}'  # 画像ファイル名は1列目と仮定
            label = self.dataframe.iloc[idx, 1]  # ラベルは2列目と仮定

            # 画像を読み込む
            img = Image.open(img_name).convert('RGB')

            # 任意の前処理を適用
            if self.transform:
                img = self.transform(img)

            return img, label
    
    train_data, val_data = train_test_split(image_label_df, test_size=0.2, random_state=32)

    train_dataset = CustomDataset(
        train_data,
        transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(256), 
                                transforms.CenterCrop(256),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                normalize]),
    )

    val_dataset = CustomDataset(
        val_data,
        transform=transforms.Compose([transforms.ToTensor(), 
                                transforms.Resize(256),
                                transforms.CenterCrop(config['image_size']),
                                normalize]),
    )

    train_loader = torch.utils.data.DataLoader(
                train_dataset, 
                batch_size=config['batch_size'],
                shuffle=True, 
                num_workers=config['num_workers'],
                pin_memory=True, )
                # sampler=train_loader)

    val_loader = torch.utils.data.DataLoader(
                val_dataset, 
                batch_size=config['batch_size'], 
                shuffle=False,
                num_workers=config['num_workers'],
                pin_memory=True, )
                # sampler=val_loader)

    print(f"Data loaded: there are {len(train_dataset)} train images.")
    print(f"Data loaded: there are {len(val_dataset)} val images.")

    train_loss_list = [] 
    train_accuracy_list = [] 
    val_loss_list = [] 
    val_accuracy_list = [] 

    print("Starting training !")
    for epoch in range(0,config['epoch']):

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, device, config)
        val_loss, val_acc = val(val_loader, model, criterion, epoch, device, config)

        print("--------------------------------------------------------------------------------------------")
        print(f"{epoch}epoch")
        print(f"Trian_Loss: {train_loss:.4f}, Train_Accuracy: {train_acc:.4f}")
        print(f"Test_Loss: {val_loss:.4f}, Test_Accuracy: {val_acc:.4f}")

        wandb.log({"train_accuracy": train_acc,
                   "train_loss": train_loss,
                   "val_accuracy": val_acc,
                   "val_loss" : val_loss,
                   "epoch" : epoch
                   })

        train_loss_list.append(train_loss)
        train_accuracy_list.append(train_acc)
        val_loss_list.append(val_loss)
        val_accuracy_list.append(val_acc)

        if epoch % config['save_epoch'] == 0:
            save_path = config['save_path']
            save_file = f'{save_path}/checkpoint-{epoch}.pth'
            torch.save(model, save_file)
    
    torch.save(model, f'{save_path}/final.pth')

    wandb.alert(
        title='学習が完了しました。',
        text=f'final_val_acc :{val_acc}',
    )

    wandb.finish()


def train(train_loader, model, criterion, optimizer, epoch, device, config):
    model.train()

    train_losses = 0
    train_accuracies = 0
    
    for images, labels in train_loader:
        
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()       

        y_pred_prob = model(images)
        loss = criterion(y_pred_prob, labels)

        loss.backward()
        optimizer.step()

        train_losses += loss.item()
        y_pred_labels = torch.max(y_pred_prob, 1)[1]
        train_accuracies += torch.sum(y_pred_labels == labels).item() / len(labels)

    epoch_train_loss = train_losses / len(train_loader)
    epoch_train_accuracy = train_accuracies / len(train_loader)

    return epoch_train_loss, epoch_train_accuracy

def val(val_loader, model, criterion, epoch, device, config):
    model.eval()

    test_losses = 0
    test_accuracies = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            y_pred_prob = model(images)
            
            loss = criterion(y_pred_prob, labels)
            
            test_losses += loss.item()
            y_pred_labels = torch.max(y_pred_prob, 1)[1]
            test_accuracies += torch.sum(y_pred_labels == labels).item() / len(labels)
    
    epoch_test_loss = test_losses / len(val_loader)
    epoch_test_accuracy = test_accuracies / len(val_loader)    
    
    return epoch_test_loss, epoch_test_accuracy

if __name__ == '__main__':
    main()