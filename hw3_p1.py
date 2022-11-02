import enum
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from PIL import Image
import torchvision.models as models
import imageio
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_pretrained_vit import ViT
import os


class p1_data(Dataset):
    def __init__(self, path, transform=None):
        self.imgpaths = glob.glob(os.path.join(path,'*.jpg'))
        self.transform = transform
    def __getitem__(self, index) :
        img = Image.open(self.imgpaths[index]).convert('RGB')
        label = int(os.path.basename(self.imgpaths[index]).split('_')[0])
        if self.transform is not None:

            img = self.transform(img)
        return img, label
    def __len__(self):
        return len(self.imgpaths)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = ViT('B_16_imagenet1k', pretrained=True, num_classes=37).to(device)

traindataset = p1_data('hw3_data/p1_data/train',transform=transforms.Compose([
    transforms.Resize(model.image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]))

valdataset = p1_data('hw3_data/p1_data/val',transform=transforms.Compose([
    transforms.Resize(model.image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]))
    
trainLoader = DataLoader(traindataset,batch_size=16)
valLoader = DataLoader(valdataset,batch_size=16)

for i, param in enumerate(model.parameters()):
    if(i<198):
        param.requires_grad = False

def train(model, epoch, log_interval=100):
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    correct = 0
    for ep in range(epoch):
        model.train()
        correct = 0
        train_loss = 0
        cnt = 0
        for data, target in tqdm(trainLoader):
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            output = model(data)
            # print(output.shape)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            train_loss += criterion(output, target).item()
            cnt+=1

        train_loss /= cnt
        print(f'Train Epoch: {ep+1}\tLoss: {train_loss:.6f}, Accuracy: {correct}/{len(trainLoader.dataset)} ({100. * correct / len(trainLoader.dataset):.0f}%)')

        val(model,ep) 
        scheduler.step()

def val(model,ep):
    criterion = nn.CrossEntropyLoss()
    model.eval() 
    val_loss = 0
    correct = 0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target in valLoader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(valLoader)
    print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(valLoader.dataset),
        100. * correct / len(valLoader.dataset)))

    if(100. * correct / len(valLoader.dataset)>90):
        torch.save(model, ('{}_{:.0f}.pth').format(ep,100. * correct / len(valLoader.dataset)))
    

train(model,10)