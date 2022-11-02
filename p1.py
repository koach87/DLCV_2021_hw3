import argparse
from PIL import Image
import torch
import glob
import os
from torch.utils.data import DataLoader,Dataset
import pandas as pd
import torchvision.transforms as transforms

class p1_data_out(Dataset):
    def __init__(self, path, transform=None):
        self.imgpaths = glob.glob(os.path.join(path,'*'))
        self.transform = transform
    def __getitem__(self, index) :
        img = Image.open(self.imgpaths[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, os.path.basename(self.imgpaths[index])
    def __len__(self):
        return len(self.imgpaths)

def output(model,test_dataloader, csv_path):
    labels = []
    fns = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for data,fn in test_dataloader:
        output = model(data.to(device))
        pred = output.max(1, keepdim=False)[1].tolist()
        labels += pred
        fns += [i for i in fn]
    df = pd.DataFrame({
            "filename":fns,
            "label":labels
        })
    df.to_csv(csv_path,index = False)

parser =  argparse.ArgumentParser(description='Use to predict image for HW3_p1')
parser.add_argument( '--img_path', type=str, default='', help='path to testing images in the target domain' )
parser.add_argument( '--csv_path', type=str, default='', help='path to your output prediction file')
args = parser.parse_args()

img_path = args.img_path
csv_path = args.csv_path
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('p1.pth', map_location=device).to(device)
test_dataset = p1_data_out(img_path, transform=transforms.Compose([
    transforms.Resize(model.image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]))
test_dataloader = DataLoader(test_dataset)
print('start predicting')
output(model, test_dataloader,csv_path)
print('done')