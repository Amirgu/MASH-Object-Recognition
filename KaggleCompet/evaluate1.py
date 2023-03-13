import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
import torch.cuda
from model import Net

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--model', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='inception/kaggle5.csv', metavar='D',
                    help="name of the output csv file")

args = parser.parse_args()
import torchvision
use_cuda = torch.cuda.is_available()
from model import Net
path1='resnet/trnormal/model_14.pth'
path2='resnet/trnormal/model_26.pth'
model1= Net()
model1.load_state_dict(torch.load(path1))

model2=Net()
model2.load_state_dict(torch.load(path2))
model2.eval()
model1.eval()
if use_cuda:
    print('Using GPU')
    model1.cuda()
    model2.cuda()
else:
    print('Using CPU')

from data import data_val_transforms
species = [
    '004.Groove_billed_Ani',
'009.Brewer_Blackbird',
'010.Red_winged_Blackbird',
'011.Rusty_Blackbird',
'012.Yellow_headed_Blackbird',

'013.Bobolink',
'014.Indigo_Bunting',
'015.Lazuli_Bunting',
'016.Painted_Bunting',

'019.Gray_Catbird',
'020.Yellow_breasted_Chat',
'021.Eastern_Towhee',
'023.Brandt_Cormorant',
'026.Bronzed_Cowbird',
'028.Brown_Creeper',
'029.American_Crow',
'030.Fish_Crow',
'031.Black_billed_Cuckoo',
'033.Yellow_billed_Cuckoo',

'034.Gray_crowned_Rosy_Finch']

test_dir = args.data + '/test_images/mistery_category'
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

import numpy as np

def evalu(path,model):   
    i=0
    s=np.zeros(20)
    for x in species:
        
        test_dir1 = path + x
        t=0.
                
        for f in tqdm(os.listdir(test_dir1)):
            t+=1.
            if 'jpg' in f:
                data = data_val_transforms(pil_loader(test_dir1 + '/' + f))
                        
                data = data.view(1, data.size(0), data.size(1), data.size(2))
                    
                
                        
                if use_cuda:
                    data = data.cuda()
                    output = model(data)
                        
                    pred = output.data.max(1, keepdim=True)[1]
                    
                if  pred.tolist()[0][0] == i:
                    
                    s[i]+=1.
                    
        
                
        s[i]=s[i]/(t)
        i+=1

    return s
z=np.zeros(20)
path='bird_dataset/val_images/'

def whoperfombetter(model1,model2):
    output_file = open(args.outfile, "w")
    output_file.write("Id,Category\n")
    
    k=0
    s1=evalu(path,model1)
    s2=evalu(path,model2)
    print(s1)
    print(s2)
    z=np.maximum(s1,s2)
    print(z)
    
    for f in tqdm(os.listdir(test_dir)):
        if 'jpg' in f:
            k+=1
            data = data_val_transforms(pil_loader(test_dir + '/' + f))
            data = data.view(1, data.size(0), data.size(1), data.size(2))
            if use_cuda:
                data = data.cuda()
            output1 = model1(data)
            output2=model2(data)
            pred1 = output1.data.max(1, keepdim=True)[1]
            pred2 = output2.data.max(1, keepdim=True)[1]
            x1=pred1.tolist()[0][0]
            x2=pred2.tolist()[0][0]
            if x1 != x2:

                print([x1,x2],'image :',str(k))
                
    


whoperfombetter(model1,model2)
print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle competition website')


