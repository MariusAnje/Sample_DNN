import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import PIL
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Sample_AlexNet
from SampleNN import *
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
import torchvision

float_net = torchvision.models.alexnet(pretrained=True).to(device)
gt  = float_net.state_dict()
b = gt.keys()
N = 16
m = 8

sample_net = Sample_AlexNet.alexnet().to(device)
lol = sample_net.state_dict()
a = lol.keys()
for i in range(len(a)):
    lol[a[i]] = gt[b[i]]
    print (lol[a[i]].shape)
sample_net.load_state_dict(lol)
sampleStateDict(sample_net, N, m)
sample_net.eval()

N = 16
m = 8
delt = pow(2,-m)
Q = pow(2, N-1) - 1
ImageSize = 224
img = torchvision.transforms.ToTensor()(plt.imread('timg.jpg'))
img = nn.functional.interpolate(img.view(1,3,650,1200),size=(ImageSize,ImageSize),mode='bilinear')
img = (img/delt).to(torch.int16)
img = img.to(torch.float) * delt

sample_net.eval()
a = open('car.csv', 'w+')
for i in list((mSample(N,m)(sample_net(img))/delt).view(-1).detach().numpy()):
    a.write('%d,'%i)