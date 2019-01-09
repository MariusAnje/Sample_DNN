import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt

# 这3个有用的
InChannel = 0
OutChannel = 0
# filter的文件路径，按照自己的路径调整一下吧
filterFilePath = './Weights/features.0.weight.pt'

# 这些不用改
N = 16
m = 8
delt = pow(2,-m)
Q = pow(2, N-1) - 1
intWeights = torch.load(filterFilePath)
floatWeights = intWeights*delt
conv = nn.Conv2d(1, 1, kernel_size=11, stride=4, padding=2)
state_dict = conv.state_dict()
state_dict['weight'] = floatWeights[InChannel][OutChannel].view(state_dict['weight'].size())
state_dict['bias'] = torch.zeros(state_dict['bias'].size())
conv.load_state_dict(state_dict)
img = torchvision.transforms.ToTensor()(plt.imread('timg.jpg'))
img = nn.functional.interpolate(img.view(1,3,650,1200),size=(227,227),mode='bilinear')
img = (img/delt).to(torch.int16)
img = img.to(torch.float) * delt
output = ((conv(img[:,InChannel,:,:].view(1,1,227,227))/delt).to(torch.int16).to(torch.float) * delt).view(55,55)
F = open(str(InChannel) + '.' + str(OutChannel) + '.output.csv','w+')
strOutput = ''
for i in range(len(output)):
    for j in range(len(output)):
        strOutput += '%.8f, '%(output[i][j])
    strOutput += '\n'
F.write(strOutput)
F.close()
