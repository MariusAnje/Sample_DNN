import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt

# These three should be changed
InChannel = 0

# Start and end of output channels
OutChannelStart = 0
OutChannelEnd = 16

# file path filters
filterFilePath = './Weights/features.0.weight.pt'

# dont't touch
N = 16
m = 8
delt = pow(2,-m)
Q = pow(2, N-1) - 1
intWeights = torch.load(filterFilePath)
floatWeights = intWeights*delt
conv = nn.Conv2d(1, OutChannelEnd - OutChannelStart, kernel_size=11, stride=4, padding=0)
state_dict = conv.state_dict()
a = floatWeights[OutChannelStart:OutChannelEnd,InChannel,:,:]
state_dict['weight'] = floatWeights[OutChannelStart:OutChannelEnd,InChannel,:,:].view(state_dict['weight'].size())
b = state_dict['weight'][OutChannelStart:OutChannelEnd,InChannel,:,:]
state_dict['bias'] = torch.zeros(state_dict['bias'].size())
conv.load_state_dict(state_dict)

ImageSize = 224
LargeSize = 227
img = torchvision.transforms.ToTensor()(plt.imread('timg.jpg'))
img = nn.functional.interpolate(img.view(1,3,650,1200),size=(ImageSize,ImageSize),mode='bilinear')
img = (img/delt).to(torch.int16)
img = img.to(torch.float) * delt
ttt = torch.zeros(1,3,LargeSize,LargeSize)
ttt[:,:,2:226,2:226] = img
img = ttt
output = ((conv(img[:,InChannel,:,:].view(1,1,LargeSize,LargeSize))/delt).to(torch.int16).to(torch.float) * delt).view(-1,55,55)
F = open(str(InChannel) + '.' + str(OutChannelStart) + '.' + str(OutChannelEnd) + '.outputFloat.csv','w+')
strOutputFloat = ''
for i in range(len(output[0])):
    for j in range(len(output[0])):
        for k in range(OutChannelEnd - OutChannelStart):
            strOutputFloat += '%.8f, '%(output[k][i][j])
    strOutputFloat += '\n'
F.write(strOutputFloat)
F.close()
F = open(str(InChannel) + '.' + str(OutChannelStart) + '.' + str(OutChannelEnd) + '.outputInt.csv','w+')
strOutputInt = ''
for i in range(len(output[0])):
    for j in range(len(output[0])):
        for k in range(OutChannelEnd - OutChannelStart):
            strOutputInt += '%6d, '%(output[k][i][j]/delt)
    strOutputInt += '\n'
F.write(strOutputInt)
F.close()
intImg = img/delt
print intImg.size()
F = open('intImg_padded.csv','w+')
intImgFile = ''
for i in range(227):
    for j in range(227):
        for k in range(3):
            intImgFile += '%6d, '%(intImg[0][k][i][j])
    intImgFile += '\n'
F.write(intImgFile)
F.close()