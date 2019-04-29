import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import models
from SampleNN import *

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
import torchvision
from torchvision import transforms
from tqdm import tqdm

N_F = 16
m_F = 8
N_w = 8
m_w = 7
pretrained = 'good.pt'
#pretrained = None


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root = '/home/yanzy/SEU_NNFailure/CIFAR_10/data', train=True,
                                        download=False, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2, pin_memory=True)

testset = torchvision.datasets.CIFAR10(root='/home/yanzy/SEU_NNFailure/CIFAR_10/data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                         shuffle=False, num_workers=2, pin_memory=True)

classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

sample_net = models.nin(N=N_F, m=m_F).to(device)

if pretrained:
    State_File = torch.load(pretrained)
    lol = sample_net.state_dict()
    a = lol.keys()
    #print (len(a))
    gt = State_File
    b = gt.keys()
    for i in range(len(a)):
        lol[a[i]] = gt[b[i]]
    sample_net.load_state_dict(lol)
else:
    for m in sample_net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.05)
            m.bias.data.normal_(0, 0.0)

def val(N_F, m_F, N_w, m_w):
    torch.cuda.empty_cache()
    TMP = protectStateDict(sample_net)
    sampleStateDict(sample_net,N=N_w,m=m_w)
    sample_net.eval()
    correct = 0
    total = 0
    count = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            images = mSample(N=N_F,m=m_F)(images)
            outputs = sample_net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            count += labels.size(0)
            correct += (predicted == labels).sum().item()
            #print correct, total
    sample_net.load_state_dict(TMP)
    torch.cuda.empty_cache()

   #print('Accuracy of the network on the %d test images: %.3f %%' % (total,
    #    100.0 * correct / total))
    return 100.0 * correct / total

criterion_s = nn.CrossEntropyLoss()
base_lr = float(1e-3)
base_lr = 0.1
param_dict = dict(sample_net.named_parameters())
params = []
#print(len(param_dict.keys()))

for key, value in param_dict.items():
    if key == 'classifier.23.weight':
        params += [{'params':[value], 'lr':0.1 * base_lr, 
            'momentum':0.95, 'weight_decay':0.0001}]
    elif key == 'classifier.23.bias':
        params += [{'params':[value], 'lr':0.1 * base_lr, 
            'momentum':0.95, 'weight_decay':0.0000}]
    elif 'weight' in key:
        params += [{'params':[value], 'lr':1.0 * base_lr,
            'momentum':0.95, 'weight_decay':0.0001}]
    else:
        params += [{'params':[value], 'lr':2.0 * base_lr,
            'momentum':0.95, 'weight_decay':0.0000}]

optimizer_s = optim.SGD(params, lr=0.1, momentum=0.9)

def adjust_learning_rate(optimizer, epoch):
    update_list = [120, 200, 240, 280]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return

def train(epoch):
    sample_net.train()
    running_loss = 0.0
    count = 0
    for data in tqdm(train_loader, leave = False):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = mSample(N=N_F,m=m_F)(inputs)

        # zero the parameter gradients
        optimizer_s.zero_grad()

        # forward + backward + optimize
        TMP = protectStateDict(sample_net)
        sampleStateDict(sample_net)
        outputs = sample_net(inputs)
        loss = criterion_s(outputs, labels)
        loss.backward()
        sample_net.load_state_dict(TMP)
        del TMP
        torch.cuda.empty_cache()
        optimizer_s.step()
        count += labels.size(0)

        # print statistics
        running_loss += loss.item()
    return running_loss

acc_m = 0
l_m = 100000
Loader = tqdm(range(320)) 
for epoch in Loader:
    adjust_learning_rate(optimizer_s, epoch)
    torch.cuda.empty_cache()
    loss = train(epoch)
    torch.cuda.empty_cache()
    TMP = protectStateDict(sample_net)
    sampleStateDict(sample_net)
    acc = val(N_F, m_F, N_w, m_w)
    sample_net.load_state_dict(TMP)
    del TMP
    if acc > acc_m:
        acc_m = acc
    if loss < l_m:
        l_m = loss
    Loader.set_description('l:%.2f, m:%.2f, a:%.2f, m:%.2f'%(loss, l_m, acc, acc_m))