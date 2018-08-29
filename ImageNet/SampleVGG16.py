import torch
import torchvision
import torch.nn as nn
from SampleNN import *

class sFeatures(nn.Module):
    def __init__(self):
        super(sFeatures, self).__init__()
        self.L0  = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.L1  = nn.ReLU(inplace = True)
        self.L2  = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.L3  = nn.ReLU(inplace = True)
        self.L4  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.L5  = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.L6  = nn.ReLU(inplace = True)
        self.L7  = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.L8  = nn.ReLU(inplace = True)
        self.L9  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.L10 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.L11 = nn.ReLU(inplace = True)
        self.L12 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.L13 = nn.ReLU(inplace = True)
        self.L14 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.L15 = nn.ReLU(inplace = True)
        self.L16 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.L17 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.L18 = nn.ReLU(inplace = True)
        self.L19 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.L20 = nn.ReLU(inplace = True)
        self.L21 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.L22 = nn.ReLU(inplace = True)
        self.L23 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.L24 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.L25 = nn.ReLU(inplace = True)
        self.L26 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.L27 = nn.ReLU(inplace = True)
        self.L28 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.L29 = nn.ReLU(inplace = True)
        self.L30 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    
    def forward(self, x):
        x = oSample()(x)
        x = oSample()(self.L1(self.L0(x)))
        x = oSample()(self.L4(self.L3(self.L2(x))))
        x = oSample()(self.L6(self.L5(x)))
        x = oSample()(self.L9(self.L8(self.L7(x))))
        x = oSample()(self.L11(self.L10(x)))
        x = oSample()(self.L13(self.L12(x)))
        x = oSample()(self.L16(self.L15(self.L14(x))))
        x = oSample()(self.L18(self.L17(x)))
        x = oSample()(self.L20(self.L19(x)))
        x = oSample()(self.L23(self.L22(self.L21(x))))
        x = oSample()(self.L25(self.L24(x)))
        x = oSample()(self.L27(self.L26(x)))
        x = oSample()(self.L30(self.L29(self.L28(x))))
        return x

class sClassifier(nn.Module):
    def __init__(self):
        super(sClassifier, self).__init__()
        num_classes = 1000
        self.L0 = nn.Linear(512 * 7 * 7, 4096)
        self.L1 = nn.ReLU(True)
        self.L2 = nn.Dropout()
        self.L3 = nn.Linear(4096, 4096)
        self.L4 = nn.ReLU(True)
        self.L5 = nn.Dropout()
        self.L6 = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        x = oSample()(self.L2(self.L1(self.L0(x))))
        x = oSample()(self.L5(self.L4(self.L3(x))))
        x = oSample()(self.L6(x))
        return x

class SampleVGG16(nn.Module):
    def __init__(self):
        super(SampleVGG16,self).__init__()
        self.features = sFeatures()
        self.classifier = sClassifier()
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)