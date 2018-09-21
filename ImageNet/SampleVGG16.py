import torch
import torchvision
import torch.nn as nn
from SampleNN import *

class sFeatures(nn.Module):
    """Total bits N, bits of decimal fraction m"""
    def __init__(self,N = 16, m = 6, bn = True):
        super(sFeatures, self).__init__()
        self.N = N
        self.m = m
        # Boolean, if Ture, with BatchNorm
        self.bn  = bn
        # Written separatly, making it easier to read, though longer 
        if bn:
            self.L0  = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.B1  = nn.BatchNorm2d(64)
            self.L1  = nn.ReLU(inplace = True)
            self.L2  = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.B2  = nn.BatchNorm2d(64)
            self.L3  = nn.ReLU(inplace = True)
            self.L4  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            self.L5  = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.B3  = nn.BatchNorm2d(128)
            self.L6  = nn.ReLU(inplace = True)
            self.L7  = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.B4  = nn.BatchNorm2d(128)
            self.L8  = nn.ReLU(inplace = True)
            self.L9  = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            self.L10 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.B5  = nn.BatchNorm2d(256)
            self.L11 = nn.ReLU(inplace = True)
            self.L12 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.B6  = nn.BatchNorm2d(256)
            self.L13 = nn.ReLU(inplace = True)
            self.L14 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.B7  = nn.BatchNorm2d(256)
            self.L15 = nn.ReLU(inplace = True)
            self.L16 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            self.L17 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.B8  = nn.BatchNorm2d(512)
            self.L18 = nn.ReLU(inplace = True)
            self.L19 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.B9  = nn.BatchNorm2d(512)
            self.L20 = nn.ReLU(inplace = True)
            self.L21 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.B10  = nn.BatchNorm2d(512)
            self.L22 = nn.ReLU(inplace = True)
            self.L23 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            self.L24 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.B11 = nn.BatchNorm2d(512)
            self.L25 = nn.ReLU(inplace = True)
            self.L26 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.B12 = nn.BatchNorm2d(512)
            self.L27 = nn.ReLU(inplace = True)
            self.L28 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.B13 = nn.BatchNorm2d(512)
            self.L29 = nn.ReLU(inplace = True)
            self.L30 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            
        else:
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
        N, m = self.N, self.m
        if self.bn:
            x = mSample(N, m)(x)
            x = mSample(N, m)(self.L1(self.B1(self.L0(x))))
            x = mSample(N, m)(self.L4(self.L3(self.B2(self.L2(x)))))
            x = mSample(N, m)(self.L6(self.B3(self.L5(x))))
            x = mSample(N, m)(self.L9(self.L8(self.B4(self.L7(x)))))
            x = mSample(N, m)(self.L11(self.B5(self.L10(x))))
            x = mSample(N, m)(self.L13(self.B6(self.L12(x))))
            x = mSample(N, m)(self.L16(self.L15(self.B7(self.L14(x)))))
            x = mSample(N, m)(self.L18(self.B8(self.L17(x))))
            x = mSample(N, m)(self.L20(self.B9(self.L19(x))))
            x = mSample(N, m)(self.L23(self.L22(self.B10(self.L21(x)))))
            x = mSample(N, m)(self.L25(self.B11(self.L24(x))))
            x = mSample(N, m)(self.L27(self.B12(self.L26(x))))
            x = mSample(N, m)(self.L30(self.L29(self.B13(self.L28(x)))))

        else:
            x = mSample(N, m)(x)
            x = mSample(N, m)(self.L1(self.L0(x)))
            x = mSample(N, m)(self.L4(self.L3(self.L2(x))))
            x = mSample(N, m)(self.L6(self.L5(x)))
            x = mSample(N, m)(self.L9(self.L8(self.L7(x))))
            x = mSample(N, m)(self.L11(self.L10(x)))
            x = mSample(N, m)(self.L13(self.L12(x)))
            x = mSample(N, m)(self.L16(self.L15(self.L14(x))))
            x = mSample(N, m)(self.L18(self.L17(x)))
            x = mSample(N, m)(self.L20(self.L19(x)))
            x = mSample(N, m)(self.L23(self.L22(self.L21(x))))
            x = mSample(N, m)(self.L25(self.L24(x)))
            x = mSample(N, m)(self.L27(self.L26(x)))
            x = mSample(N, m)(self.L30(self.L29(self.L28(x))))
            
        return x

class sClassifier(nn.Module):
    def __init__(self, N = 16, m = 6):
        super(sClassifier, self).__init__()
        self.N = N
        self.m = m
        num_classes = 1000
        self.L0 = nn.Linear(512 * 7 * 7, 4096)
        self.L1 = nn.ReLU(True)
        self.L2 = nn.Dropout()
        self.L3 = nn.Linear(4096, 4096)
        self.L4 = nn.ReLU(True)
        self.L5 = nn.Dropout()
        self.L6 = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        x = mSample(self.N, self.m)(self.L2(self.L1(self.L0(x))))
        x = mSample(self.N, self.m)(self.L5(self.L4(self.L3(x))))
        x = mSample(self.N, self.m)(self.L6(x))
        return x

class SampleVGG16(nn.Module):
    def __init__(self, N = 16, m = 6, bn = True):
        super(SampleVGG16,self).__init__()
        self.features = sFeatures(N, m, bn)
        self.classifier = sClassifier(N, m)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)