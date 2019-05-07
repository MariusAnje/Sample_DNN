# This document is forked from https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from SampleNN_Print import mSample


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        N = 16
        m = 8
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=0),
            mSample(N=N,m=m, name = 'conv1'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            mSample(N=N,m=m, name = 'conv1_Pool'),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            mSample(N=N,m=m, name = 'conv2'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            mSample(N=N,m=m, name = 'conv2_Pool'),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            mSample(N=N,m=m, name = 'conv3'),
            nn.ReLU(inplace=True),
            mSample(N=N,m=m, name = 'conv3_ReLU'),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            mSample(N=N,m=m, name = 'conv4'),
            nn.ReLU(inplace=True),
            mSample(N=N,m=m, name = 'conv4_ReLU'),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            mSample(N=N,m=m, name = 'conv5'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            mSample(N=N,m=m, name='FC1_In'),
            nn.Linear(256 * 6 * 6, 4096),
            mSample(N=N,m=m, name='FC1_Out'),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            mSample(N=N,m=m, name='FC2_In'),
            nn.Linear(4096, 4096),
            mSample(N=N,m=m, name='FC2_Out'),
            nn.ReLU(inplace=True),
            mSample(N=N,m=m, name='FC3_In'),
            nn.Linear(4096, num_classes),
            mSample(N=N,m=m, name='FC4_In'),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model
