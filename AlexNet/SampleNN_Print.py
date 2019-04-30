# This file goes with chips so we use cut tail and head

import torch

def int2str(inputs):
    a = ''
    val = inputs
    if inputs < 0:
        flag = 1
        inputs = abs(inputs)
    else:
        flag = 0
    for i in range(15):
        a += str(inputs%2)
        inputs = inputs/2
    if flag:
        a += '1'
    else:
        a += '0'
    return a[::-1]

def write2file(M):
    happy = open('happy.txt','a')
    happy.write("SSSSSSS\n")
    if len(M.shape) == 4:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                for k in range(M.shape[2]):
                    for l in range(M.shape[3]):
                        happy.write(int2str(int(M[i][j][k][l]))+' ')
                    happy.write('\n')
                happy.write('\n')
    if len(M.shape) == 2:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                happy.write(int2str(int(M[i][j]))+' ')
            happy.write('\n')
    if len(M.shape) == 1:
        for i in range(M.shape[0]):
            happy.write(int2str(int(M[i]))+' ')
        happy.write('\n')
    happy.close()
    

class xSample(torch.autograd.Function):
    """
    This method is deprecated
    """
    def __init__(ctx):
        N = 16
        ctx.Q = pow(2, N-1) - 1
        ctx.delt = pow(2,-8)
    def forward(ctx, inputs):
        Q = ctx.Q
        delt = ctx.delt
        M = (inputs/delt).round().to(torch.int16)
        S = delt*M.to(torch.float32)
        return S
    def backward(ctx, g):
        return g

class oSample(torch.autograd.Function):
    """
    This method is deprecated
    """
    def __init__(ctx):
        N = 8
        ctx.Q = pow(2, N-1) - 1
        ctx.delt = pow(2,-5)
    def forward(ctx, inputs):
        Q = ctx.Q
        delt = ctx.delt
        M = (inputs/delt).round().to(torch.int8)
        S = delt*M.to(torch.float32)
        return S
    def backward(ctx, g):
        return g

class mSample_F(torch.autograd.Function):
    """
    res = clamp(round(input/pow(2,-m)) * pow(2, -m), -pow(2, N-1), pow(2, N-1) - 1)
    """
    
    def __init__(ctx, N = 16, m = 6):
        ctx.delt = pow(2,-m)
        ctx.Q = pow(2, N-1) - 1
        
    def forward(ctx, inputs):
        Q = ctx.Q
        delt = ctx.delt
        M = (inputs.to(torch.float32)/delt).to(torch.int16).to(torch.float32)
        write2file(M)
        return delt*M
    def backward(ctx, g):
        return g

class mSample(torch.nn.Module):
    """
    A module wrapper of mSample Function.
    """
    def __init__(self, N = 16, m = 6):
        super(mSample, self).__init__()
        self.N = N
        self.m = m

    def forward(self, input):
        return mSample_F(N = self.N, m = self.m)(input)
    
    def extra_repr(self):
        s = ('N = %d, m = %d'%(self.N, self.m))
        return s
    
def sampleStateDict(net,N = 16, m = 6):
    """
    Quantize a state dict of one pytorch.model
    the operation is inplace
    input net:torch.model, N: data width, m: accuracy
    """
    Dict = net.state_dict()
    Key = Dict.keys()
    for i in Key:
        happy = open('happy.txt','a')
        happy.write(i+'\n')
        happy.close()
        Dict[i] = mSample(N,m)(Dict[i])
    net.load_state_dict(Dict)
    del Dict

def protectStateDict(net):
    """
    make a entirely new memory space for a state dict to protect it
    Since all pytorch items are using chain table,
    so some tricky methods are used
    Todo: better method should be used
    """
    Dict = net.state_dict()
    Key = Dict.keys()
    for i in Key:
        Dict[i] = Dict[i]*1
    return Dict
