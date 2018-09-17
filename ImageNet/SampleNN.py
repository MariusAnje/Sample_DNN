# a very radical version in sampling

import torch

class xSample(torch.autograd.Function):
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

class mSample(torch.autograd.Function):
    """
    res = round(input/pow(2,-m)) * pow(2, -m)
    the rounded part is bounded to int8 or int16,
    larger number will overflow
    """
    
    def __init__(ctx, N = 16, m = 6):
        ctx.delt = pow(2,-m)
        if N == 16:
            ctx.DType = torch.int16
        elif N == 8:
            ctx.DType = torch.int8
        else:
            raise Exception("We only support int16 and int8")
    def forward(ctx, inputs):
        delt = ctx.delt
        M = (inputs.to(torch.float32)/delt)
        M = M.round().to(ctx.DType)
        S = M.to(torch.float32)*delt
        return S
    def backward(ctx, g):
        return g
    
def sampleStateDict(net,N = 16, m = 6):
    """
    Quantize a state dict of one pytorch.model
    the operation is inplace
    input net:torch.model, N: data width, m: accuracy
    """
    Dict = net.state_dict()
    Key = Dict.keys()
    for i in Key:
        Dict[i] = mSample(N,m)(Dict[i])
    net.load_state_dict(Dict)

def protectStateDict(net):
    """
    make a entirely new memory space for a state dict to protect it
    Since all pytorch items are using chain table,
    so some tricky methods are used
    """
    Dict = net.state_dict()
    Key = Dict.keys()
    for i in Key:
        Dict[i] = Dict[i]*1
    return Dict