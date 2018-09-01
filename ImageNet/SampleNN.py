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
        ctx.delt = pow(2,-7)
    def forward(ctx, inputs):
        Q = ctx.Q
        delt = ctx.delt
        M = (inputs/delt).round().to(torch.int16)
        S = delt*M.to(torch.float32)
        return S
    def backward(ctx, g):
        return g
    
def sampleStateDict(net):
    Dict = net.state_dict()
    Key = Dict.keys()
    for i in Key:
        Dict[i] = oSample()(Dict[i])
    net.load_state_dict(Dict)

def protectStateDict(net):
    Dict = net.state_dict()
    Key = Dict.keys()
    for i in Key:
        Dict[i] = Dict[i]*1
    return Dict