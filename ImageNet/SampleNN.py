import torch

class oSample(torch.autograd.Function):
    def forward(ctx, inputs):
        N = 16
        Q = pow(2, N-1) - 1
        delt = pow(2,-8)
        M = (inputs/delt).round()
        M[M>=Q] = Q
        M[M<-Q] = -(Q+1)
        S = delt*M
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