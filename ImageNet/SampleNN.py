import torch

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
        M = (inputs.to(torch.float32)/delt).round().clamp(-Q-1,Q)
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