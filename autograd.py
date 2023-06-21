### Derivative of the net output w.r.t input time ###

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device


def autograd(net, Input):
    net.backward(torch.ones(net.shape).to(device), create_graph=True, retain_graph=True)
    d_net = Input.grad
    Input.grad = None
    
    return d_net
