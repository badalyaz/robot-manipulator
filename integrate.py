### Integration of the net output w.r.t input time ###

import torch
from torchdiffeq import odeint

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device



def integrate(net, n, Init_cond, index):
    
    t = ( torch.arange(1, (n + 1))/n).to(device).double()
    
    q0 = Init_cond[index, 0]

    q = q0 + odeint(net, 0*q0, t)
    
    return q