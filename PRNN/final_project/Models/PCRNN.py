import torch
import torch.nn as nn
import torch.nn.functional as F

# character RNN based on GRU but using the PRNN formulation
class PCRNN(nn.Module):
    def __init__(self, n_inputs, n_hidden,n_output,device):
        super(PCRNN, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.dt = nn.Parameter(torch.Tensor([0])).to(device)
        self.a = nn.Parameter(torch.Tensor([1])).to(device)
        self.sig = nn.Sigmoid()
        self.encoder = nn.Linear(n_inputs, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_output)
        self.recurrent = nn.Linear(n_hidden,n_hidden)
        self.hidden_init = nn.Linear(1,n_hidden)
        self.device = device
        
    def forward(self, x):
        x=x.permute(1,0,2)
        hidden_dummy = torch.ones(x.shape[1],1).to(self.device)
        output = torch.ones(x.shape[0],x.shape[1],self.n_output).to(self.device)
        hidden = self.hidden_init(hidden_dummy)
        for i in range(x.shape[0]):
            hidden = (1-self.sig(self.dt))*hidden+self.sig(self.dt)*torch.tanh(self.encoder(x[i,:,:])+self.a*self.recurrent(hidden))
            output[i,:,:] = self.decoder(hidden)
        
        return output, hidden