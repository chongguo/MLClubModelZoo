import torch
import torch.nn as nn
import torch.nn.functional as F

# RNN for character prediction and is block regularizable
class BRNN(nn.Module):
    def __init__(self, n_inputs, n_hidden,n_output,n_partitions,device):
        super(BRNN, self).__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_partitions = n_partitions
        self.size_partitions = int(n_hidden/n_partitions)
        self.ss = nn.Softsign()
        self.sp = nn.Softplus()
        self.encoder = nn.ModuleList([
            nn.Linear(n_inputs, self.size_partitions) for n in range(n_partitions)
        ])
        self.decoder = nn.Linear(n_hidden, n_output)
        self.recurrent = nn.ModuleList([
            nn.Linear(n_hidden, self.size_partitions) for n in range(n_partitions)
        ])
        self.modulator = nn.ModuleList([
            nn.Linear(n_hidden,n_hidden,bias=False) for n in range(n_partitions)
        ])
        self.hidden_init = nn.Linear(1,n_hidden)
        self.device = device
        
    def forward(self, x):
        x=x.permute(1,0,2)
        hidden_dummy = torch.ones(x.shape[1],1).to(self.device)
        hidden = self.hidden_init(hidden_dummy)
        hidden_new = torch.ones(x.shape[1],self.n_hidden).to(self.device)
        output = torch.ones(x.shape[0],x.shape[1],self.n_output).to(self.device)
        for i in range(x.shape[0]):
            for n in range(self.n_partitions):
                hidden_new[:,n*self.size_partitions:(n+1)*self.size_partitions] = \
                self.ss(self.encoder[n](x[i,:,:]) + \
                        self.recurrent[n](self.sp(torch.mul(self.modulator[n](hidden),hidden))))
                hidden = hidden_new.clone()
            output[i,:,:] = self.decoder(hidden)
        return output, hidden