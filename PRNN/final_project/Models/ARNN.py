import torch
import torch.nn as nn
import torch.nn.functional as F

# RNN for character prediction and is block regularizable, no modulator
class ARNN(nn.Module):
    def __init__(self, n_input, n_hidden,n_output,n_part,k,device):
        super(ARNN, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_part = n_part
        self.k = k # not used yet :(
        self.n_blocks = 2*n_part+2*n_part**2 # count the total number of blocks
        self.size_partitions = int(n_hidden/n_part)
        self.ss = nn.Softsign()
        self.sg = nn.Sigmoid()
        self.encoder = nn.Linear(n_input, n_hidden)
        self.recurrent = nn.Linear(n_hidden, n_hidden,bias=False)
        self.forget = nn.Linear(n_hidden, n_hidden,bias=False)
        self.decoder = nn.Linear(n_hidden, n_output)
        self.hidden_init = nn.Linear(1,n_hidden)
        self.regularize = nn.Linear(self.n_blocks,1,bias=False) 
        self.regularize.weight.data.uniform_(0.009, 0.011)
        self.reprojection()
        self.device = device
        self.to(device)
        
    def forward(self, x):
        x=x.permute(1,0,2)
        hidden_dummy = torch.ones(x.shape[1],1).to(self.device)
        hidden = self.hidden_init(hidden_dummy)
        hidden_new = torch.ones(x.shape[1],self.n_hidden).to(self.device)
        output = torch.ones(x.shape[0],x.shape[1],self.n_output).to(self.device)
        for i in range(x.shape[0]):
            forget = self.sg(self.forget(hidden))
            hidden_new = self.ss(self.encoder(x[i,:,:]) + self.recurrent(hidden))
            hidden = torch.mul(1-forget,hidden) + torch.mul(forget,hidden_new)
            output[i,:,:] = self.decoder(hidden)
        return output, hidden
    
    def block_norm(self,norm='L1'):
        Lnorms = []
        
        if norm == 'L1':
            pnorm = 1
        elif norm == 'L2':
            pnorm = 2
            
        for p, param in enumerate(self.parameters()):
            if param.requires_grad and len(param.shape)==2:
                # recurrent and forget weights
                if param.shape[1]==self.n_hidden and param.shape[0]==self.n_hidden:
                    for i in range(self.n_part):
                        for j in range(self.n_part):
                            Lnorms.append(param[i*self.size_partitions:(i+1)*self.size_partitions,j*self.size_partitions:(j+1)*self.size_partitions].norm(p=pnorm))
                # the encoder weights
                elif param.shape[1]==self.n_input and param.shape[0]==self.n_hidden:
                    for i in range(self.n_part):
                        Lnorms.append(param[i*self.size_partitions:(i+1)*self.size_partitions,:].norm(p=pnorm))
                # the decoder weights
                elif param.shape[1]==self.n_hidden and param.shape[0]==self.n_output:
                    for i in range(self.n_part):
                        Lnorms.append(param[:,i*self.size_partitions:(i+1)*self.size_partitions].norm(p=pnorm))
        # turn into a single tensor of norms
        Lnorms = torch.stack(Lnorms)
        regval = self.regularize(Lnorms)
        return regval
    
    def hypergrad(self):
        hgrad = []
        # hypergrad first caculate the appropriate gradient of the regularizers
        for p, param in enumerate(self.parameters()):
            if param.requires_grad and len(param.shape)==2:
                # the recurrena and forget weights
                if param.shape[1]==self.n_hidden and param.shape[0]==self.n_hidden:
                    for i in range(self.n_part):
                        for j in range(self.n_part):
                            hgrad.append(torch.dot(param.grad.data[i*self.size_partitions:(i+1)*self.size_partitions,j*self.size_partitions:(j+1)*self.size_partitions].flatten(),torch.sign(param.data[i*self.size_partitions:(i+1)*self.size_partitions,j*self.size_partitions:(j+1)*self.size_partitions].flatten())))
                # the encoder weights
                elif param.shape[1]==self.n_input and param.shape[0]==self.n_hidden:
                    for i in range(self.n_part):
                        hgrad.append(torch.dot(param.grad.data[i*self.size_partitions:(i+1)*self.size_partitions,:].flatten(),torch.sign(param.data[i*self.size_partitions:(i+1)*self.size_partitions,:].flatten())))
                # the decoder weights
                elif param.shape[1]==self.n_hidden and param.shape[0]==self.n_output:
                    for i in range(self.n_part):
                        hgrad.append(torch.dot(param.grad.data[:,i*self.size_partitions:(i+1)*self.size_partitions].flatten(),torch.sign(param.data[:,i*self.size_partitions:(i+1)*self.size_partitions].flatten())))
        
        self.regularize.weight.grad.data = torch.stack(hgrad).view(1,self.n_blocks)
        # zero all gradients except the hyperparameters
        for name, param in self.named_parameters():
            if not name=='regularize.weight':
                param.grad.data.zero_()
    
    def zero_regularize_grad(self):
        return self.regularize(torch.ones(self.n_blocks).to(self.device))
    
    def reprojection(self):
        # weight are iteratively rectified
        # !!need to be called after each hyperparameter optimizer step!!
        self.regularize.weight.data = self.regularize.weight.clamp(min=0)