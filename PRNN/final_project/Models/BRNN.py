import torch
import torch.nn as nn
import torch.nn.functional as F

# RNN for character prediction and is block regularizable
class BRNN(nn.Module):
    def __init__(self, n_input, n_hidden,n_output,n_part,k,device):
        super(BRNN, self).__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_part = n_part
        self.k = k # not used yet :(
        self.n_blocks = 2*n_part+n_part**2+ n_part**3 # count the total number of blocks
        self.size_partitions = int(n_hidden/n_part)
        self.ss = nn.Softsign()
        self.sp = nn.Softplus()
        self.encoder = nn.ModuleList([
            nn.Linear(n_input, self.size_partitions) for n in range(n_part)
        ])
        self.recurrent = nn.ModuleList([
            nn.Linear(n_hidden, self.size_partitions) for n in range(n_part)
        ])
        self.modulator = nn.ModuleList([
            nn.Linear(n_hidden,n_hidden,bias=False) for n in range(n_part)
        ])
        self.decoder = nn.Linear(n_hidden, n_output)
        self.hidden_init = nn.Linear(1,n_hidden)
        self.regularize = nn.Linear(self.n_blocks,1,bias=False) 
        self.regularize.weight.data.uniform_(0, 0.01)
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
            for n in range(self.n_part):
                hidden_new[:,n*self.size_partitions:(n+1)*self.size_partitions] = \
                self.ss(self.encoder[n](x[i,:,:]) + \
                        self.recurrent[n](self.sp(torch.mul(self.modulator[n](hidden),hidden))))
                hidden = hidden_new.clone()
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
                # the modulator weights
                if param.shape[1]==self.n_hidden and param.shape[0]==self.n_hidden:
                    for i in range(self.n_part):
                        for j in range(self.n_part):
                            Lnorms.append(param[i*self.size_partitions:(i+1)*self.size_partitions,j*self.size_partitions:(j+1)*self.size_partitions].norm(p=pnorm))
                # the recurrent weights
                elif param.shape[1]==self.n_hidden and param.shape[0]==self.size_partitions:
                    for i in range(self.n_part):
                        Lnorms.append(param[:,i*self.size_partitions:(i+1)*self.size_partitions].norm(p=pnorm))
                # the encoder weights
                elif param.shape[1]==self.n_input and param.shape[0]==self.size_partitions:
                    Lnorms.append(param.norm(p=pnorm))
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
                # the modulator weights
                if param.shape[1]==self.n_hidden and param.shape[0]==self.n_hidden:
                    for i in range(self.n_part):
                        for j in range(self.n_part):
                            hgrad.append(torch.dot(param.grad.data[i*self.size_partitions:(i+1)*self.size_partitions,j*self.size_partitions:(j+1)*self.size_partitions].flatten(),torch.sign(param.data[i*self.size_partitions:(i+1)*self.size_partitions,j*self.size_partitions:(j+1)*self.size_partitions].flatten())))
                # the recurrent weights
                elif param.shape[1]==self.n_hidden and param.shape[0]==self.size_partitions:
                    for i in range(self.n_part):
                        hgrad.append(torch.dot(param.grad.data[:,i*self.size_partitions:(i+1)*self.size_partitions].flatten(),torch.sign(param.data[:,i*self.size_partitions:(i+1)*self.size_partitions].flatten())))
                # the encoder weights
                elif param.shape[1]==self.n_input and param.shape[0]==self.size_partitions:
                        hgrad.append(torch.dot(param.grad.data.flatten(),torch.sign(param.data.flatten())))
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