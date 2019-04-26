import torch

def get_accuracy(logit, target):
    batch_size = target.shape[0]
    accuracy = (torch.max(logit, 1)[1].data == target.data).type(torch.DoubleTensor).mean()*100
    return accuracy.item()

def nparam(ninputs,nhidden,noutputs):
    return ninputs*(nhidden+1) + nhidden*(nhidden+1)+nhidden*(noutputs+1)

#def sample(net, f_encode, f_invert, size, prime='The Duchess'):
    #net.eval()
    #prime = HotcodeAnna(IntcodeAnna(prime))
    
    # First off, run through the prime character
    #for i in range(len(prime)-9):
    #    intchar = net(prime[i:i+10,:])

    #chars.(char)
    
    # Now pass in the previous character and get a new one
    #for ii in range(size):
        #char, h = net.forward(, )
    #    intchar = torch.max(char[-1,:])[1]
    #    intchars.append(intchar)
    
    #return ''.join(f_invert(intchar))