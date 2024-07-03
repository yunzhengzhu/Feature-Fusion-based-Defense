import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# ### mindssc feature #####
def mindssc(img, delta=1, sigma=0.8, normalize=False):
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor
    device = img.device
    dtype = img.dtype
    
    # define start and end locations for self-similarity pattern
    four_neighbourhood = torch.Tensor([[0, 1], 
                                      [1, 0],
                                      [2, 1],
                                      [1, 2]]).long() # 4, 2 (left, top, right, bottom)
                                                         #      (-x, -y, +x, +y)
    
    # squared distances
    dist = pdist(four_neighbourhood.unsqueeze(0)).squeeze(0) # 4, 4
    
    # define comparison mask
    x, y = torch.meshgrid(torch.arange(4), torch.arange(4))
    mask = ((x > y).view(-1) & (dist == 2).view(-1)) # 16
    
    # build kernel
    idx_shift1 = four_neighbourhood.unsqueeze(1).repeat(1,4,1).view(-1,2)[mask, :] # 4, 2
    idx_shift2 = four_neighbourhood.unsqueeze(0).repeat(4,1,1).view(-1,2)[mask, :] # 4, 2
    mshift1 = torch.zeros(4, 1, 3, 3).to(dtype).to(device)
    mshift1.view(-1)[torch.arange(4) * 9 + idx_shift1[:, 0] * 3 + idx_shift1[:, 1]] = 1
    mshift2 = torch.zeros(4, 1, 3, 3).to(dtype).to(device)
    mshift2.view(-1)[torch.arange(4) * 9 + idx_shift2[:, 0] * 3 + idx_shift2[:, 1]] = 1
    rpad = nn.ReplicationPad2d(delta)
    
    # compute patch-ssd
    ssd = smooth(((F.conv2d(rpad(img), mshift1, dilation=delta) - F.conv2d(rpad(img), mshift2, dilation=delta)) ** 2), sigma)
    
    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    mind_var = torch.mean(mind, 1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)
    mind /= mind_var
    mind = torch.exp(-mind).to(dtype)
   
    if normalize:
        # minmax normalization
        min_vals = mind.amin(dim=(1, 2), keepdim=True)
        max_vals = mind.amax(dim=(1, 2), keepdim=True)
        
        mind = (mind - min_vals) / (max_vals - min_vals)

    #permute to have same ordering as C++ code
    mind = mind[:, torch.Tensor([2, 0, 1, 3]).long(), :, :]
    return mind

def pdist(x, p=2):
    if p==1:
        dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).sum(dim=2)
    elif p==2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
        dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0
    return dist

def smooth(img, sigma):
    device = img.device
    
    sigma = torch.tensor([sigma]).to(device)
    N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1
    
    weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N).to(device), 2) / (2 * torch.pow(sigma, 2)))
    weight /= weight.sum()
    
    img = filter1D(img, weight, 0)
    img = filter1D(img, weight, 1)
    return img

def filter1D(img, weight, dim, padding_mode='replicate'):
    B, C, H, W = img.shape
    N = weight.shape[0]
    
    padding = torch.zeros(4,)
    padding[[2 - 2 * dim, 3 - 2 * dim]] = N//2
    padding = padding.long().tolist()
    
    view = torch.ones(4,)
    view[dim + 2] = -1
    view = view.long().tolist()
    return F.conv2d(F.pad(img.view(B*C, 1, H, W), padding, mode=padding_mode), weight.view(view)).view(B, C, H, W)

