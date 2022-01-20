import torch
from torch import nn
import torch.nn.functional as F


class OrthogonalLoss(nn.Module):
    def __init__(self):
        super(OrthogonalLoss, self).__init__()
    
    def forward(self, features, descriptor ,labels):
        features = F.normalize(features, dim =1)
        labels_equal = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
        labels_not_equal = ~labels_equal

        neg_dis = torch.matmul(features,descriptor.T)*labels_not_equal
      
        dim = features.size(1)
        gor = torch.pow(torch.mean(neg_dis),2) + torch.clamp(torch.mean(torch.pow(neg_dis,2))-1.0/dim, min=0.0)
       
        return gor




