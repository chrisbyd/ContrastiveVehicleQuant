import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .contrast_loss import SupConLoss
from .orthogonal_loss import OrthogonalLoss

def make_loss(cfg, num_classes):
    sampler = cfg.DATALOADER.SAMPLER
    triplet = TripletLoss(cfg.SOLVER.MARGIN)
    orthogonal = OrthogonalLoss()
    supcon = SupConLoss()

    if cfg.MODEL.LABEL_SMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, num_classes:", num_classes)
    else:
        xent = F.cross_entropy

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return xent(score, target)
    elif sampler == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif sampler == 'softmax_triplet':
        def loss_func(score, feat, descriptor, target):
            return xent(score, target) + triplet(feat, descriptor ,target)[0]
    elif sampler == 'softmax_contrast':
        def loss_func(score, feat, descriptor, target):
            return xent(score, target) + supcon(feat, target, descriptor, target) 
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


