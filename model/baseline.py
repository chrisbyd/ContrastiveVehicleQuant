import torch
from torch import nn

from .backbones.resnet import ResNet
from .quantization_head import GPQSoftMaxNet
from utils.functions import my_soft_assignment

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, cfg):
        super(Baseline, self).__init__()
        last_stride, model_path = cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH
        self.cfg = cfg
        self.base = ResNet(last_stride)
        self.base.load_param(model_path)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.raw_feature = nn.Linear(2048, self.cfg.len_code * self.cfg.n_book)
        self.q_head = GPQSoftMaxNet(True, n_book = cfg.n_book, len_code= cfg.intn_word, intn_word= cfg.intn_word)

        self.bottleneck = nn.BatchNorm1d(self.cfg.len_code * self.cfg.n_book)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.cfg.len_code * self.cfg.n_book, self.num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def load_param(self, model_path):
        param = torch.load(model_path)
        for i in param:
            if 'fc' in i: continue
            if i not in self.state_dict().keys(): continue
            if param[i].shape != self.state_dict()[i].shape: continue
            self.state_dict()[i].copy_(param[i])

    def forward(self, x):
        global_feat = self.gap(self.base(x))  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
        
      # feat = self.bottleneck(global_feat)  # normalize for angular softmax
        raw_feat  = self.bottleneck(self.raw_feature(global_feat))
        global_descriptor = my_soft_assignment(self.q_head.Z, feat, len_code= self.cfg.len_code,
                            self.cfg.alpha, device= 'cuda')
        if self.training:
            cls_score = self.classifier(raw_feat)
            return cls_score, raw_feat, global_descriptor  # global feature for triplet loss
        else:
            return raw_feat


