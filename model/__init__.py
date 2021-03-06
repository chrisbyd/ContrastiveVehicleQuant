from .baseline import Baseline
from .sync_bn import convert_model

def build_model(cfg, num_classes):
    if cfg.MODEL.NAME == 'resnet50':
        model = Baseline(num_classes, cfg)
    return model

