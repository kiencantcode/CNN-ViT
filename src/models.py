import torch.nn as nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


def build_model(model_cfg, num_classes):
    raise NotImplementedError


def get_param_groups(model, lr_head, lr_backbone, weight_decay):
    raise NotImplementedError
