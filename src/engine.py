def build_scheduler(optimizer, cfg, steps_per_epoch):
    raise NotImplementedError


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, cfg):
    raise NotImplementedError


def evaluate(model, loader, criterion, device):
    raise NotImplementedError


def fit(model, loaders, cfg, run_dir, device):
    raise NotImplementedError
