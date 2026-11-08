def load_config(path, overrides=None):
    raise NotImplementedError


def set_seed(seed):
    raise NotImplementedError


def count_parameters(model):
    raise NotImplementedError


def get_hardware_info():
    raise NotImplementedError


def make_run_dir(cfg):
    raise NotImplementedError
