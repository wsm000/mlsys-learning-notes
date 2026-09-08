import torch


def reference(x, y, output, n_elements):
    return (x.float() + y.float()).to(x.dtype)

