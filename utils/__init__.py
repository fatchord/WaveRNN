# Make it explicit that we do it the Python 3 way
from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import *

import sys
import torch

from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path

# Credit: Ryuichi Yamamoto (https://github.com/r9y9/wavenet_vocoder/blob/1717f145c8f8c0f3f85ccdf346b5209fa2e1c920/train.py#L599)
# Modified by: Ryan Butler (https://github.com/TheButlah)
# workaround for https://github.com/pytorch/pytorch/issues/15716
# the idea is to return outputs and replicas explicitly, so that making pytorch
# not to release the nodes (this is a pytorch bug though)

_output_ref = None
_replicas_ref = None

def data_parallel_workaround(model, *input):
    global _output_ref
    global _replicas_ref
    device_ids = list(range(torch.cuda.device_count()))
    output_device = device_ids[0]
    replicas = torch.nn.parallel.replicate(model, device_ids)
    # input.shape = (num_args, batch, ...)
    inputs = torch.nn.parallel.scatter(input, device_ids)
    # inputs.shape = (num_gpus, num_args, batch/num_gpus, ...)
    replicas = replicas[:len(inputs)]
    outputs = torch.nn.parallel.parallel_apply(replicas, inputs)
    y_hat = torch.nn.parallel.gather(outputs, output_device)
    _output_ref = outputs
    _replicas_ref = replicas
    return y_hat


def import_from_file(name, path):
    """Programmatically imports a module"""
    if not Path(path).exists():
        raise FileNotFoundError('"%s" doesn\'t exist!' % path)
    spec = spec_from_file_location(name, path)
    if spec is None:
        raise ValueError('could not load module from "%s"' % path)
    m = module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
