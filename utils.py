import matplotlib.pyplot as plt
import time, sys, math
import numpy as np
import torch
from torch.autograd import Variable 
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import librosa
from scipy.io import wavfile

def display(string, variables) :
    sys.stdout.write(f'\r{string}' % variables)

def load_wav(filename, sample_rate, encode_16bits=True) :
    x = librosa.load(filename, sr=sample_rate)[0]
    if encode_16bits == True : 
        x = np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)
    return x

def save_wav(y, filename, sample_rate) :
    if y.dtype != 'int16' : y *= 2**15
    y = np.clip(y, -2**15, 2**15 - 1)
    wavfile.write(filename, sample_rate, y.astype(np.int16))

def split_signal(x) :
    unsigned = x + 2**15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine

def combine_signal(coarse, fine) :
    return coarse * 256 + fine - 2**15

def time_since(started) :
    elapsed = time.time() - started
    m = int(elapsed // 60)
    s = int(elapsed % 60)
    if m >= 60 :
        m = m % 60
        h = int(m // 60)
        return f'{h}h {m}m {s}s'
    else :
        return f'{m}m {s}s'

