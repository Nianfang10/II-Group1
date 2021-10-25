# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 17:34:31 2021

@author: ruswang
"""

import numpy as np
import random
from dataset_windows_kmeans import SatelliteSet 
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import h5py
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import numpy.ma as ma
from tqdm import trange
import time

background = [30.05786977,46.8303811,28.48738374,10.61075796];
trees = [35.83015141,52.9400713,36.84029915,11.24572248];
clouds = [191.08824343,212.47070071,206.48104315,15.35621212];

def NN(input_signal):
    input_signal_cp = input_signal.copy();
    #input_label_cp = input_label.copy();
    len_signal = len(input_signal_cp[0]);
    fo
    return classified_label;

if __name__ == "__main__":
    dataset = SatelliteSet(windowsize = 32,test=False)
    dl = DataLoader(
        dataset,batch_size=1) 
    
    for data in tqdm(dl):
        signal = 255*data[0][0];
        label = data[1][0];
        
        outputlabel = NN(signal);
        