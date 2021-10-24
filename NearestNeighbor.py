# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 14:43:59 2021

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
#For each 
 
def NearestNeighbor(input_signal, input_label):
    input_signal_cp = np.copy(input_signal)
    input_label_cp = np.copy(input_label)
    input_row, input_col = input_signal_cp.shape
    pixels_labels = np.zeros((input_row, input_col))
    background = input_signal_cp[input_label_cp == 0] 
    background_center = np.sum(background)/len(background)
    trees = input_signal_cp[input_label_cp == 1] 
    trees_center = np.sum(trees)/len(trees)
    clouds = input_signal_cp[input_label_cp == 2] 
    clouds_center = np.sum(clouds)/len(clouds)
    return background_center,trees_center,clouds_center
       
if __name__ == "__main__":
    dataset = SatelliteSet(windowsize = 32,test=False)
    dl = DataLoader(
        dataset,batch_size=1)
    rbackground_center = np.zeros(4);
    rtrees_center = np.zeros(4);
    rclouds_center = np.zeros(4);
    background_center = np.zeros(4);
    trees_center = np.zeros(4);
    clouds_center = np.zeros(4);
    num = 1
    for data in tqdm(dl):
        signal = 255*data[0][0];
        label = data[1][0];
        for i in range(0,4):
            background_center[i],trees_center[i],clouds_center[i] = NearestNeighbor(signal[i],label);
            rbackground_center[i] = (background_center[i]+rbackground_center[i])/num;
            rtrees_center[i] = (trees_center[i]+rtrees_center[i])/num;
            rclouds_center[i] = (clouds_center[i]+rclouds_center[i])/num;
            #print(background_center)            
        print(rbackground_center)
        num += 1;
        