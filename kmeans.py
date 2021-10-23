# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 15:24:14 2021

@author: XPS
"""

# Random classification
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
#from KMeans import distance, kmeans, save_result
#BATCH_SIZE = 16
#NUM_WORKERS = 8
#NUM_EPOCHS = 5
#CUDA_DEVICE = 'cuda:0'

def loss_function(present_center, pre_center):
    present_center = np.array(present_center)
    previous_center = np.array(pre_center)
    return np.sum((present_center-previous_center)**2)

def classifer(input_signal, center):
    input_row, input_col = input_signal.shape #shape of input image
    pixels_labels = np.zeros((input_row, input_col))
    pixel_distance_t = []
    for i in range(input_row):
        for j in range(input_col):
            for k in range(len(center)):
                distance_t = np.sum(abs((input_signal[i,j])-center[k])**2)
                #distance_t = np.sum(abs((input_signal[i,j]).astype(int)-center[k].astype(int))**2)
                pixel_distance_t.append(distance_t) 
            pixels_labels[i, j] = int(pixel_distance_t.index(min(pixel_distance_t)))
            pixel_distance_t = [] 
    return pixels_labels
            
def k_means(input_signal, center_num, threshold):
    #center_num = 3
    input_signal_cp = np.copy(input_signal)
    input_row, input_col = input_signal_cp.shape
    pixels_labels = np.zeros((input_row, input_col))
    
    #rows and columns of random cluster center
    initial_center_row_num = [i for i in range(input_row)]
    random.shuffle(initial_center_row_num)
    initial_center_row_num = initial_center_row_num[:center_num]
    
    initial_center_col_num = [i for i in range(input_col)]
    random.shuffle(initial_center_col_num)
    initial_center_col_num = initial_center_col_num[:center_num]
    
    #current cluster center
    present_center = []
    for i in range(center_num):
        present_center.append(input_signal_cp[initial_center_row_num[i], initial_center_row_num[i]])
        #print(present_center)
    pixels_labels = classifer(input_signal_cp, present_center)
    num = 0 #itteration number
    while True:
        pre_centet = present_center.copy()
        #Calculate current cluster center  
        for n in range(center_num):
            temp = np.where(pixels_labels == n)
            if len(input_signal_cp[temp]) == 0:
                #print("len(input_signal_cp[temp]) == 0")
                present_center[n] = 0
            else:
                present_center[n] = sum(input_signal_cp[temp].astype(int)) / len(input_signal_cp[temp])
            
        pixels_labels = classifer(input_signal_cp, present_center)
        loss = loss_function(present_center, pre_centet)
        num = num + 1
        #print("Step:"+ str(num) + "  Loss:" + str(loss))
        if loss <= threshold:
            break
    return pixels_labels,present_center
                

'''def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid((images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=64):
    for images in dl:
        show_images(images, nmax)
        break'''
    
if __name__ == "__main__":
    dataset = SatelliteSet(windowsize = 32,test=False)
    dl = DataLoader(
        dataset,batch_size=1)
    i =0;
    null = [];
    similarity = [];
    colormap = [[47, 79, 79], [0, 255, 0], [255, 255, 255], [0, 0, 0]]
    # 0 = background, 1 = palm oil trees, 2 = clouds
    colormap = np.asarray(colormap)
    for data in dl:
        orglabel = data[1][0];
        orglabelnp = orglabel.numpy();
        RGB = 255*data[0][0][0];
        if 99 in orglabelnp:
            null.append(i);
        else:
            pixlabel, center = k_means(RGB,3,1);
            #print(pixlabel);
            realpixlabel = pixlabel.copy();
            center0 = int(center[0]);
            center1 = int(center[1]);
            center2 = int(center[2]);
            npRGB = RGB.numpy();
            index0 = np.argwhere(npRGB == center0);
            if len(index0) != 0:
                klabel0 = pixlabel[index0[0][0]][index0[0][1]];
                olabel0 = orglabelnp[index0[0][0]][index0[0][1]];
                realpixlabel[pixlabel == klabel0] = olabel0;
            index1 = np.argwhere(npRGB == center1);
            if len(index1) != 0:                
                klabel1 = pixlabel[index1[0][0]][index1[0][1]];
                olabel1 = orglabelnp[index1[0][0]][index1[0][1]];
                realpixlabel[pixlabel == klabel1] = olabel1;
            index2 = np.argwhere(npRGB == center2);
            if len(index2) != 0:
                klabel2 = pixlabel[index2[0][0]][index2[0][1]];
                olabel2 = orglabelnp[index2[0][0]][index2[0][1]];
                realpixlabel[pixlabel == klabel2] = olabel2;
            similarity.append((realpixlabel==orglabelnp).sum()/realpixlabel.size);
        i += 1;
        
        
        
        #print(pixlabel[i])
        i+=1;
        #print(255*data[0][0][0]);
        
    #img, lable = tqdm(dl)
    #for x, y in tqdm(dl):
     #   x = np.transpose(x, [0, 2, 3, 1])
      #  y = np.where(y == 99, 3, y)
       # k_means(x,3,0.02)
    #device = torch.device(CUDA_DEVICE)
    
    
    