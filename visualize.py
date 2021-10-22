import h5py
import numpy as np
import matplotlib.pylab as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pdb
from networks import CNN_Model
from tqdm import tqdm
from dataset_windows import SatelliteSet

BATCH_SIZE = 1
NUM_WORKERS = 4
CUDA_DEVICE = 'cuda:0'
if  __name__ == '__main__':
    test_dataset = SatelliteSet(windowsize=1098,split='test')
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        shuffle = False
    )

    net = CNN_Model()
    net.load_state_dict(torch.load(f'./checkpoint/best-{net.codename}.pth')['net'])
    net.to(torch.device(CUDA_DEVICE))
    net.eval()


    idx = 0
    for batch in tqdm(test_dataloader):
        
        output = net(batch[0].to(torch.device(CUDA_DEVICE)))
        #print(batch[0].shape)
        mask = ((batch[0][:,0,:,:]+batch[0][:,1,:,:]+batch[0][:,2,:,:]+batch[0][:,3,:,:])!=0).to(torch.device(CUDA_DEVICE))
        
        predictions = torch.argmax(output, dim = 1) * mask
        #print(batch[0][0,:,:,:].shape,mask.shape)
        #predictions = predictions.cpu().detach().numpy()
        #print(predictions.shape)
        #plt.imshow(predictions[1])
        #plt.show()
        gt = batch[1].to(torch.device(CUDA_DEVICE))
        mask = (gt != 99)
        gt = gt * mask 
        acc = torch.mean(( predictions == gt).float())
        print('Acc = %0.4f%%' %(acc*100))
        
        plt.rcParams["figure.figsize"] = (10,6)
        f,axarr = plt.subplots(ncols=2, nrows=1)
        #axarr.set_title('Acc = %0.4f%%' %(acc*100))
        axarr[0].set_title('Prediction')
        axarr[0].imshow(predictions[0,:,:].cpu().detach().numpy())
        axarr[1].set_title('Ground Truth')
        axarr[1].imshow(gt[0,:,:].cpu().detach().numpy())
        #plt.show()
        plt.savefig('./visualization/'+str(idx)+'.JPG')
        idx = idx + 1

        #pdb.set_trace()
        #print(predictions)
