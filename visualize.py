import h5py
import numpy as np
import matplotlib.pylab as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pdb
from networks import CNN_Model, UNet, SegNet
from tqdm import tqdm
from dataset_windows import SatelliteSet

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

BATCH_SIZE = 1
NUM_WORKERS = 0
CUDA_DEVICE = 'cuda:0'
if  __name__ == '__main__':
    test_dataset = SatelliteSet(windowsize=720,split='test')
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        shuffle = False
    )

    #net = CNN_Model()
    net = SegNet()
    net.load_state_dict(torch.load('./checkpoint/SegNet_2021-10-27-06-09-16_93.8442482213122.pth')['net'])
    net.to(torch.device(CUDA_DEVICE))
    net.eval()


    
    # count0 = 0
    # count1 = 0
    # count2 = 0
    # for batch in tqdm(test_dataloader):
    #     count0 = count0 + torch.sum((batch[1]==0)).detach().numpy()
    #     count1 = count1 + torch.sum((batch[1]==1)).detach().numpy()
    #     count2 = count2 + torch.sum((batch[1]==2)).detach().numpy()

    # print(1,count0/count1,count0/count2)
    # pdb.set_trace()
    total_test_loss = 0
    true_label_list = []
    pred_label_list = []


    idx = 0
    
    for batch in tqdm(test_dataloader):
        
        output = net(batch[0].to(torch.device(CUDA_DEVICE)))
        #print(batch[0].shape)
        mask = ((batch[0][:,0,:,:]+batch[0][:,1,:,:]+batch[0][:,2,:,:]+batch[0][:,3,:,:])==0).to(torch.device(CUDA_DEVICE))
        
        predictions = torch.argmax(output, dim = 1) 
       
        gt = batch[1].to(torch.device(CUDA_DEVICE))
        mask = (gt == 99)
        gt[mask] = 3
        predictions[mask] = 3
        acc = torch.mean(( predictions == gt).float())
        #print('Acc = %0.4f%%' %(acc*100))
        true_label_list.append(gt.view(-1).cpu().detach().numpy())
        pred_label_list.append(predictions.view(-1).cpu().detach().numpy())
        

        plt.rcParams["figure.figsize"] = (10,6)
        f,axarr = plt.subplots(ncols=2, nrows=1)
        axarr[0].set_title('Prediction')
        axarr[0].imshow(predictions[0,:,:].cpu().detach().numpy())
        axarr[1].set_title('Ground Truth')
        axarr[1].imshow(gt[0,:,:].cpu().detach().numpy())
        #plt.show()
        
        plt.savefig('./visualization/'+net.codename +'/'+str(idx)+'.JPG')
        plt.close()
        idx = idx + 1
        torch.cuda.empty_cache()
        


        #pdb.set_trace()
        #print(predictions)
    
    y_true = np.concatenate(true_label_list)
    y_pred = np.concatenate(pred_label_list)
    accuracy = accuracy_score(y_true,y_pred)
    precision, recall, f1 = precision_recall_fscore_support(y_true,y_pred,average='macro')[:-1]
    Confusion_M = confusion_matrix(y_true, y_pred)
    print("confusion matrix\n", Confusion_M[0:3,0:3])
    print("accuracy: ", accuracy)
    print("precision: ",precision)
    print("recall: ",recall)
    print("f1: ",f1)
