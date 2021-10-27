from matplotlib.pyplot import get
from numpy import logical_not, split
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler

from tqdm import tqdm

from dataset_windows import SatelliteSet
from networks import UpdatingMean, CNN_Model, UNet, SegNet
import pdb
import wandb
from datetime import datetime

BATCH_SIZE = 8
NUM_WORKERS = 2
NUM_EPOCHS = 300
CUDA_DEVICE = 'cuda:0'
WINDOW_SIZE = 256
Learning_rate = 0.002
def train_one_epoch(net, optimizer, dataloader):
    #raise NotImplementedError
    loss_aggregator = UpdatingMean()

    
    #set to training mode
    net.train()
    count = 4
    for batch in dataloader:
        
        optimizer.zero_grad()
        output = net.forward(batch[0].to(torch.device(CUDA_DEVICE)))

        #unlabeled_mask = (batch[1] != 99)
        #print(output.shape,batch[1].shape)
        weights = torch.from_numpy(np.array([0.2, 1, 0.8])).float().to(torch.device(CUDA_DEVICE))
        loss = F.cross_entropy(output, batch[1].to(torch.device(CUDA_DEVICE)),weight = weights, ignore_index = 99,reduction='mean')

        count += 1
        # if(count%1000 == 0):
        #     print("loss:",loss.data)

        loss.backward()
        optimizer.step()
        wandb.log({"Training loss": loss})
        

        loss_aggregator.add(loss.item())
        
    
    return loss_aggregator.mean()


def compute_accuracy(output, labels):
    predictions = torch.argmax(output, dim = 1)
    #print(predictions.shape)
    mask = (labels != 99)
    #print(output.shape, predictions.shape,labels.shape)
    # pdb.set_trace()
    return torch.mean((predictions*mask == labels*mask).float())

def run_validation_epoch(net,dataloader):
    accuracy_aggregator = UpdatingMean()
    loss_aggregator = UpdatingMean()
    # Put the network in evaluation mode.
    net.eval()
    # Loop over batches.
    for batch in dataloader:
        # Forward pass only.
        output = net.forward(batch[0].to(torch.device(CUDA_DEVICE)))

        # Compute the accuracy using compute_accuracy.
        accuracy = compute_accuracy(output, batch[1].to(torch.device(CUDA_DEVICE)))
        weights = torch.from_numpy(np.array([0.2, 1, 0.8])).float().to(torch.device(CUDA_DEVICE))

        loss = F.cross_entropy(output, batch[1].to(torch.device(CUDA_DEVICE)),weight = weights, ignore_index = 99,reduction='mean')
        #wandb.log({"Validating loss": loss,"Validating acc.": accuracy})

        # Save accuracy value in the aggregator.
        accuracy_aggregator.add(accuracy.item())
        loss_aggregator.add(loss.item())

    return accuracy_aggregator.mean(),loss_aggregator.mean()


if __name__ == '__main__':
    train_dataset = SatelliteSet(windowsize=WINDOW_SIZE,split='train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        shuffle = True
    )

    validate_dataset = SatelliteSet(windowsize=WINDOW_SIZE,split='validate')
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        shuffle = False
    )

    test_dataset = SatelliteSet(windowsize=WINDOW_SIZE,split='test')
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        shuffle = False
    )


    device = torch.device(CUDA_DEVICE)

    
    #net = CNN_Model()
    #net = UNet()
    net = SegNet()
    net.to(device)

    optimizer = Adam(net.parameters(),lr = Learning_rate,weight_decay = 0.00005)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = [50,100,150], gamma = 0.4, verbose = False)
    #optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9)
    
    #val_acc,val_loss = run_validation_epoch(net, validate_dataloader)
    #print('Initial accuracy:%.4f%%, loss:%.4f' %(val_acc*100,val_loss))
    wandb.init(project = 'II_project1',name = net.codename + '_scheduler_weighted_loss')
    config = wandb.config
    config.batch_size = BATCH_SIZE
    config.epochs = NUM_EPOCHS
    config.lr = Learning_rate

    

    best_accuarcy = 0
    for epoch_idx in tqdm(range(NUM_EPOCHS)):
        #training part
        loss = train_one_epoch(net,optimizer,train_dataloader)
        print('[Epoch %02d]Training Loss = %0.4f' %(epoch_idx + 1, loss))

        #validate part
        val_acc, val_loss = run_validation_epoch(net, validate_dataloader)
        print('[Epoch %02d]Validating Acc.: %.4f%%, Loss:%.4f' % (epoch_idx + 1, val_acc * 100, val_loss))
        
        wandb.log({"epoch": epoch_idx,"Validating loss": val_loss,"Validating acc.": val_acc})

        scheduler.step()

        #save checkpoint
        checkpoint = {
            'epoch_idx': epoch_idx,
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        if val_acc > best_accuarcy:
            best_accuarcy = val_acc
            

        if epoch_idx % 10 == 0:
            test_acc,test_loss = run_validation_epoch(net, test_dataloader)
            print('[Epoch %02d] Test Acc.: %.4f' % (epoch_idx + 1, test_acc * 100) + '%')
            wandb.log({"Testing loss": test_loss,"Test acc.": test_acc})
            dt=datetime.now()
            date=dt.strftime('%Y-%m-%d-%H-%M-%S')
            torch.save(checkpoint, f'checkpoint/{net.codename}_'+date+'_'+str(val_acc * 100)+'.pth')
    
    print('Best validating acc. %.4f%%',best_accuarcy)
    wandb.log({
        "Best acc.":best_accuarcy
    })

       






