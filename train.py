from matplotlib.pyplot import get
from numpy import split
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

from tqdm import tqdm

from dataset_windows import SatelliteSet
from networks import UpdatingMean, CNN_Model
import pdb
BATCH_SIZE = 16
NUM_WORKERS = 8
NUM_EPOCHS = 5
CUDA_DEVICE = 'cuda:0'
def train_one_epoch(net, optimizer, dataloader):
    #raise NotImplementedError
    loss_aggregator = UpdatingMean()

    
    #set to training mode
    net.train()
    count = 0
    for batch in tqdm(dataloader):
        
        optimizer.zero_grad()
        
        
        output = net.forward(batch[0].to(torch.device(CUDA_DEVICE)))
        
        #print(predictions)
        
        
        #pdb.set_trace()
        
        # mse_loss = torch.nn.MSELoss()
        # output = torch.argmax(output, dim = 3)
        # print(output.shape,batch[1].shape)

        #unlabeled_mask = (batch[1] != 99)
        #print(output.shape,batch[1].shape)
        loss = F.cross_entropy(output, batch[1].to(torch.device(CUDA_DEVICE)), ignore_index = 99)

        count += 1
        if(count%1000 == 0):
            print("loss:",loss.data)

        loss.backward()
        optimizer.step()
        

        loss_aggregator.add(loss.item())
        
    
    return loss_aggregator.mean()


def compute_accuracy(output, labels):
    predictions = torch.argmax(output, dim = 1)
    mask = (labels != 99)
    #print(output.shape, predictions.shape,labels.shape)
    # pdb.set_trace()
    return torch.mean((predictions*mask == labels*mask).float())

def run_validation_epoch(net,dataloader):
    accuracy_aggregator = UpdatingMean()
    # Put the network in evaluation mode.
    net.eval()
    # Loop over batches.
    for batch in tqdm(dataloader):
        # Forward pass only.
        output = net.forward(batch[0].to(torch.device(CUDA_DEVICE)))

        # Compute the accuracy using compute_accuracy.
        accuracy = compute_accuracy(output, batch[1].to(torch.device(CUDA_DEVICE)))

        # Save accuracy value in the aggregator.
        accuracy_aggregator.add(accuracy.item())

    return accuracy_aggregator.mean()


if __name__ == '__main__':
    train_dataset = SatelliteSet(windowsize=32,split='train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        shuffle = True
    )

    validate_dataset = SatelliteSet(windowsize=32,split='validate')
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        shuffle = True
    )

    device = torch.device(CUDA_DEVICE)
    

    net = CNN_Model()
    net.to(device)

    optimizer = Adam(net.parameters())

    

    best_accuarcy = 0
    for epoch_idx in range(NUM_EPOCHS):
        #training part
        loss = train_one_epoch(net,optimizer,train_dataloader)
        print('Epoch %02d, Loss = %0.4f' %(epoch_idx + 1, loss))

        #validate part
        acc = run_validation_epoch(net, validate_dataloader)
        print('[Epoch %02d] Acc.: %.4f' % (epoch_idx + 1, acc * 100) + '%')

        #save checkpoint
        checkpoint = {
            'epoch_idx': epoch_idx,
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        if acc > best_accuarcy:
            best_accuarcy = acc
            torch.save(checkpoint, f'checkpoint/best-{net.codename}.pth')

        







