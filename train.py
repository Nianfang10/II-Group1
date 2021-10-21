from matplotlib.pyplot import get
from numpy import split
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, Adadelta

from tqdm import tqdm

from dataset_windows import SatelliteSet
from networks import UpdatingMean, CNN_Model
import pdb
BATCH_SIZE = 16
NUM_WORKERS = 8
NUM_EPOCHS = 500
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
        # if(count%1000 == 0):
        #     print("loss:",loss.data)

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
    train_dataset = SatelliteSet(windowsize=128,split='train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        shuffle = True
    )

    validate_dataset = SatelliteSet(windowsize=128,split='validate')
    validate_dataloader = DataLoader(
        validate_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        shuffle = True
    )

    test_dataset = SatelliteSet(windowsize=128,split='test')
    test_dataloader = DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        shuffle = True
    )


    device = torch.device(CUDA_DEVICE)

    
    net = CNN_Model()
    #torch.nn.init.xavier_normal_(CNN_Model.parameters())
    net.to(device)

    optimizer = Adam(net.parameters())
    #optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9)
    
    acc = run_validation_epoch(net, validate_dataloader)
    print('Initial accuracy:%.4f' %(acc*100)+'%')

    

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

        if epoch_idx % 50 == 0:
            acc = run_validation_epoch(net, test_dataloader)
            print('[Epoch %02d] Test Acc.: %.4f' % (epoch_idx + 1, acc * 100) + '%')

        







