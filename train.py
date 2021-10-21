from matplotlib.pyplot import get
from numpy import logical_not, split
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, Adadelta

from tqdm import tqdm

from dataset_windows import SatelliteSet
from networks import UpdatingMean, CNN_Model
import pdb
import wandb


BATCH_SIZE = 16
NUM_WORKERS = 2
NUM_EPOCHS = 500
CUDA_DEVICE = 'cuda:0'
WINDOW_SIZE = 256
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
    for batch in tqdm(dataloader):
        # Forward pass only.
        output = net.forward(batch[0].to(torch.device(CUDA_DEVICE)))

        # Compute the accuracy using compute_accuracy.
        accuracy = compute_accuracy(output, batch[1].to(torch.device(CUDA_DEVICE)))
        loss = F.cross_entropy(output, batch[1].to(torch.device(CUDA_DEVICE)), ignore_index = 99)
        #wandb.log({"Validating loss": loss,"Validating acc.": accuracy})

        # Save accuracy value in the aggregator.
        accuracy_aggregator.add(accuracy.item())
        loss_aggregator.add(loss.item())

    return accuracy_aggregator.mean(),loss_aggregator.mean()


if __name__ == '__main__':
    wandb.init(project = 'II_project1',name = "Model 1")
    # config = wandb.config
    # config.batch_size = BATCH_SIZE
    # config.epochs = NUM_EPOCHS
    # config.lr = 0.01

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

    
    net = CNN_Model()
    #torch.nn.init.xavier_normal_(CNN_Model.parameters())
    net.to(device)

    optimizer = Adam(net.parameters())
    #optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9)
    
    #val_acc,val_loss = run_validation_epoch(net, validate_dataloader)
    #print('Initial accuracy:%.4f%%, loss:%.4f' %(val_acc*100,val_loss))

    

    best_accuarcy = 0
    for epoch_idx in range(NUM_EPOCHS):
        #training part
        loss = train_one_epoch(net,optimizer,train_dataloader)
        print('Epoch %02d,Training Loss = %0.4f' %(epoch_idx + 1, loss))

        #validate part
        val_acc, val_loss = run_validation_epoch(net, validate_dataloader)
        print('[Epoch %02d] Validating Acc.: %.4f%%, Loss:%.4f' % (epoch_idx + 1, val_acc * 100, val_loss))
        
        wandb.log({"Validating loss": val_loss,"Validating acc.": val_acc})

        #save checkpoint
        checkpoint = {
            'epoch_idx': epoch_idx,
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
        }

        if val_acc > best_accuarcy:
            best_accuarcy = val_acc
            torch.save(checkpoint, f'checkpoint/best-{net.codename}.pth')

        if epoch_idx % 10 == 0:
            test_acc,test_loss = run_validation_epoch(net, test_dataloader)
            print('[Epoch %02d] Test Acc.: %.4f' % (epoch_idx + 1, test_acc * 100) + '%')
            wandb.log({"Testing loss": test_loss,"Test acc.": test_acc})
    
    print('Best validating acc. %.4f%%',best_accuarcy)
    wandb.log({
        "Best acc.":best_accuarcy
    })

        







