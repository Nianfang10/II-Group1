from re import S
import torch
import torch.nn as nn
import pdb

WINDOW_SIZE = 128
class UpdatingMean():
    def __init__(self) -> None:
        self.sum = 0
        self.n = 0

    def mean(self):
        return self.sum / self.n

    def add(self,loss):
        self.sum += loss
        self.n += 1


class CNN_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.codename = 'CNN'

        self.layers = nn.Sequential(
            nn.Conv2d(4, 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),#[N, 16, 64, 64]
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(), #[N, 32, 32, 32]
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            #nn.BatchNorm2d(64),
            nn.ReLU(), #[N, 64, 32, 32]
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(), #[N, 64, 32, 32]
            #nn.Softmax(dim=1)
            nn.Upsample(scale_factor=2, mode = 'bilinear'),
            ##[N, 64, 64, 64]
            nn.Conv2d(64, 3, 1, stride=1),
            nn.Softmax(dim=1),
        )

        self.Classifier = nn.Sequential(
            #nn.Linear(16*WINDOW_SIZE*WINDOW_SIZE,3*WINDOW_SIZE*WINDOW_SIZE),
            
        )


    def forward(self, batch):

        x = self.layers(batch)
        # print(x.shape)
        # pdb.set_trace()

        b = x.size(0)
        #x = x.view(b,WINDOW_SIZE,WINDOW_SIZE,-1)
        #x = self.Classifier(x.view(b,-1)).view(b,3,WINDOW_SIZE,WINDOW_SIZE)
        
        # pdb.set_trace()
        return x

    