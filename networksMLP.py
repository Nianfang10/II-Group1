from re import S
import torch
import torch.nn as nn
import pdb
BATCH_SIZE = 4
WINDOW_SIZE = 16
class UpdatingMean():
    def __init__(self) -> None:
        self.sum = 0
        self.n = 0

    def mean(self):
        return self.sum / self.n

    def add(self,loss):
        self.sum += loss
        self.n += 1

class MLP(nn.Module):
    def __init__(self,n_Channel = 4):
        super(MLP, self).__init__()
        #WINDOW_SIZE = 32
        self.n_Channel = n_Channel
        self.layer = nn.Sequential(
            #nn.Linear(input_size, hidden_size),
            nn.Linear(BATCH_SIZE*n_Channel*WINDOW_SIZE*WINDOW_SIZE,BATCH_SIZE*8*WINDOW_SIZE*WINDOW_SIZE),
            nn.ReLU(),
            nn.Linear(BATCH_SIZE*8*WINDOW_SIZE*WINDOW_SIZE, BATCH_SIZE*3*WINDOW_SIZE*WINDOW_SIZE))
        
    def forward(self,x):
        x = x.view(-1)
        #x =torch.reshape(x,(-1,))
        x = self.layer(x).view(BATCH_SIZE,3,WINDOW_SIZE,WINDOW_SIZE)
        return x

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
            #nn.Softmax(dim=1),
        )

        self.Classifier = nn.Sequential(
            #nn.Linear(16*WINDOW_SIZE*WINDOW_SIZE,3*WINDOW_SIZE*WINDOW_SIZE),
            
        )


    def forward(self, batch):

        x = self.layers(batch)
        #print(x.shape,batch.shape)
        # pdb.set_trace()

        b = x.size(0)
        #x = x.view(b,WINDOW_SIZE,WINDOW_SIZE,-1)
        #x = self.Classifier(x.view(b,-1)).view(b,3,WINDOW_SIZE,WINDOW_SIZE)
        
        # pdb.set_trace()
        return x



class conv_block(nn.Module):
    """
    Convolution Block 
    
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_channels, out_channels):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2,  mode = 'bilinear'),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

    
class UNet(nn.Module):
    def __init__(self,in_channels = 4, out_channels = 3):
        super().__init__()

        self.codename = 'UNet'

        filters = [32,64,128,256]
        self.Maxpool = nn.MaxPool2d( 2, stride = 2)
        
        self.Conv1 = conv_block(in_channels,filters[0])
        self.Conv2 = conv_block(filters[0],filters[1])
        self.Conv3 = conv_block(filters[1],filters[2])
        self.Conv4 = conv_block(filters[2],filters[3])
        
        self.Up4 = up_conv(filters[3],filters[2])
        self.Up_conv4 = conv_block(filters[3],filters[2])
        
        self.Up3 = up_conv(filters[2],filters[1])
        self.Up_conv3 = conv_block(filters[2],filters[1])
        
        self.Up2 = up_conv(filters[1],filters[0])
        self.Up_conv2 = conv_block(filters[1],filters[0])
        
        
        self.Up_conv1 = conv_block(filters[0],out_channels)
        

    def forward(self, x):
        b1 = self.Conv1(x) #[size,size]
        
        b2 = self.Maxpool(b1) #[size/2,size/2]
        b2 = self.Conv2(b2)
        
        b3 = self.Maxpool(b2) #[size/4,size/4]
        b3 = self.Conv3(b3)
        
        b4 = self.Maxpool(b3) #[size/8,size/8]
        b4 = self.Conv4(b4)
        
        c4 = self.Up4(b4)   #[size/4,size/4]
        c4 = torch.cat((b3,c4),dim = 1)
        c4 = self.Up_conv4(c4)
        
        c3 = self.Up3(c4)  #[size/2,size/2]
        c3 = torch.cat((b2,c3),dim = 1)
        c3 = self.Up_conv3(c3)
        
        c2 = self.Up2(c3)   #[size,size]
        c2 = torch.cat((b1,c2),dim = 1)
        c2 = self.Up_conv2(c2)
        
        c1 = self.Up_conv1(c2)
        
        return c1
