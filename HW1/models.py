import numpy as np
import torch
import torch.nn as nn

class MANN(nn.Module):
    def __init__(self,num_classes,samples_per_class,embed_size=784):
        super().__init__()
        self.num_classes=num_classes
        self.samples_per_classe=samples_per_class
        self.embed_size=embed_size
        self.conv=nn.Sequential(
            nn.Conv2d(1,64,3),
            nn.BatchNorm2d(64,momentum=1,affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,64,3),
            nn.BatchNorm2d(64,momentum=1,affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,64,3),
            nn.BatchNorm2d(64,momentum=1,affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Flatten()
        )
        self.lstm=nn.LSTM(64+self.num_classes,128,bidirectional=True,dropout=0.5)
        self.linear=nn.Linear(128*2,self.num_classes)


    def forward(self, input_images, input_labels):
        b,_,n,_=input_labels.shape
        input_images=torch.tensor(input_images).view(-1,1,28,28)
        #input_images=input_images.to(device)
        input_images=self.conv(input_images)
        input_images=input_images.reshape(b,-1,64)
        input_labels=torch.tensor(input_labels).view(b,-1,n)
        #input_labels=input_labels.to(device)
        input_labels[:,-self.num_classes:]=0
        x=torch.cat((input_images,input_labels),-1).transpose(0,1)
        x,_=self.lstm(x)
        x=self.linear(x)
        return x