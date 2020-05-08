import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from utils import get_images,load_images,image_file_to_array
from data_generator import DataGenerator
from models import MANN

def train(num_classes,num_samples_per_class,meta_batch_size,steps,lr,root,device):
  test_losses=[]
  test_accuracies=[]
  gen=DataGenerator(root,num_classes,num_samples_per_class+1)
  model=MANN(num_classes,num_samples_per_class+1,device)
  model=model.to(device)
  criterion=nn.CrossEntropyLoss()
  optimizer=optim.Adam(model.parameters(),lr=lr)
  for n in range(1,steps+1):
    i,l=gen.sample_batch("train",meta_batch_size)
    n_labels=l[:,-1:].squeeze(1).reshape(-1,num_classes)
    target=torch.tensor(n_labels.argmax(axis=1))
    out=model(i,l)
    n_out=out[-num_classes:].transpose(0,1).contiguous().view(-1,num_classes)
    optimizer.zero_grad()
    target=target.to(device)
    loss=criterion(n_out,target)
    loss.backward()
    optimizer.step()
    if n%250==0:
      with torch.no_grad():
        i,l=gen.sample_batch("test",100)
        n_labels=l[:,-1:].squeeze(1).reshape(-1,num_classes)
        target=torch.tensor(n_labels.argmax(axis=1))
        target=target.to(device)
        out=model(i,l)
        n_out=out[-num_classes:].transpose(0,1).contiguous().view(-1,num_classes)
        pred=torch.tensor(n_out.argmax(axis=1))
        test_loss=criterion(n_out,target)
        test_losses.append(test_loss.item())
        test_acc=(1.0*(pred==target)).mean().item()
        print("Step: ",n,"\tTest accuracy: ",test_acc)
        test_accuracies.append(test_acc)
  return test_accuracies