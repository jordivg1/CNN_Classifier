import glob
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import wandb
import torch.nn.functional as F
from sklearn.utils import class_weight
from torch import optim

tamañominimo = 130

lmap2 = {'cereals': 0, 'coffee':1, 'friedegg':2, 'juice':3, 'milk':4,'pancake':5,'salat':6,'sandwich':7,'scrambledegg':8, 'tea':9   }

class MyDataset(Dataset):

  def __init__(self, feats, labs):
    features = []
    labels = []

    for file in feats:
        aux1 = np.load(file)
        features.append(aux1)
        print(aux1.shape)


    for file in labs:
        aux = pd.read_csv(file, index_col=None).iloc[:, 1:]
        aux = aux.replace(lmap2)
        print(aux.shape)
        labels.append(aux)

    label = pd.concat(labels, ignore_index=True)
    feat = np.concatenate(features)

    




    print(label.shape,feat.shape)

    self.x_train=torch.tensor(np.array(feat),dtype=torch.float64)
    self.y_train=torch.tensor(np.array(label),dtype=torch.long)

  def __len__(self):
    return len(self.y_train)
  
  def __getitem__(self,idx):
    return self.x_train[idx],self.y_train[idx]




class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 6, 3,padding = 1)
        self.conv12 = nn.Conv2d(6, 6, 3,padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3,padding = 1)
        self.conv22 = nn.Conv2d(16, 16, 3,padding = 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 3,padding = 1)
        self.pool3 = nn.MaxPool2d(2, 4)
        self.conv4 = nn.Conv2d(32, 64, 3,padding = 1)
        self.pool4 = nn.MaxPool2d(2, 4)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv12(F.relu(self.conv1(x)))))
        
        x = self.pool(F.relu(self.conv22(F.relu(self.conv2(x)))))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(self.fc3(x))
        return x


if __name__ == '__main__':

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  sweep_config = {}
  parameters_dict = {
    'learning_rate': {
        'values': [0.00009,0.000095,0.000088],
        #'value': 0.01

        }  ,
    'epochs': {
        'value': 750
        },
    'batch_size': {
          'value': 100
        },
    }
  sweep_config['parameters'] = parameters_dict
  sweep_config['method'] =  'grid' 

  sweep_id = wandb.sweep(sweep_config, project="cnnclassifier")



  
  print(device)
  # Set fixed random number seed
  feattrain = ['partitions2/s1cnn.npy','partitions2/s2cnn.npy','partitions2/s4cnn.npy']
  labtrain = ['partitions2/s1lab3.csv','partitions2/s2lab3.csv','partitions2/s4lab3.csv']

  featval = ['partitions2/s3cnn.npy']
  labval = ['partitions2/s3lab3.csv']

  
  
  


  
  # Run the training loop
  def train():
    with wandb.init():
        # Prepare dataset
      
        myDstrain=MyDataset(feattrain,labtrain)
        train_loader=DataLoader(myDstrain,wandb.config['batch_size'],shuffle=True)
        myDsval = MyDataset(featval,labval)
        validate_loader=DataLoader(myDsval,wandb.config['batch_size'],shuffle = True) 

        print(myDstrain.__len__(),myDsval.__len__())
        # Initialize the MLP
        cnn = Net().double()
        wandb.watch(cnn)
        print(cnn)
        cnn.to(device)

        dataiter = iter(train_loader)
        images, labels = dataiter.next()
        
        print(images.shape,labels.shape)
        print(images,labels)

          
          # Define the loss function and optimizer
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(cnn.parameters(), wandb.config['learning_rate'])

        step = 0
        for epoch in range(0,wandb.config['epochs']): 
            # Print epoch
            print(f'Starting epoch {epoch+1}')
            # Set current loss value
            current_loss = 0.0
            correct = 0
            correctrain = 0
            running_vall_loss = 0.0
            
            # Iterate over the DataLoader for training data
            for i, data in enumerate(train_loader, 0):
              step+=1
              cnn.train()
              # Get inputs
              inputs, targets = data[0].to(device), data[1].to(device)


              # Zero the gradients
              optimizer.zero_grad()
              
              # Perform forward pass
              outputs = cnn(inputs)
              # Compute loss
              loss = loss_function(outputs, targets.squeeze(1))
              
              # Perform backward pass
              loss.backward()
              
              # Perform optimization
              optimizer.step()
              
              # Print statistics
              correctrain += (torch.argmax(outputs,dim=1) == targets.squeeze(1)).float().sum()
              current_loss += loss.item()
              if i % 10 == 9:
                  print('Loss after mini-batch %5d: %.3f' %
                        (i + 1, current_loss / 10))
                  wandb.log({"Training-loss": current_loss / 10}, step=step)
                  current_loss = 0.0
            accuracytrain = 100 * correctrain / myDstrain.__len__()
            wandb.log({"Train Accuracy":accuracytrain})
            #Validation step
            with torch.no_grad(): 
                cnn.eval() 
                for i,data in enumerate(validate_loader,0): 
                   inputs, targets = data[0].to(device), data[1].to(device) 
                   outputs = cnn(inputs) 
                   print(torch.argmax(outputs,dim=1))
                   val_loss = loss_function(outputs, targets.squeeze(1))
					
                   # The label with the highest value will be our prediction 
                   running_vall_loss += val_loss.item()  
                   correct += (torch.argmax(outputs,dim=1) == targets.squeeze(1)).float().sum()
                accuracy = 100 * correct / myDsval.__len__()
                wandb.log({"Validation-loss": running_vall_loss / len(validate_loader),"Accuracy":accuracy,"epoch": epoch})
                print("Val Accuracy = {}".format(accuracy)) 
          # Process is complete.
        print('Training process has finished.')

  #if __name__  == '__main__':
    #train()
  wandb.agent(sweep_id, train)
  #PATH = './s1s2.pth'
  #torch.save(mlp.state_dict(), PATH)
