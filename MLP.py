import torch  
import torchvision  
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torch.optim.lr_scheduler as lr_scheduler

import torch  
import torchvision  
import torchvision.transforms as transforms 

#set seed for reliable results
torch.manual_seed(33)

transform = transforms.Compose([
   
    # Changes the data's format to a Tensor
    transforms.ToTensor(),
    
    #Performs transformation on the dataset with respect to the data's means and standard deviations
    
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
])

#Load data into two groups
# Training set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform)
# Testing set
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)


BATCH_SIZE = 128
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                          shuffle=False)


examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)







#MLP class object
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten() 
       
        self.fc1 = nn.Linear(32*32*3, 2040)  
        #using batch normalization
        self.bn1 = nn.BatchNorm1d(2040)

        self.fc2 = nn.Linear(2040, 1024)  
        self.bn2 = nn.BatchNorm1d(1024)

        self.fc3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc4 = nn.Linear(1024, 1024) 
        self.bn4 = nn.BatchNorm1d(1024)

        self.fc5= nn.Linear(1024, 10)
        self.output = nn.LogSoftmax(dim=1)
        #Using dropout to manage over-fitting
        self.dropout = nn.Dropout(0.15)
#The function that does a feed forward of the data through the network
    def forward(self, x):
      
      x = self.flatten(x) 
     
      x = F.relu(self.bn1(self.fc1(x))) 
      x = self.dropout(x)
      x = F.relu(self.bn2(self.fc2(x)))  
      x = self.dropout(x)
      x = F.relu(self.bn3(self.fc3(x)))
      x = self.dropout(x)
      x = F.relu(self.bn4(self.fc4(x)))
      x = self.dropout(x)
      x = self.fc5(x)
      x = self.output(x)  
      return x  
    

#selects device 
device = ("cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Creat the model and send its parameters to the appropriate device
mlp = MLP().to(device)



with torch.no_grad(): 
  mlp.eval()  
  x = example_data.to(device)
  outputs = mlp(x)  

 





#Training function
def train(net, train_loader, criterion, optimizer, device):
    net.train()  
    running_loss = 0.0  
    #Iterate through the data 
    for data in train_loader:
        inputs, labels = data  
        inputs, labels = inputs.to(device), labels.to(device) 
        optimizer.zero_grad() 
        outputs = net(inputs) 
        loss = criterion(outputs, labels)  

      


        loss.backward()  
        optimizer.step() 
        running_loss += loss.item()  
        #return training loss
    return running_loss / len(train_loader)

#Testing function
def test(net, test_loader, device):
    net.eval() 
    correct = 0
    total = 0
    #zero out the gradients
    with torch.no_grad():  
        #Iterate through the data 
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) 
            outputs = net(inputs) 
            _, predicted = torch.max(outputs.data, 1)  
            total += labels.size(0)
            correct += (predicted == labels).sum().item() 
    #calculate accuracy percentage
    return correct / total


mlp = MLP().to(device)
#Define our learning rate and momentum values
LEARNING_RATE = 0.01
MOMENTUM = 0.9

#Loss function
criterion = nn.NLLLoss()
#Function that generates new evaluated parameter values
optimizer = optim.SGD(mlp.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
#changes learning rate after 6"steps", a variable that is incremented with each epoch
scheduler = lr_scheduler.StepLR(optimizer, step_size = 6, gamma=0.1)


for epoch in range(15):
    #obtain estimation of training loss using the train function
    train_loss = train(mlp, train_loader, criterion, optimizer, device)
    test_acc = test(mlp, test_loader, device)
    print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")
    scheduler.step()



with torch.no_grad():  
  mlp.eval() 
  x = example_data.to(device)
  outputs = mlp(x)  


