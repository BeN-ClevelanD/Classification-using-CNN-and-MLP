import torch 
import torchvision 
import torchvision.transforms as transforms  
import torch.optim as optim 
import torch.nn as nn  
import torch.nn.functional as F 

import torch.optim as optim 
import torch.optim.lr_scheduler as lr_scheduler


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


#selects device 
device = ("cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)





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




#Cnn class object
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #using batch normalization
        self.bn1 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn2 = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)
        #Using dropout to manage over-fitting
        self.drop = nn.Dropout(0.25)
#The function that does a feed forward of the data through the network
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = self.drop(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x
    



cnn = CNN().to(device)
#Define our learning rate and momentum values
LEARNING_RATE = 1e-1
MOMENTUM = 0.9

#Loss function
criterion = nn.CrossEntropyLoss()
#Function that generates new evaluated parameter values
optimizer = optim.SGD(cnn.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
#changes learning rate after 7 "steps", a variable that is incremented with each epoch
scheduler = lr_scheduler.StepLR(optimizer, step_size = 7, gamma=0.1)
print("15 epoch training time")
print(f"Connecting to {device}  ")
for epoch in range(15):
    #obtain estimation of training loss using the train function
    train_loss = train(cnn, train_loader, criterion, optimizer, device)
    test_acc = test(cnn, test_loader, device)
    print(f"Epoch {epoch+1}: Training loss = {train_loss:.4f}, Testing accuracy = {test_acc:.4f}")
    scheduler.step()








with torch.no_grad():  
  cnn.eval()  
  x = [data for data in test_loader][0][0].to(device)
  outputs = cnn(x)  

  


  