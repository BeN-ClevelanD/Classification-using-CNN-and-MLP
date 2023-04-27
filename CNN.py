import torch  # Main Package
import torchvision  # Package for Vision Related ML
import torchvision.transforms as transforms  # Subpackage that contains image transforms
import torch.optim as optim # Optimizers
import torch.nn as nn  # Layers
import torch.nn.functional as F # Activation Functions

import torch.optim as optim # Optimizers
import torchvision.transforms as transforms



#ADD DATA AUG, ADD DROPOUT< 

# Create the transform sequence
transform = transforms.Compose([
    
    transforms.ToTensor(),  # Convert to Tensor
    # Normalize Image to [-1, 1] first number is mean, second is std deviation
    transforms.Normalize((0.5,), (0.5,)) 
])

# Load MNIST dataset
# Train
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform)
# Test
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)

# Send data to the data loaders
BATCH_SIZE = 64
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                          shuffle=False)


# Identify device
device = ("cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")




# Define the training and testing functions
def train(net, train_loader, criterion, optimizer, device):
    net.train()  # Set model to training mode.
    running_loss = 0.0  # To calculate loss across the batches
    for data in train_loader:
        inputs, labels = data  # Get input and labels for batch
        inputs, labels = inputs.to(device), labels.to(device)  # Send to device
        optimizer.zero_grad()  # Zero out the gradients of the ntwork i.e. reset
        outputs = net(inputs)  # Get predictions
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Propagate loss backwards
        optimizer.step()  # Update weights
        running_loss += loss.item()  # Update loss
    return running_loss / len(train_loader)



def test(net, test_loader, device):
    net.eval()  # We are in evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Don't accumulate gradients
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device) # Send to device
            outputs = net(inputs)  # Get predictions
            _, predicted = torch.max(outputs.data, 1)  # Get max value
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # How many are correct?
    return correct / total




# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
      
        self.fc2 = nn.Linear(120, 84)
     
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(0.15)
     

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
      
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
      
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    



cnn = CNN().to(device)

LEARNING_RATE = 0.1
MOMENTUM = 0.9

# Define the loss function, optimizer, and learning rate scheduler
criterion = nn.CrossEntropyLoss() # Use this if not using softmax layer
optimizer = optim.SGD(cnn.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

# Train the MLP for 5 epochs
for epoch in range(15):
    train_loss = train(cnn, train_loader, criterion, optimizer, device)
    test_acc = test(cnn, test_loader, device)
    print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")
    





# Test on a batch of data
with torch.no_grad():  # Don't accumlate gradients
  cnn.eval()  # We are in evalutation mode (i.e No dropout)
  x = [data for data in test_loader][0][0].to(device)
  outputs = cnn(x)  # Alias for cnn.forward

  # Print example output.
  print(outputs[0])
  print(f'Prediction: {torch.max(outputs, 1)[1][0]}')


  