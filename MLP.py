import torch  # Main Package
import torchvision  # Package for Vision Related ML
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn  # Layers
import torch.nn.functional as F # Activation Functions
import torch.optim as optim # Optimizers


import torch  # Main Package
import torchvision  # Package for Vision Related ML
import torchvision.transforms as transforms  # Subpackage that contains image transforms

# Create the transform sequence
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert to Tensor
    # Normalize Image to [-1, 1] first number is mean, second is std deviation
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0,247, 0.243, 0.261)) 
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
])

# Load MNIST dataset
# Train
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                      download=True, transform=transform)
# Test
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)

# Send data to the data loaders
BATCH_SIZE = 128
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                          shuffle=False)


examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

print(example_data.shape)

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])



# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten() # For flattening the 2D image
        #self.fc1 = nn.Linear(32*32*3, 1024)  # Input is image with shape (28x28)
        #self.fc2 = nn.Linear(1024, 2048)  # First HL
        #self.fc3 = nn.Linear(2048, 1024)
        #self.fc4= nn.Linear(1024, 10) # Second HL
        self.fc1 = nn.Linear(32*32*3, 2000)  # Input is image with shape (28x28)
        self.fc2 = nn.Linear(2000, 2000)  # First HL
        self.fc3 = nn.Linear(2000, 2000)
        self.fc4= nn.Linear(2000, 10)
        self.output = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
      # Batch x of shape (B, C, W, H)
      x = self.flatten(x) # Batch now has shape (B, C*W*H)
      x = F.relu(self.fc1(x))  # First Hidden Layer
      x = F.relu(self.fc2(x))  # Second Hidden Layer
      x = self.dropout(x)
      x = F.relu(self.fc3(x))
      x = self.dropout(x)
      x = self.fc4(x)  # Output Layer
      x = self.output(x)  # For multi-class classification
      return x  # Has shape (B, 10)
    
# Identify device
device = ("cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Creat the model and send its parameters to the appropriate device
mlp = MLP().to(device)


# Test on a batch of data
with torch.no_grad():  # Don't accumlate gradients
  mlp.eval()  # We are in evalutation mode
  x = example_data.to(device)
  outputs = mlp(x)  # Alias for mlp.forward

  # Print example output.
  print(torch.exp(outputs[0]))





# Define the training and testing functions
def train(net, train_loader, criterion, optimizer, device):
    net.train()  # Set model to training mode.
    running_loss = 0.0  # To calculate loss across the batches
    for data in train_loader:
        inputs, labels = data  # Get input and labels for batch
        inputs, labels = inputs.to(device), labels.to(device)  # Send to device
        optimizer.zero_grad()  # Zero out the gradients of the ntwork i.e. reset
        outputs = net(inputs)  # Get predictions  (forward pass)
        loss = criterion(outputs, labels)  # Calculate loss (calculation of error)
        loss.backward()  # Propagate loss backwards (backwards propagate)
        optimizer.step()  # Update weights  (adjust weight)
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


mlp = MLP().to(device)

LEARNING_RATE = 0.01
MOMENTUM = 0.9

# Define the loss function, optimizer, and learning rate scheduler
criterion = nn.NLLLoss()
optimizer = optim.SGD(mlp.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=0.001)

# Train the MLP for 5 epochs
for epoch in range(15):
    train_loss = train(mlp, train_loader, criterion, optimizer, device)
    test_acc = test(mlp, test_loader, device)
    print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f}, Test accuracy = {test_acc:.4f}")


# Test on a batch of data
with torch.no_grad():  # Don't accumlate gradients
  mlp.eval()  # We are in evalutation mode
  x = example_data.to(device)
  outputs = mlp(x)  # Alias for mlp.forward

  # Print example output.
  print(torch.exp(outputs[0]))
  print(f'Prediction: {torch.max(outputs, 1)[1][0]}')