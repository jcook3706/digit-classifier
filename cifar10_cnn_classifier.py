import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import numpy as np

# Parameters and variables
numEpochs = 100
earlyStoppingPatience = 10
earlyStoppingCount = 0
bestValLoss = float('inf')
usingCuda = False

# Function definitions
def evaluate(model, val_loader, criterion):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = running_loss / len(val_loader)
    return accuracy, avg_loss

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Device object for GPU
    print("CUDA is available. Using GPU.")
    usingCuda = True
else:
    device = torch.device("cpu")   # Device object for CPU
    print("CUDA is not available. Using CPU.")

# Download and load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Convert NumPy array to PyTorch tensor
data_tensor = torch.tensor(train_dataset.data).permute(0, 3, 1, 2)  # Convert to tensor and change dimensions

# Compute mean and std for each channel
mean = torch.mean(data_tensor.float(), dim=(0, 2, 3))/256.0  # Calculate mean
std = torch.std(data_tensor.float(), dim=(0, 2, 3))/256.0    # Calculate standard deviation

print("Mean of each channel:", mean)
print("Standard deviation of each channel:", std)

# Define transformation for MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image or numpy array to torch.Tensor
    transforms.Normalize(mean, std)  # Normalize the data
])

# Download and load the MNIST dataset
cifarTrain = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Define the sizes for the training and validation sets
train_size = 40000
val_size = len(cifarTrain) - train_size

# Split the dataset into training and validation sets
trainset, valset = torch.utils.data.random_split(cifarTrain, [train_size, val_size])
print(f'Trainset size: {trainset.__len__()}')
print(f'Valset size: {valset.__len__()}')

# Create training and validation dataloaders
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
valLoader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
print(f'Testset size: {testset.__len__()}')
testLoader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv1s = nn.Conv2d(32, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2s = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3s = nn.Conv2d(128, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(3, 16, 3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv1s(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv2s(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv3s(x))
        x = x.view(-1, 128 * 4 * 4)
        # x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Create an instance of the network
net = Net()
net.to(device)

# Define the loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# optimizer = optim.AdamW(net.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
# Show performance of model on validation partition before training
valAccuracy, valLoss = evaluate(net, valLoader, criterion)
print(f'Validation Accuracy: {valAccuracy:.4f}, Validation Loss: {valLoss:.4f}')

# Train the network
start_time = time.time()
train_losses = []
validation_losses = []
num_total_epochs = 0
for epoch in range(numEpochs):  # Loop over the dataset multiple times
    num_total_epochs += 1
    last_time = time.time()
    running_loss = 0.0
    net.train()
    for i, data in enumerate(trainLoader, 0):
        inputs, labels = data
        inputs.to(device)
        labels.to(device)
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()  # Update weights
        running_loss += loss.item()
        if i % 1000 == 999:  # Print every 1000 mini-batches
            print(f'Epoch {epoch + 1}, Mini-batch {i + 1}, Loss: {running_loss / 1000:.4f}')
            running_loss = 0.0
    print(f'Time for training epoch {epoch+1}: {(time.time()-last_time):4f}')

    trainAccuracy, trainLoss = evaluate(net, trainLoader, criterion)
    train_losses.append(trainLoss)

    valAccuracy, valLoss = evaluate(net, valLoader, criterion)
    validation_losses.append(valLoss)
    print(f'Validation Accuracy: {valAccuracy:.4f}, Validation Loss: {valLoss:.4f}')

    # Check for early stopping
    if valLoss < bestValLoss:
        bestValLoss = valLoss
        earlyStoppingCount = 0
    else:
        earlyStoppingCount += 1
        if earlyStoppingCount > earlyStoppingPatience:
            print(f'Early sotpping now, validation loss not improved in {earlyStoppingPatience} epochs.')
            break
    
    # Scheduler Step
    scheduler.step(valLoss)

print(f'Time to train CNN classifier on training partition of CIFAR10: {(time.time()-start_time):.4f} seconds')

# Save the trained model
# PATH = './mnist_cnn.pth'
# torch.save(net.state_dict(), PATH)

# Test on the testset
start_time = time.time()
testAccuracy, testLoss = evaluate(net, testLoader, criterion)
print(f'Time to run inference on the test partition of MNIST: {(time.time()-start_time):.4f} seconds')
print(f'Test Accuracy: {testAccuracy:.4f}, Test Loss: {testLoss:.4f}')

# Graph train and test loss
x = np.linspace(0, num_total_epochs, num_total_epochs)
plt.plot(x, train_losses, color='red', label="Training Loss")
plt.plot(x, validation_losses, color='blue', label="Validation Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss During Model Training')
plt.show()