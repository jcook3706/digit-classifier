import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
            # images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = running_loss / len(val_loader)
    return accuracy, avg_loss

# Check for CUDA availability
# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    device = torch.device("cuda")  # Device object for GPU
    print("CUDA is available. Using GPU.")
    usingCuda = True
else:
    device = torch.device("cpu")   # Device object for CPU
    print("CUDA is not available. Using CPU.")

# Calculate mean and stdev of MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
print(type(train_dataset.data))
mean = torch.mean(train_dataset.data.float())/255.0
stdev = torch.std(train_dataset.data.float())/255.0
print(f'Mean of MNIST dataset: {mean}')
print(f'Standard Deviation of MNIST dataset: {stdev}')

# Define transformation for MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image or numpy array to torch.Tensor
    transforms.Normalize(mean, stdev)  # Normalize the data
])

# Download and load the MNIST dataset
mnistTrain = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Define the sizes for the training and validation sets
train_size = 50000
val_size = len(mnistTrain) - train_size

# Split the dataset into training and validation sets
trainset, valset = torch.utils.data.random_split(mnistTrain, [train_size, val_size])
print(f'Trainset size: {trainset.__len__()}')
print(f'Valset size: {valset.__len__()}')

# Create training and validation dataloaders
trainLoader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
valLoader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
print(f'Testset size: {testset.__len__()}')
testLoader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Define the CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 4 * 4)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create an instance of the network
net = Net()

# Define the loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
# Show performance of model on validation partition before training
valAccuracy, valLoss = evaluate(net, valLoader, criterion)
print(f'Validation Accuracy: {valAccuracy:.4f}, Validation Loss: {valLoss:.4f}')

# Train the network
for epoch in range(numEpochs):  # Loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):
        inputs, labels = data
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()  # Update weights
        running_loss += loss.item()
        if i % 1000 == 999:  # Print every 1000 mini-batches
            print(f'Epoch {epoch + 1}, Mini-batch {i + 1}, Loss: {running_loss / 1000:.3f}')
            running_loss = 0.0

    valAccuracy, valLoss = evaluate(net, valLoader, criterion)
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

print('Finished Training')

# Save the trained model
# PATH = './mnist_cnn.pth'
# torch.save(net.state_dict(), PATH)

# TODO: Test on the testset

testAccuracy, testLoss = evaluate(net, testLoader, criterion)
print(f'Test Accuracy: {testAccuracy:.4f}, Test Loss: {testLoss:.4f}')