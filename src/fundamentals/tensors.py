import torch

# Create tensor
x = torch.tensor([[1, 2], [3, 4]])

print("Tensor:")
print(x)

print("\nShape:")
print(x.shape)

print("\nData Type:")
print(x.dtype)

print("\nDevice:")
print(x.device)

# GPU check
if torch.cuda.is_available():
    x = x.to("cuda")

    print("\nMoved to GPU:")
    print(x)

    print("\nGPU Device:")
    print(x.device)
else:
    print("\nCUDA GPU not available.")

x: torch.Tensor = torch.tensor(2.0, requires_grad=True)

x = torch.tensor(2.0, requires_grad=True)



#======================================================================================================================
#Basic Creation and working with tensors
x= torch.tensor([1,2,3])



torch.zeros(3,4)
print(x) # 3 rows and 4 columns of zeros
torch.ones(2,2)
print(x) # 2 rows and 2 columns of ones

torch.rand(2,3) # random values between 0 and 1
torch.randn(2,3) # random values from a normal distribution

torch.arange(0,10,2) # values from 0 to 10 with a step of 2
torch.linspace(0,1,5) # 5 values between 0 and 1

x.shape
x.device

#Tensor indexing and slicing
y = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print(y[0]) # first row
print(y[:,1]) # second column
print(y[1:3, 0:2]) # submatrix of rows 1-2 and columns 0-1

print("\n After reshaping:")
y.reshape(1,9) # reshape to 1 row and 9 columns
print(y) # original tensor remains unchanged

print("n After reshaping with view:")
y.view(1,9) # another way to reshape\
print(y) # original tensor remains unchanged

print("\n After transposing:")
y.t() # transpose
print(y) # original tensor remains unchanged

print("\n After permuting:")
y.permute(1,0) # permute dimensions
print(y) # original tensor remains unchanged

y.flatten() # flatten to 1D
print(y) # original tensor remains unchanged

print("\n After squeezing and unsqueezing:")
y.squeeze() # remove dimensions of size 1
print(y) # original tensor remains unchanged
y.unsqueeze(0) # add a dimension at position 0
print(y) # original tensor remains unchanged

y.grad
print(y.grad) # None, since y does not require gradients

z= y * 2
print(z) # element-wise multiplication

#matrix opereations
import torch

a = torch.tensor([
    [1, 2],
    [3, 4]
])

b = torch.tensor([
    [5, 6],
    [7, 8]
])

print(a @ b)

#broadcasting
import torch

x = torch.tensor([
    [1, 2],
    [3, 4]
])

print(x + 10)


#==============================================================================================================
#autograd

x = torch.tensor(2.0, requires_grad=True)

y = x ** 2

y.backward()

print(x.grad)

#bigger function in autograd

x = torch.tensor(2.0, requires_grad=True)

y = x**2 + 3*x + 1

y.backward()

print(x.grad)

#Gradient accumulation
import torch

x = torch.tensor(2.0, requires_grad=True)

y = x**2
y.backward()

print(x.grad)  # 4

y = x**2
y.backward()

print(x.grad)  # 8
# just grad and tensor with requires_grad=True will accumulate gradients, so we need to zero them out before the next backward pass
x.grad.zero_()  # zero out gradients

#======================================================================================================================


#Neural Network Basics 
import torch.nn as nn
import torch.optim as optim

# 1. Define a simple feedforward network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 4)   # input size=2, hidden size=4
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)   # hidden size=4, output size=1

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 2. Create model, loss, optimizer
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 3. Dummy training loop
for epoch in range(100):
    inputs = torch.randn(10, 2)   # batch of 10 samples, each with 2 features
    targets = torch.randn(10, 1)  # random target values

#working neural network using MINST dataset
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 1. Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 2. Define a simple feedforward network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)   # input layer
        self.fc2 = nn.Linear(128, 64)      # hidden layer
        self.fc3 = nn.Linear(64, 10)       # output layer (10 digits)

    def forward(self, x):
        x = x.view(-1, 28*28)              # flatten image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# model = Net()

3. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# # 4. Training loop
for epoch in range(5):  # 5 epochs
    for inputs, labels in trainloader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# # 5. Testing loop
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in testloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total:.2f}%")
# lets do the rest tomorrow 

print(f"Accuracy on test set: {100 * correct / total:.2f}%")

correct, total =0, 0
with torch.no_grad():
    for inputs, labels in testloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
