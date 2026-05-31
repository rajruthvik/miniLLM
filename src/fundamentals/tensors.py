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

