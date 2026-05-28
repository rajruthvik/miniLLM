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
    import torch

import torch

x = torch.tensor([
    [1, 2],
    [3, 4]
])

print(x[0][1])

x = torch.tensor(2.0, requires_grad=True)

y = x ** 2

y.backward()

print(x.grad)

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.tensor([1, 2, 3]).to(device)

print(x)
print(x.device)
