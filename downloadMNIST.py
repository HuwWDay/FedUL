import torchvision
print("torchvision version:", torchvision.__version__)
torchvision.datasets.MNIST(root='./data', train=True, download=True)
torchvision.datasets.MNIST(root='./data', train=False, download=True)
print("MNIST dataset downloaded successfully.")