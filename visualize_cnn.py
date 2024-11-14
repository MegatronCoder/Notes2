import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1)
        ])
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        layer_outputs = []
        for i, layer in enumerate(self.layers):
            x = self.relu(layer(x))
            layer_outputs.append(x)
            if i < 2:  # Apply pooling only after the first two layers
                x = self.pool(x)
        return layer_outputs

def visualize_layer_outputs(outputs):
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    for i, output in enumerate(outputs):
        # Select the first channel for visualization
        axs[i].imshow(output[0, 0].detach().cpu().numpy(), cmap='gray')
        axs[i].set_title(f'Layer {i+1} Output')
        axs[i].axis('off')
    plt.tight_layout()
    plt.show()

# Load and preprocess data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_data = MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = DataLoader(mnist_data, batch_size=1, shuffle=True)

# Initialize the model
model = CustomCNN()

# Get a sample image
sample_image, _ = next(iter(data_loader))

# Forward pass through the model
layer_outputs = model(sample_image)

# Visualize the original image
plt.figure(figsize=(6, 6))
plt.imshow(sample_image[0, 0], cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.show()

# Visualize layer outputs
visualize_layer_outputs(layer_outputs)

# Print shape of each layer's output
for i, output in enumerate(layer_outputs):
    print(f"Layer {i+1} output shape: {output.shape}")
