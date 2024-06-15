"""
### CNN IMPLEMENTATION USING PYTORCH

"""

import torch
import torch.nn as nn                           #imports the neural network module which contains the nn superclass
import torch.optim as optim                     #imports the optimization algorithms such as gradient descent, adam etc
import torch.nn.functional as F                 #has all the parameter-less functions, imports the activation functions(relu etc), but those can also be found in the nn package
from torch.utils.data import DataLoader         #this provides a dataset class for data representation and a dataloader for iterating over the data among other things.
import torchvision.datasets as datasets         #pytorch comes with datasets which can be imported through this
import torchvision.transforms as transforms     #has methods to perform data augmentation operations such as cropping, resizing, normalization etc.

from pathlib import Path

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root_folder_path, transform=None, label=1):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.image_paths = os.listdir(root_folder_path)
        self.labels = [label] * len(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_folder_path, self.image_paths[idx])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label

# Set paths to the root folders
covid_folder_path = "/content/drive/MyDrive/X-ray/Enhanced_COVID_Images"
non_covid_folder_path = "/content/drive/MyDrive/X-ray/Enhanced_Non-COVID_Images"

# Define transformations for data augmentation (you can customize these)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# Create custom datasets
covid_dataset = CustomDataset(root_folder_path=covid_folder_path, transform=transform, label=1)
non_covid_dataset = CustomDataset(root_folder_path=non_covid_folder_path, transform=transform, label=0)

# Combine datasets
combined_dataset = torch.utils.data.ConcatDataset([covid_dataset, non_covid_dataset])

# Split the combined dataset into training and testing sets
train_size = int(0.8 * len(combined_dataset))
test_size = len(combined_dataset) - train_size
train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size])

# Create dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Display random images from the training dataset
def display_random_images(dataloader, num_images=10):
    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    # Display random images
    random_indices = np.random.choice(len(labels), num_images, replace=False)

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i, index in enumerate(random_indices):
        image = images[index].permute(1, 2, 0).numpy()  # Convert to NumPy array and rearrange channels
        label = labels[index].item()

        axes[i].imshow(image)
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')

    plt.show()

# Display random images from the training dataset
print("Random images from the training dataset:")
display_random_images(train_dataloader)

# Display random images from the testing dataset
print("Random images from the testing dataset:")
display_random_images(test_dataloader)

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 56 * 56, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

in_channels = 3
batch_size = 64
learning_rate = 0.01

#output labels
num_classes =2

#number of epochs the model is training for
num_epochs = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_losses = []

def train(epoch):

    train_loss=0

    for batch_idx, (data, targets) in enumerate(train_dataloader):
        data = data.to(device=device)
        targets = targets.to(device=device)
        scores = model(data)
        loss = criterion(scores, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss = train_loss/len(train_dataloader)

    train_losses.append(train_loss)

for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_dataloader:
        outputs = model(inputs)
        predicted = (outputs.squeeze() > 0.5).int()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")