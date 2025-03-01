import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch import nn, optim
from tqdm import tqdm

# Define paths
train_data_dir = "/BIRD-SPECIES/dataset/train_data2/train_data2/"
save_path = "/Users/prakharjain/PycharmProjects/BIRD/BIRD-SPECIES/EfficientNetB0-Finetuned.pth"  # Path to save the trained model

# Parameters
img_size = (224, 224)
batch_size = 32
epochs = 20  # Increase number of epochs for better training

# Define data transformations (with augmentation for training)
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load training dataset
train_dataset = ImageFolder(root=os.path.join(train_data_dir), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained EfficientNetB0 model (with ImageNet weights)
model = models.efficientnet_b0(weights='IMAGENET1K_V1')  # Use new weights parameter

# Modify the classifier layer to match the number of classes
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(train_dataset.classes))

# Move model to the device (GPU/CPU)
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Scheduler to adjust learning rate during training (optional)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate average loss and accuracy for the epoch
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Step the scheduler
    scheduler.step()

    # Save the model after every epoch
    torch.save(model.state_dict(), save_path)

print(f"Model training complete. Saved to {save_path}")
