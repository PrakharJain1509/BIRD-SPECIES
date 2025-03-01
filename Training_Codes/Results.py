import os
import torch
from torchvision import transforms, models
from torch import nn
from PIL import Image

# Define paths
test_image_path = "nature-tropics-background-bird-wallpaper-preview.jpg"  # Path to the image you want to test
save_path = "/Users/prakharjain/PycharmProjects/BIRD/BIRD-SPECIES/EfficientNetB0-Finetuned.pth"  # Path where the trained model is saved

# Parameters
img_size = (224, 224)

# Define data transformations (for testing)
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and transform the image
image = Image.open(test_image_path)
image = transform(image).unsqueeze(0)  # Add batch dimension

# Define device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = models.efficientnet_b0(weights=None)  # No weights needed as we'll load our trained model
num_ftrs = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_ftrs, len(os.listdir("/path/to/your/dataset/classes")))  # Adjust for number of classes

# Load model weights
model.load_state_dict(torch.load(save_path))

# Move model to the device (GPU/CPU)
model.to(device)
model.eval()  # Set model to evaluation mode

# Move the image to the device
image = image.to(device)

# Perform inference
with torch.no_grad():  # Disable gradient calculations for inference
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

# Print the predicted class
predicted_class = predicted.item()  # Get the predicted class index
class_names = os.listdir("/path/to/your/dataset/classes")  # Get the class names from the dataset folder
print(f"Predicted class: {class_names[predicted_class]}")
