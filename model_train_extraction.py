import torch
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
import os
import json

# Define the transformation for preprocessing images
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Function to extract features from an image using a pre-trained ResNet model
def extract_features(image_path, model):
    image = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Move the input and model to GPU for faster processing if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        # Forward pass to get features
        features = model(input_batch)

    # Move features to CPU and convert to a list
    features = features.squeeze().cpu().numpy().tolist()

    return features


# Load the pre-trained ResNet model
resnet_model = models.resnet18(pretrained=True)
resnet_model.fc = torch.nn.Linear(512, 1000)  # Modify the output layer to match your desired feature size
resnet_model.eval()

# Set the model to evaluation mode (no training)
resnet_model.eval()

# Directory paths for pets and celebrities
pets_path = './dataset/pets'
celebrities_path = './dataset/celebrities'

# Extract features for pet images
pet_features = []
for pet_image in os.listdir(pets_path):
    pet_image_path = os.path.join(pets_path, pet_image)
    pet_feature = extract_features(pet_image_path, resnet_model)
    pet_features.append({'features': pet_feature, 'label': 'pet', 'image_path': pet_image_path})

# Extract features for celebrity images
celeb_features = []
for celeb_dir in os.listdir(celebrities_path):
    celeb_dir_path = os.path.join(celebrities_path, celeb_dir)
    for celeb_image in os.listdir(celeb_dir_path):
        celeb_image_path = os.path.join(celeb_dir_path, celeb_image)
        celeb_feature = extract_features(celeb_image_path, resnet_model)
        celeb_features.append({'features': celeb_feature, 'label': celeb_dir, 'image_path': celeb_image_path})

# Save the extracted features to a JSON file
output_file_path = './features.json'
all_features = pet_features + celeb_features
with open(output_file_path, 'w') as f:
    json.dump(all_features, f)

print(f'Feature extraction completed. Features saved to {output_file_path}')
