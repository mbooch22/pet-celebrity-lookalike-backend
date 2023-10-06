# pet-celebrity-lookalike-backend/feature_extraction.py
import sys
import torch
from torchvision import models, transforms
from PIL import Image
import json
from scipy.spatial.distance import cosine
import os


def extract_features(image_path, model_path):
    # print(f"Image Path: {image_path}")
    # print(f"Model Path: {model_path}")
    # Load the image
    image = Image.open(image_path)
    image = image.convert('RGB')

    # Define the transformation
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Preprocess the image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Load the pre-trained ResNet model
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Linear(512, 1000)
    # model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Extract features
    with torch.no_grad():
        output = model(input_batch)

    # Convert features to JSON and print
    features = output.squeeze().numpy().tolist()

    return features


def load_celebrity_features(celebrity_features_path):
    with open(celebrity_features_path, 'r') as f:
        celebrity_features = json.load(f)
    return celebrity_features


def find_closest_matches(user_features, celebrity_features, num_matches=3):
    # Calculate similarity with each celebrity
    similarities = [
        {
            'label': celeb['label'],
            'similarity': calculate_similarity(user_features, celeb['features']),
            'image_path': os.path.basename(celeb['image_path'])
        }
        for celeb in celebrity_features
    ]

    # Sort by similarity (higher is better)
    sorted_matches = sorted(similarities, key=lambda x: x['similarity'], reverse=True)

    # Return the top N matches
    return sorted_matches[:num_matches]


def calculate_similarity(feature1, feature2):
    # Use cosine similarity, TODO try other distance metrics
    return 1 - cosine(feature1, feature2)


if __name__ == "__main__":
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    num_matches = int(sys.argv[3])

    if num_matches > 10:
        num_matches = 10

    user_features = extract_features(image_path, model_path)
    celebrity_features_path = 'src/celebrity_features.json'

    celebrity_features = load_celebrity_features(celebrity_features_path)

    # Find closest matches
    closest_matches = find_closest_matches(user_features, celebrity_features, num_matches)

    # print("Closest Matches:")
    # for match in closest_matches:
    #     print(f"Label: {match['label']}, Similarity: {match['similarity']}")

    print(json.dumps(closest_matches))
