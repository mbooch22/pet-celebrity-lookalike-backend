import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
from collections import Counter

# Load the features from the JSON file
with open('./features.json', 'r') as f:
    data = json.load(f)

# Extract features and labels from the data
features = [item['features'] for item in data]
labels = [item['label'] for item in data]

# Convert features and labels to PyTorch tensors
X_train_tensor = torch.tensor(features, dtype=torch.float32)
# Convert labels to unique indices
label_to_index = {label: i for i, label in enumerate(set(labels))}
y_train_tensor = torch.tensor([label_to_index[label] for label in labels], dtype=torch.long)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


# Define a simple classifier
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)


# Initialize the classifier
input_size = len(features[0])  # Assuming each feature vector has the same length
output_size = len(set(labels))  # Number of unique celebrity labels
classifier = SimpleClassifier(input_size, output_size)

# Calculate class weights
class_counts = Counter(labels)
class_weights = torch.tensor([1.0 / class_counts[label] for label in label_to_index], dtype=torch.float32)

# Normalize class weights
sum_weights = sum(class_weights)
class_weights_normalized = class_weights / sum_weights if sum_weights > 0 else class_weights

# Modify the loss function
criterion = nn.CrossEntropyLoss(weight=class_weights_normalized)
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# Train the model
epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = classifier(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# Evaluate the model
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor([label_to_index[label] for label in y_test], dtype=torch.long)

with torch.no_grad():
    test_outputs = classifier(X_test_tensor)
    _, predicted = torch.max(test_outputs, 1)
    accuracy = accuracy_score(y_test_tensor.numpy(), predicted.numpy())

# Save the model
torch.save(classifier.state_dict(), './resnet18New.pth')

print(f'Test Accuracy: {accuracy * 100:.2f}%')
