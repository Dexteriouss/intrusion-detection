import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------------------------

# Read the training and test datasets
train_data = pd.read_csv('train_kdd_small.csv')
test_data = pd.read_csv('test_kdd_small.csv')

# Identify categorical and numerical features
categorical_columns = ['protocol_type', 'service', 'flag']
numerical_columns = [col for col in train_data.columns if col not in categorical_columns + ['label']]

# Encoding -----------------------------------------------------------------------------------------------

# Create labels for the categorical features
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = preprocessing.LabelEncoder()
    train_data[column] = label_encoders[column].fit_transform(train_data[column])       # Encodes the categorical features, then applies the encoding
    test_data[column] = label_encoders[column].transform(test_data[column])             # Only applies the previous encoding to prevent cross-contamination/learn from test data

# Scale the numerical features
scaler = preprocessing.StandardScaler()
train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])     # Encodes the numerical features, then applies the encoding
test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])           # Only applies the previous encoding to prevent cross-contamination/learn from test data

# Create a label encoder for the target variable
label_encoder = preprocessing.LabelEncoder()
train_data['label'] = label_encoder.fit_transform(train_data['label'])                  # Encodes the target variable, then applies the encoding
test_data['label'] = label_encoder.transform(test_data['label'])                        # Only applies the previous encoding to prevent cross-contamination/learn from test data

# Processing for use in NN ------------------------------------------------------------------------------

# Separate features and target
feature_train_data = train_data.drop('label', axis=1)
target_train_data = train_data['label']
feature_test_data = test_data.drop('label', axis=1)
target_test_data = test_data['label']

# Convert to PyTorch tensors
feature_train_tensor = torch.FloatTensor(feature_train_data.values)
target_train_tensor = torch.LongTensor(target_train_data.values)
feature_test_tensor = torch.FloatTensor(feature_test_data.values)
target_test_tensor = torch.LongTensor(target_test_data.values)

# Create data loaders
BATCH_SIZE = 32
train_dataset = TensorDataset(feature_train_tensor, target_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
test_dataset = TensorDataset(feature_test_tensor, target_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print("Preprocessed")

# --------------------------------------------------------------------------------------------------------
# Neural Network Parameters
# --------------------------------------------------------------------------------------------------------

# Define hyperparameters
LEARNING_RATE = 0.00001
NUM_EPOCHS = 20
input_shape = feature_train_data.shape[1]  # Get input shape from preprocessed data

# --------------------------------------------------------------------------------------------------------
# Neural Network Architecture
# --------------------------------------------------------------------------------------------------------

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.inputlayer = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.3)
        self.hiddenlayer = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.outputlayer = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.inputlayer(x))
        x = torch.relu(self.hiddenlayer(x))
        x = self.outputlayer(x)
        return x

# --------------------------------------------------------------------------------------------------------
# Training and Evaluation
# --------------------------------------------------------------------------------------------------------

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork(input_shape, num_classes=len(label_encoder.classes_)).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training history
history = {
    'train_loss': [],
    'train_acc': [],
    'validation_loss': [],
    'validation_acc': []
}

# Training loop
print("Training the model...")
for epoch in range(NUM_EPOCHS):
    model = model.train()
    train_running_loss = 0.0
    correct = 0
    total = 0
    
    # Use the train_loader to iterate through batches - largely influenced from PyTorch Quick Start
    for i, (inputs, labels) in enumerate(train_loader):
        # Attach to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        train_running_loss += loss.detach().item()

        predicted = torch.max(outputs.data, 1)[1]
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Calculate epoch statistics
    train_loss = train_running_loss / len(train_loader)
    train_acc = correct / total
    
    # Validation
    model.eval()
    validation_loss = 0.0
    validation_correct = 0
    validation_total = 0
    
    # Use the test_loader to iterate through batches
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        validation_loss += loss.item()
        predicted = torch.max(outputs.data, 1)[1]
        validation_total += labels.size(0)
        validation_correct += (predicted == labels).sum().item()
    
    validation_loss = validation_loss / len(test_loader)
    validation_acc = validation_correct / validation_total
    
    # Store history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['validation_loss'].append(validation_loss)
    history['validation_acc'].append(validation_acc)
    
    # Print progress
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, Val Loss: {validation_loss:.4f}, Val Accuracy: {validation_acc:.4f}')

# Final evaluation
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    # Use the test_loader for final evaluation
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert to numpy arrays to ensure consistent types
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# Calculate metrics
accuracy = accuracy_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions, average='weighted')

# Print metrics
print(f"\nTest Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, target_names=label_encoder.classes_))

# --------------------------------------------------------------------------------------------------------
# Plotting Accuracy Graph
# --------------------------------------------------------------------------------------------------------

def plot_performance(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(8, 5))
    
    # Plot accuracy
    plt.subplot(1, 1, 1)
    plt.plot(history['train_acc'])
    plt.plot(history['validation_acc'])
    plt.title('Model Accuracy')
    plt.xticks(range(0, len(history['train_acc'])))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('neural_network_performance.png')
    plt.close()

# Plot the performance
plot_performance(history)

RESULTSFILE = "Part_2_Results.txt"

# Save results to results file
with open(RESULTSFILE, "w") as f:
    f.write("Neural Network Results\n")
    # Parameters
    f.write("Parameters:\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Number of Epochs: {NUM_EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")

# Save Metrics to results file
with open(RESULTSFILE, "w") as f:
    # Metrics
    f.write("Performance Metrics:\n")
    f.write(f"Final Accuracy: {accuracy:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n\n")

