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

RESULTSFILE = "Part_2_Results.txt"

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

# --------------------------------------------------------------------------------------------------------
# Neural Network Parameters
# --------------------------------------------------------------------------------------------------------

# Define hyperparameters
LEARNING_RATE = 0.00001
NUM_EPOCHS = 35
input_size = feature_train_data.shape[1]  # Get input size from preprocessed data

# Save parameters to results file
with open(RESULTSFILE, "w") as f:
    f.write("NEURAL NETWORK RESULTS\n")
    f.write("-" * 50 + "\n\n")
    # Parameters
    f.write("PARAMETERS:\n")
    f.write(f"Learning Rate: {LEARNING_RATE}\n")
    f.write(f"Number of Epochs: {NUM_EPOCHS}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n\n")
    f.write("RUNTIME METRICS:\n")

# --------------------------------------------------------------------------------------------------------
# Neural Network Architecture
# --------------------------------------------------------------------------------------------------------

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.inputlayer = nn.Linear(input_size, 64)
        #self.dropout1 = nn.Dropout(0.5)
        self.hiddenlayer = nn.Linear(64, 32)
        #self.dropout2 = nn.Dropout(0.3)
        self.outputlayer = nn.Linear(32, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.inputlayer(x))
        x = torch.relu(self.hiddenlayer(x))
        x = self.outputlayer(x)
        return x

# --------------------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------------------

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork(input_size, num_classes=len(label_encoder.classes_)).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training history
train_acc_history = []
validation_acc_history = []

# Training loop
for epoch in range(NUM_EPOCHS):
    model = model.train()
    train_running_loss = 0.0
    correct = 0
    total = 0
  
    # Use the train_loader to iterate through batches - largely influenced from PyTorch Quick Start
    for inputs, labels in train_loader:
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
    train_acc = correct / total
    
    # Validation
    model.eval()
    validation_correct = 0
    validation_total = 0
    
    # Use the test_loader to iterate through batches
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        predicted = torch.max(outputs.data, 1)[1]
        validation_total += labels.size(0)
        validation_correct += (predicted == labels).sum().item()
    
    # Metrics
    validation_acc = validation_correct / validation_total
    
    # Store metrics history
    train_acc_history.append(train_acc)
    validation_acc_history.append(validation_acc)
    
    # Add epoch training to results file
    with open(RESULTSFILE, "a") as f:
        f.write(f'Epoch {epoch+1}, Training Accuracy: {train_acc:.4f}, Validation Accuracy: {validation_acc:.4f}\n')

# --------------------------------------------------------------------------------------------------------
# Final Evaluation and Metrics
# --------------------------------------------------------------------------------------------------------
model.eval()
all_predictions = []
all_labels = []

# Use the test_loader for final evaluation
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    predicted = torch.max(outputs.data, 1)[1]
    all_predictions.extend(predicted.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

# Convert to numpy arrays to ensure consistent types
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)

# Calculate metrics
accuracy = accuracy_score(all_labels, all_predictions)
f1 = f1_score(all_labels, all_predictions, average='weighted')

# Save metrics to results file
with open(RESULTSFILE, "a") as f:
    # Metrics
    f.write("\n\nFINAL PERFORMANCE METRICS:\n")
    f.write(f"Final Accuracy: {accuracy:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n\n")

# --------------------------------------------------------------------------------------------------------
# Plotting Accuracy Graph
# --------------------------------------------------------------------------------------------------------

def plot_performance(train_acc_history, validation_acc_history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(8, 5))
    
    # Plot accuracy
    #plt.subplot(1)
    plt.plot(train_acc_history)
    plt.plot(validation_acc_history)
    plt.title('Model Accuracy')
    plt.xticks(range(0, len(train_acc_history)))
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('NN Accuracy Metrics.png')
    plt.close()

# Plot the performance
plot_performance(train_acc_history, validation_acc_history)

