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
batch_size = 32
train_dataset = TensorDataset(feature_train_tensor, target_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_dataset = TensorDataset(feature_test_tensor, target_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print("Preprocessed")

# --------------------------------------------------------------------------------------------------------
# Neural Network Parameters
# --------------------------------------------------------------------------------------------------------

# Define hyperparameters
learning_rate = 0.001
num_epochs = 5
input_shape = feature_train_data.shape[1]  # Get input shape from preprocessed data
print(input_shape)

# --------------------------------------------------------------------------------------------------------
# Neural Network Architecture
# --------------------------------------------------------------------------------------------------------

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.inputlayer = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.5)
        self.hiddenlayer = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.5)
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
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training history
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

# Training loop
print("Training the model...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Use the train_loader to iterate through batches
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Calculate epoch statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        # Use the test_loader to iterate through batches
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_loss = val_loss / len(test_loader)
    val_acc = val_correct / val_total
    
    # Store history
    history['train_loss'].append(epoch_loss)
    history['train_acc'].append(epoch_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # Print progress
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')

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
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')

# Print metrics
print(f"\nTest Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, target_names=label_encoder.classes_))

# --------------------------------------------------------------------------------------------------------
# Plotting Performance
# --------------------------------------------------------------------------------------------------------

def plot_performance(history):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'])
    plt.plot(history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('neural_network_performance.png')
    plt.close()

# Plot the performance
plot_performance(history)

# Save results to file
with open("Neural_Network_Results.txt", "w") as f:
    f.write("Neural Network Results (PyTorch)\n")
    f.write("=" * 30 + "\n\n")
    f.write(f"Model Architecture:\n")
    f.write(str(model) + "\n\n")
    f.write("Hyperparameters:\n")
    f.write(f"Learning Rate: {learning_rate}\n")
    f.write(f"Number of Epochs: {num_epochs}\n")
    f.write(f"Batch Size: {batch_size}\n")
    f.write(f"Device: {device}\n\n")
    f.write("Performance Metrics:\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1 Score: {f1:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(all_labels, all_predictions, target_names=label_encoder.classes_))

print("Results saved to Neural_Network_Results.txt")
print("Performance plot saved to neural_network_performance.png")

