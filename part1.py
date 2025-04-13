from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

RESULTSFILE = "Part_1_Results.txt"

# --------------------------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------------------------

# Import the training and test datasets
train_data = pd.read_csv('train_kdd_small.csv')
test_data = pd.read_csv('test_kdd_small.csv')

# Identify categorical and numerical features
categorical_columns = ['protocol_type', 'service', 'flag']
numerical_columns = [col for col in train_data.columns if col not in categorical_columns + ['label']]

# Create labels for the categorical features
label_encoders = {}
for column in categorical_columns:
    label_encoders[column] = preprocessing.LabelEncoder()
    train_data[column] = label_encoders[column].fit_transform(train_data[column])       # Encodes the categorical features, then applies the encoding
    test_data[column] = label_encoders[column].transform(test_data[column])             # Only applies the previous encoding to prevent cross-contamination/learning from test data

# Scale the numerical features
scaler = preprocessing.StandardScaler()
train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])     # Scales the numerical features, then applies the scales
test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])           # Only applies the previous scaling to prevent cross-contamination/learning from test data

# Create a label encoder for the target variable
label_encoder = preprocessing.LabelEncoder()
train_data['label'] = label_encoder.fit_transform(train_data['label'])                  # Encodes the target variable, then applies the encoding
test_data['label'] = label_encoder.transform(test_data['label'])                        # Only applies the previous encoding to prevent cross-contamination/learning from test data

# Save the encoded datasets to new files - For debugging
train_data.to_csv('encoded_train_kdd.csv', index=False)
test_data.to_csv('encoded_test_kdd.csv', index=False)

# Separate features and target
feature_train_data = train_data.drop('label', axis=1)
target_train_data = train_data['label']
feature_test_data = test_data.drop('label', axis=1)
target_test_data = test_data['label']


# --------------------------------------------------------------------------------------------------------
# Running Models
# --------------------------------------------------------------------------------------------------------

# Initialize models
logistic_regression = LogisticRegression()
support_vector_machine = SVC()
random_forest = RandomForestClassifier()

# Clear results file
with open(RESULTSFILE, "w") as f:
    f.write("")

# Function to evaluate models
def run_model(model, feature_train_data, target_train_data, feature_test_data, target_test_data):
    # Train the model - Uses training data
    model.fit(feature_train_data, target_train_data)

    # Make predictions - Uses test data
    label_prediction = model.predict(feature_test_data)

    # Ensure Data Types are the same for comparison
    comparison_values = target_test_data.values
    
    # Calculate metrics - Uses built-in functions from sklearn to avoid calculation errors
    accuracy = accuracy_score(comparison_values, label_prediction)
    f1 = f1_score(comparison_values, label_prediction, average="weighted")

    # Write the results to a txt file for submission
    with open(RESULTSFILE, "a") as f:
        f.write(f"Model: {model}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"F1 Score: {f1}\n\n")

# Run Models
run_model(logistic_regression, feature_train_data, target_train_data, feature_test_data, target_test_data)
run_model(support_vector_machine, feature_train_data, target_train_data, feature_test_data, target_test_data)
run_model(random_forest, feature_train_data, target_train_data, feature_test_data, target_test_data)


