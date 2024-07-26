import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
file_path = 'data/processed/final_embeddings.xlsx'
data = pd.read_excel(file_path)

# Separate features and target
X = data.iloc[:, :-1]  
y = data.iloc[:, -1]   

# Ensure target is integer type
y = y.astype(int)

# Display first few rows of labels and unique values
print(f"First few rows of labels to validate:\n{y.head()}")
print(f"Unique values in the label column:\n{y.unique()}")

# Validate labels
valid_labels = {1, 2, 3}
if not set(y.unique()).issubset(valid_labels):
    raise ValueError("Labels should be numeric (1, 2, 3).")

# Adjust labels to start from 0
y = y - 1

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Shapes of X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

# Initialize classifiers
nb_classifier = GaussianNB()
rf_classifier = RandomForestClassifier(random_state=42)
xgb_classifier = xgb.XGBClassifier(random_state=42)

# Create a VotingClassifier
voting_classifier = VotingClassifier(estimators=[
    ('nb', nb_classifier),
    ('rf', rf_classifier),
    ('xgb', xgb_classifier)
], voting='hard')

# Train the VotingClassifier
voting_classifier.fit(X_train, y_train)

# Predict on test data
y_pred = voting_classifier.predict(X_test)

# Calculate accuracy and F1 scores
accuracy = voting_classifier.score(X_test, y_test)
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')

print(f"Accuracy: {accuracy}")
print(f"Macro F1-score: {macro_f1}")
print(f"Micro F1-score: {micro_f1}")

# Generate classification report
target_names = ['Supportive', 'Opposed', 'Neutral']  # Updated class names
report = classification_report(y_test, y_pred, target_names=target_names, labels=[0, 1, 2], zero_division=0)
print(f"Classification Report:\n{report}")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Voting Classifier')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.savefig('voting_classifier_confusion_matrix.svg', format='svg')
plt.show()
