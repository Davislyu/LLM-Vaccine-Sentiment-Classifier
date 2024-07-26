import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = 'data/processed/final_embeddings.xlsx'
data = pd.read_excel(file_path)

# Separate features and target
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Ensure target is integer type
y = y.astype(int)

# Print unique values to ensure correct labels
print(f"First few rows of labels to validate:\n{y.head()}")
print(f"Unique values in the label column:\n{y.unique()}")

# Check for valid labels
valid_labels = {1, 2, 3}
if not set(y.unique()).issubset(valid_labels):
    raise ValueError("Labels should be numeric (1, 2, 3).")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Shapes of X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

# Train Support Vector Machine
svm_classifier = SVC(kernel='linear', random_state=42)  # Linear kernel for simplicity
svm_classifier.fit(X_train, y_train)

# Predict on test data
y_pred = svm_classifier.predict(X_test)

# Calculate metrics
accuracy = svm_classifier.score(X_test, y_test)
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')

print(f"Accuracy: {accuracy}")
print(f"Macro F1-score: {macro_f1}")
print(f"Micro F1-score: {micro_f1}")

# Print classification report
target_names = ['1', '2', '3']
report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
print(f"Classification Report:\n{report}")

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.savefig('svm_confusion_matrix.svg', format='svg')
plt.close()

# Extract support vectors
support_vectors = svm_classifier.support_vectors_

# Plotting support vectors (only if data is 2D for visualization purposes)
# If data has more than 2 dimensions, you should use a dimensionality reduction technique like PCA first.
# Here, we assume 2D data for simplicity.
if X_train.shape[1] == 2:
    plt.figure(figsize=(10, 8))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=30, cmap='winter', marker='o', label='Training data')
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], facecolors='none', edgecolors='r', s=100, label='Support Vectors')
    plt.title('Support Vectors')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig('svm_support_vectors.svg', format='svg')
    plt.close()
