import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'data/processed/final_embeddings.xlsx'
data = pd.read_excel(file_path)

# Separate features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

y = y.astype(int)


# Validate labels
valid_labels = {1, 2, 3}
if not set(y.unique()).issubset(valid_labels):
    raise ValueError("Labels should be numeric (1, 2, 3).")

target_names = ['Supportive', 'Opposed', 'Neutral']
label_classes = [1, 2, 3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Shapes of X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

rf_classifier = RandomForestClassifier(
    n_estimators=200,          # Number of trees in the forest
    max_depth=10,              # Maximum depth of the trees
    min_samples_split=5,       # Minimum number of samples required to split an internal node
    min_samples_leaf=2,        # Minimum number of samples required to be at a leaf node
    max_features='sqrt',       # Number of features to consider when looking for the best split
    random_state=42            # Random state for reproducibility
)
rf_classifier.fit(X_train, y_train)

# Predict on test data
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy and F1 scores
accuracy = rf_classifier.score(X_test, y_test)
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')

print(f"Accuracy: {accuracy}")
print(f"Macro F1-score: {macro_f1}")
print(f"Micro F1-score: {micro_f1}")

# Generate classification report
report = classification_report(y_test, y_pred, target_names=target_names, labels=label_classes, zero_division=0)
print(f"Classification Report:\n{report}")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=label_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Random Forest Classifier')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.savefig('random_forest_confusion_matrix.svg', format='svg')
plt.show()
