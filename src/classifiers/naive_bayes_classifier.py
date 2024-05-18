import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, f1_score

# Load the data
file_path = 'data/processed/final_embeddings.xlsx'
data = pd.read_excel(file_path)

# Extract features and labels
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]  # The last column, which contains the labels

# Convert labels to integers (if they are not already)
y = y.astype(int)

# Validate the labels
print(f"First few rows of labels to validate:\n{y.head()}")
print(f"Unique values in the label column:\n{y.unique()}")

# Ensure the labels are 1, 2, 3
valid_labels = {1, 2, 3}
if not set(y.unique()).issubset(valid_labels):
    raise ValueError("Labels should be numeric (1, 2, 3).")

# Encode the labels
label_classes = [1, 2, 3]
print(f"Encoded labels:\n{y.head()}")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Shapes of X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

# Train the Naive Bayes classifier
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = nb_classifier.predict(X_test)

# Evaluate the model
accuracy = nb_classifier.score(X_test, y_test)
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')

print(f"Accuracy: {accuracy}")
print(f"Macro F1-score: {macro_f1}")
print(f"Micro F1-score: {micro_f1}")

# Generate classification report
target_names = ['1', '2', '3']
report = classification_report(y_test, y_pred, target_names=target_names, labels=label_classes, zero_division=0)
print(f"Classification Report:\n{report}")
