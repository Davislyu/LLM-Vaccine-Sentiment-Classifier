import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

file_path = 'data/processed/final_embeddings.xlsx'
data = pd.read_excel(file_path)

X = data.iloc[:, :-1]  
y = data.iloc[:, -1]   

y = y.astype(int)

print(f"First few rows of labels to validate:\n{y.head()}")
print(f"Unique values in the label column:\n{y.unique()}")

valid_labels = {1, 2, 3}
if not set(y.unique()).issubset(valid_labels):
    raise ValueError("Labels should be numeric (1, 2, 3).")

label_classes = [1, 2, 3]
print(f"Encoded labels:\n{y.head()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Shapes of X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = rf_classifier.score(X_test, y_test)
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')

print(f"Accuracy: {accuracy}")
print(f"Macro F1-score: {macro_f1}")
print(f"Micro F1-score: {micro_f1}")

target_names = ['1', '2', '3']
report = classification_report(y_test, y_pred, target_names=target_names, labels=label_classes, zero_division=0)
print(f"Classification Report:\n{report}")
