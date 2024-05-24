import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb

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

y = y - 1

label_classes = [0, 1, 2]
print(f"Encoded labels:\n{y.head()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Shapes of X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

# Check the distribution of the labels
print(f"Label distribution in training set: {pd.Series(y_train).value_counts()}")
print(f"Label distribution in test set: {pd.Series(y_test).value_counts()}")

# Print the amount of label "1" in training and test sets
print(f"Amount of label '1' in training set: {(y_train == 0).sum()}")
print(f"Amount of label '1' in test set: {(y_test == 0).sum()}")

xgb_classifier = xgb.XGBClassifier(random_state=42)
xgb_classifier.fit(X_train, y_train)

y_pred = xgb_classifier.predict(X_test)

accuracy = xgb_classifier.score(X_test, y_test)
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')




print(f"Accuracy: {accuracy}")

print(f"Macro F1-score: {macro_f1}")







print(f"Micro F1-score: {micro_f1}")

y_test = y_test + 1
y_pred = y_pred + 1

target_names = ['1', '2', '3']
report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
print(f"Classification Report:\n{report}")
