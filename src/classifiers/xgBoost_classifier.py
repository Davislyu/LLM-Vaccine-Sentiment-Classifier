import pandas as  pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'data/processed/final_embeddings.xlsx'
data = pd.read_excel(file_path)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

y = y.astype(int)


valid_labels = {1, 2, 3}
if not set(y.unique()).issubset(valid_labels):
    raise ValueError("Labels should be numeric (1, 2, 3).")

y = y - 1

label_classes = [0, 1, 2]
print(f"Encoded labels:\n{y.head()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Shapes of X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")

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

target_names = ['Supportive', 'Opposed', 'Neutral']

report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
print(f"Classification Report:\n{report}")

cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3])

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - XGBoost Classifier')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.savefig('xgboost_confusion_matrix.svg', format='svg')
plt.show()
