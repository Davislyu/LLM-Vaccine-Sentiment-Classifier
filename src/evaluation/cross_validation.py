import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.metrics import make_scorer, f1_score
from tqdm import tqdm


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

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

scoring = make_scorer(f1_score, average='macro')

def cross_validate_with_progress(estimator, X, y, cv, scoring, model_name):
    cv_scores = []
    for train_index, test_index in tqdm(cv.split(X, y), total=cv.get_n_splits(), desc=f"Cross-Validation - 😈{model_name}😈"):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        score = f1_score(y_test, y_pred, average='macro')
        cv_scores.append(score)
    return cv_scores

nb_classifier = GaussianNB()
nb_cv_scores = cross_validate_with_progress(nb_classifier, X, y, cv=kf, scoring=scoring, model_name="Naive Bayes")

rf_classifier = RandomForestClassifier(random_state=42)
rf_cv_scores = cross_validate_with_progress(rf_classifier, X, y, cv=kf, scoring=scoring, model_name="Random Forest")

xgb_classifier = xgb.XGBClassifier(random_state=42)
xgb_cv_scores = cross_validate_with_progress(xgb_classifier, X, y, cv=kf, scoring=scoring, model_name="XGBoost")

voting_classifier = VotingClassifier(estimators=[
    ('nb', nb_classifier),
    ('rf', rf_classifier),
    ('xgb', xgb_classifier)
], voting='hard')

voting_cv_scores = cross_validate_with_progress(voting_classifier, X, y, cv=kf, scoring=scoring, model_name="Voting Classifier")

print(f"Naive Bayes 10-Fold Cross-Validation Macro F1-scores: {nb_cv_scores}")
print(f"Naive Bayes Mean Macro F1-score: {np.mean(nb_cv_scores)}")

print(f"Random Forest 10-Fold Cross-Validation Macro F1-scores: {rf_cv_scores}")
print(f"Random Forest Mean Macro F1-score: {np.mean(rf_cv_scores)}")

print(f"XGBoost 10-Fold Cross-Validation Macro F1-scores: {xgb_cv_scores}")
print(f"XGBoost Mean Macro F1-score: {np.mean(xgb_cv_scores)}")

print(f"Voting Classifier 10-Fold Cross-Validation Macro F1-scores: {voting_cv_scores}")
print(f"Voting Classifier Mean Macro F1-score: {np.mean(voting_cv_scores)}")
