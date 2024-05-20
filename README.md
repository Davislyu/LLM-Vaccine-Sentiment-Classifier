
# LLM-Vaccine-Sentiment-Classifier

This project aims to classify Twitter posts regarding COVID-19 vaccines into supportive, opposed, or irrelevant categories using various machine learning models including Naive Bayes, Random Forest, and XGBoost. The project leverages embedding models such as COVID-Twitter-BERT (CT-BERT) and Sentence-BERT (SBERT) for feature extraction.

## Project Structure

```
LLM-Vaccine-Sentiment-Classifier/
│
├── data/
│ └── processed/
│ └── final_embeddings.xlsx
│
├── src/
│ ├── classifiers/
│ │ ├── naive_bayes_classifier.py
│ │ ├── random_forest_classifier.py
│ │ ├── xgboost_classifier.py
│ │ └── voting_classifier.py
│ └── evaluation/
│ └── cross_validation.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Data

The data used in this project is stored in an Excel file located at `data/processed/final_embeddings.xlsx`. This file contains the tweet embeddings and their corresponding labels.

## Embedding Models

### CT-BERT

COVID-Twitter-BERT (CT-BERT) is a transformer-based model fine-tuned specifically for COVID-19 related text. It provides contextual embeddings that capture the nuances of language used in tweets about COVID-19 vaccines. You can find the CT-BERT model on Hugging Face [here](https://huggingface.co/digitalepidemiologylab/covid-twitter-bert).

### SBERT

Sentence-BERT (SBERT) is a modification of BERT that uses siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine similarity. You can find the SBERT model on Hugging Face [here](https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens).

## Models

### Naive Bayes

The Naive Bayes classifier achieved the following performance metrics:

- **Accuracy**: 0.7107
- **Macro F1-score**: 0.7067
- **Micro F1-score**: 0.7107

### Random Forest

The Random Forest classifier achieved the following performance metrics:

- **Accuracy**: 0.7619
- **Macro F1-score**: 0.7573
- **Micro F1-score**: 0.7619

### XGBoost

The XGBoost classifier achieved the following performance metrics:

- **Accuracy**: 0.7862
- **Macro F1-score**: 0.7843
- **Micro F1-score**: 0.7862

### Ensemble Voting Classifier

The Ensemble Voting classifier achieved the following performance metrics:

- **Accuracy**: 0.7655
- **Macro F1-score**: 0.7622
- **Micro F1-score**: 0.7655

## Cross-Validation Results

### Naive Bayes

- **10-Fold Cross-Validation Macro F1-scores**: 
  - 0.6869506146908032
  - 0.7057253862391563
  - 0.7082123843455866
  - 0.7154541095017644
  - 0.7081590134676249
  - 0.7506295938657551
  - 0.6943318608661774
  - 0.7183845466880182
  - 0.7140522779181637
  - 0.7203213011698221
- **Mean Macro F1-score**: 0.7122221088752871

### Random Forest

- **10-Fold Cross-Validation Macro F1-scores**: 
  - 0.7500277601221228
  - 0.7672225972178861
  - 0.7544733193993615
  - 0.7493114200049827
  - 0.7716516421431496
  - 0.7954943356032286
  - 0.7452530546208895
  - 0.7401998053484454
  - 0.764328231292517
  - 0.758727175921576
- **Mean Macro F1-score**: 0.7596689341674159

### XGBoost

- **10-Fold Cross-Validation Macro F1-scores**: 
  - 0.779121962706227
  - 0.7883579530836989
  - 0.7806530126618464
  - 0.7721074197750838
  - 0.7973539872921398
  - 0.8246499105628962
  - 0.7850879434342879
  - 0.8210243046034328
  - 0.7978669817097942
  - 0.814729786451319
- **Mean Macro F1-score**: 0.7960953262280726

### Ensemble Voting Classifier

- **10-Fold Cross-Validation Macro F1-scores**: 
  - 0.7464253325564107
  - 0.7780922279139973
  - 0.755819448010008
  - 0.7567971660892505
  - 0.7762816890430445
  - 0.8039189027258297
  - 0.7502319668747216
  - 0.7691881495649714
  - 0.7726332069466385
  - 0.7768332848978009
- **Mean Macro F1-score**: 0.7686221374622674

### Summary of Results:
- **Naive Bayes**:
  - **Mean Macro F1-score**: 0.7122

- **Random Forest**:
  - **Mean Macro F1-score**: 0.7597

- **XGBoost**:
  - **Mean Macro F1-score**: 0.7961

- **Ensemble Voting**:
  - **Mean Macro F1-score**: 0.7686

### Interpretation:
- **XGBoost** performed the best among the individual classifiers, achieving a high mean Macro F1-score of 0.7961.
- The **Ensemble Voting Classifier** performed well with a mean Macro F1-score of 0.7686.
- **Random Forest** and **Naive Bayes** followed with mean Macro F1-scores of 0.7597 and 0.7122, respectively.

### Detailed Results:
| Model                 | Accuracy | Macro F1-score | Micro F1-score |
|-----------------------|----------|----------------|----------------|
| Naive Bayes           | 0.7107   | 0.7067         | 0.7107         |
| Random Forest         | 0.7619   | 0.7573         | 0.7619         |
| XGBoost               | 0.7862   | 0.7843         | 0.7862         |
| Ensemble Voting       | 0.7655   | 0.7622         | 0.7655         |

### 10-Fold Cross-Validation Results:
| Model           | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean Macro F1-score |
|-----------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|---------------------|
| Naive Bayes     | 0.6870 | 0.7057 | 0.7082 | 0.7155 | 0.7082 | 0.7506 | 0.6943 | 0.7184 | 0.7141 | 0.7203  | 0.7122              |
| Random Forest   | 0.7500 | 0.7672 | 0.7545 | 0.7493 | 0.7717 | 0.7955 | 0.7453 | 0.7402 | 0.7643 | 0.7587  | 0.7597              |
| XGBoost         | 0.7791 | 0.7884 | 0.7807 | 0.7721 | 0.7974 | 0.8246 | 0.7851 | 0.8210 | 0.7979 | 0.8147  | 0.7961              |
| Ensemble Voting | 0.7464 | 0.7781 | 0.7558 | 0.7568 | 0.7763 | 0.8039 | 0.7502 | 0.7692 | 0.7726 | 0.7768  | 0.7686              |

## Installation

To run this project, you need to have Python installed. You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

### Naive Bayes Classifier

To run the Naive Bayes classifier, execute the following command:

```bash
python src/classifiers/naive_bayes_classifier.py
```

### Random Forest Classifier

To run the Random Forest classifier, execute the following command:

```bash
python src/classifiers/random_forest_classifier.py
```

### XGBoost Classifier

To run the XGBoost classifier, execute the following command:

```bash
python src/classifiers/xgboost_classifier.py
```
### XGBoost Classifier

To run the voting ensable classifier, execute the following command:

```bash
python src/classifiers/voting_classifier.py
```

### Cross-Validation

To run the cross-validation script for all classifiers, execute the following command:

```bash
python src/evaluation/cross_validation.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
