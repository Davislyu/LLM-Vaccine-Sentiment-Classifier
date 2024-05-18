
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

- **Accuracy**: 0.8024
- **Macro F1-score**: 0.7924
- **Micro F1-score**: 0.8024

### Random Forest

The Random Forest classifier achieved the following performance metrics:

- **Accuracy**: 0.8193
- **Macro F1-score**: 0.8098
- **Micro F1-score**: 0.8193

### XGBoost

The XGBoost classifier achieved the following performance metrics:

- **Accuracy**: 0.8224
- **Macro F1-score**: 0.8140
- **Micro F1-score**: 0.8224

### Ensemble Voting Classifier

The Ensemble Voting classifier achieved the following performance metrics:

- **Accuracy**: 0.8254
- **Macro F1-score**: 0.8159
- **Micro F1-score**: 0.8254

## Cross-Validation Results

### Naive Bayes

- **10-Fold Cross-Validation Macro F1-scores**: 
  - 0.768991710852176
  - 0.7767794585842744
  - 0.7925819534723044
  - 0.8250161015451783
  - 0.7957712553620482
  - 0.832956450956451
  - 0.7407991820402785
  - 0.8247175793394281
  - 0.801288206703993
  - 0.7608739247280377
- **Mean Macro F1-score**: 0.791977582358417

### Random Forest

- **10-Fold Cross-Validation Macro F1-scores**: 
  - 0.8016892373644198
  - 0.8187189425561519
  - 0.8014392324093818
  - 0.8393489396828683
  - 0.8325971808982157
  - 0.835821418559013
  - 0.7952306108635274
  - 0.8330482461861054
  - 0.7890355086526175
  - 0.7970890698640917
- **Mean Macro F1-score**: 0.8144018387036391

### XGBoost

- **10-Fold Cross-Validation Macro F1-scores**: 
  - 0.8158351289039313
  - 0.8027214995626185
  - 0.8366675147290064
  - 0.8468542539310316
  - 0.8385612468671679
  - 0.838825601816941
  - 0.8114670209805276
  - 0.8473403533069915
  - 0.7960239716268959
  - 0.8091030789825971
- **Mean Macro F1-score**: 0.824339967070771

### Ensemble Voting Classifier

- **10-Fold Cross-Validation Macro F1-scores**: 
  - 0.8008516264294615
  - 0.808682753164557
  - 0.8055989341265416
  - 0.8535025132591328
  - 0.840007215007215
  - 0.8448675394922859
  - 0.8012656322498534
  - 0.8458656995486251
  - 0.7983264290553684
  - 0.786758953517907
- **Mean Macro F1-score**: 0.8185727295850947

### Summary of Results:
- **Naive Bayes**:
  - **Mean Macro F1-score**: 0.791977582358417

- **Random Forest**:
  - **Mean Macro F1-score**: 0.8144018387036391

- **XGBoost**:
  - **Mean Macro F1-score**: 0.824339967070771

- **Ensemble Voting**:
  - **Mean Macro F1-score**: 0.8185727295850947

### Interpretation:
- **XGBoost** performed the best among the individual classifiers, achieving a high mean Macro F1-score of 0.8243.
- The **Ensemble Voting Classifier** performed well with a mean Macro F1-score of 0.8186.
- **Random Forest** and **Naive Bayes** followed with mean Macro F1-scores of 0.8144 and 0.7920, respectively.

### Detailed Results:
| Model                 | Accuracy | Macro F1-score | Micro F1-score |
|-----------------------|----------|----------------|----------------|
| Naive Bayes           | 0.8024   | 0.7924         | 0.8024         |
| Random Forest         | 0.8193   | 0.8098         | 0.8193         |
| XGBoost               | 0.8224   | 0.8140         | 0.8224         |
| Ensemble Voting       | 0.8254   | 0.8159         | 0.8254         |

### 10-Fold Cross-Validation Results:
| Model           | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Fold 6 | Fold 7 | Fold 8 | Fold 9 | Fold 10 | Mean Macro F1-score |
|-----------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|---------|---------------------|
| Naive Bayes     | 0.7690 | 0.7768 | 0.7926 | 0.8250 | 0.7958 | 0.8330 | 0.7408 | 0.8247 | 0.8013 | 0.7609  | 0.7920              |
| Random Forest   | 0.8017 | 0.8187 | 0.8014 | 0.8393 | 0.8326 | 0.8358 | 0.7952 | 0.8330 | 0.7890 | 0.7971  | 0.8144              |
| XGBoost         | 0.8158 | 0.8027 | 0.8367 | 0.8469 | 0.8386 | 0.8388 | 0.8115 | 0.8473 | 0.7960 | 0.8091  | 0.8243              |
| Ensemble Voting | 0.8009 | 0.8087 | 0.8056 | 0.8535 | 0.8400 | 0.8449 | 0.8013 | 0.8459 | 0.7983 | 0.7868  | 0.8186              |

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
