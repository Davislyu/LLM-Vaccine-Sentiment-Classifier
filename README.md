
# LLM-Vaccine-Sentiment-Classifier

This project aims to classify Twitter posts regarding COVID-19 vaccines into supportive, opposed, or irrelevant categories using various machine learning models including Naive Bayes, Random Forest, and XGBoost. The project leverages embedding models such as CityBird (CT-BERT) and Sentence-BERT (SBERT) for feature extraction.

## Project Structure

```
LLM-Vaccine-Sentiment-Classifier/
│
├── data/
│   └── processed/
│       └── final_embeddings.xlsx
│
├── src/
│   └── classifiers/
│       ├── naive_bayes_classifier.py
│       ├── random_forest_classifier.py
│       └── xgboost_classifier.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Data

The data used in this project is stored in an Excel file located at `data/processed/final_embeddings.xlsx`. This file contains the tweet embeddings and their corresponding labels.

## Embedding Models

### CT-BERT

CityBird (CT-BERT) is a transformer-based model fine-tuned specifically for COVID-19 related text. It provides contextual embeddings that capture the nuances of language used in tweets about COVID-19 vaccines. You can find the CT-BERT model on Hugging Face [here](https://huggingface.co/digitalepidemiologylab/covid-twitter-bert).

### SBERT

Sentence-BERT (SBERT) is a modification of BERT that uses siamese and triplet network structures to derive semantically meaningful sentence embeddings that can be compared using cosine similarity. You can find the SBERT model on Hugging Face [here](https://huggingface.co/sentence-transformers/stsb-roberta-large).

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

## Results

The performance of each classifier is summarized below:

| Model       | Accuracy | Macro F1-score | Micro F1-score |
|-------------|----------|----------------|----------------|
| Naive Bayes | 0.8024   | 0.7924         | 0.8024         |
| Random Forest | 0.8193 | 0.8098         | 0.8193         |
| XGBoost     | 0.8224   | 0.8140         | 0.8224         |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
