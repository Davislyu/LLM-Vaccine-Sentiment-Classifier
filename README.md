# Twitter Posts Classification Project

This project classifies Twitter posts regarding COVID-19 vaccines into supportive, opposed, or irrelevant categories using embedding models such as CT-BERT and SBERT. The embeddings are generated using the CT-BERT model, refined using the SBERT model, and then saved into an Excel file.

## Project Structure

- `data/`
  - `raw/`: Contains the raw dataset (e.g., `tweets.xlsx`).
  - `processed/`: Contains processed datasets (e.g., `final_embeddings.xlsx`).
- `notebooks/`: Jupyter notebooks for exploratory data analysis (if needed).
- `src/`
  - `embeddings/`: Code to generate embeddings (`generate_embeddings.py`).
  - `models/`: Code for classification models (`classify_tweets.py`).
  - `main.py`: Main script to run the project.
- `tests/`: Unit tests.
- `.gitignore`: Specifies files and directories to be ignored by git.
- `README.md`: Project description and instructions.
- `requirements.txt`: List of required Python libraries.
- `setup.py`: Setup script for the project.

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Installation

1. **Clone the repository:**
   ```sh
   git clone <repository-url>
   cd project-root
Install dependencies:
   ``` pip install -r requirements.txt ```

## Running the Project

   Prepare the raw data:
      Place your raw dataset (e.g., tweets.xlsx) in the data/raw/ directory.

   Generate embeddings and refine them
   ``` python src/main.py ```

    :
        The refined embeddings will be saved in data/processed/final_embeddings.xlsx.

## Project Files

    data/raw/tweets.xlsx: The input dataset containing the tweets.
    data/processed/final_embeddings.xlsx: The output file containing the refined embeddings.
    src/embeddings/generate_embeddings.py: Script to generate and refine embeddings.
    src/main.py: Main script to run the embedding generation process.

## Example

To run the project, follow these steps:

 Ensure you have the necessary data in data/raw/tweets.xlsx.
    Run the main script:
    ```python src/main.py```
