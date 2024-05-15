import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from embeddings.generate_embeddings import generate_embeddings
from models.classify_tweets import classify_tweets

def main():
    # נתיב לדאטאסט
    raw_data_path = os.path.join('data', 'raw', 'tweets.xlsx')
    processed_data_path = os.path.join('data', 'processed', 'final_embeddings.xlsx')

    # יצירת האימבדינגס
    generate_embeddings(raw_data_path, processed_data_path)

    # סיווג הטוויטים
    classify_tweets(processed_data_path)

if __name__ == "__main__":
    main()