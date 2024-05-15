import torch

def classify_tweets(embeddings_path):
    embeddings = torch.load(embeddings_path)
    # כאן מיובא הקוד שמבצע את הסיווג
    # דוגמה בסיסית של הדפסת האימבדינגס
    print(embeddings)