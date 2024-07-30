import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from tqdm import tqdm
import os

def load_data(filename):
    df = pd.read_excel(filename)
    return df["text"].tolist()

def get_ctbert_embeddings(texts, model, tokenizer, batch_size=32):
    model.eval()  
    all_embeddings = []

   
    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Processing texts with CT-BERT"):
        batch_texts = texts[i * batch_size : (i + 1) * batch_size]

        
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=300,
        )

        with torch.no_grad():  
            outputs = model(**inputs)
            # Calculate the mean embedding 
            mean_embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(mean_embeddings)

    return torch.cat(all_embeddings, dim=0)

def get_sbert_embeddings(texts, model, tokenizer, batch_size=32):
    model.eval()  # Set the model to evaluation mode
    all_embeddings = []

    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Processing texts with SBERT"):
        batch_texts = texts[i * batch_size : (i + 1) * batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=300,
        )

        with torch.no_grad():  
            outputs = model(**inputs)
            mean_embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(mean_embeddings)

    return torch.cat(all_embeddings, dim=0)

def generate_embeddings(input_file, output_file):
    tokenizer_ct = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert")
    model_ct = AutoModel.from_pretrained("digitalepidemiologylab/covid-twitter-bert")
    
    tokenizer_sbert = AutoTokenizer.from_pretrained("sentence-transformers/stsb-roberta-large")
    model_sbert = AutoModel.from_pretrained("sentence-transformers/stsb-roberta-large")

    texts = load_data(input_file)  
    ctbert_embeddings = get_ctbert_embeddings(texts, model_ct, tokenizer_ct, batch_size=16)  # Generate embeddings for CT-BERT
    sbert_embeddings = get_sbert_embeddings(texts, model_sbert, tokenizer_sbert, batch_size=16)  # Generate embeddings for SBERT

    combined_embeddings = ctbert_embeddings + sbert_embeddings

    embeddings_df = pd.DataFrame(combined_embeddings.detach().numpy())
    embeddings_df.to_excel(output_file, index=False)  

    print(f"Final embeddings saved to '{output_file}'.")

def main():
    input_file = os.path.join("data", "raw", "tweets.xlsx")
    output_file = os.path.join("data", "processed", "final_embeddings.xlsx")
    generate_embeddings(input_file, output_file)

if __name__ == "__main__":
    main()
