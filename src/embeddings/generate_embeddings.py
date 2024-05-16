import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from tqdm import tqdm
import os

def load_data(filename):
    df = pd.read_excel(filename)
    return df["text"].tolist()

# Function to get embeddings from CT-BERT
def get_ctbert_embeddings(texts, model, tokenizer, batch_size=32):
    model.eval()  # Set the model to evaluation mode
    #Evaluation mode in PyTorch is a state set for the model during inference (i.e., when you are using the model for prediction rather than training). 
    all_embeddings = []

    # Calculate the number of batches needed based on the size of the texts and batch size
    num_batches = (len(texts) + batch_size - 1) // batch_size

    for i in tqdm(range(num_batches), desc="Processing texts with CT-BERT"):
        batch_texts = texts[i * batch_size : (i + 1) * batch_size]

        # Tokenize the batch of texts
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=300,
        )

        # Compute embeddings without calculating gradients (to prevent unnecessary updates)
        with torch.no_grad():  
            outputs = model(**inputs)
            # Calculate the mean embedding across all words in the text
            mean_embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(mean_embeddings)

    # Concatenate all embeddings into one tensor
    return torch.cat(all_embeddings, dim=0)

# Function to get embeddings from SBERT
def get_sbert_embeddings(texts, model, tokenizer, batch_size=32):
    model.eval()  # Set the model to evaluation mode
    all_embeddings = []

    # Calculate the number of batches needed based on the size of the texts and batch size
    num_batches = (len(texts) + batch_size - 1) // batch_size

    # Process each batch of texts
    for i in tqdm(range(num_batches), desc="Processing texts with SBERT"):
        batch_texts = texts[i * batch_size : (i + 1) * batch_size]

        # Tokenize the batch of texts
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=300,
        )

        # Compute embeddings without calculating gradients (to prevent unnecessary updates)
        with torch.no_grad():  
            outputs = model(**inputs)
            # Calculate the mean embedding across all words in the text
            mean_embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(mean_embeddings)

    # Concatenate all embeddings into one tensor
    return torch.cat(all_embeddings, dim=0)

# Function to generate embeddings from both models and save them to an Excel file
def generate_embeddings(input_file, output_file):
    # Load the tokenizers and models for CT-BERT and SBERT
    tokenizer_ct = AutoTokenizer.from_pretrained("digitalepidemiologylab/covid-twitter-bert")
    model_ct = AutoModel.from_pretrained("digitalepidemiologylab/covid-twitter-bert")
    
    tokenizer_sbert = AutoTokenizer.from_pretrained("sentence-transformers/stsb-roberta-large")
    model_sbert = AutoModel.from_pretrained("sentence-transformers/stsb-roberta-large")

    texts = load_data(input_file)  # Load texts from the Excel file
    ctbert_embeddings = get_ctbert_embeddings(texts, model_ct, tokenizer_ct, batch_size=16)  # Generate embeddings for CT-BERT
    sbert_embeddings = get_sbert_embeddings(texts, model_sbert, tokenizer_sbert, batch_size=16)  # Generate embeddings for SBERT

    # Combine embeddings from both models, maintaining dimensional consistency
    combined_embeddings = ctbert_embeddings + sbert_embeddings

    # Convert embeddings to a pandas DataFrame
    embeddings_df = pd.DataFrame(combined_embeddings.detach().numpy())
    embeddings_df.to_excel(output_file, index=False)  # Save the DataFrame to an Excel file

    print(f"Final embeddings saved to '{output_file}'.")

# Main function to run the entire process
def main():
    input_file = os.path.join("data", "raw", "tweets.xlsx")
    output_file = os.path.join("data", "processed", "final_embeddings.xlsx")
    generate_embeddings(input_file, output_file)

if __name__ == "__main__":
    main()
