import torch
import time
from transformers import BertTokenizer
from bert import SmallBERT, build_bert

# Assuming you've defined the SmallBERT class and build_bert function as shown above

def generate_embedding(model, text, tokenizer, max_length=16):
    """
    Generates the embedding for the input text using the SmallBERT model.
    
    Args:
        model: The SmallBERT model.
        text (str): The input text to generate embeddings for.
        tokenizer: Tokenizer to preprocess the input text.
        max_length (int): Maximum length for padding/truncating the input text.
    
    Returns:
        torch.Tensor: The embedding (last hidden state) for the input text.
    """
    # Tokenize and preprocess the input text
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_length)

    # Extract input IDs and attention mask from the tokenizer output
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Measure time before generating embeddings
    start_time = time.time()
    
    # Pass the inputs through the model to get the embeddings
    with torch.no_grad():  # Disable gradient calculation for inference
        embeddings = model(input_ids, attention_mask)
    
    # Measure time after generating embeddings
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Time taken to generate embeddings: {elapsed_time:.4f} seconds")
    
    return embeddings

# Example usage
if __name__ == "__main__":
    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-small')
    model = build_bert(args={'max_text_len': 16})

    # Sample text
    text = "This is a sample sentence to generate embeddings."

    # Generate embeddings and measure time
    embeddings = generate_embedding(model, text, tokenizer)

    print("Embeddings shape:", embeddings.shape)