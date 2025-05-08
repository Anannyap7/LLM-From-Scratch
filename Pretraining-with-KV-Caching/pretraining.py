import torch
import torch.nn as nn
import tiktoken
import requests
import os
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset, DataLoader
from architecture import GPTModel


# INITIALIZE DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# INITIALIZE MODEL CONFIGS
GPT_CONFIG_124M = {
    "vocab_size": 50257,       # Vocabulary Size
    "context_length": 256,    # Context length (max number of input tokens model can handle)
    "emb_dim": 768,            # Embedding dimension
    "num_heads": 12,             # Number of attention heads
    "num_layers": 12,            # Number of layers/transformer blocks
    "drop_rate": 0.1,          # Dropour rate
    "qkv_bias": False,         # Query-Key-Value bias while weight initialization
    "max_batch_size": 5,       # max batch size for cached keys and values matrix
    "max_seq_len": 256        # max seg length for cached keys and values matrix 
}

# LOAD TEXT FILE
def load_text_file(file_path):
    try:
        # Convert to Path object for better path handling
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r', encoding="utf-8") as file:
            return file.read()
        
    except Exception as e:
        print(f"Error loading file: {e}")
        raise

# INSTANTIATE TOKENIZER
tokenizer = tiktoken.get_encoding("gpt2")

# CREATE INPUT-OUTPUT PAIRS USING SLIDING WINDOW APPROACH
class DatasetCreation(Dataset):
    def __init__(self, text, max_length, stride):
        self.inputs, self.targets = [], []
        
        encoded_text = tokenizer.encode(text)
        
        for i in range(0, len(encoded_text)-max_length, stride):
            input_chunk = encoded_text[i : i+max_length]
            target_chunk = encoded_text[i+1 : i+max_length+1]
            self.inputs.append(torch.tensor(input_chunk))
            self.targets.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

# SPLIT DATASET AND CREATE DATA-LOADERS

# Function to create dataloader
def create_dataloader(text, max_length=256, stride=128, batch_size=4, shuffle=True, drop_last=True, num_workers=0):
    dataset = DatasetCreation(text, max_length, stride)
    # Calls the getitem function to create batches
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader

# Load and split the dataset
file_path = "../the-verdict.txt"
print(f"Loading file from: {file_path}")
raw_text = load_text_file(file_path)
print(f"Raw text length: {len(raw_text)}")

# Split the data
train_ratio = 0.9
split_index = int(len(raw_text) * train_ratio)
train_data = raw_text[:split_index]
val_data = raw_text[split_index:]
print(f"Train data length: {len(train_data)}")
print(f"Validation data length: {len(val_data)}")

# Create dataloaders
train_loader = create_dataloader(
    train_data,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    batch_size=2,
    shuffle=True,
    drop_last=True,
    num_workers=0
)

val_loader = create_dataloader(
    val_data,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    batch_size=2,
    shuffle=True,
    drop_last=True,
    num_workers=0
)


# HELPER FUNCTIONS
def text_to_token_ids(text, tokenizer):
    encoded_text = tokenizer.encode(text)
    encoded_text = torch.tensor(encoded_text, dtype=torch.long)  # Convert list to tensor
    encoded_text = encoded_text.unsqueeze(0)  # to create a batch dimension
    return encoded_text

def token_ids_to_text(token_ids, tokenizer):
    token_ids = token_ids.squeeze(0)  # remove the batch dimension
    decoded_text = tokenizer.decode(token_ids.tolist())
    return decoded_text


# FUNCTIONS TO CALCULATE LOSS IN A BATCH AND LOSS OVER ALL BATCHES

# Calculate loss within a batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device) # (B, num_tokens)
    logits = model(input_batch) # (B, num_tokens, vocab_size)
    # this function takes the argmax from vocab_size automatically for logits, but
    # it needs the logits to be of the form (num_samples, num_classes); which is
    # (B*num_tokens, vocab_size) in our case.
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

# Calculate loss over all the batches
def calc_loss_batches(data_loader, model, device, num_batches=None):
    total_loss = 0
    
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else: # If the number of batches is greater than the length of the dataloader
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
        
    return total_loss


# TEXT GENERATION FUNCTION DURING INFERENCE
def generate(model, inputs, max_new_tokens, context_size, top_k=None, temp_scale = 0.0, eos_token=50256, use_kv_cache=False):
    """
    Generate text using the model with optional KV caching.
    
    Args:
        model: The GPT model
        inputs: Initial input tokens (shape: [batch_size, seq_len])
        max_new_tokens: Maximum number of new tokens to generate
        context_size: Maximum context length to maintain
        top_k: Number of top tokens to consider for sampling (None for all tokens)
        temp_scale: Temperature for sampling (0.0 for greedy sampling)
        eos_token: End of sequence token ID
        use_kv_cache: Whether to use KV caching for faster generation
    """
    # Step 1: Enable KV caching if specified
    # This will make the model store and reuse key-value pairs during generation
    if use_kv_cache:
        model.toggle_kv_cache(True)
    
    # Step 2: Initial forward pass
    # With KV caching: Process all initial tokens and cache their key-value pairs
    # Without KV caching: Process all tokens normally
    if use_kv_cache:
        with torch.no_grad():
            logits = model(inputs, start_pos=0)  # start_pos=0 for initial sequence
    else:
        with torch.no_grad():
            logits = model(inputs)
    
    # Step 3: Generate new tokens one by one
    for i in range(max_new_tokens):
        # Get the logits for the last token only
        # Shape: [batch_size, vocab_size]
        logits = logits[:, -1, :]
        
        # Step 4: Apply top-K sampling if specified
        # This helps in generating more diverse and interesting text
        if top_k is not None:
            # Get the top K logits and their values
            top_k_logits, _ = torch.topk(logits, top_k)
            # Get the minimum value among top K logits
            min_value = top_k_logits[:, -1]
            # Replace all logits below min_value with -inf
            # This effectively removes them from consideration
            logits = torch.where(
                condition=logits < min_value,
                input=torch.tensor(float('-inf')).to(device),
                other=logits
            )
        
        # Step 5: Apply temperature scaling
        # Higher temperature = more random, lower = more focused
        if temp_scale > 0.0:
            # Scale logits by temperature
            logits = logits/temp_scale
            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)
            # Sample from the probability distribution
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy sampling (take most likely token)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1)
        
        # Step 6: Check for end of sequence
        if next_token == eos_token:
            break
        
        # Step 7: Process the new token
        if use_kv_cache:
            # With KV caching: Only process the new token
            # The model will use cached key-value pairs for previous tokens
            with torch.no_grad():
                logits = model(next_token, start_pos=inputs.shape[1])
        else:
            # Without KV caching: Process the entire sequence again
            # Concatenate new token to input sequence
            inputs = torch.cat((inputs, next_token), dim=1)
            # Keep only the latest context to prevent sequence from growing too long
            inputs = inputs[:, -context_size:]
            with torch.no_grad():
                logits = model(inputs)
    
    # Step 8: Clean up - disable KV caching after generation
    if use_kv_cache:
        model.toggle_kv_cache(False)
    
    return inputs


# PRE-TRAINING THE MODEL
def train_model(model, train_loader, val_loader, optimizer, num_epochs, eval_freq, eval_iter, start_context, tokenizer, device):
    train_losses, val_losses, step = [], [], 0
    
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            
            step += 1
            
            if step % eval_freq == 0:
                train_loss, val_loss = evaluate(model, train_loader, val_loader, eval_iter, device)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} Step ({step:06d})"
                      f"Train Loss {train_loss}, Val Loss {val_loss}")
                
        generate_and_print_sample(model, start_context, tokenizer, device)
        
    return train_losses, val_losses

# Code to evaluate the model losses in evaluation model
def evaluate(model, train_loader, val_loader, eval_iter, device):
    model.eval()
    
    with torch.no_grad():
        train_loss = calc_loss_batches(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_batches(val_loader, model, device, num_batches=eval_iter) 
        
    model.train()
    return train_loss, val_loss

# Code to generate sample text after each epoch to check model progress
def generate_and_print_sample(model, start_context, tokenizer, device):
    encoded_text = text_to_token_ids(start_context, tokenizer).to(device)
    
    model.eval()
    with torch.no_grad():
        token_ids = generate(model, encoded_text, max_new_tokens=50, context_size=model.pos_emb.weight.shape[0], top_k=25, temp_scale=25)
    
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()
    
    
# SAMPLE USAGE
torch.manual_seed(123)

# Initialize and train the model
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 10

# Train the model
train_losses, val_losses = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context="Every effort moves you",
    tokenizer=tokenizer,
    device=device
)

# Save model for later use
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict()
}, "model_and_optimizer.pth")

# Load the saved model for inference
checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)

# Example: Generate text with KV caching
print("\nGenerating with KV caching:")
model.eval()
token_ids = generate(
    model=model,
    inputs=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temp_scale=1.4,
    use_kv_cache=True  # Enable KV caching for faster generation
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


# Function to measure generation time
def measure_generation_time(model, prompt, num_tokens, use_kv_cache):
    model.eval()
    inputs = text_to_token_ids(prompt, tokenizer).to(device)
    
    # Warm up run
    with torch.no_grad():
        _ = generate(
            model=model,
            inputs=inputs,
            max_new_tokens=5,
            context_size=GPT_CONFIG_124M["context_length"],
            top_k=25,
            temp_scale=1.4,
            use_kv_cache=use_kv_cache
        )
    
    # Measure time
    start_time = time.time()
    with torch.no_grad():
        _ = generate(
            model=model,
            inputs=inputs,
            max_new_tokens=num_tokens,
            context_size=GPT_CONFIG_124M["context_length"],
            top_k=25,
            temp_scale=1.4,
            use_kv_cache=use_kv_cache
        )
    end_time = time.time()
    
    return end_time - start_time

# Measure and plot inference times
# Test different sequence lengths
token_lengths = [10, 20, 50, 100, 200]
times_with_cache = []
times_without_cache = []

prompt = "Every effort moves you"
print("\nMeasuring inference times...")

for num_tokens in token_lengths:
    print(f"\nGenerating {num_tokens} tokens...")
    
    # Measure time without KV cache
    time_without_cache = measure_generation_time(model, prompt, num_tokens, use_kv_cache=False)
    times_without_cache.append(time_without_cache)
    print(f"Time without KV cache: {time_without_cache:.2f} seconds")
    
    # Measure time with KV cache
    time_with_cache = measure_generation_time(model, prompt, num_tokens, use_kv_cache=True)
    times_with_cache.append(time_with_cache)
    print(f"Time with KV cache: {time_with_cache:.2f} seconds")

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(token_lengths, times_without_cache, 'b-o', label='Without KV Cache')
plt.plot(token_lengths, times_with_cache, 'r-o', label='With KV Cache')
plt.xlabel('Number of Generated Tokens')
plt.ylabel('Generation Time (seconds)')
plt.title('Inference Time Comparison: With vs Without KV Caching')
plt.legend()
plt.grid(True)

# Add speedup ratio
speedup_ratios = [t1/t2 for t1, t2 in zip(times_without_cache, times_with_cache)]
plt.figure(figsize=(10, 6))
plt.plot(token_lengths, speedup_ratios, 'g-o')
plt.xlabel('Number of Generated Tokens')
plt.ylabel('Speedup Ratio (Without Cache / With Cache)')
plt.title('KV Caching Speedup Ratio')
plt.grid(True)

# Save the plots
plt.savefig('inference_time_comparison.png')
plt.savefig('speedup_ratio.png')
print("\nPlots have been saved as 'inference_time_comparison.png' and 'speedup_ratio.png'")