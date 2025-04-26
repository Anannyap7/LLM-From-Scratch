"""
The tokenizer for GPT models doesn't use an <|unk|> token for out-of-vocabulary words.
Instead, GPT models use a byte pair encoding tokenizer, which breaks words down into
subword units.
"""

# Implementing BPE is complicated, so we will use an existing open-source library called tiktoken
# pip install tiktoken

import tiktoken
import requests
import re
import torch
from importlib.metadata import version
from torch.utils.data import Dataset, DataLoader

# Check the version of tiktoken installed
print("tiktoken version: ", version("tiktoken"))

# Import "The Verdict" short story
url = ("https://raw.githubusercontent.com/rasbt/"
       "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
       "the-verdict.txt")

response = requests.get(url)
if response.status_code == 200:
    with open("the-verdict.txt", "wb") as file:
        file.write(response.content)
else:
    print("Failed to download file:", response.status_code)

# Instantiate BPE tokenizer from tiktoken: tokenization and conversion to token IDs in a single step
tokenizer = tiktoken.get_encoding("gpt2")

# Sample usage of BPE tokenizer
'''
    It breaks down the unknown word into subwords or individual characters which have tokens
    associated with them. This unknown word is then represented as a sequence of subword
    tokens or characters.
'''
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace"
)
encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
decoded = tokenizer.decode(encoded)
print("Sample Text: ", text)
print("Encoded Text: ", encoded)
print("Decoded Text: ", decoded)
print('------------------------------------------------------------------------------------')

'''
DATA SAMPLING USING SLIDING WINDOW  
'''

# Read text and encode it
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
encoded_text = tokenizer.encode(raw_text)
encoded_sample = encoded_text[50:] # for interesting text passage

# Create input-target pairs using x and y variables for input and targets
context_size = 4 # determines the no of tokens included in the input
x = encoded_sample[:context_size]
y = encoded_sample[1:context_size+1]
print(f"\nx: {x}")
print(f"y:      {y}")
# Illustrate input-target pairs created
for i in range(1, context_size+1):
    context = encoded_sample[:i]
    desired = encoded_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
print('------------------------------------------------------------------------------------')

'''
IMPLEMENTATION OF SLIDING WINDOW APPROACH USING DATA LOADER IN PYTORCH
'''

# Creating the dataset and defining how individual rows are fetched from the dataset
'''
    - max_length is for the context size of input text -> length of input_chunk list
    - stride: the number of positions the inputs shift across batches, emulating a sliding window approach
    - Example:  Input of batch-1 where batch_size=1 and stride=1 -> "In the heart of"
                Input of batch-2 where batch_size=1 and stride=1 -> "the heart of the"
'''
class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(text)
        
        for i in range(0, len(token_ids)-max_length, stride):
            input_chunk = token_ids[i : i+max_length]
            target_chunk = token_ids[i+1 : i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_ids)
    
    # returns a single row from the dataset
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# Function for loading the dataset created in batches via DataLoader which can be fed into the LLM model       
def create_dataloader_v1(text, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last, #  s
        num_workers=num_workers # number of CPU processeds to use for preprocessing
    )
    return dataloader

'''
SAMPLE USAGE TO CHECK THE WORKING OF GPTDatasetV1 AND create_dataloader_v1
'''
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# stride is 4 to avoid missing a single word and to avoid overlap between the batches which could lead to overfitting
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
# Converts dataloader into a Python iterator to fetch the next entry via next() function
data_iter = iter(dataloader)
inputs, targets = next(data_iter) # accessing the first batch which will contain 8 input lists of size 4 each
print("\nInputs:\n", inputs)
print("\nTargets:\n", targets)
print('------------------------------------------------------------------------------------')

'''
CONVERT TOKEN IDs INTO EMBEDDING VECTORS
    - Embedding is a mapping from discrete objects to points in a conmtinuous vector space.
    - To convert data into a format that neural networks can prcocess.
    - Intially, the embedding weights are initialized with random values. These are optimized 
    during LLM training as a part of LLM optimization.
    - Embedding layers perform a lookup operation, retrieving the embedding vector corresponding
    to the token ID from the embedding layer's weight matrix.
'''

'''
ADD POSITIONAL ENCODINGS
    - The embedding layer converts the token ID into the same vector representation regardles of
    where it is located in the input sequence.
    - Self-attention mechanism of LLMs do not have a notion of position or order for the tokens
    within a sequence. Thus, we need to add position-aware embeddings to the token emdedding vectors
    - Such that LLMs can understand relationship between tokens and make context-aware predictions.
    - Absolute Positional Embedding: Each position in input sequence -> unique embedding to
    embedding vector of token.
    - Relative Positional Embedding: Depends on distance between tokens. Model learns the 
    relationship in terms of "how far apart". Can generalize better to sequences of varying lengths.
'''

vocab_size = 50257
output_dim = 256 # size of each embedding vector
# create embedding layers
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# Embed each token in each batch to a 256-dimensional vector
# Batch-size of 8 with 4 tokens in each input sequence: 8 x 4 x 256 tensor
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
token_embeddings = token_embedding_layer(inputs)
print("\nShape after embedding input tokens: ", token_embeddings.shape) # [8,4,256]
print("Single-batch of input data after embedding: ", token_embeddings)
print('------------------------------------------------------------------------------------')

# Add absolute embeddings
# PyTorch will add the 4 × 256–dimensional pos_embeddings tensor to each 4 × 256–dimensional 
# token embedding tensor in each of the eight batches. Since input is given one-by-one, 
# the same 4 x 256 pos embedding can be used for each input sequence.
context_length = 4
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length)) 
print("\nShape of positional encodings: ", pos_embeddings.shape)
print("Positional Encodings: ", pos_embeddings)
print('------------------------------------------------------------------------------------')

# Final input to the LLM model
input_embeddings = token_embeddings + pos_embeddings
print("\nShape of the final input to the LLM model: ", input_embeddings.shape)
print("Single-batch of input data after positional encoding: ", input_embeddings)
print('------------------------------------------------------------------------------------')