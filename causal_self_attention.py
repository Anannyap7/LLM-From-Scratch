'''
CAUSAL SELF-ATTENTION WITH TRAINABLE WEIGHTS
    - Future token embeddings are masked out and only the previous
    and current embeddings are send to the model for training
    with the purpose of predicting the next token.
'''
import torch
import torch.nn as nn
    
inputs = torch.tensor(
   [[0.43, 0.15, 0.89], # Your     (x^1)
    [0.55, 0.87, 0.66], # journey  (x^2) -> query token
    [0.57, 0.85, 0.64], # starts   (x^3)
    [0.22, 0.58, 0.33], # with     (x^4)
    [0.77, 0.25, 0.10], # one      (x^5)
    [0.05, 0.80, 0.55]] # step     (x^6)
)
print("Embedded input sequence:\n ", inputs)
d_in = inputs.shape[-1]
d_out = 2

# Create a batch
batch = torch.stack((inputs, inputs), dim=0) # Two input sequences with 6 tokens each and embedding size 3 each token
print("\nBatch Shape: ", batch.shape)

'''
CAUSAL SELF-ATTENTION CLASS
    - Create a mask to fill the future tokens (above the diagonal of the matrix)
    with -inf before calculating the attention weights using softmax function.
    - e^(-inf) = 0 when using softmax.
    - Use dropout to avoid overfitting while training. Dropout in deep learning 
    is a technique where randomly selected hidden layer units are ignored during 
    training, effectively “dropping” them out. This method helps prevent 
    overfitting by ensuring that a model does not become overly reliant on any 
    specific set of hidden layer units.
    - Dropout in the attention mechanism is typically applied at two specific times: 
        - after calculating the attention weights or 
        - after applying the attention weights to the value vectors.
    - When applying dropout to an attention weight matrix with a rate of 50%, 
    half of the elements in the matrix are randomly set to zero. To compensate 
    for the reduction in active elements, the values of the remaining elements 
    in the matrix are scaled up by a factor of 1/0.5 = 2.
    - This scaling is crucial to maintain the overall balance of the attention
    weights, ensuring that the average influence of the attention mechanism remains 
    consistent during both the training and inference phases.
'''
class CausalAttention(nn.Module):
    # Initialize trainable weights for calculating Query, Key and Value
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias = False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        # Calculate queries, keys and values for all input elements
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        print("\nKeys Shape: ", keys.shape)
        
        # Calculate attention scores and scaled-dot product attention weights for all input elements
        # Transpose dimensions 1 and 2 while keeping the batch dimension same.
        attn_scores = queries @ keys.transpose(1,2)
        
        # Replaces all True positions in the upper triangular matrix (1s above diagonal and 0s below) created with -inf.
        # In PyTorch, operations with trailing _ are performed in place.
        # Prevents applying unnecessary masking when context_length (max seq len) is larger than num_tokens (actual input).
        masked_scores = attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = self.dropout(torch.softmax(masked_scores/keys.shape[-1]**0.5, dim=-1))
        
        # Calculate context vector for all input elements
        context_vec = attn_weights @ values
        return context_vec
    
# Testing the working of SelfAttention_v2 class
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.2)
context_vec = ca(batch)
print("\nContext Vectors shape: ", context_vec.shape)
print("\nContext vectors of entire batch:\n ", context_vec)