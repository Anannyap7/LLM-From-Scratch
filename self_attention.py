'''
SELF-ATTENTION WITH TRAINABLE WEIGHTS
'''
import torch
import torch.nn as nn

'''
SELF-ATTENTION VERSION-1
'''
class SelfAttention_v1(nn.Module):
    # Initialize trainable weights for calculating Query, Key and Value
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
        self.W_key = nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
        self.W_value = nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
        
    def forward(self, x):
        # Here x is the input sequence. Calculate the queries, keys and values
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        
        # Calculate attention scores
        attn_scores = queries @ keys.T
        # Normalize attention scores to calculate attention weights
        # Divide by embedding dimension (d_out) of keys to avoid small gradients that softmax
        # would result in if the embedding dimension is greater that 1000.
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        # Calculate the context vector for each input element
        context_vec = attn_weights @ values
        return context_vec
    
# Testing the working of SelfAttention_v1 class
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

torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print("\nContext vectors (V1) of the input sequence:\n ", sa_v1(inputs))

'''
SELF-ATTENTION USING PYTORCH'S LINEAR LAYERS
    - The nn.Linear()is a fully connected (dense) layer in PyTorch. 
    - It applies a linear transformation to input data using a weight matrix and a bias vector.
    - Not only does it do matrix multiplication (Y = X @ W + b), it also initializes a weight 
    matrix of the shape specified in nn.Linear().
    - More sophisticated weight initialization compared to nn.Parameter().
'''
class SelfAttention_v2(nn.Module):
    # Initialize trainable weights for calculating Query, Key and Value
    def __init__(self, d_in, d_out, qkv_bias = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        
    def forward(self, x):
        # Calculate queries, keys and values for all input elements
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        # Calculate attention scores and scaled-dot product attention weights for all input elements
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        
        # Calculate context vector for all input elements
        context_vec = attn_weights @ values
        return context_vec
    
# Testing the working of SelfAttention_v2 class
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
context_vec = sa_v2(inputs)
print("\nContext vectors (V2) of the input sequence:\n ", context_vec)