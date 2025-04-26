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

'''
STRAIGHT-FORWARD MHSA
    - For example, if we use this MultiHeadAttentionWrapper class with two attention heads (via num_heads=2) and 
    CausalAttention output dimension d_out=2, we get a four-dimensional context vector (d_out*num_heads=4)
'''
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])
        
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1) # concatenate along column dimension

# Demonstrate the use of straight-forward mhsa
torch.manual_seed(123)
context_length = batch.shape[1] # number of tokens
d_in, d_out = 3, 2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)

'''
- The first dimension of the resulting context_vecs tensor is 2 since we have two input texts 
(the input texts are duplicated, which is why the context vectors are exactly the same for those). 
- The second dimension refers to the 6 tokens in each input. 
- The third dimension refers to the four-dimensional embedding of each token.
'''
print("SIMPLE MHA:")
print("Shape of context vector: ", context_vecs.shape)
print("Context Vector: ", context_vecs)
print('------------------------------------------------------------------------------')
'''
PARALLELIZABLE MHSA
    - Previous method processes the heads sequentially. There are separate weight matrices
    for keys, queries and values and requires more time and computational resources.
    - We want to process the heads in parallel and compute all the outputs simultaneously
    - Here, a larger weight matrix with the desired/final output dimension you want is 
    initialized and then the keys, queries and values are calculated once with this weight
    matrix. These are the split according to the number of heads mentioned in the code,
    hence giving rise to multiple queries, keys and values.
    - Example: final dimension you want is d_out=4 and number of heads=2, thus, dimension
    of each key, query and value would be 4/2 = 2. Thus, we get 2 keys, queries and values.
'''
class MultiHeadedAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0) # since head_dim = d_out/num_heads
        
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out) # linear layer to combine the head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        
        # tensor shape: b, num_tokens, d_out
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        
        # split the matrix by adding a num_heads dimension.
        # matrix is unrolled from (b, num_tokens, d_out) to (b, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # Transpose from shape (b,num_tokens, num_heads,head_dim) to 
        # (b,num_heads, num_tokens,head_dim) for matrix multiplication
        queries = queries.transpose(1,2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        
        attn_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Tensor shape: (b, num_tokens, n_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1,2)
        # combine the split dimensions/heads
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # optional linear projection
        context_vec = self.out_proj(context_vec)
        return context_vec
    
# Sample Usage
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out=2
mha = MultiHeadedAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print("PARALLELIZABLE MHA:")
print("Shape of context vector: ", context_vecs.shape)
print("Context Vector: ", context_vecs)