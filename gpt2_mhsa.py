import torch
import torch.nn as nn

context_length = 1024   # Sequence length
embedding_size = 768    # Input embedding dimension

# Create a random tensor with shape (context_length, embedding_size)
input_tensor = torch.randn(context_length, embedding_size)
batch = torch.stack((input_tensor, input_tensor), dim=0)
print("Input Tensor Shape: ", input_tensor.shape)
print("Batch Shape: ", batch.shape)
print("Batch Input:\n", batch)

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
'''
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out=768
mha = MultiHeadedAttention(d_in, d_out, context_length, 0.0, num_heads=12)
context_vecs = mha(batch)
print('\n------------------------------------------------------------------------------')
print("PARALLELIZABLE MHA:")
print("Shape of context vector: ", context_vecs.shape)
print("Context Vector:\n", context_vecs)
''' 