import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_emb_dim, output_emb_dim, context_length, dropout, 
                 num_heads, max_batch_size, max_seq_len, qkv_bias=False):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.input_emb_dim = input_emb_dim
        self.output_emb_dim = output_emb_dim
        self.num_heads = num_heads
        self.head_dim = output_emb_dim // num_heads
        self.context_length = context_length
        
        # Initialize variables for KV caching
        self.max_batch_size = max_batch_size # max batch size that cache can store
        self.max_seq_len = max_seq_len # max context/seq length to be stored in cache
        self.kv_caching_enabled = False
        
        # Initialize the query, key and value weights
        self.W_query = nn.Linear(input_emb_dim, output_emb_dim, bias = qkv_bias)
        self.W_key = nn.Linear(input_emb_dim, output_emb_dim, bias = qkv_bias)
        self.W_value = nn.Linear(input_emb_dim, output_emb_dim, bias = qkv_bias)
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        
        # Additional linear projection cause it is heavily used in most LLM architectures
        self.proj_output = nn.Linear(output_emb_dim, output_emb_dim)
        
        # Initialize register buffers for causal masking and caching keys and values
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.register_buffer('cache_k', None)
        self.register_buffer('cache_v', None)
        
    def forward(self, inputs: torch.Tensor, start_pos: int = None):
        
        if self.kv_caching_enabled and not self.training:
            assert start_pos is not None, "Must provide start_pos argument if using KV caching"
            
        batch_size, num_tokens, input_emb_dim = inputs.shape # (B, N, in_dim)
        
        # Calculate the query, keys and values
        Q = self.W_query(inputs) # from (B, N, in_dim) to (B, N, out_dim); N=1 if KV caching
        K = self.W_key(inputs)
        V = self.W_value(inputs)
        
        # Splitting output_emb_dim into num_heads
        Q = Q.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        K = K.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        V = V.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        
        # Use KV caching if enabled
        if self.kv_caching_enabled and not self.training:
            assert self.max_batch_size >= batch_size, "Length of max_batch_size for cache should always be greater than batch_size"
            
            if self.cache_k is None or self.cache_v is None:
                self.cache_k = torch.zeros(self.max_batch_size, self.max_seq_len, self.num_heads, self.head_dim, device = inputs.device)
                self.cache_v = torch.zeros_like(self.cache_k, device = inputs.device)
            
            # Caching will only be done for the new tokens (num_tokens=1) after the initial caching.
            self.cache_k[:batch_size, start_pos : start_pos + num_tokens, :, :] = K
            self.cache_v[:batch_size, start_pos : start_pos + num_tokens, :, :] = V
            
            # Extract cached keys and values for computation
            K = self.cache_k[:batch_size, :start_pos + num_tokens]
            V = self.cache_v[:batch_size, :start_pos + num_tokens]
            
        # Reshape Q, K, V for further calculation from [B, N, num_heads, head_dim] 
        # to [B, num_heads, N, head_dim] since we want to parallely compute attention for each head.
        # So now, each head will have the shape [num_tokens, head_dim].
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)
        
        # Reshape K to calculate attention scores
        attention_scores = Q @ K.transpose(2,3) # Shape: (B, num_heads, N, N)
        # Causal Masking
        attention_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        
        # K.shape[-1] = head_dim and dim=-1 since softmax along column of QK matrix (NxN)
        attention_weights = torch.softmax(attention_scores/K.shape[-1]**0.5, dim=-1)
        # Dropout on attention weights
        attention_weights = self.dropout(attention_weights) # Shape: (B, num_heads, N, N)
        
        # Calculate the Context vector
        Z = attention_weights @ V # Shape: (B, num_heads, N, head_dim)
        Z = Z.transpose(1,2) # Shape: (B, num_heads, head_dim, N)
        Z = Z.reshape(batch_size, num_tokens, self.output_emb_dim) # Shape: (B, N, out_dim)
        context_vec = self.proj_output(Z) # Linear Projection. No shape change.
        
        return context_vec # Shape: (B, N, out_dim)
        
if __name__ == '__main__':
    
    # Initialize variables
    context_length = 1024 # num_tokens (length to feed into the model)
    emb_dim = 768
    max_batch_size = 5
    max_seq_len = 1024
    num_heads = 12
    dropout = 0.3
    
    # Create batched input tensor
    input_tensor = torch.randn(2, context_length, emb_dim)
    
    # Extract dimensions
    batch_size, num_tokens, input_dim = input_tensor.shape
    
    # Run MHSA
    mhsa = MultiHeadSelfAttention(input_dim, emb_dim, context_length, dropout, num_heads, max_batch_size, max_seq_len)
    context_vec = mhsa(input_tensor)
    print("Context Vector without KV Caching: ", context_vec)