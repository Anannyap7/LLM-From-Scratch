import torch
import torch.nn as nn
from mhsa import MultiHeadSelfAttention

GPT_CONFIG_124M = {
    "vocab_size": 50257,       # Vocabulary Size
    "context_length": 1024,    # Context length (max number of input tokens model can handle)
    "emb_dim": 768,            # Embedding dimension
    "num_heads": 12,             # Number of attention heads
    "num_layers": 12,            # Number of layers/transformer blocks
    "drop_rate": 0.1,          # Dropour rate
    "qkv_bias": False,         # Query-Key-Value bias while weight initialization
    "max_batch_size": 5,       # max batch size for cached keys and values matrix
    "max_seq_len": 1024        # max seg length for cached keys and values matrix 
}

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(cfg["emb_dim"]*4, cfg["emb_dim"])
        )
    
    def forward(self, x):
        output = self.layers(x)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.use_kv_cache = False
        self.mhsa = MultiHeadSelfAttention(cfg["emb_dim"], cfg["emb_dim"], 
                                           cfg["context_length"], cfg["drop_rate"],
                                           cfg["num_heads"], cfg["max_batch_size"], 
                                           cfg["max_seq_len"], cfg["qkv_bias"])
        self.dropout = nn.Dropout(cfg["drop_rate"]) # for generalizability
        self.ff = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg["emb_dim"], eps=1e-5, elementwise_affine=True)
        self.ln2 = nn.LayerNorm(cfg["emb_dim"], eps=1e-5, elementwise_affine=True)
    
    def forward(self, x, start_pos: int = None):
        # sets the variable in MultiHeadSelfAttention class
        self.mhsa.kv_caching_enabled = self.use_kv_cache and not self.training
        if self.mhsa.kv_caching_enabled:
            assert start_pos is not None, "Must provide start_pos during inference for using KV caching"
        
        skip = x
        x = self.ln1(x)
        x = self.mhsa(x, start_pos)
        x = self.dropout(x)
        x += skip # skip/residual connection
        
        skip = x
        x = self.ln2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x += skip # skip/residual connection
        
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.use_kv_cache = False
        self.mhsa = MultiHeadSelfAttention(cfg["emb_dim"], cfg["emb_dim"], 
                                           cfg["context_length"], cfg["drop_rate"],
                                           cfg["num_heads"], cfg["max_batch_size"], 
                                           cfg["max_seq_len"], cfg["qkv_bias"])
        
        # Token and Positional Embeddings
        '''
        nn.Embedding: 
        -> One-hot encoding of token ID into the dimension of the 1st param.
        Eg: token ID at 1st index = [1,0,0,...]
        -> nn.Linear from one-hot encoded vector into embedding size for each token.
        '''
        self.token_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        
        # Dropout layer for generalizability
        self.dropout = nn.Dropout(cfg["drop_rate"])
        
        # Multiple Transformer Blocks
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["num_layers"])])
        
        # Final Layer Norm
        self.final_ln = nn.LayerNorm(cfg["emb_dim"], eps=1e-5, elementwise_affine=True)
        
        # Output prediction head
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"])
        
    def forward(self, x, start_pos: int = None):
        # sets the variable for MHSA
        self.mhsa.kv_caching_enabled = self.use_kv_cache and not self.training
        if self.mhsa.kv_caching_enabled:
            assert start_pos is not None, "Must provide start_pos during inference for using KV caching"
        
        batch_size, num_tokens = x.shape
        token_embeddings = self.token_emb(x) # (B, N, emb_dim)
        
        # No arange when using KV caching since it would result in 0 for num_tokens=1.
        # This would overlap with the pos_emb for the token at the 1st index.
        if self.use_kv_cache and num_tokens == 1: # After the 1st pass
            pos_embeddings = self.pos_emb(torch.tensor([start_pos], device = x.device)) # (B, 1, emb_dim)
        else:
            pos_embeddings = self.pos_emb(torch.arange(num_tokens, device = x.device)) # (B, N, emb_dim)
        
        input_embeddings = token_embeddings + pos_embeddings # (B, N, emb_dim)
        
        # Dropout
        input_embeddings = self.dropout(input_embeddings) # (B, N, emb_dim)
        
        # Transformer Blocks
        for block in self.trf_blocks:
            input_embeddings = block(input_embeddings, start_pos) # (B, N, emb_dim/out_dim)
        
        # Final Layer Norm
        input_embeddings = self.final_ln(input_embeddings) # (B, N, emb_dim/out_dim)
        
        # Prediction Head
        logits = self.out_head(input_embeddings) # (B, N, vocab_size)
        
        return logits
    
    def toggle_kv_cache(self, use_kv_cache: bool):
        '''
        For dynamically enabling/disabling KV caching in all transformer blocks
        '''
        # sets the variable for GPTModel class
        self.use_kv_cache = use_kv_cache
        for block in self.trf_blocks:
            # sets the varaible for TransformerBlock class
            block.use_kv_cache = use_kv_cache