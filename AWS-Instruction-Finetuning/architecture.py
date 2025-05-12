import torch
import torch.nn as nn
import tiktoken
from mhsa import MultiHeadedAttention

GPT_CONFIG_124M = {
    "vocab_size": 50257,       # Vocabulary Size
    "context_length": 1024,    # Context length (max number of input tokens model can handle)
    "emb_dim": 768,            # Embedding dimension
    "n_heads": 12,             # Number of attention heads
    "n_layers": 12,            # Number of layers/transformer blocks
    "drop_rate": 0.1,          # Dropour rate
    "qkv_bias": False          # Query-Key-Value bias while weight initialization
}

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # nn.Embedding parameters: num_embeddings, embedding_dim
        # Converts token IDs into dense embeddings: (batch_size, seq_length) → (batch_size, seq_length, emb_dim)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # Converts token indices into positional embeddings
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # Placeholder for transformer blocks
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        # Layer Normalization
        self.final_norm = LayerNorm(cfg["emb_dim"])
        '''
        - Converts the token embeddings into logits: (batch_size,context_len,emd_dim) -> 
        (batch_size,context_len,vocab_size)
        - Each 768-dimensional token vector is projected into 50,257 dimensions using a weight 
        matrix of shape (768, 50000).
        - Each number in this 50257 long tensor for each token represents the score for a word 
        in the vocab.
        - When softmax function is applied to these scores, they get converted into probabilities.
        - The highest probability for a particular token in this 50257 tensor, corresponds to the
        next predicted token.
        '''
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        
    def forward(self, in_idx):
        # in_idx are the token IDs for each token obtained using tokenizer
        batch_size, seq_len = in_idx.shape
        token_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = token_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadedAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
        
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x += shortcut # skip connection
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x += shortcut # skip connection
        
        return x
    
# Layer Normalization
'''
Using keepdim=True in operations like mean or variance calculation ensures that the output tensor 
retains the same number of dimensions as the input tensor, even though the operation reduces 
the tensor along the dimension specified via dim. For instance, without keepdim=True, 
the returned mean tensor would be a two-dimensional vector [0.1324, 0.2170] 
instead of a 2 × 1–dimensional matrix [[0.1324], [0.2170]].
'''
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        # initialize scale and shift as learnable parameters
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x):
        # normalize along the layer dimension, dim=-1
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        norm_x = self.scale * norm_x + self.shift
        return norm_x
    
# GELU Activation Function
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        x = 0.5 * x * (1 + torch.tanh(
                   torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                    (x + 0.044715 * torch.pow(x, 3))
                ))
        return x
    
# Feed Forward Neural Network Module
'''
Shape of the output tensor should be same as the shape of the input tensor.
'''
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            )
    
    def forward(self, x):
        return self.layers(x)

# Example deep neural network to demonstrate residual/skip connections
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
        ])
        
    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x += layer_output
            else:
                x = layer_output
        return x

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # Crop the input tokens to the specified context size of the LLM.
        # Only the last x input tokens are considered since they would provide a better
        # context for next word generation.
        idx_crop = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_crop)
        
        logits = logits[:, -1, :] # take only the last input token to generate the next word
        probs = torch.softmax(logits, dim=-1)
        id_next = torch.argmax(probs, dim=-1, keepdim=True)
        # Appends sampled index to the running sequence, where idx has shape (batch, n_tokens+1)
        idx = torch.cat((idx, id_next), dim=1)
        
    return idx

# Initialize the input and tokenizer
'''
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
text1 = "Every effort moves you"
text2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(text1)))
batch.append(torch.tensor(tokenizer.encode(text2)))
batch = torch.stack(batch, dim=0)
print("Input batched token IDs:\n", batch)

# Initialize the DummyGPTModel
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("\nOutput Shape: ", logits.shape)
print("Output Logits:\n", logits)
print("-------------------------------------------------------------------------------------------------")

# Sample Inference on untrained dummy gpt2 model
start_context = "Hello I am"
print("Context: ", start_context)
encoded = tokenizer.encode(start_context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("Encoded Context: ", encoded_tensor)

model.eval()
out = generate_text_simple(model=model, idx=encoded_tensor, 
                           max_new_tokens=6, context_size=GPT_CONFIG_124M["context_length"])
print("Output Tensor: ", out)

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print("Generated Context: ", decoded_text)

'''