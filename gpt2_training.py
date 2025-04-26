import torch
import tiktoken
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from gpt2_architecture import GPTModel
from gpt2_architecture import generate_text_simple
from byte_pair_encoding import create_dataloader_v1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lower context length since it would save computational resources while training
GPT_CONFIG_124M = {
    "vocab_size": 50257,       # Vocabulary Size
    "context_length": 256,     # Context length (max number of input tokens model can handle)
    "emb_dim": 768,            # Embedding dimension
    "n_heads": 12,             # Number of attention heads
    "n_layers": 12,            # Number of layers/transformer blocks
    "drop_rate": 0.1,          # Dropour rate
    "qkv_bias": False          # Query-Key-Value bias while weight initialization
}

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Load text file for training and validation
file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()
    
total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Number of Characters: ", total_characters)
print("Number of Tokens: ", total_tokens)
print("-------------------------------------------------------------------------------------------------")

# Split dataset into train and validation
train_ratio = 0.9
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# Create batched data loader for train and test dataset
'''
- targets are inputs shifted by 1 position.
- max_length is for the context size of input text.
- stride: the number of positions the inputs shift across batches, emulating a sliding window approach.
- drop_last: drops the last batch when the number of examples in your dataset is not divisible by your batch_size.
'''
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    shuffle=True,
    drop_last=True,
    num_workers=0
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    shuffle=True,
    drop_last=True,
    num_workers=0
)

# Check for correct creation of train and validation loaders
print("Train Loader:")
for x, y in train_loader:
    print(x.shape, y.shape)
    
print("\nValidation Loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

print("-------------------------------------------------------------------------------------------------")


'''HELPER FUNCTIONS'''
# Function to convert input text into token IDs
def text_to_token_ids(text, tokenizer):
    # allows the tokenization of the endoftext word in the sentence.
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # to accommodate the batch dimension
    return encoded_tensor

# Function to convert output token ids to text
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # removes the batch dimension
    decoded = tokenizer.decode(flat.tolist())
    return decoded

# Function to compute the training and validation loss
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    # collapses the batch_size and num_tokens dimensions into a single dimension, producing a tensor of shape
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

def calc_loss_batches(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float('nan')
    # Iterates over all batches if no fixed num_batches is specified
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduces the number of batches to match the total number of batches in data_loader
        # if num_batches exceeds the number of batches in the data_loader
        num_batches = min(num_batches, len(data_loader))
    # Average loss over all batches
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss/num_batches

# Modified text generation function for more diversity
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # Gets logits of the most recent context
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond) # shape: [batch_size, context_length, vocab_size]
        logits = logits[:,-1,:] # shape: [batch_size, vocab_size]
    
        # Filters logits with top-k sampling
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1] # shape: [batch_size, number of top logits]
            # we apply PyTorch’s where function to set the logit values of tokens that are below the lowest 
            # logit value within our top-k selection to negative infinity (-inf)
            logits = torch.where(
                condition = logits < min_val,
                input = torch.tensor(float('-inf')).to(logits.device), # Assigns –inf to these lower logits
                other = logits # Retains the original logits for all other tokens
            )
        # Applies temperature scaling for varied next token selection
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            # num_samples: number of tokens to select/draw for next word generation in a single iteration
            idx_next = torch.multinomial(probs, num_samples=1)
        # Greedy sampling applied if no temperature scaling is done
        else:
            # dim=0: across rows (i.e., column-wise max) | dim=1: across columns (i.e., row-wise max)
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        # Stops generating early if end-of-sequence token is encountered
        if idx_next == eos_id:
            break
        
        # Concatenate the next token with the current context for next token generation
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

# Plot the train and validation losses
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()


'''
MAIN FUNCTIONS FOR PRE-TRAINING LLMs
'''
# Training loop
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    # tracks the number of tokens processed and tracks the number of gradient updates
    tokens_seen, global_step = 0, -1
    
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # resets loss gradients from the previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # calculates loss gradients
            optimizer.step() # updates model weights using loss gradients
            tokens_seen += input_batch.numel() # number of tokens in the input batch
            global_step += 1
            
            '''
            - We recalculate the training loss in evaluation model since the evaluation mode
            does not consider dropout and batch/layer norm and we need to calculate the validation loss
            in the eval model. This ensures that the train and val loss are calculated in the 
            same conditions for model evaluation.
            - These losses can be used to analyse the hyperparameter tuning needed for the model
            to perform well.
            '''
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )
        generate_and_print_sample(model, tokenizer, device, start_context)
    
    return train_losses, val_losses, track_tokens_seen

# Model evaluation for hyperparameter tuning
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_batches(train_loader, model, device, num_batches = eval_iter)
        val_loss = calc_loss_batches(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss     

# To track whether the model improves during the training or not
def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded_text = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(model, idx=encoded_text, max_new_tokens=50, context_size=context_size, top_k=25,
    temperature=1.4)
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


'''
SAMPLE USAGE
'''

'''
# Training
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs=10
train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader, val_loader, 
                                                           optimizer, device, num_epochs=num_epochs,
                                                           eval_freq=5, eval_iter=5,
                                                           start_context="Every effort moves you",
                                                           tokenizer=tokenizer)

# Save model for later use or pre-training
torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, 
           "model_and_optimizer.pth")
'''

# To use the saved model
'''
checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train()
'''

'''
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# Inference
model.to(device)
model.eval()
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
'''

'''
# Example to illustrate how the dimensions work when encoding and decoding.
example1 = torch.tensor([[3626, 6100, 345],[1107, 588, 11311]]) # 2 input examples already mapped to token ids and batched
example2 = torch.tensor([[[16657],[ 339],[42826]],[[49906],[29669],[41751]]])
print(example1[0]) # tensor([3626, 6100,  345])
print(example2[0].flatten()) # tensor([16657,   339, 42826])
print(example2[0].flatten().squeeze(0)) # tensor([16657,   339, 42826])
print(example2[0].flatten().squeeze(0).tolist()) # [16657, 339, 42826]
print(example2.flatten().squeeze(0).tolist()) # [16657, 339, 42826, 49906, 29669, 41751]
'''