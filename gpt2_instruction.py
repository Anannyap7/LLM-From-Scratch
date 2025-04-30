import json
import os
import urllib.request
import random
import torch
import tiktoken
import time
import re
import psutil

from torch.utils.data import Dataset, DataLoader
from functools import partial
from gpt_download import download_and_load_gpt2
from gpt2_pretrained import load_weights_into_gpt
from gpt2_architecture import GPTModel
from gpt2_training import generate, text_to_token_ids, token_ids_to_text
from tqdm import tqdm


'''
INITIALIZE TOKENIZER
'''
tokenizer = tiktoken.get_encoding("gpt2")

'''
INITIALIZE THE DEVICE
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)
print()

'''
INITIALIZE THE MODEL CONFIGURATIONS
'''
BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
CHOOSE_MODEL = "gpt2-medium (355M)"
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
)
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval();


'''
LOAD DATA
'''
# Function to download and load json file which contain 1,100 instruction-reponse pairs
def download_and_load_file(file_path, url):
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    with open(file_path, "r") as file:
        data = json.load(file)
    return data
file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)

# Check whether data is loaded correctly
data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))
print("Example entry:\n", data[50])
print('-------------------------------------------------------------------')


'''
FORMAT DATA
'''
# Format the data entries into Alpaca-style input format
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    
    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    
    return instruction_text + input_text

# Check the functioning of data formatter
print("\nFORMATTED DATA ENTRY EXAMPLE:\n")
model_input = format_input(data[50])
desired_response = f"\n\n### Output:\n{data[50]['output']}"
print(model_input+desired_response)
print('-------------------------------------------------------------------')


'''
SPLIT DATA INTO TRAIN, VALIDATION AND TEST
'''
def random_split(data, train_split=0.85, val_split=0.05):
    random_seed = 42
    random.shuffle(data)
    
    train_portion = int(len(data) * train_split)
    val_portion = int(len(data) * val_split)
    test_portion = len(data) - (train_portion + val_portion)

    train_data = data[:train_portion]
    val_data = data[train_portion:train_portion+val_portion]
    test_data = data[train_portion+val_portion:]
    
    return train_data, val_data, test_data

# Check the data split
train_data, val_data, test_data = random_split(data)
print("\nTraining set length: ", len(train_data))
print("Validation set length: ", len(val_data))
print("Testing set length: ", len(test_data))
print('-------------------------------------------------------------------')


'''
CUSTOM DATASET CLASS
'''
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.encoded_texts = []
        
        for entry in data:
            instruction_and_input = format_input(entry)
            response = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_and_input + response
            self.encoded_texts.append(tokenizer.encode(full_text))
            
    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    def __len__(self):
        return len(self.data)
    
# Custom padding function which minimizes unnecessary padding by only extending sequences to match the longest one in each batch, not the whole dataset.
'''
Since we need to create both inputs and targets batch, we need to ensure that when the input
is shifted by 1 to create the target sequence, an eos token is appended to maintain the uniformity
in the length of the target token in the batch.

We replace all but the first instance of the end-of-text token, which we use as padding, with the
placeholder value -100, while keeping the initial end-of-text token in each target sequence.
This is to ensure that the padding tokens do not contribute to the training loss.
1 padding token is kept as it is to allow the model to learn when to generate an eos token in
response to instructions, which is an indicator that the geenrated response is complete.

Additionally, we introduce an allowed_ max_length parameter to optionally limit the length of the samples. 
This adjustment will be useful if you plan to work with your own datasets that exceed the 1,024-token 
context size supported by the GPT-2 model.
'''
def custom_padding(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device=device):
    max_seq_len = max([len(item)+1 for item in batch])
    inputs_lst, targets_lst = [], []
    
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id] # 1 extra token added to create the target sequence
        padded = new_item + [pad_token_id] * (max_seq_len - len(new_item))
        
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze() # extracts indices wherever value in tensor is non-zero
        if indices.numel() > 1: # .numel() counts the number of elements in a tensor
            targets[indices[1:]] = ignore_index
            
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
            
        inputs_lst.append(inputs)
        targets_lst.append(targets)
        
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

# The functools.partial in Python is a tool that allows the creation of new functions by pre-filling or 
# "freezing" some arguments of an existing function.
customized_padding = partial(custom_padding, device=device, allowed_max_length=1024)


'''
INITIALIZE THE DATALOADERS
'''
num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
val_dataset = InstructionDataset(val_data, tokenizer)
test_dataset = InstructionDataset(test_data, tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size = batch_size,
    collate_fn = customized_padding,
    shuffle = True,
    drop_last = True,
    num_workers = num_workers
)

val_loader = DataLoader(
    val_dataset,
    batch_size = batch_size,
    collate_fn = customized_padding,
    shuffle = False,
    drop_last = False,
    num_workers = num_workers
)

test_loader = DataLoader(
    test_dataset,
    batch_size = batch_size,
    collate_fn = customized_padding,
    shuffle = False,
    drop_last = False,
    num_workers = num_workers
)

# Examine the dimensions of input and target batches generated by the training loader.
print("Train Loader Input and Target Dimensions:\n")
for inputs, targets in train_loader:
    print(inputs.shape, targets.shape)
print("\nNumber of Batches in Training Set: ", len(train_loader))
print('-------------------------------------------------------------------')


'''
PERFORMANCE OF THE MODEL BEFORE FINE-TUNING
'''
torch.manual_seed(123)
input_text = format_input(val_data[0])
print("Input to the Model: \n", input_text)
token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)
response_text = generated_text[len(input_text):].strip()
print("\nModel's Response before Fine-Tuning: \n", response_text)
print('-------------------------------------------------------------------')


'''
LOSS CALCULATION
'''
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
        
    for i, (inputs, targets) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(inputs, targets, model, device)
            total_loss += loss.item()
        else:
            break
    
    return total_loss/num_batches


'''
TRAINING FUNCTION
'''
def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, global_step = [], [], 0
    
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            global_step += 1
            
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch} (Step {global_step:06d}): Train Loss {train_loss:.3f}, Val Loss {val_loss:.3f}")
                
        generate_and_print_sample(model, tokenizer, device, start_context)
        
    return train_losses, val_losses

def evaluate_model(model, train_loader, val_loader, device, eval_iter=5):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    return train_loss, val_loss

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
FINE-TUNING THE PRE-TRAINED LLM - ALL LAYERS
'''  
start_time = time.time()
torch.manual_seed(123)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.00005, weight_decay=0.1
)
num_epochs = 2
train_losses, val_losses = train_model(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")
print('-------------------------------------------------------------------')

# Save the fine-tuned model
file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")


'''
QUALITATIVE MODEL EVALUATION
'''
torch.manual_seed(123)
for entry in test_data[:3]:
    input_text = format_input(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    
    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("----------------------------------------------------------------------")
    

'''
SAVE ALL THE TEST SET RESPONSES FOR AUTOMATED MODEL EVALUATION BY COMPARING THE RESULTS WITH ANOTHER LLM
'''
for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    test_data[i]["model_response"] = response_text
with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)
    

'''
LOAD AND CHECK WHETHER THE OLLAMA MODEL IS RUNNING PROPERLY
'''
def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running
ollama_running = check_if_running("ollama")

if not ollama_running:
    raise RuntimeError("Ollama not running. Launch ollama before proceeding.")

print("Ollama running:", check_if_running("ollama"))
print("----------------------------------------------------------------------")


'''
USE THE API FOR OLLAMA FOR RESPONSE EVALUATION
This code sends a chat message (prompt) to an AI model (like LLaMA 3) running at 
http://localhost:11434/api/chat, and gets a text response back from it.
'''
def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    data = {
        "model": model, # what model to use
        "messages": [{"role": "user", "content": prompt}], # what messages the user sent
        # settings for deterministic output
        "options": {
            "seed": 123,
            "temperature": 0, # to reduce randomness in output
            "num_ctx": 2048 # it controls the maximum number of tokens the model can use to understand the input and generate a response.
        }
    }
    
    # Convert python dictionary to JSON string to send to the API
    payload = json.dumps(data).encode("utf-8")
    request = urllib.request.Request(url, data=payload, method="POST")
    # This adds a header to the HTTP request and tells the server that it is sending the data in JSON format.
    request.add_header("Content-Type", "application/json")
    
    # Opens the connection, sends the request and reads the response line-by-line
    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            # Decodes it from byte to text
            line = response.readline().decode("utf-8")
            if not line:
                break
            # Converts the line from json format to python dictionary
            response_json = json.loads(line)
            # Pulls the generated text from response and adds it to the string
            response_data += response_json["message"]["content"]
            
    return response_data
        

'''
EVALUATE THE MODEL'S RESPONSE ON 3 TEST EXAMPLES USING OLLAMA - SCORING
'''
for entry in test_data[:3]:
    prompt = (f"Given the input `{format_input(entry)}`"
              f"and the correct output `{entry['output']}`,"
              f"score the model response `{entry['model_response']}`"
              f"on a scale from 0 to 100 where 100 is the best score."
            )
    print("\nDataset response:")
    print(">>", entry['output'])
    print("\nModel response:")
    print(">>", entry["model_response"])
    print("\nScore:")
    print(">>", query_model(prompt))
    print("\n************************************")
    
print("----------------------------------------------------------------------")


'''
GET AN AVERAGE SCORE OF THE MODEL PERFORMANCE ON THE ENTIRE TEST SET
'''
def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue
    
    return scores

scores = generate_model_scores(test_data, "model_response")
print(f"Number of scores: {len(scores)} of {len(test_data)}")
print(f"Average score: {sum(scores)/len(scores):.2f}\n")