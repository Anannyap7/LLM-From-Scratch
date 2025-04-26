import json
import os
import urllib.request
import random
import torch
import tiktoken

from torch.utils.data import Dataset


'''
INITIALIZE TOKENIZER
'''
tokenizer = tiktoken.get_encoding("gpt2")


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
def custom_padding(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    max_seq_len = max([len(item)+1 for item in batch])
    inputs_lst, targets_lst = [], []
    
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id] # 1 extra token added to create the target sequence
        
        if len(item) < max_seq_len:
            padded = new_item + [pad_token_id] * (max_seq_len - len(item))
        
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squueze() # extracts indices wherever value in tensor is non-zero
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