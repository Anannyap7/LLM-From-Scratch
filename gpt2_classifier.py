import urllib.request
import zipfile
import os
from pathlib import Path
import pandas as pd
import tiktoken
import torch
import time

from torch.utils.data import Dataset, DataLoader
from gpt_download import download_and_load_gpt2
from gpt2_pretrained import load_weights_into_gpt
from gpt2_architecture import GPTModel, generate_text_simple
from gpt2_training import text_to_token_ids, token_ids_to_text


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Download the SMS Spam Collection dataset
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip" 
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

# Initialize model configuration
CHOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,
    "qkv_bias": True
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOSE_MODEL])

# Load and initialize the pretrained gpt2 model
model_size = CHOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval()

torch.manual_seed(123)

# Initialize variables
num_workers = 0 # no multiprocessing
batch_size = 8 # for dataloader
num_classes = 2 # spam and ham: to replace the last layer of the pre-trained model architecture for fine-tuning


'''
DATASET PREPARATION
'''
# Function to download and unzip the dataset
def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return

    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())
   
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)
    
    original_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(original_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")
    
download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)

# Load the dataset as pandas DataFrame
df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])

# Function to undersample the "ham/not spam" class in order to achieve a balanced dataset
print("Unbalanced label counts:\n", df["Label"].value_counts())

def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123
    )
    balanced_df = pd.concat([
        ham_subset, df[df["Label"] == "spam"]
    ])
    # Convert the labels to binary values
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    return balanced_df

balanced_df = create_balanced_dataset(df)
print("Balanced label counts:\n", balanced_df["Label"].value_counts())

# Splitting the dataset
def random_split(df, train_frac, validation_frac):
    # shuffle the dataset
    # frac=1 means 100% of the data and reset_index() resets the index after shuffling
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_index = int(len(df) * train_frac)
    val_index = train_index + int(len(df) * validation_frac)
    
    train_df = df[:train_index]
    val_df = df[train_index:val_index]
    test_df = df[val_index:]
    
    return train_df, val_df, test_df

train_df, val_df, test_df = random_split(balanced_df, 0.8, 0.1)

# Convert the DataFrames to CSV files for reusability
train_df.to_csv("train.csv", index=None)
val_df.to_csv("val.csv", index=None)
test_df.to_csv("test.csv", index=None)

# Define a custom dataset class which inherits from PyTorch's Dataset class and adds padding to the sequences smaller than the max length
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]
        
        if max_length == None:
            self.max_length = self._longest_encoded_length()
            
        else:
            self.max_length = max_length
            # truncate the sequences longer than the max length
            for encoded_text in self.encoded_texts:
                if len(encoded_text) > self.max_length:
                    encoded_text = encoded_text[:max_length]
                    
        # add padding to the sequences shorter than the max length
        for encoded_text in self.encoded_texts:
            encoded_text += [pad_token_id] * (self.max_length - len(encoded_text))
                
    def __getitem__(self, index):
        encoded_text = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded_text, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
        
    def __len__(self):
        return len(self.data)
    
    def _longest_encoded_length(self):
        return max(len(encoded_text) for encoded_text in self.encoded_texts)
    
train_dataset = SpamDataset("train.csv", tokenizer=tokenizer, max_length=None)
val_dataset = SpamDataset("val.csv", tokenizer=tokenizer, max_length=train_dataset.max_length)
test_dataset = SpamDataset("test.csv", tokenizer=tokenizer, max_length=train_dataset.max_length)

# Create DataLoaders for the datasets
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True # drop the last incomplete batch
)

# shuffle=False and drop_last=False for validation and test sets since we want to evaluate the model on all data
# and no training is involved, so the order of dataset does not matter.
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False # keep the last incomplete batch
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False # keep the last incomplete batch
)

# Check the working of the DataLoader
for input_batch, target_batch in train_loader:
    pass
print("Input batch shape: ", input_batch.shape)
print("Target batch shape: ", target_batch.shape)
print("Number of training batches: ", len(train_loader))
print("Number of validation batches: ", len(val_loader))
print("Number of test batches: ", len(test_loader))
print("------------------------------------------------------------------------------------")


'''
DEFINING FINE-TUNING LAYERS
'''
# Modify the last layer of the GPT model to fine-tune it for classification.
'''
Example: Input tensor shape: [1,4] - batch size 1 and 4 tokens.
Output tensor shape: [1,4,2] - batch size 1, 4 tokens and 2 classes.
'''
# For this we need to freeze all the other layers and modify the output layer to map from embedding dimension to the number of classes.
for params in model.parameters():
    params.requires_grad = False
# The requires_grad is True by default for the output layer: only layer to be trained.
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
# Fine-tuning additional layers can lead to a noticeable performance boost.
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True
    

'''
MODEL EVALUATION

Remember that we are interested in fine-tuning this model to return a class label indicating whether a model input is “spam” or “not spam.” 
We don’t need to finetune all four output rows; instead, we can focus on a single output token. 
In particular, we will focus on the last row corresponding to the last output token.
We convert the last row of the output tensor to a class label using a softmax function.
Since we need to find the highest probability score, we can directly use argmax instead of softmax.

WHY?
the last token in a sequence accu- mulates the most information since it is the only token with access to data from all the previous tokens. 
Therefore, in our spam classification task, we focus on this last token during the fine-tuning process.
'''

model.to(device)

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, total_predictions = 0, 0
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
        
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            with torch.no_grad(): # disables gradient tracking
                logits = model(input_batch)[:,-1,:] # logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)
            
            total_predictions += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    
    accuracy = correct_predictions / total_predictions
    return accuracy

# Accuracy before fine-tuning
train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)
print("ACCURACY BEFORE FINE-TUNING:\n")
print(f"Training Accuracy: {train_accuracy*100:.2f}%")
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print("------------------------------------------------------------------------------------")


'''
LOSS FUNCTION
'''
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    
    logits = model(input_batch)[:,-1,:] # logits of the last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    
    if len(data_loader) == 0:
        return float("nan")
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


'''
TRAINING LOOP FOR FINE-TUNING
'''
def train_classifier(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # resets losss gradients from the previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # calculates loss gradients
            optimizer.step() # updates model weights using loss gradients
            examples_seen += input_batch.shape[0]
            global_step += 1
            
            # Optional step to evaluate the model during training after a certain number of iterations
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )
        
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    
    return train_losses, val_losses, train_accs, val_accs, examples_seen

# Initialize the optimizer and training loop
start_time = time.time()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

train_losses, val_losses, train_accs, val_accs, examples_seen = \
    train_classifier(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=5, eval_freq=50,
        eval_iter=5
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")
print("------------------------------------------------------------------------------------")

# calculate the performance metrics for the training, validation, and test sets across the entire dataset
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)
print(f"Final Training accuracy: {train_accuracy*100:.2f}%")
print(f"Final Validation accuracy: {val_accuracy*100:.2f}%")
print(f"Final Test accuracy: {test_accuracy*100:.2f}%")
print("------------------------------------------------------------------------------------")


'''
MODEL INFERENCE
'''
def classify_review(text, model, tokenizer, device, max_length=None,pad_token_id=50256):
    model.eval()
    
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]
    
    input_ids = input_ids[:min(max_length, supported_context_length)] # truncate the input sequence to the max length
    
    input_ids += [pad_token_id] * (supported_context_length - len(input_ids)) # add padding to the input sequence
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0) # add batch dimension
    
    with torch.no_grad():
        logits = model(input_tensor)[:,-1,:]
    predicted_label = torch.argmax(logits, dim=-1).item()
    
    return "spam" if predicted_label == 1 else "not spam"

# Example usage - 1
text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)
print(classify_review(
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
))

# Example usage - 2
text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)
print(classify_review(
    text_2, model, tokenizer, device, max_length=train_dataset.max_length
))


'''
MODEL SAVING
'''
torch.save(model.state_dict(), "spam_classifier.pth")