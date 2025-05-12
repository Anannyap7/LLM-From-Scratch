import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import re
import sys
import traceback

from functools import partial
from architecture import GPTModel
from pretrained_model import load_weights_into_gpt

'''
SAGEMAKER TRAINING SCRIPT FOR GPT-2 INSTRUCTION FINE-TUNING
This script is intended to be used as the entry_point for a SageMaker PyTorch Estimator.
It loads data from the SageMaker training channel, fine-tunes the GPT-2 model, and saves the model to model_dir.
All relevant comments are preserved for clarity.
'''

def print_exception():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback)

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

def custom_padding(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    max_seq_len = max([len(item)+1 for item in batch])
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (max_seq_len - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, tokenizer):
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
    return train_losses, val_losses

def evaluate_model(model, train_loader, val_loader, device, eval_iter=5):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    return train_loss, val_loss

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

def model_fn(model_dir):
    # This is required for SageMaker inference, but not used in training
    pass

def main():
    '''
    When you launch a SageMaker training job, you must specify the S3 paths of your training, validation, and test data (these are the URIs printed by your modified data_prep.py script, e.g., s3://instructionbucketsagemaker/pytorch_instruction_finetuning/train_data.json).
    SageMaker will automatically download each file from S3 to the AWS training machine/instance and set environment variables to point to the local directories where the data is available.
    Your training script should read the data from these local directories on the AWS machine, not directly from S3.
    
    The SageMaker training job will automatically set the following environment variables:
    - SM_CHANNEL_TRAINING: The local path to the training data on AWS machine (downloaded from the S3 path you provided for training)
    - SM_CHANNEL_VALIDATION: The local path to the validation data on AWS machine (downloaded from the S3 path you provided for validation)
    - SM_MODEL_DIR: The local path where the model should be saved (SageMaker will upload this to S3 after training)
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--eval-freq', type=int, default=5)
    parser.add_argument('--eval-iter', type=int, default=5)
    parser.add_argument('--allowed-max-length', type=int, default=1024)
    parser.add_argument('--model-size', type=str, default="355M")
    # You need to instantiate the estimator such that the script runs on Sagemaker (AWS Machine) and does not return NoneType error when run locally.
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train_file', type=str, default="train_data.json")
    parser.add_argument('--val_file', type=str, default="val_data.json")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    # Load tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Model config
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }
    model_configs = {
        "124M": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "355M": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "774M": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "1558M": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    BASE_CONFIG.update(model_configs[args.model_size])

    # Load pre-trained weights (assumes weights are available in the container or downloaded)
    from gpt2_download import download_and_load_gpt2
    settings, params = download_and_load_gpt2(
        model_size=args.model_size,
        models_dir="gpt2"
    )
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.to(device)
    model.train()

    # Load data
    with open(os.path.join(args.train, args.train_file), "r") as f:
        train_data = json.load(f)
    with open(os.path.join(args.val, args.val_file), "r") as f:
        val_data = json.load(f)

    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)

    customized_padding = partial(custom_padding, device=device, allowed_max_length=args.allowed_max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=customized_padding,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=customized_padding,
        shuffle=False,
        drop_last=False
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train_losses, val_losses = train_model(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.epochs, eval_freq=args.eval_freq, eval_iter=args.eval_iter,
        tokenizer=tokenizer
    )

    # Save the fine-tuned model
    file_name = f"{re.sub(r'[ ()]', '', args.model_size)}-sft.pth"
    save_path = os.path.join(args.model_dir, file_name)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print_exception()
        sys.exit(1) 