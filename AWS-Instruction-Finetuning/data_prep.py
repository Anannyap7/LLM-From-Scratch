import json
import os
import random
import urllib.request
import sagemaker
from sagemaker.s3 import S3Uploader
import boto3 # used for handling S3 buckets (data storage)

'''
DATA PREPARATION AND LOADING SCRIPT
This script handles downloading, formatting, splitting, and saving the instruction dataset for GPT-2 fine-tuning.
All relevant comments are preserved for clarity.
'''

# Function to download and load json file which contain 1,100 instruction-response pairs
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

data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))
print("Example entry:\n", data[50])
print('-------------------------------------------------------------------')

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

# Split data into train, validation, and test sets
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

train_data, val_data, test_data = random_split(data)
print("\nTraining set length: ", len(train_data))
print("Validation set length: ", len(val_data))
print("Testing set length: ", len(test_data))
print('-------------------------------------------------------------------')

# Save dataset locally as json files.
os.makedirs("data", exist_ok=True)
with open("data/train_data.json", "w") as file:
    json.dump(train_data, file, indent=2)
with open("data/val_data.json", "w") as file:
    json.dump(val_data, file, indent=2)
with open("data/test_data.json", "w") as file:
    json.dump(test_data, file, indent=2)

print("Data preparation complete. Files saved in ./data directory.")

# ---------------- SageMaker-specific code below ----------------
region = 'us-east-2'  # Specify the AWS region you want to use
boto_sess = boto3.Session(region_name=region)  # Create a new boto3 session with the specified region
sm_boto3 = boto_sess.client("sagemaker")  # Create a SageMaker client using the boto3 session (in the correct region)
sess = sagemaker.Session(boto_session=boto_sess)  # Create a SageMaker session object, also using the same boto3 session
bucket = 'instructionbucketsagemaker'  # Name of the S3 bucket you want to use for storing data
print("Using bucket: ", bucket)  # Print the bucket name to confirm which bucket is being used

# Upload the json files to the S3 bucket
def upload_data(bucket, data_prefix):
    train_path = sess.upload_data(path="data/train_data.json", bucket=bucket, key_prefix=data_prefix)
    val_path = sess.upload_data(path="data/val_data.json", bucket=bucket, key_prefix=data_prefix)
    test_path = sess.upload_data(path="data/test_data.json", bucket=bucket, key_prefix=data_prefix)
    print(f"Uploaded data to:\n{train_path}\n{val_path}\n{test_path}")
    return train_path, val_path, test_path

data_prefix = "pytorch_instruction_finetuning"
train_path, val_path, test_path = upload_data(bucket, data_prefix)