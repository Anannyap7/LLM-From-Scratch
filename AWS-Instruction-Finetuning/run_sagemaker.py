import sagemaker
from sagemaker.pytorch import PyTorch
from data_prep import upload_data
import os
import json

# Set up SageMaker session
sagemaker_session = sagemaker.Session()

# TODO: Replace with your SageMaker execution role ARN (Found in IAM Roles Page)
role = 'arn:aws:iam::182399726768:role/service-role/AmazonSageMaker-ExecutionRole-20250509T160359'

# S3 URIs for your data (update if your paths are different)
train_s3_path, val_s3_path, test_s3_path = upload_data('instructionbucketsagemaker', "pytorch_instruction_finetuning")

# Define the PyTorch estimator: This assigns a SageMaker training job to the AWS machine/instance
estimator = PyTorch(
    entry_point='train.py',                # Your training script
    source_dir='.',                        # Directory containing train.py
    role=role,                             # IAM role
    framework_version='1.12.0',            # version of PyTorch supported by SageMaker
    py_version='py38',                     # version of Python supported by SageMaker
    instance_count=1,                      # number of instances/machines to use for training (1 is enough)
    instance_type='ml.m5.2xlarge',           # Choose the type of machine you want to use for training
    hyperparameters={                      # Other hyperparameters are specified as default in train.py
        'train_file': 'train_data.json',
        'val_file': 'val_data.json'
    }
)

# Launch the training job
estimator.fit({
    'training': train_s3_path,
    'validation': val_s3_path
})

print('SageMaker training job launched! Check the SageMaker console for progress.')


# Deploy the model to a SageMaker endpoint
predictor = estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.c5.2xlarge',  # Or another suitable instance type
    endpoint_name='gpt2-instruction-endpoint'  # Optional: name your endpoint
) 

print("Model deployed! Endpoint name:", predictor.endpoint_name)

# ---- Inference on test data ----
# Load the first example from test_data.json
with open('data/test_data.json', 'r') as f:
    test_data = json.load(f)
example = test_data[0]

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

input_text = format_input(example)
payload = {"text": input_text}

predictor.content_type = "application/json"
result = predictor.predict(json.dumps(payload))

print("Input:")
print(input_text)
print("\nModel Output:")
print(result) 