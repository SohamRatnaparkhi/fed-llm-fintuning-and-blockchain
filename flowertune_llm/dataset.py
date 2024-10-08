from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from datasets import load_dataset

FDS = None  # Cache FederatedDataset


def formatting_prompts_func(example):
    output_texts = []
    # Constructing a standard Alpaca (https://github.com/tatsu-lab/stanford_alpaca#data-release) prompt
    mssg = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    for i in range(len(example["instruction"])):
        res = ""
        if i < len(example['response']):
            res = example['response'][i]
        else:
            break
        text = f"{mssg}\n### Instruction:\n{example['instruction'][i]}\n### Response: {res}"
        output_texts.append(text)
    return output_texts


def get_tokenizer_and_data_collator_and_propt_formatting(tokenizer):
    # From: https://huggingface.co/docs/trl/en/sft_trainer
    # model_path = '/home/sohamr/projects/def-ssamet-ab/sohamr/models/models--openai-community--gpt2'
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_name, use_fast=True, padding_side="right"
    # )
    tokenizer.pad_token = tokenizer.eos_token
    response_template_with_context = "\n### Response:"  # alpaca response tag
    response_template_ids = tokenizer.encode(
        response_template_with_context, add_special_tokens=False
    )[2:]
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, tokenizer=tokenizer
    )

    return tokenizer, data_collator, formatting_prompts_func


def load_data(partition_id: int, num_partitions: int, dataset_name: str):
    """Load partition data."""
    # Only initialize `FederatedDataset` once
    global FDS
    if FDS is None:
        # local_dataset_path = "/home/sohamr/projects/def-ssamet-ab/sohamr/datasets/pankajrajdeo___drug_bank"
        
        # Load the local dataset
        local_dataset = load_dataset(dataset_name)
        
        # Create the partitioner
        partitioner = IidPartitioner(num_partitions=num_partitions)
        
        # Create FederatedDataset with the local dataset
        FDS = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )
        # partitioner.dataset = local_dataset
    client_trainset = FDS.load_partition(partition_id, "train")
    # client_trainset = make_alpaca_gpt4(client_trainset)
    # client_trainset = make_medical_qa(client_trainset)
    # client_trainset = make_dataset_drug_bank(client_trainset)
    client_trainset = make_databricks(client_trainset)
    return client_trainset


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict


def make_dataset_drug_bank(client_trainset):
    # Step 1: Limit the dataset to the first 1000 rows
    client_trainset = client_trainset.select(
        range(min(1000, len(client_trainset))))

    # Step 2: Rename the 'Name' column to 'instruction'
    if "Name" in client_trainset.column_names:
        client_trainset = client_trainset.rename_column("Name", "instruction")

    # Step 3: List of columns to be concatenated
    columns_to_concat = ["Description", "Indication",
                         "Pharmacodynamics", "Mechanism of Action", "Toxicity"]

    # Step 4: Replace null values with empty strings in the specified columns
    for column in columns_to_concat:
        if column in client_trainset.column_names:
            client_trainset = client_trainset.map(
                lambda x: {column: "" if x[column] is None else str(x[column])})

    # Step 5: Combine the specified columns into a new column called 'response'
    def combine_columns(example):
        response = ""
        for column in columns_to_concat:
            if column in example:
                response += str(example[column])
        return {"response": response}

    client_trainset = client_trainset.map(combine_columns)

    # Optional: Remove the original columns if they are no longer needed
    columns_to_remove = set(columns_to_concat) & set(
        client_trainset.column_names)
    client_trainset = client_trainset.remove_columns(list(columns_to_remove))

    return client_trainset

def make_medical_qa(client_trainset):
    client_trainset = client_trainset.rename_column("Answer", "response")
    client_trainset = client_trainset.rename_column(
        "Question", "instruction")

    return client_trainset

def make_alpaca_gpt4(client_trainset):
    client_trainset = client_trainset.rename_column("output", "response")
    return client_trainset

def make_databricks(client_trainset):
    # Replace null values in context column with empty string
    client_trainset = client_trainset.map(lambda x: {"context": "" if x["context"] is None else x["context"]})
    
    # Combine context and instruction columns
    def combine_context_instruction(example):
        combined = f"{example['context']} {example['instruction']}".strip()
        return {"instruction": combined}
    
    client_trainset = client_trainset.map(combine_context_instruction)
    
    # Remove the original context column
    if "context" in client_trainset.column_names:
        client_trainset = client_trainset.remove_columns(["context"])

    return client_trainset