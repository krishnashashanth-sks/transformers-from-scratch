import torch

# Helper function to calculate perplexity (optional, but good for language models)
def calculate_perplexity(loss):
    return torch.exp(loss)

def group_texts(examples,block_size):
    # Concatenate all texts from a batch
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # Drop the last incomplete batch to keep sequence length consistent
    total_length = (total_length // block_size) * block_size

    # Split by chunks of block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    # For language modeling, inputs are `input_ids` and labels are `input_ids` shifted by one
    result["labels"] = result["input_ids"].copy()
    return result