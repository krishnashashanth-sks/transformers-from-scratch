import torch

def custom_collate_fn(batch):
    """Custom collate function to handle dictionary outputs from CustomDataset."""
    # Each item in 'batch' is a dictionary from __getitem__
    # We want to create a new dictionary where each key holds a batch of tensors
    collated_batch = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            # Stack tensors along a new dimension (batch dimension)
            collated_batch[key] = torch.stack([item[key] for item in batch])
        else:
            # For non-tensor items, e.g., if we had lists or other types, handle accordingly.
            # For now, we assume all relevant outputs are tensors.
            # If there were non-tensor items like lists of strings, we might just collect them:
            collated_batch[key] = [item[key] for item in batch]
    return collated_batch
