import requests
import os

# 1. Define the directory path and file path
data_dir = '/content/data'
file_path = os.path.join(data_dir, 'tinyshakespeare.txt')

# 2. Create the directory if it doesn't exist
os.makedirs(data_dir, exist_ok=True)
print(f"Ensured directory '{data_dir}' exists.")

# 3. Download the 'tinyshakespeare.txt' dataset
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
response = requests.get(url)

if response.status_code == 200:
    with open(file_path, 'wb') as f:
        f.write(response.content)
    print(f"'tinyshakespeare.txt' downloaded and saved to '{file_path}'.")
else:
    print(f"Failed to download file. Status code: {response.status_code}")

# Read the content of the downloaded text file into a string variable
with open(file_path, 'r', encoding='utf-8') as f:
    text_data = f.read()

print(f"Successfully read {len(text_data)} characters from '{file_path}'.")
# Display the first 500 characters to verify
print("\nFirst 500 characters of the dataset:")
print(text_data[:500])

#Enchance Tokenization and Data Loading 
import torch
from torch.utils.data import Dataset, DataLoader

# 1. Extract all unique characters from the text_data
chars = sorted(list(set(text_data)))
vocab_size_char = len(chars)
print(f"New character-level vocabulary size: {vocab_size_char}")

# 2. Map each unique character to a numerical index and vice-versa
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

print("Char to index mapping example:", {k: char_to_idx[k] for k in list(char_to_idx)[:5]})
print("Index to char mapping example:", {k: idx_to_char[k] for k in list(idx_to_char)[:5]})

# 3. Convert the entire text_data into a sequence of numerical indices
encoded_text = torch.tensor([char_to_idx[char] for char in text_data], dtype=torch.long)

print(f"Original text data length: {len(text_data)} characters")
print(f"Encoded text tensor shape: {encoded_text.shape}")
print("First 10 numericalized characters:", encoded_text[:10].tolist())
print("Last 10 numericalized characters:", encoded_text[-10:].tolist())

def get_sequence_and_target(data_tensor, block_size, idx):
    # Input sequence: characters from index idx to idx + block_size - 1
    input_sequence = data_tensor[idx : idx + block_size]
    # Target sequence: characters from index idx + 1 to idx + block_size
    target_sequence = data_tensor[idx + 1 : idx + block_size + 1]
    return input_sequence, target_sequence

# Example usage with max_seq_len
# Ensure max_seq_len is defined, which it is from previous cells (max_seq_len = 10)

example_idx = 0
input_ex, target_ex = get_sequence_and_target(encoded_text, max_seq_len, example_idx)

print(f"Input sequence (indices): {input_ex.tolist()}")
print(f"Input sequence (chars): {''.join([idx_to_char[i.item()] for i in input_ex])}")
print(f"Target sequence (indices): {target_ex.tolist()}")
print(f"Target sequence (chars): {''.join([idx_to_char[i.item()] for i in target_ex])}")

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        # Total number of possible sequences is (length of data - block_size)
        # A sequence is 'block_size' long, and its target is the next 'block_size' characters.
        # So, the last possible starting index for an input sequence is len(self.data) - self.block_size - 1.
        # This means there are (len(self.data) - self.block_size - 1) + 1 possible sequences.
        # For example, if data = [0,1,2,3,4] and block_size = 2:
        # idx=0: [0,1]->[1,2]
        # idx=1: [1,2]->[2,3]
        # idx=2: [2,3]->[3,4]
        # The last sequence starts at index len(data) - block_size - 1. So, total is len(data) - block_size.
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Input sequence: characters from index idx to idx + block_size - 1
        input_sequence = self.data[idx : idx + self.block_size]
        # Target sequence: characters from index idx + 1 to idx + block_size
        target_sequence = self.data[idx + 1 : idx + self.block_size + 1]
        return input_sequence, target_sequence
        
from torch.utils.data import Dataset, DataLoader

# 6. Instantiate the custom Dataset
dataset = TextDataset(encoded_text, max_seq_len)
print(f"Dataset instantiated with {len(dataset)} possible sequences.")

# 7. Create a DataLoader instance
batch_size = 64 # Define a suitable batch size
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"DataLoader created with batch size {batch_size}.")
