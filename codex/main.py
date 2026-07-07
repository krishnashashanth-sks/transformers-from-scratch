from torch.utils.data import DataLoader
import torch # Import torch for device definition
import os
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from dataset import CodeDataset
from model import TransformerDecoder
from train import train_step
from utils import create_attention_mask
import torch.nn as nn

# Prepare some dummy code for training the tokenizer
dummy_code_corpus = [
    "import math\ndef calculate_area(radius):\n    return math.pi * radius**2\n",
    "class MyShape:\n    def __init__(self, color):\n        self.color = color\n",
    "print('Hello, World!') # Simple print statement\nfor i in range(10):\n    if i % 2 == 0:\n        print(f'Even number: {i}')\n"
    "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n-1):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
    "import numpy as np\ndef matrix_multiply(A, B):\n    return np.matmul(A, B)",
    "class Animal:\n    def __init__(self, name):\n        self.name = name\n    def speak():\n        raise NotImplementedError",
    "class Dog(Animal):\n    def speak(self):\n        return f'{self.name} says Woof!'",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b",
    "result = list(fibonacci(10))",
    "def greet(name):\n    print(f'Hello, {name}!')",
    "greet('World')",
    "a = 10\nb = 20\nc = a + b # Simple addition"
]

# Initialize a Byte-level BPE tokenizer
tokenizer = ByteLevelBPETokenizer()

# Train the tokenizer
# In a real scenario, you would pass a list of file paths here
tokenizer.train_from_iterator(dummy_code_corpus, vocab_size=1000, min_frequency=2,
                              special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

# Save the tokenizer files (vocab.json and merges.txt)
tokenizer_path = "./code_tokenizer"
# Create the directory if it doesn't exist
os.makedirs(tokenizer_path, exist_ok=True)
tokenizer.save_model(tokenizer_path)

print(f"Tokenizer saved to {tokenizer_path}/")

loaded_tokenizer = Tokenizer(BPE.from_file(vocab=f"{tokenizer_path}/vocab.json",
                                        merges=f"{tokenizer_path}/merges.txt"))
loaded_tokenizer.pre_tokenizer = ByteLevel()
loaded_tokenizer.post_processor = BertProcessing(
    ("</s>", loaded_tokenizer.token_to_id("</s>")),
    ("<s>", loaded_tokenizer.token_to_id("<s>")),)

pad_token_id = loaded_tokenizer.token_to_id("<pad>")
eos_token_id = loaded_tokenizer.token_to_id("</s>")
bos_token_id = loaded_tokenizer.token_to_id("<s>")
# Model parameters needed for dataset and dataloader setup
max_seq_len = 50 # Maximum sequence length the model can handle

# Encode all dummy texts using the loaded_tokenizer
# `loaded_tokenizer` and `max_seq_len` are assumed to be available from previous cells.
encoded_all_texts = [loaded_tokenizer.encode(text) for text in dummy_code_corpus]

# Create the CodeDataset instance
code_dataset = CodeDataset(encoded_all_texts, max_seq_len, pad_token_id, eos_token_id)

# Create a DataLoader for batching
batch_size = 2 # Reuse batch_size from previous cell
data_loader = DataLoader(code_dataset, batch_size=batch_size, shuffle=True)
# --- Training setup with the CodeDataset for the advanced model ---

# Model parameters
vocab_size = loaded_tokenizer.get_vocab_size()
embed_dim = 512
num_heads = 8
ff_dim = 2048
num_layers = 4   # More layers for an advanced model
max_seq_len = 50 # Maximum sequence length the model can handle
dropout_rate = 0.1

# Instantiate the Transformer Decoder model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerDecoder(vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len, dropout_rate).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id) # Standard for multi-class classification (token prediction)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # Reduced learning rate

num_epochs = 10 # Increased number of epochs

for epoch in range(num_epochs):
    for input_ids,labels in data_loader:
        loss=train_step(model, optimizer, input_ids, labels, pad_token_id, device)
        print("Epoch {epoch}, Loss: {loss}")

model.eval() # Set model to evaluation mode
with torch.no_grad(): # Disable gradient calculation
    test_input_ids, test_target_ids = next(iter(data_loader)) # Take one batch from the DataLoader for testing
    test_input_ids = test_input_ids.to(device)

    test_combined_mask = create_attention_mask(test_input_ids, pad_token_id, device)
    test_output_logits = model(test_input_ids, mask=test_combined_mask)

    # Get predicted token IDs (e.g., greedy decoding)
    predicted_token_ids = torch.argmax(test_output_logits, dim=-1)

    print("\nTest input token IDs shape:", test_input_ids.shape)
    print("Test output logits shape:", test_output_logits.shape)
    print("Predicted token IDs for test input (first sequence, first 10 tokens):", predicted_token_ids[0, :10])
    print("Actual target IDs for test input (first sequence, first 10 tokens):", test_target_ids[0, :10])