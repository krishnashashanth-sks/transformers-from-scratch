from dataset import tgt_vocab,src_vocab
import torch
from layers import LabelSmoothing
from model import iQTransformer
from losses import SimpleLossCompute
from train import train
from dataset import train_dataloader, prepared_data
from utils import greedy_decode,make_src_mask,ids_to_tokens

padding_idx = tgt_vocab.word2idx['<pad>']

model = iQTransformer(src_vocab.n_words, tgt_vocab.n_words, N=2, d_model=256, d_ff=1024, num_heads=4, dropout=0.1, num_bits=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)

padding_idx = tgt_vocab.word2idx['<pad>']
loss_criterion = LabelSmoothing(size=tgt_vocab.n_words, padding_idx=padding_idx, smoothing=0.1)
compute_loss = SimpleLossCompute(loss_criterion)

num_epochs = 5 # Set a reasonable number of epochs for demonstration

train(num_epochs,model,train_dataloader,optimizer,device,src_vocab,tgt_vocab,compute_loss)

# Ensure model is in evaluation mode
model.eval()

# Collect translations for qualitative assessment
results = []

with torch.no_grad():
    for i, sample in enumerate(prepared_data):
        # a. Extract and prepare src_ids
        src_ids = torch.tensor(sample['src_ids'], dtype=torch.long).unsqueeze(0).to(device) # Add batch dimension
        src_text = sample['src_text']
        ground_truth_tgt_text = sample['tgt_text']

        # b. Move source tensor to device (already done above)

        # c. Generate the source mask
        src_mask = make_src_mask(src_ids, src_vocab.word2idx['<pad>'])

        # d. Call the greedy_decode function
        # Define max_len: typically source length + a buffer, or a fixed max
        max_len = src_ids.size(1) + 10 # Adding a buffer of 10 tokens

        decoded_ids = greedy_decode(model,
                                    src_ids,
                                    src_mask,
                                    max_len,
                                    tgt_vocab.word2idx['<sos>'],
                                    tgt_vocab.word2idx['<eos>'],
                                    src_vocab.word2idx['<pad>'],
                                    tgt_vocab.word2idx['<pad>'],
                                    device)

        # e. Convert decoded IDs back to words
        decoded_tokens = ids_to_tokens(decoded_ids.squeeze(0).tolist(), tgt_vocab)

        # f. Post-process the list of decoded tokens
        final_translation_tokens = []
        for token in decoded_tokens:
            if token in ['<sos>', '<pad>']:
                continue
            if token == '<eos>':
                break
            final_translation_tokens.append(token)
        final_translation = ' '.join(final_translation_tokens)

        # g. Print the results
        print(f"\n--- Sample {i+1} ---")
        print(f"Source: {src_text}")
        print(f"Ground Truth: {ground_truth_tgt_text}")
        print(f"Generated Translation: {final_translation}")
        results.append({
            'source': src_text,
            'ground_truth': ground_truth_tgt_text,
            'generated': final_translation
        })
