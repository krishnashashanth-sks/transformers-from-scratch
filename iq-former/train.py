import time
from utils import make_src_mask,make_tgt_mask

def train(num_epochs,model,train_dataloader,optimizer,device,src_vocab,tgt_vocab,compute_loss):
    for epoch in range(num_epochs):
        model.train() # Set the model to training mode
        total_loss = 0
        start_time = time.time()

        for i, batch in enumerate(train_dataloader):
            # Explicitly zero gradients at the beginning of each batch processing
            optimizer.zero_grad()

            # a. Get data from the batch
            src_ids = batch['src_ids']
            tgt_input_ids = batch['tgt_input_ids']
            tgt_output_ids = batch['tgt_output_ids']

            # b. Move to device
            src_ids = src_ids.to(device)
            tgt_input_ids = tgt_input_ids.to(device)
            tgt_output_ids = tgt_output_ids.to(device)

            # c. Create masks
            src_mask = make_src_mask(src_ids, src_vocab.word2idx['<pad>'])
            tgt_mask = make_tgt_mask(tgt_input_ids, tgt_vocab.word2idx['<pad>'])

            # d. Perform forward pass
            model_output = model(src_ids, tgt_input_ids, src_mask, tgt_mask)

            # Pass model_output through the generator layer to get log-probabilities over vocabulary
            output_for_loss = model.generator(model_output)

            # e. Calculate loss (now SimpleLossCompute only returns the tensor)
            # Flatten the model output and target for loss computation
            norm = (tgt_output_ids != tgt_vocab.word2idx['<pad>']).sum().item()
            loss_tensor = compute_loss(output_for_loss.contiguous().view(-1, output_for_loss.size(-1)),
                                    tgt_output_ids.contiguous().view(-1),
                                    normalize_batch_size=False)

            # Perform backpropagation and update parameters in the main loop
            loss_tensor.backward()
            optimizer.step()
            # optimizer.zero_grad() is already called at the start of the loop

            # Accumulate loss as a scalar for logging
            total_loss += loss_tensor.item() / norm

        # f. Print loss after each epoch
        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}, Time: {epoch_duration:.2f}s")
