from main import SEQ_LEN

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop.
    total_length = (total_length // SEQ_LEN) * SEQ_LEN
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + SEQ_LEN] for i in range(0, total_length, SEQ_LEN)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result
