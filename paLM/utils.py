
#  real_tokenize_multilingual function 
def real_tokenize_multilingual(texts, tokenizer,max_length=128):
    encoded_inputs = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    return encoded_inputs['input_ids']
