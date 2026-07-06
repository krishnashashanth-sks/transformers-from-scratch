class RoBERTaConfig:
    """Configuration for a RoBERTa-base equivalent model."""
    def __init__(self):
        self.vocab_size = 50265  # RoBERTa-base vocabulary size
        self.hidden_size = 768  # Dimensionality of the encoder layers and the pooler layer
        self.num_attention_heads = 12  # Number of attention heads for each attention layer in the Transformer encoder
        self.num_hidden_layers = 12  # Number of hidden layers in the Transformer encoder
        self.intermediate_size = 3072  # Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder
        self.max_position_embeddings = 514  # The maximum sequence length that this model might ever be used with

    def __repr__(self):
        return (
            f"RoBERTaConfig(\n"
            f"    vocab_size={self.vocab_size},\n"
            f"    hidden_size={self.hidden_size},\n"
            f"    num_attention_heads={self.num_attention_heads},\n"
            f"    num_hidden_layers={self.num_hidden_layers},\n"
            f"    intermediate_size={self.intermediate_size},\n"
            f"    max_position_embeddings={self.max_position_embeddings}\n"
            f")"
        )
