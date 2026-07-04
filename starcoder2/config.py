class StarCoderV2Config:
  def __init__(
      self,vocab_size=50257,
      hidden_size=768,
      num_hidden_layers=12,
      num_attention_heads=12,
      num_key_value_heads=2,
      intermediate_size=3072,
      rope_theta=1000.0,
      dropout_rate=0.1,
      norm_epsilon=1e-5,
      max_position_embeddings=1024
  ):
    self.vocab_size=vocab_size
    self.hidden_size=hidden_size
    self.num_hidden_layers=num_hidden_layers
    self.num_attention_heads=num_attention_heads
    self.num_key_value_heads=num_key_value_heads
    self.intermediate_size=intermediate_size
    self.rope_theta=rope_theta
    self.dropout_rate=dropout_rate
    self.norm_epsilon=norm_epsilon
    self.max_position_embeddings=max_position_embeddings
    assert self.num_attention_heads % self.num_key_value_heads==0,"num_attention_heads must be divisible by num_key_value_heads for GOA"
    self.num_key_value_groups=self.num_attention_heads//self.num_key_value_heads
