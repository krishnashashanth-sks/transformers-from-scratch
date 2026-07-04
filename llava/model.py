import torch.nn as nn
from layers import *

class LLaVA(nn.Module):
  def __init__(self,vision_encoder_params,projection_layer_params,language_model_params):
    super().__init__()
    self.vision_encoder=VisionEncoder(
        image_size=vision_encoder_params['image_size'],
        patch_size=vision_encoder_params['patch_size'],
        in_channels=vision_encoder_params['in_channels'],
        embed_dim=vision_encoder_params['embed_dim'],
        num_heads=vision_encoder_params['num_heads'],
        num_layers=vision_encoder_params['num_layers'],
        mlp_dim=vision_encoder_params['mlp_dim'],
    )
    self.projection_layer=ProjectionLayer(
        input_dim=vision_encoder_params['input_dim'],
        output_dim=vision_encoder_params['output_dim']
    )
    self.language_model=LanguageModel(
        vocab_size=vision_encoder_params['vocab_size'],
        embed_dim=vision_encoder_params['embed_dim'],
        max_seq_len=vision_encoder_params['max_seq_len'],
        num_heads=vision_encoder_params['num_heads'],
        num_layers=vision_encoder_params['num_layers'],
        mlp_dim=vision_encoder_params['mlp_dim']
    )
  def forward(self,pixel_values,input_ids):
    vision_features=self.vision_encoder(pixel_values)
    projected_visual_features=self.projection_layer(vision_features)
    return self.language_model(input_ids,projected_visual_features)

print("LLaVA model and its sub-components defined successfully.")