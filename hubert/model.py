
import torch.nn as nn

class TransformerEncoder(nn.Module):
  def __init__(self,input_dim,model_dim,num_heads,num_layers,dropout=0.1):
    super(TransformerEncoder,self).__init__()
    self.input_projection=nn.Linear(input_dim,model_dim)
    encoder_layer=nn.TransformerEncoderLayer(d_model=model_dim,nhead=num_heads,dropout=dropout,batch_first=True)
    self.transformer_encoder=nn.TransformerEncoder(encoder_layer,num_layers=num_layers)
  def forward(self,x):
    x=x.permute(0,2,1)
    x=self.input_projection(x)
    return self.transformer_encoder(x)
    
class PredictionHead(nn.Module):
  def __init__(self,model_dim,num_clusters):
    super(PredictionHead,self).__init__()
    self.prediction_layer=nn.Linear(model_dim,num_clusters)
  def forward(self,transformer_output):
    return self.prediction_layer(transformer_output)