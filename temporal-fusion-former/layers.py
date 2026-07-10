import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Implement Gated Linear Unit (GLU)
class GLU(nn.Module):
    def __init__(self, input_size, output_size):
        super(GLU, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.gate = nn.Linear(input_size, output_size)

    def forward(self, x):
        lin = self.linear(x)
        gate = torch.sigmoid(self.gate(x))
        return lin * gate

# 2. Implement Gated Residual Network (GRN)
class GRN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.0):
        super(GRN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU() # FIX: Removed extraneous comma
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout_rate)
        self.glu = GLU(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)

        if input_size != output_size:
            self.skip_connection = nn.Linear(input_size, output_size)
        else:
            self.skip_connection = None

    def forward(self, x, context=None):
        hidden = self.linear1(x)
        hidden = self.elu(hidden)

        hidden = self.linear2(hidden)
        hidden = self.dropout(hidden)

        gated_output = self.glu(hidden)

        if self.skip_connection is not None:
            residual = self.skip_connection(x)
        else:
            residual = x

        output = self.layer_norm(gated_output + residual)
        return output

# 3. Implement Variable Selection Network (VSN)
class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, num_inputs, hidden_size, output_size, dropout_rate=0.0):
        super(VariableSelectionNetwork, self).__init__()
        self.num_inputs = num_inputs
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.context_grn = GRN(input_size, hidden_size, hidden_size, dropout_rate=dropout_rate)

        self.input_embeddings = nn.ModuleList([
            nn.Linear(input_size, hidden_size) for _ in range(num_inputs)
        ])

        # FIX: `attention_grn` should output 1 for each input item, not `num_inputs`. Reshape later.
        self.attention_grn = GRN(hidden_size * 2, hidden_size, 1, dropout_rate=dropout_rate) # Output 1 scalar weight per input

        self.final_grn = GRN(hidden_size, hidden_size, output_size, dropout_rate=dropout_rate) # Input to final_grn is hidden_size (sum of weighted embeddings)

    def forward(self, inputs, context=None):
        if not isinstance(inputs, list):
            inputs = [inputs]

        is_sequence = len(inputs[0].shape) == 3

        if is_sequence:
            batch_size, time_steps, _ = inputs[0].shape
            flat_inputs = [inp.reshape(batch_size * time_steps, -1) for inp in inputs]
            if context is not None:
                # FIX: If context is already a sequence (B, T, H), it just needs to be reshaped
                context_expanded = context.reshape(batch_size * time_steps, -1)
            else:
                context_expanded = None # context is optional
        else:
            batch_size, _ = inputs[0].shape
            flat_inputs = inputs
            context_expanded = context

        if context_expanded is not None:
            processed_context = self.context_grn(context_expanded)
        else:
            # If no context, create a zero tensor with batch_size*time_steps for sequence or batch_size for non-sequence
            processed_context = torch.zeros(flat_inputs[0].shape[0], self.hidden_size, device=flat_inputs[0].device)

        embeddings = [self.input_embeddings[i](flat_input) for i, flat_input in enumerate(flat_inputs)]

        # Concatenate embeddings with processed_context for attention GRN
        attention_inputs = [torch.cat([embedding, processed_context], dim=-1) for embedding in embeddings]
        attention_inputs_stacked = torch.stack(attention_inputs, dim=1) # (batch*timesteps, num_inputs, 2*hidden_size)
        attention_grn_input = attention_inputs_stacked.view(-1, 2 * self.hidden_size) # (batch*timesteps*num_inputs, 2*hidden_size)

        raw_attention_weights = self.attention_grn(attention_grn_input) # Shape: (batch*timesteps*num_inputs, 1)

        # FIX: Reshape raw_attention_weights correctly to (batch*timesteps, num_inputs) for softmax
        # Ensure the view operation correctly reflects the original batch structure
        raw_attention_weights = raw_attention_weights.view(flat_inputs[0].shape[0], self.num_inputs) # (B*T, N)

        attention_weights = F.softmax(raw_attention_weights, dim=-1)

        weighted_embeddings = []
        for i, embedding in enumerate(embeddings):
            weighted_embeddings.append(embedding * attention_weights[:, i].unsqueeze(-1))

        summed_weighted_embeddings = torch.stack(weighted_embeddings, dim=0).sum(dim=0) # (batch*timesteps or batch_size, hidden_size)

        output = self.final_grn(summed_weighted_embeddings)

        if is_sequence:
            output = output.view(batch_size, time_steps, -1)
            attention_weights = attention_weights.view(batch_size, time_steps, -1)

        return output, attention_weights

# 4. Implement Static Covariate Encoder
class StaticCovariateEncoder(nn.Module):
    def __init__(self, input_categorical_sizes, input_real_size, hidden_size, output_size, dropout_rate=0.0):
        super(StaticCovariateEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.static_categorical_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=size, embedding_dim=hidden_size)
            for size in input_categorical_sizes
        ])

        embedding_total_size = len(input_categorical_sizes) * hidden_size
        grn_input_size = embedding_total_size + input_real_size

        self.static_grn = GRN(
            input_size=grn_input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            dropout_rate=dropout_rate
        )

    def forward(self, static_categorical_data, static_real_data):
        embedded_categorical_features = []
        for i, cat_data in enumerate(static_categorical_data):
            embedded_categorical_features.append(self.static_categorical_embeddings[i](cat_data.long()))

        if embedded_categorical_features:
            concatenated_categorical_embeddings = torch.cat(embedded_categorical_features, dim=-1)
        else:
            batch_size = static_real_data.shape[0]
            concatenated_categorical_embeddings = torch.empty(batch_size, 0, device=static_real_data.device)

        if static_real_data is not None and static_real_data.shape[-1] > 0:
            combined_static_input = torch.cat([concatenated_categorical_embeddings, static_real_data], dim=-1)
        else:
            combined_static_input = concatenated_categorical_embeddings

        static_context_vector = self.static_grn(combined_static_input)

        return static_context_vector

# 5. Implement Dynamic Covariate Encoder
class DynamicCovariateEncoder(nn.Module):
    def __init__(
        self,
        known_categorical_sizes,
        known_real_size,
        unknown_categorical_sizes,
        unknown_real_size,
        hidden_size,
        num_gru_layers,
        dropout_rate=0.0
    ):
        super(DynamicCovariateEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_gru_layers = num_gru_layers

        self.known_cat_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=size, embedding_dim=hidden_size)
            for size in known_categorical_sizes
        ])
        self.known_real_projections = nn.ModuleList([
            nn.Linear(1, hidden_size) for _ in range(known_real_size)
        ]) if known_real_size > 0 else nn.ModuleList()

        self.unknown_cat_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=size, embedding_dim=hidden_size)
            for size in unknown_categorical_sizes
        ])
        self.unknown_real_projections = nn.ModuleList([
            nn.Linear(1, hidden_size) for _ in range(unknown_real_size)
        ]) if unknown_real_size > 0 else nn.ModuleList()

        num_known_dynamic_features = len(known_categorical_sizes) + known_real_size
        num_unknown_dynamic_features = len(unknown_categorical_sizes) + unknown_real_size

        self.known_dynamic_vsn = VariableSelectionNetwork(
            input_size=hidden_size,
            num_inputs=num_known_dynamic_features,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate
        )

        self.unknown_dynamic_vsn = VariableSelectionNetwork(
            input_size=hidden_size,
            num_inputs=num_unknown_dynamic_features,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate
        )

        self.history_combine_grn = GRN(
            input_size=2 * hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate
        )

        self.future_combine_grn = GRN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True
        )

    def forward(
        self,
        historical_known_categorical_data,
        historical_known_real_data,
        historical_unknown_categorical_data,
        historical_unknown_real_data,
        future_known_categorical_data,
        future_known_real_data,
        static_context
    ):
        batch_size = historical_known_real_data.shape[0]
        encoder_length = historical_known_real_data.shape[1]
        decoder_length = future_known_real_data.shape[1]

        static_context_encoder = static_context.unsqueeze(1).expand(-1, encoder_length, -1)
        static_context_decoder = static_context.unsqueeze(1).expand(-1, decoder_length, -1)

        historical_known_inputs_transformed = []
        for i, cat_data in enumerate(historical_known_categorical_data):
            historical_known_inputs_transformed.append(self.known_cat_embeddings[i](cat_data.long()))
        for i, real_data in enumerate(historical_known_real_data.unbind(dim=-1)):
            historical_known_inputs_transformed.append(self.known_real_projections[i](real_data.unsqueeze(-1)))

        selected_known_historical_features, _ = self.known_dynamic_vsn(
            inputs=historical_known_inputs_transformed,
            context=static_context_encoder
        )

        historical_unknown_inputs_transformed = []
        for i, cat_data in enumerate(historical_unknown_categorical_data):
            historical_unknown_inputs_transformed.append(self.unknown_cat_embeddings[i](cat_data.long()))
        for i, real_data in enumerate(historical_unknown_real_data.unbind(dim=-1)):
            historical_unknown_inputs_transformed.append(self.unknown_real_projections[i](real_data.unsqueeze(-1)))

        selected_unknown_historical_features, _ = self.unknown_dynamic_vsn(
            inputs=historical_unknown_inputs_transformed,
            context=static_context_encoder
        )

        combined_historical_features = torch.cat([
            selected_known_historical_features,
            selected_unknown_historical_features
        ], dim=-1)

        processed_historical_features = self.history_combine_grn(combined_historical_features)

        lstm_output, _ = self.lstm(processed_historical_features)

        future_known_inputs_transformed = []
        for i, cat_data in enumerate(future_known_categorical_data):
            future_known_inputs_transformed.append(self.known_cat_embeddings[i](cat_data.long()))
        for i, real_data in enumerate(future_known_real_data.unbind(dim=-1)):
            future_known_inputs_transformed.append(self.known_real_projections[i](real_data.unsqueeze(-1)))

        selected_known_future_features, _ = self.known_dynamic_vsn(
            inputs=future_known_inputs_transformed,
            context=static_context_decoder
        )

        processed_future_features = self.future_combine_grn(selected_known_future_features)

        return lstm_output, processed_future_features

# 6. Implement MultiHeadAttention Mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate=0.0):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        matmul_qk = torch.matmul(q, k.transpose(-2, -1))

        dk = torch.tensor(self.head_dim, dtype=torch.float32).to(query.device)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        if mask is not None:
            scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scaled_attention_logits, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, v)

        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.embed_dim)

        output = self.out_proj(output)

        return output, attention_weights

# --- Helper function for categorical sizes ---
def get_categorical_sizes(df, feature_list):
    original_feature_list = [f.replace('_encoded', '') if '_encoded' in f else f for f in feature_list]
    return [df[f].nunique() + 1 for f in original_feature_list]

