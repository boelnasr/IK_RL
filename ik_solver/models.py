import torch
import torch.nn as nn
import torch.nn.functional as F


# Multi-Head Attention Class
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Layers for query, key, and value projections
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)

        # Output projection layer
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Additional layers for stability and flexibility
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Scaling factor for dot-product attention
        self.scale = self.head_dim ** 0.5

    def forward(self, query, key, value, mask=None, causal_mask=False):
        batch_size = query.size(0)

        # Project inputs to queries, keys, and values
        Q = (
            self.query_layer(query)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # [B, num_heads, seq_len, head_dim]
        K = (
            self.key_layer(key)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.value_layer(value)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, num_heads, seq_len, seq_len]

        # Apply masks if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Causal mask to prevent attending to future positions
        if causal_mask:
            seq_len = scores.size(-1)
            causal_mask = (
                torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
                .unsqueeze(0)
                .unsqueeze(0)
            )
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))

        # Softmax to get attention probabilities, then apply dropout
        attn_weights = self.dropout(F.softmax(scores, dim=-1))

        # Calculate attention output
        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, seq_len, head_dim]
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.embed_dim)
        )  # [B, seq_len, embed_dim]

        # Project output and apply residual connection and layer norm
        attn_output = self.out_proj(attn_output)
        attn_output = self.layer_norm(query + attn_output)  # Residual connection

        return attn_output


# Joint Actor with optional Attention
class JointActor(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, use_attention=False, num_heads=4):
        super(JointActor, self).__init__()
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim

        # Optional feature extractor layer before attention
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Multi-head attention layer if enabled
        if use_attention:
            self.attention = MultiHeadAttention(embed_dim=hidden_dim, num_heads=num_heads)

        # Updated actor network with additional layers
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Tanh bounds actions between -1 and 1
        )

        # Log standard deviation for action exploration
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Adjusted to be a vector per action dimension

        # Weight initialization
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, state):
        # Apply feature extraction
        x = self.feature_extractor(state)

        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x, x, x) + x  # Residual connection with attention

        # Pass through the actor network
        action_mean = self.actor(x)

        # Ensure std is not too small for numerical stability
        action_std = torch.clamp(self.log_std.exp().expand_as(action_mean), min=1e-3)

        return action_mean, action_std


class CentralizedCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_agents, use_attention=False, num_heads=4, dropout=0.1):
        super(CentralizedCritic, self).__init__()

        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents

        # Feature extraction with layer normalization
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Optional multi-head attention layer
        if use_attention:
            self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
            self.attention_norm = nn.LayerNorm(hidden_dim)

        # Core critic network with residual connections and dropout
        self.critic_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)),
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)),
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)),
        ])

        # Final output layer to estimate state value for each agent
        self.output_layer = nn.Linear(hidden_dim, num_agents)

        # Weight initialization
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, states):
        # Feature extraction and normalization
        x = self.feature_extractor(states)  # [batch_size, hidden_dim]

        # Apply attention with residual connection if enabled
        if self.use_attention:
            # nn.MultiheadAttention expects [seq_len, batch_size, embed_dim]
            x = x.unsqueeze(0)  # [1, batch_size, hidden_dim]
            attn_out, _ = self.attention(x, x, x)  # [1, batch_size, hidden_dim]
            attn_out = attn_out.squeeze(0)  # [batch_size, hidden_dim]
            x = x.squeeze(0) + self.attention_norm(attn_out)  # [batch_size, hidden_dim]

        # Pass through critic layers with residual connections
        for layer in self.critic_layers:
            x = x + layer(x)  # [batch_size, hidden_dim]

        # Output layer for value estimation
        x = self.output_layer(x)  # [batch_size, num_agents]

        # Debugging output shape
        #print(f"Output shape from CentralizedCritic before returning: {x.shape}")

        return x  # [batch_size, num_agents]