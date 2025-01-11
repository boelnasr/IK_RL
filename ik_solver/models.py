import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear layers for Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
        # Store attention components
        self.attention_weights = None

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()

        # Project inputs
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax and dropout
        attention_weights = F.softmax(attn_scores, dim=-1)
        self.attention_weights = attention_weights.detach()
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        out = torch.matmul(attention_weights, V)

        # Reshape and combine heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, embed_dim)

        # Final projection
        out = self.output_proj(out)

        return out



class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim=2048, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.attention_weights = None

    def forward(self, x, mask=None):
        attended = self.attention(x, mask)
        self.attention_weights = self.attention.attention_weights  # Save weights for visualization
        x = self.layer_norm1(x + attended)
        ff_out = self.feed_forward(x)
        x = self.layer_norm2(x + ff_out)
        return x

    def get_attention_weights(self):
        return self.attention_weights


class JointActor(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim=1, use_attention=True, num_heads=4, dropout=0.1):
        super(JointActor, self).__init__()
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim

        # Input normalization for stability
        self.input_norm = nn.LayerNorm(input_dim)

        # Enhanced feature extractor with residual connections
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        if use_attention:
            self.transformer = TransformerEncoderBlock(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=hidden_dim * 4,
                dropout=dropout
            )

        # Policy network with residual connections
        self.policy_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ),
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ),
            nn.Linear(hidden_dim, action_dim)
        ])

        # Log standard deviation for actions
        self.log_std = nn.Parameter(torch.ones(action_dim) * -2.0)
        self.log_std_min = -20
        self.log_std_max = -1

        # Initialize weights
        self.apply(self._init_weights)

        # Store attention weights
        self.attention_weights = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            module.bias.data.zero_()

    def forward(self, state):
        # Input normalization
        x = self.input_norm(state)
        x = self.feature_extractor(x)

        # Attention mechanism
        if self.use_attention:
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # Add sequence dimension for transformer
            x = self.transformer(x)
            self.attention_weights = self.transformer.get_attention_weights()  # Store attention weights
            x = x.squeeze(1)  # Remove sequence dimension

        # Policy network with residual connections
        identity = x
        for layer in self.policy_net[:-1]:
            x = layer(x) + identity
            identity = x

        # Final action computation
        action_mean = self.policy_net[-1](x)
        action_mean = torch.tanh(action_mean) * 0.1  # Scale actions for safety

        # Bounded standard deviation
        action_std = torch.exp(torch.clamp(self.log_std, min=self.log_std_min, max=self.log_std_max))

        return action_mean, action_std

    def get_attention_weights(self):
        """Retrieve attention weights for visualization."""
        return self.attention_weights if self.use_attention else None

class CentralizedCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_agents, use_attention=True, num_heads=4, dropout=0.1):
        super(CentralizedCritic, self).__init__()
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        
        # Input processing
        self.input_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Attention mechanism
        if use_attention:
            self.transformer = TransformerEncoderBlock(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=hidden_dim * 4,
                dropout=dropout
            )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, np.sqrt(2))
            module.bias.data.zero_()

    def forward(self, state):
        x = self.input_layer(state)
        
        if self.use_attention:
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            x = self.transformer(x)
            x = x.squeeze(1)
            
        values = self.value_net(x)
        return values

    def get_attention_weights(self):
        if self.use_attention:
            return self.transformer.get_attention_weights()
        return None