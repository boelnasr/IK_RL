import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.2):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
        # Store attention components
        self.attention_weights = None

    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.size()

        # Project inputs to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Optional mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax + dropout
        attention_weights = F.softmax(attn_scores, dim=-1)
        self.attention_weights = attention_weights.detach()
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of V
        out = torch.matmul(attention_weights, V)

        # Reshape and combine heads
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, embed_dim)

        # Final linear projection
        out = self.output_proj(out)

        return out


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim=2048, dropout=0.2):
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
        """
        x: (batch_size, seq_len, embed_dim)
        """
        attended = self.attention(x, mask)
        self.attention_weights = self.attention.attention_weights  # store for visualization

        # Residual connection + LayerNorm
        x = self.layer_norm1(x + attended)

        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.layer_norm2(x + ff_out)
        return x

    def get_attention_weights(self):
        return self.attention_weights


class JointActor(nn.Module):
    """
    JointActor uses:
      1. Input layer norm + feature extraction (MLP).
      2. Optional Transformer block for attention over features/time.
      3. A GRU for temporal processing (multi-step if seq_len > 1).
      4. A final policy network (with residual connections) to output action means.
      5. log_std for action standard deviations.
    """
    def __init__(self, input_dim, hidden_dim, action_dim=1,
                 use_attention=True, num_heads=4, dropout=0.2,
                 gru_layers=1):
        super(JointActor, self).__init__()
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers

        # 1. Input normalization for stability
        self.input_norm = nn.LayerNorm(input_dim)

        # 2. Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        # 3. Optional Transformer block
        if use_attention:
            self.transformer = TransformerEncoderBlock(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=hidden_dim * 4,
                dropout=dropout
            )
        else:
            self.transformer = None

        # 4. GRU for temporal processing
        #    If your input is always single-step, seq_len=1, the GRU won't do much,
        #    but it still can store hidden states for partial observability.
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,  # (batch, seq_len, input_size)
            dropout=dropout if gru_layers > 1 else 0.0
        )

        # 5. Policy network with residual connections
        self.policy_blocks = nn.ModuleList([
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
            )
        ])
        self.policy_out = nn.Linear(hidden_dim, action_dim)

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

    def forward(self, state_seq):
        """
        Handles both single-step and multi-step sequences.
        state_seq: (batch_size, input_dim) or (batch_size, seq_len, input_dim)
        """
        # If state_seq has 2 dimensions, add a seq_len dimension of 1
        if len(state_seq.shape) == 2:
            state_seq = state_seq.unsqueeze(1)  # (batch_size, 1, input_dim)

        batch_size, seq_len, _ = state_seq.shape

        # 1. Normalize inputs
        state_seq = self.input_norm(state_seq)

        # 2. Feature extraction for each time step
        feats = self.feature_extractor(state_seq)

        # 3. Optional Transformer over time or features
        if self.use_attention:
            feats = self.transformer(feats)  # shape still (batch, seq_len, hidden_dim)
            self.attention_weights = self.transformer.get_attention_weights()

        # 4. GRU
        gru_out, _ = self.gru(feats)  # (batch_size, seq_len, hidden_dim)

        # 5. Take the last time step's output
        last_out = gru_out[:, -1, :]  # (batch_size, hidden_dim)

        # 6. Policy network with residual blocks
        x = last_out
        identity = x
        for block in self.policy_blocks:
            x = block(x) + identity
            identity = x

        # Final action computation
        action_mean = self.policy_out(x)
        action_mean = torch.tanh(action_mean) * 0.1  # Scale actions for safety

        # Bounded standard deviation
        action_std = torch.exp(
            torch.clamp(self.log_std, min=self.log_std_min, max=self.log_std_max)
        )

        return action_mean, action_std



    def get_attention_weights(self):
        return self.attention_weights


class CentralizedCritic(nn.Module):
    """
    A feed-forward (optionally attention-based) critic that processes
    a concatenated state of multiple agents (or a combined global state).
    Outputs one value per agent or a single global value, depending on final layer.
    """
    def __init__(self, state_dim, hidden_dim, num_agents,
                 use_attention=False, num_heads=4, dropout=0.2):
        super(CentralizedCritic, self).__init__()
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        
        # 1. Input processing
        self.input_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # 2. Optional Transformer
        if use_attention:
            self.transformer = TransformerEncoderBlock(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=hidden_dim * 4,
                dropout=dropout
            )
        else:
            self.transformer = None
        
        # 3. Value network (feed-forward)
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_agents)  # or 1 if you want a single global value
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, np.sqrt(2))
            module.bias.data.zero_()

    def forward(self, state):
        """
        state: (batch_size, state_dim)
          If multi-agent, you might have state_dim = sum(all agents' obs dims),
          or some combined global representation.
        """
        x = self.input_layer(state)

        if self.use_attention:
            # Expand to (batch_size, seq_len=1, hidden_dim) for the Transformer
            x = x.unsqueeze(1)
            x = self.transformer(x)
            x = x.squeeze(1)
            
        values = self.value_net(x)
        return values

    def get_attention_weights(self):
        if self.use_attention and self.transformer is not None:
            return self.transformer.get_attention_weights()
        return None



