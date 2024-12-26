import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with a built-in residual connection and layer norm.
    Expects input shape: (batch_size, seq_len, embed_dim).
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Query, Key, Value projections
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout & LayerNorm for the output of the attention block
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Scaling factor for dot-product attention
        self.scale = self.head_dim ** 0.5

    def forward(self, x, mask=None, causal_mask=False):
        """
        x: Tensor of shape [batch_size, seq_len, embed_dim].
        mask: Optional attention mask (shape [batch_size, seq_len]).
        causal_mask: If True, applies a causal (look-ahead) mask for language-like tasks.
        """
        batch_size, seq_len, _ = x.size()

        # Project inputs to Q, K, V
        Q = self.query_layer(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key_layer(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value_layer(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch_size, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute raw attention scores: (Q dot K^T) / sqrt(head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, num_heads, seq_len, seq_len]

        # Apply mask (e.g., padding mask) if provided
        if mask is not None:
            # mask: [batch_size, seq_len] -> shape broadcasting for multihead
            # We expand dims to [batch_size, 1, 1, seq_len] or similar
            # Then the logic masked_fill(mask == 0, -inf)
            expanded_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, seq_len]
            scores = scores.masked_fill(expanded_mask == 0, float('-inf'))

        # Apply causal mask if needed
        if causal_mask:
            causal = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
            scores = scores.masked_fill(causal == 0, float('-inf'))

        # Softmax over last dimension (seq_len), then dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, V)  # [B, num_heads, seq_len, head_dim]

        # Reshape back to [batch_size, seq_len, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Output projection
        attn_output = self.out_proj(attn_output)

        # Residual connection + LayerNorm
        out = self.layer_norm(x + attn_output)
        return out


class TransformerFeedForward(nn.Module):
    """
    Simple 2-layer feed-forward network (often called position-wise feed-forward in Transformers).
    """
    def __init__(self, embed_dim, ff_dim=2048, dropout=0.1):
        super(TransformerFeedForward, self).__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        # Another residual connection + layernorm
        x = self.layer_norm(residual + x)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    A single Transformer encoder block: MH Attention -> residual + LN -> FeedForward -> residual + LN.
    """
    def __init__(self, embed_dim, num_heads=4, ff_dim=2048, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = TransformerFeedForward(embed_dim, ff_dim, dropout=dropout)

    def forward(self, x, mask=None, causal_mask=False):
        # Multi-head Self-Attention
        x = self.attention(x, mask=mask, causal_mask=causal_mask)
        # Feed Forward
        x = self.ffn(x)
        return x


class JointActor(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim, use_attention=False, num_heads=4, dropout=0.1):
        super(JointActor, self).__init__()
        self.use_attention = use_attention
        self.hidden_dim = hidden_dim

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Optional Transformer block
        if use_attention:
            self.transformer_block = TransformerEncoderBlock(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                ff_dim=4 * hidden_dim,  # typical feed-forward dimension
                dropout=dropout
            )

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # output in [-1, 1], adjust if your env needs different bounds
        )

        # Log std as a learnable parameter
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Init
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, state):
        """
        Expects state: [batch_size, input_dim].
        If using attention, we treat it as (batch_size, seq_len=1, hidden_dim).
        """
        # Feature extraction
        x = self.feature_extractor(state)  # [batch_size, hidden_dim]

        if self.use_attention:
            # Reshape to [batch_size, seq_len=1, hidden_dim]
            x = x.unsqueeze(1)
            x = self.transformer_block(x)  # still [batch_size, 1, hidden_dim]
            x = x.squeeze(1)  # back to [batch_size, hidden_dim]

        # Actor head
        action_mean = self.actor(x)  # [batch_size, action_dim]

        # log_std -> clamp for numerical stability
        # We expand to match the shape of action_mean
        log_std_clamped = torch.clamp(self.log_std, min=-2.0, max=2.0)
        action_std = torch.exp(log_std_clamped).expand_as(action_mean)

        return action_mean, action_std


class CentralizedCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_agents, use_attention=False, num_heads=4,
                 ff_dim=2048, dropout=0.1, num_transformer_blocks=1):
        """
        state_dim: dimension of the concatenated (global) state
        hidden_dim: dimension for each transformer's embedding / MLP dimension
        num_agents: how many agents (output dimension)
        """
        super(CentralizedCritic, self).__init__()

        self.use_attention = use_attention
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents

        # Initial feature extraction
        self.input_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

        # If using attention-based approach, stack multiple transformer blocks
        self.transformer_blocks = nn.ModuleList()
        if use_attention:
            for _ in range(num_transformer_blocks):
                block = TransformerEncoderBlock(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout
                )
                self.transformer_blocks.append(block)

        # Additional MLP after attention (or after input) for value estimation
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Final output for value function: shape [batch_size, num_agents]
        self.output_layer = nn.Linear(hidden_dim, num_agents)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, states):
        """
        states: [batch_size, state_dim].
        If use_attention=True, we treat each batch element as (batch_size, seq_len=1, hidden_dim).
        """
        # Feature extraction
        x = self.input_layer(states)  # [batch_size, hidden_dim]

        if self.use_attention:
            # Convert to [batch_size, seq_len=1, hidden_dim]
            x = x.unsqueeze(1)  # seq_len=1
            for block in self.transformer_blocks:
                x = block(x)  # [batch_size, 1, hidden_dim]

            x = x.squeeze(1)  # back to [batch_size, hidden_dim]

        # MLP
        x = self.mlp(x)  # [batch_size, hidden_dim]

        # Output: one value per agent
        values = self.output_layer(x)  # [batch_size, num_agents]
        return values
