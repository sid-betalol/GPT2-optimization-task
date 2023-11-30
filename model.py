import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint

# Configuration class for model parameters
@dataclass
class GPT2Config:
    vocab_size: int
    max_position_embeddings: int
    n_layer: int
    n_head: int
    n_embd: int
    bias : bool
    dropout : float
    weight_tying : bool

class LayerNorm(nn.Module):

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

# Multi-head self-attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections
        self.c_attn = nn.Linear(config.n_embd, config.n_embd * 3, bias = config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = config.bias)
        # split into heads
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        #dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.f_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.max_position_embeddings, config.max_position_embeddings))
                                        .view(1, 1, config.max_position_embeddings, config.max_position_embeddings))

    def forward(self, x):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.f_dropout(self.c_proj(y))
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).type_as(inv_freq)
        freqs = torch.einsum('n , d -> n d', t, inv_freq)
        self.register_buffer('sin', freqs.sin())
        self.register_buffer('cos', freqs.cos())

    def forward(self, x):
        seq_len, dimension = x.shape[-2:]
        sin, cos = self.sin[:seq_len, :dimension//2], self.cos[:seq_len, :dimension//2]
        return torch.cat((x[..., :dimension//2] * cos + x[..., dimension//2:] * sin,
                          x[..., :dimension//2] * sin - x[..., dimension//2:] * cos), dim=-1)
    
class GroupedMultiHeadAttention(nn.Module):
    def __init__(self, config, num_groups=2):
        super().__init__()
        self.num_groups = num_groups
        self.group_size = config.n_head // num_groups
        self.scaled_dot_attn = MultiHeadAttention(config)

    def forward(self, x):
        B, T, C = x.size()
        # Splitting heads into groups
        x = x.view(B, T, self.num_groups, self.group_size, C // self.group_size)
        # Applying attention within each group
        x = torch.cat([self.scaled_dot_attn(x[:, :, i]) for i in range(self.num_groups)], dim=2)
        return x.view(B, T, C)

class SlidingWindowAttention(nn.Module):
    def __init__(self, config, window_size=64):
        super().__init__()
        self.window_size = window_size
        self.attn = MultiHeadAttention(config)

    def forward(self, x):
        B, T, C = x.size()
        # Applying attention within a sliding window
        padded_x = torch.nn.functional.pad(x, (0, 0, self.window_size // 2, self.window_size // 2))
        windows = padded_x.unfold(1, self.window_size, 1)
        y = torch.cat([self.attn(window) for window in windows], dim=1)
        return y

# Feed-forward network
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = nn.functional.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# Transformer block (combining attention and feed-forward network)
class TransformerBlock(nn.Module):
    def __init__(self, config, use_rope=False, use_group_query=False, use_sliding_window=False):
        
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)


        # Add the appropriate attention mechanism based on the flags
        if use_group_query:
            self.attn = GroupedMultiHeadAttention(config)
        elif use_sliding_window:
            self.attn = SlidingWindowAttention(config)
        else:
            self.attn = MultiHeadAttention(config)

        # RoPE integration
        if use_rope:
            self.rope = RotaryPositionalEmbedding(config.n_embd)
        else:
            self.rope = None

        self.mlp = FeedForward(config)

    def forward(self, x):
        # Apply Rotary Positional Embedding if enabled
        if self.rope is not None:
            x = self.rope(x)

        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Model(nn.Module):
    def __init__(self, config, use_rope=False, use_group_query=False, use_sliding_window=False, use_checkpointing=False):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.use_rope = use_rope

        if not self.use_rope:
            self.wpe = nn.Embedding(config.max_position_embeddings, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config, use_rope, use_group_query, use_sliding_window) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.use_checkpointing = use_checkpointing
        self.config = config

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        if config.weight_tying:
            self.wte.weight = self.head.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):

        x = self.wte(x)

        if not self.use_rope:
            positions = torch.arange(0, x.shape[1], dtype=torch.long, device=x.device)
            x = x + self.wpe(positions)

        x = self.drop(x)

        for block in self.blocks:
            if self.use_checkpointing:
                x = checkpoint(block, x)
            else:
                x = block(x)
        x = self.ln_f(x)
        x = self.head(x)
        return x
    
    @torch.no_grad()
    def load_state_dict(self, ckpt):

        ckpt = {key.replace("h.", "blocks."):val for key, val in ckpt.items() if ".attn.bias" not in key}

        ckpt["head.weight"] = ckpt["wte.weight"]

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        for k in ckpt.keys():
            if any(k.endswith(w) for w in transposed):
                ckpt[k] = ckpt[k].t()

        super().load_state_dict(ckpt)

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens, temperature=1.0, top_k=None):

        B, T = tokens.shape

        for _ in range(max_new_tokens):
            tokens_cond = tokens if tokens.size(1) <= self.config.max_position_embeddings else tokens[:, -self.config.max_position_embeddings:]
            logits = self(tokens_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            tokens_next = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat((tokens, tokens_next), dim=1)

        return tokens
