# This file containts the implementation the transformer block used in experiments
import torch
import torch.nn as nn
from torch.nn import functional as F

# block_size
# vocab_size
# n_layer
# n_head
# n_embd

class Head(nn.Module):
    """one head of self-attention"""
    def __init__(self, head_size, d_embd, seq_length):
        super().__init__()
        self.head_size = head_size
        self.d_embd = d_embd
        self.seq_length = seq_length # context length
        self.key = nn.Linear(d_embd, head_size, bias = False)
        self.query = nn.Linear(d_embd, head_size, bias = False)
        self.value = nn.Linear(d_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(seq_length, seq_length)))
    
    def forward(self, x):
        B, T, C = x.shape # Batch size, seq_length, d_embd
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # Compute attention score
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5 # (B, T, h) @ (B, h, T) --> (B, T, T)
        wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = 1) # (B, T, T)
        # Perform the weighted aggregation
        out = wei @ v # (B, T, T) @ (B, T, h) --> (B, T, h)
        return out

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, d_embd, seq_length):
        super().__init__()
        head_size = d_embd // num_heads
        self.heads = nn.ModuleList([Head(head_size, d_embd, seq_length) for _ in range(num_heads)]) # head_size * num_heads = d_embd
        self.proj = nn.Linear(d_embd, d_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1) # (B, T, d_embd), as if we have just used an nn.Embedding
        out = self.proj(out)
        return out

class MLP(nn.Module):
    def __init__(self, d_embd):
        super().__init__()
        self.c_fc = nn.Linear(d_embd, 4 * d_embd)
        self.relu = nn.ReLU()
        self.c_proj = nn.Linear(4 * d_embd, d_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.relu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, d_embd, num_heads, seq_length):
        super().__init__()
        self.sa = MultiHeadAttention(num_heads, d_embd, seq_length)
        self.mlp = MLP(d_embd)
        self.ln1 = nn.LayerNorm(d_embd)
        self.ln2 = nn.LayerNorm(d_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_embd, num_heads, seq_length, n_layer, device):
        # n_layer is the number of transformer blocks
        super().__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.d_embd = d_embd
        self.num_heads = num_heads
        self.seq_length = seq_length
        self.n_layer = n_layer

        # Embedding tables
        self.token_embedding = nn.Embedding(vocab_size, d_embd)
        self.positional_encoding = nn.Embedding(seq_length, d_embd)

        self.transformer = nn.ModuleList([Block(d_embd, num_heads, seq_length) for _ in range(n_layer)])

        self.lm_head = nn.Linear(d_embd, vocab_size, bias = False) # Final classifier

        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.n_layer) ** -0.5 # Reduce std when w have Residual connections
            torch.nn.init.normal_(module.weight, mean = 0.0, std = std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
        
    def forward(self, idx, targets = None):
        # idx is the list of tokens, targets is of shape (B, T) the next tokens
        B, T = idx.shape
        tok_embd = self.token_embedding(idx) # (B, T, d_embd)
        pos_embd = self.positional_encoding(torch.arange(T, device = self.device)) # (T, d_embd)
        x = tok_embd + pos_embd

        for block in self.transformer:
            x = block(x)
        logits = self.lm_head(x)

        if targets is None:
            loss =  None
        else:
            B, T, p = logits.shape # p = vocab_size
            logits = logits.view(B * T, p)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -self.seq_length:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C) # softmax is applied along each row
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

        
# Simple NN for MNIST classification
class Mnist_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.output = nn.Linear(128, 10)
    
    def forward(self, x):
        # x of shape (B, 28*28)
        x = self.linear(x)
        x = self.relu(x)
        logits = self.output(x)
        return logits

