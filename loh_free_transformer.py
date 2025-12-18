"""
Hierarchical VAE with split head attention.
- First N heads: bidirectional (encoder)
- Second H-N heads: causal (decoder)
- Each layer is a VAE with binary latents
- ladder prior
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import numpy as np
import torch.distributed as dist
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, BlockMask
from torch.optim import AdamW
from tqdm import tqdm
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Dict, Any, Optional

from text8_loader import load_text8, Text8FixedLoader

# =============================================================================
# Performance config
# =============================================================================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True
torch.set_float32_matmul_precision('high')

# =============================================================================
# Config
# =============================================================================

@dataclass
class VAEConfig:
    # Data
    vocab_size: int = 27
    seq_len: int = 256
    max_tokens: int = 2**16
    batch_size: int = 128
    
    # Architecture
    dim: int = 768
    heads: int = 12
    encoder_heads: int = 6  # Bidirectional heads (default 1)
    depth: int = 12
    bits_per_layer: int = 4  # Binary latent bits per position per layer
    dtype: str = "bfloat16"  # Training dtype (rope stays float32)
    
    # Training
    lr: float = 3e-4
    warmup_steps: int = 1000
    max_steps: int = 1_000_000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # VAE
    kl_weight: float = 1.0
    kl_warmup_steps: int = 50  # Steps to ramp KL weight 0 -> kl_weight
    kl_threshold: float = 1.5/12  # Free bits per position per layer
    temperature: float = 1.0
    span_dropout: float = 0.5  # Probability of masking random span in causal attention
    
    # Logging
    log_every: int = 50
    eval_every: int = 500  # Evaluate true BPC (from prior)
    eval_crops: int = 100  # Number of crops for training eval
    test_crops: int = 10000  # Number of crops for final test eval (paper uses 1M but that's slow)
    sample_every: int = 1000
    save_every: int = 1000

    @property
    def head_dim(self):
        return self.dim // self.heads
    
    @property
    def decoder_heads(self):
        return self.heads - self.encoder_heads
    
    @property
    def torch_dtype(self):
        return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[self.dtype]


CONFIG = VAEConfig()

# Character mappings
CHAR_TO_IDX = {chr(ord('a') + i): i for i in range(26)}
CHAR_TO_IDX[' '] = 26
IDX_TO_CHAR = {v: k for k, v in CHAR_TO_IDX.items()}

# =============================================================================
# Distributed utils
# =============================================================================

def get_local_rank(): return int(os.environ.get("LOCAL_RANK", 0))
def get_rank(): return int(os.environ.get("RANK", 0))
def get_world_size(): return int(os.environ.get("WORLD_SIZE", 1))
def is_master(): return get_rank() == 0

# =============================================================================
# Pretty printing
# =============================================================================

ANSI = SimpleNamespace(
    COLORS=['\033[92m', '\033[96m', '\033[93m', '\033[91m', 
            '\033[94m', '\033[95m', '\033[97m', '\033[90m'],
    RESET='\033[0m',
    BOLD='\033[1m', 
    DIM='\033[2m',
)

def color(idx): 
    return ANSI.COLORS[idx % len(ANSI.COLORS)]

def tokens_to_text(tokens):
    return ''.join(IDX_TO_CHAR.get(t.item(), '?') for t in tokens)

def print_samples(samples, title="Samples"):
    print(f"\n{color(1)}{'═' * 80}{ANSI.RESET}")
    print(f"{ANSI.BOLD}{color(1)} {title}{ANSI.RESET}")
    print(f"{color(1)}{'═' * 80}{ANSI.RESET}")
    for i, s in enumerate(samples):
        disp = s.replace(' ', '·')[:70]
        print(f"  {ANSI.BOLD}[{i+1}]{ANSI.RESET} {disp}")
    print(f"{color(1)}{'═' * 80}{ANSI.RESET}\n")

# =============================================================================
# Initialization
# =============================================================================

def trunc_normal_(m, std=0.02):
    nn.init.trunc_normal_(m.weight, std=std)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)

def zero_init_(m):
    nn.init.zeros_(m.weight)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)

# =============================================================================
# RoPE
# =============================================================================

def make_rope_cache(seq_len, head_dim, device, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(head_dim // 2, device=device).float() / (head_dim // 2)))
    pos = torch.arange(seq_len, device=device).float()
    angles = pos.unsqueeze(-1) * inv_freq
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin):
    orig_dtype = x.dtype
    x = x.float()
    cos = cos.float().view(1, -1, 1, cos.shape[-1])
    sin = sin.float().view(1, -1, 1, sin.shape[-1])
    x_re, x_im = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    out = torch.stack([x_re * cos - x_im * sin, x_re * sin + x_im * cos], -1)
    return out.flatten(-2).to(orig_dtype)

# =============================================================================
# Binary Mapper (discrete VAE bottleneck)
# =============================================================================

NAT = math.log(2)

def binary_entropy(logits):
    """H(Bernoulli(sigmoid(logits))) in nats."""
    prob = logits.sigmoid()
    not_prob = 1. - prob
    return -(prob * F.logsigmoid(logits) + not_prob * F.logsigmoid(-logits)).sum(dim=-1)


class BinaryMapper(nn.Module):
    """Maps continuous logits to discrete binary codes with straight-through."""
    
    def __init__(self, bits: int, kl_threshold: float = 0.0):
        super().__init__()
        self.bits = bits
        self.num_codes = 2 ** bits
        self.kl_threshold = kl_threshold
        
        power_two = 2 ** torch.arange(bits)
        codes = (torch.arange(self.num_codes)[:, None].bitwise_and(power_two) != 0).float()
        
        self.register_buffer('power_two', power_two, persistent=False)
        self.register_buffer('codes', codes, persistent=False)  # [num_codes, bits]

    def forward(
        self,
        posterior_logits: torch.Tensor,  # [..., bits]
        prior_logits: torch.Tensor,      # [..., bits]
        temperature: float = 1.0,
        straight_through: bool = True,
    ):
        """
        Sample from posterior, compute KL to prior.
        Returns one_hot codes, thresholded KL, raw KL.
        """
        dtype = posterior_logits.dtype
        
        # Sample from posterior
        prob = (posterior_logits / temperature).sigmoid()
        sampled_bits = (torch.rand_like(posterior_logits) <= prob).float()
        indices = (self.power_two * sampled_bits.long()).sum(dim=-1)
        one_hot = F.one_hot(indices, self.num_codes).to(dtype)
        
        # KL(posterior || prior)
        q = posterior_logits.sigmoid()
        kl_per_bit = (
            q * (F.logsigmoid(posterior_logits) - F.logsigmoid(prior_logits)) +
            (1 - q) * (F.logsigmoid(-posterior_logits) - F.logsigmoid(-prior_logits))
        )
        kl_raw = kl_per_bit.sum(dim=-1)
        
        # Free bits threshold
        kl = F.relu(kl_raw - self.kl_threshold)
        
        # Straight-through gradient
        if straight_through:
            codes = self.codes.to(dtype)
            log_soft = (
                F.logsigmoid(posterior_logits) @ codes.T +
                F.logsigmoid(-posterior_logits) @ (1 - codes).T
            )
            soft = log_soft.exp()
            one_hot = one_hot + soft - soft.detach()
        
        return one_hot, kl, kl_raw


# =============================================================================
# Attention context
# =============================================================================

@dataclass
class DualAttnCtx:
    """Context for dual-head attention (encoder bidir + decoder causal)."""
    block_mask: BlockMask
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    num_heads: int


# =============================================================================
# Dual-Head Attention
# =============================================================================

class DualAttention(nn.Module):
    """
    Attention where first half of heads are bidirectional (encoder),
    second half are causal (decoder).
    """

    def __init__(self, dim: int, heads: int, encoder_heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.encoder_heads = encoder_heads
        self.decoder_heads = heads - encoder_heads

        # Unfused q/k/v
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        self.out_enc = nn.Linear(self.heads * self.head_dim, dim, bias=False)
        self.out_dec = nn.Linear(self.decoder_heads * self.head_dim, dim, bias=False)

        trunc_normal_(self.q_proj)
        trunc_normal_(self.k_proj)
        trunc_normal_(self.v_proj)
        zero_init_(self.out_enc)
        zero_init_(self.out_dec)

    def forward(self, x: torch.Tensor, block_mask: BlockMask, rope_cos: torch.Tensor, rope_sin: torch.Tensor):
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.heads, self.head_dim)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)
        q = norm(q)
        k = norm(k)

        q = q.transpose(1, 2)  # [B, H, N, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = flex_attention(q, k, v, block_mask=block_mask, scale=self.scale)
        out = out.transpose(1, 2)  # [B, N, H, D]

        # Split into encoder (bidir) and decoder (causal) outputs, encoder gets both
        enc_out = out.reshape(B, N, -1)
        dec_out = out[:, :, self.encoder_heads:, :].reshape(B, N, -1)

        return self.out_enc(enc_out), self.out_dec(dec_out)


class MLP(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * mult, bias=False)
        self.fc2 = nn.Linear(dim * mult, dim, bias=False)
        trunc_normal_(self.fc1)
        zero_init_(self.fc2)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)).square())


# =============================================================================
# VAE Layer
# =============================================================================
def norm(x):
    return F.rms_norm(x, (x.shape[-1],), eps=1e-5)

class VAELayer(nn.Module):
    """
    Single layer of hierarchical VAE.
    - Dual attention: encoder (bidir) + decoder (causal)
    - Binary latent with ladder prior (prior from decoder, z_{<i} in residual)
    """
    
    def __init__(self, cfg: VAEConfig, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        
        
        # Dual attention
        self.attn = DualAttention(cfg.dim, cfg.heads, cfg.encoder_heads)
        
        # MLP
        self.mlp = MLP(cfg.dim)
        
        # Posterior: from encoder output (bidir, sees everything)
        self.posterior_proj = nn.Linear(cfg.dim, cfg.bits_per_layer, bias=True)
        
        # Prior: from decoder output (causal, z_{<i} in residual stream)
        self.prior_proj = nn.Linear(cfg.dim, cfg.bits_per_layer, bias=True)
        
        # z to hidden
        self.z_to_hidden = nn.Linear(2 ** cfg.bits_per_layer, cfg.dim, bias=False)
        
        # Binary mapper
        self.mapper = BinaryMapper(cfg.bits_per_layer, kl_threshold=cfg.kl_threshold)
        
        # Init
        trunc_normal_(self.posterior_proj, std=0.02)
        trunc_normal_(self.prior_proj, std=0.02)
        zero_init_(self.z_to_hidden)

    def forward(
        self,
        x: torch.Tensor,           # [B, N, D]
        block_mask: BlockMask,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        temperature: float = 1.0,
        use_prior: bool = False,
        z_cache: Optional[torch.Tensor] = None,  # [N, num_codes] cached z one-hots
    ):
        # Dual attention
        x_norm = norm(x)
        enc_out, dec_out = self.attn(x_norm, block_mask, rope_cos, rope_sin)
        
        # Posterior from encoder (bidir)
        posterior_logits = self.posterior_proj(enc_out)
        
        # Prior from decoder (causal) - z_{<i} already in x via residual
        prior_logits = self.prior_proj(dec_out)
        
        # Sample z
        if use_prior:
            # Inference: sample from prior
            z_one_hot, _, _ = self.mapper(prior_logits, prior_logits, temperature, straight_through=False)
            # Use cached z where available
            if z_cache is not None:
                cache_len = z_cache.shape[0]
                z_one_hot = z_one_hot.squeeze(0)  # [N, num_codes]
                z_one_hot[:cache_len] = z_cache
                z_one_hot = z_one_hot.unsqueeze(0)  # [1, N, num_codes]
            kl = torch.zeros_like(x[:, :, 0])
            kl_raw = kl
        else:
            # Training: sample from posterior, KL to prior
            z_one_hot, kl, kl_raw = self.mapper(posterior_logits, prior_logits, temperature)
        
        # Project z and update residual
        z_hidden = self.z_to_hidden(z_one_hot)
        x = x + dec_out + z_hidden
        x = x + self.mlp(norm(x))
        
        return x, kl, kl_raw, z_one_hot.squeeze(0)  # Return z for caching


# =============================================================================
# Hierarchical VAE
# =============================================================================

def offsets_to_batch_ids(total_len: int, offsets: torch.Tensor) -> torch.Tensor:
    return torch.bucketize(torch.arange(total_len, device=offsets.device), offsets[:-1], right=True) - 1


class HierarchicalVAE(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        
        # Token embedding
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        trunc_normal_(self.token_embed, std=0.02)
        
        # VAE layers
        self.layers = nn.ModuleList([VAELayer(cfg, i) for i in range(cfg.depth)])
        
        # Output head
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        trunc_normal_(self.head, std=0.02)
        
        # RoPE cache
        self.register_buffer('rope_cos', None, persistent=False)
        self.register_buffer('rope_sin', None, persistent=False)

    def init_rope(self, device):
        self.rope_cos, self.rope_sin = make_rope_cache(
            self.cfg.seq_len, self.cfg.head_dim, device
        )

    def make_ctx(self, offsets, training: bool = False, span_dropout: float = 0.5) -> DualAttnCtx:
        """Create dual-head attention context with optional span masking."""
        N = int(offsets[-1])
        device = offsets.device
        num_heads = self.cfg.heads
        encoder_heads = self.cfg.encoder_heads
        
        batch_ids = offsets_to_batch_ids(N, offsets)
        seq_positions = torch.arange(N, device=device) - offsets[batch_ids]
        
        rope_cos = self.rope_cos[seq_positions]
        rope_sin = self.rope_sin[seq_positions]
        
        if training and span_dropout > 0:
            # Random span per position for causal heads
            max_span_end = seq_positions + 1
            span_end_rel = torch.randint(1, 64, (N,), device=device).clamp(max=max_span_end)
            span_len = torch.randint(4, 32, (N,), device=device)
            span_start_rel = (span_end_rel - span_len).clamp(min=0)
            
            seq_starts = offsets[batch_ids]
            span_start = seq_starts + seq_positions - span_end_rel + 1
            span_end = seq_starts + seq_positions - span_start_rel + 1
            
            use_span = torch.rand(N, device=device) < span_dropout
            
            def dual_mask_mod(b, h, q_idx, kv_idx):
                same_seq = batch_ids[q_idx] == batch_ids[kv_idx]
                is_causal_head = h >= encoder_heads
                causal_ok = kv_idx <= q_idx
                
                # Mask out random span for causal heads
                in_span = (kv_idx >= span_start[q_idx]) & (kv_idx < span_end[q_idx])
                causal_ok = causal_ok & ~(in_span & use_span[q_idx])
                
                return same_seq & (~is_causal_head | causal_ok)
        else:
            # Inference or no span dropout: standard dual mask
            def dual_mask_mod(b, h, q_idx, kv_idx):
                same_seq = batch_ids[q_idx] == batch_ids[kv_idx]
                is_causal_head = h >= encoder_heads
                causal_ok = kv_idx <= q_idx
                return same_seq & (~is_causal_head | causal_ok)
        
        block_mask = create_block_mask(
            dual_mask_mod,
            B=None, H=num_heads, Q_LEN=N, KV_LEN=N,
            device=device, _compile=True
        )
        
        return DualAttnCtx(block_mask, rope_cos, rope_sin, num_heads)

    def _forward_impl(
        self,
        x: torch.Tensor,
        block_mask: BlockMask,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        temperature: float,
        use_prior: bool,
        z_cache: Optional[List[torch.Tensor]] = None,  # [depth] list of [cache_len, num_codes]
    ):
        """Core forward - compilable."""
        N = x.shape[1]
        depth = len(self.layers)
        
        kl_all = torch.zeros(depth, N, device=x.device, dtype=x.dtype)
        kl_raw_all = torch.zeros(depth, N, device=x.device, dtype=x.dtype)
        z_all = []  # Store z for each layer
        
        for i, layer in enumerate(self.layers):
            layer_cache = z_cache[i] if z_cache is not None else None
            x, kl, kl_raw, z = layer(
                x, block_mask, rope_cos, rope_sin,
                temperature, use_prior, layer_cache
            )
            kl_all[i] = kl.squeeze(0)
            kl_raw_all[i] = kl_raw.squeeze(0)
            z_all.append(z)
        
        logits = self.head(norm(x)).squeeze(0)  # [N, V]
        
        return logits, kl_all, kl_raw_all, z_all


    @torch.compile(fullgraph=True)
    def _forward_fullgraph(
        self,
        x: torch.Tensor,
        block_mask: BlockMask,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        temperature: float,
        use_prior: bool,
        z_cache: Optional[List[torch.Tensor]] = None,  # [depth] list of [cache_len, num_codes]
    ):
        return self._forward_impl(x, block_mask, rope_cos, rope_sin, temperature, use_prior, z_cache)
    
    @torch.compile(dynamic=True)
    def _forward_dynamic(
        self,
        x: torch.Tensor,
        block_mask: BlockMask,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        temperature: float,
        use_prior: bool,
        z_cache: Optional[List[torch.Tensor]] = None,  # [depth] list of [cache_len, num_codes]
    ):
        return self._forward_impl(x, block_mask, rope_cos, rope_sin, temperature, use_prior, z_cache)



    def forward(
        self,
        tokens: torch.Tensor,  # [N]
        ctx: DualAttnCtx,
        temperature: float = 1.0,
        use_prior: bool = False,
        z_cache: Optional[List[torch.Tensor]] = None,  # [depth] list of [cache_len, num_codes]
    ):
        x = self.token_embed(tokens).unsqueeze(0)  # [1, N, D]
        
        forward_impl = self._forward_fullgraph if use_prior else self._forward_dynamic

        logits, kl_per_layer, kl_raw_per_layer, z_all = forward_impl(
            x, ctx.block_mask, ctx.rope_cos, ctx.rope_sin,
            temperature, use_prior, z_cache
        )
        
        return SimpleNamespace(
            logits=logits,
            kl_per_layer=kl_per_layer,
            kl_raw_per_layer=kl_raw_per_layer,
            kl_total=kl_per_layer.sum(dim=0).mean(),
            z_all=z_all,  # [depth] list of [N, num_codes]
        )

    @torch.no_grad()
    def sample(self, num_samples=1, device=None, temperature=1.0):
        if device is None:
            device = next(self.parameters()).device
        
        seq_len = self.cfg.seq_len
        depth = self.cfg.depth
        num_codes = 2 ** self.cfg.bits_per_layer
        dtype = self.cfg.torch_dtype
        samples = []
        
        for _ in range(num_samples):
            tokens = torch.zeros(seq_len, dtype=torch.long, device=device)
            # z_cache[layer] = [cached_len, num_codes]
            z_cache = [torch.empty(0, num_codes, device=device, dtype=dtype) for _ in range(depth)]
            
            for i in range(seq_len - 1):
                offsets = torch.tensor([0, i + 1], device=device)
                ctx = self.make_ctx(offsets)
                out = self.forward(tokens[:i+1], ctx, temperature, use_prior=True, z_cache=z_cache)
                
                # Update cache with new z at position i
                for l in range(depth):
                    z_cache[l] = out.z_all[l][:i+1]  # Keep z for positions 0..i
                
                probs = F.softmax(out.logits[-1] / temperature, dim=-1)
                tokens[i + 1] = torch.multinomial(probs, 1)
            
            samples.append(tokens_to_text(tokens))
        
        return samples


# =============================================================================
# Training
# =============================================================================

def train_step(model: HierarchicalVAE, tokens: torch.Tensor, ctx: DualAttnCtx, cfg: VAEConfig, step: int):
    out = model(tokens, ctx, cfg.temperature)
    
    # Reconstruction loss (next token prediction from causal decoder)
    recon_loss = F.cross_entropy(out.logits[:-1], tokens[1:])
    
    # KL loss with warmup
    kl_loss = out.kl_total
    kl_weight = cfg.kl_weight * min(1.0, step / cfg.kl_warmup_steps)
    
    # Total loss
    loss = recon_loss + kl_weight * kl_loss
    
    # Metrics (all in bits)
    recon = recon_loss.item() / math.log(2)  # Reconstruction term (decoder sees z from posterior)
    
    # Per-layer KL (thresholded and raw)
    kl_per_layer = out.kl_per_layer.mean(dim=1)  # [depth]
    kl_raw_per_layer = out.kl_raw_per_layer.mean(dim=1)  # [depth]
    
    # Rate (total bits per position)
    rate = kl_per_layer.sum().item() / math.log(2)
    rate_raw = kl_raw_per_layer.sum().item() / math.log(2)
    
    # ELBO = recon + rate_raw (true coding cost upper bound)
    elbo = recon + rate_raw
    
    return loss, SimpleNamespace(
        recon=recon,
        rate=rate,
        rate_raw=rate_raw,
        elbo=elbo,
        kl_weight=kl_weight,
        kl_per_layer=[k.item() / math.log(2) for k in kl_per_layer],
        kl_raw_per_layer=[k.item() / math.log(2) for k in kl_raw_per_layer],
    )


@torch.no_grad()
def eval_true_bpc(model: HierarchicalVAE, tokens: torch.Tensor, ctx: DualAttnCtx, temperature: float = 1.0):
    """
    Evaluate true BPC by sampling z from prior (causal, no future leakage).
    This is the actual compression rate at test time.
    """
    out = model(tokens, ctx, temperature, use_prior=True)
    loss = F.cross_entropy(out.logits[:-1], tokens[1:])
    return loss.item() / math.log(2)


@torch.no_grad()
def evaluate(
    model: HierarchicalVAE,
    data: np.ndarray,
    cfg: VAEConfig,
    device: torch.device,
    num_crops: int = 1000,
    desc: str = "Eval",
) -> dict:
    """
    Evaluate on random crops from data.
    
    Returns dict with:
        - true_bpc: actual compression (z from prior)
        - elbo: upper bound (recon + rate with posterior)
        - recon: reconstruction term
        - rate: KL term
    """
    model.eval()
    seq_len = cfg.seq_len
    data_len = len(data)
    
    true_bpc_sum = 0.0
    elbo_sum = 0.0
    recon_sum = 0.0
    rate_sum = 0.0
    
    for _ in tqdm(range(num_crops), desc=desc, disable=not is_master()):
        # Random crop
        start = torch.randint(0, data_len - seq_len - 1, (1,)).item()
        tokens = torch.tensor(data[start:start + seq_len], dtype=torch.long, device=device)
        
        offsets = torch.tensor([0, seq_len], device=device)
        ctx = model.make_ctx(offsets, training=False)
        
        # True BPC (prior) - z from prior, no future leakage
        out_prior = model(tokens, ctx, cfg.temperature, use_prior=True)
        true_loss = F.cross_entropy(out_prior.logits[:-1], tokens[1:])
        true_bpc_sum += true_loss.item() / math.log(2)
        
        # ELBO components (posterior) - use raw KL for true rate
        out_post = model(tokens, ctx, cfg.temperature, use_prior=False)
        recon_loss = F.cross_entropy(out_post.logits[:-1], tokens[1:])
        kl_raw_per_layer = out_post.kl_raw_per_layer.mean(dim=1)
        rate = kl_raw_per_layer.sum().item() / math.log(2)
        recon = recon_loss.item() / math.log(2)
        
        recon_sum += recon
        rate_sum += rate
        elbo_sum += recon + rate
    
    return {
        'true_bpc': true_bpc_sum / num_crops,
        'elbo': elbo_sum / num_crops,
        'recon': recon_sum / num_crops,
        'rate': rate_sum / num_crops,
    }


def main():
    cfg = CONFIG
    
    dist.init_process_group("nccl")
    device = torch.device(f"cuda:{get_local_rank()}")
    
    model = HierarchicalVAE(cfg).to(device=device, dtype=cfg.torch_dtype)
    model.init_rope(device)  # RoPE stays float32

    if is_master():
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_bits = cfg.depth * cfg.bits_per_layer
        print(f"\n{ANSI.BOLD}{'═' * 60}{ANSI.RESET}")
        print(f"{ANSI.BOLD}{color(0)} Hierarchical VAE (Dual-Head){ANSI.RESET}")
        print(f"{'═' * 60}")
        print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
        print(f"  Config: depth={cfg.depth}, dim={cfg.dim}, heads={cfg.heads}, dtype={cfg.dtype}")
        print(f"  Heads: {cfg.encoder_heads} encoder (bidir) + {cfg.decoder_heads} decoder (causal)")
        print(f"  Latent: {cfg.bits_per_layer} bits/layer × {cfg.depth} layers = {total_bits} bits/pos")
        print(f"  Max rate: {total_bits:.1f} bits/pos")
        print(f"  Span dropout: {cfg.span_dropout}")
        print(f"{'═' * 60}\n")
    
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    
    # Optimizer
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if param.ndim == 1 or 'bias' in name or 'norm' in name or 'embed' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer = AdamW([
        {'params': decay_params, 'weight_decay': cfg.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9)
    
    train_data, valid_data, test_data = load_text8()
    train_loader = Text8FixedLoader(
        train_data,
        max_tokens=cfg.max_tokens,
        seq_len=cfg.seq_len,
        batch_size=cfg.batch_size,
        shuffle=True
    )
    
    if is_master():
        print(f"  Train data: {len(train_data):,} chars\n")
    
    model.train()
    step = 0
    ema = {}
    
    pbar = tqdm(train_loader, disable=not is_master(), desc="Training", total=cfg.max_steps)
    for batch in pbar:
        if step > cfg.max_steps:
            break
        
        # LR warmup
        warmup_frac = min(step / cfg.warmup_steps, 1.0)
        for pg in optimizer.param_groups:
            pg['lr'] = cfg.lr * warmup_frac
        
        tokens = batch.tokens.to(device)
        offsets = batch.offsets.to(device)
        ctx = model.make_ctx(offsets, training=True, span_dropout=cfg.span_dropout)
        
        model.zero_grad(set_to_none=True)
        loss, metrics = train_step(model, tokens, ctx, cfg, step)
        loss.backward()
        
        for p in model.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        
        # EMA
        for k in ['recon', 'rate', 'rate_raw', 'elbo']:
            v = getattr(metrics, k)
            ema[k] = v if k not in ema else 0.99 * ema[k] + 0.01 * v
        
        pbar.set_postfix(elbo=f"{ema['elbo']:.3f}", recon=f"{ema['recon']:.3f}", rate=f"{ema['rate']:.2f}", β=f"{metrics.kl_weight:.2f}")
        
        if is_master() and step % cfg.log_every == 0:
            kl_str = " ".join(f"{k:.2f}" for k in metrics.kl_per_layer)
            kl_raw_str = " ".join(f"{k:.2f}" for k in metrics.kl_raw_per_layer)
            true_bpc_str = f" │ {color(5)}true{ANSI.RESET}={ema.get('true_bpc', 0):.3f}" if 'true_bpc' in ema else ""
            print(f"  step {step:6d} │ {color(0)}elbo{ANSI.RESET}={ema['elbo']:.3f} │ "
                  f"{color(1)}recon{ANSI.RESET}={ema['recon']:.3f} │ {color(2)}rate{ANSI.RESET}={ema['rate']:.2f}/{metrics.rate_raw:.2f} │ "
                  f"{color(3)}β{ANSI.RESET}={metrics.kl_weight:.3f}{true_bpc_str}")
            print(f"           │ {color(4)}kl{ANSI.RESET}=[{kl_str}] │ {color(6)}raw{ANSI.RESET}=[{kl_raw_str}]")
        
        # Evaluate on validation set (z from prior, no future leakage)
        if is_master() and cfg.eval_every > 0 and step > 0 and step % cfg.eval_every == 0:
            eval_results = evaluate(model, valid_data, cfg, device, num_crops=cfg.eval_crops, desc="Valid")
            ema['true_bpc'] = eval_results['true_bpc'] if 'true_bpc' not in ema else 0.9 * ema['true_bpc'] + 0.1 * eval_results['true_bpc']
            print(f"  {color(5)}>>> valid: true_bpc={eval_results['true_bpc']:.4f} elbo={eval_results['elbo']:.4f} "
                  f"(recon={eval_results['recon']:.4f} + rate={eval_results['rate']:.3f}){ANSI.RESET}")
            model.train()
        
        if is_master() and cfg.sample_every > 0 and step > 0 and step % cfg.sample_every == 0:
            model.eval()
            try:
                samples = model.sample(num_samples=3, device=device, temperature=1.0)
                print_samples(samples, title=f"Samples @ step {step}")
            except Exception as e:
                tqdm.write(f"{ANSI.COLORS[3]}Sampling failed: {e}{ANSI.RESET}")
            model.train()
        
        if is_master() and cfg.save_every > 0 and step % cfg.save_every == 0:
            torch.save({'step': step, 'model': model.state_dict(), 'cfg': cfg},
                      'checkpoint_dual_vae.pt')
        
        step += 1
    
    # Final test evaluation
    if is_master():
        print(f"\n{ANSI.BOLD}{'═' * 60}{ANSI.RESET}")
        print(f"{ANSI.BOLD}{color(0)} Final Test Evaluation ({cfg.test_crops} crops){ANSI.RESET}")
        print(f"{'═' * 60}")
        
        test_results = evaluate(model, test_data, cfg, device, num_crops=cfg.test_crops, desc="Test")
        
        print(f"\n  {color(0)}TRUE BPC{ANSI.RESET}  = {test_results['true_bpc']:.4f}  ← comparable to table")
        print(f"  {color(1)}ELBO{ANSI.RESET}      = {test_results['elbo']:.4f}  (upper bound)")
        print(f"  {color(2)}recon{ANSI.RESET}     = {test_results['recon']:.4f}")
        print(f"  {color(3)}rate{ANSI.RESET}      = {test_results['rate']:.4f} bits/char")
        print(f"\n{'═' * 60}\n")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
