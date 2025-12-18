# Dual-Head Hierarchical VAE

Split-head attention VAE for character-level text modeling.

## Architecture

Each transformer layer contains a binary VAE:
- **First N heads**: bidirectional (encoder) — sees full context for posterior
- **Remaining heads**: causal (decoder) — computes prior from z_{<i}

Latents flow through the residual stream, creating a ladder prior across depth. Giving you an encoder / decoder with shared weights and a bunch of computational reuse.

## Key Ideas

| Component | Description |
|-----------|-------------|
| Dual attention | Single QKV projection, split mask per head |
| Binary codes | K bits/layer → 2^K discrete codes, straight-through gradient |
| Free bits | Per-layer KL threshold prevents posterior collapse |
| Span dropout | Random causal span masking for regularization |

