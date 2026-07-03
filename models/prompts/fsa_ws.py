"""
FSA-WS — Frequency- and Similarity-Aware Warm Start
====================================================
Standalone initializer for new prompt slots in SparsePromptMemory.

Math
----
Given n existing slots with usage counts uᵢ, prototypes pᵢ, and
parameter tensors θᵢ, and a context vector z_t:

    ũᵢ    = uᵢ / Σuⱼ                              (normalize usage)
    aᵢ    = (α·ũᵢ + β·Sim(z_t, pᵢ)) / T           (logits)
    wᵢ    = exp(aᵢ) / Σexp(aⱼ)                    (softmax weights)
    θ_new = Σ wᵢ · θᵢ                             (warm-start init)

When the pool is empty (n=0), returns a zero tensor of shape prompt_dim.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List


def fsa_ws_init(
    z_t: Tensor,
    prompt_dim: tuple,
    prototypes: List[Tensor],
    usage: List[int],
    prompts: nn.ParameterDict,
    alpha: float,
    beta: float,
    temp_T: float,
) -> Tensor:
    """
    Compute a warm-start initialization tensor for a new prompt slot.

    Args:
        z_t        : Context representation that triggered slot creation.
        prompt_dim : Shape of each prompt tensor.
        prototypes : Detached CPU prototype tensors, one per existing slot.
        usage      : Integer usage counts, one per existing slot.
        prompts    : nn.ParameterDict holding existing slot parameters.
        alpha      : Weight for the usage-frequency term.
        beta       : Weight for the cosine-similarity term.
        temp_T     : Softmax temperature (must be > 0).

    Returns:
        theta_new — CPU tensor of shape prompt_dim, detached from the graph.
    """
    n = len(prototypes)

    if n == 0:
        return torch.zeros(prompt_dim)

    # --- Step 1: normalize usage counts ---
    u = torch.tensor(usage[:n], dtype=torch.float32)
    u_sum = u.sum()
    u_tilde = u / u_sum if u_sum > 0 else torch.full((n,), 1.0 / n)

    # --- Step 2: cosine similarities Sim(z_t, p_i) ---
    z_flat = z_t.detach().cpu().flatten().unsqueeze(0)  # (1, D)
    sims = torch.tensor(
        [
            F.cosine_similarity(
                z_flat, prototypes[i].flatten().unsqueeze(0)
            ).item()
            for i in range(n)
        ],
        dtype=torch.float32,
    )

    # --- Step 3: logits and softmax weights ---
    a = (alpha * u_tilde + beta * sims) / temp_T
    w = F.softmax(a, dim=0)  # (n,)

    # --- Step 4: weighted sum of existing parameters ---
    theta_new = torch.zeros(prompt_dim)
    for i in range(n):
        theta_new = theta_new + w[i].item() * prompts[f"slot_{i}"].data.cpu()

    return theta_new
