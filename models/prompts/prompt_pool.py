"""
Sparse Prompt Memory with MoE Sparse Soft Routing
==================================================
Refactored from the discrete-slot SparsePromptMemory to a fully data-driven
Mixture of Experts (MoE) Sparse Soft Routing architecture.

Design Rationale
----------------
In the original design, prompt slots were discrete entities indexed by
cosine-similarity thresholds (delta_reuse / delta_new).  This introduces
a hard, similarity-based clustering inductive bias that may not align with
the true underlying concept boundaries in non-stationary TSF.

The refactored design replaces the discrete slot retrieval + routing table with:

  1. A continuous expert bank  Θ ∈ ℝ^(num_experts × D)
  2. A differentiable router    R: ℝ^D → ℝ^(num_experts)
  3. Top-K sparse dispatch      — only top-k experts activate per (B, C)
  4. A masked load-balancing auxiliary loss — expert usage is only
     penalised for channels that triggered Actual Drift.

Mathematical Specification
--------------------------
Given a query z_q ∈ ℝ^(B×C×D):

  Step 1 — Router logits:
    l = R(z_q) ∈ ℝ^(B×C×E)        (E = num_experts)

  Step 2 — Softmax:
    p = softmax(l, dim=-1) ∈ ℝ^(B×C×E),  Σₑ pₑ = 1

  Step 3 — Top-K sparsity:
    p_topK = topK_mask(p, k=top_k)
    p_norm = p_topK / (Σₑ p_topKₑ + ε)   ← re-normalise per (B,C)

  Step 4 — Weighted dispatch:
    Θ_dispatch = Θ[indices_topK]          ← gather [B×C×K×D]
    θ          = einsum('BCK, BCKD→BCD', p_norm, Θ_dispatch)

  Step 5 — Load-balancing loss (active channels only):
    f_i = one_hot(argmax(p, dim=-1))              ← non-diff; detached
    P_i = mean(p[:, drift_mask, i])               ← differentiable
    L_aux = α · Σᵢ (f_i · P_i)                   ← only over active C

The router receives gradients from L_aux through p, so it learns to route
more evenly across experts for channels marked as drifting (drift_mask=1).
Channels with drift_mask=0 are excluded entirely from the auxiliary loss,
preserving their expert affinity (they are "stable").
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional
import math


class SparsePromptMemory(nn.Module):
    """
    MoE-based Sparse Prompt Memory for continual online TSF.

    Components
    ----------
      prompts : nn.Parameter [num_experts, prompt_dim]
                — continuous expert bank; all experts are always present
      router  : nn.Linear [prompt_dim → num_experts]
                — lightweight, fully differentiable

    Public API
    ----------
      retrieve_prompt(z_query)           → (theta, routing_probs, dispatch_indices)
      force_prompt(target_idx_tensor)    → theta_forced
      compute_load_balancing_loss(...)  → l_aux (scalar)
      route_novelty(routing_probs)      → routing_confidence
      update_usage(dispatch_indices)    → None

    Notes
    -----
      • Unlike the discrete-slot version, there is no `max_slots` growth
        boundary; `num_experts` is a fixed hyper-parameter.
      • The `route_novelty` method is replaced by `route_novelty` returning
        a confidence score (entropy-based) rather than a discrete decision,
        giving the streaming loop more flexibility.
      • `update_usage` now works on Top-K indices so that multi-expert
        usage is tracked fairly.
    """

    def __init__(
        self,
        prompt_dim: int | tuple,
        num_experts: int,
        top_k: int = 2,
        temperature: float = 1.0,
        load_balancing_alpha: float = 1e-3,
        rich_router: Optional[nn.Module] = None,
        use_noop: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        prompt_dim : int or tuple
        num_experts : int
        top_k : int
        temperature : float
        load_balancing_alpha : float
        rich_router : Optional[nn.Module]
            If provided, replaces the default nn.Linear router.
            Must accept (z_features, err_history) and output [B, C, E] or [B, C, E+1].
            When use_noop=True, output must be [B, C, E+1] (last col = no-op).
        use_noop : bool
            If True, adds a no-op expert (frozen zeros) at index num_experts.
            Router must output E+1 logits.
        """
        super().__init__()

        if isinstance(prompt_dim, tuple):
            prompt_dim_int = math.prod(prompt_dim)
        else:
            prompt_dim_int = int(prompt_dim)

        self.prompt_dim           = prompt_dim_int
        self.num_experts          = num_experts
        self.top_k                = top_k
        self.temperature          = temperature
        self.load_balancing_alpha = load_balancing_alpha
        self.use_noop             = use_noop
        # Total expert slots seen by the router (E real + 1 noop if enabled)
        self.n_router_out         = num_experts + (1 if use_noop else 0)

        # Expert bank
        self.prompts: nn.Parameter = nn.Parameter(
            torch.randn(num_experts, self.prompt_dim) * 0.02
        )

        # No-op expert: frozen zeros at index num_experts
        # When router selects this slot, theta contribution = 0 (frozen baseline)
        if use_noop:
            self.register_buffer(
                'noop_prompt',
                torch.zeros(1, self.prompt_dim)
            )
        else:
            self.noop_prompt = None

        # Router: linear (default) or injectable rich router
        if rich_router is not None:
            self.router = rich_router
            self._use_rich_router = True
        else:
            self.router = nn.Linear(self.prompt_dim, self.n_router_out, bias=False)
            self._use_rich_router = False

        self._usage: torch.Tensor = torch.zeros(num_experts)

    def _all_prompts(self) -> Tensor:
        """Return real expert prompts plus optional no-op prompt."""
        if self.use_noop:
            return torch.cat([self.prompts, self.noop_prompt], dim=0)
        return self.prompts

    def _fallback_prompt_index(self, device: torch.device) -> Tensor:
        """Index used when a router emits an invalid prompt id."""
        fallback = self.num_experts if self.use_noop else 0
        return torch.tensor(fallback, device=device, dtype=torch.long)

    # ======================================================================
    # Public API
    # ======================================================================

    def retrieve_prompt(
        self, z_query: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Retrieve an aggregated expert prompt via sparse soft routing.

        Parameters
        ----------
        z_query : Tensor, shape [B, C, D]
                  Global regime embedding from GlobalFusion.
                  B = batch (typically 1 in strict streaming),
                  C = number of channels (CI backbone channels),
                  D = prompt_dim.

        Returns
        -------
        theta : Tensor, shape [B, C, D]
                Aggregated prompt tensor; ready to be broadcast-added to
                H_local in the framework's dual-path.
        routing_probs : Tensor, shape [B, C, E]
                        Full softmax routing probabilities (before Top-K mask).
                        Required by compute_load_balancing_loss.
        dispatch_indices : Tensor, shape [B, C, K]
                           Integer indices of the Top-K selected experts.
        """
    def _linear_router_logits(self, z_query: Tensor) -> Tensor:
        """Standard linear router path: z_query [B,C,D] → logits [B,C,E(+1)]."""
        B, C, D = z_query.shape
        z_flat = z_query.view(B * C, D)
        logits_flat = self.router(z_flat)         # [B*C, E(+1)]
        return logits_flat.view(B, C, -1)         # [B, C, E(+1)]

    def retrieve_prompt(
        self,
        z_query: Tensor,
        z_features: Optional[Tensor] = None,
        err_history: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Retrieve an aggregated expert prompt via sparse soft routing.

        Parameters
        ----------
        z_query    : [B, C, D]    — mean-pooled patch embedding (always required)
        z_features : [B, C, 4D]   — rich window features (for RichMLPRouter)
        err_history: [B, C, K]    — error history (for RichMLPRouter)

        Returns
        -------
        theta            : [B, C, D]
        routing_probs    : [B, C, E(+1)]
        dispatch_indices : [B, C, K]
        """
        B, C, _ = z_query.shape

        # ---- Router logits ----
        if self._use_rich_router:
            if z_features is None:
                if z_query.shape[-1] == 4 * self.prompt_dim:
                    z_features = z_query
                else:
                    z_features = torch.cat([z_query] * 4, dim=-1)  # legacy fallback
            logits = self.router(z_features, err_history)   # [B, C, E+1]
        else:
            logits = self._linear_router_logits(z_query)    # [B, C, E(+1)]

        # ---- Softmax ----
        routing_probs = F.softmax(logits / self.temperature, dim=-1)  # [B, C, E(+1)]

        # ---- Top-K sparsity ----
        # top_k is applied over real experts only if use_noop;
        # no-op can win by having the highest logit naturally.
        values_topk, indices_topk = torch.topk(
            routing_probs, k=self.top_k, dim=-1, sorted=True
        )  # [B, C, K]

        sum_topk = values_topk.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        probs_topk_norm = values_topk / sum_topk   # [B, C, K]

        # ---- Expert prompts (including no-op at index num_experts) ----
        all_prompts = self._all_prompts()  # [E(+1), D]
        valid_prompt = (indices_topk >= 0) & (indices_topk < all_prompts.shape[0])
        fallback_idx = self._fallback_prompt_index(indices_topk.device)
        safe_indices = torch.where(valid_prompt, indices_topk, fallback_idx)

        # If a mismatched router emits an out-of-bank slot, treat it as no-op.
        probs_topk_norm = probs_topk_norm * valid_prompt.float()
        probs_topk_norm = probs_topk_norm / probs_topk_norm.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-8)

        expert_prompts = all_prompts[safe_indices]   # [B, C, K, D]
        theta = torch.einsum('bck,bckd->bcd', probs_topk_norm, expert_prompts)

        return theta, routing_probs, safe_indices

    # ======================================================================
    # Delayed Update Path (BDLA phase)
    # ======================================================================

    def force_prompt(
        self,
        target_idx_tensor: Tensor,
        z_query_history: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Delayed-update path: rebuild the routing graph and force specific expert(s).

        This method is the core of the BDLA-safe stateless design.  At inference
        time (step t) the router produces dispatch decisions; at update time
        (step t+H) we must retroactively replay those decisions while still
        allowing the router to receive gradients.

        Why rebuilding the graph is necessary
        -------------------------------------
        The naive approach would store routing_probs.detach() in the buffer and
        pass it directly to the loss.  This severs the computation graph: the
        Router never sees gradients from the delayed update, so it never learns
        from the BDLA feedback loop.

        The correct approach (this method) is:
          1. Replay z_query_history through the Router → routing_probs_history
             (the Router's matmul now carries a grad_fn, re-attaching it to the
              active computation graph).
          2. Use torch.gather on routing_probs_history to retrieve the exact
             probabilities assigned to each forced expert.
          3. Perform the weighted aggregation as in retrieve_prompt, but with
             the forced probabilities instead of Top-K softmax ones.

        Parameters
        ----------
        target_idx_tensor : Tensor, shape [B, C, K]
                           Integer indices of the K experts that were active at
                           inference time.  Stored in the BDLABuffer; has no
                           grad (it was detached when pushed).
        z_query_history : Tensor, shape [B, C, D]
                          The historical query tensor (z_global) that was stored
                          in the BDLABuffer alongside the prediction.  This is
                          the key to graph reconstruction: passing it through
                          self.router re-creates the routing probabilities for
                          the exact historical timestep, not the current one.

        Returns
        -------
        theta_forced : Tensor, shape [B, C, D]
                       Aggregated forced prompt — differentiable.
        routing_probs_history : Tensor, shape [B, C, E]
                                Full softmax probabilities from the reconstructed
                                routing head — carries grad_fn so the Router
                                receives gradients through the load-balancing loss.
        """
        target_idx_tensor = target_idx_tensor.long()
        B, C, K = target_idx_tensor.shape
        device  = target_idx_tensor.device
        D = self.prompt_dim
        E = self.n_router_out

        # ---- Replay routing (graph reconstruction) ----
        if self._use_rich_router:
            if z_query_history.shape[-1] == 4 * self.prompt_dim:
                z_features = z_query_history
            else:
                z_features = torch.cat([z_query_history] * 4, dim=-1)  # legacy fallback
            logits_history = self.router(z_features, None)          # [B, C, E(+1)]
        else:
            z_flat = z_query_history.view(B * C, D)
            logits_history = self.router(z_flat).view(B, C, E)      # [B, C, E(+1)]

        routing_probs_history = F.softmax(
            logits_history / self.temperature, dim=-1
        )  # [B, C, E(+1)]

        # ---- Gather forced expert probabilities ----
        n_router_slots = routing_probs_history.shape[-1]
        valid_router = (target_idx_tensor >= 0) & (target_idx_tensor < n_router_slots)
        safe_router_idx = target_idx_tensor.clamp(
            min=0, max=max(n_router_slots - 1, 0)
        )
        probs_gathered = torch.gather(
            routing_probs_history, dim=-1,
            index=safe_router_idx,
        )  # [B, C, K]
        probs_gathered = probs_gathered * valid_router.float()

        sum_probs = probs_gathered.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        probs_forced_norm = probs_gathered / sum_probs  # [B, C, K]

        # ---- Gather expert prompts (noop = zeros at index num_experts) ----
        all_prompts = self._all_prompts()
        valid_prompt = (target_idx_tensor >= 0) & (target_idx_tensor < all_prompts.shape[0])
        fallback_idx = self._fallback_prompt_index(device)
        safe_prompt_idx = torch.where(valid_prompt, target_idx_tensor, fallback_idx)
        forced_prompts = all_prompts[safe_prompt_idx]  # [B, C, K, D]
        forced_prompts = forced_prompts * valid_prompt.unsqueeze(-1).float()

        if K > 1:
            theta_forced = torch.einsum('bck,bckd->bcd', probs_forced_norm, forced_prompts)
        else:
            theta_forced = forced_prompts.squeeze(2)

        return theta_forced, routing_probs_history



    # ======================================================================
    # Masked Load-Balancing Loss
    # ======================================================================

    def compute_load_balancing_loss(
        self,
        routing_probs: Tensor,
        drift_mask: Tensor,
        alpha: float = 1e-3,
    ) -> Tensor:
        """
        Masked load-balancing auxiliary loss.

        This loss encourages the router to use all experts fairly, but ONLY
        for channels marked as drifting (drift_mask == 1).  Stable channels
        (drift_mask == 0) are excluded so their expert affinity is
        undisturbed.

        Mathematical derivation
        -----------------------
        For expert e ∈ {1,…,E} over a mini-batch with N_active active
        (channel, step) pairs (where N_active = Σ drift_mask):

          P_e = (1/N_active) · Σₙ  pₑⁿ          (differentiable mean prob)
          f_e = (1/N_active) · Σₙ  𝟙[argmax(pⁿ) = e]   (non-diff frequency)

        The auxiliary loss is:
          L_aux = α · Σₑ  f_e · P_e

        Using the one-hot representation of argmax:
          f_e  = (1/N_active) · Σₙ  one_hot(argmax(pⁿ))ₑ
               = (1/N_active) · Σₙ  ŵₑⁿ        where ŵ = detach(softmax)

        Since f_e is detached, gradients flow only through P_e (via p),
        which is exactly what we want: the router learns to reduce the
        product f_e · P_e, i.e. to flatten the probability distribution
        for drifting channels.

        Parameters
        ----------
        routing_probs : Tensor, shape [B, C, E]
                        Full softmax probabilities (before Top-K).
        drift_mask : Tensor, shape [B, C]
                     Binary mask (float 0.0 or 1.0).
                     1 = channel triggered Actual Drift this step → penalise.
                     0 = stable channel → exclude from loss.
        alpha : float, default 1e-3
                Scaling coefficient.  Typical range: [1e-4, 1e-2].

        Returns
        -------
        l_aux : Tensor, scalar
                The load-balancing auxiliary loss.
        """
        B, C, E = routing_probs.shape
        device  = routing_probs.device

        # ------------------------------------------------------------------
        # Flatten spatial dimensions to (B·C,)
        # ------------------------------------------------------------------
        p_flat = routing_probs.view(B * C, E)               # (B*C, E)
        m_flat = drift_mask.view(B * C).float()             # (B*C,)

        # ------------------------------------------------------------------
        # a) Count active (drifting) (B·C) pairs
        # ------------------------------------------------------------------
        N_active = m_flat.sum().clamp(min=1.0)              # scalar

        # ------------------------------------------------------------------
        # b) Differentiable mean probability density per expert P_e
        #    Only count active channels.
        # ------------------------------------------------------------------
        # p_flat   [B*C, E]
        # m_flat   [B*C, 1]
        # → masked sum [1, E]
        masked_prob_sum = (p_flat * m_flat.unsqueeze(-1)).sum(dim=0)   # (E,)
        P = masked_prob_sum / N_active                                       # (E,)

        # ------------------------------------------------------------------
        # c) Non-differentiable assignment frequency f_e (one-hot + detach)
        # ------------------------------------------------------------------
        # argmax over expert dimension → indices [B*C, 1]
        argmax_idx = routing_probs.argmax(dim=-1).long().detach()  # (B, C)
        argmax_flat = argmax_idx.view(B * C, 1)                    # (B*C, 1)

        # one-hot encoding: w_onehot [B*C, E]
        w_onehot = F.one_hot(argmax_flat.squeeze(-1), num_classes=E).float()  # (B*C, E)

        # Apply drift mask (zero out stable channels) then normalise
        f = (w_onehot * m_flat.unsqueeze(-1)).sum(dim=0) / N_active           # (E,)

        # ------------------------------------------------------------------
        # d) Load-balancing auxiliary loss
        # ------------------------------------------------------------------
        #   L_aux = α · Σₑ (f_e · P_e)
        # f and P are both (E,) vectors
        l_aux = alpha * E * torch.dot(f, P)

        # ------------------------------------------------------------------
        # e) NaN/Inf guard (should never fire, but defensive)
        # ------------------------------------------------------------------
        if not torch.isfinite(l_aux):
            return torch.zeros([], device=device, dtype=routing_probs.dtype)

        return l_aux

    # ======================================================================
    # Routing novelty / confidence metric
    # ======================================================================

    def route_novelty(self, routing_probs: Tensor) -> Tensor:
        """
        Compute a routing confidence score (1 − normalised entropy).

        High confidence (→ 1)  : router is decisive; one expert dominates.
        Low confidence (→ 0)   : router is uncertain; regime boundary may be near.

        This gives the streaming loop a scalar signal for novelty detection
        without hard thresholding on expert indices.

        Parameters
        ----------
        routing_probs : Tensor, shape [B, C, E]

        Returns
        -------
        confidence : Tensor, shape [B, C]
                     Per-(batch, channel) confidence score.
        """
        # Entropy of softmax distribution: H = -Σ p·log(p)
        # For a uniform distribution over E experts: H_uniform = log(E)
        # Normalised entropy ∈ [0, 1]: 1 - H / log(E)
        H = -(routing_probs * (routing_probs + 1e-8).log()).sum(dim=-1)   # (B, C)
        H_uniform = math.log(self.num_experts)
        confidence = 1.0 - H / H_uniform
        return confidence

    # ======================================================================
    # Usage tracking
    # ======================================================================

    def update_usage(self, dispatch_indices: Tensor) -> None:
        """
        Increment the usage counter for the dispatched expert(s).

        Parameters
        ----------
        dispatch_indices : Tensor, shape [B, C, K]
                           Top-K expert indices from the most recent
                           retrieve_prompt call.
        """
        B, C, K = dispatch_indices.shape
        flat_indices = dispatch_indices.view(B * C, K).long()   # (B*C, K)
        usage_delta = torch.zeros(self.num_experts, device=dispatch_indices.device)
        for k in range(K):
            idx_k = flat_indices[:, k]
            # Filter out no-op/out-of-bank indices before scatter_add.
            valid_mask = (idx_k >= 0) & (idx_k < self.num_experts)
            if valid_mask.any():
                valid_idx = idx_k[valid_mask]
                usage_delta.scatter_add_(
                    0, valid_idx,
                    torch.ones(valid_idx.shape[0], device=dispatch_indices.device)
                )
        self._usage += usage_delta.detach().cpu()


    def get_usage(self) -> Tensor:
        """Return the current usage counters for all experts."""
        return self._usage.clone()

    def reset_usage(self) -> None:
        """Reset all usage counters to zero."""
        self._usage.zero_()

    # ======================================================================
    # Utilities
    # ======================================================================

    def extra_repr(self) -> str:
        return (
            f"prompt_dim={self.prompt_dim}, "
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"temperature={self.temperature}, "
            f"load_balancing_alpha={self.load_balancing_alpha}"
        )

    def __repr__(self) -> str:
        return f"SparsePromptMemory({self.extra_repr()})"
