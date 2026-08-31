# SPDX-License-Identifier: MIT
"""PagedSparseReuseCache: paged KV layout, B==1, bf16.

Replaces HF DynamicCache + _to_paged double-materialization with a single
paged buffer (num_pages, H_kv, page_size, D) that matches the layout
fmha_sm100 already expects. CrossLayerSparseReuse / sparse_reuse_attention_forward
stay unchanged; this just provides paged views. Hooks into the HF transformers
Cache abstraction.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import Cache, CacheLayerMixin


PAGE_SIZE = 128


class PagedSparseCacheLayer(CacheLayerMixin):
    """Per-layer paged KV store.

    Physical layout: (max_pages, num_kv_heads, page_size, head_dim) bf16.
    Token `pos` lives at (pos // page_size, :, pos % page_size, :).
    B==1, contiguous allocation, no page_table indirection.
    """

    is_sliding = False
    is_compileable = False

    def __init__(self, max_pages: int, num_kv_heads: int, head_dim: int,
                 page_size: int = PAGE_SIZE,
                 dtype: torch.dtype = torch.bfloat16,
                 device: str | torch.device = "cuda"):
        super().__init__()
        self.max_pages = int(max_pages)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.page_size = int(page_size)
        self.dtype = dtype
        self.device = torch.device(device) if isinstance(device, str) else device
        self.k_paged: torch.Tensor | None = None
        self.v_paged: torch.Tensor | None = None
        self.seq_len: int = 0
        self.is_initialized = False

    def lazy_initialization(self, key_states: torch.Tensor,
                            value_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        # Zero-init: the FMHA kernel reads full physical pages, so any slot
        # beyond seqused_k must be a benign zero, not random memory.
        self.k_paged = torch.zeros(
            self.max_pages, self.num_kv_heads, self.page_size, self.head_dim,
            dtype=self.dtype, device=self.device)
        self.v_paged = torch.zeros_like(self.k_paged)
        self.is_initialized = True

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               cache_kwargs: dict | None = None,
               *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        """Scatter (1, H_kv, S_new, D) into paged slots; return paged views.

        Returned tensors are 4D paged (num_pages, H_kv, page_size, D), NOT
        (B, H_kv, S, D). sparse_reuse_attention_forward understands this and
        skips _to_paged; eager/sdpa do not.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        assert key_states.shape[0] == 1, "cache only supports B==1"

        S_new = key_states.shape[-2]
        cache_position = None
        if cache_kwargs is not None:
            cache_position = cache_kwargs.get("cache_position")
        if cache_position is None:
            cache_position = torch.arange(
                self.seq_len, self.seq_len + S_new,
                device=self.device, dtype=torch.int64)
        else:
            cache_position = cache_position.to(self.device).long()
        # Contiguous append: new tokens occupy [seq_len, seq_len + S_new), so the
        # highest written position is seq_len + S_new - 1. Compute it as a plain
        # Python int instead of cache_position.max().item() -- that .item() was a
        # GPU->CPU sync firing every layer every decode step (32x per step),
        # stalling the host. No D2H here now.
        max_pos = self.seq_len + S_new - 1
        assert max_pos < self.max_pages * self.page_size, (
            f"position {max_pos} exceeds cache capacity "
            f"{self.max_pages * self.page_size}")

        page_idx = cache_position // self.page_size
        slot_idx = cache_position % self.page_size

        k_in = key_states[0].permute(1, 0, 2).contiguous()
        v_in = value_states[0].permute(1, 0, 2).contiguous()

        self.k_paged[page_idx, :, slot_idx, :] = k_in
        self.v_paged[page_idx, :, slot_idx, :] = v_in

        self.seq_len = max(self.seq_len, max_pos + 1)
        return self.k_paged, self.v_paged

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.seq_len, 0

    def get_seq_length(self) -> int:
        return self.seq_len

    def get_max_cache_shape(self) -> int:
        return self.max_pages * self.page_size

    # transformers <5 name; kept so older call sites keep working.
    get_max_length = get_max_cache_shape

    def reset(self) -> None:
        # Zero the slack so the partial-page tail of a previous run can't leak.
        if self.is_initialized:
            self.k_paged.zero_()
            self.v_paged.zero_()
        self.seq_len = 0

    def resize(self, new_max_pages: int) -> None:
        """Resize the pool to exactly `new_max_pages` and reset seq_len.

        Always leaves the buffers zero-initialized. Call between cases when the
        next prompt length is known to avoid sizing by the largest context.
        """
        new_max_pages = int(new_max_pages)
        assert new_max_pages > 0, f"new_max_pages must be > 0, got {new_max_pages}"
        if (not self.is_initialized) or self.max_pages != new_max_pages:
            # Drop old buffers before allocating to keep peak memory low.
            self.k_paged = None
            self.v_paged = None
            self.max_pages = new_max_pages
            self.k_paged = torch.zeros(
                self.max_pages, self.num_kv_heads, self.page_size, self.head_dim,
                dtype=self.dtype, device=self.device)
            self.v_paged = torch.zeros_like(self.k_paged)
            self.is_initialized = True
        else:
            self.k_paged.zero_()
            self.v_paged.zero_()
        self.seq_len = 0


class PagedSparseReuseCache(Cache):
    """Top-level Cache wrapping one PagedSparseCacheLayer per layer.

    `is_paged = True` lets sparse_reuse_attention_forward detect the layout
    and skip _to_paged.
    """

    is_paged = True

    def __init__(self, num_layers: int, max_kv_len: int,
                 num_kv_heads: int, head_dim: int,
                 page_size: int = PAGE_SIZE,
                 dtype: torch.dtype = torch.bfloat16,
                 device: str | torch.device = "cuda"):
        max_pages = (max_kv_len + page_size - 1) // page_size
        layers = [
            PagedSparseCacheLayer(
                max_pages=max_pages, num_kv_heads=num_kv_heads,
                head_dim=head_dim, page_size=page_size,
                dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        super().__init__(layers=layers)
        self.num_layers = num_layers
        self.max_pages = max_pages
        self.page_size = page_size

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()

    def prepare_for(self, max_kv_len: int) -> None:
        """Resize every layer's pool to exactly fit `max_kv_len` tokens."""
        new_max_pages = (int(max_kv_len) + self.page_size - 1) // self.page_size
        assert new_max_pages > 0, (
            f"prepare_for requires max_kv_len > 0, got {max_kv_len}")
        for layer in self.layers:
            layer.resize(new_max_pages)
        self.max_pages = new_max_pages

    @property
    def paged_bytes(self) -> int:
        total = 0
        for layer in self.layers:
            if layer.is_initialized and layer.k_paged is not None:
                total += layer.k_paged.numel() * layer.k_paged.element_size()
                total += layer.v_paged.numel() * layer.v_paged.element_size()
        return total
