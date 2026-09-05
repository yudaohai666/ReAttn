# Custom OpenCompass model classes for patched models
import os
import sys
from typing import Optional, Callable
from transformers import AutoConfig


current_dir = os.path.dirname(os.path.abspath(__file__))
# Insert the repo root (two levels up from eval/benchmarks) so that `import pbs_attn` works
# regardless of the working directory or PYTHONPATH at subprocess launch time.
abs_path = os.path.join(current_dir, '../..')
abs_path = os.path.normpath(abs_path)
if abs_path not in sys.path:
    sys.path.insert(0, abs_path)

from opencompass.models.huggingface_above_v4_33 import HuggingFacewithChatTemplate
from opencompass.registry import MODELS
from pbs_attn.patch.compat import patch_tokenizer_encode_plus
from pbs_attn.patch.huggingface import (
    apply_patch_with_prefill,
    get_meanpooling_prefill,
    get_minference_prefill,

    get_xattention_prefill,
    get_flexprefill_prefill,
    get_flashattn_prefill,

    get_permuted_block_sparse_attn_fwd,
    get_reuse_v1_prefill,
    get_duo_attention_prefill,
)

# transformers >= 5 dropped tokenizer.batch_encode_plus, which OpenCompass still
# calls in its generate/score paths. Restore it before any tokenizer is loaded.
patch_tokenizer_encode_plus()


@MODELS.register_module()
class PatchedHuggingFaceCausalLM(HuggingFacewithChatTemplate): 
    def __init__(self, 
                 path: str,
                 patch_type: str = 'meanpooling',
                 patch_kwargs: dict = {},
                 **kwargs):
        self.patch_type = patch_type
        self.patch_kwargs = patch_kwargs
        super().__init__(path=path, **kwargs)
    
    def _load_model(self, path: str, kwargs: dict, peft_path: Optional[str] = None, peft_kwargs: dict = dict()):
        """Override _load_model to apply patch after model loading."""
        # Inject YaRN RoPE config for Qwen3 before loading (applies to all paths)
        model_config = AutoConfig.from_pretrained(path)
        self._is_qwen3 = getattr(model_config, "model_type", None) == "qwen3"
        if self._is_qwen3:
            model_config.rope_parameters = {
                "rope_theta": 1000000,
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 32768,
            }
            model_config.max_position_embeddings = 131072
            kwargs["config"] = model_config

        # sparse_reuse loads its own model (custom attn_implementation + paged
        # cache) instead of the vanilla HF model + prefill monkey-patch.
        if self.patch_type == 'sparse_reuse':
            self._load_sparse_reuse_model(path, kwargs)
            return

        # First load the model normally
        super()._load_model(path, kwargs, peft_path, peft_kwargs)

        # Patch tokenizer to disable thinking mode for Qwen3
        if self._is_qwen3:
            _orig_apply_chat_template = self.tokenizer.apply_chat_template
            def _apply_chat_template_no_think(*args, **kw):
                kw.setdefault('enable_thinking', False)
                return _orig_apply_chat_template(*args, **kw)
            self.tokenizer.apply_chat_template = _apply_chat_template_no_think

        # Then apply the appropriate patch
        prefill_fn = self._get_prefill_function()
        if prefill_fn is not None:
            print(f"🔧 Applying {self.patch_type} patch to model...")
            self.model = apply_patch_with_prefill(self.model, prefill_fn)
        else:
            print("⚠️  No patch applied - using original model")

    def _load_sparse_reuse_model(self, path: str, kwargs: dict):
        """Load a model with cross-layer sparse-reuse attention.

        Unlike the prefill-patch methods, sparse_reuse replaces the attention
        implementation and drives a PagedSparseReuseCache, so it needs its own
        loader (``load_model_with_reuse``) and a generate() wrapper that
        prepares/resets that cache per call. B==1 only.
        """
        # Lazy import: fmha_sm100 / cutlass are only needed for this method.
        import os as _os
        import sys as _sys
        import torch
        _repo_root = _os.path.dirname(_os.path.dirname(current_dir))
        if _repo_root not in _sys.path:
            _sys.path.insert(0, _repo_root)
        from sparse_attn.sparse_reuse import load_model_with_reuse

        pk = dict(self.patch_kwargs)
        label_path = pk.pop('label_path', None)
        if not label_path:
            raise ValueError("sparse_reuse requires 'label_path' in patch_kwargs")

        dtype = kwargs.get('torch_dtype', torch.bfloat16)
        if isinstance(dtype, str):
            dtype = {'torch.float16': torch.float16, 'torch.bfloat16': torch.bfloat16,
                     'torch.float': torch.float}.get(dtype, torch.bfloat16)

        # Reserve enough paged-cache capacity for the longest prompt + output.
        max_out_len = getattr(self, 'max_out_len', 2048) or 2048
        init_kv_len = int(self.max_seq_len or 128 * 1024) + int(max_out_len) + 32

        print(f"🔧 Loading sparse_reuse model (label={label_path})...")
        model, _tok, cache = load_model_with_reuse(
            model_path=path,
            label_path=label_path,
            dtype=dtype,
            device='cuda',
            paged_cache_max_kv_len=init_kv_len,
            model_config=kwargs.get('config', None),
            **pk,
        )
        model.generation_config.do_sample = False
        self.model = model
        self.reuse_cache = cache
        self._wrap_generate_with_reuse_cache()

    def _wrap_generate_with_reuse_cache(self):
        """Inject the PagedSparseReuseCache into every model.generate call.

        OpenCompass calls ``self.model.generate(**tokens, **generation_kwargs)``;
        the sparse_reuse path additionally needs the paged cache sized, reset,
        and passed as ``past_key_values``.
        """
        cache = self.reuse_cache
        orig_generate = self.model.generate

        def generate(*args, **kwargs):
            input_ids = kwargs.get('input_ids')
            if input_ids is None and args:
                input_ids = args[0]
            seq_len = int(input_ids.shape[-1]) if input_ids is not None else 0
            max_new = int(kwargs.get('max_new_tokens', 0) or 0)
            cache.prepare_for(seq_len + max_new + 32)
            cache.reset()
            kwargs['past_key_values'] = cache
            kwargs['use_cache'] = True
            return orig_generate(*args, **kwargs)

        self.model.generate = generate

    def _get_prefill_function(self) -> Optional[Callable]:
        """Get the appropriate prefill function based on patch_type."""
        if self.patch_type == 'meanpooling':
            return get_meanpooling_prefill(**self.patch_kwargs)
        elif self.patch_type == 'minference':
            return get_minference_prefill(**self.patch_kwargs)
        elif self.patch_type == 'xattention':
            return get_xattention_prefill(**self.patch_kwargs)
        elif self.patch_type == 'flexprefill':
            return get_flexprefill_prefill(**self.patch_kwargs)
        elif self.patch_type == 'flashattn':
            return get_flashattn_prefill(**self.patch_kwargs)
        elif self.patch_type == 'pbs':
            return get_permuted_block_sparse_attn_fwd(**self.patch_kwargs)
        elif self.patch_type == 'reuse_v1':
            return get_reuse_v1_prefill(**self.patch_kwargs)
        elif self.patch_type == 'duo':
            return get_duo_attention_prefill(**self.patch_kwargs)
        else:
            print(f"⚠️  Unknown patch_type: {self.patch_type}")
            
            return None


# Convenience classes for specific patches
@MODELS.register_module()
class MeanPoolingHuggingFaceCausalLM(PatchedHuggingFaceCausalLM):
    """HuggingFace CausalLM with MeanPooling sparse attention."""
    def __init__(self, path: str, **kwargs):
        patch_kwargs = kwargs.pop('patch_kwargs', {})
        super().__init__(path=path, patch_type='meanpooling', patch_kwargs=patch_kwargs, **kwargs)


@MODELS.register_module()
class MinferenceHuggingFaceCausalLM(PatchedHuggingFaceCausalLM):
    """HuggingFace CausalLM with Minference sparse attention."""
    def __init__(self, path: str, **kwargs):
        patch_kwargs = kwargs.pop('patch_kwargs', {})
        super().__init__(path=path, patch_type='minference', patch_kwargs=patch_kwargs, **kwargs)


@MODELS.register_module()
class XAttentionHuggingFaceCausalLM(PatchedHuggingFaceCausalLM):
    """HuggingFace CausalLM with XAttention sparse attention."""
    def __init__(self, path: str, **kwargs):
        patch_kwargs = kwargs.pop('patch_kwargs', {})
        super().__init__(path=path, patch_type='xattention', patch_kwargs=patch_kwargs, **kwargs)


@MODELS.register_module()
class FlexPrefillHuggingFaceCausalLM(PatchedHuggingFaceCausalLM):
    """HuggingFace CausalLM with FlexPrefill sparse attention."""
    def __init__(self, path: str, **kwargs):
        patch_kwargs = kwargs.pop('patch_kwargs', {})
        super().__init__(path=path, patch_type='flexprefill', patch_kwargs=patch_kwargs, **kwargs) 


@MODELS.register_module()
class FlashAttnHuggingFaceCausalLM(PatchedHuggingFaceCausalLM):
    """HuggingFace CausalLM with FlashAttention."""
    def __init__(self, path: str, **kwargs):
        patch_kwargs = kwargs.pop('patch_kwargs', {})
        super().__init__(path=path, patch_type='flashattn', patch_kwargs=patch_kwargs, **kwargs)


@MODELS.register_module()
class PBSHuggingFaceCausalLM(PatchedHuggingFaceCausalLM):
    """HuggingFace CausalLM with Permuted Block Sparse (PBS) attention."""
    def __init__(self, path: str, **kwargs):
        patch_kwargs = kwargs.pop('patch_kwargs', {})
        super().__init__(path=path, patch_type='pbs', patch_kwargs=patch_kwargs, **kwargs)
