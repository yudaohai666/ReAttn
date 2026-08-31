# coding=utf-8
"""Compatibility shims for third-party baselines built against older transformers."""

import typing

_PATCHED = False


class _AvailabilityResult(tuple):
    """``(available, version)`` tuple that is also usable as a plain bool.

    transformers >= 5 changed ``_is_package_available`` to always return a
    ``(bool, version)`` tuple.  Older code (e.g. minference) still writes
    ``if _is_package_available("pkg"):``, which is always true for a non-empty
    tuple and therefore imports packages that are not installed.
    """

    def __bool__(self):
        return bool(self[0])


def patch_is_package_available():
    """Make ``transformers.utils.import_utils._is_package_available`` dual-form."""
    global _PATCHED
    if _PATCHED:
        return
    import transformers.utils.import_utils as import_utils

    original = import_utils._is_package_available
    if getattr(original, "_pbs_attn_patched", False):
        _PATCHED = True
        return

    def _is_package_available(pkg_name, return_version=False):
        result = original(pkg_name, return_version=return_version)
        if isinstance(result, tuple):
            return _AvailabilityResult(result)
        # Very old transformers returned a bare bool (or bool, version).
        return _AvailabilityResult((result, None))

    _is_package_available._pbs_attn_patched = True
    import_utils._is_package_available = _is_package_available
    _PATCHED = True


def patch_tokenizer_encode_plus():
    """Restore ``batch_encode_plus``/``encode_plus`` on the tokenizer base.

    transformers >= 5 dropped these methods from the tokenizer backends
    (e.g. ``TokenizersBackend``), but OpenCompass 0.5.x still calls
    ``tokenizer.batch_encode_plus(...)`` in its generate/score paths. Both are
    thin wrappers over ``__call__`` in modern transformers, so we delegate.
    """
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if getattr(PreTrainedTokenizerBase, "_pbs_attn_encode_plus_patched", False):
        return
    if hasattr(PreTrainedTokenizerBase, "batch_encode_plus"):
        # transformers still provides it natively; nothing to do.
        return

    def batch_encode_plus(self, batch_text_or_text_pairs, **kwargs):
        return self(batch_text_or_text_pairs, **kwargs)

    def encode_plus(self, text, text_pair=None, **kwargs):
        if text_pair is not None:
            return self(text, text_pair, **kwargs)
        return self(text, **kwargs)

    PreTrainedTokenizerBase.batch_encode_plus = batch_encode_plus
    PreTrainedTokenizerBase.encode_plus = encode_plus
    PreTrainedTokenizerBase._pbs_attn_encode_plus_patched = True


def import_minference_prefill():
    """Import ``Minference_prefill``, working around upstream import issues."""
    patch_is_package_available()
    try:
        from pbs_attn.baselines.Minference import Minference_prefill
    except NameError as e:
        # Some minference releases reference ``Tuple`` without importing it.
        if "name 'Tuple' is not defined" not in str(e):
            raise
        import importlib.util
        import sys

        module_name = "minference.ops.pit_sparse_flash_attention_v2"
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            raise
        module = importlib.util.module_from_spec(spec)
        module.Tuple = typing.Tuple
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        from pbs_attn.baselines.Minference import Minference_prefill
    return Minference_prefill
