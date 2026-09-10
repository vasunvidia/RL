"""Low-precision recipes for Qwen3.5 GRPO."""

from __future__ import annotations

from typing import Optional

from transformer_engine.pytorch.custom_recipes.quantizer_factory_zoo import (
    mxfp8_fwd_high_precision_bwd_factory,
)
from transformer_engine.pytorch.quantization import QuantizerRole
from transformer_engine.pytorch.tensor.identity_tensor import IdentityQuantizer


def install_mcore_routed_mxfp8_fwd_alignment() -> None:
    """Use MXFP8 block-32 alignment for MCore's CustomRecipe path."""
    from megatron.core import fp8_utils
    from megatron.core.transformer.moe import moe_utils

    original = moe_utils.get_fp8_align_size

    def get_fp8_align_size(fp8_recipe):
        recipe = getattr(fp8_recipe, "value", fp8_recipe)
        if str(recipe).lower() == "custom":
            return 32
        return original(fp8_recipe)

    fp8_utils.get_fp8_align_size = get_fp8_align_size
    moe_utils.get_fp8_align_size = get_fp8_align_size


install_mcore_routed_mxfp8_fwd_alignment()


def routed_expert_mxfp8_fwd_factory(role: Optional[QuantizerRole]):
    """Apply forward MXFP8 only to routed GroupedLinear GEMMs.

    Qwen's routed experts use GroupedLinear. The recipe keeps its shared expert
    ungrouped, so every other GEMM retains the identity quantizer.
    """
    if role is not None and role.module_type == "grouped_linear":
        return mxfp8_fwd_high_precision_bwd_factory(role)
    return IdentityQuantizer()


def install_mcore_routed_mxfp8_fwd_checkpoint_safe_globals() -> None:
    """Allow MCore checkpoints to restore this custom recipe by identity."""
    from megatron.core.safe_globals import SafeUnpickler

    required_globals = {
        ("transformer_engine.common.recipe", "CustomRecipe"),
        (__name__, "routed_expert_mxfp8_fwd_factory"),
    }
    SafeUnpickler._SAFE_CLASSES = frozenset(
        set(SafeUnpickler._SAFE_CLASSES) | required_globals
    )


install_mcore_routed_mxfp8_fwd_checkpoint_safe_globals()
