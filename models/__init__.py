# Model exports for the trm_bellman package.

from models.value_head import LatentValueHead
from models.edit_policy import EditPolicyHead
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1Carry,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)

__all__ = [
    # Main model
    "TinyRecursiveReasoningModel_ACTV1",
    "TinyRecursiveReasoningModel_ACTV1Config",
    "TinyRecursiveReasoningModel_ACTV1Carry",
    "TinyRecursiveReasoningModel_ACTV1InnerCarry",
    # RL heads
    "LatentValueHead",
    "EditPolicyHead",
]
