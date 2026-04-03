"""Environment registrations."""

from rewardhack_gym.envs.code.interface_conformance import InterfaceConformanceCodeEnvironment
from rewardhack_gym.envs.code.patch_verification import PatchVerificationCodeEnvironment
from rewardhack_gym.envs.code.spec_overfit import SpecOverfitCodeEnvironment
from rewardhack_gym.envs.math.constraint_sensitive import ConstraintSensitiveMathEnvironment
from rewardhack_gym.envs.math.reasoning_validity import ReasoningValidityMathEnvironment
from rewardhack_gym.envs.math.symbolic_mismatch import SymbolicMismatchMathEnvironment

__all__ = [
    "ConstraintSensitiveMathEnvironment",
    "InterfaceConformanceCodeEnvironment",
    "PatchVerificationCodeEnvironment",
    "ReasoningValidityMathEnvironment",
    "SpecOverfitCodeEnvironment",
    "SymbolicMismatchMathEnvironment",
]

