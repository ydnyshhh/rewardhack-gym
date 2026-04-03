from __future__ import annotations

_BOOTSTRAPPED = False


def bootstrap_builtin_environments() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    from rewardhack_gym.envs.code.interface_conformance import InterfaceConformanceCodeEnvironment
    from rewardhack_gym.envs.code.patch_verification import PatchVerificationCodeEnvironment
    from rewardhack_gym.envs.code.spec_overfit import SpecOverfitCodeEnvironment
    from rewardhack_gym.envs.math.constraint_sensitive import ConstraintSensitiveMathEnvironment
    from rewardhack_gym.envs.math.reasoning_validity import ReasoningValidityMathEnvironment
    from rewardhack_gym.envs.math.symbolic_mismatch import SymbolicMismatchMathEnvironment

    # Importing the modules is sufficient because registration happens at module scope.
    _ = (
        InterfaceConformanceCodeEnvironment,
        PatchVerificationCodeEnvironment,
        SpecOverfitCodeEnvironment,
        ConstraintSensitiveMathEnvironment,
        ReasoningValidityMathEnvironment,
        SymbolicMismatchMathEnvironment,
    )
    _BOOTSTRAPPED = True
