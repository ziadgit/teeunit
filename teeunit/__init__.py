"""
TeeUnit: Multi-Agent Arena Environment

A turn-based multi-agent deathmatch environment wrapping real Teeworlds,
for LLM reinforcement learning. Built on the OpenEnv framework.
"""

from .models import (
    # Enums
    WeaponType,
    WEAPON_NAMES,
    # Input/Observation/State
    TeeInput,
    TeeAction,  # Alias for backwards compatibility
    TeeObservation,
    TeeState,
    StepResult,
    GameConfig,
    # Supporting types
    VisiblePlayer,
    VisibleProjectile,
    VisiblePickup,
    KillEvent,
    AgentScore,
)

from .client import (
    TeeEnv,
    SyncTeeEnv,
    LocalTeeEnv,
    TeeEnvError,
    make_env,
)

__version__ = "0.2.0"
__all__ = [
    # Version
    "__version__",
    # Enums
    "WeaponType",
    "WEAPON_NAMES",
    # Core types
    "TeeInput",
    "TeeAction",
    "TeeObservation",
    "TeeState",
    "StepResult",
    "GameConfig",
    # Supporting types
    "VisiblePlayer",
    "VisibleProjectile",
    "VisiblePickup",
    "KillEvent",
    "AgentScore",
    # Client
    "TeeEnv",
    "SyncTeeEnv",
    "LocalTeeEnv",
    "TeeEnvError",
    "make_env",
]
