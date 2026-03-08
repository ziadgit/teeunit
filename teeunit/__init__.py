"""
TeeUnit: Multi-Agent Arena Environment

A turn-based multi-agent deathmatch environment for LLM reinforcement learning,
inspired by Teeworlds. Built on the OpenEnv framework.
"""

from .models import (
    # Enums
    ActionType,
    Direction,
    WeaponType,
    TerrainType,
    PickupType,
    # Action/Observation/State
    TeeAction,
    TeeObservation,
    TeeState,
    StepResult,
    GameConfig,
    # Supporting types
    VisibleEnemy,
    VisiblePickup,
    AgentScore,
)

from .client import (
    TeeEnv,
    SyncTeeEnv,
    LocalTeeEnv,
    TeeEnvError,
    make_env,
)

__version__ = "0.1.0"
__all__ = [
    # Version
    "__version__",
    # Enums
    "ActionType",
    "Direction",
    "WeaponType",
    "TerrainType",
    "PickupType",
    # Core types
    "TeeAction",
    "TeeObservation",
    "TeeState",
    "StepResult",
    "GameConfig",
    # Supporting types
    "VisibleEnemy",
    "VisiblePickup",
    "AgentScore",
    # Client
    "TeeEnv",
    "SyncTeeEnv",
    "LocalTeeEnv",
    "TeeEnvError",
    "make_env",
]
